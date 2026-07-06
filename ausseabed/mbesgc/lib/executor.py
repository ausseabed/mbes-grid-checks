"""
Manages process of executing checks
"""

from typing import Dict, List, Tuple
from osgeo import gdal
import logging
import numpy as np
import numpy.ma as ma
import os
import tempfile
from pathlib import Path

from .check_utils import get_check
from .data import InputFileDetails, BandType, InputFileDetailsError
from .tiling import get_tiles, Tile
from .gridcheck import GridCheck
from .pinkchart import PinkChartProcessor

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, input_file_details: List[InputFileDetails], check_classes):
        self.input_file_details = input_file_details
        self.tile_size_x = 40000
        self.tile_size_y = 40000
        self.checks = check_classes

        # used to store the results of each check as the checks are run
        # across multiple tiles
        self.check_result_cache: Dict[Tuple[InputFileDetails, str], GridCheck] = {}

        self.spatial_export = False
        self.spatial_export_location = None
        self.spatial_qajson = True

        # list of temporary directories that need to be cleaned up after the Executor
        # has completed processing
        self.temp_dirs: list[str] = []

        # this source input files before any preprocessing is performed
        self.source_input_file_details: List[InputFileDetails] | None = None

    def _preprocess(self):
        """
        Performs some preprocessing of the input datasets. eg; transformation
        of the input rasters to line up with the pink chart.

        If a pink chart file is given for an input file details set then this
        will recreate all input files to match up with the pink chart, and
        also generate a raster version of the pink chart.
        """

        self.source_input_file_details = list(self.input_file_details)

        # iterate by index as we'll be replacing ifds as we iterate over this list
        for ifd_index in range(0, len(self.input_file_details)):
            ifd = self.input_file_details[ifd_index]

            if ifd.pink_chart_filename is None:
                # then there's no pre processing required, so skip to next ifd
                continue

            processed_ifd = ifd.clone()
            # replace this set of input files (ifd) with the preprocessed version
            self.input_file_details[ifd_index] = processed_ifd

            # each ifd is a set of input files, the pink chart stuff needs to be
            # done per each set of these inputs
            if ifd.pink_chart_filename is None:
                # don't do any of the following if there is no pink chart specified
                continue
            # temp_dir = tempfile.TemporaryDirectory()
            # self.temp_dirs.append(temp_dir)
            # temp_dir_path = Path(temp_dir.name)
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            temp_dir_path = Path(temp_dir)

            raster_inputs = []
            raster_outputs = []
            pc_output = temp_dir_path.joinpath(
                Path(ifd.pink_chart_filename).stem + "_pinkchart.tif"
            )

            for input_file, band_index, band_type in ifd.input_band_details:
                output_file = temp_dir_path.joinpath(Path(input_file).stem + ".tif")
                # The pink chart processor processes whole geotiffs including all the bands
                # to make sure we don't unecessarily process the same input raster multiple
                # times (as would be the case with a multi band geotiff), we filter out duplicates
                # here.
                # Also, while reprocessing duplicates worked on MacOS it failed on Windows
                if Path(input_file) not in raster_inputs:
                    raster_inputs.append(Path(input_file))
                    raster_outputs.append(output_file)

                processed_ifd.add_band_details(str(output_file), band_index, band_type)

            pcp = PinkChartProcessor(
                raster_inputs, Path(ifd.pink_chart_filename), raster_outputs, pc_output
            )
            pcp.process()

            # update the size of the rasters so the correct tiling strategy is calculated later
            processed_ifd.size_x = pcp.size_x
            processed_ifd.size_y = pcp.size_y
            processed_ifd.geotransform = pcp.geotransform

            # now that we have a raster version of the pink chart we can add it as a band of data
            # to the input file details
            processed_ifd.add_band_details(str(pc_output), 1, BandType.pinkChart)

    def _load_band_tile(
        self,
        filename: str | None,
        band_index: int | None,
        tile: Tile | None,
        zeroed_nulls: bool = False,
    ) -> np.ndarray | None:
        # function may be called even when the band was not given as input
        # by the user. In such cases the filename and band index will be
        # None. It's up to the checks later on to handle being given None
        # instead of a numpy array
        if filename is None or band_index is None or tile is None:
            return None

        src_ds = gdal.Open(filename)
        if src_ds is None:
            raise RuntimeError(f"Could not open {filename}")

        src_band = src_ds.GetRasterBand(band_index)
        band_data = np.array(
            src_band.ReadAsArray(
                tile.min_x, tile.min_y, tile.max_x - tile.min_x, tile.max_y - tile.min_y
            )
        )

        # we need to mask the nodata values otherwise whatever value is used
        # for nodata will appear in the results
        # zeroed_nulls was included to specifically resolve the issue documented at
        # https://github.com/ausseabed/finder-grid-checks/issues/2
        nodata = src_band.GetNoDataValue()
        if nodata is None:
            # TODO; this should return a masked array like the other two cases
            return band_data
        elif np.isnan(nodata):
            # we need a special case for when NaN is used as nodata because NaN != NaN
            mask = np.isnan(band_data)
            if zeroed_nulls:
                band_data[mask] = 0
            masked_band_data = ma.masked_where(mask, band_data)
            return masked_band_data
        else:
            mask = band_data == nodata
            if zeroed_nulls:
                band_data[mask] = 0
            masked_band_data = ma.masked_where(mask, band_data)
            return masked_band_data

    def _load_data(self, ifd: InputFileDetails, tile: Tile):
        """
        Loads the 3 input bands for the given tile
        """
        depth_file, depth_band_idx = ifd.get_band(BandType.depth)
        density_file, density_band_idx = ifd.get_band(BandType.density)
        uncertainty_file, uncertainty_band_idx = ifd.get_band(BandType.uncertainty)
        pinkchart_file, pinkchart_band_idx = ifd.get_band(BandType.pinkChart)

        depth_data = self._load_band_tile(depth_file, depth_band_idx, tile)
        # makes sense for density to have null data of any value converted to zero
        density_data = self._load_band_tile(density_file, density_band_idx, tile, True)
        uncertainty_data = self._load_band_tile(
            uncertainty_file, uncertainty_band_idx, tile
        )
        pinkchart_data = self._load_band_tile(pinkchart_file, pinkchart_band_idx, tile)

        if density_data is not None:
            density_data = density_data.astype(int)

        return (depth_data, density_data, uncertainty_data, pinkchart_data)

    def _get_output_file_location(
        self, ifd: InputFileDetails, check: GridCheck
    ) -> str | None:
        if self.spatial_export_location is None:
            return None
        check_path = os.path.join(ifd.get_common_filename(), check.name)
        return os.path.join(self.spatial_export_location, check_path)

    def _run_checks(
        self,
        ifd: InputFileDetails,
        tile: Tile,
        depth_data,
        density_data,
        uncertainty_data,
        pinkchart_data,
        is_stopped=None,
    ):
        """
        Runs each of the checks assigned to each file (via the
        InputFileDetails) on the loaded data arrays
        """
        # total number of check that will be run. Not all of them included in
        # the ifd.check_ids_and_params list will be run here as the checks
        # may not be implented by this plugin
        total_check_count = 0
        for check_id, check_params in ifd.check_ids_and_params:
            if get_check(check_id, self.checks) is None:
                # then the check is not supported by this tool
                # so skip and move on
                continue
            total_check_count += 1

        count = 0
        for check_id, check_params in ifd.check_ids_and_params:
            if is_stopped is not None and is_stopped():
                return

            check_class = get_check(check_id, self.checks)
            if check_class is None:
                # then the check is not supported by this tool
                # so skip and move on
                continue
            check = check_class(check_params)

            check.spatial_export = self.spatial_export
            check.spatial_export_location = self._get_output_file_location(ifd, check)
            check.spatial_qajson = self.spatial_qajson

            check.check_started()
            try:
                check.run(
                    ifd,
                    tile,
                    depth_data,
                    density_data,
                    uncertainty_data,
                    pinkchart_data,
                )
                check.check_ended()
            except Exception as e:
                check.execution_status = "failed"
                check.error_message = str(e)

                logger = logging.getLogger(__name__)
                logger.error(e, exc_info=True)

            # if this check has already been run on a different tile we need
            # to merge the results together. Then when all tiles have been run
            # we'll have a single entry for each check in `check_result_cache`
            # that is the result of all tiles merged
            src_ifd = ifd
            if src_ifd.source is not None:
                # make sure we're using the actual source input file details and
                # not a clone. If we use the clone the qajson won't be updated
                # correctly
                src_ifd = src_ifd.source

            if (src_ifd, check_id) in self.check_result_cache:
                last_check = self.check_result_cache[(src_ifd, check_id)]
                check.merge_results(last_check)
            self.check_result_cache[(src_ifd, check_id)] = check

            count += 1
            self.__update_tile_progress(0.2 + count / total_check_count * 0.8)

    def __update_progress(self, progress):
        """Calls the progress callback directly. Passing a value of 1.0
        to this function will set the progress bar to 100%
        """
        if self._progress_callback is None:
            return
        else:
            self._progress_callback(progress)

    def __update_tile_progress(self, progress):
        """Recalculates the progress so that it's between the `self._tile_start_progress`
        and `self._tile_end_progress` values. The `progess` value given to this is assumed
        to be the progress of processing a single tile. eg; progress of 1.0 means that tile
        has been completed (not all tiles)
        """
        if self._progress_callback is None:
            return
        else:
            delta_prog = self._tile_end_progress - self._tile_start_progress
            adjusted_prog = delta_prog * progress + self._tile_start_progress
            self._progress_callback(adjusted_prog)

    def run(self, progress_callback=None, qajson_update_callback=None, is_stopped=None):
        logger.info(f"Processing with tile size {self.tile_size_x},{self.tile_size_y}")

        self._progress_callback = progress_callback
        # preprocess the data
        # - generate pink chart raster, and clip existing rasters to pink chart
        self.__update_progress(0)
        validation_failed = False
        for input_file_detail in self.input_file_details:
            passed, messages = input_file_detail.validate()
            if not passed:
                validation_failed = True
                for msg in messages:
                    logger.error(msg)
        if validation_failed:
            raise InputFileDetailsError("Error occured during validation, check logs")

        self.__update_progress(0.025)
        self._preprocess()

        self.__update_progress(0.05)

        # clear out any previously run checks
        self.check_result_cache = {}

        # collect list of input files, and the list of tiles to be used for
        # each of these input files
        files_and_tiles = []

        # build up a list of all files, and the tiles that need to be loaded
        # and processed for each file
        total_tile_count = 0
        for input_file_detail in self.input_file_details:
            tiles = get_tiles(
                min_x=0,
                min_y=0,
                max_x=input_file_detail.size_x,
                max_y=input_file_detail.size_y,
                size_x=self.tile_size_x,
                size_y=self.tile_size_y,
            )
            total_tile_count += len(tiles)
            file_and_tile = (input_file_detail, tiles)
            files_and_tiles.append(file_and_tile)

        total_file_and_tile_count = 0
        for ifd, tiles in files_and_tiles:
            for _, tile in enumerate(tiles):
                total_file_and_tile_count += 1
        self._tile_start_progress = 0.05

        processed_tile_count = 0
        # loop over each input file
        for ifd, tiles in files_and_tiles:
            # and for each input file loop over the necessary tiles
            # It's much more performant do only load the data for each tile
            # once, and then run all the checks over the loaded tile
            # before moving onto the next
            for _, tile in enumerate(tiles):
                # we use this to help calculate progress info in the __update_tile_progress function
                self._tile_start_progress = (
                    0.05 + processed_tile_count / total_file_and_tile_count * 0.95
                )
                self._tile_end_progress = (
                    0.05 + (processed_tile_count + 1) / total_file_and_tile_count * 0.95
                )
                if is_stopped is not None and is_stopped():
                    return

                self.__update_tile_progress(0)

                depth_data, density_data, uncertainty_data, pinkchart_data = (
                    self._load_data(ifd, tile)
                )

                self.__update_tile_progress(0.2)

                self._run_checks(
                    ifd,
                    tile,
                    depth_data,
                    density_data,
                    uncertainty_data,
                    pinkchart_data,
                    is_stopped,
                )
                processed_tile_count += 1

                self.__update_progress(self._tile_end_progress)
