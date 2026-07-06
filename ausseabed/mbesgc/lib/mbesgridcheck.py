from typing import List, Any
from ausseabed.qajson.model import QajsonParam, QajsonOutputs, QajsonExecution
from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.tiling import Tile

import collections
import logging
import numpy as np
import numpy.ma as ma
import geojson
from geojson import MultiPolygon
from osgeo import gdal, ogr, osr
from affine import Affine

from .gridcheck import GridCheck, GridCheckState

logger = logging.getLogger(__name__)


class DensityCheck(GridCheck):
    """
    Performs check based on density data and input arrays. This looks to
    determine if each node (grid cell) satisfies a minimum count (the value
    stored in the density layer) and whether a percentage of nodes overall
    satisfy a different minimum count.
    """

    id = "5e2afd8a-2ced-4de8-80f5-111c459a7175"
    name = "Density Check"
    version = "1"
    input_params = [
        QajsonParam("Minimum Soundings per node", 5),
        QajsonParam("Minimum Soundings per node percentage", 95.0),
    ]
    parameter_help_link = "user_manual_qax_MBESGC.html#mbesgc-density-params"

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        # if a percentage `_min_spn_p` of all nodes is above a threshold
        # `_min_spn` then this check will fail
        self._min_spn = self.get_param("Minimum Soundings per node")
        self._min_spn_p = self.get_param("Minimum Soundings per node percentage")

        self.tiles_geojson = MultiPolygon()
        self.extents_geojson = MultiPolygon()

        # amount of padding to place around failing pixels
        # this simplifies the geometry, and enlarges the failing area that
        # will allow it to be shown in the UI more easily
        self.pixel_growth = 5

        # was this check run without a density layer (the only layer it needs)
        self.missing_density = False

        self.density_histogram: dict[int, int] = {}

    def run(
        self,
        ifd: InputFileDetails,
        tile: Tile,
        depth,
        density,
        uncertainty,
        pinkchart,
        progress_callback=None,
    ):

        assert ifd.geotransform is not None
        # this check only requires the density layer, so check it is given
        # if not mark this check as aborted
        self.missing_density = density is None
        if self.missing_density:
            logger.info("No density data provided, aborting density check")
            self.execution_status = "aborted"
            self.error_message = "Missing density data"
            self.density_histogram = {}
            # we cant run the check so return
            return

        self.total_cell_count = int(density.count())
        # skip processing this chunk of data if it contains only nodata
        if self.total_cell_count == 0:
            self.density_histogram = {}
            return

        # generate histogram of counts
        # unique_vals will be the soundings per node
        # unique_counts is the total number of times the unique_val soundings
        # count was found.
        unique_vals, unique_counts = np.unique(density, return_counts=True)
        hist = {}
        for val, count in zip(unique_vals, unique_counts):
            if isinstance(val, ma.core.MaskedConstant):
                continue
            # following gets serialized to JSON and as numpy types are not
            # supported by default we convert the float32 and int64 types to
            # plain python ints
            hist[int(val)] = int(count)

        self.density_histogram = hist

        if not (self.spatial_export or self.spatial_export_location):
            # if we don't generate spatial outputs, then there's no
            # need to do any further processing
            return

        bad_cells_mask = density < self._min_spn
        bad_cells_mask.fill_value = False
        bad_cells_mask = bad_cells_mask.filled()
        bad_cells_mask_int8 = bad_cells_mask.astype(np.int8)

        src_affine = Affine.from_gdal(*ifd.geotransform)
        tile_affine = src_affine * Affine.translation(tile.min_x, tile.min_y)

        # there are similarities in how the qajson spatial outputs are generated
        # and how the exported files are generated, however as it is possible
        # to generate both there's not a lot of common steps we can share. Hence
        # theres a bit of duplication; unfortunately duplication in code AND
        # processing.
        if self.spatial_qajson:
            self.extents_geojson = ifd.get_extents_feature()

            tile_ds = gdal.GetDriverByName("MEM").Create(
                "",
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Byte,
            )

            # grow out failed pixels to make them more obvious. We've already
            # calculated the pass/fail stats so this won't impact results.
            bad_cells_mask_int8_grow = self._grow_pixels(
                bad_cells_mask_int8, self.pixel_growth
            )

            # simplify distance is calculated as the distance pixels are grown out
            # `ifd.geotransform[1]` is pixel size
            assert ifd.geotransform is not None
            simplify_distance = self.pixel_growth * ifd.geotransform[1]

            tile_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_band = tile_ds.GetRasterBand(1)
            tile_band.WriteArray(bad_cells_mask_int8_grow, 0, 0)
            tile_band.SetNoDataValue(0)
            tile_band.FlushCache()
            tile_ds.SetProjection(ifd.projection)

            ogr_srs = osr.SpatialReference()
            ogr_srs.ImportFromWkt(ifd.projection)

            ogr_driver = ogr.GetDriverByName("MEM")
            ogr_dataset = ogr_driver.CreateDataSource("shapemask")
            ogr_layer = ogr_dataset.CreateLayer("shapemask", srs=ogr_srs)

            # used the input raster data 'tile_band' as the input and mask, if not
            # used as a mask then a feature that outlines the entire dataset is
            # also produced
            gdal.Polygonize(
                tile_band,
                tile_band,
                ogr_layer,
                -1,
                [],
                callback=None,
            )

            ogr_simple_driver = ogr.GetDriverByName("MEM")
            ogr_simple_dataset = ogr_simple_driver.CreateDataSource("failed_poly")
            ogr_simple_layer = ogr_simple_dataset.CreateLayer("failed_poly", srs=None)

            self._simplify_layer(
                ogr_layer,
                ogr_simple_layer,
                simplify_distance,
            )

            ogr_srs_out = osr.SpatialReference()
            ogr_srs_out.ImportFromEPSG(4326)
            transform = osr.CoordinateTransformation(ogr_srs, ogr_srs_out)

            for feature in ogr_simple_layer:
                transformed = feature.GetGeometryRef()
                transformed.Transform(transform)

                geojson_feature = geojson.loads(feature.ExportToJson())
                self.tiles_geojson.coordinates.append(
                    geojson_feature.geometry.coordinates
                )

            ogr_simple_dataset.Destroy()
            ogr_dataset.Destroy()

        if self.spatial_export:
            tf = self._get_tmp_file("density_failed", "tif", tile)
            tile_ds = gdal.GetDriverByName("GTiff").Create(
                tf,
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Byte,
                options=["COMPRESS=DEFLATE"],
            )

            tile_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_band = tile_ds.GetRasterBand(1)
            tile_band.WriteArray(bad_cells_mask_int8, 0, 0)
            tile_band.SetNoDataValue(0)
            tile_band.FlushCache()
            tile_ds.SetProjection(ifd.projection)

            ogr_srs = osr.SpatialReference()
            ogr_srs.ImportFromWkt(ifd.projection)

            sf = self._get_tmp_file("density_failed", "shp", tile)
            ogr_driver = ogr.GetDriverByName("ESRI Shapefile")
            ogr_dataset = ogr_driver.CreateDataSource(sf)
            ogr_layer = ogr_dataset.CreateLayer("density_failed", srs=ogr_srs)

            # used the input raster data 'tile_band' as the input and mask, if not
            # used as a mask then a feature that outlines the entire dataset is
            # also produced
            gdal.Polygonize(
                tile_band,
                tile_band,
                ogr_layer,
                -1,
                [],
                callback=None,
            )

            tile_ds = None
            ogr_dataset.Destroy()

            self._move_tmp_dir()

        # # includes only the tile boundaries, used for debug
        # tile_geojson = tile.to_geojson(ifd.projection, ifd.geotransform)
        # self.tiles_geojson.coordinates.append(tile_geojson.coordinates)

    def merge_results(self, last_check: GridCheck):
        """
        merge the density histogram of the last_check into the current check
        """
        self.start_time = last_check.start_time
        assert isinstance(last_check, DensityCheck)

        for soundings_count, last_count in last_check.density_histogram.items():
            if soundings_count in self.density_histogram:
                self.density_histogram[soundings_count] += last_count
            else:
                self.density_histogram[soundings_count] = last_count

        self.tiles_geojson.coordinates.extend(last_check.tiles_geojson.coordinates)

        self._merge_temp_dirs(last_check)

    def get_outputs(self) -> QajsonOutputs:

        if len(self.density_histogram) == 0:
            # then there's been nothing but nodata for all tiles
            # posisbly caused by pink chart (coverage) not overlapping
            # raster data.
            self.execution_status = "failed"
            msg = (
                "Density check did not find any non-nodata data in inputs. "
                "This could be a result of a coverage area file not intersecting "
                "grid data."
            )
            logger.error(msg)
            self.error_message = msg

        execution = QajsonExecution(
            start=self.start_time,
            end=self.end_time,
            status=self.execution_status,
            error=self.error_message,
        )

        if self.execution_status == "aborted" or self.execution_status == "failed":
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[self.error_message] if self.error_message else None,
                data={},
                check_state=GridCheckState.cs_fail,
            )

        # sort the list of sounding counts and the number of occurrences
        counts = collections.OrderedDict(sorted(self.density_histogram.items()))

        messages = []
        data: dict[str, Any] = {}
        check_state = None

        lowest_sounding_count, occurrences = next(iter(counts.items()))

        total_soundings = sum(counts.values())
        under_threshold_soundings = 0
        for sounding_count, occurrences in counts.items():
            if sounding_count >= self._min_spn:
                break
            under_threshold_soundings += occurrences

        percentage_over_threshold = (
            1.0 - under_threshold_soundings / total_soundings
        ) * 100.0

        if percentage_over_threshold < self._min_spn_p:
            messages.append(
                f"{percentage_over_threshold:.1f}% of nodes were found to have a "
                f"sounding count above {self._min_spn}. This is required to"
                f" be {self._min_spn_p}% of all nodes"
            )
            check_state = GridCheckState.cs_fail

        if check_state is None:
            check_state = GridCheckState.cs_pass

        str_key_counts = collections.OrderedDict()
        for key, val in counts.items():
            str_key_counts[str(key)] = val

        data["chart"] = {
            "type": "histogram",
            "data": str_key_counts,
        }

        if self.spatial_qajson:
            data["map"] = self.tiles_geojson
            data["extents"] = self.extents_geojson

        data["summary"] = {
            "total_soundings": total_soundings,
            "percentage_over_threshold": percentage_over_threshold,
            "under_threshold_soundings": under_threshold_soundings,
        }

        result = QajsonOutputs(
            execution=execution,
            files=None,
            count=None,
            percentage=None,
            messages=messages,
            data=data,
            check_state=check_state,
        )

        return result


class TvuCheck(GridCheck):
    """
    Total Vertical Uncertainty check. An allowable value is calculated on a
    per node basis that is derived from some constants (constant depth error,
    factor of depth dependent error) and the depth itself. This is then
    compared against the calculated uncertainty value within the data.
    """

    id = "b5c0469c-6559-4aea-bf9c-d0b337550e89"
    name = "Total Vertical Uncertainty Check"
    version = "1"
    input_params = [
        QajsonParam("Constant Depth Error", 0.5),
        QajsonParam("Factor of Depth Dependent Errors", 0.013),
        QajsonParam("Acceptable Area Percentage", 100.0),
    ]
    parameter_help_link = "user_manual_qax_MBESGC.html#mbesgc-tvu-params"

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        self._depth_error = self.get_param("Constant Depth Error")
        self._depth_error_factor = self.get_param("Factor of Depth Dependent Errors")
        self._area_percentage = self.get_param("Acceptable Area Percentage")

        self.tiles_geojson = MultiPolygon()
        self.extents_geojson = MultiPolygon()

        # amount of padding to place around failing pixels
        # this simplifies the geometry, and enlarges the failing area that
        # will allow it to be shown in the UI more easily
        self.pixel_growth: int = 5

        self.total_cell_count: int = 0
        self.failed_cell_count: int = 0

        self.missing_depth: bool = False
        self.missing_uncertainty: bool = False

    def merge_results(self, last_check: GridCheck):
        assert isinstance(last_check, TvuCheck)
        self.start_time = last_check.start_time

        self.total_cell_count += last_check.total_cell_count
        self.failed_cell_count += last_check.failed_cell_count

        self.tiles_geojson.coordinates.extend(last_check.tiles_geojson.coordinates)

        self._merge_temp_dirs(last_check)

    def run(
        self,
        ifd: InputFileDetails,
        tile: Tile,
        depth,
        density,
        uncertainty,
        pinkchart,
        progress_callback=None,
    ):
        # run check on tile data
        assert ifd.geotransform is not None

        # this check only requires the density layer, so check it is given
        # if not mark this check as aborted
        self.missing_depth = depth is None
        self.missing_uncertainty = uncertainty is None

        if self.missing_depth and self.missing_uncertainty:
            self.execution_status = "aborted"
            self.error_message = "Missing depth and uncertainty data"
        elif self.missing_depth:
            self.execution_status = "aborted"
            self.error_message = "Missing depth data"
        elif self.missing_uncertainty:
            self.execution_status = "aborted"
            self.error_message = "Missing uncertainty data"

        if self.execution_status == "aborted":
            # we can't run the check, so return
            logger.info(f"{self.error_message}, aborting TVU check")
            return

        a = self._depth_error
        b = self._depth_error_factor

        # some tools produce negative uncertainty values which will cause
        # problems with the threshold check. So calculate the abs values
        # and use this to check against.
        uncertainty = np.absolute(uncertainty)

        # count of all cells/nodes/pixels that are not NaN in the uncertainty
        # array
        self.total_cell_count = int(uncertainty.count())

        # skip processing this chunk of data if it contains only nodata
        if self.total_cell_count == 0:
            self.failed_cell_count = 0
            return

        # calculate allowable uncertainty based on equation and depth data
        allowable_uncertainty = np.sqrt(a**2 + (b * depth) ** 2)

        failed_uncertainty = uncertainty > allowable_uncertainty

        failed_uncertainty.fill_value = False
        failed_uncertainty = failed_uncertainty.filled()
        failed_uncertainty_int8 = failed_uncertainty.astype(np.int8)

        # count of cells that failed the check
        self.failed_cell_count = int(failed_uncertainty.sum())

        if not (self.spatial_export or self.spatial_export_location):
            # if we don't generate spatial outputs, then there's no
            # need to do any further processing
            return

        src_affine = Affine.from_gdal(*ifd.geotransform)
        tile_affine = src_affine * Affine.translation(tile.min_x, tile.min_y)

        ogr_srs = osr.SpatialReference()
        ogr_srs.ImportFromWkt(ifd.projection)

        # see notes on density check for more details on the processing performed
        # below
        if self.spatial_qajson:
            self.extents_geojson = ifd.get_extents_feature()

            # grow out failed pixels to make them more obvious. We've already
            # calculated the pass/fail stats so this won't impact results.
            failed_uncertainty_int8_grow = self._grow_pixels(
                failed_uncertainty_int8, self.pixel_growth
            )

            # simplify distance is calculated as the distance pixels are grown out
            # `ifd.geotransform[1]` is pixel size
            assert ifd.geotransform is not None
            simplify_distance = self.pixel_growth * ifd.geotransform[1]

            tile_ds = gdal.GetDriverByName("MEM").Create(
                "",
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Float32,
            )
            tile_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_band = tile_ds.GetRasterBand(1)
            tile_band.WriteArray(allowable_uncertainty, 0, 0)
            tile_band.SetNoDataValue(0)
            tile_band.FlushCache()
            tile_ds.SetProjection(ifd.projection)

            tile_failed_ds = gdal.GetDriverByName("MEM").Create(
                "",
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Byte,
            )
            tile_failed_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_failed_band = tile_failed_ds.GetRasterBand(1)
            tile_failed_band.WriteArray(failed_uncertainty_int8_grow, 0, 0)
            tile_failed_band.SetNoDataValue(0)
            tile_failed_band.FlushCache()
            tile_failed_ds.SetProjection(ifd.projection)

            ogr_driver = ogr.GetDriverByName("MEM")
            ogr_dataset = ogr_driver.CreateDataSource("shapemask")
            ogr_layer = ogr_dataset.CreateLayer("shapemask", srs=ogr_srs)

            # used the input raster data 'tile_band' as the input and mask, if not
            # used as a mask then a feature that outlines the entire dataset is
            # also produced
            gdal.Polygonize(
                tile_failed_band,
                tile_failed_band,
                ogr_layer,
                -1,
                [],
                callback=None,
            )

            ogr_simple_driver = ogr.GetDriverByName("MEM")
            ogr_simple_dataset = ogr_simple_driver.CreateDataSource("failed_poly")
            ogr_simple_layer = ogr_simple_dataset.CreateLayer("failed_poly", srs=None)

            self._simplify_layer(ogr_layer, ogr_simple_layer, simplify_distance)

            ogr_srs_out = osr.SpatialReference()
            ogr_srs_out.ImportFromEPSG(4326)
            transform = osr.CoordinateTransformation(ogr_srs, ogr_srs_out)

            for feature in ogr_simple_layer:
                # transform feature into epsg:4326 before export to geojson
                transformed = feature.GetGeometryRef()
                transformed.Transform(transform)

                geojson_feature = geojson.loads(feature.ExportToJson())

                self.tiles_geojson.coordinates.append(
                    geojson_feature.geometry.coordinates
                )

            ogr_simple_dataset.Destroy()
            ogr_dataset.Destroy()

        if self.spatial_export:
            au = self._get_tmp_file("allowable_uncertainty", "tif", tile)
            tile_ds = gdal.GetDriverByName("GTiff").Create(
                au,
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Float32,
                options=["COMPRESS=DEFLATE"],
            )
            tile_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_band = tile_ds.GetRasterBand(1)
            tile_band.WriteArray(allowable_uncertainty, 0, 0)
            tile_band.SetNoDataValue(0)
            tile_band.FlushCache()
            tile_ds.SetProjection(ifd.projection)

            tf = self._get_tmp_file("failed_uncertainty", "tif", tile)
            tile_failed_ds = gdal.GetDriverByName("GTiff").Create(
                tf,
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Byte,
                options=["COMPRESS=DEFLATE"],
            )
            tile_failed_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_failed_band = tile_failed_ds.GetRasterBand(1)
            tile_failed_band.WriteArray(failed_uncertainty_int8, 0, 0)
            tile_failed_band.SetNoDataValue(0)
            tile_failed_band.FlushCache()
            tile_failed_ds.SetProjection(ifd.projection)

            sf = self._get_tmp_file("failed_uncertainty", "shp", tile)
            ogr_driver = ogr.GetDriverByName("ESRI Shapefile")
            ogr_dataset = ogr_driver.CreateDataSource(sf)
            ogr_layer = ogr_dataset.CreateLayer("failed_uncertainty", srs=ogr_srs)

            # used the input raster data 'tile_band' as the input and mask, if not
            # used as a mask then a feature that outlines the entire dataset is
            # also produced
            gdal.Polygonize(
                tile_failed_band,
                tile_failed_band,
                ogr_layer,
                -1,
                [],
                callback=None,
            )

            tile_ds = None
            tile_failed_ds = None
            ogr_dataset.Destroy()

            self._move_tmp_dir()

    def get_outputs(self) -> QajsonOutputs:

        if self.total_cell_count == 0:
            # then there's been nothing but nodata for all tiles
            # posisbly caused by pink chart (coverage) not overlapping
            # raster data.
            self.execution_status = "failed"
            msg = (
                "TVU check did not find any non-nodata data in inputs. "
                "This could be a result of a coverage area file not intersecting "
                "grid data."
            )
            logger.error(msg)
            self.error_message = msg

        execution = QajsonExecution(
            start=self.start_time,
            end=self.end_time,
            status=self.execution_status,
            error=self.error_message,
        )
        data = {}

        if self.execution_status == "completed":
            data = {
                "failed_cell_count": self.failed_cell_count,
                "total_cell_count": self.total_cell_count,
                "fraction_failed": self.failed_cell_count / self.total_cell_count,
            }

            if self.spatial_qajson:
                data["map"] = self.tiles_geojson
                data["extents"] = self.extents_geojson

        if self.execution_status == "aborted" or self.execution_status == "failed":
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[self.error_message] if self.error_message else None,
                data=data,
                check_state=GridCheckState.cs_fail,
            )

        percent_failed = self.failed_cell_count / self.total_cell_count * 100.0
        if (100.0 - percent_failed) < self._area_percentage:
            msg = (
                f"{self.failed_cell_count} nodes failed the TVU check this "
                f"represents {percent_failed:.1f}% of all nodes within data."
                f"The area based percentage of nodes that passed the allowable "
                f"TVU did not exceed the Acceptable Area Percentage specified ({self._area_percentage:.1f}%)"
            )
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[msg],
                data=data,
                check_state=GridCheckState.cs_fail,
            )
        else:
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[],
                data=data,
                check_state=GridCheckState.cs_pass,
            )


class ResolutionCheck(GridCheck):
    """
    Determines what areas of the grid satisfy a resolution check. The check
    first calculates a feature detection size (fds) on a per pixel basis, two
    values for this are calculated above and below a threshold depth. In both
    cases the fds value is calculated from the pixels depth and linear equation
    parameters provided as input parameters to this check. eg;

        fds = Depth Multiplier * depth + Depth Constant

    This equation is calculated using different a different Depth Multiplier
    and Depth Constant depending on whether the depth at that location is
    above of below the threshold.

    Once fds has been calculated per pixel a boolean check is performed to find
    pixels where the grid resolution is lower than the fds * feature
    detection multiplier. If any pixel is false, then the QA check fails.

    """

    id = "c73119ea-4f79-4001-86e3-11c4cbaaeb2d"
    name = "Resolution Check"
    version = "1"

    # default values taken from IHO - 1a spec
    input_params = [
        QajsonParam("Feature Detection Size Multiplier", 0.5),
        QajsonParam("Threshold Depth", 40.0),
        QajsonParam("Above Threshold FDS Depth Multiplier", 0.0),
        QajsonParam("Above Threshold FDS Depth Constant", 2.0),
        QajsonParam("Below Threshold FDS Depth Multiplier", 0.05),
        QajsonParam("Below Threshold FDS Depth Constant", 0.0),
    ]
    parameter_help_link = "user_manual_qax_MBESGC.html#mbesgc-resolution-params"

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        self._fds_multiplier = self.get_param("Feature Detection Size Multiplier")

        self._threshold_depth = self.get_param("Threshold Depth")

        self._a_fds_depth_multiplier = self.get_param(
            "Above Threshold FDS Depth Multiplier"
        )
        self._a_fds_depth_constant = self.get_param(
            "Above Threshold FDS Depth Constant"
        )

        self._b_fds_depth_multiplier = self.get_param(
            "Below Threshold FDS Depth Multiplier"
        )
        self._b_fds_depth_constant = self.get_param(
            "Below Threshold FDS Depth Constant"
        )

        self.tiles_geojson = MultiPolygon()
        self.extents_geojson = MultiPolygon()

        # amount of padding to place around failing pixels
        # this simplifies the geometry, and enlarges the failing area that
        # will allow it to be shown in the UI more easily
        self.pixel_growth = 5

        self.total_cell_count = 0
        self.failed_cell_count = 0

        self.missing_depth = False

    def merge_results(self, last_check: GridCheck):
        assert isinstance(last_check, ResolutionCheck)
        self.start_time = last_check.start_time

        self.total_cell_count += last_check.total_cell_count
        self.failed_cell_count += last_check.failed_cell_count

        self.tiles_geojson.coordinates.extend(last_check.tiles_geojson.coordinates)

        self._merge_temp_dirs(last_check)

    def run(
        self,
        ifd: InputFileDetails,
        tile: Tile,
        depth,
        density,
        uncertainty,
        pinkchart,
        progress_callback=None,
    ):
        # run check on tile data

        # this check only requires the depth layer, so check it is given
        # if not mark this check as aborted
        self.missing_depth = depth is None
        if self.missing_depth:
            self.execution_status = "aborted"
            self.error_message = "Missing depth data"
            logger.info(f"{self.error_message}, aborting resolution check")

            # we cant run the check so return
            return

        assert ifd.geotransform is not None
        self.grid_resolution = ifd.geotransform[1]

        # count of all cells/nodes/pixels that are not NaN in the uncertainty
        # array
        self.total_cell_count = int(depth.count())

        # skip processing this chunk of data if it contains only nodata
        if self.total_cell_count == 0:
            self.failed_cell_count = 0
            return

        abs_depth = np.abs(depth)
        abs_threshold_depth = abs(self._threshold_depth)

        # refer to docs at top of class defn, this is described there
        fds = np.piecewise(
            abs_depth,
            [
                abs_depth < abs_threshold_depth,
                abs_depth >= abs_threshold_depth,
            ],
            [
                lambda d: self._a_fds_depth_multiplier * d + self._a_fds_depth_constant,
                lambda d: self._b_fds_depth_multiplier * d + self._b_fds_depth_constant,
            ],
        )

        fds = np.ma.masked_where(np.ma.getmask(depth), fds)
        allowable_grid_size = fds * self._fds_multiplier

        # The idea of the standard here is that the deeper the water gets the
        # less ability you have to pick up features on the seafloor and also
        # features become less important the deeper the water gets as under
        # keel clearance for ships becomes less of an issue.
        failed_resolution = allowable_grid_size < self.grid_resolution

        failed_resolution.fill_value = False
        failed_resolution = failed_resolution.filled()
        failed_resolution_int8 = failed_resolution.astype(np.int8)

        # count of cells that failed the check
        self.failed_cell_count = int(failed_resolution.sum())

        if not (self.spatial_export or self.spatial_export_location):
            # if we don't generate spatial outputs, then there's no
            # need to do any further processing
            return

        src_affine = Affine.from_gdal(*ifd.geotransform)
        tile_affine = src_affine * Affine.translation(tile.min_x, tile.min_y)

        ogr_srs = osr.SpatialReference()
        ogr_srs.ImportFromWkt(ifd.projection)

        # see notes on density check for more details on the processing performed
        # below
        if self.spatial_qajson:
            self.extents_geojson = ifd.get_extents_feature()

            # grow out failed pixels to make them more obvious. We've already
            # calculated the pass/fail stats so this won't impact results.
            failed_resolution_int8_grow = self._grow_pixels(
                failed_resolution_int8, self.pixel_growth
            )

            # simplify distance is calculated as the distance pixels are grown out
            # `ifd.geotransform[1]` is pixel size
            assert ifd.geotransform
            simplify_distance = self.pixel_growth * ifd.geotransform[1]

            tile_failed_ds = gdal.GetDriverByName("MEM").Create(
                "",
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Byte,
            )
            tile_failed_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_failed_band = tile_failed_ds.GetRasterBand(1)
            tile_failed_band.WriteArray(failed_resolution_int8_grow, 0, 0)
            tile_failed_band.SetNoDataValue(0)
            tile_failed_band.FlushCache()
            tile_failed_ds.SetProjection(ifd.projection)

            ogr_driver = ogr.GetDriverByName("MEM")
            ogr_dataset = ogr_driver.CreateDataSource("shapemask")
            ogr_layer = ogr_dataset.CreateLayer("shapemask", srs=ogr_srs)

            # used the input raster data 'tile_band' as the input and mask, if not
            # used as a mask then a feature that outlines the entire dataset is
            # also produced
            gdal.Polygonize(
                tile_failed_band,
                tile_failed_band,
                ogr_layer,
                -1,
                [],
                callback=None,
            )

            ogr_simple_driver = ogr.GetDriverByName("MEM")
            ogr_simple_dataset = ogr_simple_driver.CreateDataSource("failed_poly")
            ogr_simple_layer = ogr_simple_dataset.CreateLayer("failed_poly", srs=None)

            self._simplify_layer(
                ogr_layer,
                ogr_simple_layer,
                simplify_distance,
            )

            ogr_srs_out = osr.SpatialReference()
            ogr_srs_out.ImportFromEPSG(4326)
            transform = osr.CoordinateTransformation(ogr_srs, ogr_srs_out)

            for feature in ogr_simple_layer:
                # transform feature into epsg:4326 before export to geojson
                transformed = feature.GetGeometryRef()
                transformed.Transform(transform)

                geojson_feature = geojson.loads(feature.ExportToJson())

                self.tiles_geojson.coordinates.append(
                    geojson_feature.geometry.coordinates
                )

            ogr_simple_dataset.Destroy()
            ogr_dataset.Destroy()

        if self.spatial_export:
            allowable_grid_size.fill_value = -9999.0
            allowable_grid_size = allowable_grid_size.filled()

            ar = self._get_tmp_file("allowable_resolution", "tif", tile)
            tile_ds = gdal.GetDriverByName("GTiff").Create(
                ar,
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Float32,
                options=["COMPRESS=DEFLATE"],
            )
            tile_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_band = tile_ds.GetRasterBand(1)
            tile_band.WriteArray(allowable_grid_size, 0, 0)
            tile_band.SetNoDataValue(-9999.0)
            tile_band.FlushCache()
            tile_ds.SetProjection(ifd.projection)

            tf = self._get_tmp_file("failed_resolution", "tif", tile)
            tile_failed_ds = gdal.GetDriverByName("GTiff").Create(
                tf,
                tile.max_x - tile.min_x,
                tile.max_y - tile.min_y,
                1,
                gdal.GDT_Byte,
                options=["COMPRESS=DEFLATE"],
            )
            tile_failed_ds.SetGeoTransform(tile_affine.to_gdal())

            tile_failed_band = tile_failed_ds.GetRasterBand(1)
            tile_failed_band.WriteArray(failed_resolution_int8, 0, 0)
            tile_failed_band.SetNoDataValue(0)
            tile_failed_band.FlushCache()
            tile_failed_ds.SetProjection(ifd.projection)

            sf = self._get_tmp_file("failed_resolution", "shp", tile)
            ogr_driver = ogr.GetDriverByName("ESRI Shapefile")
            ogr_dataset = ogr_driver.CreateDataSource(sf)
            ogr_layer = ogr_dataset.CreateLayer("failed_resolution", srs=ogr_srs)

            # used the input raster data 'tile_band' as the input and mask, if not
            # used as a mask then a feature that outlines the entire dataset is
            # also produced
            gdal.Polygonize(
                tile_failed_band,
                tile_failed_band,
                ogr_layer,
                -1,
                [],
                callback=None,
            )
            tile_ds = None
            tile_failed_ds = None
            ogr_dataset.Destroy()

            self._move_tmp_dir()

    def get_outputs(self) -> QajsonOutputs:

        if self.total_cell_count == 0:
            # then there's been nothing but nodata for all tiles
            # posisbly caused by pink chart (coverage) not overlapping
            # raster data.
            self.execution_status = "failed"
            msg = (
                "Resolution check did not find any non-nodata data in inputs. "
                "This could be a result of a coverage area file not intersecting "
                "grid data."
            )
            logger.error(msg)
            self.error_message = msg

        execution = QajsonExecution(
            start=self.start_time,
            end=self.end_time,
            status=self.execution_status,
            error=self.error_message,
        )

        data = {}
        if self.execution_status == "completed":
            data = {
                "failed_cell_count": self.failed_cell_count,
                "total_cell_count": self.total_cell_count,
                "fraction_failed": self.failed_cell_count / self.total_cell_count,
                "grid_resolution": self.grid_resolution,
            }

            if self.spatial_qajson:
                data["map"] = self.tiles_geojson
                data["extents"] = self.extents_geojson

        if self.execution_status == "aborted" or self.execution_status == "failed":
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[self.error_message] if self.error_message else None,
                data=data,
                check_state=GridCheckState.cs_fail,
            )
        elif self.failed_cell_count > 0:
            percent_failed = self.failed_cell_count / self.total_cell_count * 100
            msg = (
                f"{self.failed_cell_count} nodes failed the resolution check "
                f"this represents {percent_failed:.1f}% of all nodes within "
                "data."
            )
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[msg],
                data=data,
                check_state=GridCheckState.cs_fail,
            )
        else:
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[],
                data=data,
                check_state=GridCheckState.cs_pass,
            )
