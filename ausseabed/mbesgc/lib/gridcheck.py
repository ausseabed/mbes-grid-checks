'''
Definition of Grid Checks implemented in mbesgc
'''
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from ausseabed.qajson.model import QajsonParam, QajsonOutputs, QajsonExecution
from .data import InputFileDetails
from .tiling import Tile

import collections
import numpy as np
import numpy.ma as ma
import geojson
from geojson import MultiPolygon
from osgeo import gdal, ogr, osr
from affine import Affine


class GridCheckState(str, Enum):
    cs_pass = 'pass'
    cs_warning = 'warning'
    cs_fail = 'fail'


class GridCheckResult:
    '''
    Encapsulates the various outputs from a grid check
    '''
    def __init__(
            self,
            state: GridCheckState,
            messages: List = []):
        self.state = state
        self.messages = messages


class GridCheck:
    '''
    Base class for all grid checks
    '''

    def __init__(self, input_params: List[QajsonParam]):
        self.input_params = input_params

        # used to record time check was started, and when it completed
        self.start_time = None
        self.end_time = None

        # did the check execute successfully
        self.execution_status = 'draft'
        # any error messages that occured during running of the check
        self.error_message = None

    def check_started(self):
        '''
        to be called before first call to checkc `run` function. Initialises
        the check
        '''
        if self.start_time is None:
            # only set the start time if the start_time is None, as this
            # function may be called multiple times as each tile is processed
            self.start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        self.execution_status = 'running'

    def check_ended(self):
        '''
        to be called after last call to check `run` function. Finalises
        the check
        '''
        self.end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

        if self.execution_status == 'running':
            self.execution_status = 'completed'

    def get_param(self, param_name: str) -> Any:
        ''' Gets the parameter value from the list of QajsonParams. Returns
        None if no parameter found. If multiple parameters share the same
        name, the first will be returned.
        '''
        if len(self.input_params) == 0:
            return None

        param = next(
            param
            for param in self.input_params
            if param.name == param_name
        )
        if param is None:
            return None
        else:
            return param.value

    def run(
            ifd: InputFileDetails,
            tile: Tile,
            depth,
            density,
            uncertainty,
            progress_callback=None):
        '''
        Abstract function definition for how each check should implement its
        run method.

        Args:
            ifd (InputFileDetails): details of the file the data has been
                loaded from
            tile (Tile): pixel coordinates of the data loaded into the input
                arrays (depth, density, uncertainty)
            depth (numpy): Depth/elevation data
            density (numpy): Density data
            uncertainty (numpy): Uncertainty data
            progress_callback (function): optional callback function to
                indicate progress to caller

        Returns:
            nothing?

        '''
        raise NotImplementedError

    def merge_results(self, last_check: GridCheck) -> None:
        '''
        Abstract function definition for how each check should merge results.

        Checks are run on a tile by tile (chunck of input data) basis. To get
        the complete results the results from each chunk need to be merged.
        This merging needs to be implemented within this function.

        Must be overwritten by child classes
        '''
        raise NotImplementedError

    def get_outputs(self) -> QajsonOutputs:
        '''
        Gets the results of this check in a QaJson format
        '''
        raise NotImplementedError


class DensityCheck(GridCheck):
    '''
    Performs check based on density data and input arrays. This looks to
    determine if each node (grid cell) statisfies a minimum count (the value
    stored in the density layer) and whether a percentage of nodes overall
    satisfy a different minumum count.
    '''

    id = '5e2afd8a-2ced-4de8-80f5-111c459a7175'
    name = 'Density Check'
    version = '1'
    input_params = [
        QajsonParam("Minimum Soundings per node", 5),
        QajsonParam("Minimum Soundings per node at percentage", 5),
        QajsonParam("Minumum Soundings per node percentage", 95),
    ]

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        # if any node has fewer soundings than this value, it will fail
        self._min_spn = self.get_param('Minimum Soundings per node')

        # if a percentage `_min_spn_p` of all nodes is above a threshold
        # `_min_spn_ap` then this check will also fail
        self._min_spn_p = self.get_param(
            'Minumum Soundings per node percentage')
        self._min_spn_ap = self.get_param(
            'Minimum Soundings per node at percentage')

        self.tiles_geojson = MultiPolygon()

    def run(
            self,
            ifd: InputFileDetails,
            tile: Tile,
            depth,
            density,
            uncertainty,
            progress_callback=None):
        # generate histogram of counts
        # unique_vals will be the soundings per node
        # unique_counts is the total number of times the unique_val soundings
        # count was found.
        unique_vals, unique_counts = np.unique(density, return_counts=True)
        hist = {}
        for (val, count) in zip(unique_vals, unique_counts):
            if type(val) is ma.core.MaskedConstant:
                continue
            # following gets serialized to JSON and as numpy types are not
            # supported by default we convert the float32 and int64 types to
            # plain python ints
            hist[int(val)] = int(count)

        self.density_histogram = hist

        bad_cells_mask = density < self._min_spn
        bad_cells_mask.fill_value = False
        bad_cells_mask = bad_cells_mask.filled()
        bad_cells_mask_int16 = bad_cells_mask.astype(np.int16)

        src_affine = Affine.from_gdal(*ifd.geotransform)
        tile_affine = src_affine * Affine.translation(
            tile.min_x,
            tile.min_y
        )
        tile_ds = gdal.GetDriverByName('MEM').Create(
            '',
            tile.max_x - tile.min_x,
            tile.max_y - tile.min_y,
            1,
            gdal.GDT_Int16
        )
        tf = '/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/t.tif'
        # tile_ds = gdal.GetDriverByName('GTiff').Create(
        #     tf,
        #     tile.max_x - tile.min_x,
        #     tile.max_y - tile.min_y,
        #     1,
        #     gdal.GDT_Int16
        # )
        tile_ds.SetGeoTransform(tile_affine.to_gdal())

        tile_band = tile_ds.GetRasterBand(1)
        tile_band.WriteArray(bad_cells_mask_int16, 0, 0)
        tile_band.SetNoDataValue(0)
        tile_band.FlushCache()
        tile_ds.SetProjection(ifd.projection)


        # dst_layername = "POLYGONIZED_STUFF"
        # drv = ogr.GetDriverByName("ESRI Shapefile")
        # dst_ds = drv.CreateDataSource(tf + ".shp")
        # dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )
        ogr_srs = osr.SpatialReference()
        ogr_srs.ImportFromWkt(ifd.projection)

        ogr_driver = ogr.GetDriverByName('Memory')
        ogr_dataset = ogr_driver.CreateDataSource('shapemask')
        ogr_layer = ogr_dataset.CreateLayer('shapemask', srs=ogr_srs)

        # used the input raster data 'tile_band' as the input and mask, if not
        # used as a mask then a feature that outlines the entire dataset is
        # also produced
        gdal.Polygonize(
            tile_band,
            tile_band,
            ogr_layer,
            -1,
            [],
            callback=None
        )

        ogr_srs_out = osr.SpatialReference()
        ogr_srs_out.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(ogr_srs, ogr_srs_out)

        for feature in ogr_layer:
            transformed = feature.GetGeometryRef()
            transformed.Transform(transform)

            geojson_feature = geojson.loads(feature.ExportToJson())
            self.tiles_geojson.coordinates.extend(
                geojson_feature.geometry.coordinates
            )

        ogr_dataset.Destroy()

        # # includes only the tile boundaries, used for debug
        # tile_geojson = tile.to_geojson(ifd.projection, ifd.geotransform)
        # self.tiles_geojson.coordinates.append(tile_geojson.coordinates)

    def merge_results(self, last_check: GridCheck):
        '''
        merge the density histogram of the last_check into the current check
        '''
        self.start_time = last_check.start_time

        for soundings_count, last_count in last_check.density_histogram.items():
            if soundings_count in self.density_histogram:
                self.density_histogram[soundings_count] += last_count
            else:
                self.density_histogram[soundings_count] = last_count

        self.tiles_geojson.coordinates.extend(
            last_check.tiles_geojson.coordinates
        )

    def get_outputs(self) -> QajsonOutputs:

        execution = QajsonExecution(
            start=self.start_time,
            end=self.end_time,
            status=self.execution_status,
            error=self.error_message
        )

        if len(self.density_histogram) == 0:
            # there's no data to check, so fail
            return QajsonOutputs(
                execution=execution,
                files=None,
                count=None,
                percentage=None,
                messages=[
                    'No counts were extracted, was a valid raster provided'],
                data=None,
                check_state=GridCheckState.cs_fail
            )

        # sort the list of sounding counts and the number of occurances
        counts = collections.OrderedDict(
            sorted(self.density_histogram.items()))

        messages = []
        data = {}
        check_state = None

        lowest_sounding_count, occurances = next(iter(counts.items()))
        if lowest_sounding_count < self._min_spn:
            messages.append(
                f'Minimum sounding count of {lowest_sounding_count} occured '
                f'{occurances} times'
            )
            check_state = GridCheckState.cs_fail

        total_soundings = sum(counts.values())
        under_threshold_soundings = 0
        for sounding_count, occurances in counts.items():
            if sounding_count >= self._min_spn_ap:
                break
            under_threshold_soundings += occurances

        percentage_over_threshold = \
            (1.0 - under_threshold_soundings / total_soundings) * 100.0

        if percentage_over_threshold < self._min_spn_p:
            messages.append(
                f'{percentage_over_threshold}% of nodes were found to have a '
                f'sounding count below {self._min_spn_ap}. This is required to'
                f' be {self._min_spn_p}% of all nodes'
            )
            check_state = GridCheckState.cs_fail

        if check_state is None:
            check_state = GridCheckState.cs_pass

        str_key_counts = collections.OrderedDict()
        for key, val in counts.items():
            str_key_counts[str(key)] = val

        data['chart'] = {
            'type': 'histogram',
            'data': str_key_counts
        }

        data['map'] = self.tiles_geojson

        result = QajsonOutputs(
            execution=execution,
            files=None,
            count=None,
            percentage=None,
            messages=messages,
            data=data,
            check_state=check_state
        )

        return result


class TvuCheck(GridCheck):
    '''
    Total Vertical Uncertainty check. An allowable value is calculated on a
    per node basis that is derived from some constants (constant depth error,
    factor of depth dependent error) and the depth itself. This is then
    compared against the calculated uncertainty value within the data.
    '''

    id = 'b5c0469c-6559-4aea-bf9c-d0b337550e89'
    name = 'Total Vertical Uncertainty Check'
    version = '1'
    input_params = [
        QajsonParam("Constant Depth Error", 1.0),
        QajsonParam("Factor of Depth Dependent Errors", 1.0)
    ]

    def __init__(self, input_params: List[QajsonParam]):
        super().__init__(input_params)

        self._depth_error = self.get_param('Constant Depth Error')
        self._depth_error_factor = self.get_param(
            'Factor of Depth Dependent Errors')

    def merge_results(self, last_check: GridCheck):
        self.start_time = last_check.start_time

    def run(
            self,
            ifd: InputFileDetails,
            tile: Tile,
            depth,
            density,
            uncertainty,
            progress_callback=None):
        # not implemented yet
        pass

    def get_outputs(self) -> QajsonOutputs:

        execution = QajsonExecution(
            start=self.start_time,
            end=self.end_time,
            status=self.execution_status,
            error=self.error_message
        )

        return QajsonOutputs(
            execution=execution,
            files=None,
            count=None,
            percentage=None,
            messages=['TVU Check not implemented'],
            data=None,
            check_state=GridCheckState.cs_fail
        )
