'''
Definition of Grid Checks implemented in mbesgc
'''
from __future__ import annotations
from datetime import datetime
from enum import Enum
from pathlib import PurePath
from tempfile import TemporaryDirectory
import distutils
from typing import List, Any
from ausseabed.qajson.model import QajsonParam, QajsonOutputs
from .data import InputFileDetails
from .tiling import Tile

import os
import scipy.ndimage as ndimage
from osgeo import ogr


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

        self.spatial_export = False
        self.spatial_export_location = None
        self.spatial_qajson = True

        self.temp_dir = None
        self.temp_base_dir = None
        self.temp_dir_all = []

    def check_started(self):
        '''
        to be called before first call to checkc `run` function. Initialises
        the check
        '''
        if self.spatial_export_location is not None and self.spatial_export:
            # create a temp folder to keep all the tiled chunks of data
            p = PurePath(self.spatial_export_location)
            self.temp_dir = TemporaryDirectory()
            d = os.path.join(self.temp_dir.name, p.parent.name)
            d = os.path.join(d, p.name)
            if not os.path.exists(d):
                os.makedirs(d)
            self.temp_base_dir = d
            self.temp_dir_all.append(self.temp_dir)

        if self.start_time is None:
            # only set the start time if the start_time is None, as this
            # function may be called multiple times as each tile is processed
            self.start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        self.execution_status = 'running'

    def _merge_temp_dirs(self, last_check: GridCheck):
        self.temp_dir_all.extend(last_check.temp_dir_all)

    def _get_tmp_file(self, name: str, extension: str, tile: Tile) -> str:
        n = f"{name}_{tile.min_x}_{tile.min_y}.{extension}"
        return os.path.join(self.temp_base_dir, n)

    def _move_tmp_dir(self):
        distutils.dir_util.copy_tree(
            self.temp_base_dir,
            self.spatial_export_location)
        # shutil.copy(self.temp_base_dir, self.spatial_export_location)

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
            pinkchart,
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
            pinkchart (numpy): Pink Chart data
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

    def _simplify_layer(self, in_lyr, out_lyr, simplify_distance):
        '''
        Creates a simplified layer from an input layer using GDAL's
        simplify function
        '''
        for in_feat in in_lyr:
            geom = in_feat.GetGeometryRef()
            simple_geom = geom.SimplifyPreserveTopology(simplify_distance)
            self.__add_geom(simple_geom, out_lyr)

    def __add_geom(self, geom, out_lyr):
        feature_def = out_lyr.GetLayerDefn()
        out_feat = ogr.Feature(feature_def)
        out_feat.SetGeometry(geom)
        out_lyr.CreateFeature(out_feat)

    def _grow_pixels(self, data_array, pixel_growth):
        '''
        Used for boolean data arrays, will grow out a non-zero (true) pixel
        value by a certain number of pixels. Helps fatten up areas that fail
        a check and supports more simple ploygonised geometry.
        '''
        def test_func(values):
            return values.max()

        return ndimage.generic_filter(
            data_array,
            test_func,
            size=(pixel_growth, pixel_growth)
        )
