from pathlib import Path
import logging
import math
import numpy as np
import os

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from typing import List

from .tiling import get_tiles


logger = logging.getLogger(__name__)


class Extents():
    """
    Extents object for managing bounding box type information of raster datasets
    """

    @classmethod
    def from_geotransform(cls, geo_transform: List[float], sizeX: int, sizeY: int):
        """
        Builds an extent object based on a gdal geotransform array and the raster
        size (given in pixels)
        """
        min_x = geo_transform[0]
        max_y = geo_transform[3]
        max_x = min_x + geo_transform[1] * sizeX
        min_y = max_y + geo_transform[5] * sizeY
        return Extents(min_x, min_y, max_x, max_y)

    def __init__(self, min_x: float, min_y: float, max_x: float, max_y:float) -> None:
        self.min_x: float = min_x
        self.min_y: float = min_y
        self.max_x: float = max_x
        self.max_y: float = max_y

    def to_list(self) -> List[float]:
        return [self.min_x, self.min_y, self.max_x, self.max_y]

    def __eq__(self, other: "Extents") -> bool:
        return (
            self.min_x == other.min_x and
            self.min_y == other.min_y and
            self.max_x == other.max_x and
            self.max_y == other.max_y
        )
    
    def __str__(self) -> str:
        return f"{self.min_x}, {self.min_y}, {self.max_x}, {self.max_y}"


class PinkChartProcessor():
    """
    Pre processes the grid data to accomodate the inclusion of the pink
    chart. Pink chart, or the coverage area, is the area over which the
    QA checks will be run.
    """

    def __init__(
            self,
            source_rasters: List[Path],
            source_pinkchart: Path,
            output_rasters: List[Path],
            output_pinkchart_raster: Path
        ) -> None:
        """
        Requires a paths to `source_rasters` that includes the bathymetry data, more
        than one source_raster may be provided if the bathy data has different bands
        split into multiple files. It is not intended to cater for different bathy
        datasets at one.
        The `source_pinkchart` is a vector file including features that make up
        the coverage area of interest.
        `output_rasters` is a one-to-one mapping of the `source_rasters` but each
        source raster has been aligned with the rasterised version of the pinkchart.
        The `output_pinkchart_raster` is path to where a rasterised version of the
        pink chart will be created.
        """
        self.raster_files = source_rasters
        self.output_raster_files = output_rasters

        self.pinkchart_file = source_pinkchart
        self.rasterised_file = output_pinkchart_raster

        # size of the output rasters in pixels
        self.size_x = None
        self.size_y = None

        self.geotransform = None
        
    def _calc_ideal_value(
            self,
            res: float,
            source_val: float,
            target_val: float,
            is_min: bool
        ) -> float:
        """
        Calculates the value that should be used as an extent based on a source
        value (where it gets the alignment from), a resolution (of the raster data),
        and a target value. The returned value will always be the `source_val` +/- 
        a multiple of the `res`.
        """
        d = source_val - target_val        
        d_units = d / res

        if is_min:
            d_units = math.ceil(d_units)
        elif not is_min:
            d_units = math.floor(d_units)

        return source_val - d_units * res

    def _calc_ideal_extents(
            self,
            source_res_x: float,
            source_res_y: float,
            source_extents: Extents,
            target_extents: Extents
        ) -> Extents:
        """
        Generate a new ideal set of extents that allows a raster with source_res_x,
        source_res_y and source extents to fit into it without resampling. The returned
        extents will always be larger than the source_extents.

        Assumes consistent projection.
        """

        i_min_x = self._calc_ideal_value(source_res_x, source_extents.min_x, target_extents.min_x, True)
        i_min_y = self._calc_ideal_value(source_res_y, source_extents.min_y, target_extents.min_y, True)
        i_max_x = self._calc_ideal_value(source_res_x, source_extents.max_x, target_extents.max_x, False)
        i_max_y = self._calc_ideal_value(source_res_y, source_extents.max_y, target_extents.max_y, False)

        return Extents(i_min_x, i_min_y, i_max_x, i_max_y)

    def _rasterize(
            self,
            dataset: gdal.Dataset,
            layer: ogr.Layer,
        ) -> None:
        """
        Rasterizes the `layer` into the `dataset`. All features are burnt into the
        raster and a value of 1 is used.
        """
        gdal.RasterizeLayer(dataset, [1], layer, burn_values=[1])

    def _warp(
            self,
            source: gdal.Dataset,
            output: Path,
            extents: Extents,
            res_x: float,
            res_y: float,
            cutline_dataset_name: str = None,
            cutline_layer_name: str = None,
        ) -> None:
        """
        Performs a GDAL warp operation. Pushes the `source` data into the extents and
        resolution given.
        Will use cutline to clips source raster data if cutline dataset and layer
        details are given.
        """
        # get the datatype from the first band, in a tiff file all bands share the
        # same type
        band: gdal.Band = source.GetRasterBand(1)
        dt = band.DataType
        nodata = band.GetNoDataValue()

        drv_tiff: gdal.Driver = gdal.GetDriverByName("GTiff")
        out_raster: gdal.Dataset = drv_tiff.Create(
            str(output),
            int((extents.max_x - extents.min_x) / res_x),
            int((extents.max_y - extents.min_y) / res_y),
            source.RasterCount,
            dt,
            options=["COMPRESS=DEFLATE"]
        )
        gt = [
            extents.min_x,
            res_x,
            0,
            extents.max_y,
            0,
            -res_y,
        ]

        out_raster.SetGeoTransform(gt)
        out_raster.SetProjection(source.GetProjection())

        # we need to explicitly set the no data for each band, passing this in as
        # a WarpOption isn't enough
        for band_index in range(1, source.RasterCount+1):
            aband: gdal.Band = out_raster.GetRasterBand(band_index)
            aband.SetNoDataValue(nodata)

        if (cutline_dataset_name is None or cutline_layer_name is None):
            options = gdal.WarpOptions(srcNodata=nodata, dstNodata=nodata)
        else:
            options = gdal.WarpOptions(
                srcNodata=nodata,
                dstNodata=nodata,
                cutlineDSName=cutline_dataset_name,
                cutlineLayer=cutline_layer_name
            )
        gdal.Warp(out_raster, source, options=options)

        out_raster.FlushCache()

        self.size_x = out_raster.RasterXSize
        self.size_y = out_raster.RasterYSize
        self.geotransform = gt

        del out_raster

    def process(self):
        """
        Generate a rasterised version of the source pinkchart and updated
        versions of the source raster data that lines up with the pinkchart
        raster
        """
        # Open up one of the source raster files to get some details about the
        # dataset, we use these later to calculate extents for the pinkchart raster
        # that align with this raster
        data_raster: gdal.Dataset = gdal.Open(str(self.raster_files[0].absolute()))
        data_raster_proj: str = data_raster.GetProjection()
        data_raster_size_x = data_raster.RasterXSize
        data_raster_size_y = data_raster.RasterYSize
        data_raster_gt = data_raster.GetGeoTransform()
        data_raster_extents = Extents.from_geotransform(data_raster_gt, data_raster_size_x, data_raster_size_y)
        res_x = abs(data_raster_gt[1])
        res_y = abs(data_raster_gt[5])

        # get the current extents of the pinkchart vector
        # NOTE: It has been observed that sometimes the extents of a shapefile don't necessarily
        # match that of the features within the shapefile. One shapefile presented significantly
        # larger extents than any feature it contained, as such more processing (and memory) was
        # used than really required.
        pc_vector: ogr.DataSource = ogr.Open(str(self.pinkchart_file.absolute()))
        pc_layer: ogr.Layer = pc_vector.GetLayer()
        pc_layer_extent_values = pc_layer.GetExtent(force=1)
        pc_layer_min_x, pc_layer_max_x, pc_layer_min_y, pc_layer_max_y = pc_layer_extent_values

        ogr_srs_raster = osr.SpatialReference()
        ogr_srs_raster.ImportFromWkt(data_raster_proj)

        ogr_srs_pc = pc_layer.GetSpatialRef()

        if str(ogr_srs_pc) == str(ogr_srs_raster):
            # then don't do a coordinate transform as it is in the same CRS and this has some
            # undesired side effects
            pc_layer_min_x_trans, pc_layer_min_y_trans  = (pc_layer_min_x, pc_layer_min_y)
            pc_layer_max_x_trans, pc_layer_max_y_trans = (pc_layer_max_x, pc_layer_max_y)
        else:
            transform = osr.CoordinateTransformation(ogr_srs_pc, ogr_srs_raster)
            pc_layer_min_x_trans, pc_layer_min_y_trans, _  = transform.TransformPoint(pc_layer_min_x, pc_layer_min_y)
            pc_layer_max_x_trans, pc_layer_max_y_trans, _  = transform.TransformPoint(pc_layer_max_x, pc_layer_max_y)

        pc_layer_extents = Extents(pc_layer_min_x_trans, pc_layer_min_y_trans, pc_layer_max_x_trans, pc_layer_max_y_trans)

        # expand out the extents of the pinkchart extents so that these extents
        # will line up with the source raster data
        tapped_extents = self._calc_ideal_extents(res_x, res_y, data_raster_extents, pc_layer_extents)

        # create a new raster, the pinkchart raster data will be written to this
        drv_tiff: gdal.Driver = gdal.GetDriverByName("GTiff")
        pc_raster: gdal.Dataset = drv_tiff.Create(
            str(self.rasterised_file.absolute()),
            int((tapped_extents.max_x - tapped_extents.min_x) / res_x),
            int((tapped_extents.max_y - tapped_extents.min_y) / res_y),
            1,
            gdal.gdalconst.GDT_Byte,
            options=["COMPRESS=DEFLATE"]
        )
        pc_raster_gt = list(data_raster_gt)
        pc_raster_gt[0] = tapped_extents.min_x
        pc_raster_gt[3] = tapped_extents.max_y
        pc_raster.SetGeoTransform(pc_raster_gt)
        pc_raster.SetProjection(data_raster_proj)

        # rasterize the pink chart
        self._rasterize(pc_raster, pc_layer)

        # GDAL can be a little picky about when it actually writes data to
        # file. Runnning these following lines seems to be the best way to
        # ensure GDAL actually writes the data.
        pc_raster.FlushCache()
        del pc_raster

        pc_ds: gdal.Dataset = gdal.Open(str(self.rasterised_file.absolute()))
        # we create this raster above, so it's always band 1
        pc_band: gdal.Band = pc_ds.GetRasterBand(1)

        # now warp each of the source rasters into the ideal extents of the
        # pink chart
        for index, src_filename in enumerate(self.raster_files):
            dest_filename = self.output_raster_files[index]
            data_raster: gdal.Dataset = gdal.Open(str(src_filename.absolute()))

            self._warp(
                data_raster,
                dest_filename,
                tapped_extents,
                res_x,
                res_y,
                cutline_dataset_name=str(self.pinkchart_file.absolute()),
                cutline_layer_name=pc_layer.GetName()
            )

            # warp the source data into a dataset with the same extents as the
            # pink chart. While it has the same extents of the pink chart, it
            # isn't clipped to the pink chart at this stage.
            fn, ext = os.path.splitext(dest_filename)
            warped_filename = f"{fn}.warp{ext}"
            self._warp(
                data_raster,
                warped_filename,
                tapped_extents,
                res_x,
                res_y
            )

            ds_source: gdal.Dataset = gdal.Open(warped_filename)
            ds_source_datatype = ds_source.GetRasterBand(1).DataType

            # create a new dataset that will include a clipped version of the input
            # data
            ds_output: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(
                str(dest_filename.absolute()),
                ds_source.RasterXSize,
                ds_source.RasterYSize,
                ds_source.RasterCount,
                ds_source_datatype,
                options=["COMPRESS=DEFLATE"]
            )
            ds_output.SetProjection(pc_ds.GetProjection())
            ds_output.SetGeoTransform(pc_ds.GetGeoTransform())

            # we're going to process this data on a block by block basis, as this is
            # a nice compromise between being fast and not loading the entire dataset
            # into memory
            size_x = ds_source.RasterXSize
            size_y = ds_source.RasterYSize
            tile_size_x, tile_size_y = ds_source.GetRasterBand(1).GetBlockSize()
            tiles = get_tiles(
                0,
                0,
                size_x,
                size_y,
                tile_size_x,
                tile_size_y
            )

            # copy description and nodata value to each output band
            for band_index in range(1, ds_source.RasterCount+1):
                band_source: gdal.Band = ds_source.GetRasterBand(band_index)
                band_output: gdal.Band = ds_output.GetRasterBand(band_index)
                band_output.SetNoDataValue(band_source.GetNoDataValue())
                band_output.SetDescription(band_source.GetDescription())

            # now loop through each on of the tiles (geotiff blocks)
            for i, tile in enumerate(tiles):
                # read the pink chart (coverage area) data for this tile
                # we only need to read this once for all the bands
                pc_data = np.array(pc_band.ReadAsArray(
                    tile.min_x,
                    tile.min_y,
                    tile.max_x - tile.min_x,
                    tile.max_y - tile.min_y
                ))

                # loop through each one of the bands in the source dataset
                for band_index in range(1, ds_source.RasterCount+1):
                    # load the input data
                    band_source: gdal.Band = ds_source.GetRasterBand(band_index)
                    band_source_data = np.array(band_source.ReadAsArray(
                        tile.min_x,
                        tile.min_y,
                        tile.max_x - tile.min_x,
                        tile.max_y - tile.min_y
                    ))
                    
                    # this is where the data is clipped to the pink chart (coverage
                    # area) dataset. Basically we just replace all the indexes in the
                    # band_source_data array where the pink chart is 0 with nodata
                    # and leave the rest of the source data unchanged.
                    band_source_data[pc_data == 0] = band_source.GetNoDataValue()

                    # now write the modified source data back to the output file
                    band_output: gdal.Band = ds_output.GetRasterBand(band_index)
                    band_output.WriteRaster(
                        tile.min_x, tile.min_y,
                        tile.width, tile.height,
                        band_source_data.tobytes(),
                        tile.width, tile.height,
                        ds_source_datatype
                    )

