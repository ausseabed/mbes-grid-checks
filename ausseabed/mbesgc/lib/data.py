"""
Module handles mapping input files of varying types (tif, bag) to conventions
implemented within mdes-grid-checks such as the mapping of band numbers
to what they represent
"""

from __future__ import annotations
from enum import Enum
from osgeo import gdal, osr, ogr
from geojson import MultiPolygon
from typing import Tuple, List, Type, Dict, TYPE_CHECKING
import os
import os.path
from pathlib import Path

from ausseabed.qajson.model import (
    QajsonCheck,
    QajsonParam,
    QajsonRoot,
    QajsonQa,
    QajsonDataLevel,
    QajsonInfo,
    QajsonGroup,
    QajsonInputs,
    QajsonFile,
)
from ausseabed.qajson.utils import latest_schema_version

GdalGeoTransform = tuple[float, float, float, float, float, float]


class BandType(str, Enum):
    depth = "depth"
    density = "density"
    uncertainty = "uncertainty"
    pinkChart = "pinkChart"


class InputFileDetailsError(RuntimeError):
    """Error raised when issues are identified with InputFileDetails"""

    pass


class InputFileDetails:
    def __init__(self) -> None:
        self.size_x: int = 0
        self.size_y: int = 0

        # geotransform of the raster data file. This is needed to georeference
        # the output data arrays later
        self.geotransform: GdalGeoTransform | None = None
        self.projection: str | None = None

        self.input_band_details: list[tuple[str, int, BandType]] = []

        # pink chart filename, if one was given
        self.pink_chart_filename: str | None = None

        # list of check uuids that this input file will be run through
        self.check_ids_and_params: list[Tuple[str, List[QajsonParam]]] = []

        # keep track of the qajson check entry so that results can be written
        # to it.
        self.qajson_checks: List[QajsonCheck] = []

        # keep track of the original InputFileDetails object
        # this is set for all clones
        self.source: InputFileDetails | None = None

    def add_band_details(
        self,
        input_file: str,
        band_index: int,
        band_type: BandType,
    ) -> None:
        ibd = (input_file, band_index, band_type)
        self.input_band_details.append(ibd)

    @property
    def band_count(self):
        return len(self.input_band_details)

    def has_same_inputs(self, other: "InputFileDetails") -> bool:
        """compares this InputFileDetails against another InputFileDetails to
        see if they have the same set of input files/bands/band types. Details do
        not need to be in order.
        """
        for o_input_file, o_band_index, o_band_type in self.input_band_details:
            found = False
            for i_input_file, i_band_index, i_band_type in other.input_band_details:
                if (
                    i_input_file == o_input_file
                    and i_band_index == o_band_index
                    and i_band_type == o_band_type
                ):
                    found = True
            if not found:
                return False
        return True

    def clear_band_details(self):
        self.input_band_details.clear()

    def get_band(self, band_type: BandType) -> Tuple[str, int] | Tuple[None, None]:
        band_details = next(
            (ibd for ibd in self.input_band_details if ibd[2] == band_type), None
        )
        if band_details is None:
            return None, None
        else:
            return band_details[0], band_details[1]

    def validate(self) -> Tuple[bool, List[str]]:
        """Run a series of checks on this input to identify any issues that
        would require the user to modify input data.
        Returns a boolean indicating if validation was passed and a list of
        messages identifying the validation issues.
        """
        validation_messages: List[str] = []
        # there should be at max 3 input bands (depth, density, uncertainty)
        if len(self.input_band_details) > 3:
            validation_messages.append(
                f"A maximum of 3 input bands is expected, "
                f"but {len(self.input_band_details)} were provided."
            )

        # there should be no duplication in input bands
        # build up a dict that includes a count of each band type
        bandtypes_and_count: Dict[BandType, int] = {}
        for _, _, band_type in self.input_band_details:
            if band_type in bandtypes_and_count:
                bandtypes_and_count[band_type] = bandtypes_and_count[band_type] + 1
            else:
                bandtypes_and_count[band_type] = 1

        # now raise errors as appropriate
        dup_msg: List[str] = []
        for band_type, count in bandtypes_and_count.items():
            if count > 1:
                dup_msg.append(f"{count} bands were found with type {band_type}")
        msg = f"Found more than 1 band defined with the same data type ({', '.join(dup_msg)})"
        if len(dup_msg) >= 1:
            validation_messages.append(msg)

        # check that all input bands have nodata values, all bands have
        # the same size and resolution
        size_x = None
        size_y = None
        res_x = None
        res_y = None

        for filename, band_index, band_type in self.input_band_details:
            ds = gdal.Open(filename)
            if ds is None:
                raise RuntimeError(f"Could not open {filename}")

            proj_str = ds.GetProjection()
            ogr_srs = osr.SpatialReference()
            try:
                ogr_srs.ImportFromWkt(proj_str)
                valid = ogr_srs.Validate()
                if valid != ogr.OGRERR_NONE:
                    raise RuntimeError("CRS failed validation")
            except Exception:
                validation_messages.append(
                    f"{filename} has invalid Coordinate Reference System (CRS) information"
                )
                validation_messages.append(f'CRS string is "{proj_str}"')

            band: gdal.Band = ds.GetRasterBand(band_index)
            if band.GetNoDataValue() is None:
                msg = f"band index {band_index} in file {filename} has no nodata value assigned"
                validation_messages.append(msg)

            if res_x is None or res_y is None:
                _, res_x, _, _, _, res_y = ds.GetGeoTransform()

            else:
                _, ds_res_x, _, _, _, ds_res_y = ds.GetGeoTransform()
                if ds_res_x != res_x:
                    msg = (
                        f"Input raster bands differ in x resolution ({res_x} != "
                        f"{ds_res_x}) and should be the same across all "
                        "bands in a single dataset"
                    )
                    validation_messages.append(msg)
                if ds_res_y != res_y:
                    msg = (
                        f"Input raster bands differ in y resolution ({res_y} != "
                        f"{ds_res_y}) and should be the same across all "
                        "bands in a single dataset"
                    )
                    validation_messages.append(msg)

            if size_x is None:
                size_x = ds.RasterXSize
            elif size_x != ds.RasterXSize:
                msg = (
                    f"Input raster bands differ in x size ({size_x} != "
                    f"{ds.RasterXSize}) and should be the same across all "
                    "bands in a single dataset"
                )
                validation_messages.append(msg)

            if size_y is None:
                size_y = ds.RasterYSize
            elif size_y != ds.RasterYSize:
                msg = (
                    f"Input raster bands differ in y size ({size_y} != "
                    f"{ds.RasterYSize}) and should be the same across all "
                    "bands in a single dataset"
                )
                validation_messages.append(msg)

        # if there are no validation messages, then assume validation is ok
        return len(validation_messages) == 0, validation_messages

    def get_common_filename(self) -> str:
        """For multi-band tiffs this will return the name of the file (no
        extension or full path). For single-band tiffs where there are multiple
        files this will include the common component of the name shared by all
        filenames.
        """
        if len(self.input_band_details) == 0:
            # no files, shouldn't ever happen
            return ""
        elif len(self.input_band_details) == 1:
            # then only a single file has been specified
            input_file, _, _ = self.input_band_details[0]
            fn = Path(input_file).stem
            return fn
        else:
            all_names = [
                Path(input_file).stem for input_file, _, _ in self.input_band_details
            ]
            min_length = min([len(name) for name in all_names])
            end_pos = 0
            for i in range(min_length):
                char_from_first = all_names[0][i]
                if all([char_from_first == test_name[i] for test_name in all_names]):
                    # then chars are ok, so continue on
                    end_pos += 1
                    # pass
                else:
                    # then this char is not the same, so we need to break
                    break

            a_name = all_names[0]
            if end_pos < 5:
                # then there was insufficient common characters at the start of the file names
                # so just use the first filename
                # the 5 characters in this case was chosen arbitrarily
                return all_names[0]
            else:
                return a_name[:end_pos]

    def get_extents_feature(self) -> MultiPolygon:
        """Gets the extents of this input file based on the geotransform as a geojson feature"""
        assert self.geotransform is not None
        minx = self.geotransform[0]
        maxy = self.geotransform[3]
        maxx = minx + self.geotransform[1] * self.size_x
        miny = maxy + self.geotransform[5] * self.size_y

        ogr_srs = osr.SpatialReference()
        ogr_srs.ImportFromWkt(self.projection)
        ogr_srs_out = osr.SpatialReference()
        ogr_srs_out.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(ogr_srs, ogr_srs_out)

        transformed_bounds = transform.TransformBounds(minx, miny, maxx, maxy, 2)
        minx, miny, maxx, maxy = transformed_bounds

        polygon = MultiPolygon(
            [
                [
                    [
                        (miny, minx),
                        (miny, maxx),
                        (maxy, maxx),
                        (maxy, minx),
                    ]
                ]
            ]
        )

        return polygon

    def __repr__(self):
        bd = [f"  {fn} {bi} {bt}" for (fn, bi, bt) in self.input_band_details]
        bds = "\n".join(bd)
        return (
            f"size: {self.size_x}, {self.size_y}\n"
            f"pink chart file: {self.pink_chart_filename}\n"
            f"{bds}"
        )

    def clone(self):
        """Returns a new instance of InputFileDetails. Note: does not
        clone input_band_details list.
        """
        ifd = InputFileDetails()
        ifd.source = self
        ifd.size_x = self.size_x
        ifd.size_y = self.size_y
        ifd.geotransform = self.geotransform
        ifd.projection = self.projection
        ifd.pink_chart_filename = self.pink_chart_filename
        ifd.check_ids_and_params = self.check_ids_and_params
        ifd.qajson_checks = list(self.qajson_checks)
        # only thing we don't clone
        ifd.input_band_details = []

        return ifd


def _get_tiff_details(input_files) -> InputFileDetails:
    """
    Single tiffs include all 3 bands
    """
    ifd = InputFileDetails()

    for input_file in input_files:
        raster: gdal.Dataset = gdal.Open(input_file)
        if raster is None:
            raise RuntimeError(f"input file {input_file} could not be opened")
        size_x = raster.RasterXSize
        size_y = raster.RasterYSize
        geotransform = raster.GetGeoTransform()
        projection = raster.GetProjection()

        # assumes the size,proj,geotransform of all input bands is the
        # same, which is not enforced when there are multiple input
        # files
        ifd.size_x = size_x
        ifd.size_y = size_y
        ifd.geotransform = geotransform
        ifd.projection = projection

        file_added = False
        for band_index in range(1, raster.RasterCount + 1):
            band: gdal.Band = raster.GetRasterBand(band_index)
            band_name: str = band.GetDescription().lower()
            if "depth" in band_name:
                ifd.add_band_details(input_file, band_index, BandType.depth)
                file_added = True
            elif "density" in band_name:
                ifd.add_band_details(input_file, band_index, BandType.density)
                file_added = True
            elif "uncertainty" in band_name:
                ifd.add_band_details(input_file, band_index, BandType.uncertainty)
                file_added = True

        name_only = Path(input_file).stem.lower()

        if file_added:
            # already added, so skip this
            pass
        elif raster.RasterCount == 1 and "depth" in name_only:
            # then the band type is assumed by the filename
            ifd.add_band_details(input_file, 1, BandType.depth)
        elif raster.RasterCount == 1 and "density" in name_only:
            ifd.add_band_details(input_file, 1, BandType.density)
        elif raster.RasterCount == 1 and "uncertainty" in name_only:
            ifd.add_band_details(input_file, 1, BandType.uncertainty)
        else:
            # then we need to assume the default ordering of bands
            # 1. depth
            # 2. density
            # 3. uncertainty

            # first clear out anything that may have been added. Users must
            # label all bands, or no labels will be used at all
            ifd.clear_band_details()
            for band_index in range(1, raster.RasterCount):
                if band_index == 1:
                    band_type = BandType.depth
                elif band_index == 2:
                    band_type = BandType.density
                else:
                    band_type = BandType.uncertainty
                ifd.add_band_details(input_file, band_index, band_type)

    return ifd


def _get_bag_details(input_file):
    """
    Bag file bands are split across multiple files. This function identifies
    these files, the bands within them, and their size (in px).
    """
    fn_no_extension = os.path.splitext(input_file)[0]
    input_file_density = f"{fn_no_extension}_Density.bag"
    if not os.path.isfile(input_file_density):
        raise RuntimeError(
            f"Could not find density file for bag , expected {input_file_density}"
        )

    depth_raster = gdal.Open(input_file)
    depth_size_x = depth_raster.RasterXSize
    depth_size_y = depth_raster.RasterYSize
    density_raster = gdal.Open(input_file_density)
    density_size_x = density_raster.RasterXSize
    density_size_y = density_raster.RasterYSize

    if depth_size_x != density_size_x or depth_size_y != density_size_y:
        raise RuntimeError(
            "mismatch in data sizes across depth and density inputs. "
            "Both files must have the same size."
        )

    if depth_raster.RasterCount != 2:
        raise RuntimeError(
            f"input file ({input_file}) has {depth_raster.RasterCount} bands"
            " and 2 are expected."
        )

    if density_raster.RasterCount != 2:
        raise RuntimeError(
            f"input file ({input_file_density}) has "
            f"{density_raster.RasterCount} bands and 2 are expected."
        )

    ifd = InputFileDetails()
    ifd.size_x = depth_size_x
    ifd.size_y = depth_size_y
    # band order is assumed based on convention
    ifd.add_band_details(input_file, 1, BandType.depth)
    ifd.add_band_details(input_file, 2, BandType.uncertainty)
    ifd.add_band_details(input_file_density, 1, BandType.density)
    return ifd


def get_input_details(
    inputfiles: List[str],
    relative_to: str | None = None,
) -> List[InputFileDetails]:
    """
    Extracts relevant band numbers, size(px), and appropriate file names
    from list of input files.
    """
    inputdetails = []

    # update list of input files for relative path if one has been
    # provided
    for i in range(0, len(inputfiles)):
        inputfile = inputfiles[i]
        if not os.path.isfile(inputfile) and relative_to is not None:
            test_rel_file = os.path.join(relative_to, inputfile)
            if os.path.isfile(test_rel_file):
                inputfiles[i] = test_rel_file

    if len(inputfiles) == 0:
        raise RuntimeError("No gridded input files provided")

    _first = inputfiles[0].lower()
    if _first.endswith(".tif") or _first.endswith(".tiff"):
        # assume all files are tif files if the first one is
        tifdetails = _get_tiff_details(inputfiles)
        inputdetails.append(tifdetails)
    elif _first.endswith("_density.bag"):
        # ignore these bag files, we'll handle these in the next if case
        pass
    elif _first.endswith(".bag"):
        bagdetails = _get_bag_details(inputfiles[0])
        inputdetails.append(bagdetails)

    return inputdetails


def inputs_from_qajson_checks(
    qajson_checks: List[QajsonCheck],
    relative_to: str | None = None,
) -> List[InputFileDetails]:

    inputs: List[InputFileDetails] = []
    for qajson_check in qajson_checks:
        check_id = qajson_check.info.id
        assert qajson_check.inputs is not None
        grid_filenames = [
            qajson_file.path
            for qajson_file in qajson_check.inputs.files
            if qajson_file.file_type == "Survey DTMs"
        ]
        check_inputs = get_input_details(grid_filenames, relative_to)

        pc_filenames = [
            qajson_file.path
            for qajson_file in qajson_check.inputs.files
            if qajson_file.file_type == "Coverage Area"
        ]

        # loop through all the new sets of InputFileDetails that have been identified
        for ci in check_inputs:
            cid_and_params = (check_id, qajson_check.inputs.params)

            added = False
            # now loop through all the existing InputFileDetails
            for existing_ifd in inputs:
                if existing_ifd.has_same_inputs(ci):
                    # then we've already identified this as a set of input data that
                    # needs to be processed, so instead of duplicating it in the list
                    # (that would result in it being re-read) we just add another check
                    # and set of input params to it
                    existing_ifd.check_ids_and_params.append(cid_and_params)
                    existing_ifd.qajson_checks.append(qajson_check)
                    added = True
            if not added:
                ci.check_ids_and_params.append(cid_and_params)
                ci.qajson_checks.append(qajson_check)
                if len(pc_filenames) > 0:
                    ci.pink_chart_filename = pc_filenames[0]
                inputs.append(ci)

    return inputs


def qajson_from_inputs(
    input: InputFileDetails,
    check_classes: List[Type["GridCheck"]],
) -> QajsonRoot:

    checks = []
    for check_class in check_classes:
        info = QajsonInfo(
            id=check_class.id,
            name=check_class.name,
            description=None,
            version=check_class.version,
            group=QajsonGroup("", "", ""),
        )

        input_file_path, _, _ = input.input_band_details[0]
        input_file = QajsonFile(
            path=input_file_path,
            file_type="Survey DTMs",
            description=None,
        )

        inputs = QajsonInputs(
            files=[input_file],
            params=check_class.input_params,
        )
        check = QajsonCheck(
            info=info,
            inputs=inputs,
            outputs=None,
        )
        checks.append(check)

    datalevel = QajsonDataLevel(checks=checks)
    qa = QajsonQa(
        version=latest_schema_version(),
        raw_data=QajsonDataLevel([]),
        survey_products=datalevel,
        chart_adequacy=None,
    )
    root = QajsonRoot(qa)

    return root


# function is used by the QAX plugin, we have it here as the same code is needed
# in the finder-grid-check plugin
def get_file_details(filename: str) -> str:
    """Return some details about the raster file that's been provided. In this
    case a list of the bands, and the resolution of the dataset.
    """
    ifds = get_input_details([filename])
    band_types: list[str | BandType] = []
    for ifd in ifds:
        for _, _, band_type in ifd.input_band_details:
            band_types.append(band_type)
        if len(ifd.input_band_details) == 0:
            band_types.append("Type not identified")
        band_types.append(f"{ifd.size_x}{chr(0x00D7)}{ifd.size_y}")

    return "\n".join(band_types)


if TYPE_CHECKING:
    from .gridcheck import GridCheck
