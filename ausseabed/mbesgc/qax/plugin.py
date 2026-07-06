from typing import List, Callable, Tuple
from pathlib import Path


from ausseabed.mbesgc.lib.allchecks import all_checks
from ausseabed.mbesgc.lib.data import inputs_from_qajson_checks, get_file_details
from ausseabed.mbesgc.lib.executor import Executor

from hyo2.qax.lib.plugin import QaxCheckToolPlugin, QaxCheckReference, QaxFileType
from ausseabed.qajson.model import QajsonRoot, QajsonInputs


class MbesGridChecksQaxPlugin(QaxCheckToolPlugin):
    # supported raw data file types
    file_types = [
        QaxFileType(
            name="Shapefile",
            extension="shp",
            group="Coverage Area",
            icon="shp.png",
        ),
        QaxFileType(
            name="GeoTIFF",
            extension="tiff",
            group="Survey DTMs",
            icon="tif.png",
        ),
        QaxFileType(
            name="GeoTIFF",
            extension="tif",
            group="Survey DTMs",
            icon="tif.png",
        ),
        QaxFileType(
            name="BAG file",
            extension="bag",
            group="Survey DTMs",
            icon="bag.png",
        ),
    ]

    def __init__(self):
        super(MbesGridChecksQaxPlugin, self).__init__()
        # name of the check tool
        self.name = "MBES Grid Checks"
        self._check_references = self._build_check_references()

        self.exe = None

    def _build_check_references(self) -> List[QaxCheckReference]:
        data_level = "survey_products"
        check_refs = []

        # loop through each check class, defining the QaxCheckRefs
        for mgc_check_class in all_checks:
            cr = QaxCheckReference(
                id=mgc_check_class.id,
                name=mgc_check_class.name,
                data_level=data_level,
                description=None,
                supported_file_types=MbesGridChecksQaxPlugin.file_types,
                default_input_params=mgc_check_class.input_params,
                version=mgc_check_class.version,
                parameter_help_link=mgc_check_class.parameter_help_link,
            )
            check_refs.append(cr)

        return check_refs

    def checks(self) -> List[QaxCheckReference]:
        return self._check_references

    def __check_files_match(self, a: QajsonInputs, b: QajsonInputs) -> bool:
        """Checks if the input files in a are the same as b. This is used
        to match the plugin's output with the QAJSON outputs that must be
        updated with the check results.
        """
        set_a = set([str(p.path) for p in a.files])
        set_b = set([str(p.path) for p in b.files])
        return set_a == set_b

    def get_summary_details(self, qajson: QajsonRoot) -> List[Tuple[str, str]]:
        # it may be worth moving these header files to their own
        # dedicated qax plugin, that is always run irrespective
        # of what plugins are selected by the user.
        # Currently if the user doesn't run the MBES Grid Checks
        # plugin (this one), then these header fields won't be
        # available

        percentage_node_number = "5"

        # look through all the checks in the qajson to find the denisty check
        # and from this density check pull out the min soundings per node at percentage
        # parameter as this needs to be included in the summary label
        density_check = next(
            (
                c
                for c in qajson.qa.survey_products.checks
                if c.info.name == "Density Check"
            ),
            None,
        )
        if density_check:
            min_s_a_p = next(
                (
                    p
                    for p in (
                        density_check.inputs.params if density_check.inputs else []
                    )
                    if p.name == "Minimum Soundings per node at percentage"
                ),
                None,
            )
            if min_s_a_p:
                percentage_node_number = min_s_a_p.value

        return [
            ("header", "File Name"),
            ("header", "Latest Update"),
            ("header", "Summary"),
            ("header", "Number of Nodes"),
            ("DENSITY", "Number of Nodes with density fails"),
            (
                "DENSITY",
                r"% of nodes with "
                + str(percentage_node_number)
                + " soundings or greater",
            ),
            ("DENSITY", r"100% of nodes on SF"),
            ("DENSITY", "Density Check comment"),
            ("UNCERTAINTY", "Number of Nodes with Uncertainty Fails"),
            ("UNCERTAINTY", r"% of Nodes with  Uncertainty Fails"),
            ("UNCERTAINTY", "TVU Check comment"),
            ("RESOLUTION", "Resolution Check QAX Message"),
        ]

    def _revision_from_filename(self, filename: str) -> str:
        """Extracts  revision id from the filename. This is indicated by a token
        of the filename starting with `r`
        """
        potential_separators = ["-", "_", " "]
        name_only = filename
        separator = None
        separator_count = 0
        for sep in potential_separators:
            c = name_only.count(sep)
            if c > separator_count:
                separator = sep
                separator_count = c

        if separator_count == 0:
            return name_only

        name_tokens = name_only.split(separator)
        for t in name_tokens:
            if t.startswith("r") and len(t) > 1:
                return t
        return ""

    def get_summary_value(
        self,
        field_section: str,
        field_name: str,
        filename: str,
        qajson: QajsonRoot,
    ) -> object:
        """ """
        checks = self._get_qajson_checks(qajson)
        file_checks = self._checks_filtered_by_file(filename, checks)

        density_check = None
        density_checks = self._checks_filtered_by_name("Density Check", file_checks)
        # should really only be one
        if len(density_checks) >= 1:
            density_check = density_checks[0]

            # check if the density check failed, or was aborted. If so then
            # we can't rely on any of its outputs so this shouldn't be
            # included in the summary either
            if density_check.outputs.execution.status != "completed":
                density_check = None

        tvu_check = None
        tvu_checks = self._checks_filtered_by_name(
            "Total Vertical Uncertainty Check", file_checks
        )
        if len(tvu_checks) >= 1:
            tvu_check = tvu_checks[0]
            if tvu_check.outputs.execution.status != "completed":
                tvu_check = None

        res_check = None
        res_checks = self._checks_filtered_by_name("Resolution Check", file_checks)
        if len(res_checks) >= 1:
            res_check = res_checks[0]
            if res_check.outputs.execution.status != "completed":
                res_check = None

        if field_section == "header" and field_name == "File Name":
            return Path(filename).name
        elif field_section == "header" and field_name == "Latest Update":
            fn = Path(filename).name
            return self._revision_from_filename(fn)
        elif field_section == "header" and field_name == "Summary":
            return ""
        elif field_section == "header" and field_name == "Number of Nodes":
            if density_check:
                density_data = density_check.outputs.data
                node_count = 0
                for _, v in density_data["chart"]["data"].items():
                    node_count += v
                return node_count
            else:
                return "No density check"
        elif (
            field_section == "DENSITY"
            and field_name == "Number of Nodes with density fails"
        ):
            if density_check:
                density_data = density_check.outputs.data
                summary_data = density_data["summary"]
                return summary_data["under_threshold_soundings"]
            else:
                return "No density check"
        elif field_section == "DENSITY" and field_name.startswith(r"% of nodes with"):
            if density_check:
                density_data = density_check.outputs.data
                summary_data = density_data["summary"]
                return summary_data["percentage_over_threshold"]
            else:
                return "No density check"
        elif field_section == "DENSITY" and field_name == r"100% of nodes on SF":
            # TODO: need to understand this metric
            return ""
        elif field_section == "DENSITY" and field_name == "Density Check comment":
            # User entered field (entered into the XLSX), so just leave empty
            return ""
        elif (
            field_section == "UNCERTAINTY"
            and field_name == "Number of Nodes with Uncertainty Fails"
        ):
            if tvu_check:
                return tvu_check.outputs.data["failed_cell_count"]
            else:
                return "No TVU check"
        elif (
            field_section == "UNCERTAINTY"
            and field_name == r"% of Nodes with  Uncertainty Fails"
        ):
            if tvu_check:
                return tvu_check.outputs.data["fraction_failed"] * 100
            else:
                return "No TVU check"
        elif field_section == "UNCERTAINTY" and field_name == "TVU Check comment":
            return ""
        elif (
            field_section == "RESOLUTION"
            and field_name == "Resolution Check QAX Message"
        ):
            if res_check:
                return res_check.outputs.check_state
            else:
                return "No resolution check"

        else:
            return "No summary value"

    def run(
        self,
        qajson: QajsonRoot,
        progress_callback: Callable | None = None,
        qajson_update_callback: Callable | None = None,
        is_stopped: Callable | None = None,
    ) -> None:
        grid_data_checks = qajson.qa.survey_products.checks
        ifd_list = inputs_from_qajson_checks(grid_data_checks)

        self.exe = Executor(ifd_list, all_checks)

        # set options coming from QAX
        self.exe.spatial_qajson = self.spatial_outputs_qajson
        self.exe.spatial_export = self.spatial_outputs_export
        self.exe.spatial_export_location = self.spatial_outputs_export_location

        if (
            self.gridprocessing_tile_x is not None
            and self.gridprocessing_tile_y is not None
        ):
            # the executor defines a default tile size, don't overide this if the
            # gridprocessing_tile_x or gridprocessing_tile_y haven't been set
            self.exe.tile_size_x = self.gridprocessing_tile_x
            self.exe.tile_size_y = self.gridprocessing_tile_y

        # the check_runner callback accepts only a float, whereas the qax
        # qwax plugin check tool callback requires a referece to a check tool
        # AND a progress value. Hence this little mapping function,
        def pg_call(check_runner_progress):
            progress_callback(self, check_runner_progress)

        self.exe.run(pg_call, qajson_update_callback, is_stopped)

        for (ifd, check_id), check in self.exe.check_result_cache.items():
            for i, qajson_check in enumerate(ifd.qajson_checks):
                # the input file details includes a number of qajson check references
                # we need to make sure we only update the output qajson for the current
                # check.
                if qajson_check.info.id == check_id:
                    qajson_check.outputs = check.get_outputs()

        # MBESGC runs all checks over each tile of an input file, therefore
        # it's only possible to update the qajson once all checks have been
        # completed.
        if qajson_update_callback is not None:
            qajson_update_callback()

        # # the checks runner produces an array containing a listof checks
        # # each check being a dictionary. Deserialise these using the qa json
        # # datalevel class
        # out_dl = QajsonDataLevel.from_dict(
        #     {'checks': self.check_runner.output})
        #
        # # now loop through all raw_data (Mate only does raw data) checks in
        # # the qsjson and update the right checks with the check runner output
        # for out_check in out_dl.checks:
        #     # find the check definition in the input qajson.
        #     # note: both check and id must match. The same check implmenetation
        #     # may be include twice but with diffferent names (this is
        #     # supported)
        #     in_check = next(
        #         (
        #             c
        #             for c in qajson.qa.raw_data.checks
        #             if (
        #                 c.info.id == out_check.info.id and
        #                 c.info.name == out_check.info.name and
        #                 self.__check_files_match(c.inputs, out_check.inputs))
        #         ),
        #         None
        #     )
        #     if in_check is None:
        #         # this would indicate a check was run that was not included
        #         # in the input qajson. *Should never occur*
        #         raise RuntimeError(
        #             "Check {} ({}) found in output that was not "
        #             "present in input"
        #             .format(out_check.info.name, out_check.info.id))
        #     # replace the input qajson check outputs with the output generated
        #     # by the check_runner
        #     in_check.outputs = out_check.outputs

    def get_file_details(self, filename: str) -> str:
        """Return some details about the raster file that's been provided. In this
        case a list of the bands, and the resolution of the dataset.
        """
        return get_file_details(filename)
