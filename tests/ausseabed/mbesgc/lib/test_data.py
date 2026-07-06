import unittest
import json

from ausseabed.qajson.model import QajsonCheck

from ausseabed.mbesgc.lib.data import (
    inputs_from_qajson_checks,
    InputFileDetails,
    BandType,
)

check01_str = """
{
    "info": {
        "id": "5e2afd8a-2ced-4de8-80f5-111c459a7175",
        "name": "Density Check",
        "version": "1",
        "group": {
            "id": "",
            "name": "",
            "description": ""
        }
    },
    "inputs": {
        "files": [
            {
                "path": "/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/tests/test_data/ds02/in2018_c01_cs_CUBE_2m_Rev2.shp",
                "file_type": "Coverage Area"
            },
            {
                "path": "/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/tests/test_data/ds02/in2018_c01_cs_CUBE_2m_Rev2.tif",
                "file_type": "Survey DTMs"
            }
        ],
        "params": [
            {
                "name": "Minimum Soundings per node",
                "value": 5
            },
            {
                "name": "Minimum Soundings per node at percentage",
                "value": 9
            },
            {
                "name": "Minimum Soundings per node percentage",
                "value": 95.0
            }
        ]
    }
}
"""

check02_str = """
{
    "info": {
        "id": "5e2afd8a-2ced-4de8-80f5-111c459a7175",
        "name": "Density Check",
        "version": "1",
        "group": {
            "id": "",
            "name": "",
            "description": ""
        }
    },
    "inputs": {
        "files": [
            {
                "path": "/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/tests/test_data/ds01/in2018_c01clip_CombinedSurface_CUBE_2m_Rev2.shp",
                "file_type": "Coverage Area"
            },
            {
                "path": "/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/tests/test_data/ds01/in2018_c01clip_CombinedSurface_CUBE_2m_Rev2_uncertainty.tif",
                "file_type": "Survey DTMs"
            },
            {
                "path": "/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/tests/test_data/ds01/in2018_c01clip_CombinedSurface_CUBE_2m_Rev2_density.tif",
                "file_type": "Survey DTMs"
            },
            {
                "path": "/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/tests/test_data/ds01/in2018_c01clip_CombinedSurface_CUBE_2m_Rev2_depth.tif",
                "file_type": "Survey DTMs"
            }
        ],
        "params": [
            {
                "name": "Minimum Soundings per node",
                "value": 5
            },
            {
                "name": "Minimum Soundings per node at percentage",
                "value": 9
            },
            {
                "name": "Minimum Soundings per node percentage",
                "value": 95.0
            }
        ]
    }
}
"""


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check_01 = QajsonCheck.from_dict(json.loads(check01_str))
        cls.check_02 = QajsonCheck.from_dict(json.loads(check02_str))

    def test_x(self):
        inputs = inputs_from_qajson_checks([self.check_01])
        print("inputs for check")
        for input in inputs:
            print(input)

        inputs = inputs_from_qajson_checks([self.check_02])
        print("inputs for check")
        for input in inputs:
            print(input)

    def test_has_same_inputs(self):
        a = InputFileDetails()
        a.add_band_details("f1", 1, BandType.depth)
        a.add_band_details("f1", 2, BandType.density)
        a.add_band_details("f1", 3, BandType.uncertainty)

        b = InputFileDetails()
        b.add_band_details("f1", 2, BandType.density)
        b.add_band_details("f1", 1, BandType.depth)
        b.add_band_details("f1", 3, BandType.uncertainty)

        self.assertTrue(a.has_same_inputs(b))

        c = InputFileDetails()
        c.add_band_details("f2", 1, BandType.depth)
        c.add_band_details("f2", 2, BandType.density)
        c.add_band_details("f2", 3, BandType.uncertainty)

        self.assertFalse(a.has_same_inputs(c))

    def test_get_common_filename(self):
        inputs = inputs_from_qajson_checks([self.check_01])
        self.assertEqual("in2018_c01_cs_CUBE_2m_Rev2", inputs[0].get_common_filename())

        inputs = inputs_from_qajson_checks([self.check_02])
        self.assertEqual(
            "in2018_c01clip_CombinedSurface_CUBE_2m_Rev2_",
            inputs[0].get_common_filename(),
        )

    def test_validate_duplicate_input_bands(self):
        a = InputFileDetails()
        a.add_band_details("f1", 1, BandType.depth)
        a.add_band_details("f1", 2, BandType.depth)
        a.add_band_details("f1", 3, BandType.uncertainty)

        passed, _ = a.validate()
        self.assertFalse(passed)

    def test_validate_more_than_3_inputs(self):
        a = InputFileDetails()
        a.add_band_details("f1", 1, BandType.depth)
        a.add_band_details("f1", 2, BandType.density)
        a.add_band_details("f1", 3, BandType.uncertainty)
        a.add_band_details("f1", 3, "blah")

        passed, _ = a.validate()
        self.assertFalse(passed)

    def test_validate_ok(self):
        a = InputFileDetails()
        a.add_band_details("f1", 1, BandType.depth)
        a.add_band_details("f1", 2, BandType.density)
        a.add_band_details("f1", 3, BandType.uncertainty)

        passed, messages = a.validate()
        self.assertTrue(passed)
        self.assertEqual(len(messages), 0)
