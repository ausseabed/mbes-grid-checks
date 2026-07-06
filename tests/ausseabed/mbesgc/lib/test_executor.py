import unittest
import json

from ausseabed.mbesgc.lib.allchecks import all_checks
from ausseabed.qajson.model import QajsonCheck

from ausseabed.mbesgc.lib.data import inputs_from_qajson_checks
from ausseabed.mbesgc.lib.executor import Executor

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
                "path": "/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/tests/test_data/ds02/in2018_c01_CombinedSurface_CUBE_2m_Rev2.shp",
                "file_type": "Coverage Area"
            },
            {
                "path": "/Users/lachlan/work/projects/qa4mb/repo/mbes-grid-checks/tests/test_data/ds02/in2018_c01_CombinedSurface_CUBE_2m_Rev2.tif",
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


class TestExecutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check_01 = QajsonCheck.from_dict(json.loads(check01_str))
        cls.check_02 = QajsonCheck.from_dict(json.loads(check02_str))

    def test_x(self):
        # inputs = inputs_from_qajson_checks([self.check_01])
        # print("inputs for check")
        # for input in inputs:
        #     print(input)

        inputs = inputs_from_qajson_checks([self.check_02])
        # print("inputs for check")
        # for input in inputs:
        #     print(input)

        exe = Executor(inputs, all_checks)
        exe._preprocess()
