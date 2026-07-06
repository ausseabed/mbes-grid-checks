import numpy as np
import unittest

from ausseabed.qajson.model import QajsonParam

from ausseabed.mbesgc.lib.gridcheck import GridCheckState
from ausseabed.mbesgc.lib.mbesgridcheck import DensityCheck
from ausseabed.mbesgc.lib.data import InputFileDetails
from ausseabed.mbesgc.lib.tiling import Tile


class TestDensityCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # these objects are often needed by checks, but only for the generation of
        # spatial outputs. These dummy objects can be used where these spatial
        # outputs are not being tested.
        cls.dummy_ifd = InputFileDetails()
        cls.dummy_ifd.size_x = 5
        cls.dummy_ifd.size_y = 5
        cls.dummy_ifd.geotransform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cls.dummy_ifd.projection = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'  # noqa

        cls.dummy_tile = Tile(0, 0, 5, 5)

        # set up some dummy data
        mask = [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, True],
            [False, False, True, True],
        ]
        depth_data = [
            [-40, -40, -40, -40],
            [-40, -60, -80, -40],
            [-40, -60, -70, -40],
            [-40, -30, -70, -40],
            [-40, -40, -40, -40],
        ]
        cls.depth = np.ma.array(
            np.array(depth_data, dtype=np.float32),
            mask=mask,
        )
        density_data = [
            [10, 1, 9, 9],
            [10, 2, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10],
        ]
        cls.density = np.ma.array(
            np.array(density_data, dtype=np.float32),
            mask=mask,
        )
        uncertainty_data = [
            [0.7, 0.7, 0.2, 0.2],
            [0.7, 0.4, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.9],
            [0.2, 0.2, 0.9, 0.0],
            [0.2, 0.2, 0.2, 0.0],
        ]
        cls.uncertainty = np.ma.array(
            np.array(uncertainty_data, dtype=np.float32), mask=mask
        )

    def test_result_hist_chunk_merge(self):
        res_a = {
            0: 3,
            1: 5,
            2: 7,
            5: 8,
            10: 1,
        }
        res_b = {
            0: 1,
            2: 3,
            4: 2,
            5: 3,
            9: 1,
        }

        c_a = DensityCheck([])
        c_a.density_histogram = res_a

        c_b = DensityCheck([])
        c_b.density_histogram = res_b

        c_a.merge_results(c_b)

        self.assertEqual(c_a.density_histogram[0], 4)
        self.assertEqual(c_a.density_histogram[1], 5)
        self.assertEqual(c_a.density_histogram[2], 10)
        self.assertEqual(c_a.density_histogram[4], 2)
        self.assertEqual(c_a.density_histogram[5], 11)
        self.assertEqual(c_a.density_histogram[9], 1)
        self.assertEqual(c_a.density_histogram[10], 1)

    def test_density_threshold(self):
        input_params = [
            QajsonParam("Minimum Soundings per node", 5),
            QajsonParam("Minimum Soundings per node percentage", 95),
        ]

        check = DensityCheck(input_params)
        check.run(
            ifd=self.dummy_ifd,
            tile=self.dummy_tile,
            depth=self.depth,
            density=self.density,
            uncertainty=self.uncertainty,
            pinkchart=None,
        )

        density_histogram = check.density_histogram

        # check the counts in the density histogram match that of the density
        # array above excluding the the values that are masked out with the
        # mask array
        self.assertEqual(density_histogram[1], 1)
        self.assertEqual(density_histogram[2], 1)
        self.assertEqual(density_histogram[9], 2)
        self.assertEqual(density_histogram[10], 13)

        # now check the output data

        output = check.get_outputs()
        # check should fail as there are two values below the min soundings
        # count of 5
        self.assertEqual(output.check_state, GridCheckState.cs_fail)

    def test_density_threshold_at_percentage(self):
        input_params = [
            QajsonParam("Minimum Soundings per node", 5),
            QajsonParam("Minimum Soundings per node percentage", 89),
        ]

        check = DensityCheck(input_params)
        check.run(
            ifd=self.dummy_ifd,
            tile=self.dummy_tile,
            depth=self.depth,
            density=self.density,
            uncertainty=self.uncertainty,
            pinkchart=None,
        )

        # now check the output data
        output = check.get_outputs()
        # two out of 17 nodes are below the min soundings per node at
        # percentage of 5. This works out to 82%, and the input param specifies
        # 95% so this should fail
        # "Minimum Soundings per node" is set to 0 so this wont be tripped.
        self.assertEqual(output.check_state, GridCheckState.cs_fail)
