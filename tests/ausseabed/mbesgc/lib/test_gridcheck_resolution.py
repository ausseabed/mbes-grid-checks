import numpy as np
import unittest

from ausseabed.qajson.model import QajsonParam

from ausseabed.mbesgc.lib.mbesgridcheck import ResolutionCheck
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
        cls.dummy_ifd.geotransform = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        cls.dummy_ifd.projection = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')  # noqa

        cls.dummy_tile = Tile(0, 0, 5, 5)

        # set up some dummy data
        mask = [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, True],
            [False, False, True, True]
        ]
        depth_data = [
            [-40, -40, -40, -40],
            [-40, -60, -80, -40],
            [-40, -60, -70, -40],
            [-40, -30, -70, -40],
            [-40, -40, -40, -40]
        ]
        cls.depth = np.ma.array(
            np.array(depth_data, dtype=np.float32),
            mask=mask
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
            mask=mask
        )
        uncertainty_data = [
            [0.7, 0.7, 0.2, 0.2],
            [0.7, 0.4, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.9],
            [0.2, 0.2, 0.9, 0.0],
            [0.2, 0.2, 0.2, 0.0]
        ]
        cls.uncertainty = np.ma.array(
            np.array(uncertainty_data, dtype=np.float32),
            mask=mask
        )

    def test_resolution(self):
        # default values taken from IHO - 1a spec
        input_params = [
            QajsonParam("Feature Detection Size Multiplier", 1.5),
            QajsonParam("Threshold Depth", 40.0),
            QajsonParam("Above Threshold FDS Depth Multiplier", 0.0),
            QajsonParam("Above Threshold FDS Depth Constant", 2),
            QajsonParam("Below Threshold FDS Depth Multiplier", 0.025),
            QajsonParam("Below Threshold FDS Depth Constant", 0)
        ]

        check = ResolutionCheck(input_params)
        check.run(
            ifd=self.dummy_ifd,
            tile=self.dummy_tile,
            depth=self.depth,
            density=self.density,
            uncertainty=self.uncertainty,
            pinkchart=None
        )

        # 17 because three of the cells are masked
        self.assertEqual(check.total_cell_count, 17)

        # note: grid resolution of the dummy data is 2, this is taken from
        # the geo transform assigned to the dummy inputfiledetails object

        # calculated feature detection size using input params and dummy depth
        # data.
        # [1.   1.   1.   1.  ]
        # [1.   1.5  2.   1.  ]
        # [1.   1.5  1.75 1.  ]
        # [1.   2.   1.75 1.  ]
        # [1.   1.   1.   1.  ]

        # We could use the feature detection size multiplier to scale the
        # 'calculated feature detection size' but instead the check divides
        # the grid resolution by this value. So for the purposes of this test
        # we use a "Feature Detection Size Multiplier" value of 1.5 to get
        # some nodes to fail; in this case it means any nodes with a calculated
        # feature detection size below 1.333 will fail this check.

        # and these values exceed threshold in 11 locations. Note that it's not
        # 14 locations due to the masking of 3 cells (lower right corner)
        self.assertEqual(check.failed_cell_count, 11)
