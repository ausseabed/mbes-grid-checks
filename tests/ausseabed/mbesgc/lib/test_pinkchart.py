from pathlib import Path
import unittest


from ausseabed.mbesgc.lib.pinkchart import PinkChartProcessor, Extents

class TestPinkChart(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raster_file = Path('./tests/test_data/pink_chart/portfairy.tiff')
        cls.raster_file_out = Path('./tests/test_data/pink_chart/portfairy_output.tiff')
        cls.pinkchart_file = Path('./tests/test_data/pink_chart/pinkchart.shp')
        cls.rasterised_file = Path('./tests/test_data/pink_chart/pinkchart_outside_raster.tif')

    def test_files_exist(self):
        # print("test data files in " + str(self.raster_file.absolute().parent))
        self.assertTrue(self.raster_file.exists())
        self.assertTrue(self.raster_file.is_file())

        self.assertTrue(self.pinkchart_file.exists())
        self.assertTrue(self.pinkchart_file.is_file())

    def test_ideal_value(self):
        pc = PinkChartProcessor(None, None, None, None)
        self.assertAlmostEqual(
            pc._calc_ideal_value(0.5, -4, -6.3, True),
            -6.5
        )

        self.assertAlmostEqual(
            pc._calc_ideal_value(0.5, -6.3, -4, True),
            -4.3
        )

        self.assertAlmostEqual(
            pc._calc_ideal_value(0.5, 1, -0.1, True),
            -0.5
        )

        self.assertAlmostEqual(
            pc._calc_ideal_value(0.5, 1, 2.1, False),
            2.5
        )

        self.assertAlmostEqual(
            pc._calc_ideal_value(0.5, 5, 4.1, False),
            4.5
        )

        self.assertAlmostEqual(
            pc._calc_ideal_value(0.5, 1, 2.2, True),
            2.0
        )

    def test_calc_ideal_extents(self):
        pc = PinkChartProcessor(None, None, None, None)

        source_res_x = 0.5
        source_res_y = 0.5
        source_extents = Extents(-4, 1, 1, 5)
        target_extents = Extents(-6.3, -0.1, 2.1, 4.1)

        ideal_extents = pc._calc_ideal_extents(source_res_x, source_res_y, source_extents, target_extents)
        # print()
        # print("source = " + str(source_extents))
        # print("target = " + str(target_extents))
        # print("ideal  = " + str(ideal_extents))

        expected_extents = Extents(-6.5, -0.5, 2.5, 4.5)
        self.assertEqual(ideal_extents, expected_extents)

    def test_generate_pinkchart_raster(self):
        pc = PinkChartProcessor(
            [self.raster_file],
            self.pinkchart_file,
            [self.raster_file_out],
            self.rasterised_file
        )

        pc.process()
    

    def test_generate_pinkchart_raster_different_projections(self):
        pc = PinkChartProcessor(
            [Path('./tests/test_data/ds02/in2018_c01_cs_CUBE_2m_Rev2.tif')],
            Path('./tests/test_data/ds02/in2018_c01_cs_CUBE_2m_Rev2.shp'),
            [Path('./tests/test_data/ds02_out/in2018_c01_cs_CUBE_2m_Rev2.tif')],
            Path('./tests/test_data/ds02_out/in2018_c01_cs_CUBE_2m_Rev2_pc.tif')
        )

        pc.process()
