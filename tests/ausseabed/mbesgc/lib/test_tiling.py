import unittest

from ausseabed.mbesgc.lib.tiling import get_tiles


class TestTiling(unittest.TestCase):
    def test_get_tiles(self):
        min_x = 0
        min_y = 0
        max_x = 14
        max_y = 10
        size_x = 5
        size_y = 3

        tiles = get_tiles(min_x, min_y, max_x, max_y, size_x, size_y)

        self.assertEqual(tiles[0].min_x, min_x)
        self.assertEqual(tiles[0].min_y, min_y)

        self.assertEqual(tiles[-1].max_x, max_x)
        self.assertEqual(tiles[-1].max_y, max_y)

        self.assertEqual(len(tiles), 3 * 4)
