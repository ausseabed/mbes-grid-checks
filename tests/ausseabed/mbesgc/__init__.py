import unittest

from ausseabed.mbesgc.lib.tiling import get_tiles


class TestTiling(unittest.TestCase):

    def test_get_tiles(self):
        min_x = 0
        min_y = 0
        max_x = 14
        max_y = 6
        size_x = 4
        size_y = 4

        tiles = get_tiles(min_x, min_y, max_x, max_y, size_x, size_y)

    def test_always_fail(self):
        self.assertEqual(1, 2)
