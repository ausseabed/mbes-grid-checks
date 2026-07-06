from affine import Affine
from geojson import Polygon
from osgeo import osr


class Tile:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    @property
    def width(self):
        return self.max_x - self.min_x

    @property
    def height(self):
        return self.max_y - self.min_y

    def to_geojson(self, projection, geotransform):
        fwd = Affine.from_gdal(*geotransform)

        min_x_min_y = fwd * (self.min_x, self.min_y)
        max_x_min_y = fwd * (self.max_x, self.min_y)
        max_x_max_y = fwd * (self.max_x, self.max_y)
        min_x_max_y = fwd * (self.min_x, self.max_y)

        coordinates = [
            min_x_min_y,
            max_x_min_y,
            max_x_max_y,
            min_x_max_y,
            min_x_min_y,
        ]

        src_proj = osr.SpatialReference()
        src_proj.ImportFromWkt(projection)
        dst_proj = osr.SpatialReference()
        dst_proj.ImportFromEPSG(4326)
        dst_proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        coord_trans = osr.CoordinateTransformation(src_proj, dst_proj)

        coordinates = [
            coord_trans.TransformPoint(coord[0], coord[1], 0.0)[:2]
            for coord in coordinates
        ]

        return Polygon(coordinates=coordinates)

    def __repr__(self):
        return f"({self.min_x}, {self.min_y}) ({self.max_x}, {self.max_y})"


def get_tiles(min_x, min_y, max_x, max_y, size_x, size_y):
    """
    Breaks the given extents down into a number of tiles based on a tile size.
    Edge tiles will have a smaller dimension if the extents are not divisible
    by the size.
    """
    assert min_x < max_x
    assert min_y < max_y
    assert size_x != 0
    assert size_y != 0

    dX = int(max_x) - int(min_x)
    dY = int(max_y) - int(min_y)
    whole_steps_x = dX / int(size_x)
    whole_steps_y = dY / int(size_y)
    rX = dX % whole_steps_x
    rY = dY % whole_steps_y

    tiles = []
    for y in range(int(min_y), int(max_y), int(size_y)):
        next_y = y + int(size_y)
        next_y = max_y if next_y > max_y else next_y
        for x in range(int(min_x), int(max_x), int(size_x)):
            next_x = x + int(size_x)
            next_x = max_x if next_x > max_x else next_x
            tile = Tile(x, y, next_x, next_y)
            tiles.append(tile)

    return tiles
