import unittest

from ausseabed.mbesgc.qax.plugin import MbesGridChecksQaxPlugin

class TestGrouping(unittest.TestCase):

    def test_extract_revision(self):
        fn = "foo_bar_r123_xyz"

        plugin = MbesGridChecksQaxPlugin()
        revision = plugin._revision_from_filename(fn)

        self.assertEqual(revision, 'r123')
