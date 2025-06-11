# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys
import unittest

import max.mojo.importer  # noqa

# Put the current directory (containing mojo_module.mojo) on the Python module
# lookup path.
sys.path.insert(0, "")


# Imports from 'mojo_module.mojo'
import mojo_module  # type: ignore


class TestMojoPythonInterop(unittest.TestCase):
    def test_pyinit(self):
        self.assertTrue(mojo_module)

    def test_plus_one(self):
        self.assertEqual(mojo_module.plus_one(5), 6)


if __name__ == "__main__":
    unittest.main()
