# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys
import unittest

import max.mojo.importer  # noqa
import numpy as np

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

    def test_parallel_wrapper(self):
        arr = np.random.uniform(size=(10000)).astype(np.float32)
        arr = mojo_module.parallel_wrapper(arr)
        assert np.allclose(arr[0], np.max(arr))


if __name__ == "__main__":
    unittest.main()
