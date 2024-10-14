# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(MSTDL-894): Support running this test on Linux
# REQUIRES: system-darwin

# RUN: python3 -m mojo-pybind.main %S/mojo_module.mojo
# RUN: python3 %s

import sys
import os
import unittest

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

# Force the Mojo standard library to load the libpython for the current Python
# process.
os.environ["MOJO_PYTHON_LIBRARY"] = sys.executable

# Imports from 'mojo_module.so'
import mojo_module


class TestMojoPythonInterop(unittest.TestCase):
    def test_pyinit(self):
        self.assertTrue(mojo_module)

    def test_pytype_reg_trivial(self):
        self.assertEqual(mojo_module.Int.__name__, "Int")

    def test_pytype_empty_init(self):
        # Tests that calling the default constructor on a wrapped Mojo type
        # is possible.
        mojo_int = mojo_module.Int()

        self.assertEqual(type(mojo_int), mojo_module.Int)


if __name__ == "__main__":
    unittest.main()
