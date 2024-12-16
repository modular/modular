# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(MSTDL-894): Support running this test on Linux
# REQUIRES: system-darwin

# RUN: kgen-translate -import-mojo %S/module.mojo -gen-pybind --mojo-enable-prebuilt-packages -o %T/module.mlir
# RUN: python3 -m mojo-pybind.main %T/module.mlir
# RUN: python3 %s

import os
import sys
import unittest

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

# Force the Mojo standard library to load the libpython for the current Python
# process.
os.environ["MOJO_PYTHON_LIBRARY"] = sys.executable

# Imports from 'feature_overview.so'
import module


class TestMojoPythonInterop(unittest.TestCase):
    def test_pyinit(self):
        self.assertTrue(module)

    def test_pytype_reg_trivial(self):
        self.assertEqual(module.Int.__name__, "Int")


if __name__ == "__main__":
    unittest.main()
