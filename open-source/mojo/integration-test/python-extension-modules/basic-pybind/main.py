# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: MODULAR_MOJO_PYBIND=enabled %bare-mojo build %S/mojo_module.mojo --gen-py
# RUN: python3 %s

import sys
import unittest

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

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

        self.assertEqual(repr(mojo_int), "0")

        mojo_module.incr_int(mojo_int)

        self.assertEqual(repr(mojo_int), "1")


if __name__ == "__main__":
    unittest.main()
