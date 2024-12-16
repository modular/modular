# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import sys
import unittest

import max._mojo.mojo_importer  # noqa

# Put the current directory (containing mojo_module.mojo) on the Python module
# lookup path.
sys.path.insert(0, "")

# TODO(MOCO-1375): Remove env var restriction on --gen-py use
os.environ["MODULAR_MOJO_PYBIND"] = "enabled"

# Imports from 'mojo_module.mojo'
import mojo_module  # type: ignore


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


if __name__ == "__main__":
    unittest.main()
