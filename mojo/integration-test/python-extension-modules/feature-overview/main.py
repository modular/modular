# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(MSTDL-894): Support running this test on Linux
# REQUIRES: system-darwin
# RUN: python3 -m mojo-pybind.main %S/feature_overview.mojo
# RUN: python3 %s

import sys
import os
import unittest

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

# Force the Mojo standard library to load the libpython for the current Python
# process.
os.environ["MOJO_PYTHON_LIBRARY"] = sys.executable

# Imports from 'feature_overview.so'
import feature_overview


class TestMojoPythonInterop(unittest.TestCase):
    def test_case_return_arg_tuple(self):
        result = feature_overview.case_return_arg_tuple(
            1, 2, "three", ["four", "four.B"]
        )

        self.assertEqual(result, (1, 2, "three", ["four", "four.B"]))

    def test_case_raise_empty_error(self):
        with self.assertRaises(ValueError) as cm:
            feature_overview.case_raise_empty_error()

        self.assertEqual(cm.exception.args, ())

    def test_case_raise_string_error(self):
        with self.assertRaises(ValueError) as cm:
            feature_overview.case_raise_string_error()

        self.assertEqual(cm.exception.args, ("sample value error",))

    def test_case_mojo_raise(self):
        with self.assertRaises(Exception) as cm:
            feature_overview.case_mojo_raise()

        self.assertEqual(cm.exception.args, ("Mojo error",))

    def test_case_create_mojo_type_instance(self):
        person = feature_overview.Person()

        self.assertEqual(type(person).__name__, "Person")

        self.assertEqual(person.name(), "John Smith")


if __name__ == "__main__":
    unittest.main()
