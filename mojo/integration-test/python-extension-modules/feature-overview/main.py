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

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

# Force the Mojo standard library to load the libpython for the current Python
# process.
os.environ["MOJO_PYTHON_LIBRARY"] = sys.executable

# Imports from 'feature_overview.so'
import feature_overview


def test_case_return_arg_tuple():
    result = feature_overview.case_return_arg_tuple(
        1, 2, "three", ["four", "four.B"]
    )

    assert result == (1, 2, "three", ["four", "four.B"])


def test_case_raise_empty_error():
    try:
        # This is expected to raise
        feature_overview.case_raise_empty_error()

        print("expected exception to be raised")
        exit(-1)
    except ValueError as e:
        # Do nothing, we caught the error we expected.
        pass


def test_case_raise_string_error():
    try:
        # This is expected to raise
        feature_overview.case_raise_string_error()

        print("expected exception to be raised")
        exit(-1)
    except ValueError as e:
        # Assert that we caught the error we expected.
        assert str(e) == "sample value error"


def test_case_mojo_raise():
    try:
        # This is expected to raise
        feature_overview.case_mojo_raise()

        print("expected exception to be raised")
        exit(-1)
    except Exception as e:
        # Assert that we caught the error we expected.
        assert str(e) == "Mojo error"


if __name__ == "__main__":
    test_case_return_arg_tuple()
    test_case_raise_empty_error()
    test_case_raise_string_error()
    test_case_mojo_raise()
