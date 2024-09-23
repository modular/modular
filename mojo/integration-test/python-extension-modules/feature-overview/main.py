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


if __name__ == "__main__":
    test_case_return_arg_tuple()
