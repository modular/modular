# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(MSTDL-894): Support running this test on Linux
# REQUIRES: system-darwin
# RUN: python3 -m mojo-pybind.main %S/bindings.mojo
# RUN: python3 %s

import sys
import os
from pathlib import Path

# Put the current directory (containing bindings.so) on the Python module lookup
# path.
sys.path.insert(0, "")

# Force the Mojo standard library to load the libpython for the current Python
# process.
os.environ["MOJO_PYTHON_LIBRARY"] = sys.executable

# Imports from 'bindings.so'
import bindings

if __name__ == "__main__":
    pass

    result = bindings.mojo_count_args(1, 2)

    assert result == 2

    print("Result from Mojo 🔥:", result)
