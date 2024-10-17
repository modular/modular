# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: python3 -m mojo-pybind.main --raw-bindings %S/mojo_module.mojo
# RUN: python3 %s

import sys

# Put the current directory (containing module.so) on the Python module lookup
# path.
sys.path.insert(0, "")

# Imports from 'mojo_module.so'
import mojo_module

if __name__ == "__main__":
    pass

    result = mojo_module.mojo_count_args(1, 2)

    assert result == 2

    print("Result from Mojo 🔥:", result)
