# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: system-darwin
# RUN: %mojo-build %s -o %t
# RUN: %t | FileCheck %s

from std.python import Python
from std.sys import argv


def main() raises:
    var python = Python()

    # CHECK: This was built inside of python
    var py_string = Python.evaluate("'This was built' + ' inside of python'")
    print(python.as_string_slice(py_string.__str__()))
