# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: system-darwin
# XFAIL: asan && !system-darwin
# RUN: mojo build %mojo_cpu_build_arch %s -o %t
# RUN: %t | FileCheck %s

from IO import print
from PythonInterface import Python
from Sys import argv


def main() -> None:
    var python = Python()

    # CHECK: This was built inside of python
    let py_string = Python.evaluate("'This was built' + ' inside of python'")
    print(python.__str__(py_string.__str__()))
