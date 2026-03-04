# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -DVAR=1 %s | FileCheck %s
# RUN: %mojo -D VAR=1 %s | FileCheck %s

from std.sys import get_defined_bool


def main() raises:
    # CHECK: True
    print(get_defined_bool["VAR", False]())
