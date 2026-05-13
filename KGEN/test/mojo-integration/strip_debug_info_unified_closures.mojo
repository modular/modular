# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mkdir -p %t.closure-dir
# RUN: mojo package %S/inputs/closure -o %t.closure-dir/closure.mojoc
# RUN: mojo -debug-level=line-tables -I %t.closure-dir %s 4 | FileCheck %s

from std.sys import argv
from closure import emitLoad


def main() raises:
    var x = SIMD[DType.int, 1](atol(argv()[1]))
    # CHECK: 4
    emitLoad(x)
