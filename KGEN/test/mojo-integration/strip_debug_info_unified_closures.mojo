# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mkdir -p %t.closure-dir
# RUN: mojo package %S/inputs/closure -o %t.closure-dir/closure.mojopkg
# RUN: mojo -debug-level=line-tables -I %t.closure-dir %s 4 | FileCheck %s

from sys import argv
from closure import emitLoad


def main():
    var x = SIMD[DType.int, 1](atol(argv()[1]))
    # CHECK: 4
    emitLoad(x)
