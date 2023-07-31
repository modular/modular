# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: system-darwin
# RUN: mojo build %mojo_cpu_build_arch %s -o %t
# RUN: %t --arg1 | FileCheck %s

from IO import print
from Sys import argv


fn main():
    # CHECK: This was called inside of `fn` main
    print("This was called inside of `fn` main")

    # CHECK: --arg1
    print(argv()[1])
