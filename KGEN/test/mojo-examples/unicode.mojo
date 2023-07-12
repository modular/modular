# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s | FileCheck %s
# RUN: kgen %s -emit -debug-level=full --O0 -o /dev/null

from IO import print


fn main():
    # CHECK: Hello, 🔥!
    print("Hello, 🔥!")
