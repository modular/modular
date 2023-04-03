# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s | FileCheck %s

from IO import print


fn main():
    # CHECK: Hello, 🔥!
    print("Hello, 🔥!")
