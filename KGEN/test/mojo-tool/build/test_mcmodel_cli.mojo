# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: x86_64-linux
# COM: This check only makes sense for generating an ELF object file.
# RUN: %mojo-build %s --mcmodel=medium --large-data-threshold=2 -o %t
# RUN: llvm-objdump %t -t | FileCheck %s

# COM: check that string constant is in .lrodata section
# (for any data size that's larger than large-data-threshold)
# CHECK: .lrodata
fn main():
    print("hello world.")
