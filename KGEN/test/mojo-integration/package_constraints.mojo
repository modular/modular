# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Create the inner package with ABSOLUTE file paths.
# Create the outer package with RELATIVE file paths.
# When the main executable imports both packages, there will be a conflict:
# The `constrained_method` function will come from the inner package, and use
# absolute paths. But the usage of `constrained_method` in the outer package
# will use relative paths, and result in a mismatch in def metadata.

# RUN: mojo precompile %S/inputs/where_package -o %S/where_package.mojoc
# RUN: mojo precompile %S/inputs/wrapper_where_package -o %S/wrapper_where_package.mojoc -strip-file-prefix=%S/inputs
# RUN: mojo %s -I %S | FileCheck %s

from where_package import constrained_method
from wrapper_where_package import use_constrained_method


def main():
    # CHECK: result: 42 2
    comptime result = constrained_method[42]()
    comptime result2 = use_constrained_method()
    print("result:", result, result2)
