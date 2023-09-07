# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo %s 2>&1 | FileCheck %s


# CHECK: could not find a 'main' function to execute
@export
fn foo() -> Float32:
    return 0.0
