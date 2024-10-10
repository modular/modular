# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo %s 2>&1 | FileCheck %s


# CHECK: module does not define a `main` function
@export
fn foo() -> Float32:
    return 0.0
