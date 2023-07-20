# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: MODULAR_DERIVED_PATH="foo" not %mojo %s 2>&1 | FileCheck %s
# CHECK: unable to locate module 'Builtin'


fn main() -> None:
    pass
