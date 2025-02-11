# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo package -kgenModule -disable-builtins -I %P/test_dependency %P/target -o %T/target.mlirbc
# RUN: kgen-opt %T/target.mlirbc | FileCheck %s

from test_dependency import *

@export
fn anchor() -> int:
    return use_me()
