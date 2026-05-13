# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo precompile -kgenModule -disable-builtins -I %P/test_dependency %P/target -o %t.target.mlirbc
# RUN: kgen-opt %t.target.mlirbc | FileCheck %s

from test_dependency import *


@export
def anchor() -> __mlir_type.index:
    return use_me()
