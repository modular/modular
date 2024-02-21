# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics -I=%S %s | FileCheck %s

from test_package.module import *


# CHECK-LABEL: lit.func @"foo
fn foo():
    let x = Wrapper(__mlir_attr.`33 : index`)
    let y = x.data


# Even though ParameterizedType is referenced in an alias in Wrapper, the alias
# itself is unused by this file. ParameterizedType should have been removed as
# an unreachable decl.
# CHECK-NOT: lit.struct.decl @ParameterizedType
