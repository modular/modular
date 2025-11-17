# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -I=%S %s | FileCheck %s

from test_package.module import *


# CHECK-LABEL: lit.fn @"foo
fn foo():
    var x = Wrapper(33)
    var y = x.data


# Even though ParameterizedType is referenced in an alias in Wrapper, the alias
# itself is unused by this file. ParameterizedType should have been removed as
# an unreachable decl.
# CHECK-NOT: lit.struct.decl @ParameterizedType
