# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics -split-input-file --mojo-disable-parser-caching=true -I=%S %s | FileCheck %s

from test_package.module import *


# CHECK-LABEL: lit.func @"foo
fn foo():
    let x = Wrapper(33)
    print(x.data)


# Even though ParameterizedType is referenced in an alias in Wrapper, the alias
# itself is unused by this file. ParameterizedType should have been removed as
# an unreachable decl.
# CHECK-NOT: lit.struct.decl @ParameterizedType
