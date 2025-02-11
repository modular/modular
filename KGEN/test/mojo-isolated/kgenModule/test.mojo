# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo package -kgenModule -disable-builtins -I %S/test_dependency %S/target -o %T/target.mlirbc
# RUN: kgen-opt %T/target.mlirbc | FileCheck %s

# CHECK: kgen.generator export @anchor() -> index
# CHECK: kgen.generator @"test_dependency::impl::use_me()"() -> index
# CHECK: kgen.generator @"test_dependency::impl::child()"() -> index
# CHECK-NOT: dead

fn main():
    pass
