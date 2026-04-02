# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo package -kgenModule -disable-builtins -I %S/inputs/test_dependency %S/inputs/target -o %t.target.mlirbc
# RUN: kgen-opt %t.target.mlirbc | FileCheck %s

# CHECK: kgen.generator export @"kgen_module.mojo.tmp.target::impl::anchor()"() -> index
# CHECK-SAME: linkageName = "anchor" : !kgen.string
# CHECK: kgen.generator @"test_dependency::impl::use_me()"() -> index
# CHECK: kgen.generator @"test_dependency::impl::child()"() -> index
# CHECK-NOT: dead


def main():
    pass
