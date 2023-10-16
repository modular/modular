# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s

# COM: Check that closure structs are deduplicated.

# CHECK-COUNT-1: lit.struct.decl @"_CI_
# CHECK-COUNT-1: lit.struct.decl @"_CW_


fn use(a: Int):
    pass


fn makes_escaping_closure(a: Int):
    fn dummy(n: Int) escaping:
        use(a)

    fn duplicate(n: Int) escaping:
        use(a)
