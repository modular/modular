# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Check that closure structs are deduplicated.

# CHECK-COUNT-1: lit.struct.decl @"`_CI_
# CHECK-COUNT-1: lit.struct.decl @"fn(


fn use(a: Int):
    pass


fn makes_escaping_closure(a: Int):
    fn dummy(n: Int):
        use(a)

    fn duplicate(n: Int):
        use(a)
