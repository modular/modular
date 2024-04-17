# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Check that closure structs are deduplicated.

# CHECK-COUNT-1: lit.struct.decl @"`_CI_
# CHECK-COUNT-1: lit.struct.decl @"fn(


fn use(a: int):
    pass


fn makes_escaping_closure(a: int):
    fn dummy(n: int):
        use(a)

    fn duplicate(n: int):
        use(a)
