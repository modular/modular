# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | FileCheck %s

# COM: Check that closure structs are deduplicated.

# CHECK-COUNT-1: lit.struct.decl @"`_CI_
# CHECK-COUNT-1: lit.struct.decl @"fn(


alias Int = __mlir_type.index


fn use(a: Int):
    pass


fn makes_escaping_closure(a: Int):
    fn dummy(n: Int) escaping:
        use(a)

    fn duplicate(n: Int) escaping:
        use(a)
