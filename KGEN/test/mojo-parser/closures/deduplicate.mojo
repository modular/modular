# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Check that closure structs are deduplicated.

# CHECK-COUNT-1: lit.struct.decl @"`_CI_
# CHECK-COUNT-1: lit.struct.decl @"def(


def use(a: Int):
    pass


def makes_escaping_closure(a: Int):
    def dummy(n: Int):
        use(a)

    def duplicate(n: Int):
        use(a)
