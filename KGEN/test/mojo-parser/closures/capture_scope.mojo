# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


def use(x: Int):
    pass


# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.field field0 : !Int


def makes_escaping_closure(x: Int):
    def bar():
        try:
            use(x)
        except:
            pass
