# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use(x: Int):
    pass


# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.field field0 : !Int


fn makes_escaping_closure(x: Int):
    fn bar():
        try:
            use(x)
        except:
            pass
