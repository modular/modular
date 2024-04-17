# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use(x: int):
    pass


# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.field field0 : index


fn makes_escaping_closure(x: int):
    fn bar():
        try:
            use(x)
        except:
            pass
