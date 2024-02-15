# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


fn use(x: int):
    pass


# CHECK: lit.struct.decl @"`_CI_
# CHECK-NEXT: lit.struct.field field0 : index


fn makes_escaping_closure(x: int):
    fn bar() escaping:
        try:
            use(x)
        except:
            pass
