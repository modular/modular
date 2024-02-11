# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | FileCheck %s


fn use(x: Int):
    pass


# CHECK: lit.struct.decl @"`_CI_
# CHECK-NEXT: lit.struct.field field0 : index


fn makes_escaping_closure(x: Int):
    fn bar() escaping:
        try:
            use(x)
        except:
            pass
