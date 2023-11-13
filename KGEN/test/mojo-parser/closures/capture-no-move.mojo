# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s

# COM: Capture type cannot be moved.


struct StringNoMove:
    fn __copyinit__(inout self, existing: Self):
        pass

    fn __del__(owned self):
        pass


fn use(x: StringNoMove):
    pass


# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: copyInit =
# CHECK-SAME: destructor =
# CHECK-SAME: moveInit =
# CHECK: lit.struct.field field0 : !StringNoMove
# CHECK: lit.func @"__del__
# CHECK: lit.func @"__copyinit__
# CHECK: lit.func @"__moveinit__
fn makes_escaping_closure_from_nomove(m: StringNoMove):
    fn foo() escaping:
        use(m)
