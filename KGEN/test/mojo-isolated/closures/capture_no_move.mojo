# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Capture type cannot be moved.


struct StringNoMove:
    fn __copyinit__(inout self, existing: Self):
        pass

    fn __del__(owned self):
        pass


fn use(x: StringNoMove):
    pass


# CHECK: lit.struct.decl @"`_CI_
# CHECK-NEXT: destructor :!lit.signature
# CHECK-NEXT: move :!lit.signature
# CHECK-NEXT: copy :!lit.signature
# CHECK: lit.struct.field field0 : !StringNoMove
# CHECK: lit.func @"__del__
# CHECK: lit.func @"__copyinit__
# CHECK: lit.func @"__moveinit__
fn makes_escaping_closure_from_nomove(m: StringNoMove):
    fn foo():
        use(m)
