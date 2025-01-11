# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Capture type cannot be moved.


struct StringNoMove:
    fn __copyinit__(out self, existing: Self):
        pass

    fn __del__(owned self):
        pass


fn use(x: StringNoMove):
    pass


# CHECK: lit.struct.decl @"`_CI_
# CHECK-NEXT: destructor :!lit.generator
# CHECK-NEXT: move :!lit.generator
# CHECK-NEXT: copy :!lit.generator
# CHECK: lit.struct.field field0 : !StringNoMove
# CHECK: lit.fn @"__del__
# CHECK: lit.fn @"__copyinit__
# CHECK: lit.fn @"__moveinit__
fn makes_escaping_closure_from_nomove(m: StringNoMove):
    fn foo():
        use(m)
