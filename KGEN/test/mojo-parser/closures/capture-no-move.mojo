# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s

# COM: Capture type cannot be moved.


struct StringNoMove:
    var size: Int

    fn __init__(inout self, sz: Int):
        self.size = sz

    fn __copyinit__(inout self, existing: Self):
        pass

    fn __del__(owned self):
        pass


fn use(x: StringNoMove):
    pass


# CHECK: lit.struct.decl @"_CI_{{.*}}::StringNoMove, /)
# CHECK: lit.struct.field field0 : !StringNoMove
# CHECK: lit.func @"__del__
# CHECK: lit.func @"__copyinit__
# CHECK: lit.func @"__moveinit__
fn makes_escaping_closure_from_nomove(m: StringNoMove):
    fn foo() escaping:
        use(m)
