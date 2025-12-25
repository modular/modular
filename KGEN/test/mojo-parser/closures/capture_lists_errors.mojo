# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics


fn takeIt[T: fn () unified -> None, //](state: T):
    state()

struct MoveMe(Movable):
    var x:Int
    fn __moveinit__(out self, deinit other: Self):
        self.x = other.x
    fn __del__(deinit self:Self):
        pass

fn use(d:MoveMe):
    pass

# CHECK-LABEL:  lit.fn @"toy
fn toy(var byMove: MoveMe): # expected-note {{'byMove' declared here}}
    fn myclosure() unified {var byMove^}:
        use(byMove)

    use(byMove) # expected-error {{use of uninitialized value 'byMove'}}
    takeIt(myclosure)
