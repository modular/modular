# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics


def takeIt[T: def () -> None, //](state: T):
    state()

struct MoveMe(Movable):
    var x:Int
    def __init__(out self, *, deinit take: Self):
        self.x = take.x
    def __del__(deinit self:Self):
        pass

def use(d:MoveMe):
    pass

# CHECK-LABEL:  lit.fn @"toy
def toy(var byMove: MoveMe): # expected-note {{'byMove' declared here}}
    def myclosure() {var byMove^}:
        use(byMove)

    use(byMove) # expected-error {{use of uninitialized value 'byMove'}}
    takeIt(myclosure)
