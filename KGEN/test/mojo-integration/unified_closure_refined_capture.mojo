# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

# A scope assumption (`comptime assert conforms_to(...)`) refines a generic
# parameter's bound for method/name resolution. Verify that the same refinement
# now reaches a unified closure's by-value captures: by-copy `{var}` and
# by-move `{var x^}` previously read the unrefined declared bound and rejected
# the implicit copy / move (MOCO-4229).


trait Weak(Copyable, ImplicitlyDeletable, Movable):
    pass


struct Thing(TrivialRegisterPassable, Weak):
    var v: Int

    def __init__(out self, v: Int):
        self.v = v


# NOTE: no `Copyable` here — `Copyable` requires `Movable`, which would make
# the declared bound movable and defeat the refinement test below.
trait WeakNoMove(ImplicitlyDeletable):
    pass


struct HeavyThing(Movable, WeakNoMove):
    var v: Int

    def __init__(out self, v: Int):
        self.v = v


def capture_by_copy[T: Weak](z: T):
    comptime assert conforms_to(type_of(z), ImplicitlyCopyable)

    # The by-copy capture must consult the refined bound to copy `z`.
    def f() {var z}:
        _ = z

    f()


def capture_by_ref[T: Weak](z: T):
    comptime assert conforms_to(type_of(z), ImplicitlyCopyable)

    # The by-ref form already worked; keep it covered to guard the witness path.
    def g() {ref z}:
        _ = z

    g()


def capture_by_move[T: WeakNoMove](var z: T):
    comptime assert conforms_to(type_of(z), Movable)

    # The by-move capture must consult the refined bound to move `z`.
    def h() {var z^}:
        _ = z

    h()


def main():
    capture_by_copy(Thing(1))
    capture_by_ref(Thing(2))
    capture_by_move(HeavyThing(3))
    # CHECK: ok
    print("ok")
