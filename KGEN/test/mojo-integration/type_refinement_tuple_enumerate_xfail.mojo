# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Type refinement for `enumerate` tuple unpacking in generic for loops.
#
# XFAIL: *
# TODO(MOCO-4361): Remove the XFAIL and fold these cases back into
# `type_refinement_tuple.mojo` once generic `enumerate` unpacking compiles
# again. Since `Tuple` became conditionally `Deinitable`, `enumerate`
# yields `Tuple[Int, Iterator.Element]` and `Iterator.Element: Movable` erases
# the element's `Deinitable` refinement in a generic context, so the
# per-iteration tuple temporary can't be dropped. (Concrete-element `enumerate`
# still compiles; only the generic form regressed.)
#
# RUN: %mojo -debug-level full %s | FileCheck %s


trait Greetable:
    def greet(self) -> String:
        ...


struct Dog(
    Copyable,
    Deinitable,
    Greetable,
    ImplicitlyCopyable,
    Movable,
):
    var name: String

    def __init__(out self, name: String):
        self.name = name

    def greet(self) -> String:
        return "Woof! I'm " + self.name


# `enumerate`: index + element unpack (value iteration).
def test_enumerate_bind[
    T: ImplicitlyCopyable & Deinitable
](items: List[T]) where conforms_to(T, Greetable):
    for i, item in enumerate(items):
        print("enum:", i, item.greet())


# `enumerate` with `var` index and element bindings.
def test_enumerate_var[
    T: ImplicitlyCopyable & Deinitable
](items: List[T]) where conforms_to(T, Greetable):
    for var idx, var item in enumerate(items):
        print("enum_var:", idx, item.greet())


# `ref` enumerate pair: mixed index (value) and `ref` subscript on item.
def test_enumerate_ref_subscript[
    T: ImplicitlyCopyable & Deinitable
](items: List[T]) where conforms_to(T, Greetable):
    for ref pair in enumerate(items):
        print("enum_ref:", pair[0], pair[1].greet())


def main():
    var dogs = List[Dog]()
    dogs.append(Dog("Rex"))
    dogs.append(Dog("Bella"))

    # CHECK: enum: 0 Woof! I'm Rex
    # CHECK: enum: 1 Woof! I'm Bella
    test_enumerate_bind(dogs)

    # CHECK: enum_var: 0 Woof! I'm Rex
    # CHECK: enum_var: 1 Woof! I'm Bella
    test_enumerate_var(dogs)

    # CHECK: enum_ref: 0 Woof! I'm Rex
    # CHECK: enum_ref: 1 Woof! I'm Bella
    test_enumerate_ref_subscript(dogs)
