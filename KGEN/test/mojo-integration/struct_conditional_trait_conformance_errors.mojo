# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test end-to-end conditional trait conformance - negative case.
# This tests that a struct with a conditional conformance correctly fails to
# satisfy the trait when the condition is NOT met.

# RUN: not %mojo %s 2>&1 | FileCheck %s


# A wrapper struct with conditional Copyable conformance.
# ConditionalCopyableWrapper[T] is Copyable if and only if T is Copyable.
struct ConditionalCopyableWrapper[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDestructible,
    Movable,
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn __moveinit__(out self, deinit take: Self):
        self.value = take.value^

    fn __copyinit__(
        out self, copy: Self, /
    ) where conforms_to(Self.T, Copyable):
        self.value = rebind_var[Self.T](
            trait_downcast[Copyable](copy.value).copy()
        )


# A function that requires Copyable
fn needs_copyable[T: Copyable](x: T):
    pass


# A movable-only type (not Copyable)
struct MovableOnlyType(ImplicitlyDestructible, Movable):
    var x: Int

    fn __init__(out self, x: Int):
        self.x = x

    fn __moveinit__(out self, deinit take: Self):
        self.x = take.x


# ===========================================================================
# Test 2: Closure with conditionally-conforming return type (negative case)
# ===========================================================================


trait Printable:
    fn print_value(self):
        ...


# A type that is ImplicitlyCopyable but NOT Printable
struct NonPrintableType(ImplicitlyCopyable):
    var x: Int

    fn __init__(out self, x: Int):
        self.x = x

    fn __moveinit__(out self, deinit take: Self):
        self.x = take.x

    fn __copyinit__(out self, copy: Self, /):
        self.x = copy.x


# A wrapper with CONDITIONAL Printable conformance.
struct PrintableWrapper[T: ImplicitlyCopyable](
    ImplicitlyCopyable,
    Printable where conforms_to(T, Printable),
):
    var value: Self.T

    fn __init__(out self, value: Self.T):
        self.value = value

    fn __moveinit__(out self, deinit take: Self):
        self.value = take.value

    fn __copyinit__(out self, copy: Self, /):
        self.value = copy.value

    fn print_value(self) where conforms_to(Self.T, Printable):
        trait_downcast[Printable](self.value).print_value()


fn use_printable_closure[
    T: Printable & ImplicitlyCopyable, C: fn() unified -> T
](impl: C):
    var result = impl()
    result.print_value()


# ===========================================================================
# Test: Unsound call with symbolic type parameter is rejected
# ===========================================================================
# When T has no Copyable bound, ConditionalCopyableWrapper[T] is not provably
# Copyable. Passing it to a function requiring Copyable must be rejected at
# parse time, not deferred to elaboration.


# CHECK: argument type 'ConditionalCopyableWrapper[T]' does not conform to trait 'Copyable'
fn unsound_generic_call[
    T: ImplicitlyDestructible & Movable
](x: ConditionalCopyableWrapper[T]):
    needs_copyable(x)


# ===========================================================================
# Test: Unsound variadic pack call is rejected
# ===========================================================================
# Tuple[*types] with *types: Movable has no Copyable bound on its elements,
# so its conditional Copyable conformance (AllCopyable) can't be proven.


# CHECK: argument type 'Tuple[types]' does not conform to trait 'Copyable'
fn unsound_variadic_call[*types: Movable](t: Tuple[*types]):
    needs_copyable(t)


# ===========================================================================
# Test: where clause on wrong trait doesn't prove conformance
# ===========================================================================
# A where clause for Intable does not help prove Copyable conformance.


# CHECK: argument type 'ConditionalCopyableWrapper[T]' does not conform to trait 'Copyable'
fn wrong_where_clause[
    T: ImplicitlyDestructible & Movable
](x: ConditionalCopyableWrapper[T]) where conforms_to(T, Intable):
    needs_copyable(x)


fn main():
    # ConditionalCopyableWrapper[MovableOnlyType] should NOT be Copyable because
    # MovableOnlyType is not Copyable.
    var wrapped = ConditionalCopyableWrapper(MovableOnlyType(42))
    # CHECK: argument type 'ConditionalCopyableWrapper[MovableOnlyType]' does not conform to trait 'Copyable'
    needs_copyable(wrapped)

    # PrintableWrapper[NonPrintableType] should NOT conform to Printable because
    # NonPrintableType is not Printable.
    var captured = NonPrintableType(42)

    fn make_wrapper() unified {var} -> PrintableWrapper[NonPrintableType]:
        return PrintableWrapper(captured)

    # CHECK: 'use_printable_closure' parameter 'T' has 'Printable & ImplicitlyCopyable' type, but value has type 'AnyStruct[PrintableWrapper[NonPrintableType]]'
    use_printable_closure[
        PrintableWrapper[NonPrintableType], type_of(make_wrapper)
    ](make_wrapper)
