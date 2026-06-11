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
struct ConditionalCopyableWrapper[T: ImplicitlyDeletable & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDeletable,
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def __moveinit__(out self, deinit move: Self):
        self.value = move.value^

    def __copyinit__(
        out self, copy: Self, /
    ) where conforms_to(Self.T, Copyable):
        self.value = rebind_var[Self.T](
            trait_downcast[Copyable](copy.value).copy()
        )


# A function that requires Copyable
def needs_copyable[T: Copyable](x: T):
    pass


# A movable-only type (not Copyable)
struct MovableOnlyType(ImplicitlyDeletable, Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x

    def __moveinit__(out self, deinit move: Self):
        self.x = take.x


# ===========================================================================
# Test 2: Closure with conditionally-conforming return type (negative case)
# ===========================================================================


trait Printable:
    def print_value(self):
        ...


# A type that is ImplicitlyCopyable but NOT Printable
struct NonPrintableType(ImplicitlyCopyable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x

    def __moveinit__(out self, deinit move: Self):
        self.x = take.x

    def __copyinit__(out self, copy: Self, /):
        self.x = copy.x


# A wrapper with CONDITIONAL Printable conformance.
struct PrintableWrapper[T: ImplicitlyCopyable](
    ImplicitlyCopyable,
    Printable where conforms_to(T, Printable),
):
    var value: Self.T

    def __init__(out self, value: Self.T):
        self.value = value

    def __moveinit__(out self, deinit move: Self):
        self.value = take.value

    def __copyinit__(out self, copy: Self, /):
        self.value = copy.value

    def print_value(self) where conforms_to(Self.T, Printable):
        trait_downcast[Printable](self.value).print_value()


def use_printable_closure[
    T: Printable & ImplicitlyCopyable, C: def() -> T
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
def unsound_generic_call[
    T: ImplicitlyDeletable & Movable
](x: ConditionalCopyableWrapper[T]):
    needs_copyable(x)


# ===========================================================================
# Test: Unsound variadic pack call is rejected
# ===========================================================================
# Tuple[*types] with *types: Movable has no Copyable bound on its elements,
# so its conditional Copyable conformance (AllCopyable) can't be proven.


# CHECK: argument type 'Tuple[types]' does not conform to trait 'Copyable'
def unsound_variadic_call[*types: Movable](t: Tuple[*types]):
    needs_copyable(t)


# ===========================================================================
# Test: where clause on wrong trait doesn't prove conformance
# ===========================================================================
# A where clause for Intable does not help prove Copyable conformance.


# CHECK: argument type 'ConditionalCopyableWrapper[T]' does not conform to trait 'Copyable'
def wrong_where_clause[
    T: ImplicitlyDeletable & Movable
](x: ConditionalCopyableWrapper[T]) where conforms_to(T, Intable):
    needs_copyable(x)


# ===========================================================================
# Test: Alias-resolved field type with unsatisfied conditional conformance
# ===========================================================================
# When a struct uses comptime aliases to build its field type and the
# conditional conformance constraint isn't met, the compiler must reject
# attempts to copy the struct.


@fieldwise_init
struct AliasWrapper[T: ImplicitlyDeletable & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDeletable,
    Movable,
):
    comptime Inner = ConditionalCopyableWrapper[Self.T]
    var _field: Self.Inner


# CHECK: argument type 'AliasWrapper[MovableOnlyType]' does not conform to trait 'Copyable'
def alias_needs_copyable():
    var w = AliasWrapper(_field=ConditionalCopyableWrapper(MovableOnlyType(1)))
    needs_copyable(w)


# ===========================================================================
# Test: Conditional RegisterPassable conformance rejected for non-RP type
# ===========================================================================
# ConditionalRPWrapper[T] is RegisterPassable only when T is RegisterPassable.
# Passing ConditionalRPWrapper[MovableOnlyType] (where MovableOnlyType is NOT
# RegisterPassable) to a function requiring RegisterPassable must be rejected.


struct ConditionalRPWrapper[T: ImplicitlyDeletable & Movable](
    ImplicitlyDeletable,
    Movable,
    RegisterPassable where conforms_to(T, RegisterPassable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


def needs_rp[T: RegisterPassable](x: T):
    pass


def main():
    # ConditionalCopyableWrapper[MovableOnlyType] should NOT be Copyable because
    # MovableOnlyType is not Copyable.
    var wrapped = ConditionalCopyableWrapper(MovableOnlyType(42))
    # CHECK: argument type 'ConditionalCopyableWrapper[MovableOnlyType]' does not conform to trait 'Copyable'
    needs_copyable(wrapped)

    # PrintableWrapper[NonPrintableType] should NOT conform to Printable because
    # NonPrintableType is not Printable.
    var captured = NonPrintableType(42)

    def make_wrapper() {var} -> PrintableWrapper[NonPrintableType]:
        return PrintableWrapper(captured)

    # CHECK: 'use_printable_closure' parameter 'T' has 'Printable & ImplicitlyCopyable' type, but value has type 'AnyStruct[PrintableWrapper[NonPrintableType]]'
    use_printable_closure[
        PrintableWrapper[NonPrintableType], type_of(make_wrapper)
    ](make_wrapper)

    # ConditionalRPWrapper[MovableOnlyType] should NOT be RegisterPassable
    # because MovableOnlyType is not RegisterPassable.
    var rp_wrapped = ConditionalRPWrapper(MovableOnlyType(99))
    # CHECK: argument type 'ConditionalRPWrapper[MovableOnlyType]' does not conform to trait 'RegisterPassable'
    needs_rp(rp_wrapped)
