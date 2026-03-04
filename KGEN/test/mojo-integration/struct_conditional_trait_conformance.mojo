# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test end-to-end conditional trait conformance - positive cases.
# This tests that a struct with a conditional conformance (e.g., Copyable where
# conforms_to(T, Copyable)) correctly satisfies the trait when the condition is
# met across various patterns.

# RUN: %mojo -debug-level full %s | FileCheck %s


# ===========================================================================
# Basic Types for Testing
# ===========================================================================


# A simple copyable type
struct CopyableType(Copyable, ImplicitlyDestructible, Movable):
    var x: Int

    fn __init__(out self, x: Int):
        self.x = x


# An implicitly-copyable type (ImplicitlyCopyable ⊃ Copyable ⊃ Movable)
struct ImplCopyableType(
    Copyable, ImplicitlyCopyable, ImplicitlyDestructible, Movable
):
    var x: Int

    fn __init__(out self, x: Int):
        self.x = x


# ===========================================================================
# Test 1: Simple Conditional Conformance
# ===========================================================================


# A simple wrapper struct with conditional Copyable conformance.
# Wrapper[T] is Copyable if and only if T is Copyable.
struct SimpleWrapper[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDestructible,
    Movable,
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.value = rebind_var[Self.T](
            trait_downcast[Copyable](copy.value).copy()
        )


fn needs_copyable[T: Copyable](x: T):
    print("needs_copyable: Type is Copyable!")


fn test_simple_conditional():
    var wrapped = SimpleWrapper(CopyableType(42))
    # CHECK: needs_copyable: Type is Copyable!
    needs_copyable(wrapped)


# ===========================================================================
# Test 2: Nested Wrappers
# ===========================================================================


fn test_nested_wrappers():
    # SimpleWrapper[CopyableType] is Copyable
    # SimpleWrapper[SimpleWrapper[CopyableType]] is also Copyable
    var inner = SimpleWrapper(CopyableType(42))
    var outer = SimpleWrapper(inner^)
    # CHECK: needs_copyable: Type is Copyable!
    needs_copyable(outer)


# ===========================================================================
# Test 3: Function Type Conversion with Conditional Conformance
# ===========================================================================

# This tests that function type conversion correctly handles types with
# conditional trait conformance. The fix in ExprConversions.cpp ensures
# we pass the concrete type when checking conformance for function argument
# covariance.


fn takes_copyable_via_trait[T: Copyable & ImplicitlyDestructible](x: T):
    var copy = x.copy()
    _ = copy^
    # CHECK: Function type conversion with conditional works
    print("Function type conversion with conditional works")


fn test_function_type_conversion():
    # SimpleWrapper[CopyableType] has conditional Copyable conformance
    # This tests that the conformance is correctly detected
    var wrapped = SimpleWrapper(CopyableType(42))
    takes_copyable_via_trait(wrapped.copy())


# ===========================================================================
# Test 4: Unified Closures with Conditionally-Conforming Return Types
# ===========================================================================
# TODO(#77058): Re-enable after lazy conformance is added.
# Eager Conformance broke concrete-return-type closure patterns with
# conditional conformance — the closure wrapper's internal parameter type
# is no longer resolved to the concrete type during conformance checking.
#
# trait Printable:
#     fn print_value(self):
#         ...
#
#
# @fieldwise_init
# struct SimplePrintable(ImplicitlyCopyable, Printable):
#     var x: Int
#
#     fn print_value(self):
#         print("SimplePrintable:", self.x)
#
#
# struct PrintableWrapper[T: ImplicitlyCopyable](
#     ImplicitlyCopyable,
#     Printable where conforms_to(T, Printable),
# ):
#     var value: Self.T
#
#     fn __init__(out self, value: Self.T):
#         self.value = value
#
#     fn __moveinit__(out self, deinit take: Self):
#         self.value = take.value
#
#     fn __copyinit__(out self, copy: Self, /):
#         self.value = copy.value
#
#     fn print_value(self) where conforms_to(Self.T, Printable):
#         trait_downcast[Printable](self.value).print_value()
#
#
# fn use_printable_closure[
#     T: Printable & ImplicitlyCopyable, C: fn() unified -> T
# ](impl: C):
#     var result = impl()
#     result.print_value()
#
#
# fn test_closure_with_conditional_return():
#     var captured = SimplePrintable(42)
#
#     fn make_wrapper() unified {var} -> PrintableWrapper[SimplePrintable]:
#         return PrintableWrapper(captured)
#
#     # COM: CHECK: SimplePrintable: 42
#     use_printable_closure[
#         PrintableWrapper[SimplePrintable], type_of(make_wrapper)
#     ](make_wrapper)
#
#
# fn test_nested_conditional_closure():
#     var val = SimplePrintable(100)
#
#     fn make_nested() unified {
#         var val
#     } -> PrintableWrapper[PrintableWrapper[SimplePrintable]]:
#         return PrintableWrapper(PrintableWrapper(val))
#
#     # COM: CHECK: SimplePrintable: 100
#     use_printable_closure[
#         PrintableWrapper[PrintableWrapper[SimplePrintable]],
#         type_of(make_nested),
#     ](make_nested)


# ===========================================================================
# Test 5: Diamond Pattern with Both Branches Conditional
# ===========================================================================

# Diamond hierarchy:
#        Base
#       /    \
#  DerivedA  DerivedB
#       \    /
#     DiamondStruct


# A type that is both Copyable and Intable
struct CopyableIntableType(Copyable, ImplicitlyDestructible, Intable, Movable):
    var x: Int

    fn __init__(out self, x: Int):
        self.x = x

    fn __int__(self) -> Int:
        return self.x


trait Base:
    pass


trait DerivedA(Base):
    pass


trait DerivedB(Base):
    pass


# DiamondBothConditional conforms to:
# - DerivedA when T is Copyable
# - DerivedB when T is Intable
# - Base when T is Copyable OR T is Intable (must be explicitly listed)
# The derived constraints imply Base's OR constraint via the weakening rule:
# A implies (A OR B), so conforms_to(T, Copyable) implies the OR.
struct DiamondBothConditional[T: ImplicitlyDestructible & Movable](
    # Base must be explicitly listed with its OR constraint
    Base where conforms_to(T, Copyable) or conforms_to(T, Intable),
    # DerivedA's constraint implies Base's via weakening: A implies (A OR B)
    DerivedA where conforms_to(T, Copyable),
    # DerivedB's constraint implies Base's via weakening: B implies (A OR B)
    DerivedB where conforms_to(T, Intable),
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^


fn needs_base[T: Base](x: T):
    print("needs_base: Type conforms to Base!")


fn needs_derived_a[T: DerivedA](x: T):
    print("needs_derived_a: Type conforms to DerivedA!")


fn needs_derived_b[T: DerivedB](x: T):
    print("needs_derived_b: Type conforms to DerivedB!")


fn test_diamond_both_conditional():
    # CopyableIntableType is both Copyable and Intable
    # So DiamondBothConditional[CopyableIntableType] conforms to all three traits
    var diamond = DiamondBothConditional(CopyableIntableType(42))

    # CHECK: needs_base: Type conforms to Base!
    needs_base(diamond)
    # CHECK: needs_derived_a: Type conforms to DerivedA!
    needs_derived_a(diamond)
    # CHECK: needs_derived_b: Type conforms to DerivedB!
    needs_derived_b(diamond)


# ===========================================================================
# Test 6: Diamond Pattern with One Unconditional Branch
# ===========================================================================

# When one path is unconditional, the ancestor is also unconditional
# (True OR cond) = True


# A movable-only type (not Copyable)
struct MovableOnlyType(ImplicitlyDestructible, Movable):
    var x: Int

    fn __init__(out self, x: Int):
        self.x = x


struct DiamondOneUnconditional[T: ImplicitlyDestructible & Movable](
    DerivedA where conforms_to(T, Copyable),
    DerivedB,
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^


fn test_diamond_one_unconditional():
    # MovableOnlyType is NOT Copyable
    # But DiamondOneUnconditional[MovableOnlyType] still conforms to Base
    # because DerivedB is unconditional
    var diamond = DiamondOneUnconditional(MovableOnlyType(42))

    # CHECK: needs_base: Type conforms to Base!
    needs_base(diamond)
    # CHECK: needs_derived_b: Type conforms to DerivedB!
    needs_derived_b(diamond)


# ===========================================================================
# Test 7: Explicit Ancestor in Inheritance List
# ===========================================================================


# When an ancestor is explicitly listed, it uses the explicit constraint
# (or no constraint if listed unconditionally)
struct ExplicitAncestor[T: ImplicitlyDestructible & Movable](
    Base,
    DerivedA where conforms_to(T, Copyable),
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^


fn test_explicit_ancestor():
    # MovableOnlyType is NOT Copyable
    # But ExplicitAncestor[MovableOnlyType] conforms to Base
    # because Base is explicitly listed without a constraint
    var explicit = ExplicitAncestor(MovableOnlyType(42))

    # CHECK: needs_base: Type conforms to Base!
    needs_base(explicit)


# ===========================================================================
# Test 8: Multiple Conditional Conformances
# ===========================================================================


struct MultipleConditional[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    Intable where conforms_to(T, Intable),
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    fn __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.data = rebind_var[Self.T](
            trait_downcast[Copyable](copy.data).copy()
        )

    fn __int__(self) -> Int where conforms_to(Self.T, Intable):
        return 0


fn needs_intable[T: Intable](x: T):
    print("needs_intable: Type is Intable!")


fn test_multiple_conditional():
    # CopyableIntableType is both Copyable and Intable
    var multi = MultipleConditional(CopyableIntableType(42))

    # CHECK: needs_copyable: Type is Copyable!
    needs_copyable(multi)
    # CHECK: needs_intable: Type is Intable!
    needs_intable(multi)


# ===========================================================================
# Test 9: Stronger conformance constraint implies weaker method constraint
# ===========================================================================


trait RequiresMethod:
    fn method(self):
        ...


struct StrongerConformance[T: ImplicitlyDestructible & Movable](
    # Conformance requires BOTH Copyable AND Intable
    RequiresMethod where conforms_to(T, Copyable) and conforms_to(T, Intable),
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    # Method only requires Copyable - satisfied by stronger conformance constraint
    fn method(self) where conforms_to(Self.T, Copyable):
        print("StrongerConformance.method")


fn test_stronger_implies_weaker():
    # CopyableIntableType is both Copyable and Intable
    var s = StrongerConformance(CopyableIntableType(42))
    # CHECK: StrongerConformance.method
    s.method()


# ===========================================================================
# Test 10: Diamond with Same Constraint on Both Paths (auto-propagated)
# ===========================================================================

# Uses the Base/DerivedA/DerivedB trait diamond from Test 5 above.
# Both DerivedA and DerivedB carry the same constraint (T: Copyable), so Base
# is reached via two paths that agree. The compiler should silently propagate
# the constraint to Base without requiring explicit listing.


struct DiamondSameConstraint[T: ImplicitlyDestructible & Movable](
    DerivedA where conforms_to(T, Copyable),
    DerivedB where conforms_to(T, Copyable),
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^


fn test_diamond_same_constraint():
    # CopyableIntableType is both Copyable and Intable, so the constraint
    # conforms_to(T, Copyable) is satisfied. Base is auto-propagated with the
    # same constraint from both paths.
    var diamond = DiamondSameConstraint(CopyableIntableType(42))

    # CHECK: needs_base: Type conforms to Base!
    needs_base(diamond)
    # CHECK: needs_derived_a: Type conforms to DerivedA!
    needs_derived_a(diamond)
    # CHECK: needs_derived_b: Type conforms to DerivedB!
    needs_derived_b(diamond)


# ===========================================================================
# Test 11: Diamond with Reordered Compound Constraints (logical equivalence)
# ===========================================================================

# Uses the same Base/DerivedA/DerivedB diamond. Both paths carry a compound
# constraint but in different operand order:
#   DerivedA where Copyable and Intable
#   DerivedB where Intable and Copyable
# These are logically equivalent but produce different MLIR attrs (And is not
# commutative in the attribute representation). The compiler uses mutual
# implication to recognize them as agreeing and auto-propagate to Base.


struct DiamondReorderedConstraint[T: ImplicitlyDestructible & Movable](
    DerivedA where conforms_to(T, Copyable) and conforms_to(T, Intable),
    DerivedB where conforms_to(T, Intable) and conforms_to(T, Copyable),
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^


fn test_diamond_reordered_constraint():
    # CopyableIntableType satisfies both Copyable and Intable.
    # The two paths to Base carry "Copyable and Intable" vs "Intable and Copyable"
    # which are logically equivalent, so Base should auto-propagate.
    var diamond = DiamondReorderedConstraint(CopyableIntableType(42))

    # CHECK: needs_base: Type conforms to Base!
    needs_base(diamond)
    # CHECK: needs_derived_a: Type conforms to DerivedA!
    needs_derived_a(diamond)
    # CHECK: needs_derived_b: Type conforms to DerivedB!
    needs_derived_b(diamond)


# ===========================================================================
# Test: Violated constraint takes precedence over unprovable
# ===========================================================================
# When a method has multiple where clauses and one is unprovable but another
# is directly contradicted by the conformance constraint, the method should
# be rejected (Violated) rather than flagged as an error (Unprovable).
# This allows the trait's default implementation to be used.


trait ViolatedPrecedenceTrait:
    fn get_label(read self) -> Int:
        return 0


struct ViolatedTakesPrecedence[T: ImplicitlyDestructible & Movable](
    ImplicitlyDestructible,
    Movable,
    ViolatedPrecedenceTrait where conforms_to(T, Copyable),
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    # First where clause: unprovable (Intable is independent of Copyable).
    # Second where clause: contradicts the conformance (conformance requires
    # Copyable, but this requires NOT Copyable).
    # The method should be Violated (rejected), not Unprovable (error).
    fn get_label(
        read self,
    ) -> Int where conforms_to(Self.T, Intable) where not conforms_to(
        Self.T, Copyable
    ):
        return 1


fn needs_violated_precedence[T: ViolatedPrecedenceTrait](x: T):
    print("violated_precedence:", x.get_label())


fn test_violated_precedence():
    var v = ViolatedTakesPrecedence(CopyableType(1))
    # CHECK: violated_precedence: 0
    needs_violated_precedence(v)


# ===========================================================================
# Test 12: Conditional Movable transfer via conforms_to(T, Copyable)
# ===========================================================================
# The struct is conditionally Movable gated by conforms_to(T, Copyable).
# The transfer operator (^) should work when T satisfies the condition.


struct MovableViaCopyable[T: ImplicitlyDestructible & Movable](
    ImplicitlyDestructible,
    Movable where conforms_to(T, Copyable),
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^


fn test_move_via_copyable():
    var original = MovableViaCopyable(CopyableType(100))
    var moved = original^
    # CHECK: move_via_copyable: 100
    print("move_via_copyable:", moved.value.x)


# ===========================================================================
# Test 13: Conditional Movable transfer via conforms_to(T, ImplicitlyCopyable)
# ===========================================================================
# Same as Test 12, but gated by conforms_to(T, ImplicitlyCopyable).
# ImplicitlyCopyable also subsumes Movable, so the transfer should work.


struct MovableViaImplCopyable[T: ImplicitlyDestructible & Movable](
    ImplicitlyDestructible,
    Movable where conforms_to(T, ImplicitlyCopyable),
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^


fn test_move_via_impl_copyable():
    var original = MovableViaImplCopyable(ImplCopyableType(200))
    var moved = original^
    # CHECK: move_via_impl_copyable: 200
    print("move_via_impl_copyable:", moved.value.x)


# ===========================================================================
# Test 14: Both Movable and Copyable as conditional conformances
# ===========================================================================
# Both Movable and Copyable are conditional.  The Copyable constraint must
# explicitly imply the Movable constraint (ancestor requirement).  Both
# transfer (^) and copy should work when conditions are met.


struct BothConditionalMoveCopy[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable) and conforms_to(T, Movable),
    ImplicitlyDestructible,
    Movable where conforms_to(T, Movable),
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Movable):
        self.value = rebind_var[Self.T](
            trait_downcast[Copyable](copy.value).copy()
        )


fn test_both_conditional_move_copy():
    var original = BothConditionalMoveCopy(CopyableType(300))
    var copied = original.copy()
    # CHECK: both_cond_copy: 300
    print("both_cond_copy:", copied.value.x)

    var another = BothConditionalMoveCopy(CopyableType(400))
    var moved = another^
    # CHECK: both_cond_move: 400
    print("both_cond_move:", moved.value.x)


# ===========================================================================
# Main
# ===========================================================================


fn main():
    test_simple_conditional()
    test_nested_wrappers()
    test_function_type_conversion()
    # TODO(#77058): Re-enable after lazy conformance is added.
    # test_closure_with_conditional_return()
    # test_nested_conditional_closure()
    test_diamond_both_conditional()
    test_diamond_one_unconditional()
    test_explicit_ancestor()
    test_multiple_conditional()
    test_stronger_implies_weaker()
    test_diamond_same_constraint()
    test_diamond_reordered_constraint()
    test_violated_precedence()
    test_move_via_copyable()
    test_move_via_impl_copyable()
    test_both_conditional_move_copy()
