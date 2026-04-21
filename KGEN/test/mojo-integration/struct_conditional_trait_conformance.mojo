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

    def __init__(out self, x: Int):
        self.x = x


# An implicitly-copyable type (ImplicitlyCopyable ⊃ Copyable ⊃ Movable)
struct ImplCopyableType(
    Copyable, ImplicitlyCopyable, ImplicitlyDestructible, Movable
):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


# ===========================================================================
# Simple Conditional Conformance
# ===========================================================================


# A simple wrapper struct with conditional Copyable conformance.
# Wrapper[T] is Copyable if and only if T is Copyable.
struct SimpleWrapper[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDestructible,
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.value = rebind_var[Self.T](
            trait_downcast[Copyable](copy.value).copy()
        )


def needs_copyable[T: Copyable](x: T):
    print("needs_copyable: Type is Copyable!")


def test_simple_conditional():
    var wrapped = SimpleWrapper(CopyableType(42))
    # CHECK: needs_copyable: Type is Copyable!
    needs_copyable(wrapped)


# ===========================================================================
# Nested Wrappers
# ===========================================================================


def test_nested_wrappers():
    # SimpleWrapper[CopyableType] is Copyable
    # SimpleWrapper[SimpleWrapper[CopyableType]] is also Copyable
    var inner = SimpleWrapper(CopyableType(42))
    var outer = SimpleWrapper(inner^)
    # CHECK: needs_copyable: Type is Copyable!
    needs_copyable(outer)


# ===========================================================================
# Function Type Conversion with Conditional Conformance
# ===========================================================================

# This tests that function type conversion correctly handles types with
# conditional trait conformance. The fix in ExprConversions.cpp ensures
# we pass the concrete type when checking conformance for function argument
# covariance.


def takes_copyable_via_trait[T: Copyable & ImplicitlyDestructible](x: T):
    var copy = x.copy()
    _ = copy^
    # CHECK: Function type conversion with conditional works
    print("Function type conversion with conditional works")


def test_function_type_conversion():
    # SimpleWrapper[CopyableType] has conditional Copyable conformance
    # This tests that the conformance is correctly detected
    var wrapped = SimpleWrapper(CopyableType(42))
    takes_copyable_via_trait(wrapped.copy())


# ===========================================================================
# Unified Closures with Conditionally-Conforming Return Types
# ===========================================================================
# TODO(#77058): Re-enable after lazy conformance is added.
# Eager Conformance broke concrete-return-type closure patterns with
# conditional conformance — the closure wrapper's internal parameter type
# is no longer resolved to the concrete type during conformance checking.
#
# trait Printable:
#     def print_value(self):
#         ...
#
#
# @fieldwise_init
# struct SimplePrintable(ImplicitlyCopyable, Printable):
#     var x: Int
#
#     def print_value(self):
#         print("SimplePrintable:", self.x)
#
#
# struct PrintableWrapper[T: ImplicitlyCopyable](
#     ImplicitlyCopyable,
#     Printable where conforms_to(T, Printable),
# ):
#     var value: Self.T
#
#     def __init__(out self, value: Self.T):
#         self.value = value
#
#     def __moveinit__(out self, deinit take: Self):
#         self.value = take.value
#
#     def __copyinit__(out self, copy: Self, /):
#         self.value = copy.value
#
#     def print_value(self) where conforms_to(Self.T, Printable):
#         trait_downcast[Printable](self.value).print_value()
#
#
# def use_printable_closure[
#     T: Printable & ImplicitlyCopyable, C: def() unified -> T
# ](impl: C):
#     var result = impl()
#     result.print_value()
#
#
# def test_closure_with_conditional_return():
#     var captured = SimplePrintable(42)
#
#     def make_wrapper() unified {var} -> PrintableWrapper[SimplePrintable]:
#         return PrintableWrapper(captured)
#
#     # COM: CHECK: SimplePrintable: 42
#     use_printable_closure[
#         PrintableWrapper[SimplePrintable], type_of(make_wrapper)
#     ](make_wrapper)
#
#
# def test_nested_conditional_closure():
#     var val = SimplePrintable(100)
#
#     def make_nested() unified {
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
# Diamond Pattern with Both Branches Conditional
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

    def __init__(out self, x: Int):
        self.x = x

    def __int__(self) -> Int:
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

    def __init__(out self, var data: Self.T):
        self.data = data^


def needs_base[T: Base](x: T):
    print("needs_base: Type conforms to Base!")


def needs_derived_a[T: DerivedA](x: T):
    print("needs_derived_a: Type conforms to DerivedA!")


def needs_derived_b[T: DerivedB](x: T):
    print("needs_derived_b: Type conforms to DerivedB!")


def test_diamond_both_conditional():
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
# Diamond Pattern with One Unconditional Branch
# ===========================================================================

# When one path is unconditional, the ancestor is also unconditional
# (True OR cond) = True


# A movable-only type (not Copyable)
struct MovableOnlyType(ImplicitlyDestructible, Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


struct DiamondOneUnconditional[T: ImplicitlyDestructible & Movable](
    DerivedA where conforms_to(T, Copyable),
    DerivedB,
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^


def test_diamond_one_unconditional():
    # MovableOnlyType is NOT Copyable
    # But DiamondOneUnconditional[MovableOnlyType] still conforms to Base
    # because DerivedB is unconditional
    var diamond = DiamondOneUnconditional(MovableOnlyType(42))

    # CHECK: needs_base: Type conforms to Base!
    needs_base(diamond)
    # CHECK: needs_derived_b: Type conforms to DerivedB!
    needs_derived_b(diamond)


# ===========================================================================
# Explicit Ancestor in Inheritance List
# ===========================================================================


# When an ancestor is explicitly listed, it uses the explicit constraint
# (or no constraint if listed unconditionally)
struct ExplicitAncestor[T: ImplicitlyDestructible & Movable](
    Base,
    DerivedA where conforms_to(T, Copyable),
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^


def test_explicit_ancestor():
    # MovableOnlyType is NOT Copyable
    # But ExplicitAncestor[MovableOnlyType] conforms to Base
    # because Base is explicitly listed without a constraint
    var explicit = ExplicitAncestor(MovableOnlyType(42))

    # CHECK: needs_base: Type conforms to Base!
    needs_base(explicit)


# ===========================================================================
# Multiple Conditional Conformances
# ===========================================================================


struct MultipleConditional[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    Intable where conforms_to(T, Intable),
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^

    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.data = rebind_var[Self.T](
            trait_downcast[Copyable](copy.data).copy()
        )

    def __int__(self) -> Int where conforms_to(Self.T, Intable):
        return 0


def needs_intable[T: Intable](x: T):
    print("needs_intable: Type is Intable!")


def test_multiple_conditional():
    # CopyableIntableType is both Copyable and Intable
    var multi = MultipleConditional(CopyableIntableType(42))

    # CHECK: needs_copyable: Type is Copyable!
    needs_copyable(multi)
    # CHECK: needs_intable: Type is Intable!
    needs_intable(multi)


# ===========================================================================
# Stronger conformance constraint implies weaker method constraint
# ===========================================================================


trait RequiresMethod:
    def method(self):
        ...


struct StrongerConformance[T: ImplicitlyDestructible & Movable](
    # Conformance requires BOTH Copyable AND Intable
    RequiresMethod where conforms_to(T, Copyable) and conforms_to(T, Intable),
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^

    # Method only requires Copyable - satisfied by stronger conformance constraint
    def method(self) where conforms_to(Self.T, Copyable):
        print("StrongerConformance.method")


def test_stronger_implies_weaker():
    # CopyableIntableType is both Copyable and Intable
    var s = StrongerConformance(CopyableIntableType(42))
    # CHECK: StrongerConformance.method
    s.method()


# ===========================================================================
# Diamond with Same Constraint on Both Paths (auto-propagated)
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

    def __init__(out self, var data: Self.T):
        self.data = data^


def test_diamond_same_constraint():
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
# Diamond with Reordered Compound Constraints (logical equivalence)
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

    def __init__(out self, var data: Self.T):
        self.data = data^


def test_diamond_reordered_constraint():
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
    def get_label(read self) -> Int:
        return 0


struct ViolatedTakesPrecedence[T: ImplicitlyDestructible & Movable](
    ImplicitlyDestructible,
    Movable,
    ViolatedPrecedenceTrait where conforms_to(T, Copyable),
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^

    # First where clause: unprovable (Intable is independent of Copyable).
    # Second where clause: contradicts the conformance (conformance requires
    # Copyable, but this requires NOT Copyable).
    # The method should be Violated (rejected), not Unprovable (error).
    def get_label(
        read self,
    ) -> Int where conforms_to(Self.T, Intable) where not conforms_to(
        Self.T, Copyable
    ):
        return 1


def needs_violated_precedence[T: ViolatedPrecedenceTrait](x: T):
    print("violated_precedence:", x.get_label())


def test_violated_precedence():
    var v = ViolatedTakesPrecedence(CopyableType(1))
    # CHECK: violated_precedence: 0
    needs_violated_precedence(v)


# ===========================================================================
# Conditional Movable transfer via conforms_to(T, Copyable)
# ===========================================================================
# The struct is conditionally Movable gated by conforms_to(T, Copyable).
# The transfer operator (^) should work when T satisfies the condition.


struct MovableViaCopyable[T: ImplicitlyDestructible & Movable](
    ImplicitlyDestructible,
    Movable where conforms_to(T, Copyable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


def test_move_via_copyable():
    var original = MovableViaCopyable(CopyableType(100))
    var moved = original^
    # CHECK: move_via_copyable: 100
    print("move_via_copyable:", moved.value.x)


# ===========================================================================
# Conditional Movable transfer via conforms_to(T, ImplicitlyCopyable)
# ===========================================================================
# Same as Test 12, but gated by conforms_to(T, ImplicitlyCopyable).
# ImplicitlyCopyable also subsumes Movable, so the transfer should work.


struct MovableViaImplCopyable[T: ImplicitlyDestructible & Movable](
    ImplicitlyDestructible,
    Movable where conforms_to(T, ImplicitlyCopyable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


def test_move_via_impl_copyable():
    var original = MovableViaImplCopyable(ImplCopyableType(200))
    var moved = original^
    # CHECK: move_via_impl_copyable: 200
    print("move_via_impl_copyable:", moved.value.x)


# ===========================================================================
# Both Movable and Copyable as conditional conformances
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

    def __init__(out self, var value: Self.T):
        self.value = value^

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Movable):
        self.value = rebind_var[Self.T](
            trait_downcast[Copyable](copy.value).copy()
        )


def test_both_conditional_move_copy():
    var original = BothConditionalMoveCopy(CopyableType(300))
    var copied = original.copy()
    # CHECK: both_cond_copy: 300
    print("both_cond_copy:", copied.value.x)

    var another = BothConditionalMoveCopy(CopyableType(400))
    var moved = another^
    # CHECK: both_cond_move: 400
    print("both_cond_move:", moved.value.x)


# ===========================================================================
# Synthesized copy ctor with conditional Copyable conformance
# ===========================================================================
# The compiler should synthesize a copy constructor that uses trait-downcast
# refinement for the T-typed field when the where-clause provides
# conforms_to(T, Copyable).  No manual copy init is written.


struct SynthCopyWrapper[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDestructible,
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


def test_synthesized_copy():
    var original = SynthCopyWrapper(CopyableType(42))
    var copied = original.copy()
    # CHECK: synth_copy: 42
    print("synth_copy:", copied.value.x)


# ===========================================================================
# Synthesized move ctor with T-typed field
# ===========================================================================
# The move init is synthesized for a struct with a T-typed field.  Because
# T: ImplicitlyDestructible & Movable, the field IS unconditionally movable,
# so the unconditional synthesis path handles it.  This verifies that the
# synthesized move init works correctly alongside conditional copy.


def test_synthesized_move():
    var original = SynthCopyWrapper(CopyableType(99))
    var moved = original^
    # CHECK: synth_move: 99
    print("synth_move:", moved.value.x)


# ===========================================================================
# Mixed fields — conditional + unconditional for copy
# ===========================================================================
# A struct with both an unconditionally copyable field (Int) and a
# conditionally copyable field (T).  The synthesized copy ctor handles
# each field via the appropriate path.


struct MixedFieldWrapper[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDestructible,
    Movable,
):
    var count: Int
    var value: Self.T

    def __init__(out self, count: Int, var value: Self.T):
        self.count = count
        self.value = value^


def test_mixed_field_copy():
    var original = MixedFieldWrapper(7, CopyableType(55))
    var copied = original.copy()
    # CHECK: mixed_copy: 7 55
    print("mixed_copy:", copied.count, copied.value.x)


# ===========================================================================
# Copy synthesis via conforms_to(T, ImplicitlyCopyable) subsumption
# ===========================================================================
# The struct is conditionally Copyable gated by conforms_to(T,
# ImplicitlyCopyable).  Because ImplicitlyCopyable subsumes Copyable, the
# compiler should synthesize a copy constructor that uses the conditional
# ImplicitlyCopyable constraint.  The move constructor is unconditional.


struct CopyableViaImplCopyable[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, ImplicitlyCopyable),
    ImplicitlyDestructible,
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


def test_copy_via_impl_copyable():
    var original = CopyableViaImplCopyable(ImplCopyableType(500))
    var copied = original.copy()
    # CHECK: copy_via_impl_copyable: 500
    print("copy_via_impl_copyable:", copied.value.x)
    var moved = original^
    # CHECK: move_with_impl_copyable: 500
    print("move_with_impl_copyable:", moved.value.x)


# ===========================================================================
# Copy + move with ImplicitlyCopyable type on Copyable constraint
# ===========================================================================
# SynthCopyWrapper requires conforms_to(T, Copyable).  ImplCopyableType
# conforms to ImplicitlyCopyable ⊃ Copyable, so the condition is met.
# Both the synthesized copy and unconditional move should work.


def test_synth_copy_with_impl_copyable_type():
    var original = SynthCopyWrapper(ImplCopyableType(600))
    var copied = original.copy()
    # CHECK: synth_copy_ic: 600
    print("synth_copy_ic:", copied.value.x)
    var moved = original^
    # CHECK: synth_move_ic: 600
    print("synth_move_ic:", moved.value.x)


# ===========================================================================
# Test: Conditional RegisterPassable Conformance
# ===========================================================================


struct ConditionalRP[T: Movable & ImplicitlyDestructible](
    ImplicitlyDestructible,
    Movable,
    RegisterPassable where conforms_to(T, RegisterPassable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


def needs_rp[T: RegisterPassable](x: T):
    print("RegisterPassable!")


def test_conditional_rp():
    # Int is RegisterPassable, so ConditionalRP[Int] should be RP.
    var x = ConditionalRP[Int](42)
    needs_rp(x)


# ===========================================================================
# conforms_to builtin evaluates where-clause constraints
# ===========================================================================
# The conforms_to builtin should return False when the conditional conformance's
# where-clause is not satisfied, even though the ConformanceOp exists in the
# struct's symbol table.


def test_conforms_to_evaluates_where_clause():
    # SimpleWrapper[CopyableType] should conform to Copyable
    # because CopyableType is Copyable.
    # CHECK: conforms_to_satisfied: True
    print(
        "conforms_to_satisfied:",
        conforms_to(SimpleWrapper[CopyableType], Copyable),
    )

    # SimpleWrapper[MovableOnlyType] should NOT conform to Copyable
    # because MovableOnlyType is NOT Copyable.
    # CHECK: conforms_to_unsatisfied: False
    print(
        "conforms_to_unsatisfied:",
        conforms_to(SimpleWrapper[MovableOnlyType], Copyable),
    )

    # MultipleConditional[CopyableIntableType] should conform to both
    # Copyable and Intable.
    # CHECK: multi_conforms_copyable: True
    print(
        "multi_conforms_copyable:",
        conforms_to(MultipleConditional[CopyableIntableType], Copyable),
    )
    # CHECK: multi_conforms_intable: True
    print(
        "multi_conforms_intable:",
        conforms_to(MultipleConditional[CopyableIntableType], Intable),
    )

    # MultipleConditional[CopyableType] should conform to Copyable
    # but NOT to Intable.
    # CHECK: copyable_only_copyable: True
    print(
        "copyable_only_copyable:",
        conforms_to(MultipleConditional[CopyableType], Copyable),
    )
    # CHECK: copyable_only_intable: False
    print(
        "copyable_only_intable:",
        conforms_to(MultipleConditional[CopyableType], Intable),
    )

    # Unconditional conformance should always return True.
    # CHECK: unconditional_movable: True
    print(
        "unconditional_movable:",
        conforms_to(SimpleWrapper[MovableOnlyType], Movable),
    )


# ===========================================================================
# conforms_to with symbolic type parameter (elaborator path)
# ===========================================================================
# When conforms_to(Wrapper[T], Trait) is called with a symbolic T, the
# expression remains unevaluated at parse time and is evaluated by the
# elaborator when T is instantiated with a concrete type. This tests that
# the elaborator's evaluateConformsToWithConstraints correctly returns
# true/false based on the conditional conformance constraint.


def check_copyable_symbolic[T: ImplicitlyDestructible & Movable]() -> Bool:
    return conforms_to(SimpleWrapper[T], Copyable)


def test_symbolic_conforms_to():
    # CHECK: symbolic_copyable_int: True
    print("symbolic_copyable_int:", check_copyable_symbolic[CopyableType]())
    # CHECK: symbolic_copyable_movable: False
    print(
        "symbolic_copyable_movable:",
        check_copyable_symbolic[MovableOnlyType](),
    )


# ===========================================================================
# Trait bound on T proves conditional conformance at call site
# ===========================================================================
# When T has a Copyable bound, SimpleWrapper[T] is provably Copyable,
# and can be passed to functions requiring Copyable.


def guarded_copyable_call[
    T: Copyable & ImplicitlyDestructible
](x: SimpleWrapper[T]):
    needs_copyable(x)


def test_guarded_conditional_call():
    var w = SimpleWrapper(CopyableType(42))
    # CHECK: guarded_call: ok
    guarded_copyable_call(w)
    print("guarded_call: ok")


# ===========================================================================
# comptime if guard enables conditional conformance call
# ===========================================================================
# A comptime if conforms_to(T, Trait) guard proves the constraint within its
# body, allowing calls that require the conditional conformance.


def conditional_copy_in_guard[
    T: ImplicitlyDestructible & Movable
](x: SimpleWrapper[T]):
    comptime if conforms_to(T, Copyable):
        # CHECK: comptime_guard_copy: ok
        needs_copyable(x)
        print("comptime_guard_copy: ok")


def test_comptime_if_guard():
    var w = SimpleWrapper(CopyableType(42))
    conditional_copy_in_guard(w)


# ===========================================================================
# where clause proves variadic conditional conformance
# ===========================================================================
# A where clause asserting AllWritable[*types] enables calling repr() on
# Tuple[*types], because the assumption is wired into doesNominalTypeConformTo
# via constraintImplies.

from std.reflection.traits import AllWritable


def repr_with_where[
    *types: Movable & Writable
](t: Tuple[*types]) -> String where AllWritable[*types]:
    return repr(t)


def test_where_clause_proves_variadic():
    var t = (1, "hello")
    # CHECK: where_variadic: Tuple[Int, String](Int(1), 'hello')
    print("where_variadic:", repr_with_where(t))


# ===========================================================================
# conforms_to with concrete types (VerifyParameters/LIT path)
# ===========================================================================
# conforms_to used as a value expression with fully concrete types is
# evaluated by the VerifyParameters pass through LITSymTabEvaluationContext.
# Without the fix, the evaluateWithContext/simplify() fallback would return
# true by only checking ConformanceOp existence, ignoring the constraint.


def test_conforms_to_value_expression():
    # Positive: nested wrapper where inner satisfies the constraint.
    # CHECK: value_nested_pos: True
    print(
        "value_nested_pos:",
        conforms_to(SimpleWrapper[SimpleWrapper[CopyableType]], Copyable),
    )

    # Negative: wrapper around a non-Copyable type must return False.
    # CHECK: value_neg: False
    print(
        "value_neg:",
        conforms_to(SimpleWrapper[MovableOnlyType], Copyable),
    )

    # Negative: nested wrapper where inner does NOT satisfy the constraint.
    # CHECK: value_nested_neg: False
    print(
        "value_nested_neg:",
        conforms_to(SimpleWrapper[SimpleWrapper[MovableOnlyType]], Copyable),
    )


# ===========================================================================
# Scope-dependent metatype upcast with conditional conformance
# ===========================================================================
# canMetaTypeUpCastTo (used by canImplicitlyConvertToType and ParamMatcher)
# needs the caller scope to prove that a conditionally-conforming type can
# be upcast to a trait metatype.


def accept_copyable_metatype[T: Copyable]():
    pass


def upcast_with_scope[T: Copyable & ImplicitlyDestructible]():
    # SimpleWrapper[T] is Copyable because T: Copyable is in scope.
    # This exercises canMetaTypeUpCastTo receiving the scope.
    accept_copyable_metatype[SimpleWrapper[T]]()


def test_metatype_upcast_with_scope():
    upcast_with_scope[CopyableType]()
    # CHECK: metatype_upcast_scope: ok
    print("metatype_upcast_scope: ok")


# ===========================================================================
# Function type conversion with where-clause scope
# ===========================================================================
# canConvertFunctionTypes checks argument conformance via checkConformance
# with the caller's declScope. This tests that a higher-order function
# accepting `def(Copyable)` can receive a function taking a conditionally-
# conforming type when the scope proves the conformance.


def apply_copyable_fn[
    T: Copyable & ImplicitlyDestructible & Movable
](f: def(SimpleWrapper[T]) thin -> None, x: SimpleWrapper[T]):
    f(x)


def print_wrapper_value[
    T: Copyable & ImplicitlyDestructible & Movable
](w: SimpleWrapper[T],):
    # CHECK: fn_conversion_scope: ok
    print("fn_conversion_scope: ok")


def test_fn_conversion_with_scope():
    var w = SimpleWrapper(CopyableType(42))
    apply_copyable_fn(print_wrapper_value[CopyableType], w)


# ===========================================================================
# Trait-to-trait upcast with scope proving conditional conformance
# ===========================================================================
# emitImplicitConversionToType uses getDeclScope() to prove that a
# conditionally-conforming type value can upcast to a base trait type.
# Here we upcast from Copyable (which refines Movable) to Movable,
# verifying the scope-aware path through checkConformance.


def accept_movable[T: Movable](x: T):
    pass


def upcast_conditional_to_base[
    T: Copyable & ImplicitlyDestructible & Movable
](x: SimpleWrapper[T]):
    # SimpleWrapper[T] is Copyable (proven by T: Copyable in scope).
    # Copyable refines Movable, so this upcast should work.
    accept_movable(x)


def test_upcast_conditional_to_base():
    var w = SimpleWrapper(CopyableType(42))
    upcast_conditional_to_base(w)
    # CHECK: upcast_conditional_base: ok
    print("upcast_conditional_base: ok")


# ===========================================================================
# TRP + conditional conformance with custom __eq__ and default __ne__
# ===========================================================================
# TrivialRegisterPassable types use a calling-convention thunk for witness
# table entries (self by-value vs by-reference). The thunk must propagate
# the callee's where-clause constraints so the indirect call can be verified.
# Tests custom __eq__ (where clause + thunk) and default __ne__ (calls __eq__).


@fieldwise_init
struct TRPEquatable[T: AnyType](
    Equatable where conforms_to(T, Equatable),
    TrivialRegisterPassable,
):
    var value: Int

    def __eq__(self, other: Self) -> Bool where conforms_to(Self.T, Equatable):
        return self.value == other.value


def needs_equatable[T: Equatable](a: T, b: T) -> Bool:
    return (a == b) and not (a != b)


def test_trp_conditional_equatable():
    var a = TRPEquatable[Int](42)
    var b = TRPEquatable[Int](42)
    var c = TRPEquatable[Int](99)
    # CHECK: trp_eq_same: True
    print("trp_eq_same:", needs_equatable(a, b))
    # CHECK: trp_eq_diff: False
    print("trp_eq_diff:", needs_equatable(a, c))


# ===========================================================================
# TRP + multiple conditional conformances with default __eq__
# ===========================================================================
# Tests multiple conditional conformances on one TRP struct. Equatable uses
# the default reflection-based __eq__ (no override). Also tests a custom
# trait with a where-clause method.


trait CustomTRPTrait:
    def get_value(self) -> Int:
        ...


@fieldwise_init
struct CustomTRPImpl(CustomTRPTrait, TrivialRegisterPassable):
    var x: Int

    def get_value(self) -> Int:
        return self.x


@fieldwise_init
struct TRPMulti[T: AnyType](
    CustomTRPTrait where conforms_to(T, CustomTRPTrait),
    Equatable where conforms_to(T, Equatable),
    TrivialRegisterPassable,
):
    var value: Int

    def get_value(self) -> Int where conforms_to(Self.T, CustomTRPTrait):
        return self.value


def needs_custom_trp[T: CustomTRPTrait](x: T) -> Int:
    return x.get_value()


def test_trp_multiple_conditional():
    # Equatable path (Int conforms to Equatable) — uses default __eq__
    var a = TRPMulti[Int](77)
    var b = TRPMulti[Int](77)
    var c = TRPMulti[Int](99)
    # CHECK: trp_multi_eq: True
    print("trp_multi_eq:", needs_equatable(a, b))
    # CHECK: trp_multi_ne: False
    print("trp_multi_ne:", needs_equatable(a, c))
    # CustomTRPTrait path (CustomTRPImpl conforms to CustomTRPTrait)
    var d = TRPMulti[CustomTRPImpl](88)
    # CHECK: trp_multi_custom: 88
    print("trp_multi_custom:", needs_custom_trp(d))


# ===========================================================================
# Implicit copy of conditionally ImplicitlyCopyable type in generic
# ===========================================================================


@fieldwise_init
struct CondWrapper[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable),
    ImplicitlyDestructible,
    Movable,
):
    var value: Self.T


def implicit_copy_in_generic[
    T: ImplicitlyCopyable & ImplicitlyDestructible
](x: CondWrapper[T]) -> CondWrapper[T]:
    return x


def test_implicit_copy_in_generic():
    var w = CondWrapper(42)
    var copy = implicit_copy_in_generic(w)
    # CHECK: implicit_copy_generic: 42
    print("implicit_copy_generic:", copy.value)


# ===========================================================================
# Move of conditionally Movable type in generic
# ===========================================================================


@fieldwise_init
struct CondMovable[T: ImplicitlyDestructible & Movable](
    ImplicitlyDestructible,
    Movable where conforms_to(T, Copyable),
):
    var value: Self.T


def move_in_generic[
    T: Copyable & ImplicitlyDestructible
](var x: CondMovable[T]) -> CondMovable[T]:
    return x^


def test_move_in_generic():
    var w = CondMovable(42)
    var moved = move_in_generic(w^)
    # CHECK: move_in_generic: 42
    print("move_in_generic:", moved.value)


# ===========================================================================
# Explicit copy of conditionally Copyable type in generic
# ===========================================================================


def explicit_copy_in_generic[
    T: Copyable & ImplicitlyDestructible
](x: CondWrapper[T]) -> CondWrapper[T]:
    return x.copy()


def test_explicit_copy_in_generic():
    var w = CondWrapper(42)
    var copy = explicit_copy_in_generic(w)
    # CHECK: explicit_copy_generic: 42
    print("explicit_copy_generic:", copy.value)


# ===========================================================================
# Copy/move synthesis through comptime type aliases
# ===========================================================================
# Before the fix, synthesized copy/move ctors called isCopyable/isMovable
# without the fn's where-clause scope, so they couldn't prove that an
# alias-resolved field type like `SimpleWrapper[T]` is Copyable even when the
# fn's where-clause asserts `conforms_to(T, Copyable)`.  The structural
# fallback (fieldConditionallyConformsToBuiltin) also couldn't match because
# the field param is `SimpleWrapper[T]`, not bare `T`.
#
# This struct is the minimal reproduction: the field type is a comptime alias
# to SimpleWrapper[T] (defined above), which is conditionally Copyable.


@fieldwise_init
struct AliasField[T: ImplicitlyDestructible & Movable](
    Copyable where conforms_to(T, Copyable),
    ImplicitlyDestructible,
    Movable,
):
    comptime Wrapped = SimpleWrapper[Self.T]
    var field: Self.Wrapped


def test_comptime_alias_conditional_conformance():
    # CHECK: alias_pos: True
    print("alias_pos:", conforms_to(AliasField[CopyableType], Copyable))
    # CHECK: alias_neg: False
    print("alias_neg:", conforms_to(AliasField[MovableOnlyType], Copyable))

    # Synthesized copy through alias-resolved field.
    var original = AliasField(field=SimpleWrapper(CopyableType(42)))
    var copied = original.copy()
    # CHECK: alias_copy: 42
    print("alias_copy:", copied.field.value.x)

    # Synthesized move through alias-resolved field.
    var to_move = AliasField(field=SimpleWrapper(CopyableType(99)))
    var moved = to_move^
    # CHECK: alias_move: 99
    print("alias_move:", moved.field.value.x)


# ===========================================================================
# Default Equatable on conditionally-conforming struct (sub-element walking)
# ===========================================================================
# Regression test for SymbolRefAttr sub-element walking in
# TypeConformsToTraitAttr. The default Equatable.__eq__ uses reflection to
# iterate fields and calls conforms_to(FieldType, Equatable) for each one.
# This path exercises constraintImplies → getCanonicalAttr on decomposed
# multi-trait conforms_to attrs. Without proper sub-element walking,
# getCanonicalAttr cannot descend into the SymbolRefAttr list, producing
# non-canonical forms that cause constraintImplies to reject valid
# subsumption (e.g. conforms_to(T, Equatable) should imply
# conforms_to(T, ImplicitlyDestructible) via ancestor expansion).


@fieldwise_init
struct DefaultEqWrapper[T: ImplicitlyDestructible & Movable](
    Equatable where conforms_to(T, Equatable),
    ImplicitlyDestructible,
    Movable,
):
    var value: Self.T


def test_default_equatable_conditional():
    var a = DefaultEqWrapper[Int](42)
    var b = DefaultEqWrapper[Int](42)
    var c = DefaultEqWrapper[Int](99)
    # CHECK: default_eq_same: True
    print("default_eq_same:", a == b)
    # CHECK: default_eq_diff: False
    print("default_eq_diff:", a == c)
    # CHECK: default_ne_same: False
    print("default_ne_same:", a != b)
    # CHECK: default_ne_diff: True
    print("default_ne_diff:", a != c)


# ===========================================================================
# Comptime trait alias on type parameter with conditional conformance
# ===========================================================================
# When T is constrained to a comptime trait alias (e.g. `comptime MyAlias =
# A & B`), the alias introduces SugarAttr/AnyTraitType/ParamType wrapping
# around the type parameter's type in the MLIR IR. The
# `simplifyConformsToAgainstTypeValue` function must unwrap these layers to
# extract the underlying TraitType and verify that it subsumes the required
# trait from the conditional conformance constraint.


comptime CopyableAlias = Copyable & ImplicitlyDestructible


def test_alias_param_conditional[T: CopyableAlias](x: SimpleWrapper[T]):
    # SimpleWrapper[T] is conditionally Copyable where conforms_to(T, Copyable).
    # T: CopyableAlias = Copyable & ImplicitlyDestructible.
    # The compiler must look through the alias sugar to see that T: Copyable.
    needs_copyable(x)
    print("alias_param_conditional: ok")


def test_alias_param_conforms_to():
    # Also verify conforms_to evaluates correctly with alias-constrained params.
    print(
        "alias_conforms_pos:",
        conforms_to(SimpleWrapper[CopyableType], Copyable),
    )
    print(
        "alias_conforms_neg:",
        conforms_to(SimpleWrapper[MovableOnlyType], Copyable),
    )


def main():
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
    test_synthesized_copy()
    test_synthesized_move()
    test_mixed_field_copy()
    test_copy_via_impl_copyable()
    test_synth_copy_with_impl_copyable_type()
    test_conforms_to_evaluates_where_clause()
    test_symbolic_conforms_to()
    test_guarded_conditional_call()
    test_comptime_if_guard()
    test_where_clause_proves_variadic()
    test_conforms_to_value_expression()
    test_metatype_upcast_with_scope()
    test_fn_conversion_with_scope()
    test_upcast_conditional_to_base()
    test_trp_conditional_equatable()
    test_trp_multiple_conditional()
    test_implicit_copy_in_generic()
    test_move_in_generic()
    test_explicit_copy_in_generic()
    test_comptime_alias_conditional_conformance()
    test_default_equatable_conditional()
    # CHECK: RegisterPassable!
    test_conditional_rp()
    # CHECK: alias_param_conditional: ok
    test_alias_param_conditional(SimpleWrapper[CopyableType](CopyableType(42)))
    # CHECK: alias_conforms_pos: True
    # CHECK: alias_conforms_neg: False
    test_alias_param_conforms_to()
