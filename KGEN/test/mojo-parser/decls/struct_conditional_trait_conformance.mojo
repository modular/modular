# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN:  %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt --kgen-print-inline-type-values | FileCheck %s

# Test file for conditional trait conformance parsing.
# This tests that `where` clauses in struct trait inheritance lists
# are correctly parsed and placed in the canonicalTrait attribute.

# Type aliases with constraints are generated for constrained trait compositions.
# Check for constrained trait type aliases containing the expected constraints:
# CHECK-DAG: @Copyable where #kgen.constraint<{{.*}}conforms_to(:{{.*}} T, [{{.*}}@AnyType, {{.*}}@Copyable, {{.*}}@Movable])
# CHECK-DAG: @Intable where #kgen.constraint<{{.*}}conforms_to(:{{.*}} T, [{{.*}}@AnyType, {{.*}}@ImplicitlyDeletable, {{.*}}@Intable])


# ===========================================================================
# Unconditional conformance - struct is always Movable
# ===========================================================================
# The struct should NOT have any constraint in its trait type.
# CHECK: lit.struct.decl @UnconditionalMovable<T: !Movable_ImplicitlyDeletable>
# CHECK-NOT: where #kgen.constraint
# CHECK-SAME: attributes
struct UnconditionalMovable[T: Movable & ImplicitlyDeletable](Movable):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


# ===========================================================================
# Single conditional conformance - Copyable only when T is Copyable
# ===========================================================================
# Verify the ConformanceOp has the constraint attached:
# CHECK: lit.struct.decl @ConditionalCopyable<T: !Movable_ImplicitlyDeletable>
# CHECK: kgen.conformance @"std::builtin::stubs::Copyable"
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!Movable_ImplicitlyDeletable T, [{{.*}}@AnyType, {{.*}}@Copyable, {{.*}}@Movable])
struct ConditionalCopyable[T: Movable & ImplicitlyDeletable](
    Copyable where conforms_to(T, Copyable), Movable
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def __init__(out self, *, deinit move: Self):
        self.value = move.value^

    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.value = rebind_var[Self.T](trait_downcast[Copyable](copy.value).copy())


# ===========================================================================
# Multiple conditional conformances - Copyable and Intable
# ===========================================================================
# Verify ConformanceOps have constraints for both Copyable and Intable:
# CHECK: lit.struct.decl @MultipleConditionalConformances<T: !Movable_ImplicitlyDeletable>
# CHECK: kgen.conformance @"std::builtin::stubs::Copyable"
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!Movable_ImplicitlyDeletable T, [{{.*}}@AnyType, {{.*}}@Copyable, {{.*}}@Movable])
# CHECK: kgen.conformance @"std::builtin::stubs::Intable"
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!Movable_ImplicitlyDeletable T, [{{.*}}@AnyType, {{.*}}@ImplicitlyDeletable, {{.*}}@Intable])
struct MultipleConditionalConformances[T: Movable & ImplicitlyDeletable](
    Copyable where conforms_to(T, Copyable),
    Intable where conforms_to(T, Intable),
    Movable,
):
    var inner: Self.T

    def __init__(out self, var inner: Self.T):
        self.inner = inner^

    def __init__(out self, *, deinit move: Self):
        self.inner = move.inner^

    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.inner = rebind_var[Self.T](trait_downcast[Copyable](copy.inner).copy())

    def __int__(self) -> Int where conforms_to(Self.T, Intable):
        return 0


# ===========================================================================
# Disproved candidate with provable alternative - selects provable
# ===========================================================================
# When a struct has both:
# - A method with `where not` that contradicts the conformance (disproved)
# - A method with matching constraints (provable)
# The provable candidate is correctly selected with no error.
#
# Just verify the struct declaration is generated (no compilation error):
# CHECK: lit.struct.decl @DisprovedWithProvableAlternative<T: !Movable_ImplicitlyDeletable>


trait WhereNotTestTrait:
    def where_not_method(self):
        ...


struct DisprovedWithProvableAlternative[T: Movable & ImplicitlyDeletable](
    WhereNotTestTrait where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    # This method is "disproved" - its constraint contradicts the conformance.
    # The conformance requires T: Copyable, but this method requires NOT that.
    def where_not_method(self) where not conforms_to(Self.T, Copyable):
        pass

    # This method is "provable" - its constraint matches the conformance.
    # It will be selected as the witness table entry.
    def where_not_method(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Witness selection with matching vs contradicting constraints
# ===========================================================================
# This test demonstrates that when selecting a witness for a trait method:
# - Overloads with provable constraints (matching conformance) are selected
# - Overloads with disproved constraints (contradicting conformance) are skipped
#
# The key is that the constraints must be on the SAME trait as the conformance.
# Using `where not conforms_to(T, Copyable)` works because it directly
# contradicts `conforms_to(T, Copyable)`.
#
# NOTE: Using UNRELATED traits like `where not conforms_to(T, Intable)` would
# NOT work - it would be "unprovable" and cause an error.
#
# CHECK: lit.struct.decl @WitnessSelectionWithWhereNot<T: !Movable_ImplicitlyDeletable>

trait Greeter:
    def greet(self): ...

struct WitnessSelectionWithWhereNot[T: Movable & ImplicitlyDeletable](
    Greeter where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    # Overload 1: Provable - constraint matches the conformance.
    # This is selected as the witness for the Greeter trait.
    def greet(self) where conforms_to(Self.T, Copyable):
        pass

    # Overload 2: Disproved - constraint directly contradicts the conformance.
    # The conformance guarantees T: Copyable, so `not conforms_to(T, Copyable)`
    # is definitively false. This overload is skipped.
    def greet(self) where not conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Multiple overloads with stronger conformance constraint
# ===========================================================================
# When the conformance has a compound constraint (A and B), we can use
# `where not B` to filter out an overload because the conformance implies B.
#
# Here: conformance is `conforms_to(T, Copyable) and conforms_to(T, Intable)`
# - Method with `where conforms_to(T, Intable)` → provable (conformance implies it)
# - Method with `where not conforms_to(T, Intable)` → disproved (conformance implies Intable)
#
# CHECK: lit.struct.decl @CompoundConformanceWithWhereNot<T: !Movable_ImplicitlyDeletable>

trait Formatter:
    def format(self): ...

struct CompoundConformanceWithWhereNot[T: Movable & ImplicitlyDeletable](
    Formatter where conforms_to(T, Copyable) and conforms_to(T, Intable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    # Overload 1: Provable - conformance implies both Copyable AND Intable,
    # so it certainly implies just Intable.
    def format(self) where conforms_to(Self.T, Intable):
        pass

    # Overload 2: Disproved - conformance implies Intable, so
    # `not conforms_to(T, Intable)` contradicts it.
    def format(self) where not conforms_to(Self.T, Intable):
        pass


# ===========================================================================
# Compound method constraint with contradicting part
# ===========================================================================
# When a method has `where not X and Y`, and the conformance implies X,
# the `not X` part contradicts the conformance, making the whole constraint
# disproved (since AND requires all parts to be true).
#
# This is the pattern the reviewer mentioned: users can add extra conditions
# with `and`, but as long as one part contradicts the conformance, the
# overload is filtered out.
#
# CHECK: lit.struct.decl @CompoundMethodConstraint<T: !Movable_ImplicitlyDeletable>

trait Processor:
    def process(self): ...

struct CompoundMethodConstraint[T: Movable & ImplicitlyDeletable](
    Processor where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    # Overload 1: Provable - constraint matches the conformance.
    # This is selected as the witness.
    def process(self) where conforms_to(Self.T, Copyable):
        pass

    # Overload 2: Disproved - the `not conforms_to(T, Copyable)` part
    # contradicts the conformance. Even though there's an extra
    # `and conforms_to(T, Intable)` condition, the contradiction on the
    # first part makes the whole AND false.
    def process(self)
        where not conforms_to(Self.T, Copyable) and conforms_to(Self.T, Intable):
        pass


# ===========================================================================
# Trait composition with conditional conformance
# ===========================================================================
# When a trait composition `A & B` has a conditional conformance, both A and B
# should get the same constraint. Verify both conformance ops have constraints.
#
# CHECK: lit.struct.decl @CompositionConditional<T: !Movable_ImplicitlyDeletable>
# CHECK: kgen.conformance @"{{.*}}CompTraitA"
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!Movable_ImplicitlyDeletable T, [{{.*}}@AnyType, {{.*}}@Copyable, {{.*}}@Movable])
# CHECK: kgen.conformance @"{{.*}}CompTraitB"
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!Movable_ImplicitlyDeletable T, [{{.*}}@AnyType, {{.*}}@Copyable, {{.*}}@Movable])

trait CompTraitA:
    def comp_a_method(self): ...

trait CompTraitB:
    def comp_b_method(self): ...

struct CompositionConditional[T: Movable & ImplicitlyDeletable](
    CompTraitA & CompTraitB where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def comp_a_method(self) where conforms_to(Self.T, Copyable):
        pass

    def comp_b_method(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Duplicate trait with same constraint (valid - no conflict)
# ===========================================================================
# Listing the same trait twice with the same constraint is redundant but valid.
#
# CHECK: lit.struct.decl @DuplicateSameConstraint<T: !Movable_ImplicitlyDeletable>
# CHECK: kgen.conformance @"{{.*}}DupSameTrait"
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!Movable_ImplicitlyDeletable T, [{{.*}}@AnyType, {{.*}}@Copyable, {{.*}}@Movable])

trait DupSameTrait:
    def dup_method(self): ...

struct DuplicateSameConstraint[T: Movable & ImplicitlyDeletable](
    DupSameTrait where conforms_to(T, Copyable),
    DupSameTrait where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def dup_method(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Composition + standalone with same constraint (valid - no conflict)
# ===========================================================================
# A & B where cond, A where cond — A appears twice but with the same
# constraint, which is valid.
#
# CHECK: lit.struct.decl @CompositionStandaloneSameConstraint<T: !Movable_ImplicitlyDeletable>
# CHECK: kgen.conformance @"{{.*}}CSTraitA"
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!Movable_ImplicitlyDeletable T, [{{.*}}@AnyType, {{.*}}@Copyable, {{.*}}@Movable])
# CHECK: kgen.conformance @"{{.*}}CSTraitB"
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!Movable_ImplicitlyDeletable T, [{{.*}}@AnyType, {{.*}}@Copyable, {{.*}}@Movable])

trait CSTraitA:
    def cs_a_method(self): ...

trait CSTraitB:
    def cs_b_method(self): ...

struct CompositionStandaloneSameConstraint[T: Movable & ImplicitlyDeletable](
    CSTraitA & CSTraitB where conforms_to(T, Copyable),
    CSTraitA where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def cs_a_method(self) where conforms_to(Self.T, Copyable):
        pass

    def cs_b_method(self) where conforms_to(Self.T, Copyable):
        pass

# ===========================================================================
# MOCO-3347
# ===========================================================================


struct MyOptional[T: Movable](
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable) and conforms_to(
        T, Copyable
    ),
    Movable,
):
    pass

# ===========================================================================
# Split conforms_to constraints imply composite ancestor constraint
# ===========================================================================
# conforms_to(T, A) AND conforms_to(T, B) should imply conforms_to(T, A & B)
# when checking that a derived trait's constraint implies its ancestor's.
# This should parse without error (no "does not imply" diagnostic).

trait SplitAncestor:
    pass

trait SplitDerived(SplitAncestor):
    pass

# CHECK-LABEL: lit.struct.decl @SplitImpliesComposite
struct SplitImpliesComposite[T: Movable & ImplicitlyDeletable](
    SplitAncestor where conforms_to(T, Copyable & Intable),
    SplitDerived where conforms_to(T, Copyable) and conforms_to(T, Intable),
    Movable,
):
    pass

# Other direction: composite implies split (subsumption).
# CHECK-LABEL: lit.struct.decl @CompositeImpliesSplit
struct CompositeImpliesSplit[T: Movable & ImplicitlyDeletable](
    SplitAncestor where conforms_to(T, Copyable) and conforms_to(T, Intable),
    SplitDerived where conforms_to(T, Copyable & Intable),
    Movable,
):
    pass


#CHECK-LABEL: lit.struct.decl @Node
struct Node[ElementType: ImplicitlyCopyable](Movable):
    var value: MyOptional[Self.ElementType]
    # CHECK-LABEL: lit.fn @"__init__
    def __init__(out self, value: MyOptional[Self.ElementType] = None):
        # `MyOptional[Self.ElementType]` is implicitly copyable.

        # CHECK: lit.memcpy %value, %0
        self.value = value
