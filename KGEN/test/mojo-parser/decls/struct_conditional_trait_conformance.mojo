# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt --kgen-print-inline-type-values | FileCheck %s

# Test file for conditional trait conformance parsing.
# This tests that `where` clauses in struct trait inheritance lists
# are correctly parsed and placed in the canonicalTrait attribute.

# Type aliases with constraints are generated for constrained trait compositions.
# Check for constrained trait type aliases containing the expected constraints:
# CHECK-DAG: @std::@builtin::@stubs::@Copyable where #kgen.constraint<conforms_to(:!Movable T, ["std::builtin::stubs::Copyable"])
# CHECK-DAG: @std::@builtin::@stubs::@Intable where #kgen.constraint<conforms_to(:!Movable T, ["std::builtin::stubs::Intable"])


# ===========================================================================
# Test 1: Unconditional conformance - struct is always Movable
# ===========================================================================
# The struct should NOT have any constraint in its trait type.
# CHECK: lit.struct.decl @UnconditionalMovable<T: !Movable>
# CHECK-NOT: where #kgen.constraint
# CHECK-SAME: attributes
struct UnconditionalMovable[T: Movable](Movable):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn __moveinit__(out self, deinit existing: Self):
        self.value = existing.value^


# ===========================================================================
# Test 2: Single conditional conformance - Copyable only when T is Copyable
# ===========================================================================
# Verify the ConformanceOp has the constraint attached:
# CHECK: lit.struct.decl @ConditionalCopyable<T: !Movable>
# CHECK: kgen.conformance @"std::builtin::stubs::Copyable"
# CHECK: } where #kgen.constraint<conforms_to(:!Movable T, ["std::builtin::stubs::Copyable"])
struct ConditionalCopyable[T: Movable](Copyable where conforms_to(T, Copyable), Movable):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

    fn __moveinit__(out self, deinit existing: Self):
        self.value = existing.value^

    fn __copyinit__(out self, existing: Self, /) where conforms_to(Self.T, Copyable):
        self.value = rebind_var[Self.T](trait_downcast[Copyable](existing.value).copy())


# ===========================================================================
# Test 3: Multiple conditional conformances - Copyable and Intable
# ===========================================================================
# Verify ConformanceOps have constraints for both Copyable and Intable:
# CHECK: lit.struct.decl @MultipleConditionalConformances<T: !Movable>
# CHECK: kgen.conformance @"std::builtin::stubs::Copyable"
# CHECK: } where #kgen.constraint<conforms_to(:!Movable T, ["std::builtin::stubs::Copyable"])
# CHECK: kgen.conformance @"std::builtin::stubs::Intable"
# CHECK: } where #kgen.constraint<conforms_to(:!Movable T, ["std::builtin::stubs::Intable"])
struct MultipleConditionalConformances[T: Movable](
    Copyable where conforms_to(T, Copyable),
    Intable where conforms_to(T, Intable),
    Movable,
):
    var inner: Self.T

    fn __init__(out self, var inner: Self.T):
        self.inner = inner^

    fn __moveinit__(out self, deinit existing: Self):
        self.inner = existing.inner^

    fn __copyinit__(out self, existing: Self, /) where conforms_to(Self.T, Copyable):
        self.inner = rebind_var[Self.T](trait_downcast[Copyable](existing.inner).copy())

    fn __int__(self) -> Int where conforms_to(Self.T, Intable):
        return 0
