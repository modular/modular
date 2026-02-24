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

    fn __moveinit__(out self, deinit take: Self):
        self.x = take.x

    fn __copyinit__(out self, copy: Self, /):
        self.x = copy.x


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

    fn __moveinit__(out self, deinit take: Self):
        self.value = take.value^

    fn __copyinit__(
        out self, copy: Self, /
    ) where conforms_to(Self.T, Copyable):
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

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^

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
# Main
# ===========================================================================


fn main():
    test_simple_conditional()
    test_nested_wrappers()
    test_function_type_conversion()
    # TODO(#77058): Re-enable after lazy conformance is added.
    # test_closure_with_conditional_return()
    # test_nested_conditional_closure()
    test_violated_precedence()
