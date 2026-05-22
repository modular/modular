# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Regression test for MOCO-4025: trait-composition default method body
# resolution.
#
# The generic function refines `T: AnyType` with a local equality trait. This
# creates an `AnyType & LocalEquatable` trait composition for method lookup. The
# inherited default method body must not be resolved while building that
# composition; it should remain tied to the declaring trait/concrete witness.

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait LocalEquatable:
    def __eq__(self, other: Self) -> Bool:
        if trait_downcast[LocalEquatable](
            __struct_field_ref(0, self)
        ) != trait_downcast[LocalEquatable](__struct_field_ref(0, other)):
            return False
        return True

    def __ne__(self, other: Self) -> Bool:
        return not self == other


struct FieldValue(LocalEquatable):
    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True


struct LocalValue(LocalEquatable):
    var field: FieldValue

    def __init__(out self):
        self.field = FieldValue()


# CHECK-LABEL: lit.fn @"eq_on_refined
def eq_on_refined[
    T: AnyType
](x: T, y: T) -> Bool where conforms_to(T, LocalEquatable):
    return x == y


def test_eq_on_refined():
    var x = LocalValue()
    var y = LocalValue()
    _ = eq_on_refined(x, y)
