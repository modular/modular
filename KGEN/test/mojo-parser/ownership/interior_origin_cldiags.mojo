# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Tests for interior origins.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics


# ===----------------------------------------------------------------------=== #
# Interior Origin Handling
# ===----------------------------------------------------------------------=== #

struct MyListInterior[T: AnyType]:
    var data: UnsafePointer[Self.T, UntrackedOrigin[mut=True]]

    def __init__(out self):
        self.data = UnsafePointer[Self.T, UntrackedOrigin[mut=True]].unsafe_dangling()

    def __del__(deinit self):
        pass # Explicit del so it isn't considered trivial and elided.

    def mutate(mut self):
        pass

    def __getitem__(
        ref self
    ) -> ref[self.data.get_ref_with_unsafe_interior_origin["element"](self)] Self.T:
        return self.data.get_ref_with_unsafe_interior_origin["element"](self)

def test_invalidate_base():
    # expected-note @+1 {{'list' declared here}}
    var list = MyListInterior[Int]()

    ref elt_ref1 = list[]
    elt_ref1 += 4
    list^.__del__()

    # Deleting list obviously invalidates it.
    # expected-error @+1 {{use of uninitialized value 'list'}}
    elt_ref1 += 4

def test_invalidate_interior():
    var list = MyListInterior[Int]()
    ref elt_ref2 = list[]
    elt_ref2 += 4
    list.mutate()   # expected-note {{origin was invalidated here}}
    # expected-error @+1 {{use of invalidated interior reference 'list["element"]'}}
    elt_ref2 += 4

# simple control flow test.
def test_if(cond: Bool):
    var list = MyListInterior[Int]()
    ref elt_ref2 = list[]
    elt_ref2 += 4
    if cond:
        list.mutate()   # expected-note {{origin was invalidated here}}
    # expected-error @+1 {{use of invalidated interior reference 'list["element"]'}}
    elt_ref2 += 4

struct TwoIntLists:
   var first: MyListInterior[Int]
   var second: MyListInterior[Int]

   def __init__(out self):
      self.first = MyListInterior[Int]()
      self.second = MyListInterior[Int]()

# Test that we can handle nested field sensitivity correctly.
def test_field_sensitive_nested_invalidation():
    var list_of_two_intlists = MyListInterior[TwoIntLists]()
    ref first_list = list_of_two_intlists[].first
    ref second_list = list_of_two_intlists[].second

    ref first_list_elt = first_list[]
    ref second_list_elt = second_list[]

    # Mutating the elements of either list is fine, and shouldn't cause a
    # problem for anything.
    first_list_elt += 4
    second_list_elt += 4

    # Mutating the first list shouldn't invalidate the second list because of
    # nested field sensitivity.
    first_list.mutate()   # expected-note {{origin was invalidated here}}
    second_list_elt += 4

    # However, it should invalidate the first list.
    # expected-error @+1 {{use of invalidated interior reference 'list_of_two_intlists["element"].first["element"]'}}
    first_list_elt += 4
