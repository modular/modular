# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Tests for interior origins.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s


struct MyList[T: Movable]:
    var data: UnsafePointer[Self.T, UntrackedOrigin[mut=True]]

    def __init__(out self):
        self.data = UnsafePointer[Self.T, UntrackedOrigin[mut=True]].unsafe_dangling()

    def __del__(deinit self):
        pass # Explicit del so it isn't considered trivial and elided.

    def mutate(mut self):
        pass

    def __getitem__(
        ref self, idx: Int
    ) -> ref[self.data.unsafe_origin_owned_rebase[origin_of(self), "element"](idx)[]] Self.T:
        return self.data.unsafe_origin_owned_rebase[origin_of(self), "element"](idx)[]


# CHECK-LABEL: lit.fn @"test0
def test0():
    # CHECK: lit.call {{.*}}MyList::@"__init__
    var list = MyList[Int]()

    # CHECK: lit.call {{.*}}MyList::@"__getitem__
    ref elt = list[4]

    # CHECK: lit.call {{.*}}Int::@"__iadd__
    elt += 4
    # CHECK: lit.call {{.*}}MyList::@"__del__
