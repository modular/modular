# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Tests for indirect origins.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s

struct MyList[T: Copyable & Movable]:
    var data: UnsafePointer[T]

    fn __init__(out self):
        self.data = UnsafePointer[T]()

    fn __del__(owned self):
        pass

    fn mutate(mut self): pass

    fn __getitem__(ref self, idx: Int) -> ref [self.data.get_unique_item_ref(idx)] T:
        return self.data.get_unique_item_ref(idx)

# CHECK-LABEL: lit.fn @"test0
fn test0():
  # CHECK: lit.call {{.*}}MyList::@"__init__
  var list = MyList[Int]()

  # CHECK: lit.call {{.*}}MyList::@"__getitem__
  var ptr = Pointer(to=list[4])
  # CHECK: lit.call {{.*}}MyList::@"__del__

  # FIXME: This is not extending the lifetime of MyList
  # CHECK: lit.call {{.*}}Int::@"__iadd__
  ptr[] += 4
