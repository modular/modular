# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes | FileCheck %s


trait Iterator:
    alias Element: AnyType


struct I(Iterator):
    alias Element = Int


struct _MapIterator[
    InnerIteratorType: Iterator, //,
    Function: fn (InnerIteratorType.Element) -> Int,
]():
    var _inner: InnerIteratorType

    fn __init__(out self):
        while True:
            pass


fn f(x: Int) -> Int:
    return 1


fn map[
    func: fn (Int) -> Int,
](ref iterable: I) -> _MapIterator[InnerIteratorType=I, Function=func]:
    return {}


# CHECK-LABEL: lit.fn @"bork(
fn bork(l: I):
    var l2 = map[f](l)
    # This shouldn't cause a crash, it should successfully destruct it.
    # CHECK: lit.call @moco_2373::@_MapIterator::@"__del__
