# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo doc %s | FileCheck %s
from std.collections import List


# CHECK: "name": "foo"
# CHECK: "type": "List"
# CHECK: "signature": "foo(x: List[Int])"
fn foo(x: List[Int]):
    pass


# CHECK: "name": "bar"
# CHECK: "type": "List"
# CHECK: "signature": "bar[x: List[Int]]()"
fn bar[x: List[Int]]():
    pass


# CHECK: "name": "baz"
# CHECK: "returns": {
# CHECK:    "type": "List"
# CHECK: }
# CHECK: "signature": "baz() -> List[Int]"
fn baz() -> List[Int]:
    return List[Int]()


# CHECK: "name": "higher_order"
# CHECK: "type": "fn(List[Int]) -> Int"
# CHECK: "signature": "higher_order(f: fn(List[Int]) -> Int) -> fn(List[Int]) -> Int"
fn higher_order(
    f: fn (std.collections.list.List[Int]) -> Int,
) -> fn (std.collections.list.List[Int]) -> Int:
    return f
