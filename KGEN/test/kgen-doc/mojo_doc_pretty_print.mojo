# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-doc %s | FileCheck %s
from std.collections import List


# CHECK: "name": "foo"
# CHECK: "type": "List"
# CHECK: "signature": "foo(x: List[Int])"
def foo(x: List[Int]):
    pass


# CHECK: "name": "bar"
# CHECK: "type": "List"
# CHECK: "signature": "bar[x: List[Int]]()"
def bar[x: List[Int]]():
    pass


# CHECK: "name": "baz"
# CHECK: "returns": {
# CHECK:    "type": "List"
# CHECK: }
# CHECK: "signature": "baz() -> List[Int]"
def baz() -> List[Int]:
    return List[Int]()


# CHECK: "name": "higher_order"
# CHECK: "type": "def(List[Int]) -> Int"
# CHECK: "signature": "higher_order(f: def(List[Int]) -> Int) -> def(List[Int]) -> Int"
def higher_order(
    f: def(List[Int]) -> Int,
) -> def(List[Int]) -> Int:
    return f
