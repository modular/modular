# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: kgen %s -emit=header | FileCheck %s


# CHECK: extern float bar();
@export("bar")
def foo() abi("C") -> Float32:
    # OK to alias, not proper main
    return 0.0


# CHECK: extern float call_me();
@export
def call_me() abi("C") -> Float32:
    return 1.0


@fieldwise_init
struct RegIntPair(TrivialRegisterPassable):
    var first: Int
    var second: Int


# CHECK: extern ssize_t first_reg(ssize_t, ssize_t);
@export
def first_reg(pair: RegIntPair) abi("C") -> Int:
    return pair.first


# This is a memory only type. The platform C ABI classifies it from its layout,
# so its declaration below matches RegIntPair's field for field.
struct MemIntPair:
    var first: Int
    var second: Int

    def __init__(out self, first: Int, second: Int):
        self.first = first
        self.second = second


# CHECK: extern ssize_t first_mem(ssize_t, ssize_t);
@export
def first_mem(pair: MemIntPair) abi("C") -> Int:
    return pair.first


# CHECK: extern int32_t main(int32_t, void *);
def main():
    _ = call_me()
