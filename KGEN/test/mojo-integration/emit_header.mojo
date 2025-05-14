# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: kgen %s -emit-header | FileCheck %s


@export("bar", ABI="C")
# CHECK: extern float bar();
fn foo() -> Float32:
    # OK to alias, not proper main
    return 0.0


@export(ABI="C")
# CHECK: extern float call_me();
fn call_me() -> Float32:
    return 1.0


@fieldwise_init
@register_passable("trivial")
struct RegIntPair:
    var first: Int
    var second: Int


# CHECK: extern ssize_t first_reg(ssize_t, ssize_t);
@export(ABI="C")
fn first_reg(pair: RegIntPair) -> Int:
    return pair.first


# CHECK: extern void make_reg_pair(ssize_t, ssize_t, ssize_t *, ssize_t *);
@export(ABI="C")
fn make_reg_pair(first: Int, second: Int) -> RegIntPair:
    return RegIntPair(first, second)


# This is a memory only type.
struct MemIntPair:
    var first: Int
    var second: Int

    fn __init__(out self, first: Int, second: Int):
        self.first = first
        self.second = second


# CHECK: extern ssize_t first_mem(void *);
@export(ABI="C")
fn first_mem(pair: MemIntPair) -> Int:
    return pair.first


# CHECK: extern void make_mem_pair(ssize_t, ssize_t, void *);
@export(ABI="C")
fn make_mem_pair(first: Int, second: Int) -> MemIntPair:
    return MemIntPair(first, second)


# CHECK: extern int32_t main(int32_t, void *);
fn main():
    _ = call_me()
