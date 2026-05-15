# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# C reference: c_abi_test_nested_structs.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_nested
# RUN: %t.dir/test_nested | FileCheck %s

from std.ffi import external_call


# ============================================================================
# Nested float struct: Outer { Inner {f32, f32}, f32 } (12 bytes)
# ABI (x86-64): SSEPair(f64, f32) — eightbyte 0 is two f32s → SSE,
#               eightbyte 1 is one f32 → SSE.
# ============================================================================
@fieldwise_init
struct Inner(TrivialRegisterPassable):
    var x: Float32
    var y: Float32


@fieldwise_init
struct Outer(TrivialRegisterPassable):
    var inner: Inner
    var z: Float32


def test_nested_float_12byte():
    var s = Outer(Inner(1.0, 2.0), 3.0)
    var result = external_call["c_func_nested_float_12byte", Outer](s)
    print("nested_float_12byte:", result.inner.x, result.inner.y, result.z)


# CHECK: nested_float_12byte: 2.0 3.0 4.0


def main():
    test_nested_float_12byte()
