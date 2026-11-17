# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# C reference: c_abi_test_nested_structs.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_nested
# RUN: %t.dir/test_nested | FileCheck %s %if x86_64-linux %{--check-prefixes=CHECK,X86%}

from std.collections import Array
from std.ffi import external_call
from std.sys import CompilationTarget


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


# ============================================================================
# Vector and nested struct: VecInner { SIMD[int32, 2], Inner } (16 bytes)
# ABI (x86-64): SSEPair(f64, f64) — a 64-bit vector is SSE.
# ============================================================================
@fieldwise_init
struct VecInner(TrivialRegisterPassable):
    var v: SIMD[DType.int32, 2]
    var inner: Inner


def test_vec_inner_16byte():
    var s = VecInner(SIMD[DType.int32, 2](1, 2), Inner(3.0, 4.0))
    var r = external_call["c_func_vec_inner_16byte", VecInner](s)
    print("vec_inner_16byte:", r.v[0], r.v[1], r.inner.x, r.inner.y)


# CHECK: vec_inner_16byte: 2 3 4.0 5.0


# ============================================================================
# Array member: FloatArray3 { Array[Float32, 3] } (12 bytes)
# ABI (x86-64): SSEPair(f64, f32) — the array is classified per element.
# ============================================================================
@fieldwise_init
struct FloatArray3(Copyable):
    var v: Array[Float32, 3]


# `external_call` cannot return a struct that is not `RegisterPassable`.
@extern("c_func_float_array_12byte")
def c_func_float_array_12byte(s: FloatArray3) abi("C") -> FloatArray3:
    ...


def test_float_array_12byte():
    var r = c_func_float_array_12byte(FloatArray3([1.0, 2.0, 3.0]))
    print("float_array_12byte:", r.v[0], r.v[1], r.v[2])


# X86: float_array_12byte: 2.0 3.0 4.0


def main():
    test_nested_float_12byte()
    test_vec_inner_16byte()
    # Not supported on arm64 yet.
    comptime if CompilationTarget.is_x86():
        test_float_array_12byte()
