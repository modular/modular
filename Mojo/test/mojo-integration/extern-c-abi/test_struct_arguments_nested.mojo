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


# ============================================================================
# Nested float struct: Box { Vec3 {f32, f32, f32}, Vec3 {f32, f32, f32} }
# (24 bytes)
# ABI (ARM64): not an HFA. Two top-level fields, but six floats once flattened,
#              and the HFA limit of four counts flattened members. Passed by
#              pointer, returned through x8.
# ABI (x86-64): MEMORY class (>16 bytes).
# ============================================================================
@fieldwise_init
struct Vec3(TrivialRegisterPassable):
    var x: Float32
    var y: Float32
    var z: Float32


@fieldwise_init
struct Box(TrivialRegisterPassable):
    var min: Vec3
    var max: Vec3


def test_nested_float_24byte():
    var s = Box(Vec3(1.0, 2.0, 3.0), Vec3(4.0, 5.0, 6.0))
    var result = external_call["c_func_nested_float_24byte", Box](s)
    print(
        "nested_float_24byte:",
        result.min.x,
        result.min.y,
        result.min.z,
        result.max.x,
        result.max.y,
        result.max.z,
    )


# CHECK: nested_float_24byte: 2.0 3.0 4.0 5.0 6.0 7.0


def weigh_box(b: Box) abi("C") -> Float32:
    return (
        b.min.x
        + 10 * b.min.y
        + 100 * b.min.z
        + 1000 * b.max.x
        + 10000 * b.max.y
        + 100000 * b.max.z
    )


def test_nested_float_24byte_callback():
    # C builds the Box and passes it to an abi("C") Mojo function.
    var result = external_call["c_call_nested_float_24byte", Float32](weigh_box)
    print("nested_float_24byte_callback:", result)


# CHECK: nested_float_24byte_callback: 654321.0


def main():
    test_nested_float_12byte()
    test_nested_float_24byte()
    test_nested_float_24byte_callback()
