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
# C reference: c_abi_test_vector_structs.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_vectors
# RUN: %t.dir/test_vectors | FileCheck %s

from std.ffi import external_call


# ============================================================================
# Vector struct: { SIMD[DType.float32, 2] } (8 bytes)
# Correct ABI (x86-64): f64 (SSE) — 8-byte all-float struct → SSE register.
# Correct ABI (arm64):  f32x2 (HFA) — 2-element float HFA → FP registers.
# ============================================================================
@fieldwise_init
struct VectorStruct8(TrivialRegisterPassable):
    var v: SIMD[.float32, 2]


def test_vec_8byte():
    var s = VectorStruct8(SIMD[.float32, 2](1.0, 2.0))
    var result = external_call["c_func_vec_8byte", VectorStruct8](s)
    print("vec_8byte:", result.v[0], result.v[1])


# CHECK: vec_8byte: 2.0 3.0


# ============================================================================
# Integer vector struct: { SIMD[DType.int32, 4], SIMD[DType.int32, 4] } (32 B)
# Correct ABI (arm64):  HVA of 2 members — a 128-bit vector is a member
#                       whatever its element type → q0-q1.
# Correct ABI (x86-64): >16 bytes → MEMORY.
# ============================================================================
@fieldwise_init
struct IntVectorPair(TrivialRegisterPassable):
    var a: SIMD[.int32, 4]
    var b: SIMD[.int32, 4]


def test_vec_int_32byte():
    var s = IntVectorPair(
        SIMD[.int32, 4](1, 2, 3, 4), SIMD[.int32, 4](5, 6, 7, 8)
    )
    var result = external_call["c_func_vec_int_32byte", IntVectorPair](s)
    print("vec_int_32byte:", result.a, result.b)


# CHECK: vec_int_32byte: [2, 3, 4, 5] [6, 7, 8, 9]


def main():
    test_vec_8byte()
    test_vec_int_32byte()
