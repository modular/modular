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
# C reference: c_abi_test_hva.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_hva
# RUN: %t.dir/test_hva | FileCheck %s

from std.ffi import external_call


# Two 128-bit integer vectors: an HVA in v0-v1 under AAPCS64.
@fieldwise_init
struct HvaI32x4x2(TrivialRegisterPassable):
    var a: SIMD[DType.int32, 4]
    var b: SIMD[DType.int32, 4]


# A 128-bit float32x4 with a 128-bit float64x2: same vector size, so still
# an HVA even though the element types differ.
@fieldwise_init
struct HvaMixed128(TrivialRegisterPassable):
    var a: SIMD[DType.float32, 4]
    var b: SIMD[DType.float64, 2]


# A 64-bit integer vector with a 64-bit float vector: same size, an HVA.
@fieldwise_init
struct HvaMixed64(TrivialRegisterPassable):
    var a: SIMD[DType.int32, 2]
    var b: SIMD[DType.float32, 2]


# A 128-bit and a 64-bit vector: different sizes, not homogeneous.
@fieldwise_init
struct NotHva(TrivialRegisterPassable):
    var a: SIMD[DType.int32, 4]
    var b: SIMD[DType.int32, 2]


def test_hva_i32x4x2():
    var v = HvaI32x4x2(
        SIMD[DType.int32, 4](1, 2, 3, 4), SIMD[DType.int32, 4](5, 6, 7, 8)
    )
    var result = external_call["c_func_hva_i32x4x2", HvaI32x4x2](v)
    print("hva_i32x4x2:", result.a[0], result.a[3], result.b[0], result.b[3])


# CHECK: hva_i32x4x2: 2 5 6 9


def test_hva_mixed128():
    var v = HvaMixed128(
        SIMD[DType.float32, 4](1.0, 1.0, 1.0, 1.0),
        SIMD[DType.float64, 2](2.0, 2.0),
    )
    var result = external_call["c_func_hva_mixed128", HvaMixed128](v)
    print("hva_mixed128:", result.a[0], result.b[0])


# CHECK: hva_mixed128: 2.0 3.0


def test_hva_mixed64():
    var v = HvaMixed64(
        SIMD[DType.int32, 2](1, 2), SIMD[DType.float32, 2](3.0, 4.0)
    )
    var result = external_call["c_func_hva_mixed64", HvaMixed64](v)
    print("hva_mixed64:", result.a[0], result.a[1], result.b[0], result.b[1])


# CHECK: hva_mixed64: 2 3 4.0 5.0


def test_not_hva():
    var v = NotHva(SIMD[DType.int32, 4](1, 2, 3, 4), SIMD[DType.int32, 2](5, 6))
    var result = external_call["c_func_not_hva", NotHva](v)
    print("not_hva:", result.a[0], result.a[3], result.b[0], result.b[1])


# CHECK: not_hva: 2 5 6 7


def main():
    test_hva_i32x4x2()
    test_hva_mixed128()
    test_hva_mixed64()
    test_not_hva()
