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

# RUN: %mojo %s | FileCheck %s

# Tests round-trip correctness of `simd_to_llvm_struct` via `__mlir_deferred_type`.

from max.gpu._utils import (
    llvm_struct_to_simd,
    simd_to_llvm_struct,
)


# @no_inline ensures each dtype/size specialization is emitted separately.
@no_inline
def round_trip[
    dtype: DType, n: SIMDLength
](v: SIMD[dtype, n]) -> SIMD[dtype, n]:
    return llvm_struct_to_simd[dtype, n](simd_to_llvm_struct[dtype, n](v))


def main():
    # CHECK: f32 x4: 1.0 2.0 3.0 4.0
    var r1 = round_trip(SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0))
    print("f32 x4:", r1[0], r1[1], r1[2], r1[3])

    # CHECK: f16 x2: 5.0 6.0
    var r2 = round_trip(SIMD[.float16, 2](5.0, 6.0))
    print("f16 x2:", r2[0], r2[1])

    # CHECK: i32 x1: 42
    var r3 = round_trip(Int32(42))
    print("i32 x1:", r3[0])

    # CHECK: i32 x8: 1 2 3 4 5 6 7 8
    var r4 = round_trip(SIMD[.int32, 8](1, 2, 3, 4, 5, 6, 7, 8))
    print("i32 x8:", r4[0], r4[1], r4[2], r4[3], r4[4], r4[5], r4[6], r4[7])

    # CHECK: bf16 x4: 1.0 2.0 3.0 4.0
    var r5 = round_trip(SIMD[.bfloat16, 4](1.0, 2.0, 3.0, 4.0))
    print("bf16 x4:", r5[0], r5[1], r5[2], r5[3])

    # CHECK: i64 x2: 100 200
    var r6 = round_trip(SIMD[.int64, 2](100, 200))
    print("i64 x2:", r6[0], r6[1])
