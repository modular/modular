# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


def main():
    simd1 = SIMD[DType.int32, 4](2, 3, 4, 5)
    simd2 = SIMD[DType.int32, 4](-1, 2, -3, 4)
    simd3 = simd1 * simd2
    print(simd3)

    var f1: Float16 = 2.5
    var f2: Float16 = -4.0
    var f3 = f1 * f2  # Implicitly of type Float16
    print(f3)

    var i8: Int8 = 8
    var f64: Float64 = 64.0
    # result = i8 * f64 # Error
    var result = Float64(i8) * f64
    print(result)

    simd4 = SIMD[DType.float32, 4](2.2, 3.3, 4.4, 5.5)
    simd5 = SIMD[DType.int16, 4](-1, 2, -3, 4)
    simd6 = simd4 * simd5.cast[DType.float32]()  # Convert with cast() method
    print("simd6:", simd6)
    simd7 = simd5 + SIMD[DType.int16, 4](simd4)  # Convert with SIMD constructor
    print("simd7:", simd7)

    num_float16 = SIMD[DType.float16, 4](3.5, -3.5, 3.5, -3.5)
    denom_float16 = SIMD[DType.float16, 4](2.5, 2.5, -2.5, -2.5)

    num_int32 = SIMD[DType.int32, 4](5, -6, 7, -8)
    denom_int32 = SIMD[DType.int32, 4](2, 3, -4, -5)

    # Result is SIMD[DType.float16, 4]
    true_quotient_float16 = num_float16 / denom_float16
    print("True float16 division:", true_quotient_float16)

    # Result is SIMD[DType.int32, 4]
    true_quotient_int32 = num_int32 / denom_int32
    print("True int32 division:", true_quotient_int32)

    # Result is SIMD[DType.float16, 4]
    var floor_quotient_float16 = num_float16 // denom_float16
    print("Floor float16 division:", floor_quotient_float16)

    # Result is SIMD[DType.int32, 4]
    var floor_quotient_int32 = num_int32 // denom_int32
    print("Floor int32 division:", floor_quotient_int32)

    # Result is SIMD[DType.float16, 4]
    var remainder_float16 = num_float16 % denom_float16
    print("Modulo float16:", remainder_float16)

    # Result is SIMD[DType.int32, 4]
    var remainder_int32 = num_int32 % denom_int32
    print("Modulo int32:", remainder_int32)

    print()

    # Result is SIMD[DType.float16, 4]
    var result_float16 = (
        denom_float16 * floor_quotient_float16 + remainder_float16
    )
    print("Result float16:", result_float16)

    # Result is SIMD[DType.int32, 4]
    var result_int32 = denom_int32 * floor_quotient_int32 + remainder_int32
    print("Result int32:", result_int32)
