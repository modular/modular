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
    # Elementwise comparison methods for SIMD-to-SIMD
    simd1 = SIMD[DType.int16, 4](-1, 2, -3, 4)
    simd2 = SIMD[DType.int16, 4](0, 1, 2, 3)

    # Elementwise comparison method returns SIMD[DType.bool, 4]
    simd3 = simd1.gt(simd2)  # SIMD[DType.bool, 4]
    print("simd1.gt(simd2):", simd3)

    # SIMD-to-scalar comparison: must use elementwise methods
    simd4 = SIMD[DType.int16, 4](-1, 2, -3, 4)
    simd5 = simd4.gt(2)  # SIMD[DType.bool, 4]
    print("simd4.gt(2):", simd5)

    simd6 = SIMD[DType.float32, 4](1.1, -2.2, 3.3, -4.4)
    simd7 = simd6.gt(0.5)  # SIMD[DType.bool, 4]
    print("simd6.gt(0.5):", simd7)

    # Scalar (size-1 SIMD) comparison works with operators
    var float1: Float16 = 12.345  # Scalar[DType.float16]
    var float2: Float32 = 0.5  # Scalar[DType.float32]
    result = Float32(float1) > float2  # Result is Bool
    print("Float32(float1) > float2:", result)

    # All elementwise comparison methods
    print("\n--- All elementwise comparison methods ---")
    simd8 = SIMD[DType.int32, 4](1, 2, 3, 2)
    simd9 = SIMD[DType.int32, 4](1, 2, 4, 2)

    print("simd8.eq(simd9):", simd8.eq(simd9))
    print("simd8.ne(simd9):", simd8.ne(simd9))
    print("simd8.lt(simd9):", simd8.lt(simd9))
    print("simd8.le(simd9):", simd8.le(simd9))
    print("simd8.gt(simd9):", simd8.gt(simd9))
    print("simd8.ge(simd9):", simd8.ge(simd9))

    # Bool-returning operators (equality/inequality for SIMD)
    print("\n--- Bool-returning operators (equality only) ---")
    print("simd8 == simd9:", simd8 == simd9)
    print("simd8 != simd9:", simd8 != simd9)

    # Scalar comparisons with Bool-returning operators
    print("\n--- Scalar comparisons ---")
    scalar1 = Scalar[DType.int32](5)
    scalar2 = Scalar[DType.int32](3)
    print("scalar1 > scalar2:", scalar1 > scalar2)
    print("scalar1 < scalar2:", scalar1 < scalar2)
    print("scalar1 >= scalar2:", scalar1 >= scalar2)
    print("scalar1 <= scalar2:", scalar1 <= scalar2)
