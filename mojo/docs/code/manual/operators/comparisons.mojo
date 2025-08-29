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
    simd1 = SIMD[DType.int16, 4](-1, 2, -3, 4)
    simd2 = SIMD[DType.int16, 4](0, 1, 2, 3)
    simd3 = simd1.gt(simd2)  # SIMD[DType.bool, 4]
    print(simd3)

    simd4 = SIMD[DType.int16, 4](-1, 2, -3, 4)
    simd5 = simd4.gt(2)  # SIMD[DType.bool, 4]
    print(simd5)

    simd6 = SIMD[DType.float32, 4](1.1, -2.2, 3.3, -4.4)
    simd7 = simd6.gt(0.5)  # SIMD[DType.bool, 4]
    print(simd7)

    var float1: Float16 = 12.345  # SIMD[DType.float16, 1]
    var float2: Float32 = 0.5  # SIMD[DType.float32, 1]
    result = Float32(float1) > float2  # Result is SIMD[DType.bool, 1]
    print(result)
