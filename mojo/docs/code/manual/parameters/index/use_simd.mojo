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
    var vector = SIMD[DType.int16, 4](1, 2, 3, 4)
    vector = vector * vector
    for i in range(4):
        print(vector[i], end=" ")
    print()

    # Example: "Using parameterized types and functions"

    # Make a vector of 4 floats.
    var small_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

    # Make a big vector containing 1.0 in float16 format.
    var big_vec = SIMD[DType.float16, 32](1.0)

    # Do some math and convert the elements to float32.
    var bigger_vec = (big_vec + big_vec).cast[DType.float32]()

    # You can write types out explicitly if you want of course.
    var bigger_vec2: SIMD[DType.float32, 32] = bigger_vec

    print("small_vec DType:", small_vec.dtype, "size:", small_vec.size)
    print(
        "bigger_vec2 DType:",
        bigger_vec2.dtype,
        "size:",
        bigger_vec2.size,
    )

    # second example

    from math import sqrt

    fn rsqrt[dt: DType, width: Int](x: SIMD[dt, width]) -> SIMD[dt, width]:
        return 1 / sqrt(x)

    var v = SIMD[DType.float16, 4](42)
    print(rsqrt(v))
