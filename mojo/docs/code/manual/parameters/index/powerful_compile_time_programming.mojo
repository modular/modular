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


fn slice[
    dtype: DType, size: Int, //
](x: SIMD[dtype, size], offset: Int) -> SIMD[dtype, size // 2]:
    alias new_size = size // 2
    var result = SIMD[dtype, new_size]()
    for i in range(new_size):
        result[i] = SIMD[dtype, 1](x[i + offset])
    return result


fn reduce_add(x: SIMD) -> Int:
    @parameter
    if x.size == 1:
        return Int(x[0])
    elif x.size == 2:
        return Int(x[0]) + Int(x[1])

    # Extract the top/bottom halves, add them, sum the elements.
    alias half_size = x.size // 2
    var lhs = slice(x, 0)
    var rhs = slice(x, half_size)
    return reduce_add(lhs + rhs)


def main():
    var x = SIMD[DType.index, 4](1, 2, 3, 4)
    print(x)
    print("Elements sum:", reduce_add(x))
