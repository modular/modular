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


fn concat[
    dtype: DType, ls_size: Int, rh_size: Int, //
](lhs: SIMD[dtype, ls_size], rhs: SIMD[dtype, rh_size]) -> SIMD[
    dtype, ls_size + rh_size
]:
    var result = SIMD[dtype, ls_size + rh_size]()

    @parameter
    for i in range(ls_size):
        result[i] = lhs[i]

    @parameter
    for j in range(rh_size):
        result[ls_size + j] = rhs[j]
    return result


def main():
    var a = SIMD[DType.float32, 2](1, 2)
    var x = concat(a, a)

    print("result type:", x.dtype, "length:", x.size)
