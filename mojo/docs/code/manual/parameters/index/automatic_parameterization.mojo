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


fn print_params(vec: SIMD):
    print(vec.dtype)
    print(vec.size)


fn print_params2[t: DType, s: Int, //](vec: SIMD[t, s]):
    print(vec.dtype)
    print(vec.size)


fn on_type():
    print(SIMD[DType.float32, 2].size)  # prints 2


fn on_instance():
    var x = SIMD[DType.int32, 2](4, 8)
    print(x.dtype)  # prints int32


fn interleave(v1: SIMD, v2: __type_of(v1)) -> SIMD[v1.dtype, v1.size * 2]:
    var result = SIMD[v1.dtype, v1.size * 2]()

    @parameter
    for i in range(v1.size):
        result[i * 2] = v1[i]
        result[i * 2 + 1] = v2[i]
    return result


fn foo[value: SIMD]():
    pass


def main():
    var v = SIMD[DType.float64, 4](1.0, 2.0, 3.0, 4.0)
    print_params(v)

    print_params2(v)

    on_type()

    on_instance()

    var a = SIMD[DType.int16, 4](1, 2, 3, 4)
    var b = SIMD[DType.int16, 4](0, 0, 0, 0)
    var c = interleave(a, b)
    print(c)
