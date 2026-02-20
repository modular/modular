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


from layout._coord import Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.argsort import argsort
from testing import assert_true


fn test_argsort[
    *,
    ascending: Bool,
    filler: fn(Int, Int) -> Float32,
]() raises:
    print("== test_argsort")

    comptime n = 16384

    var input_ptr = alloc[Float32](n)
    var input = TileTensor(input_ptr, row_major(Idx(n)))

    var indices_ptr = alloc[Int32](n)
    var indices = TileTensor(indices_ptr, row_major(Idx(n))).fill[
        use_runtime_layout=True
    ](0)

    for i in range(n):
        input[i] = filler(i, n)

    argsort[ascending=ascending](indices, input)

    for i in range(n):
        if i < n - 1:
            var lhs = input[indices[i]]
            var rhs = input[indices[i + 1]]

            comptime if ascending:
                assert_true(
                    lhs < rhs,
                    msg=String(
                        "input[Int(indices[",
                        i,
                        "])] < input[Int(indices[",
                        i + 1,
                        "])] where the rhs is ",
                        rhs,
                        " and the lhs is ",
                        lhs,
                        " and the indices are ",
                        indices[i],
                        " and ",
                        indices[i + 1],
                        " and the ascending is ",
                        ascending,
                    ),
                )
            else:
                assert_true(
                    lhs > rhs,
                    msg=String(
                        "input[Int(indices[",
                        i,
                        "])] > input[Int(indices[",
                        i + 1,
                        "])] where the rhs is ",
                        rhs,
                        " and the lhs is ",
                        lhs,
                        " and the indices are ",
                        indices[i],
                        " and ",
                        indices[i + 1],
                        " and the ascending is ",
                        ascending,
                    ),
                )
        else:
            var lhs = input[indices[i]]
            var rhs = input[indices[0]]

            comptime if ascending:
                assert_true(
                    lhs > rhs,
                    msg=String(
                        "input[Int(indices[",
                        i,
                        "])] > input[Int(indices[0])] where the rhs is ",
                        rhs,
                        " and the lhs is ",
                        lhs,
                        " and the indices are ",
                        indices[i],
                        " and ",
                        indices[0],
                        " and the ascending is ",
                        ascending,
                    ),
                )
            else:
                assert_true(
                    lhs < rhs,
                    msg=String(
                        "input[Int(indices[",
                        i,
                        "])] < input[Int(indices[0])] where the rhs is ",
                        rhs,
                        " and the lhs is ",
                        lhs,
                        " and the indices are ",
                        indices[i],
                        " and ",
                        indices[0],
                        " and the ascending is ",
                        ascending,
                    ),
                )

    input_ptr.free()
    indices_ptr.free()


def main():
    fn linear_filler(i: Int, n: Int) -> Float32:
        return Float32(i)

    fn reverse_filler(i: Int, n: Int) -> Float32:
        return Float32(n - i)

    test_argsort[ascending=True, filler=linear_filler]()
    test_argsort[ascending=True, filler=reverse_filler]()

    test_argsort[ascending=False, filler=linear_filler]()
    test_argsort[ascending=False, filler=reverse_filler]()
