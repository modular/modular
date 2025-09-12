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

from layout import *
from layout.int_tuple import product
from layout.layout_tensor import *
from testing import assert_equal


fn test_vectorize_2() raises:
    var tensor = LayoutTensor[
        DType.float32,
        Layout(IntTuple(IntTuple(16, 32), 4), IntTuple(IntTuple(32, 1), 512)),
        MutableAnyOrigin,
    ].stack_allocation[stack_alignment=16]()

    var n = product(tensor.layout.shape)
    for i in range(n):
        tensor.ptr[i] = i

    var frag = tensor._vectorize_2[__origin_of(), IntTuple(IntTuple(1, 4), 1)]()
    var crd = RuntimeTuple[IntTuple(2)]()
    var val = frag[crd]
    assert_equal(val[0], 64)
    assert_equal(val[1], 65)
    assert_equal(val[2], 66)
    assert_equal(val[3], 67)

    var frag_linear = tensor._vectorize_2[4]()
    var val_linear = frag_linear[crd]
    assert_equal(val_linear[0], 64)
    assert_equal(val_linear[1], 65)
    assert_equal(val_linear[2], 66)
    assert_equal(val_linear[3], 67)

    var three_dim_tensor = LayoutTensor[
        DType.float32,
        Layout(IntTuple(16, 32, 4), IntTuple(32, 1, 512)),
        MutableAnyOrigin,
    ].stack_allocation[stack_alignment=16]()

    n = product(three_dim_tensor.layout.shape)
    for i in range(n):
        three_dim_tensor.ptr[i] = i

    var frag_3dt = three_dim_tensor._vectorize_2[
        __origin_of(), IntTuple(1, 4, 1)
    ]()
    var val_3dt = frag_3dt[crd]
    assert_equal(val_3dt[0], 64)
    assert_equal(val_3dt[1], 65)
    assert_equal(val_3dt[2], 66)
    assert_equal(val_3dt[3], 67)

    var frag_linear_3dt = three_dim_tensor._vectorize_2[4]()
    var val_linear_3dt = frag_linear_3dt[crd]
    assert_equal(val_linear_3dt[0], 64)
    assert_equal(val_linear_3dt[1], 65)
    assert_equal(val_linear_3dt[2], 66)
    assert_equal(val_linear_3dt[3], 67)

    alias layout = Layout(IntTuple(8, 8), IntTuple(8, 1))
    var tensor2 = LayoutTensor[
        DType.float32,
        layout,
        MutableAnyOrigin,
    ].stack_allocation[stack_alignment=8]()

    n = product(tensor2.layout.shape)
    for i in range(n):
        tensor2.ptr[i] = i

    var frag2 = tensor2.vectorize[2]()
    var val2 = frag2[crd]
    assert_equal(val2[0], 32.0)
    assert_equal(val2[1], 40.0)

    n = product(tensor2.layout.shape)
    for i in range(n):
        tensor2.ptr[i] = i

    var frag3 = tensor2._vectorize_2[2]()
    var val3 = frag3[crd]
    assert_equal(val3[0], 16.0)
    assert_equal(val3[1], 17.0)

    alias layout_unknown = Layout(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE), IntTuple(UNKNOWN_VALUE, 1)
    )
    var heap = UnsafePointer[Int32].alloc[alignment=8](64)
    var tensor4 = LayoutTensor[
        DType.int32,
        layout_unknown,
        linear_idx_type = DType.int32,
        layout_int_type = DType.int32,
    ](heap, RuntimeLayout[layout_unknown]({8, 8}, {8, 1}))
    for i in range(64):
        tensor4.ptr[i] = i
    var frag4 = tensor4._vectorize_2[2]()
    var val4 = frag4[crd]
    assert_equal(val4[0], 16)
    assert_equal(val4[1], 17)


def main():
    test_vectorize_2()
