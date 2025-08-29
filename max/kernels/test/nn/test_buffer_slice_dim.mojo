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


from algorithm import elementwise
from layout import (
    LayoutTensor,
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from nn.slice import slice_dim_as_view

from utils.index import IndexList


def print_elements[dtype: DType](tensor: LayoutTensor[dtype, **_]):
    print("New shape:", tensor.runtime_layout.shape.value.canonicalize())
    print("New strides:", tensor.runtime_layout.stride.value.canonicalize())

    @always_inline
    @parameter
    fn print_elements_lambda[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](coords: IndexList[rank]):
        var index = rebind[IndexList[tensor.rank]](coords)
        var idx = tensor.runtime_layout(
            RuntimeTuple[fill_like(tensor.layout.shape, UNKNOWN_VALUE)](index)
        )
        print(tensor.ptr[idx])

    elementwise[print_elements_lambda, 1](
        tensor.runtime_layout.shape.value.canonicalize()
    )


# slice_dim
def test_slice_dim[
    dtype: DType, numelems: Int, outer_rank: Int, dim: Int
](dims: IndexList[outer_rank], start: Int, stop: Int, step: Int):
    var memory1 = InlineArray[Scalar[dtype], numelems](uninitialized=True)
    var in_tensor = LayoutTensor[dtype, Layout.row_major[outer_rank]()](
        memory1.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[outer_rank]()].row_major(dims),
    )

    print("In shape:", in_tensor.runtime_layout.shape.value.canonicalize())
    print("In strides:", in_tensor.runtime_layout.stride.value.canonicalize())

    for i in range(numelems):
        in_tensor.ptr[i] = i

    # Perform the slice even if we are testing the copy so we get the target size.
    var sliced = slice_dim_as_view[dtype, dim](
        in_tensor,
        start,
        stop,
        step,
    )

    print_elements(sliced)


# CHECK-LABEL: == test_slice_dim_basic
def test_slice_dim_basic():
    print("== test_slice_dim_basic")

    # CHECK-NEXT: In shape: (4, 4)
    # CHECK-NEXT: In strides: (4, 1)
    # CHECK-NEXT: New shape: (2, 4)
    # CHECK-NEXT: New strides: (4, 1)
    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 9.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 11.0
    # CHECK-NEXT: 12.0
    # CHECK-NEXT: 13.0
    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 15.0

    # print(torch.arange(0, 16).reshape(4, 4)[2:4:1, :].flatten())
    test_slice_dim[DType.float32, 16, 2, 0](IndexList[2](4, 4), 2, 4, 1)

    # CHECK-NEXT: In shape: (4, 4)
    # CHECK-NEXT: In strides: (4, 1)
    # CHECK-NEXT: New shape: (4, 2)
    # CHECK-NEXT: New strides: (4, 1)
    # CHECK-NEXT: 2.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 7.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 11.0
    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 15.0

    # print(torch.arange(0, 16).reshape(4, 4)[:, 2:4:1].flatten())
    test_slice_dim[DType.float32, 16, 2, 1](IndexList[2](4, 4), 2, 4, 1)


def main():
    test_slice_dim_basic()
