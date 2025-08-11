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
from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.slice import slice_dim_as_view

from utils.index import IndexList


def print_elements[
    dtype: DType, in_rank: Int
](tensor: NDBuffer[dtype, in_rank]):
    print("New shape:", tensor.get_shape())
    print("New strides:", tensor.get_strides())

    @always_inline
    @parameter
    fn print_elements_lambda[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        var index = rebind[IndexList[in_rank]](idx)
        print(tensor[index])

    elementwise[print_elements_lambda, 1](tensor.get_shape())


# slice_dim
def test_slice_dim[
    dtype: DType, numelems: Int, outer_rank: Int, dim: Int
](dims: DimList, start: Int, stop: Int, step: Int):
    var memory1 = InlineArray[Scalar[dtype], numelems](uninitialized=True)
    var in_tensor = NDBuffer[
        dtype,
        outer_rank,
    ](memory1.unsafe_ptr(), dims)

    print("In shape:", in_tensor.get_shape())
    print("In strides:", in_tensor.get_strides())

    for i in range(numelems):
        in_tensor.data[i] = i

    # Perform the slice even if we are testing the copy so we get the target size.
    var sliced = slice_dim_as_view[dtype, outer_rank, dim](
        in_tensor,
        start,
        stop,
        step,
    )

    print_elements[dtype, outer_rank](sliced)


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
    test_slice_dim[DType.float32, 16, 2, 0](DimList(4, 4), 2, 4, 1)

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
    test_slice_dim[DType.float32, 16, 2, 1](DimList(4, 4), 2, 4, 1)


def main():
    test_slice_dim_basic()
