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
from nn.arange import arange, arange_shape

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
def test_arange[
    dtype: DType,
](start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]):
    var outshape: IndexList[1]
    try:
        outshape = arange_shape[dtype, True](start, stop, step)
    except e:
        outshape = IndexList[1]()
        print(e)
    print("Expected output shape: ")
    print(outshape)

    alias max_output_size = 64

    if max_output_size < outshape[0]:
        print("Memory is larger than static limit, test failed")
        return

    var memory4 = InlineArray[Scalar[dtype], max_output_size](
        uninitialized=True
    )
    var out_tensor = NDBuffer[dtype, 1](memory4.unsafe_ptr(), outshape)

    @always_inline
    @__copy_capture(out_tensor, step, start, stop)
    @parameter
    fn arange_lambda[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        var index = rebind[IndexList[1]](idx)
        var range_val = arange[dtype, simd_width](start, stop, step, index)
        out_tensor.store[width=simd_width](index, range_val)

    elementwise[arange_lambda, 1](
        rebind[IndexList[1]](out_tensor.get_shape()),
    )

    print_elements[dtype, 1](out_tensor)


# CHECK-LABEL: == test_arrange_basic
def test_arrange_basic():
    print("== test_arrange_basic")

    # CHECK-NEXT: Expected output shape:
    # CHECK-NEXT: (6,)
    # CHECK-NEXT: New shape: (6,)
    # CHECK-NEXT: New strides: (1,)
    # CHECK-NEXT: 0
    # CHECK-NEXT: 1
    # CHECK-NEXT: 2
    # CHECK-NEXT: 3
    # CHECK-NEXT: 4
    # CHECK-NEXT: 5

    # print(np.arange(0, 6, 1))
    test_arange[DType.int32](0, 6, 1)

    # CHECK-NEXT: Expected output shape:
    # CHECK-NEXT: (3,)
    # CHECK-NEXT: New shape: (3,)
    # CHECK-NEXT: New strides: (1,)
    # CHECK-NEXT: 38
    # CHECK-NEXT: 15
    # CHECK-NEXT: -8

    # print(np.arange(38, -13, -23))
    test_arange[DType.int32](38, -13, -23)


def main():
    test_arrange_basic()
