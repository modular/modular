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

from collections import OptionalReg
from math import iota
from random import rand, seed

from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    IntTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from nn.topk import _top_k_cpu, _top_k_sampling

from utils.index import IndexList, product


struct TestTensor[rank: Int, dtype: DType](Movable):
    var storage: List[Scalar[dtype]]
    var shape: IndexList[rank]

    fn __init__(out self, shape: IndexList[rank]):
        self.storage = List[Scalar[dtype]](
            length=UInt(shape.flattened_length()), fill=0
        )
        self.shape = shape

    fn to_layout_tensor(
        ref self,
    ) -> LayoutTensor[dtype, Layout.row_major[rank](), __origin_of(self)]:
        return {
            self.storage.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[rank]()].row_major(self.shape),
        }


fn test_case_sampling[
    rank: Int,
    dtype: DType,
    fill_fn: fn[rank: Int, dtype: DType] (
        LayoutTensor[mut=True, dtype, **_]
    ) capturing [_] -> None,
](
    K: Int,
    axis: Int,
    input_shape: IndexList[rank],
    temperature: Scalar[dtype] = 1,
) raises:
    var input_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(product(input_shape))
    )
    alias layout = Layout.row_major[rank]()
    var input = LayoutTensor[dtype, layout](
        input_ptr, RuntimeLayout[layout].row_major(input_shape)
    )

    var output_shape: IndexList[rank]
    var output_idxs_shape: IndexList[rank]

    @parameter
    if rank == 1:
        output_shape = IndexList[rank](K)
        output_idxs_shape = IndexList[rank](1)
    elif rank == 2:
        output_shape = IndexList[rank](input_shape[0], K)
        output_idxs_shape = IndexList[rank](input_shape[0], 1)
    else:
        output_shape = IndexList[rank](input_shape[0], input_shape[1], K)
        output_idxs_shape = IndexList[rank](input_shape[0], input_shape[1], 1)

    var output_vals_ptr = UnsafePointer[Scalar[dtype]].alloc(
        Int(product(output_shape))
    )
    var output_idxs_ptr = UnsafePointer[Int64].alloc(
        Int(product(output_idxs_shape))
    )
    var out_vals = LayoutTensor[dtype, layout](
        output_vals_ptr, RuntimeLayout[layout].row_major(output_shape)
    )
    var out_idxs = LayoutTensor[DType.int64, layout](
        output_idxs_ptr, RuntimeLayout[layout].row_major(output_idxs_shape)
    )

    fill_fn[rank, dtype](input)

    var max_k = K

    @parameter
    if rank == 1:
        batch_size = 1
    elif rank == 2:
        batch_size = input_shape[0]
    else:
        batch_size = input_shape[0] * input_shape[1]
    var temperature_ptr = UnsafePointer[Scalar[DType.float32]].alloc(batch_size)
    for i in range(batch_size):
        temperature_ptr[i] = temperature.cast[DType.float32]()

    alias layout_1d = Layout.row_major(UNKNOWN_VALUE)
    var temperature_buf = OptionalReg(
        LayoutTensor[DType.float32, layout_1d, MutableAnyOrigin](
            temperature_ptr,
            RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
        )
    )

    var seed_ptr = UnsafePointer[Scalar[DType.uint64]].alloc(batch_size)
    for i in range(batch_size):
        seed_ptr[i] = 12
    var seed_buf = OptionalReg(
        LayoutTensor[DType.uint64, layout_1d, MutableAnyOrigin](
            seed_ptr,
            RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
        )
    )

    _top_k_sampling(
        max_k,
        input,
        out_vals,
        out_idxs,
        temperature=temperature_buf,
        seed=seed_buf,
    )

    var _xxx_no_lifetimes = input  # intentionally bad name
    var _xx_no_lifetimes = out_vals
    var _x_no_lifetimes = out_idxs

    for i in range(out_idxs.size()):
        print(out_idxs.ptr[i], end="")
        print(",", end="")
    print("")


fn test_case[
    rank: Int,
    dtype: DType,
    fill_fn: fn[rank: Int, dtype: DType] (
        LayoutTensor[mut=True, dtype, **_]
    ) capturing [_] -> None,
    largest: Bool = True,
](K: Int, axis: Int, input_shape: IndexList[rank], sorted: Bool = True):
    var input = TestTensor[rank, dtype](input_shape)

    var output_shape = input_shape
    output_shape[axis] = K
    var out_vals = TestTensor[rank, dtype](output_shape)
    var out_idxs = TestTensor[rank, DType.int64](output_shape)

    var input_buf = input.to_layout_tensor()
    fill_fn[rank, dtype](input_buf)

    _top_k_cpu[largest=largest](
        input.to_layout_tensor(),
        K,
        axis,
        out_vals.to_layout_tensor(),
        out_idxs.to_layout_tensor(),
        1,  # force multithreading for small test cases,
        sorted,
    )

    var _xxx_no_origins = input^  # intentionally bad name

    for i in range(len(out_vals.storage)):
        print(out_vals.storage[i], end="")
        print(",", end="")
    print("")
    for i in range(len(out_idxs.storage)):
        print(out_idxs.storage[i], end="")
        print(",", end="")
    print("")


fn main() raises:
    seed(1)

    @parameter
    fn fill_iota[
        rank: Int, dtype: DType
    ](buf: LayoutTensor[mut=True, dtype, **_]):
        iota(
            buf.ptr,
            buf.runtime_layout.shape.value.canonicalize().flattened_length(),
        )

    @parameter
    fn fill_rand[
        rank: Int, dtype: DType
    ](buf: LayoutTensor[mut=True, dtype, **_]):
        rand(
            buf.ptr,
            buf.runtime_layout.shape.value.canonicalize().flattened_length(),
        )

    fn test_1d_sorted():
        print("== test_1d_sorted")
        test_case[1, DType.float32, fill_iota](
            5, 0, IndexList[1](10), sorted=True
        )

    # CHECK-LABEL: test_1d_sorted
    # CHECK: 9.0,8.0,7.0,6.0,5.0,
    # CHECK: 9,8,7,6,5,
    test_1d_sorted()

    fn test_1d_notsorted():
        print("== test_1d_notsorted")
        test_case[1, DType.float32, fill_iota](
            5, 0, IndexList[1](10), sorted=False
        )

    # CHECK-LABEL: test_1d_notsorted
    # CHECK: 8.0,7.0,6.0,9.0,5.0,
    # CHECK: 8,7,6,9,5,
    test_1d_notsorted()

    fn test_axis_1():
        print("== test_axis_1")
        test_case[2, DType.float32, fill_iota](
            2, 1, IndexList[2](4, 4), sorted=True
        )

    # CHECK-LABEL: test_axis_1
    # CHECK: 3.0,2.0,7.0,6.0,11.0,10.0,15.0,14.0,
    # CHECK-NEXT: 3,2,3,2,3,2,3,2,
    test_axis_1()

    fn test_axis_1_notsorted():
        print("== test_axis_1_notsorted")
        test_case[2, DType.float32, fill_iota](
            2, 1, IndexList[2](4, 4), sorted=False
        )

    # CHECK-LABEL: test_axis_1_notsorted
    # CHECK: 3.0,2.0,7.0,6.0,11.0,10.0,15.0,14.0,
    # CHECK-NEXT: 3,2,3,2,3,2,3,2,
    test_axis_1_notsorted()

    fn test_smallest():
        print("== test_smallest")
        test_case[2, DType.float32, fill_iota, largest=False](
            2, 1, IndexList[2](4, 4), False
        )

    # CHECK-LABEL: test_smallest
    # CHECK: 0.0,1.0,4.0,5.0,8.0,9.0,12.0,13.0,
    # CHECK-NEXT: 0,1,0,1,0,1,0,1,
    test_smallest()

    fn test_axis_0():
        print("== test_axis_0")
        test_case[2, DType.float32, fill_iota](2, 0, IndexList[2](4, 4))

    # CHECK-LABEL: test_axis_0
    # CHECK: 12.0,13.0,14.0,15.0,8.0,9.0,10.0,11.0,
    # CHECK-NEXT: 3,3,3,3,2,2,2,2,
    test_axis_0()

    @parameter
    fn fill_identical[
        rank: Int, dtype: DType
    ](buf: LayoutTensor[mut=True, dtype, **_]):
        _ = buf.fill(1)

    fn test_identical():
        print("== test_identical")
        test_case[2, DType.float32, fill_identical](3, 0, IndexList[2](4, 4))

    # CHECK-LABEL: test_identical
    # CHECK: 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    # CHECK-NEXT: 0,0,0,0,1,1,1,1,2,2,2,2,
    test_identical()

    fn test_identical_large():
        print("== test_identical_large")
        test_case[2, DType.float32, fill_identical](3, 0, IndexList[2](33, 33))

    # CHECK-LABEL: test_identical_large
    # CHECK: 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
    # CHECK: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    test_identical_large()

    fn test_max_k():
        print("== test_max_k")
        test_case[2, DType.float32, fill_iota](3, 0, IndexList[2](3, 4))

    # CHECK-LABEL: test_max_k
    # CHECK: 8.0,9.0,10.0,11.0,4.0,5.0,6.0,7.0,0.0,1.0,2.0,3.0,
    # CHECK-NEXT: 2,2,2,2,1,1,1,1,0,0,0,0,
    test_max_k()

    @parameter
    fn fill_custom[
        rank: Int, dtype: DType
    ](buf: LayoutTensor[mut=True, dtype, **_]):
        var flat_buf = LayoutTensor[
            dtype,
            Layout.row_major(UNKNOWN_VALUE),
            address_space = buf.address_space,
        ](
            buf.ptr,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                IndexList[1](buf.size())
            ),
        )

        for i in range(flat_buf.size()):
            var idx = flat_buf.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)
            )
            flat_buf.ptr[idx] = flat_buf.size() - i - 1
        flat_buf[0] = -1

    fn test_5d():
        print("== test_5d")
        test_case[5, DType.float32, fill_custom](
            1, 1, IndexList[5](1, 4, 3, 2, 1)
        )

    # CHECK-LABEL: == test_5d
    # CHECK: 17.0,22.0,21.0,20.0,19.0,18.0,
    # CHECK-NEXT: 1,0,0,0,0,0,
    test_5d()

    fn test_1d_sorted_sampling() raises:
        print("== test_1d_sorted_sampling")
        alias rank = 1
        test_case_sampling[1, DType.float32, fill_iota](
            5,
            0,
            IndexList[1](10),
        )

    # CHECK-LABEL: test_1d_sorted_sampling
    # CHECK: 9,
    test_1d_sorted_sampling()

    fn test_2d_sorted_sampling() raises:
        print("== test_2d_sorted_sampling")
        test_case_sampling[2, DType.float32, fill_rand](
            5,
            1,
            IndexList[2](5, 10),
        )

    # CHECK-LABEL: test_2d_sorted_sampling
    # CHECK: 4,1,0,6,4,
    test_2d_sorted_sampling()

    fn test_3d_sorted_sampling() raises:
        print("== test_3d_sorted_sampling")
        test_case_sampling[3, DType.float32, fill_rand](
            5,
            2,
            IndexList[3](3, 5, 10),
        )

    # CHECK-LABEL: test_3d_sorted_sampling
    # 6,9,5,2,3,1,7,9,5,1,9,0,2,3,4,
    test_3d_sorted_sampling()

    @parameter
    fn ones[rank: Int, dtype: DType](buf: LayoutTensor[mut=True, dtype, **_]):
        for i in range(
            buf.runtime_layout.shape.value.canonicalize().flattened_length()
        ):
            buf.ptr[i] = 1

    fn test_1d_sorted_sampling_temp() raises:
        print("== test_1d_sorted_sampling_temp")
        alias rank = 1
        test_case_sampling[1, DType.float32, fill_rand](
            5, 0, IndexList[1](10), temperature=0.7
        )

    # CHECK-LABEL: test_1d_sorted_sampling_temp
    # CHECK: 6,
    test_1d_sorted_sampling_temp()

    fn test_2d_sorted_sampling_temp() raises:
        print("== test_2d_sorted_sampling_temp")
        test_case_sampling[2, DType.float32, fill_rand](
            5,
            1,
            IndexList[2](50, 10),
            temperature=0.7,
        )

    # CHECK-LABEL: test_2d_sorted_sampling_temp
    # CHECK: 6,6,0,0,5,2,6,4,3,1,0,4,8,0,0,0,5,7,7,4,6,3,4,2,5,3,6,7,8,6,6,5,9,7,8,3,7,4,8,6,2,8,6,4,5,7,8,3,5,0,
    test_2d_sorted_sampling_temp()

    fn test_2d_sorted_sampling_temp_zero() raises:
        print("== test_2d_sorted_sampling_temp_zero")
        test_case_sampling[2, DType.float32, fill_rand](
            5,
            1,
            IndexList[2](50, 10),
            temperature=0.0,
        )

    # CHECK-LABEL: test_2d_sorted_sampling_temp_zero
    # CHECK: 7,7,2,9,8,4,3,2,4,0,8,0,5,5,4,6,0,3,0,6,2,5,8,3,4,0,7,4,1,3,1,6,7,2,8,8,3,4,1,0,9,8,2,6,2,3,2,8,2,3,
    test_2d_sorted_sampling_temp_zero()

    fn test_deterministic_sampling() raises:
        print("== test_deterministic_sampling")
        test_case_sampling[2, DType.float32, ones](
            5,
            1,
            IndexList[2](50, 10),
        )

    # CHECK-LABEL: test_deterministic_sampling
    # CHECK: 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    test_deterministic_sampling()
