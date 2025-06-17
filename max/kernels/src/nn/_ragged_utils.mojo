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

from sys.info import _current_target, simdwidthof

from algorithm.functional import elementwise
from buffer import NDBuffer
from gpu.host._compile import get_gpu_target
from gpu.host.info import is_cpu
from layout import LayoutTensor
from runtime.asyncrt import DeviceContextPtr

from utils import IndexList


@always_inline
fn get_batch_from_row_offsets(
    row_offsets: NDBuffer[DType.uint32, 1, *_], tok_idx: Int
) -> Int:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    var row_offsets_size = row_offsets.dim[0]()

    debug_assert(
        tok_idx >= 0 and tok_idx < Int(row_offsets[row_offsets_size - 1]),
        "tok_idx is out of range of row_offsets",
    )

    var low: UInt = 0
    var high: UInt = row_offsets_size - 1
    while low + 1 != high:
        var mid = (low + high) // 2

        if tok_idx >= Int(row_offsets[mid]):
            low = mid
        elif tok_idx < Int(row_offsets[mid]):
            high = mid

    return Int(low)


@always_inline
fn get_batch_from_row_offsets(
    row_offsets: LayoutTensor[DType.uint32, **_], tok_idx: Int
) -> Int:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    var row_offsets_size = row_offsets.size()

    debug_assert(
        tok_idx >= 0 and tok_idx < Int(row_offsets[row_offsets_size - 1]),
        "tok_idx is out of range of row_offsets",
    )

    var low: UInt = 0
    var high: UInt = row_offsets_size - 1
    while low + 1 != high:
        var mid = (low + high) // 2

        if tok_idx >= Int(row_offsets[mid]):
            low = mid
        elif tok_idx < Int(row_offsets[mid]):
            high = mid

    return Int(low)


fn merge_ragged_tensors[
    rank: Int,
    type: DType, //,
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, type, rank],
    c_row_offsets: NDBuffer[mut=True, DType.uint32, 1],
    a: NDBuffer[type, rank],
    a_row_offsets: NDBuffer[DType.uint32, 1],
    b: NDBuffer[type, rank],
    b_row_offsets: NDBuffer[DType.uint32, 1],
    ctx: DeviceContextPtr,
) raises:
    @always_inline
    @parameter
    fn merge_fn[width: Int, rank_: Int](idx: IndexList[rank_]):
        constrained[rank_ == rank, "Invalid rank passed to the kernel"]()

        var a_tensor_size = a.dim[0]()
        var is_tensor_a = idx[0] < a_tensor_size

        var batch_id: Int
        var src_idx: IndexList[rank_] = idx
        if is_tensor_a:
            batch_id = get_batch_from_row_offsets(a_row_offsets, src_idx[0])
        else:
            src_idx[0] = idx[0] - a_tensor_size
            batch_id = get_batch_from_row_offsets(b_row_offsets, src_idx[0])

        var dst_idx: IndexList[rank_] = idx
        var dst_row_idx: Int = src_idx[0]

        if is_tensor_a:
            dst_row_idx += Int(b_row_offsets[batch_id])
        else:
            dst_row_idx += Int(a_row_offsets[batch_id + 1])

        dst_idx[0] = dst_row_idx

        # The elementwise function takes care of handling the scenario where
        # tensors' last dimension is not multiple of simdwidth. It will call
        # this `merge_fn`function with width = 1 for the last few elements.
        var val: SIMD[type, width]
        if is_tensor_a:
            val = a.load[width=width](src_idx)
        else:
            val = b.load[width=width](src_idx)

        c.store[width=width](rebind[IndexList[rank]](dst_idx), val)

        # Update the row offsets if this is the first element of the batch
        var is_first_element = is_tensor_a and src_idx[0] == Int(
            a_row_offsets[batch_id]
        )

        @parameter
        for i in range(1, rank):
            if idx[i] != 0:
                is_first_element = False

        if is_first_element:
            c_row_offsets.store[width=1](batch_id, dst_row_idx)

            # If this is the last batch, also update the last row offset to the total size
            if batch_id == c_row_offsets.dim[0]() - 2:
                var total_size = a.dim[0]() + b.dim[0]()
                c_row_offsets.store[width=1](batch_id + 1, total_size)

    alias compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    alias target_simd_width = simdwidthof[type, target=compile_target]()
    alias kernel_simd_width = 1 if rank == 1 else target_simd_width

    elementwise[
        func=merge_fn,
        simd_width=kernel_simd_width,
        target=target,
        _trace_description="merge_ragged_tensors",
    ](c.dynamic_shape, ctx)
