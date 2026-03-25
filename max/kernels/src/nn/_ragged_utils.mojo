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

from std.sys.info import _current_target, simd_width_of

from std.algorithm.functional import elementwise
from std.gpu.host import get_gpu_target
from std.gpu.host.info import is_cpu
from layout import LayoutTensor, TileTensor
from std.gpu.host import DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from std.utils import IndexList


@always_inline
def get_batch_from_row_offsets(
    row_offsets: LayoutTensor[DType.uint32, ...], tok_idx: Int
) -> Int:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    var row_offsets_size = row_offsets.size()

    assert tok_idx >= 0 and tok_idx < Int(
        row_offsets[row_offsets_size - 1]
    ), "tok_idx is out of range of row_offsets"

    var low: UInt = 0
    var high = UInt(row_offsets_size - 1)
    while low + 1 != high:
        var mid = (low + high) // 2

        if tok_idx >= Int(row_offsets[mid]):
            low = mid
        elif tok_idx < Int(row_offsets[mid]):
            high = mid

    return Int(low)


@always_inline
def get_batch_from_row_offsets(
    row_offsets: TileTensor[DType.uint32, ...], tok_idx: Int
) -> Int:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    comptime assert row_offsets.flat_rank == 1

    var row_offsets_size = row_offsets.num_elements()

    assert tok_idx >= 0 and tok_idx < Int(
        row_offsets[row_offsets_size - 1]
    ), "tok_idx is out of range of row_offsets"

    var low: UInt = 0
    var high = UInt(row_offsets_size - 1)
    while low + 1 != high:
        var mid = (low + high) // 2

        if tok_idx >= Int(row_offsets[mid]):
            low = mid
        elif tok_idx < Int(row_offsets[mid]):
            high = mid

    return Int(low)


@always_inline
def get_batch_and_token_idx_from_row_offsets(
    row_offsets: TileTensor[DType.uint32, ...], tok_idx: Int
) -> Tuple[Int, Int]:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    comptime assert row_offsets.flat_rank == 1

    var row_offsets_size = row_offsets.num_elements()

    assert tok_idx >= 0 and tok_idx < Int(
        row_offsets[row_offsets_size - 1]
    ), "tok_idx is out of range of row_offsets"

    var low: UInt = 0
    var high = UInt(row_offsets_size - 1)
    while low + 1 != high:
        var mid = (low + high) // 2

        if tok_idx >= Int(row_offsets[mid]):
            low = mid
        elif tok_idx < Int(row_offsets[mid]):
            high = mid

    return Int(low), Int(tok_idx - Int(row_offsets[low]))


def merge_ragged_tensors[
    rank: Int,
    dtype: DType,
    //,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, dtype, ...],
    c_row_offsets: TileTensor[mut=True, DType.uint32, ...],
    a: TileTensor[dtype, ...],
    a_row_offsets: TileTensor[DType.uint32, ...],
    b: TileTensor[dtype, ...],
    b_row_offsets: TileTensor[DType.uint32, ...],
    ctx: DeviceContextPtr,
) raises:
    comptime assert c.flat_rank == rank, "c.flat_rank must equal rank"
    comptime assert a.flat_rank == rank, "a.flat_rank must equal rank"
    comptime assert b.flat_rank == rank, "b.flat_rank must equal rank"
    comptime assert (
        c_row_offsets.flat_rank == 1
    ), "c_row_offsets.flat_rank must be 1"
    comptime assert (
        a_row_offsets.flat_rank == 1
    ), "a_row_offsets.flat_rank must be 1"
    comptime assert (
        b_row_offsets.flat_rank == 1
    ), "b_row_offsets.flat_rank must be 1"

    @always_inline
    @parameter
    def merge_fn[
        width: Int, rank_: Int, alignment: Int = 1
    ](idx: IndexList[rank_]):
        comptime assert rank_ == rank, "Invalid rank passed to the kernel"

        var a_tensor_size = Int(a.dim[0]())
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

        # Compute flat offsets for pointer load/store (Horner form).
        # Inner dimensions are the same across a, b, and c.
        @always_inline
        @parameter
        def _flat_offset[r: Int](index: IndexList[r]) -> Int:
            comptime assert r == rank
            var flat = index[0]
            comptime for i in range(1, rank):
                flat = flat * Int(c.dim[i]()) + index[i]
            return flat

        var src_flat = _flat_offset(src_idx)
        var dst_flat = _flat_offset(dst_idx)

        # The elementwise function takes care of handling the scenario where
        # tensors' last dimension is not multiple of simdwidth. It will call
        # this `merge_fn`function with width = 1 for the last few elements.
        var val: SIMD[dtype, width]
        if is_tensor_a:
            val = a.ptr.load[width=width](src_flat)
        else:
            val = b.ptr.load[width=width](src_flat)

        c.ptr.mut_cast[True]().store[width=width](dst_flat, val)

        # Update the row offsets if this is the first element of the batch
        var is_first_element = is_tensor_a and src_idx[0] == Int(
            a_row_offsets[batch_id]
        )

        comptime for i in range(1, rank):
            if idx[i] != 0:
                is_first_element = False

        if is_first_element:
            c_row_offsets[batch_id] = UInt32(dst_row_idx)

            # If this is the last batch, also update the last row offset to the total size
            if batch_id == Int(c_row_offsets.dim[0]()) - 2:
                var total_size = Int(a.dim[0]()) + Int(b.dim[0]())
                c_row_offsets[batch_id + 1] = UInt32(total_size)

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = 1 if rank == 1 else target_simd_width

    var shape = IndexList[rank]()
    comptime for i in range(rank):
        shape[i] = Int(c.dim[i]())

    elementwise[
        func=merge_fn,
        simd_width=kernel_simd_width,
        target=target,
        _trace_description="merge_ragged_tensors",
    ](shape, ctx)


def eagle_prefill_shift_tokens[
    dtype: DType,
    //,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, dtype, ...],
    tokens: TileTensor[dtype, ...],
    offsets: TileTensor[DType.uint32, ...],
    shift_next_tokens: TileTensor[dtype, ...],
    num_draft_tokens: TileTensor[DType.int64, ...],
    ctx: DeviceContextPtr,
) raises:
    """Shift ragged tokens left by 1 per request, appending bonus tokens.

    Dispatches at runtime on num_draft_tokens:
    - K=0 (prefill): shift each request's tokens left by 1, append
      shift_next_tokens
    - K>0 (decode): passthrough (copy tokens unchanged)
    """
    comptime assert output.flat_rank == 1
    comptime assert tokens.flat_rank == 1
    comptime assert offsets.flat_rank == 1
    comptime assert shift_next_tokens.flat_rank == 1
    comptime assert num_draft_tokens.flat_rank == 1

    @always_inline
    @parameter
    def shift_fn[
        width: Int, rank_: Int, alignment: Int = 1
    ](idx: IndexList[rank_]):
        comptime assert rank_ == 1

        var i = idx[0]
        var K = Int(num_draft_tokens.ptr.load[width=1](0))

        if K > 0:
            # Decode: passthrough copy
            output.ptr.mut_cast[True]().store[width=1](
                i, tokens.ptr.load[width=1](i)
            )
        else:
            # Prefill: shift left by 1 per batch, append bonus token
            var batch_id = get_batch_from_row_offsets(offsets, i)
            var end = Int(offsets[batch_id + 1])

            if i < end - 1:
                # Not the last position: copy from next position
                output.ptr.mut_cast[True]().store[width=1](
                    i, tokens.ptr.load[width=1](i + 1)
                )
            else:
                # Last position in batch: append shift_next_tokens
                output.ptr.mut_cast[True]().store[width=1](
                    i, shift_next_tokens.ptr.load[width=1](batch_id)
                )

    var shape = IndexList[1](Int(output.dim[0]()))

    elementwise[
        func=shift_fn,
        simd_width=1,
        target=target,
        _trace_description="eagle_prefill_shift_tokens",
    ](shape, ctx)


def extract_accepted_hs[
    rank: Int,
    dtype: DType,
    //,
    target: StaticString = "cpu",
](
    accepted_hs: TileTensor[mut=True, dtype, ...],
    accepted_offsets: TileTensor[mut=True, DType.uint32, ...],
    hs: TileTensor[dtype, ...],
    hs_offsets: TileTensor[DType.uint32, ...],
    first_rejected: TileTensor[DType.int64, ...],
    num_draft_tokens: Int,
    ctx: DeviceContextPtr,
    zero_fill_rejected: Bool = False,
) raises:
    """Extract accepted hidden states from target forward output.

    Handles both prefill (K=0, passthrough) and decode (K>0, extraction).
    K (num_draft_tokens) is a host-side scalar read by the caller.

    K==0 (prefill): D2D memcpy of offsets and HS, no kernel launch.
    K>0 (decode): launches kernel to extract accepted positions.

    """
    comptime assert accepted_hs.flat_rank == rank
    comptime assert hs.flat_rank == rank
    comptime assert accepted_offsets.flat_rank == 1
    comptime assert hs_offsets.flat_rank == 1
    comptime assert first_rejected.flat_rank == 1

    var dim1 = Int(hs.dim[1]())
    var local_batch = Int(first_rejected.dim[0]())

    # TODO: move this conditional into the elemwise lambda so it is compatible
    # with cuda graphs
    if num_draft_tokens == 0:
        # Prefill passthrough: D2D memcpy of offsets and hidden states.
        ctx[].enqueue_copy(
            accepted_offsets.ptr.mut_cast[True](),
            hs_offsets.ptr,
            local_batch + 1,
        )
        ctx[].enqueue_copy(
            accepted_hs.ptr.mut_cast[True](),
            hs.ptr,
            Int(hs.dim[0]()) * dim1,
        )
        return

    # Pass 1: Compute accepted_offsets (prefix sum).
    # Thread i computes accepted_offsets[i] = sum(first_rejected[0..i-1] + 1).
    @always_inline
    @parameter
    def offsets_fn[width: Int, r: Int, alignment: Int = 1](idx: IndexList[r]):
        var i = idx[0]
        var offset: UInt32 = 0
        for j in range(i):
            offset += UInt32(Int(first_rejected.ptr.load[width=1](j)) + 1)
        accepted_offsets[i] = offset

    elementwise[
        func=offsets_fn,
        simd_width=1,
        target=target,
        _trace_description="extract_accepted_hs_offsets",
    ](IndexList[1](local_batch + 1), ctx)

    if zero_fill_rejected:
        ctx[].enqueue_memset(
            DeviceBuffer[dtype](
                ctx=ctx[],
                ptr=accepted_hs.ptr,
                size=accepted_hs.num_elements(),
                owning=False,
            ),
            Scalar[dtype](0),
        )

    # Pass 2: Copy accepted hidden states.
    # Each thread handles one element in the 2-D HS tensor.
    @always_inline
    @parameter
    def copy_fn[width: Int, r: Int, alignment: Int = 1](idx: IndexList[r]):
        comptime assert r == rank

        var row = idx[0]
        var batch = get_batch_from_row_offsets(hs_offsets, row)
        var offset_in_request = row - Int(hs_offsets[batch])
        var first_rejected_idx = Int(first_rejected.ptr.load[width=1](batch))

        @always_inline
        @parameter
        def _flat_offset[r_: Int](index: IndexList[r_]) -> Int:
            comptime assert r_ == rank
            var flat = index[0]
            comptime for d in range(1, rank):
                flat = flat * Int(hs.dim[d]()) + index[d]
            return flat

        if offset_in_request <= first_rejected_idx:
            var dst_row = Int(accepted_offsets[batch]) + offset_in_request
            var dst_idx = idx
            dst_idx[0] = dst_row

            var src_flat = _flat_offset(idx)
            var dst_flat = _flat_offset(dst_idx)

            accepted_hs.ptr.mut_cast[True]().store[width=width](
                dst_flat, hs.ptr.load[width=width](src_flat)
            )

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = 1 if rank == 1 else target_simd_width

    var shape = IndexList[rank]()
    comptime for i in range(rank):
        shape[i] = Int(hs.dim[i]())

    elementwise[
        func=copy_fn,
        simd_width=kernel_simd_width,
        target=target,
        _trace_description="extract_accepted_hs_copy",
    ](shape, ctx)
