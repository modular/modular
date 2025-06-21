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
from math import ceildiv, exp, iota
from sys import alignof, simdwidthof, sizeof

import gpu.warp as warp
from algorithm.functional import parallelize_over_rows
from algorithm.reduction import _get_nd_indices_from_flat_index
from bit import log2_floor
from buffer import NDBuffer
from buffer.dimlist import DimList
from builtin.io import _printf
from builtin.sort import _quicksort
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from gpu.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext
from gpu.host.dim import Dim
from gpu.host.info import is_cpu
from gpu.memory import AddressSpace, external_memory
from gpu.random import Random
from memory import stack_allocation
from nn.gather_scatter import normalize_neg_index
from nn.reshape import reshape
from runtime.asyncrt import DeviceContextPtr

from utils import IndexList
from utils.numerics import max_or_inf, min_or_neg_inf


@always_inline
fn top_k_shape_impl[
    dtype: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[dtype, rank],
    max_k: Int,
    axis: Int,
) raises -> IndexList[
    rank
]:
    """
    Compute the output shape of a top/bottom k operation.

    Parameters:
        dtype: Data type of the input buffer.
        rank: Rank of the input.
        single_thread_blocking_override: If this function can block.

    Args:
        input: The input tensor.
        max_k: The maximum K value.
        axis: The axis value in a tensor.

    Returns:
        The output shape.
    """

    if max_k < 0 or max_k > input.get_shape()[axis]:
        raise Error("[top/bottom-k] k must be within [0, input_shape[axis]]")

    var shape = input.get_shape()
    shape[normalize_neg_index(axis, rank)] = max_k

    return shape


@always_inline
fn top_k_shape[
    dtype: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[dtype, rank],
    max_k: Int,
    axis: Int,
) raises -> IndexList[
    rank
]:
    return top_k_shape_impl[
        single_thread_blocking_override=single_thread_blocking_override
    ](input, max_k, axis)


@always_inline
fn bottom_k_shape[
    dtype: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[dtype, rank],
    max_k: Int,
    axis: Int,
) raises -> IndexList[
    rank
]:
    return top_k_shape_impl[
        single_thread_blocking_override=single_thread_blocking_override
    ](input, max_k, axis)


@always_inline
fn _adjust_top_p[
    T: DType
](
    top_p: Scalar[T],
    values: UnsafePointer[Scalar[T], **_],
    k: Int,
    total_sum: Scalar[T],
) -> Scalar[T]:
    # Align the given top_p to the cumulative probability of the tokens.
    # For example, if after top_k we have three tokens with probabilities
    # [0.7, 0.2, 0.1] and top_p = 0.8, then we should sample from the first
    # two tokens with probabilities [0.7, 0.2], so we set _top_p = 0.9.
    var _top_p = Scalar[T](1)
    if top_p < 1:
        var cum_prob = Scalar[T](0)
        for ki in range(k):
            cum_prob += values[ki]
            if cum_prob >= top_p * total_sum:
                break
        _top_p = cum_prob / total_sum
    return _top_p


fn top_k[
    rank: Int,
    dtype: DType,
    out_idx_type: DType, //,
    largest: Bool = True,
    target: StaticString = "cpu",
](
    input: NDBuffer[dtype, rank],
    max_k: Int,
    axis: Int,
    out_vals: NDBuffer[mut=True, dtype, rank],
    out_idxs: NDBuffer[mut=True, out_idx_type, rank],
    sorted: Bool,
    ctx: DeviceContextPtr,
    k: OptionalReg[NDBuffer[DType.int64, 1, MutableAnyOrigin]] = None,
) raises:
    """
    Implementation of the Top K algorithm. Returns the top or bottom K elements
    and their index along a specified axis.

    Parameters:
        rank: Rank of the input.
        dtype: Data type of the input buffer.
        out_idx_type: The data dtype of the output indices (default is DType.int64).
        largest: Whether to find the maximum (top k) or minimum value (bottom k).
        target: The target to run on.

    Args:
        input: The input tensor.
        max_k: The largest number of top elements.
        axis: The axis along which to operate.
        out_vals: Output values.
        out_idxs: Output indices.
        sorted: Indicates if the top/bottom K elements are in (stable) sorted order.
        ctx: The device call context.
        k: Per batch element k value.
    """

    var normalized_axis = normalize_neg_index(Int64(axis), rank)

    @parameter
    if is_cpu[target]():
        constrained[
            out_idx_type is DType.int64,
            "out_idx_type must be int64 for cpu",
        ]()

        alias grain_size = 1000
        _top_k_cpu[largest=largest](
            input,
            max_k,
            Int(normalized_axis),
            out_vals,
            out_idxs,
            grain_size,
            sorted,
            k=k,
        )
    else:
        if normalized_axis != rank - 1:
            raise Error("axis other than -1 not supported on GPU")
        if not sorted:
            print(
                "Warning: Unsorted top-k is not supported on GPU. Falling"
                " back to sorted top-k."
            )
        var cuda_ctx = ctx.get_device_context()
        topk_gpu[sampling=False, largest=largest](
            cuda_ctx,
            max_k,
            input,
            out_vals,
            out_idxs,
            k=k,
        )


fn _top_k_cpu[
    rank: Int,
    dtype: DType,
    out_idx_type: DType,
    largest: Bool,
](
    input: NDBuffer[dtype, rank],
    max_k: Int,
    axis: Int,
    out_vals: NDBuffer[mut=True, dtype, rank],
    out_idxs: NDBuffer[mut=True, out_idx_type, rank],
    parallelism_grain_size: Int,  # impl detail, exposed for testing
    sorted: Bool,
    k: OptionalReg[NDBuffer[DType.int64, 1, MutableAnyOrigin]] = None,
):
    var shape = input.get_shape()

    @__copy_capture(shape)
    @parameter
    fn process_rows(start_row: Int, end_row: Int):
        # Allocate the index list without initializing its elements.
        var idxs = List[Int64](unsafe_uninit_length=shape[axis])

        for row_idx in range(start_row, end_row):
            var indices = _get_nd_indices_from_flat_index(row_idx, shape, axis)
            iota(idxs)

            var batch_idx = indices[0] if axis != 0 else 0
            var k_val = max_k
            if k:
                k_val = Int(k.value()[batch_idx])

            @parameter
            @always_inline
            fn indices_to_val(idx: Int64) -> Scalar[dtype]:
                indices[axis] = Int(idx)
                return input[indices]

            @parameter
            if largest:

                @parameter
                @always_inline
                fn _val_greater_than(lhs: Int64, rhs: Int64) -> Bool:
                    return indices_to_val(lhs) > indices_to_val(rhs)

                if sorted:
                    sort[_val_greater_than](idxs)
                else:
                    _ = partition[_val_greater_than](idxs, k_val)
            else:

                @parameter
                @always_inline
                fn _val_less_than(lhs: Int64, rhs: Int64) -> Bool:
                    return indices_to_val(lhs) < indices_to_val(rhs)

                if sorted:
                    sort[_val_less_than](idxs)
                else:
                    _ = partition[_val_less_than](idxs, k_val)

            if sorted:
                # for duplicate vals, the smaller index needs to appear first
                # _quicksort is not stable, so do another pass to enforce this
                # could use a stable sorting algorithm but the complexity is O(n*log(n)*log(n))
                # this is also what tensorflow and PT do:
                # https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/kernels/topk_op.cc#L171-L172
                var i = 0
                while i < shape[axis] - 1:
                    indices[axis] = Int(idxs[i])
                    var curr = input[indices]
                    var num_equal = 1
                    for j in range(i + 1, shape[axis]):
                        indices[axis] = Int(idxs[j])
                        var next = input[indices]
                        if curr != next:
                            break
                        num_equal += 1
                    if num_equal > 1:
                        var ptr = idxs.data + i
                        sort(
                            Span[idxs.T, __origin_of(idxs)](
                                ptr=ptr, length=num_equal
                            )
                        )
                    i += num_equal

            for i in range(k_val):
                indices[axis] = Int(idxs[i])
                var val = input[indices]
                indices[axis] = i
                out_vals[indices] = val
                out_idxs[indices] = rebind[Scalar[out_idx_type]](idxs[i])

    parallelize_over_rows[process_rows](shape, axis, parallelism_grain_size)


@always_inline
fn fused_token_sampling_cpu[
    dtype: DType,
    rank: Int,
    out_idx_type: DType,
](
    max_k: Int,
    input: NDBuffer[dtype, rank],
    out_idxs: NDBuffer[mut=True, out_idx_type, rank],
    k: OptionalReg[NDBuffer[DType.int64, 1, MutableAnyOrigin]] = None,
    temperature: OptionalReg[
        NDBuffer[DType.float32, 1, MutableAnyOrigin]
    ] = None,
    top_p: OptionalReg[NDBuffer[DType.float32, 1, MutableAnyOrigin]] = None,
    seed: OptionalReg[NDBuffer[DType.uint64, 1, MutableAnyOrigin]] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume.

    Parameters:
        dtype: Data type of the input buffer.
        rank: Rank of the input.
        out_idx_type: Data type of the output indices.

    Args:
        max_k: Largest number of top elements.
        input: NDBuffer[dtype, rank] (Any shape)- The input tensor.
        out_idxs: NDBuffer[out_idx_type, rank] (shape of [input_shape[:-1]] + [1]) - The output indices.
        k: Optional device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.
    """
    constrained[out_idx_type is DType.int64, "out_idx_type must be int64"]()
    # materialize the out_vals which is of shape [input[:-1]] + [k]
    var out_vals_shape = input.get_shape()
    out_vals_shape[rank - 1] = max_k
    var out_vals = NDBuffer[dtype, rank](
        UnsafePointer[Scalar[dtype]].alloc(out_vals_shape.flattened_length()),
        out_vals_shape,
    )

    _top_k_sampling(
        max_k,
        input,
        out_vals,
        rebind[NDBuffer[DType.int64, rank, out_idxs.origin]](out_idxs),
        k,
        temperature,
        top_p,
        seed,
    )

    out_vals.data.free()


fn _top_k_sampling[
    dtype: DType,
    rank: Int,
](
    max_k: Int,
    input: NDBuffer[dtype, rank],
    out_vals: NDBuffer[mut=True, dtype, rank],
    out_idxs: NDBuffer[mut=True, DType.int64, rank],
    k: OptionalReg[NDBuffer[DType.int64, 1, MutableAnyOrigin]] = None,
    temperature: OptionalReg[
        NDBuffer[DType.float32, 1, MutableAnyOrigin]
    ] = None,
    top_p: OptionalReg[NDBuffer[DType.float32, 1, MutableAnyOrigin]] = None,
    seed: OptionalReg[NDBuffer[DType.uint64, 1, MutableAnyOrigin]] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume.

    Parameters:
        dtype: Data type of the input buffer.
        rank: Rank of the input.

    Args:
        max_k: Largest number of top elements.
        input: NDBuffer[dtype, rank] (Any shape)- The input tensor.
        out_vals: NDBuffer[dtype, rank] (shape of [input[:-1]] + [k]) - The output values.
        out_idxs: NDBuffer[DType.int64, rank] (shape of [input[:-1]] + [1]) - The output indices.
        k: Optional buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.
    """
    # Now reshape for sampling
    var orig_in_shape: IndexList[rank] = input.get_shape()
    var last_dim = orig_in_shape[rank - 1]

    alias internal_rank = 2
    var internal_bs: Int
    var internal_in_shape: IndexList[internal_rank]

    @parameter
    if rank == 1:
        internal_bs = 1
        internal_in_shape = IndexList[internal_rank](1, input.size())
    elif rank == internal_rank:
        internal_bs = orig_in_shape[0]
        internal_in_shape = rebind[IndexList[internal_rank]](orig_in_shape)
    elif rank > internal_rank:
        internal_bs = Int(orig_in_shape.flattened_length() / last_dim)
        internal_in_shape = IndexList[internal_rank](internal_bs, last_dim)
    else:
        raise Error("Unsupported input rank. Must be >= 1.")

    internal_out_shape = IndexList[internal_rank](internal_bs, max_k)
    internal_out_vals = reshape(out_vals, internal_out_shape)  # internal view
    internal_out_idxs_shape = IndexList[internal_rank](internal_bs, 1)
    internal_out_idxs = reshape(
        out_idxs, internal_out_idxs_shape
    )  # internal view
    # End reshape to internal rank

    var out_idxs_tmp = NDBuffer[DType.int64, internal_rank](
        UnsafePointer[Int64].alloc(Int(out_vals.size())),
        internal_out_shape,  # topk returns K as last dim
    )
    _top_k_cpu[rank=internal_rank, dtype=dtype, largest=True](
        reshape(input, internal_in_shape),
        max_k,
        axis=internal_rank - 1,  # Always operate on the last axis
        out_vals=internal_out_vals,
        out_idxs=out_idxs_tmp,
        sorted=True,
        parallelism_grain_size=1,
        k=k,
    )

    # Sample from the top K elements
    for batch in range(internal_bs):
        var temperature_val = Scalar[DType.float32](1.0)
        if temperature:
            temperature_val = temperature.value()[batch]

        var k_val = max_k
        if k:
            k_val = Int(k.value()[batch])

        # Calculate softmax normalization
        var max_val = internal_out_vals[batch, 0]
        var sum_exp = Scalar[dtype](0)
        var exp_vals = UnsafePointer[Scalar[dtype]].alloc(k_val)
        var temp_val = temperature_val.cast[dtype]()
        for i in range(k_val):
            var val = internal_out_vals[batch, i]
            var exp_val = exp((val - max_val) / max(temp_val, 1e-6))
            exp_vals[i] = exp_val
            sum_exp += exp_val

        # Handle top_p parameter - extract scalar value from buffer
        var top_p_val = Scalar[dtype](1.0)
        if top_p:
            top_p_val = top_p.value()[batch].cast[dtype]()
        var _top_p = _adjust_top_p[dtype](top_p_val, exp_vals, k_val, sum_exp)

        # Handle seed parameter - extract scalar value from buffer
        var seed_val = UInt64(0)
        if seed:
            seed_val = seed.value()[batch]

        # Use the same RNG as the GPU sampling implementation
        var rng_state = Random(
            seed=seed_val, offset=out_idxs_tmp[batch, 0].cast[DType.uint64]()
        )
        var rng = rng_state.step_uniform()

        # Sample using the normalized probabilities
        var r = sum_exp * _top_p * rng[0].cast[dtype]()
        for i in range(k_val):
            r -= exp_vals[i]
            if r <= 0 or i == k_val - 1:
                # Store the sampled index and value
                internal_out_idxs[batch, 0] = out_idxs_tmp[batch, i]
                break
        exp_vals.free()

        # Fill remaining positions with sentinel values for unused elements
        for remaining_k in range(k_val, max_k):
            if remaining_k < internal_out_vals.get_shape()[1]:
                internal_out_vals[batch, remaining_k] = _topk_dead_val[
                    dtype, True
                ]()
            # Note: out_idxs for sampling only has 1 element in last dim, so no need to fill indices
    out_idxs_tmp.data.free()


@always_inline("nodebug")
fn _topk_dead_val[T: DType, largest: Bool = True]() -> Scalar[T]:
    @parameter
    if largest:
        return min_or_neg_inf[T]()
    else:
        return max_or_inf[T]()


# Define the TopK_2 structure to keep track of the top element per thread
@fieldwise_init
@register_passable("trivial")
struct TopK_2[T: DType, largest: Bool = True](Copyable, Defaultable, Movable):
    var p: Int  # flattened index of the element
    var u: Scalar[T]  # value of the element

    fn __init__(out self):
        self.p = -1
        self.u = _topk_dead_val[T, largest]()

    fn insert(mut self, elem: Scalar[T], elem_id: Int):
        @parameter
        if largest:
            if elem > self.u:
                self.u = elem
                self.p = elem_id
        else:
            if elem < self.u:
                self.u = elem
                self.p = elem_id


# Function to perform warp-level reduction to find the maximum TopK_2
@always_inline
@parameter
fn _warp_reduce_topk[
    T: DType, largest: Bool
](val: TopK_2[T, largest]) -> TopK_2[T, largest]:
    """
    Performs warp-level reduction to find the maximum TopK_2 element.
    Uses shuffle down operations to efficiently compute the warp-wide
    maximum of TopK_2 values across all threads in a warp.

    Parameters:
        T: DType - Data type of the values being compared.
        largest: Bool - Whether to find the maximum or minimum value.

    Arguments:
        val: TopK_2[T, largest] - TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T, largest] - Maximum TopK_2 value across the warp.
    """
    var res = val

    # Shuffle down function for TopK_2 structure
    @parameter
    fn shuffle_down_topk2(
        v: TopK_2[T, largest], offset: Int
    ) -> TopK_2[T, largest]:
        return TopK_2[T, largest](
            u=warp.shuffle_down(v.u, offset),  # u is the value
            p=Int(warp.shuffle_down(Int32(v.p), offset)),  # p is the index
        )

    @parameter
    fn reduce_fn(
        a: TopK_2[T, largest], b: TopK_2[T, largest]
    ) -> TopK_2[T, largest]:
        @parameter
        if largest:
            return a if a.u > b.u else b
        else:
            return a if a.u < b.u else b

    # Reimplement `warp_reduce` for TopK_2 reduce and shuffle function
    alias limit = log2_floor(WARP_SIZE)

    @parameter
    for i in reversed(range(limit)):
        alias mask = 1 << i
        res = reduce_fn(res, shuffle_down_topk2(res, mask))

    return res


# Function to perform block-level reduction to find the maximum TopK_2
@always_inline
fn _block_reduce_topk[
    T: DType, largest: Bool
](val: TopK_2[T, largest]) -> TopK_2[T, largest]:
    """
    Performs a block-level reduction to find the maximum TopK_2 element.

    This function takes a TopK_2 value from each thread in a block and performs
    a reduction to find the maximum across all threads. It uses shared memory
    and warp-level reductions to efficiently compute the block-wide maximum.

    Parameters:
        T: DType - The data dtype of the values being compared.
        largest: Bool - Whether to find the maximum or minimum value.

    Arguments:
        val: TopK_2[T, largest] - The TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T, largest] - The maximum TopK_2 value across all threads in the block.

    Note:
    This function assumes that BLOCK_SIZE is a multiple of WARP_SIZE.
    It uses shared memory to store intermediate results and performs
    a final warp-level reduction to compute the block-wide maximum.
    """
    alias MAX_BLOCK_SIZE = 1024
    constrained[
        MAX_BLOCK_SIZE % WARP_SIZE == 0,
        "block size must be a multiple of the warp size",
    ]()

    # Calculate sizes for shared memory allocation
    alias p_width = simdwidthof[DType.index]()
    alias u_width = simdwidthof[Scalar[T]]()

    # Allocate shared memory for indices and values
    var p_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * p_width,
        Scalar[DType.index],
        address_space = AddressSpace.SHARED,
    ]()
    var u_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * u_width,
        Scalar[T],
        address_space = AddressSpace.SHARED,
    ]()

    # Calculate warp id and thread information
    var warp = warp_id()
    alias num_warps_needed = MAX_BLOCK_SIZE // WARP_SIZE

    # Each warp reduces its own TopK_2 value
    var warp_accum: TopK_2[T, largest] = _warp_reduce_topk[T, largest](val)

    # Store warp-level results in shared memory
    if lane_id() == 0 and warp < num_warps_needed:
        # Note: Potential bank conflict for sub 4 byte data elements
        p_sram[Int(warp) * p_width] = Scalar[DType.index](warp_accum.p)
        u_sram[Int(warp) * u_width] = warp_accum.u
    barrier()

    # Load warp results into final warp for block-level reduction
    var block_accum = TopK_2[T, largest]()
    var thread_in_final_warp = thread_idx.x < (block_dim.x // WARP_SIZE)
    if thread_in_final_warp:
        var p_idx = p_sram[lane_id() * p_width]  # loaded value is a scalar
        block_accum = TopK_2[T, largest](
            p=Int(p_idx), u=u_sram[lane_id() * u_width]  # Convert back to int
        )
    else:
        # Initialize unused threads with dummy values
        block_accum.p = -1
        block_accum.u = _topk_dead_val[T, largest]()

    # Perform final warp-level reduction for block result
    return _warp_reduce_topk[T, largest](block_accum)


fn _topk_stage1[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: UnsafePointer[Scalar[DType.int64]],
    max_k: Int,
    num_elements: Int,
    num_blocks_per_input: Int,
    in_buffer: UnsafePointer[Scalar[T]],
    local_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # Output buffer of size num_blocks_per_input * max_k
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type]
    ],  # Output buffer of size num_blocks_per_input * max_k
):
    """
    Computes the Top-K elements within each block.

    This kernel function is the first stage of a two-stage Top-K algorithm.
    Each thread block processes a portion of the input data and finds its local top-K elements.
    The local top-K results are stored in global memory for further processing in stage 2.

    Parameters:
        T: Data type of the elements.
        out_idx_type: DType - The data dtype of the output indices.
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        K: Number of top elements to select per block. Varies for each batch element.
        max_k: Largest number of top elements to keep for each batch element.
        num_elements: Size of last dimension of input buffer (vocab size).
        num_blocks_per_input: Number of blocks used to process the input data.
        in_buffer: Input buffer containing the elements to process.
        local_topk_vals: Output buffer to store the local top-K values.
        local_topk_idxs: Output buffer to store the indices of local top-K elements.

    Note:
        The output buffers (local_topk_vals and local_topk_idxs) should be of size num_blocks_per_input * max_k.
    """
    tid = thread_idx.x
    bid = block_idx.x
    block_size = block_dim.x

    batch_id = bid // num_blocks_per_input
    block_lane = bid % num_blocks_per_input

    _in_buffer = in_buffer + batch_id * num_elements

    # # Allocate shared memory for the values and indices
    var topk_sram = external_memory[
        TopK_2[T, largest],
        address_space = AddressSpace.SHARED,
        alignment = alignof[TopK_2[T, largest]](),
    ]()

    with PDL():
        # Pack the topk_vals and topk_idxs into shared memory
        var block_offset = block_lane * block_size
        var stride = block_size * num_blocks_per_input
        topk_sram[tid] = TopK_2[T, largest]()
        for i in range(tid + block_offset, num_elements, stride):
            topk_sram[tid].insert(_in_buffer[i], i)

        barrier()
        var k_batch = max_k
        if K:
            k_batch = Int(K[batch_id])
        # Prepare for K iterations to find the local top-K elements
        for k in range(k_batch):
            # Initialize each thread with its own TopK_2 value and index
            var partial = topk_sram[tid]

            # Perform block-level reduction to find the maximum TopK_2
            var total = _block_reduce_topk[T, largest](partial)

            if tid == 0:
                # Store the local top-K values and indices in global memory
                var vector_idx = total.p
                local_topk_vals[bid * max_k + k] = total.u
                local_topk_idxs[bid * max_k + k] = Scalar[DType.index](
                    vector_idx
                ).cast[out_idx_type]()

                # Remove the found maximum from consideration in the next iteration
                var orig_tid = (vector_idx - block_offset) % stride
                topk_sram[orig_tid].u = _topk_dead_val[T, largest]()

            barrier()

        # Fill remaining positions with sentinel values for unused elements
        if tid == 0:
            for remaining_k in range(k_batch, max_k):
                local_topk_vals[bid * max_k + remaining_k] = _topk_dead_val[
                    T, largest
                ]()
                local_topk_idxs[bid * max_k + remaining_k] = Scalar[
                    out_idx_type
                ](-1)


@always_inline("nodebug")
fn _get_shmem_size_stg_1[dtype: DType](block_size: Int) -> Int:
    # Get dynamic shared memory size for stage 1
    return Int(block_size * sizeof[TopK_2[dtype]]())


fn _topk_stage2[
    T: DType,
    out_idx_type: DType,
    sampling: Bool = True,
    largest: Bool = True,
](
    K: UnsafePointer[Scalar[DType.int64]],
    max_k: Int,
    num_blocks_per_input: Int,
    local_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # Input array of size n_batch * num_blocks_per_input * K
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type]
    ],  # Input array of size n_batch * num_blocks_per_input * K
    global_topk_vals: UnsafePointer[
        Scalar[T]
    ],  # sampling ? undefined : output array of size K
    global_topk_idxs: UnsafePointer[
        Scalar[out_idx_type]
    ],  # sampling ? sampled token : Output array of size K
    temperature: UnsafePointer[Scalar[DType.float32]],
    top_p: UnsafePointer[Scalar[DType.float32]],
    seed: UnsafePointer[Scalar[DType.uint64]],
):
    """
    Computes the global Top-K elements from the local Top-K results produced by stage 1.

    This kernel is designed to be executed with a single block, performing the final
    reduction step to obtain the global Top-K elements.

    Parameters:
        T: Data type of the elements.
        out_idx_type: DType - The data dtype of the output indices.
        sampling: Bool - Whether to sample a token from the top-K distribution.
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        K: Number of top elements to select per batch element.
        max_k: Largest number of top elements to keep for each batch element.
        num_blocks_per_input: Number of blocks used in stage 1.
        local_topk_vals: Pointer to local Top-K values from stage 1 (size: batch_size * num_blocks_per_input * K).
        local_topk_idxs: Pointer to local Top-K indices from stage 1 (size: batch_size * num_blocks_per_input * K).
        global_topk_vals: Pointer to store the final global Top-K values (size: batch_size * K).
        global_topk_idxs: Pointer to store the final global Top-K indices (size: batch_size * (1 if sampling else K)).
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.

    The function uses shared memory to store and process the local Top-K results,
    and performs a block-level reduction to find the global Top-K elements.
    """
    # compute the total number of elements reduced from stage 1
    var num_elem_reduced = num_blocks_per_input * max_k

    var tid = thread_idx.x
    var batch_id = block_idx.x
    # assert (block_idx.x == 0)
    # assert (grid_dim.x == 1)
    var batch_i_topk_vals = global_topk_vals + batch_id * max_k
    var batch_i_topk_idxs = global_topk_idxs + batch_id * (
        1 if sampling else max_k
    )
    var _local_topk_vals = local_topk_vals + batch_id * num_elem_reduced
    var _local_topk_idxs = local_topk_idxs + batch_id * num_elem_reduced

    # Allocate shared memory for values and indices
    var num_e_rounded = ceildiv(num_elem_reduced, WARP_SIZE) * WARP_SIZE
    var vals_smem_size = num_e_rounded
    var vals_sram = external_memory[
        Scalar[T],
        address_space = AddressSpace.SHARED,
        alignment = alignof[Scalar[T]](),
    ]()
    var idxs_sram = (vals_sram + vals_smem_size).bitcast[Int]()

    # These values are only read from in the sampling case.
    var s_val2 = UnsafePointer[Scalar[T], address_space = AddressSpace.SHARED]()
    var s_id = UnsafePointer[Int, address_space = AddressSpace.SHARED]()

    with PDL():
        # Handle the case where stage 1 is executed with a single block
        var k_batch = max_k
        if K:
            k_batch = Int(K[batch_id])
        if num_blocks_per_input == 1 and not sampling:
            if tid < k_batch:
                batch_i_topk_vals[tid] = _local_topk_vals[tid]
                # cast to out_idx_type
                batch_i_topk_idxs[tid] = _local_topk_idxs[tid]
            elif tid >= k_batch and tid < max_k:
                # Fill unused positions with sentinel values
                batch_i_topk_vals[tid] = _topk_dead_val[T, largest]()
                batch_i_topk_idxs[tid] = Scalar[out_idx_type](-1)
            return

        @parameter
        if sampling:
            # Storing the top-K logits in shmem for sampling
            s_id = (idxs_sram + vals_smem_size).bitcast[Int]()
            # The 2* below is for warp align safety
            s_val2 = (s_id + 2 * k_batch).bitcast[Scalar[T]]()

        var s_sum = stack_allocation[
            1, Scalar[T], address_space = AddressSpace.SHARED
        ]()
        s_sum[0] = Scalar[T](0)
        var max_logit = Scalar[T](0)

        # Cache local top-K results from stage 1 into shared memory
        for i in range(tid, num_elem_reduced, block_dim.x):
            vals_sram[i] = _local_topk_vals[i]
            idxs_sram[i] = i
        barrier()

        for k in range(max_k):
            if k >= k_batch:
                # Fill remaining positions with sentinel values for unused elements
                @parameter
                if not sampling:
                    if tid == 0:
                        for remaining_k in range(k, max_k):
                            batch_i_topk_vals[remaining_k] = _topk_dead_val[
                                T, largest
                            ]()
                            batch_i_topk_idxs[remaining_k] = Scalar[
                                out_idx_type
                            ](-1)
                break

            # Re-initialize partial for each thread
            var partial = TopK_2[T, largest]()
            # TODO: unroll this
            for i in range(tid, num_elem_reduced, block_dim.x):
                partial.insert(vals_sram[i], i)

            barrier()
            # Perform block-level reduction to find the maximum TopK_2
            var total: TopK_2[T, largest] = _block_reduce_topk[T, largest](
                partial
            )

            if tid == 0:

                @parameter
                if sampling:
                    if k == 0:
                        max_logit = total.u

                # Remove the found maximum from consideration in the next iteration
                idxs_sram[total.p] = -1
                vals_sram[total.p] = _topk_dead_val[T, largest]()

                @parameter
                if sampling:
                    batch_i_topk_vals[k] = total.u
                    s_id[k] = total.p
                    var temp_val = Scalar[DType.float32](1.0)
                    if temperature:
                        temp_val = temperature[batch_id]
                    total.u = exp(
                        (total.u - max_logit) / max(temp_val.cast[T](), 1e-6)
                    )
                    s_val2[k] = total.u
                    s_sum[0] += total.u
                else:
                    # Store the global top-K values and indices
                    batch_i_topk_vals[k] = total.u
                    batch_i_topk_idxs[k] = _local_topk_idxs[total.p]

                # Early exit if no valid index
                if total.p == -1:
                    break
            barrier()

        # do sampling
        @parameter
        if sampling:
            if tid == 0:
                var top_p_val = Scalar[T](1.0)
                if top_p:
                    top_p_val = top_p[batch_id].cast[T]()
                var _top_p = _adjust_top_p[T](
                    top_p_val, s_val2, k_batch, s_sum[0]
                )

                # Use the largest logit's id as the offset for the random number
                # generator, so that we don't use the same random number for every
                # token in the sequence.
                var seed_val = UInt64(0)
                if seed:
                    seed_val = seed[batch_id]
                var rng_state = Random(
                    seed=seed_val,
                    offset=_local_topk_idxs[0].cast[DType.uint64](),
                )
                var rng = rng_state.step_uniform()
                var softmax_norm = s_sum[0]
                var r = softmax_norm * _top_p * rng[0].cast[T]()
                for ki in range(k_batch):
                    var exp_logit = s_val2[ki]

                    r -= exp_logit
                    if r <= 0.0 or ki == k_batch - 1:
                        # uncomment below to return prob of largest logit
                        # batch_i_topk_vals[0] = exp_logit / softmax_norm
                        var idx: Int = s_id[ki]
                        batch_i_topk_idxs[0] = _local_topk_idxs[idx]
                        break


fn _topk_gpu[
    dtype: DType,
    rank: Int,
    out_idx_type: DType, //,
    sampling: Bool = True,
    largest: Bool = True,
](
    ctx: DeviceContext,
    max_k: Int,
    input_buf: NDBuffer[dtype, rank],
    device_local_topk_vals: NDBuffer[dtype, rank],
    device_local_topk_idxs: NDBuffer[out_idx_type, rank],
    out_vals: NDBuffer[mut=True, dtype, rank],
    out_idxs: NDBuffer[mut=True, out_idx_type, rank],
    k: OptionalReg[NDBuffer[DType.int64, 1, MutableAnyOrigin]] = None,
    temperature: OptionalReg[
        NDBuffer[DType.float32, 1, MutableAnyOrigin]
    ] = None,
    block_size: Int = 256,
    num_blocks_per_input: OptionalReg[Int] = None,
    top_p: OptionalReg[NDBuffer[DType.float32, 1, MutableAnyOrigin]] = None,
    seed: OptionalReg[NDBuffer[DType.uint64, 1, MutableAnyOrigin]] = None,
) raises:
    """Computes the Top-K elements from the input tensor using a GPU-accelerated two-stage algorithm.

    This function implements a two-stage Top-K algorithm:
    1. Stage 1 (_topk_stage1): Divides the input into blocks and computes local Top-K for each block.
    2. Stage 2 (_topk_stage2): Merges the local Top-K results to obtain the global Top-K.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        rank: Int - The rank of the input tensor.
        out_idx_type: DType - The data dtype of the output indices (default is DType.index).
        sampling: Bool - Whether to return token samples from topK dist (default is True).
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        max_k: Int
            Largest number of top elements to keep for each batch element.
        input_buf: NDBuffer[dtype, rank, DimList(batch_size,N)]
            Input tensor as a device NDBuffer.
        device_local_topk_vals: NDBuffer[dtype, 2, DimList(batch_size, num_blocks_per_input * max(K))]
            Temporary buffer for locally reduced top-K values from stage 1.
        device_local_topk_idxs: NDBuffer[DType.index, 2, DimList(batch_size, num_blocks_per_input * max(K))]
            Temporary buffer for locally reduced top-K indices from stage 1.
        out_vals: NDBuffer[dtype, 2, DimList(batch_size, max(K))]
            Output buffer on device for the K largest values.
        out_idxs: NDBuffer[DType.index, 2, DimList(batch_size, 1 if sampling else max(K))]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
        k: Optional NDBuffer[DType.int64, 1]]
            Device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling for each batch element.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: OptionalReg[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.

    The implementation uses shared memory and warp-level primitives for efficient GPU execution.
    It's modeled from the following similar algos in [InternLM]
    (https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/sampling_topk_kernels.cu)
    and [TRT-LLM]
    (https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopKKernels.cu).

    """
    constrained[rank == 2, "rank must be 2"]()
    constrained[
        not (sampling and not largest),
        "sampling not supported for largest=False",
    ]()
    # Use largest number of threads per block
    var batch_size = input_buf.get_shape()[0] if rank == 2 else 1
    var N = input_buf.get_shape()[1]
    # Define the number of blocks per grid
    var num_blocks_per_input_: Int = ceildiv(
        N, block_size
    ) if not num_blocks_per_input else num_blocks_per_input.value()
    # Calculate largest num bytes of shmem for each stage
    if block_size % WARP_SIZE != 0:
        # TODO: Need to pad in this case
        raise Error("block_size must be a multiple of WARP_SIZE")

    var shared_mem_bytes_1 = _get_shmem_size_stg_1[dtype](block_size)

    # Define grid and block dimensions for stage 1
    var grid_dim_stage1 = Dim(num_blocks_per_input_ * batch_size)
    var block_dim_stage1 = Dim(block_size)

    # Handle optional k parameter
    var k_ptr: UnsafePointer[Scalar[DType.int64]]
    if k:
        k_ptr = rebind[UnsafePointer[Scalar[DType.int64]]](k.value().data)
    else:
        k_ptr = UnsafePointer[Scalar[DType.int64]]()  # null pointer

    # Enqueue the first kernel (stage 1)
    ctx.enqueue_function[_topk_stage1[dtype, out_idx_type, largest]](
        k_ptr,
        max_k,
        N,
        num_blocks_per_input_,
        input_buf.data,
        device_local_topk_vals.data,
        device_local_topk_idxs.data,
        grid_dim=grid_dim_stage1,
        block_dim=block_dim_stage1,
        shared_mem_bytes=shared_mem_bytes_1,
        attributes=pdl_launch_attributes(),
    )

    var num_elem_reduced = (
        ceildiv(num_blocks_per_input_ * max_k, WARP_SIZE) * WARP_SIZE
    )
    var num_bytes_sample_cache = max_k * (
        sizeof[Scalar[dtype]]() + 2 * sizeof[DType.index]()
    )
    var shared_mem_bytes_2 = (
        num_elem_reduced * (sizeof[Scalar[dtype]]() + sizeof[DType.index]())
        + num_bytes_sample_cache
    )
    shared_mem_bytes_2 = Int(
        ceildiv(shared_mem_bytes_2, WARP_SIZE) * WARP_SIZE
    )  # align to warp size

    # Define grid and block dimensions for stage 2
    var grid_dim_stage2 = Dim(
        batch_size
    )  # Single block since num_elements_stage2 is small
    var block_dim_stage2 = Dim(block_size)

    # Handle optional temperature parameter
    var temp_ptr: UnsafePointer[Scalar[DType.float32]]
    if temperature:
        temp_ptr = rebind[UnsafePointer[Scalar[DType.float32]]](
            temperature.value().data
        )
    else:
        temp_ptr = UnsafePointer[Scalar[DType.float32]]()  # null pointer

    # Handle optional top_p parameter
    var top_p_ptr: UnsafePointer[Scalar[DType.float32]]
    if top_p:
        top_p_ptr = rebind[UnsafePointer[Scalar[DType.float32]]](
            top_p.value().data
        )
    else:
        top_p_ptr = UnsafePointer[Scalar[DType.float32]]()  # null pointer

    # Handle optional seed parameter
    var seed_ptr: UnsafePointer[Scalar[DType.uint64]]
    if seed:
        seed_ptr = rebind[UnsafePointer[Scalar[DType.uint64]]](
            seed.value().data
        )
    else:
        seed_ptr = UnsafePointer[Scalar[DType.uint64]]()  # null pointer

    # Enqueue the second kernel (stage 2)
    ctx.enqueue_function[_topk_stage2[dtype, out_idx_type, sampling, largest]](
        k_ptr,
        max_k,
        num_blocks_per_input_,
        device_local_topk_vals.data,
        device_local_topk_idxs.data,
        out_vals.data,
        out_idxs.data,
        temp_ptr,
        top_p_ptr,
        seed_ptr,
        grid_dim=grid_dim_stage2,
        block_dim=block_dim_stage2,
        shared_mem_bytes=shared_mem_bytes_2,
        attributes=pdl_launch_attributes(),
    )


@always_inline
fn topk_gpu[
    dtype: DType,
    rank: Int,
    out_idx_type: DType, //,
    sampling: Bool = True,
    largest: Bool = True,
](
    ctx: DeviceContext,
    max_k: Int,
    input: NDBuffer[dtype, rank],
    out_vals: NDBuffer[mut=True, dtype, rank],
    out_idxs: NDBuffer[mut=True, out_idx_type, rank],
    block_size: OptionalReg[Int] = None,
    num_blocks_per_input: OptionalReg[Int] = None,
    k: OptionalReg[NDBuffer[DType.int64, 1, MutableAnyOrigin]] = None,
    temperature: OptionalReg[
        NDBuffer[DType.float32, 1, MutableAnyOrigin]
    ] = None,
    top_p: OptionalReg[NDBuffer[DType.float32, 1, MutableAnyOrigin]] = None,
    seed: OptionalReg[NDBuffer[DType.uint64, 1, MutableAnyOrigin]] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with/without sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume or the top K values and indices across the tensor.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        rank: Int - The rank of the input tensor.
        out_idx_type: DType - The data dtype of the output indices (default is DType.index).
        sampling: Bool - Whether to return token samples from topK dist (default is True).
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        max_k: Int
            Largest number of top elements to keep for each batch element.
        input: NDBuffer[dtype, rank]
            Input tensor as a device NDBuffer.
        out_vals: NDBuffer[dtype, rank]
            Output buffer on device for the K largest values.
        out_idxs: NDBuffer[DType.index, rank]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
            Last dimension is 1 if sampling is True, otherwise K.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: OptionalReg[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.
        k: Optional NDBuffer[DType.int64, 1, MutableAnyOrigin]
            Device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.
    """
    constrained[rank > 0, "Input rank must be positive"]()
    var orig_in_shape: IndexList[rank] = input.get_shape()
    var N = orig_in_shape[rank - 1]
    var last_idx_dim = 1 if sampling else max_k

    # heuristic to set block size
    var block_size_: Int
    if input.size() <= 1024 * 64 * 3:
        block_size_ = 256
    elif input.size() <= 32000 * 256:
        block_size_ = 512
    else:
        block_size_ = 1024
    block_size_ = block_size.value() if block_size else block_size_

    # This section handles different input ranks by reshaping to a 2D tensor
    var internal_bs: Int  # Internal batch size
    alias internal_rank = 2  # We always reshape to 2D for internal processing
    var internal_input: NDBuffer[dtype, internal_rank, MutableAnyOrigin]
    var internal_out_idxs: NDBuffer[
        out_idx_type, internal_rank, MutableAnyOrigin
    ]
    var internal_out_vals: NDBuffer[dtype, internal_rank, MutableAnyOrigin]

    @parameter
    if rank == 1:
        # Handle 1D input: treat it as a single batch with one element
        internal_bs = 1
        var internal_in_shape = IndexList[internal_rank](1, input.size())
        var internal_out_vals_shape = IndexList[internal_rank](1, max_k)
        var internal_out_idxs_shape = IndexList[internal_rank](1, last_idx_dim)
        # Reshape 1D inputs to 2D
        internal_input = reshape(input, internal_in_shape)
        internal_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
        internal_out_vals = reshape(out_vals, internal_out_vals_shape)
    elif rank == internal_rank:
        # Input is already 2D, no reshaping needed
        internal_bs = orig_in_shape[0]
        internal_input = rebind[NDBuffer[dtype, internal_rank, input.origin]](
            input
        )
        internal_out_idxs = rebind[
            NDBuffer[out_idx_type, internal_rank, out_idxs.origin]
        ](out_idxs)
        internal_out_vals = rebind[
            NDBuffer[dtype, internal_rank, out_vals.origin]
        ](out_vals)
    else:  # rank > 2
        # Handle higher dimensional inputs by flattening all but the last dimension
        var _last_dim = orig_in_shape[rank - 1]
        internal_bs = Int(orig_in_shape.flattened_length() / _last_dim)

        var internal_in_shape = IndexList[internal_rank](internal_bs, _last_dim)
        var internal_out_idxs_shape = IndexList[internal_rank](
            internal_bs, last_idx_dim
        )
        var internal_out_vals_shape = IndexList[internal_rank](
            internal_bs, max_k
        )

        # Reshape higher dimensional inputs to 2D
        internal_input = reshape(input, internal_in_shape)
        internal_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
        internal_out_vals = reshape(out_vals, internal_out_vals_shape)

    # Calculate the number of blocks per input
    var num_blocks_per_input_ = min(
        ceildiv(N, block_size_), 8
    ) if not num_blocks_per_input else num_blocks_per_input.value()

    # Define shape for the kernel's internal cache buffers
    var internal_cache_shape = DimList(
        internal_bs, num_blocks_per_input_ * max_k
    )

    # Create temporary buffer for local top-K values
    var internal_vals_buf = ctx.enqueue_create_buffer[dtype](
        Int(internal_cache_shape.product())
    )
    var device_local_topk_vals = NDBuffer[dtype, internal_rank](
        internal_vals_buf._unsafe_ptr(), internal_cache_shape
    )

    # Create temporary buffer for local top-K indices
    var internal_idxs_buf = ctx.enqueue_create_buffer[out_idx_type](
        Int(internal_cache_shape.product())
    )
    var device_local_topk_idxs = NDBuffer[out_idx_type, internal_rank](
        internal_idxs_buf._unsafe_ptr(), internal_cache_shape
    )

    _topk_gpu[
        dtype=dtype,
        rank=internal_rank,
        out_idx_type=out_idx_type,
        sampling=sampling,
        largest=largest,
    ](
        ctx,
        max_k,
        internal_input,
        device_local_topk_vals,
        device_local_topk_idxs,
        internal_out_vals,
        internal_out_idxs,
        k=k,
        temperature=temperature,
        block_size=block_size_,
        num_blocks_per_input=num_blocks_per_input_,
        top_p=top_p,
        seed=seed,
    )

    # Clean up buffers
    _ = internal_vals_buf^
    _ = internal_idxs_buf^


@always_inline
fn fused_token_sampling_gpu[
    dtype: DType,
    rank: Int,
    out_idx_type: DType, //,
](
    ctx: DeviceContext,
    max_k: Int,
    input: NDBuffer[dtype, rank],
    out_idxs: NDBuffer[mut=True, out_idx_type, rank],
    block_size: OptionalReg[Int] = None,
    num_blocks_per_input: OptionalReg[Int] = None,
    k: OptionalReg[NDBuffer[DType.int64, 1, MutableAnyOrigin]] = None,
    temperature: OptionalReg[
        NDBuffer[DType.float32, 1, MutableAnyOrigin]
    ] = None,
    top_p: OptionalReg[NDBuffer[DType.float32, 1, MutableAnyOrigin]] = None,
    seed: OptionalReg[NDBuffer[DType.uint64, 1, MutableAnyOrigin]] = None,
) raises:
    """
    Top K algorithm with fused sampling.
    Returns the sampled indices from the Top-K of the innermost
    dimension of the input tensor for each row/subvolume.
    """
    var out_vals_shape = input.get_shape()
    out_vals_shape[rank - 1] = max_k
    var out_vals_buf = ctx.enqueue_create_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var out_vals = NDBuffer[dtype, rank](
        out_vals_buf._unsafe_ptr(), out_vals_shape
    )

    topk_gpu[sampling=True, largest=True](
        ctx,
        max_k,
        input,
        out_vals,
        out_idxs,
        k=k,
        temperature=temperature,
        top_p=top_p,
        block_size=block_size,
        num_blocks_per_input=num_blocks_per_input,
        seed=seed,
    )

    _ = out_vals_buf^
