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

from math import ceildiv, exp, iota
from memory import alloc
from sys import align_of, simd_width_of, size_of, env_get_bool

import gpu.primitives.warp as warp
from algorithm.functional import parallelize_over_rows
from algorithm.reduction import _get_nd_indices_from_flat_index
from bit import log2_floor
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
from gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext, DeviceBuffer
from gpu.host.info import is_cpu
from gpu.memory import AddressSpace, external_memory
from random import Random
from layout._coord import (
    ComptimeInt,
    Coord,
    CoordLike,
    DynamicCoord,
    Idx,
    RuntimeInt,
    coord_to_index_list,
)
from layout._layout import TensorLayout, Layout, RowMajorLayout, row_major
from layout._tile_tensor import TileTensor
from math import log2
from memory import stack_allocation
from nn.gather_scatter import normalize_neg_index
from nn.reshape import reshape
from os.env import getenv
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList, StaticTuple, product
from utils.numerics import max_or_inf, min_or_neg_inf


@always_inline
fn top_k_shape_impl[
    dtype: DType,
    single_thread_blocking_override: Bool,
](input: TileTensor[dtype, ...], max_k: Int, axis: Int) raises -> IndexList[
    input.rank
]:
    """
    Compute the output shape of a top/bottom k operation.

    Parameters:
        dtype: Data type of the input buffer.
        single_thread_blocking_override: If this function can block.

    Args:
        input: The input tensor.
        max_k: The maximum K value.
        axis: The axis value in a tensor.

    Returns:
        The output shape.
    """

    # Clamp max_k
    var bound_max_k = Int(input.dim(axis)) if max_k == -1 else max_k

    if bound_max_k < 0 or bound_max_k > Int(input.dim(axis)):
        raise Error("[top/bottom-k] k must be within [0, input_shape[axis]]")

    var shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )
    shape[normalize_neg_index(axis, input.rank)] = bound_max_k

    return shape


@always_inline
fn _adjust_top_p[
    T: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    top_p: Scalar[T],
    values: UnsafePointer[Scalar[T], address_space=address_space],
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
    dtype: DType,
    out_idx_type: DType,
    //,
    largest: Bool = True,
    target: StaticString = "cpu",
](
    input: TileTensor[dtype, ...],
    max_k: Int,
    axis: Int,
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    sorted: Bool,
    ctx: DeviceContextPtr,
    k: Optional[
        TileTensor[
            DType.int64,
            RowMajorLayout[RuntimeInt[DType.int64]],
            ImmutAnyOrigin,
        ],
    ] = None,
) raises:
    """
    Implementation of the Top K algorithm. Returns the top or bottom K elements
    and their index along a specified axis.

    Parameters:
        dtype: Data type of the input buffer.
        out_idx_type: The data dtype of the output indices (default == DType.int64).
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
    comptime assert (
        input.rank == out_vals.rank
    ), "input.rank must match out_vals.rank"
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"

    var normalized_axis = normalize_neg_index(Int64(axis), input.rank)

    # Clamp max_k
    var bound_max_k = 255 if max_k == -1 else max_k

    comptime if is_cpu[target]():
        comptime assert (
            out_idx_type == DType.int64
        ), "out_idx_type must be int64 for cpu"

        comptime grain_size = 1000
        _top_k_cpu[largest=largest](
            input,
            bound_max_k,
            Int(normalized_axis),
            out_vals,
            out_idxs,
            grain_size,
            sorted,
            k=k,
        )
    else:
        if normalized_axis != Scalar[DType.int](input.rank - 1):
            raise Error("axis other than -1 not supported on GPU")
        if not sorted:
            print(
                "Warning: Unsorted top-k is not supported on GPU. Falling"
                " back to sorted top-k."
            )
        var cuda_ctx = ctx.get_device_context()
        topk_gpu[sampling=False, largest=largest](
            cuda_ctx,
            bound_max_k,
            input,
            out_vals,
            out_idxs,
            k=k,
        )


fn _top_k_cpu[
    dtype: DType,
    out_idx_type: DType,
    largest: Bool,
    KLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
](
    input: TileTensor[dtype, ...],
    max_k: Int,
    axis: Int,
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    parallelism_grain_size: Int,  # impl detail, exposed for testing
    sorted: Bool,
    k: Optional[TileTensor[DType.int64, KLayoutType, ImmutAnyOrigin]] = None,
):
    comptime assert (
        input.rank == out_vals.rank
    ), "input.rank must match out_vals.rank"
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"
    comptime assert k.T.flat_rank == 1
    var shape = coord_to_index_list(input.layout.shape_coord())

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
                var k_raw = Int(k.value()[batch_idx])
                k_val = max_k if k_raw == -1 else k_raw

            # Clamp k to the size of the axis to avoid out-of-bounds access
            if k_val > shape[axis]:
                k_val = shape[axis]

            @parameter
            @always_inline
            fn indices_to_val(idx: Int64) -> Scalar[dtype]:
                indices[axis] = Int(idx)
                var input_idx = input.layout(Coord(indices))
                return input.ptr[input_idx]

            comptime if largest:

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
                    var input_idx = input.layout(Coord(indices))
                    var curr = input.ptr[input_idx]
                    var num_equal = 1
                    for j in range(i + 1, shape[axis]):
                        indices[axis] = Int(idxs[j])
                        var input_idx = input.layout(Coord(indices))
                        var next = input.ptr[input_idx]
                        if curr != next:
                            break
                        num_equal += 1
                    if num_equal > 1:
                        var ptr = idxs.unsafe_ptr() + i
                        sort(
                            Span[idxs.T, origin_of(idxs)](
                                ptr=ptr, length=num_equal
                            )
                        )
                    i += num_equal

            for i in range(k_val):
                indices[axis] = Int(idxs[i])
                var input_idx = input.layout(Coord(indices))
                var val = input.ptr[input_idx]
                indices[axis] = i
                var out_vals_idx = out_vals.layout(Coord(indices))
                var out_idxs_idx = out_idxs.layout(Coord(indices))
                out_vals.ptr[out_vals_idx] = val
                out_idxs.ptr[out_idxs_idx] = rebind[Scalar[out_idx_type]](
                    idxs[i]
                )

    parallelize_over_rows[process_rows](shape, axis, parallelism_grain_size)


@always_inline
fn fused_token_sampling_cpu[
    dtype: DType,
    out_idx_type: DType,
    KLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64]
    ],
    TopPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    SeedLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
](
    max_k: Int,
    input: TileTensor[dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    k: Optional[TileTensor[DType.int64, KLayoutType, ImmutAnyOrigin]] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
    top_p: Optional[
        TileTensor[DType.float32, TopPLayoutType, ImmutAnyOrigin]
    ] = None,
    seed: Optional[
        TileTensor[DType.uint64, SeedLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume.

    Parameters:
        dtype: Data type of the input buffer.
        out_idx_type: Data type of the output indices.
        KLayoutType: Layout type of the k buffer.
        TemperatureLayoutType: Layout type of the temperature buffer.
        TopPLayoutType: Layout type of the top_p buffer.
        SeedLayoutType: Layout type of the seed buffer.

    Args:
        max_k: Largest number of top elements.
        input: NDBuffer[dtype, rank] (Any shape)- The input tensor.
        out_idxs: NDBuffer[out_idx_type, rank] (shape of [input_shape[:-1]] + [1]) - The output indices.
        k: Optional device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.
    """
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"
    comptime assert out_idx_type == DType.int64, "out_idx_type must be int64"

    bound_max_k = 255 if max_k == -1 else max_k

    # materialize the out_vals which is of shape [input[:-1]] + [k]
    var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
    out_vals_shape[input.rank - 1] = bound_max_k
    var out_vals = TileTensor(
        alloc[Scalar[dtype]](out_vals_shape.flattened_length()),
        row_major(Coord(out_vals_shape)),
    )

    _top_k_sampling(
        bound_max_k,
        input,
        out_vals,
        TileTensor(out_idxs.ptr.bitcast[Int64](), out_idxs.layout),
        k,
        temperature,
        top_p,
        seed,
    )

    out_vals.ptr.free()


fn _top_k_sampling[
    dtype: DType,
    KLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64]
    ],
    TopPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    SeedLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
](
    max_k: Int,
    input: TileTensor[dtype, ...],
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, DType.int64, ...],
    k: Optional[TileTensor[DType.int64, KLayoutType, ImmutAnyOrigin]] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
    top_p: Optional[
        TileTensor[DType.float32, TopPLayoutType, ImmutAnyOrigin]
    ] = None,
    seed: Optional[
        TileTensor[DType.uint64, SeedLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume.

    Parameters:
        dtype: Data type of the input buffer.
        KLayoutType: Layout type of the k buffer.
        TemperatureLayoutType: Layout type of the temperature buffer.
        TopPLayoutType: Layout type of the top_p buffer.
        SeedLayoutType: Layout type of the seed buffer.

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
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert (
        input.rank == out_vals.rank
    ), "input.rank must match out_vals.rank"
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"
    comptime assert temperature.T.flat_rank == 1
    comptime assert k.T.flat_rank == 1
    comptime assert top_p.T.flat_rank == 1
    comptime assert seed.T.flat_rank == 1

    # Now reshape for sampling
    var orig_in_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )
    var last_dim = orig_in_shape[input.rank - 1]

    comptime internal_rank = 2
    var internal_bs: Int
    var internal_in_shape: IndexList[internal_rank]

    comptime if input.rank == 1:
        internal_bs = 1
        internal_in_shape = IndexList[internal_rank](1, input.numel())
    elif input.rank == internal_rank:
        internal_bs = orig_in_shape[0]
        internal_in_shape = rebind[IndexList[internal_rank]](orig_in_shape)
    elif input.rank > internal_rank:
        internal_bs = Int(
            Float64(orig_in_shape.flattened_length()) / Float64(last_dim)
        )
        internal_in_shape = IndexList[internal_rank](internal_bs, last_dim)
    else:
        raise Error("Unsupported input rank. Must be >= 1.")

    internal_out_shape = IndexList[internal_rank](internal_bs, max_k)
    internal_out_idxs_shape = IndexList[internal_rank](internal_bs, 1)

    var reshaped_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
    var reshaped_out_vals = reshape(out_vals, internal_out_shape)

    var out_idxs_tmp = TileTensor(
        alloc[Int64](out_vals.numel()),
        row_major(Coord(internal_out_shape)),  # topk returns K as last dim
    )
    var reshaped_input = reshape(input, internal_in_shape)
    _top_k_cpu[dtype=dtype, largest=True](
        reshaped_input,
        max_k,
        axis=internal_rank - 1,  # Always operate on the last axis
        out_vals=reshaped_out_vals,
        out_idxs=out_idxs_tmp,
        sorted=True,
        parallelism_grain_size=1,
        k=k,
    )

    # Sample from the top K elements
    for batch in range(internal_bs):
        var temperature_val = Float32(1.0)
        if temperature:
            temperature_val = temperature.value()[batch][0]

        var k_val = max_k
        if k:
            var k_raw = Int(k.value()[batch])
            k_val = max_k if k_raw == -1 else k_raw

        # Clamp k_val to the number of valid top-k entries available in internal_out_vals
        var avail_k = Int(reshaped_out_vals.dim[1]())
        if k_val > avail_k:
            k_val = avail_k

        # Calculate softmax normalization
        var max_val = reshaped_out_vals[batch, 0][0]
        var sum_exp = Scalar[dtype](0)
        var exp_vals = alloc[Scalar[dtype]](k_val)
        var temp_val = temperature_val.cast[dtype]()
        for i in range(k_val):
            var val = reshaped_out_vals[batch, i][0]
            var exp_val = exp((val - max_val) / max(temp_val, 1e-6))
            exp_vals[i] = exp_val
            sum_exp += exp_val

        # Handle top_p parameter - extract scalar value from buffer
        var top_p_val = Scalar[dtype](1.0)
        if top_p:
            top_p_val = top_p.value()[batch][0].cast[dtype]()
        var _top_p = _adjust_top_p[dtype](top_p_val, exp_vals, k_val, sum_exp)

        # Handle seed parameter - extract scalar value from buffer
        var seed_val = UInt64(0)
        if seed:
            seed_val = seed.value()[batch][0]

        # Use the same RNG as the GPU sampling implementation
        var rng_state = Random(seed=seed_val)
        var rng = rng_state.step_uniform()

        # Sample using the normalized probabilities
        var r = sum_exp * _top_p * rng[0].cast[dtype]()
        for i in range(k_val):
            r -= exp_vals[i]
            if r <= 0 or i == k_val - 1:
                # Store the sampled index and value
                reshaped_out_idxs[batch, 0] = out_idxs_tmp[batch, i]
                break
        exp_vals.free()

        # Fill remaining positions with sentinel values for unused elements
        for remaining_k in range(k_val, max_k):
            if remaining_k < Int(reshaped_out_vals.dim[1]()):
                reshaped_out_vals[batch, remaining_k] = _topk_dead_val[
                    dtype, True
                ]()
            # Note: out_idxs for sampling only has 1 element in last dim, so no need to fill indices
    out_idxs_tmp.ptr.free()


@always_inline("nodebug")
fn _topk_dead_val[T: DType, largest: Bool = True]() -> Scalar[T]:
    comptime if largest:
        return min_or_neg_inf[T]()
    else:
        return max_or_inf[T]()


# Define the TopK_2 structure to keep track of the top element per thread
@fieldwise_init
struct TopK_2[T: DType, largest: Bool = True](
    Defaultable, TrivialRegisterPassable
):
    var p: Int  # flattened index of the element
    var u: Scalar[Self.T]  # value of the element

    fn __init__(out self):
        self.p = -1
        self.u = _topk_dead_val[Self.T, Self.largest]()

    fn insert(mut self, elem: Scalar[Self.T], elem_id: Int):
        comptime if Self.largest:
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
    T: DType,
    largest: Bool,
    num_lanes: Int = WARP_SIZE,
    broadcast: Bool = False,
](val: TopK_2[T, largest]) -> TopK_2[T, largest]:
    """
    Performs warp-level reduction to find the maximum TopK_2 element.
    Uses shuffle down operations to efficiently compute the warp-wide
    maximum of TopK_2 values across all threads in a warp.

    Parameters:
        T: DType - Data type of the values being compared.
        largest: Bool - Whether to find the maximum or minimum value.
        num_lanes: Int - Number of lanes that participate in the reduction.
        broadcast: Bool - Whether to broadcast the result to all lanes.

    Arguments:
        val: TopK_2[T, largest] - TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T, largest] - Maximum TopK_2 value across the warp.
    """
    comptime assert (
        num_lanes.is_power_of_two()
    ), "num_lanes must be a power of two"

    var res = val

    # Shuffle function for TopK_2 structure
    @parameter
    fn shuffle_topk2(v: TopK_2[T, largest], offset: Int) -> TopK_2[T, largest]:
        comptime fn_type = fn[dtype: DType, simd_width: Int](
            val: SIMD[dtype, simd_width], offset: UInt32
        ) -> SIMD[dtype, simd_width]
        comptime xor_fn: fn_type = warp.shuffle_xor
        comptime down_fn: fn_type = warp.shuffle_down

        comptime shuffle_fn = xor_fn if broadcast else down_fn

        return TopK_2[T, largest](
            u=shuffle_fn(v.u, UInt32(offset)),  # u is the value
            p=Int(shuffle_fn(Int32(v.p), UInt32(offset))),  # p is the index
        )

    @parameter
    fn reduce_fn(
        a: TopK_2[T, largest], b: TopK_2[T, largest]
    ) -> TopK_2[T, largest]:
        comptime if largest:
            if a.u > b.u:
                return a
            elif a.u < b.u:
                return b
            return a if a.p < b.p else b
        else:
            if a.u < b.u:
                return a
            elif a.u > b.u:
                return b
            return a if a.p < b.p else b

    # Reimplement `warp_reduce` for TopK_2 reduce and shuffle function
    comptime limit = log2_floor(num_lanes)

    comptime for i in reversed(range(limit)):
        comptime mask = 1 << i
        res = reduce_fn(res, shuffle_topk2(res, mask))

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
    comptime MAX_BLOCK_SIZE = 1024
    comptime assert (
        MAX_BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"

    # Calculate sizes for shared memory allocation
    comptime p_width = simd_width_of[DType.int]()
    comptime u_width = simd_width_of[Scalar[T]]()

    # Allocate shared memory for indices and values
    var p_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * p_width,
        Scalar[DType.int],
        address_space = AddressSpace.SHARED,
    ]()
    var u_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * u_width,
        Scalar[T],
        address_space = AddressSpace.SHARED,
    ]()

    # Calculate warp id and thread information
    var warp = warp_id()
    comptime num_warps_needed = MAX_BLOCK_SIZE // WARP_SIZE

    # Each warp reduces its own TopK_2 value
    var warp_accum: TopK_2[T, largest] = _warp_reduce_topk[T, largest](val)

    # Store warp-level results in shared memory
    if lane_id() == 0 and warp < UInt(num_warps_needed):
        # Note: Potential bank conflict for sub 4 byte data elements
        p_sram[Int(warp) * p_width] = Scalar[DType.int](warp_accum.p)
        u_sram[Int(warp) * u_width] = warp_accum.u
    barrier()

    # Load warp results into final warp for block-level reduction
    var block_accum = TopK_2[T, largest]()
    var thread_in_final_warp = thread_idx.x < block_dim.x // UInt(WARP_SIZE)
    if thread_in_final_warp:
        var p_idx = p_sram[
            lane_id() * UInt(p_width)
        ]  # loaded value is a scalar
        block_accum = TopK_2[T, largest](
            p=Int(p_idx),
            u=u_sram[lane_id() * UInt(u_width)],  # Convert back to int
        )
    else:
        # Initialize unused threads with dummy values
        block_accum.p = -1
        block_accum.u = _topk_dead_val[T, largest]()

    # Perform final warp-level reduction for block result
    return _warp_reduce_topk[T, largest](block_accum)


fn _topk_stage1_old[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: UnsafePointer[Int64, ImmutAnyOrigin],
    max_k: Int,
    num_elements: Int,
    num_blocks_per_input: Int,
    in_buffer: UnsafePointer[Scalar[T], ImmutAnyOrigin],
    local_topk_vals: UnsafePointer[
        Scalar[T], MutAnyOrigin
    ],  # Output buffer of size num_blocks_per_input * max_k
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type], MutAnyOrigin
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

    batch_id = bid // UInt(num_blocks_per_input)
    block_lane = bid % UInt(num_blocks_per_input)

    _in_buffer = in_buffer + batch_id * UInt(num_elements)

    # Allocate shared memory for the values and indices
    var topk_sram = external_memory[
        TopK_2[T, largest],
        address_space = AddressSpace.SHARED,
        alignment = align_of[TopK_2[T, largest]](),
    ]()

    with PDL():
        # Pack the topk_vals and topk_idxs into shared memory
        var block_offset = block_lane * block_size
        var stride = block_size * UInt(num_blocks_per_input)
        topk_sram[tid] = TopK_2[T, largest]()
        for i in range(tid + block_offset, num_elements, stride):
            topk_sram[tid].insert(_in_buffer[i], i)
        barrier()
        var k_batch = max_k
        if K:
            var k_raw = Int(K[batch_id])
            k_batch = max_k if k_raw == -1 else k_raw
        # Prepare for K iterations to find the local top-K elements
        for k in range(k_batch):
            # Initialize each thread with its own TopK_2 value and index
            var partial = topk_sram[tid]

            # Perform block-level reduction to find the maximum TopK_2
            var total = _block_reduce_topk[T, largest](partial)

            if tid == 0:
                # Store the local top-K values and indices in global memory
                var vector_idx = total.p
                local_topk_vals[bid * UInt(max_k) + UInt(k)] = total.u
                local_topk_idxs[bid * UInt(max_k) + UInt(k)] = Scalar[
                    DType.int
                ](vector_idx).cast[out_idx_type]()

                # Remove the found maximum from consideration in the next iteration
                if total.p >= 0:
                    var orig_tid = (vector_idx - Int(block_offset)) % Int(
                        stride
                    )
                    topk_sram[orig_tid].u = _topk_dead_val[T, largest]()

            barrier()

        # Fill remaining positions with sentinel values for unused elements
        if tid == 0:
            for remaining_k in range(k_batch, max_k):
                local_topk_vals[
                    bid * UInt(max_k) + UInt(remaining_k)
                ] = _topk_dead_val[T, largest]()
                local_topk_idxs[bid * UInt(max_k) + UInt(remaining_k)] = Scalar[
                    out_idx_type
                ](-1)


fn _topk_stage1[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: UnsafePointer[Int64, ImmutAnyOrigin],
    max_k: Int,
    num_elements: Int,
    num_blocks_per_input: Int,
    in_buffer: UnsafePointer[Scalar[T], ImmutAnyOrigin],
    in_buffer_tmp: UnsafePointer[Scalar[T], MutAnyOrigin],
    local_topk_vals: UnsafePointer[
        Scalar[T], MutAnyOrigin
    ],  # Output buffer of size num_blocks_per_input * max_k
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type], MutAnyOrigin
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
        in_buffer_tmp: Temporary input buffer to store the elements to process.
        local_topk_vals: Output buffer to store the local top-K values.
        local_topk_idxs: Output buffer to store the indices of local top-K elements.

    Note:
        The output buffers (local_topk_vals and local_topk_idxs) should be of size num_blocks_per_input * max_k.
    """

    tid = thread_idx.x
    bid = block_idx.x
    block_size = block_dim.x

    batch_id = bid // UInt(num_blocks_per_input)
    block_lane = bid % UInt(num_blocks_per_input)

    var block_offset = block_lane * block_size
    var stride = block_size * UInt(num_blocks_per_input)

    _in_buffer = in_buffer + batch_id * UInt(num_elements)
    _in_buffer_tmp = in_buffer_tmp + batch_id * UInt(num_elements)

    # Copy input values to temp buffer
    for i in range(tid + block_offset, num_elements, stride):
        _in_buffer_tmp[i] = _in_buffer[i]

    var k_batch = max_k
    if K:
        var k_raw = Int(K[batch_id])
        k_batch = max_k if k_raw == -1 else k_raw

    # Clamp k_batch to the number of elements we can actually draw from
    if k_batch > num_elements:
        k_batch = num_elements

    # Allocate shared memory for the values and indices
    var topk_sram = external_memory[
        TopK_2[T, largest],
        address_space = AddressSpace.SHARED,
        alignment = align_of[TopK_2[T, largest]](),
    ]()

    with PDL():
        # Prepare for K iterations to find the local top-K elements
        for k in range(k_batch):
            topk_sram[tid] = TopK_2[T, largest]()

            # Pack the topk_vals and topk_idxs into shared memory
            for i in range(tid + block_offset, num_elements, stride):
                var val = _in_buffer_tmp[i]
                topk_sram[tid].insert(val, i)

            barrier()

            # Initialize each thread with its own TopK_2 value and index
            var partial = topk_sram[tid]

            # Perform block-level reduction to find the maximum TopK_2
            var total = _block_reduce_topk[T, largest](partial)

            if tid == 0:
                # Store the local top-K values and indices in global memory
                var vector_idx = total.p
                local_topk_vals[bid * UInt(max_k) + UInt(k)] = total.u
                local_topk_idxs[bid * UInt(max_k) + UInt(k)] = Scalar[
                    DType.int
                ](vector_idx).cast[out_idx_type]()

                if total.p >= 0:
                    # Remove the found maximum from consideration in the next iteration
                    _in_buffer_tmp[total.p] = _topk_dead_val[T, largest]()

            barrier()

        # Fill remaining positions with sentinel values for unused elements
        if tid == 0:
            for remaining_k in range(k_batch, max_k):
                local_topk_vals[
                    bid * UInt(max_k) + UInt(remaining_k)
                ] = _topk_dead_val[T, largest]()
                local_topk_idxs[bid * UInt(max_k) + UInt(remaining_k)] = Scalar[
                    out_idx_type
                ](-1)


@always_inline("nodebug")
fn _get_shmem_size_stg_1[dtype: DType](block_size: Int) -> Int:
    # Get dynamic shared memory size for stage 1
    return block_size * size_of[TopK_2[dtype]]()


fn _topk_stage2[
    T: DType,
    out_idx_type: DType,
    sampling: Bool = True,
    largest: Bool = True,
](
    K: UnsafePointer[Int64, ImmutAnyOrigin],
    max_k: Int,
    num_blocks_per_input: Int,
    local_topk_vals: UnsafePointer[
        Scalar[T], ImmutAnyOrigin
    ],  # Input array of size n_batch * num_blocks_per_input * K
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type], ImmutAnyOrigin
    ],  # Input array of size n_batch * num_blocks_per_input * K
    global_topk_vals: UnsafePointer[
        Scalar[T], MutAnyOrigin
    ],  # sampling ? undefined : output array of size K
    global_topk_idxs: UnsafePointer[
        Scalar[out_idx_type], MutAnyOrigin
    ],  # sampling ? sampled token : Output array of size K
    temperature: UnsafePointer[Float32, ImmutAnyOrigin],
    top_p: UnsafePointer[Float32, ImmutAnyOrigin],
    seed: UnsafePointer[UInt64, ImmutAnyOrigin],
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
    var batch_i_topk_vals = global_topk_vals + batch_id * UInt(max_k)
    var batch_i_topk_idxs = global_topk_idxs + batch_id * UInt(
        1 if sampling else max_k
    )
    var _local_topk_vals = local_topk_vals + batch_id * UInt(num_elem_reduced)
    var _local_topk_idxs = local_topk_idxs + batch_id * UInt(num_elem_reduced)

    # Allocate shared memory for values and indices
    var num_e_rounded = ceildiv(num_elem_reduced, WARP_SIZE) * WARP_SIZE
    var vals_smem_size = num_e_rounded
    var vals_sram = external_memory[
        Scalar[T],
        address_space = AddressSpace.SHARED,
        alignment = align_of[Scalar[T]](),
    ]()
    var idxs_sram = (vals_sram + vals_smem_size).bitcast[Int]()

    # These values are only read from in the sampling case.
    var s_val2 = type_of(vals_sram)()
    var s_id = type_of(idxs_sram)()

    with PDL():
        # Handle the case where stage 1 is executed with a single block
        var k_batch = max_k
        if K:
            var k_raw = Int(K[batch_id])
            k_batch = max_k if k_raw == -1 else k_raw

        # Clamp k_batch to not exceed the reduced elements per batch and max_k
        if k_batch > num_elem_reduced:
            k_batch = num_elem_reduced

        if num_blocks_per_input == 1 and not sampling:
            if tid < UInt(k_batch):
                batch_i_topk_vals[tid] = _local_topk_vals[tid]
                # cast to out_idx_type
                batch_i_topk_idxs[tid] = _local_topk_idxs[tid]
            elif tid >= UInt(k_batch) and tid < UInt(max_k):
                # Fill unused positions with sentinel values
                batch_i_topk_vals[tid] = _topk_dead_val[T, largest]()
                batch_i_topk_idxs[tid] = Scalar[out_idx_type](-1)
            return

        comptime if sampling:
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
                comptime if not sampling:
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
                comptime if sampling:
                    if k == 0:
                        max_logit = total.u

                # Remove the found maximum from consideration in the next iteration
                idxs_sram[total.p] = -1
                vals_sram[total.p] = _topk_dead_val[T, largest]()

                comptime if sampling:
                    comptime assert (
                        T.is_floating_point()
                    ), "T must be floating point for sampling"
                    batch_i_topk_vals[k] = total.u
                    s_id[k] = total.p
                    var temp_val = Float32(1.0)
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
        comptime if sampling:
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
                var rng_state = Random(seed=seed_val)
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
    out_idx_type: DType,
    //,
    sampling: Bool = True,
    largest: Bool = True,
    _force_old_impl: Bool = False,
    KLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64]
    ],
    TopPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    SeedLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
](
    ctx: DeviceContext,
    max_k: Int,
    input_buf: TileTensor[dtype, ...],
    device_local_topk_vals: TileTensor[dtype, ...],
    device_local_topk_idxs: TileTensor[out_idx_type, ...],
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    k: Optional[TileTensor[DType.int64, KLayoutType, ImmutAnyOrigin]] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
    block_size: Int = 256,
    num_blocks_per_input: Optional[Int] = None,
    top_p: Optional[
        TileTensor[DType.float32, TopPLayoutType, ImmutAnyOrigin]
    ] = None,
    seed: Optional[
        TileTensor[DType.uint64, SeedLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """Computes the Top-K elements from the input tensor using a GPU-accelerated two-stage algorithm.

    This function implements a two-stage Top-K algorithm:
    1. Stage 1 (_topk_stage1): Divides the input into blocks and computes local Top-K for each block.
    2. Stage 2 (_topk_stage2): Merges the local Top-K results to obtain the global Top-K.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        out_idx_type: DType - The data dtype of the output indices (default == DType.int).
        sampling: Bool - Whether to return token samples from topK dist (default is True).
        largest: Bool - Whether to find the maximum or minimum value.
        _force_old_impl: Bool - Whether to force use the old implementation.
        KLayoutType: Layout type of the k buffer.
        TemperatureLayoutType: Layout type of the temperature buffer.
        TopPLayoutType: Layout type of the top_p buffer.
        SeedLayoutType: Layout type of the seed buffer.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        max_k: Int
            Largest number of top elements to keep for each batch element.
        input_buf: NDBuffer[dtype, rank, DimList(batch_size,N)]
            Input tensor as a device NDBuffer.
        device_local_topk_vals: NDBuffer[dtype, 2, DimList(batch_size, num_blocks_per_input * max(K))]
            Temporary buffer for locally reduced top-K values from stage 1.
        device_local_topk_idxs: NDBuffer[DType.int, 2, DimList(batch_size, num_blocks_per_input * max(K))]
            Temporary buffer for locally reduced top-K indices from stage 1.
        out_vals: NDBuffer[dtype, 2, DimList(batch_size, max(K))]
            Output buffer on device for the K largest values.
        out_idxs: NDBuffer[DType.int, 2, DimList(batch_size, 1 if sampling else max(K))]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
        k: Optional NDBuffer[DType.int64, 1]]
            Device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling for each batch element.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: Optional[Int]
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
    comptime assert input_buf.rank == 2, "rank must be 2"
    comptime assert not (
        sampling and not largest
    ), "sampling not supported for largest=False"
    comptime assert (
        input_buf.rank == out_vals.rank
    ), "input.rank must match out_vals.rank"
    comptime assert (
        input_buf.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"

    # Use largest number of threads per block
    var batch_size = Int(input_buf.dim[0]()) if input_buf.rank == 2 else 1
    var N = Int(input_buf.dim[1]())

    # Do not launch gpu kernels with grid_dim = 0
    if batch_size == 0:
        return

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
    var grid_dim_stage1 = num_blocks_per_input_ * batch_size
    var block_dim_stage1 = block_size

    # Handle optional k parameter
    var k_ptr: UnsafePointer[Int64, ImmutAnyOrigin]
    if k:
        k_ptr = rebind[UnsafePointer[Int64, ImmutAnyOrigin]](k.value().ptr)
    else:
        k_ptr = UnsafePointer[Int64, ImmutAnyOrigin]()  # null pointer

    var k_size = k.value().numel() if k else 0
    var k_device = DeviceBuffer[DType.int64](ctx, k_ptr, k_size, owning=False)

    # Enqueue the first kernel (stage 1)
    comptime if env_get_bool[
        "USE_OLD_TOP_K_KERNEL", False
    ]() or _force_old_impl:
        comptime kernel_1 = _topk_stage1_old[dtype, out_idx_type, largest]
        ctx.enqueue_function_experimental[kernel_1](
            k_device,
            max_k,
            N,
            num_blocks_per_input_,
            input_buf.to_device_buffer(ctx),
            device_local_topk_vals.to_device_buffer(ctx),
            device_local_topk_idxs.to_device_buffer(ctx),
            grid_dim=grid_dim_stage1,
            block_dim=block_dim_stage1,
            shared_mem_bytes=shared_mem_bytes_1,
            attributes=pdl_launch_attributes(),
        )
    else:
        var input_buf_tmp = ctx.enqueue_create_buffer[dtype](batch_size * N)
        comptime kernel_1 = _topk_stage1[dtype, out_idx_type, largest]
        ctx.enqueue_function_experimental[kernel_1](
            k_device,
            max_k,
            N,
            num_blocks_per_input_,
            input_buf.to_device_buffer(ctx),
            input_buf_tmp,
            device_local_topk_vals.to_device_buffer(ctx),
            device_local_topk_idxs.to_device_buffer(ctx),
            grid_dim=grid_dim_stage1,
            block_dim=block_dim_stage1,
            shared_mem_bytes=shared_mem_bytes_1,
            attributes=pdl_launch_attributes(),
        )
        _ = input_buf_tmp^

    var num_elem_reduced = (
        ceildiv(num_blocks_per_input_ * max_k, WARP_SIZE) * WARP_SIZE
    )
    var num_bytes_sample_cache = max_k * (
        size_of[Scalar[dtype]]() + 2 * size_of[DType.int]()
    )
    var shared_mem_bytes_2 = (
        num_elem_reduced * (size_of[Scalar[dtype]]() + size_of[DType.int]())
        + num_bytes_sample_cache
    )
    # align to warp size
    shared_mem_bytes_2 = ceildiv(shared_mem_bytes_2, WARP_SIZE) * WARP_SIZE

    # Define grid and block dimensions for stage 2
    var grid_dim_stage2 = (
        batch_size  # Single block since num_elements_stage2 is small
    )
    var block_dim_stage2 = block_size

    # Handle optional temperature parameter
    var temp_ptr: UnsafePointer[Float32, ImmutAnyOrigin]
    if temperature:
        temp_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            temperature.value().ptr
        )
    else:
        temp_ptr = UnsafePointer[Float32, ImmutAnyOrigin]()  # null pointer
    var temp_size = temperature.value().numel() if temperature else 0

    # Handle optional top_p parameter
    var top_p_ptr: UnsafePointer[Float32, ImmutAnyOrigin]
    if top_p:
        top_p_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            top_p.value().ptr
        )
    else:
        top_p_ptr = UnsafePointer[Float32, ImmutAnyOrigin]()  # null pointer
    var top_p_size = top_p.value().numel() if top_p else 0

    # Handle optional seed parameter
    var seed_ptr: UnsafePointer[UInt64, ImmutAnyOrigin]
    if seed:
        seed_ptr = seed.value().ptr
    else:
        seed_ptr = UnsafePointer[UInt64, ImmutAnyOrigin]()  # null pointer
    var seed_size = seed.value().numel() if seed else 0

    var temp_device = DeviceBuffer[DType.float32](
        ctx,
        temp_ptr,
        temp_size,
        owning=False,
    )
    var top_p_device = DeviceBuffer[DType.float32](
        ctx,
        top_p_ptr,
        top_p_size,
        owning=False,
    )
    var seed_device = DeviceBuffer[DType.uint64](
        ctx,
        seed_ptr,
        seed_size,
        owning=False,
    )

    # Enqueue the second kernel (stage 2)
    comptime kernel_2 = _topk_stage2[dtype, out_idx_type, sampling, largest]
    ctx.enqueue_function_experimental[kernel_2](
        k_device,
        max_k,
        num_blocks_per_input_,
        device_local_topk_vals.to_device_buffer(ctx),
        device_local_topk_idxs.to_device_buffer(ctx),
        out_vals.to_device_buffer(ctx),
        out_idxs.to_device_buffer(ctx),
        temp_device,
        top_p_device,
        seed_device,
        grid_dim=grid_dim_stage2,
        block_dim=block_dim_stage2,
        shared_mem_bytes=shared_mem_bytes_2,
        attributes=pdl_launch_attributes(),
    )


@always_inline
fn topk_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    sampling: Bool = True,
    largest: Bool = True,
    _force_old_impl: Bool = False,
    KLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64]
    ],
    TopPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    SeedLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
](
    ctx: DeviceContext,
    max_k: Int,
    input: TileTensor[dtype, ...],
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    block_size: Optional[Int] = None,
    num_blocks_per_input: Optional[Int] = None,
    k: Optional[TileTensor[DType.int64, KLayoutType, ImmutAnyOrigin]] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
    top_p: Optional[
        TileTensor[DType.float32, TopPLayoutType, ImmutAnyOrigin]
    ] = None,
    seed: Optional[
        TileTensor[DType.uint64, SeedLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with/without sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume or the top K values and indices across the tensor.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        out_idx_type: DType - The data dtype of the output indices (default == DType.int).
        sampling: Bool - Whether to return token samples from topK dist (default is True).
        largest: Bool - Whether to find the maximum or minimum value.
        _force_old_impl: Bool - Whether to force use the old implementation.
        KLayoutType: Layout type of the k buffer.
        TemperatureLayoutType: Layout type of the temperature buffer.
        TopPLayoutType: Layout type of the top_p buffer.
        SeedLayoutType: Layout type of the seed buffer.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        max_k: Int
            Largest number of top elements to keep for each batch element.
        input: NDBuffer[dtype, rank]
            Input tensor as a device NDBuffer.
        out_vals: NDBuffer[dtype, rank]
            Output buffer on device for the K largest values.
        out_idxs: NDBuffer[DType.int, rank]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
            Last dimension is 1 if sampling is True, otherwise K.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: Optional[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.
        k: Optional NDBuffer[DType.int64, 1, MutAnyOrigin]
            Device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.
    """
    comptime assert input.rank > 0, "Input rank must be positive"
    var orig_in_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )
    var N = orig_in_shape[input.rank - 1]
    var last_idx_dim = 1 if sampling else max_k

    # Clamp max_k
    bound_max_k = 255 if max_k == -1 else max_k

    # heuristic to set block size
    var block_size_: Int
    if input.numel() <= 1024 * 64 * 3:
        block_size_ = 256
    elif input.numel() <= 32000 * 256:
        block_size_ = 512
    else:
        block_size_ = 1024
    block_size_ = block_size.value() if block_size else block_size_

    # This section handles different input ranks by reshaping to a 2D tensor
    var internal_bs: Int  # Internal batch size
    comptime internal_rank = 2  # We always reshape to 2D for internal processing
    var internal_input: TileTensor[
        dtype,
        Layout[
            shape_types = DynamicCoord[DType.int64, 2].element_types,
            stride_types = DynamicCoord[DType.int64, 2].element_types,
        ],
        input.origin,
        address_space = input.address_space,
    ]
    var internal_out_idxs: TileTensor[
        out_idx_type,
        Layout[
            shape_types = DynamicCoord[DType.int64, 2].element_types,
            stride_types = DynamicCoord[DType.int64, 2].element_types,
        ],
        out_idxs.origin,
        address_space = out_idxs.address_space,
    ]
    var internal_out_vals: TileTensor[
        dtype,
        Layout[
            shape_types = DynamicCoord[DType.int64, 2].element_types,
            stride_types = DynamicCoord[DType.int64, 2].element_types,
        ],
        out_vals.origin,
        address_space = out_vals.address_space,
    ]

    comptime if input.rank == 1:
        # Handle 1D input: treat it as a single batch with one element
        internal_bs = 1
        var internal_in_shape = IndexList[internal_rank](1, input.numel())
        var internal_out_vals_shape = IndexList[internal_rank](1, bound_max_k)
        var internal_out_idxs_shape = IndexList[internal_rank](1, last_idx_dim)
        # Reshape 1D inputs to 2D
        internal_input = reshape(input, internal_in_shape)
        internal_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
        internal_out_vals = reshape(out_vals, internal_out_vals_shape)
    elif input.rank == internal_rank:
        # Input is already 2D, no reshaping needed
        internal_bs = orig_in_shape[0]
        internal_input = rebind[type_of(internal_input)](
            input.make_dynamic[DType.int64]()
        )
        internal_out_idxs = rebind[type_of(internal_out_idxs)](
            out_idxs.make_dynamic[DType.int64]()
        )
        internal_out_vals = rebind[type_of(internal_out_vals)](
            out_vals.make_dynamic[DType.int64]()
        )
    else:  # rank > 2
        # Handle higher dimensional inputs by flattening all but the last dimension
        var _last_dim = orig_in_shape[input.rank - 1]
        internal_bs = Int(
            Float64(orig_in_shape.flattened_length()) / Float64(_last_dim)
        )

        var internal_in_shape = IndexList[internal_rank](internal_bs, _last_dim)
        var internal_out_idxs_shape = IndexList[internal_rank](
            internal_bs, last_idx_dim
        )
        var internal_out_vals_shape = IndexList[internal_rank](
            internal_bs, bound_max_k
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
    var internal_cache_shape = IndexList[2](
        internal_bs, num_blocks_per_input_ * bound_max_k
    )

    # Create temporary buffer for local top-K values
    var internal_vals_buf = ctx.enqueue_create_buffer[dtype](
        product(internal_cache_shape)
    )
    var device_local_topk_vals = TileTensor(
        internal_vals_buf.unsafe_ptr(),
        row_major(Coord(internal_cache_shape)),
    )

    # Create temporary buffer for local top-K indices
    var internal_idxs_buf = ctx.enqueue_create_buffer[out_idx_type](
        product(internal_cache_shape)
    )
    var device_local_topk_idxs = TileTensor(
        internal_idxs_buf.unsafe_ptr(),
        row_major(Coord(internal_cache_shape)),
    )

    _topk_gpu[
        dtype=dtype,
        out_idx_type=out_idx_type,
        sampling=sampling,
        largest=largest,
        _force_old_impl=_force_old_impl,
    ](
        ctx,
        bound_max_k,
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
    out_idx_type: DType,
    //,
    KLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64]
    ],
    TopPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    SeedLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
](
    ctx: DeviceContext,
    max_k: Int,
    min_top_p: Float32,
    input: TileTensor[dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    block_size: Optional[Int] = None,
    num_blocks_per_input: Optional[Int] = None,
    k: Optional[TileTensor[DType.int64, KLayoutType, ImmutAnyOrigin]] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
    top_p: Optional[
        TileTensor[DType.float32, TopPLayoutType, ImmutAnyOrigin]
    ] = None,
    seed: Optional[
        TileTensor[DType.uint64, SeedLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """
    Top K algorithm with fused sampling.
    Returns the sampled indices from the Top-K of the innermost
    dimension of the input tensor for each row/subvolume.
    """

    # If all items in the batch, want to sample all tokens (top_k==-1, top_p=1)
    # We can use gumbel sampling.
    if max_k == -1 and min_top_p == 1.0:
        gumbel_sampling_gpu(
            ctx,
            input,
            out_idxs,
            temperature,
            seed,
        )
        return

    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"

    var bound_max_k = 255 if max_k == -1 else max_k

    var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
    out_vals_shape[input.rank - 1] = bound_max_k
    var out_vals_buf = ctx.enqueue_create_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var out_vals = TileTensor(
        out_vals_buf.unsafe_ptr(),
        row_major(Coord(out_vals_shape)),
    )

    topk_gpu[sampling=True, largest=True](
        ctx,
        bound_max_k,
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


# ===-----------------------------------------------------------------------===#
# Sampling Kernel with the Gumbel-max trick
# ===-----------------------------------------------------------------------===#


fn apply_gumbel_noise_kernel[
    dtype: DType,
    OutputLayoutType: TensorLayout,
    InputLayoutType: TensorLayout,
    num_sms: Int,
    num_threads: Int,
](
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InputLayoutType, ImmutAnyOrigin],
    temperature: UnsafePointer[Float32, ImmutAnyOrigin],
    seed: UnsafePointer[UInt64, ImmutAnyOrigin],
):
    comptime EPS = Float32(1e-20)
    comptime LOG2 = Float32(0.6931471806)

    comptime simd_width = simd_width_of[dtype]()
    var N = Int(input.dim(1))
    comptime num_blocks_per_token = 8
    comptime group_size = num_blocks_per_token * num_threads
    comptime num_groups = num_sms // num_blocks_per_token

    var tid = Int(thread_idx.x)
    var sm_id = Int(block_idx.x)
    var group_id = sm_id // num_blocks_per_token
    var tid_in_group = tid + (sm_id % num_blocks_per_token) * num_threads

    var num_tokens = input.dim[0]()

    comptime assert (
        simd_width % 4 == 0
    ), "SIMD width must be divisible by 4 to match RNG output size."

    # split workload across blocks
    with PDL():
        if sm_id >= num_groups * num_blocks_per_token:
            return

        for tok_idx in range(group_id, num_tokens, num_groups):
            var temp_val = Float32(1.0)
            if temperature:
                temp_val = temperature[tok_idx]

            var seed_val = UInt64(0)
            if seed:
                seed_val = seed[tok_idx]

            var ld_ptr = input.ptr + tok_idx * N
            var st_ptr = output.ptr + tok_idx * N
            comptime align = align_of[SIMD[dtype, simd_width]]()

            for i in range(tid_in_group, N // simd_width, group_size):
                var rng_state = Random(
                    seed=seed_val * UInt64(N) + UInt64(i),
                )
                var input_val: SIMD[dtype, simd_width]
                if N % simd_width == 0:
                    input_val = ld_ptr.load[width=simd_width, alignment=align](
                        i * simd_width
                    )
                else:
                    input_val = ld_ptr.load[width=simd_width](i * simd_width)
                var noised_logits = input_val.cast[DType.float32]() / temp_val

                comptime for loop_i in range(simd_width // 4):
                    var rnd_val = rng_state.step_uniform()
                    rnd_val = -LOG2 * log2(-log2(rnd_val + EPS) + EPS)

                    comptime for vec_i in range(4):
                        noised_logits[4 * loop_i + vec_i] += rnd_val[vec_i]

                if N % simd_width == 0:
                    st_ptr.store[width=simd_width, alignment=align](
                        i * simd_width, noised_logits.cast[dtype]()
                    )
                else:
                    st_ptr.store[width=simd_width](
                        i * simd_width, noised_logits.cast[dtype]()
                    )

            # If N is not divisible by simd_width, handle remaining elements
            if N % simd_width != 0:
                var N_res = N % simd_width
                var rng_state = Random(
                    seed=seed_val * UInt64(N)
                    + UInt64(N - N_res)
                    + UInt64(tid_in_group),
                )
                if tid_in_group < N_res:
                    var input_val = ld_ptr.load(
                        (N - N_res) + tid_in_group
                    ).cast[DType.float32]()
                    var noised_logit = input_val / temp_val
                    var rnd_val = rng_state.step_uniform()[0]
                    rnd_val = -LOG2 * log2(-log2(rnd_val + EPS) + EPS)
                    noised_logit += rnd_val
                    st_ptr.store(
                        (N - N_res) + tid_in_group,
                        noised_logit.cast[dtype](),
                    )


@always_inline
fn gumbel_sampling_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    TemperatureLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64]
    ],
    SeedLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
](
    ctx: DeviceContext,
    input: TileTensor[dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
    seed: Optional[
        TileTensor[DType.uint64, SeedLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """
    Gumbel sampling using the Gumbel-max trick for categorical distributions.

    Applies Gumbel(0,1) noise to input logits, then selects the argmax.
    This is mathematically equivalent to sampling from softmax(logits/temperature)
    but avoids expensive softmax computation.

    Args:
        ctx: Device context for GPU operations.
        input: Input logits tensor [batch, vocab_size].
        out_idxs: Output tensor for sampled indices [batch, 1].
        temperature: Optional per-token temperature scaling [batch].
        seed: Optional per-token random seeds [batch] for reproducibility.
    """

    # create a buffer to hold the Gumbel noise applied input
    var noised_input_buf = ctx.enqueue_create_buffer[dtype](input.numel())
    var noised_input = TileTensor(noised_input_buf, input.layout)

    # Handle optional temperature parameter
    var temp_ptr: UnsafePointer[Float32, ImmutAnyOrigin]
    if temperature:
        temp_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            temperature.value().ptr
        )
    else:
        temp_ptr = UnsafePointer[Float32, ImmutAnyOrigin]()  # null pointer
    var temp_size = temperature.value().numel() if temperature else 0

    # Handle optional seed parameter
    var seed_ptr: UnsafePointer[UInt64, ImmutAnyOrigin]
    if seed:
        seed_ptr = rebind[UnsafePointer[UInt64, ImmutAnyOrigin]](
            seed.value().ptr
        )
    else:
        seed_ptr = UnsafePointer[UInt64, ImmutAnyOrigin]()  # null pointer
    var seed_size = seed.value().numel() if seed else 0

    comptime hw_info = ctx.default_device_info
    comptime gumbel_kernel = apply_gumbel_noise_kernel[
        dtype,
        noised_input.LayoutType,
        input.LayoutType,
        hw_info.sm_count,
        hw_info.max_thread_block_size,
    ]

    ctx.enqueue_function_experimental[gumbel_kernel](
        noised_input,
        input.as_immut(),
        temperature.value().to_device_buffer(ctx),
        seed.value().to_device_buffer(ctx),
        grid_dim=hw_info.sm_count,
        block_dim=hw_info.max_thread_block_size,
        attributes=pdl_launch_attributes(),
    )

    # Extract argmax after Gumbel noise application.
    var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
    out_vals_shape[input.rank - 1] = 1
    var out_vals_buf = ctx.enqueue_create_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var out_vals = TileTensor(
        out_vals_buf.unsafe_ptr(),
        row_major(Coord(out_vals_shape)),
    )

    # The old implementation of topk_gpu is correct when top_k = 1.
    topk_gpu[sampling=False, _force_old_impl=True](
        ctx,
        1,
        noised_input,
        out_vals,
        out_idxs,
    )

    _ = noised_input_buf^
    _ = out_vals_buf^
