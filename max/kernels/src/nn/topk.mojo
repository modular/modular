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

from std.math import ceildiv, exp, iota
from std.math.uutils import ufloordiv, udivmod
from std.memory import alloc
from std.sys import align_of, simd_width_of, size_of, get_defined_bool

import std.gpu.primitives.warp as warp
from std.algorithm.functional import parallelize_over_rows
from std.algorithm.reduction import _get_nd_indices_from_flat_index
from std.bit import log2_floor
from std.gpu import (
    WARP_SIZE,
    barrier,
    thread_idx,
    block_dim,
    block_idx,
    lane_id,
    warp_id,
)
from std.gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host.info import is_cpu
from std.gpu.memory import AddressSpace, external_memory
from std.sys.info import has_apple_gpu_accelerator, is_apple_gpu
from std.random import Random
from layout import (
    Coord,
    CoordLike,
    Idx,
    RowMajorLayout,
    RuntimeInt,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from layout.tile_layout import Layout
from std.math import log2
from std.memory import stack_allocation
from nn.gather_scatter import normalize_neg_index
from nn.reshape import reshape
from nn.softmax import softmax_with_temperature
from nn.topk_fi import apply_min_p_mask_kernel, topk_topp_sampling_from_prob
from std.runtime.asyncrt import DeviceContextPtr
from std.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import IndexList, product
from std.utils.numerics import max_or_inf, min_or_neg_inf
from std.gpu.primitives.grid_controls import PDLLevel

from .normalization import (
    _APPLE_STATIC_SHMEM_MAX_COUNT,
    _APPLE_STATIC_SHMEM_MAX_BYTES,
)


@always_inline
def top_k_shape_impl[
    dtype: DType
](input: TileTensor[dtype, ...], max_k: Int, axis: Int) raises -> IndexList[
    input.rank
]:
    """
    Compute the output shape of a top/bottom k operation.

    Parameters:
        dtype: Data type of the input buffer.

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
def _adjust_top_p[
    T: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    top_p: Scalar[T],
    values: UnsafePointer[Scalar[T], _, address_space=address_space],
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


def top_k[
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

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    @parameter
    def trace_information() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", input_shape, dtype),
                    "max_k=" + String(max_k),
                    "axis=" + String(axis),
                    "largest=" + String(largest),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "top_k",
        Trace[TraceLevel.OP]._get_detail_str[trace_information](),
        task_id=Int(ctx.get_device_context().id()),
    ):
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
                sorted=sorted,
                ctx=ctx.get_optional_device_context(),
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


def _top_k_cpu[
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
    ctx: Optional[DeviceContext] = None,
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
    def process_rows(start_row: Int, end_row: Int):
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
            def indices_to_val(idx: Int64) -> Scalar[dtype]:
                indices[axis] = Int(idx)
                var input_idx = input.layout(Coord(indices))
                return input.raw_load(input_idx)

            comptime if largest:

                @parameter
                @always_inline
                def _val_greater_than(lhs: Int64, rhs: Int64) -> Bool:
                    return indices_to_val(lhs) > indices_to_val(rhs)

                if sorted:
                    sort[_val_greater_than](idxs)
                else:
                    _ = partition[_val_greater_than](idxs, k_val)
            else:

                @parameter
                @always_inline
                def _val_less_than(lhs: Int64, rhs: Int64) -> Bool:
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
                    var curr = input.raw_load(input_idx)
                    var num_equal = 1
                    for j in range(i + 1, shape[axis]):
                        indices[axis] = Int(idxs[j])
                        var input_idx = input.layout(Coord(indices))
                        var next = input.raw_load(input_idx)
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
                var val = input.raw_load(input_idx)
                indices[axis] = i
                var out_vals_idx = out_vals.layout(Coord(indices))
                var out_idxs_idx = out_idxs.layout(Coord(indices))
                out_vals.raw_store(out_vals_idx, val)
                out_idxs.ptr[out_idxs_idx] = rebind[Scalar[out_idx_type]](
                    idxs[i]
                )

    parallelize_over_rows[process_rows](
        shape, axis, parallelism_grain_size, ctx
    )


@always_inline
def fused_token_sampling_cpu[
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
        input: TileTensor[dtype] (Any shape)- The input tensor.
        out_idxs: TileTensor[out_idx_type] (shape of [input_shape[:-1]] + [1]) - The output indices.
        k: Optional device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.
    """
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"
    comptime assert out_idx_type == DType.int64, "out_idx_type must be int64"

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    @parameter
    def trace_information() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", input_shape, dtype),
                    "max_k=" + String(max_k),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("cpu")](
        "fused_token_sampling",
        Trace[TraceLevel.OP]._get_detail_str[trace_information](),
    ):
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
            out_idxs.bitcast[DType.int64](),
            k,
            temperature,
            top_p,
            seed,
        )

        out_vals.ptr.free()


def _top_k_sampling[
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
        input: TileTensor[dtype] (Any shape)- The input tensor.
        out_vals: TileTensor[dtype] (shape of [input[:-1]] + [k]) - The output values.
        out_idxs: TileTensor[DType.int64] (shape of [input[:-1]] + [1]) - The output indices.
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
        internal_in_shape = IndexList[internal_rank](1, input.num_elements())
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
        alloc[Int64](out_vals.num_elements()),
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
def _topk_dead_val[T: DType, largest: Bool = True]() -> Scalar[T]:
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

    def __init__(out self):
        self.p = -1
        self.u = _topk_dead_val[Self.T, Self.largest]()

    def insert(mut self, elem: Scalar[Self.T], elem_id: Int):
        comptime if Self.largest:
            if elem > self.u:
                self.u = elem
                self.p = elem_id
        else:
            if elem < self.u:
                self.u = elem
                self.p = elem_id


struct TopKHeap[T: DType, largest: Bool, M: Int]:
    """Fixed-capacity register heap for per-thread top-M tracking.

    Stores up to M (value, index) pairs in registers. During the scan
    phase, a cached threshold provides O(1) rejection of non-competitive
    elements. All internal loops are compile-time unrolled to keep data
    in registers on GPU. Indices are stored as Int32 to reduce register
    pressure for large block sizes.
    """

    var vals: InlineArray[Scalar[Self.T], Self.M]
    var idxs: InlineArray[Int32, Self.M]
    var threshold: Scalar[Self.T]

    @always_inline
    def __init__(out self):
        self.vals = InlineArray[Scalar[Self.T], Self.M](
            fill=_topk_dead_val[Self.T, Self.largest]()
        )
        self.idxs = InlineArray[Int32, Self.M](fill=Int32(-1))
        self.threshold = _topk_dead_val[Self.T, Self.largest]()

    @always_inline
    def insert(mut self, val: Scalar[Self.T], idx: Int):
        """Insert an element, evicting the worst if full."""
        # Fast reject against threshold. When the heap has empty slots
        # the threshold equals dead_val, so the check naturally fails
        # for all real values and we fall through to the empty-slot path.
        comptime if Self.largest:
            if val <= self.threshold:
                return
        else:
            if val >= self.threshold:
                return

        var idx32 = Int32(idx)

        # Try to place in an empty slot.
        var inserted = False
        comptime for i in range(Self.M):
            if not inserted and self.idxs[i] == -1:
                self.vals[i] = val
                self.idxs[i] = idx32
                inserted = True

        if not inserted:
            # All slots full — replace the first element at threshold.
            comptime for i in range(Self.M):
                if not inserted and self.vals[i] == self.threshold:
                    self.vals[i] = val
                    self.idxs[i] = idx32
                    inserted = True
        self._update_threshold()

    @always_inline
    def _update_threshold(mut self):
        """Recompute eviction threshold (worst value in the heap)."""
        self.threshold = self.vals[0]
        comptime for i in range(1, Self.M):
            comptime if Self.largest:
                if self.vals[i] < self.threshold:
                    self.threshold = self.vals[i]
            else:
                if self.vals[i] > self.threshold:
                    self.threshold = self.vals[i]

    @always_inline
    def best(self) -> TopK_2[Self.T, Self.largest]:
        """Return the best element, ties broken by smallest index.

        Returns a dead TopK_2 (p=-1) when all entries are exhausted.
        """
        var best_u = self.vals[0]
        var best_p = self.idxs[0]
        comptime for i in range(1, Self.M):
            comptime if Self.largest:
                if self.vals[i] > best_u or (
                    self.vals[i] == best_u and self.idxs[i] < best_p
                ):
                    best_u = self.vals[i]
                    best_p = self.idxs[i]
            else:
                if self.vals[i] < best_u or (
                    self.vals[i] == best_u and self.idxs[i] < best_p
                ):
                    best_u = self.vals[i]
                    best_p = self.idxs[i]
        return TopK_2[Self.T, Self.largest](p=Int(best_p), u=best_u)

    @always_inline
    def remove(mut self, idx: Int):
        """Remove element by global index, replacing with dead value."""
        var idx32 = Int32(idx)
        comptime for i in range(Self.M):
            if self.idxs[i] == idx32:
                self.vals[i] = _topk_dead_val[Self.T, Self.largest]()
                self.idxs[i] = Int32(-1)


# Function to perform warp-level reduction to find the maximum TopK_2
@always_inline
@parameter
def _warp_reduce_topk[
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
    def shuffle_topk2(v: TopK_2[T, largest], offset: Int) -> TopK_2[T, largest]:
        comptime fn_type = def[dtype: DType, simd_width: SIMDSize](
            val: SIMD[dtype, simd_width], offset: UInt32
        ) thin -> SIMD[dtype, simd_width]
        comptime xor_fn: fn_type = warp.shuffle_xor
        comptime down_fn: fn_type = warp.shuffle_down

        comptime shuffle_fn = xor_fn if broadcast else down_fn

        return TopK_2[T, largest](
            u=shuffle_fn(v.u, UInt32(offset)),  # u is the value
            p=Int(shuffle_fn(Int32(v.p), UInt32(offset))),  # p is the index
        )

    @parameter
    def reduce_fn(
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
def _block_reduce_topk[
    T: DType,
    //,
    ascending: Bool,
    MAX_BLOCK_SIZE: Int = WARP_SIZE if is_apple_gpu() else 1024,
](val: TopK_2[T, ascending]) -> TopK_2[T, ascending]:
    """
    Performs a block-level reduction to find the maximum TopK_2 element.

    This function takes a TopK_2 value from each thread in a block and performs
    a reduction to find the maximum across all threads. It uses shared memory
    and warp-level reductions to efficiently compute the block-wide maximum.

    Parameters:
        T: DType - The data dtype of the values being compared.
        ascending: Bool - Whether to find the maximum or minimum value.
        MAX_BLOCK_SIZE: Int - The maximum number of threads in a block.

    Arguments:
        val: TopK_2[T, ascending] - The TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T, ascending] - The maximum TopK_2 value across all threads in the block.

    Note:
    This function assumes that BLOCK_SIZE is a multiple of WARP_SIZE.
    It uses shared memory to store intermediate results and performs
    a final warp-level reduction to compute the block-wide maximum.
    """
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
        address_space=AddressSpace.SHARED,
    ]()
    var u_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * u_width,
        Scalar[T],
        address_space=AddressSpace.SHARED,
    ]()

    # Calculate warp id and thread information
    var warp = warp_id()
    comptime num_warps_needed = MAX_BLOCK_SIZE // WARP_SIZE

    # Each warp reduces its own TopK_2 value
    var warp_accum: TopK_2[T, ascending] = _warp_reduce_topk[T, ascending](val)

    # Store warp-level results in shared memory
    if lane_id() == 0 and warp < num_warps_needed:
        # Note: Potential bank conflict for sub 4 byte data elements
        p_sram[warp * p_width] = Scalar[DType.int](warp_accum.p)
        u_sram[warp * u_width] = warp_accum.u
    barrier()

    # Load warp results into final warp for block-level reduction
    var block_accum = TopK_2[T, ascending]()
    var thread_in_final_warp = thread_idx.x < ufloordiv(block_dim.x, WARP_SIZE)
    if thread_in_final_warp:
        var p_idx = p_sram[lane_id() * p_width]  # loaded value is a scalar
        block_accum = TopK_2[T, ascending](
            p=Int(p_idx),  # Convert back to int
            u=u_sram[lane_id() * u_width],
        )
    else:
        # Initialize unused threads with dummy values
        block_accum.p = -1
        block_accum.u = _topk_dead_val[T, ascending]()

    # Perform final warp-level reduction for block result
    return _warp_reduce_topk[T, ascending](block_accum)


@__name(t"topk_stage1_old_no_shmem_{T}_{out_idx_type}_{largest}", mangle=True)
def _topk_stage1_old_no_shmem[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
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
    """Shared-memory-free variant of _topk_stage1_old for Apple GPUs.

    Each thread keeps its local TopK_2 in registers. Warp-level reduction
    finds the block-wide winner, which is broadcast so the owning thread
    can invalidate its own register value for the next iteration.

    Only correct when block_size <= WARP_SIZE (single warp per block),
    which is always the case on Apple GPUs.
    """

    tid = thread_idx.x
    bid = block_idx.x
    block_size = block_dim.x

    batch_id, block_lane = udivmod(bid, num_blocks_per_input)

    _in_buffer = in_buffer + batch_id * num_elements

    with PDL():
        # Each thread finds its local best element in registers.
        var block_offset = block_lane * block_size
        var stride = block_size * num_blocks_per_input
        var partial = TopK_2[T, largest]()
        for i in range(tid + block_offset, num_elements, stride):
            partial.insert(_in_buffer[i], i)

        var k_batch = max_k
        if K:
            var k_raw = Int(K.unsafe_value()[batch_id])
            k_batch = max_k if k_raw == -1 else k_raw

        # Find top-K elements via repeated warp reductions.
        for k in range(k_batch):
            # Warp-level reduction to find the best element across
            # all threads (no shared memory needed).
            var total = _warp_reduce_topk[T, largest](partial)

            # Broadcast the winner's index to all lanes so the owning
            # thread can invalidate its register.
            var winner_p = warp.broadcast(total.p)

            if tid == 0:
                local_topk_vals[bid * max_k + k] = total.u
                local_topk_idxs[bid * max_k + k] = Scalar[DType.int](
                    total.p
                ).cast[out_idx_type]()

            # The thread that owned the winning element invalidates it.
            if partial.p == winner_p:
                partial.u = _topk_dead_val[T, largest]()
                partial.p = -1

        # Fill remaining positions with sentinel values.
        if tid == 0:
            for remaining_k in range(k_batch, max_k):
                local_topk_vals[bid * max_k + remaining_k] = _topk_dead_val[
                    T, largest
                ]()
                local_topk_idxs[bid * max_k + remaining_k] = Scalar[
                    out_idx_type
                ](-1)


@__name(t"topk_stage1_old_{T}_{out_idx_type}_{largest}", mangle=True)
def _topk_stage1_old[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
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

    batch_id, block_lane = udivmod(bid, num_blocks_per_input)

    _in_buffer = in_buffer + batch_id * num_elements

    # Allocate shared memory for the values and indices
    var topk_sram = stack_allocation[
        _APPLE_STATIC_SHMEM_MAX_COUNT[TopK_2[T]],
        TopK_2[T, largest],
        address_space=AddressSpace.SHARED,
    ]() if comptime (is_apple_gpu()) else external_memory[
        TopK_2[T, largest],
        address_space=AddressSpace.SHARED,
        alignment=align_of[TopK_2[T, largest]](),
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
            var k_raw = Int(K.unsafe_value()[batch_id])
            k_batch = max_k if k_raw == -1 else k_raw
        # Prepare for K iterations to find the local top-K elements
        for k in range(k_batch):
            # Initialize each thread with its own TopK_2 value and index
            var partial = topk_sram[tid]

            # Perform block-level reduction to find the maximum TopK_2
            var total = _block_reduce_topk[ascending=largest](partial)

            if tid == 0:
                # Store the local top-K values and indices in global memory
                var vector_idx = total.p
                local_topk_vals[bid * max_k + k] = total.u
                local_topk_idxs[bid * max_k + k] = Scalar[DType.int](
                    vector_idx
                ).cast[out_idx_type]()

                # Remove the found maximum from consideration in the next iteration
                if total.p >= 0:
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


@__name(t"topk_stage1_no_shmem_{T}_{out_idx_type}_{largest}", mangle=True)
def _topk_stage1_no_shmem[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
    max_k: Int,
    num_elements: Int,
    num_blocks_per_input: Int,
    in_buffer_tmp: UnsafePointer[Scalar[T], MutAnyOrigin],
    local_topk_vals: UnsafePointer[
        Scalar[T], MutAnyOrigin
    ],  # Output buffer of size num_blocks_per_input * max_k
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type], MutAnyOrigin
    ],  # Output buffer of size num_blocks_per_input * max_k
):
    """Shared-memory-free variant of _topk_stage1 for Apple GPUs.

    Uses warp-level reduction and broadcasts instead of block-level
    reduction with shared memory. The winner is broadcast to all lanes
    and each thread writes the dead value to global memory for its own
    winning element, avoiding the need for thread 0 to write to another
    thread's global memory slot.

    Only correct when block_size <= WARP_SIZE (single warp per block).
    """

    tid = thread_idx.x
    bid = block_idx.x
    block_size = block_dim.x

    batch_id, block_lane = udivmod(bid, num_blocks_per_input)

    var block_offset = block_lane * block_size
    var stride = block_size * num_blocks_per_input

    _in_buffer_tmp = in_buffer_tmp + batch_id * num_elements

    # Hoist per-block output base pointers out of the k loop.
    var out_vals = local_topk_vals + bid * max_k
    var out_idxs = local_topk_idxs + bid * max_k

    var k_batch = max_k
    if K:
        var k_raw = Int(K.unsafe_value()[batch_id])
        k_batch = max_k if k_raw == -1 else k_raw

    # Clamp k_batch to the number of elements we can actually draw from
    if k_batch > num_elements:
        k_batch = num_elements

    comptime HEAP_SIZE = 8

    with PDL():
        # Phase 1: Single scan to build per-thread register heap.
        var heap = TopKHeap[T, largest, HEAP_SIZE]()
        for i in range(tid + block_offset, num_elements, stride):
            heap.insert(_in_buffer_tmp[i], i)

        # Phase 2: Extract winners from heaps without re-scanning.
        # Threads whose heap is exhausted fall back to a global-memory
        # re-scan so that non-top-M elements are still discoverable.
        var heap_iters = min(k_batch, HEAP_SIZE)
        for k in range(heap_iters):
            # Use heap if it has valid entries, else fall back to re-scan.
            var partial = heap.best()
            if partial.p < 0:
                partial = TopK_2[T, largest]()
                for i in range(tid + block_offset, num_elements, stride):
                    partial.insert(_in_buffer_tmp[i], i)

            var total = _warp_reduce_topk[T, largest](partial)

            var winner_p = warp.broadcast(total.p)

            if tid == 0:
                out_vals[k] = total.u
                out_idxs[k] = Scalar[DType.int](total.p).cast[out_idx_type]()

            if partial.p == winner_p and winner_p >= 0:
                heap.remove(winner_p)
                _in_buffer_tmp[winner_p] = _topk_dead_val[T, largest]()

        # Phase 3: Fallback to global-memory re-scan for remaining k.
        for k in range(heap_iters, k_batch):
            var partial = TopK_2[T, largest]()

            for i in range(tid + block_offset, num_elements, stride):
                partial.insert(_in_buffer_tmp[i], i)

            var total = _warp_reduce_topk[T, largest](partial)

            var winner_p = warp.broadcast(total.p)

            if tid == 0:
                out_vals[k] = total.u
                out_idxs[k] = Scalar[DType.int](total.p).cast[out_idx_type]()

            if partial.p == winner_p and winner_p >= 0:
                _in_buffer_tmp[winner_p] = _topk_dead_val[T, largest]()

        # Fill remaining positions with sentinel values.
        if tid == 0:
            for remaining_k in range(k_batch, max_k):
                out_vals[remaining_k] = _topk_dead_val[T, largest]()
                out_idxs[remaining_k] = Scalar[out_idx_type](-1)


@__name(t"topk_stage1_{T}_{out_idx_type}_{largest}", mangle=True)
def _topk_stage1[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
    max_k: Int,
    num_elements: Int,
    num_blocks_per_input: Int,
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

    The input data must be pre-copied into in_buffer_tmp before launching this kernel
    (via device-to-device DMA copy), allowing the copy engine to operate in parallel.

    Parameters:
        T: Data type of the elements.
        out_idx_type: DType - The data dtype of the output indices.
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        K: Number of top elements to select per block. Varies for each batch element.
        max_k: Largest number of top elements to keep for each batch element.
        num_elements: Size of last dimension of input buffer (vocab size).
        num_blocks_per_input: Number of blocks used to process the input data.
        in_buffer_tmp: Pre-copied input buffer to read and modify during top-K.
        local_topk_vals: Output buffer to store the local top-K values.
        local_topk_idxs: Output buffer to store the indices of local top-K elements.

    Note:
        The output buffers (local_topk_vals and local_topk_idxs) should be of size num_blocks_per_input * max_k.
    """

    tid = thread_idx.x
    bid = block_idx.x
    block_size = block_dim.x

    batch_id, block_lane = udivmod(bid, num_blocks_per_input)

    var block_offset = block_lane * block_size
    var stride = block_size * num_blocks_per_input

    _in_buffer_tmp = in_buffer_tmp + batch_id * num_elements

    # Hoist per-block output base pointers out of the k loop.
    var out_vals = local_topk_vals + bid * max_k
    var out_idxs = local_topk_idxs + bid * max_k

    var k_batch = max_k
    if K:
        var k_raw = Int(K.unsafe_value()[batch_id])
        k_batch = max_k if k_raw == -1 else k_raw

    # Clamp k_batch to the number of elements we can actually draw from
    if k_batch > num_elements:
        k_batch = num_elements

    # Shared memory to broadcast the winner index so the owning thread
    # can write the dead value (better L1 locality than thread 0).
    var winner_sram = stack_allocation[
        1, Int, address_space=AddressSpace.SHARED
    ]()

    comptime HEAP_SIZE = 8

    with PDL():
        # Phase 1: Single scan to build per-thread register heap.
        var heap = TopKHeap[T, largest, HEAP_SIZE]()
        for i in range(tid + block_offset, num_elements, stride):
            heap.insert(_in_buffer_tmp[i], i)

        # Phase 2: Extract winners from heaps without re-scanning.
        # Threads whose heap is exhausted fall back to a global-memory
        # re-scan so that non-top-M elements are still discoverable.
        var heap_iters = min(k_batch, HEAP_SIZE)
        for k in range(heap_iters):
            # Use heap if it has valid entries, else fall back to re-scan.
            var partial = heap.best()
            if partial.p < 0:
                partial = TopK_2[T, largest]()
                for i in range(tid + block_offset, num_elements, stride):
                    partial.insert(_in_buffer_tmp[i], i)

            var total = _block_reduce_topk[ascending=largest](partial)

            if tid == 0:
                out_vals[k] = total.u
                out_idxs[k] = Scalar[DType.int](total.p).cast[out_idx_type]()
                winner_sram[0] = total.p
            barrier()

            var winner_p = winner_sram[0]
            if partial.p == winner_p and winner_p >= 0:
                heap.remove(winner_p)
                _in_buffer_tmp[winner_p] = _topk_dead_val[T, largest]()

        # Phase 3: Fallback to global-memory re-scan for remaining k.
        for k in range(heap_iters, k_batch):
            var partial = TopK_2[T, largest]()

            for i in range(tid + block_offset, num_elements, stride):
                var val = _in_buffer_tmp[i]
                partial.insert(val, i)

            var total = _block_reduce_topk[ascending=largest](partial)

            if tid == 0:
                out_vals[k] = total.u
                out_idxs[k] = Scalar[DType.int](total.p).cast[out_idx_type]()
                winner_sram[0] = total.p
            barrier()

            var winner_p = winner_sram[0]
            if partial.p == winner_p and winner_p >= 0:
                _in_buffer_tmp[winner_p] = _topk_dead_val[T, largest]()

        # Parallel sentinel fill using all threads.
        for remaining_k in range(k_batch + tid, max_k, block_size):
            out_vals[remaining_k] = _topk_dead_val[T, largest]()
            out_idxs[remaining_k] = Scalar[out_idx_type](-1)


@always_inline("nodebug")
def _get_shmem_size_stg_1[dtype: DType](block_size: Int) -> Int:
    # Get dynamic shared memory size for stage 1
    return block_size * size_of[TopK_2[dtype]]()


@__name(t"topk_stage2_{T}_{out_idx_type}_{sampling}_{largest}", mangle=True)
def _topk_stage2[
    T: DType,
    out_idx_type: DType,
    sampling: Bool = True,
    largest: Bool = True,
](
    K: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
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
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    top_p: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    min_p: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    seed: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]],
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
        min_p: Per-row min-p threshold. Tokens with probability below
            ``min_p * max_prob`` are excluded from sampling.
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
    var vals_sram = stack_allocation[
        _APPLE_STATIC_SHMEM_MAX_COUNT[TopK_2[T]],
        Scalar[T],
        address_space=AddressSpace.SHARED,
    ]() if comptime (is_apple_gpu()) else external_memory[
        Scalar[T],
        address_space=AddressSpace.SHARED,
        alignment=align_of[Scalar[T]](),
    ]()

    var idxs_sram = (vals_sram + vals_smem_size).bitcast[Int]()

    # SAFETY: Only dereferenced inside `comptime if sampling` blocks;
    # overwritten with real pointers before use when sampling is enabled.
    var s_val2 = type_of(vals_sram).unsafe_dangling()
    var s_id = type_of(idxs_sram).unsafe_dangling()

    with PDL():
        # Handle the case where stage 1 is executed with a single block
        var k_batch = max_k
        if K:
            var k_raw = Int(K.unsafe_value()[batch_id])
            k_batch = max_k if k_raw == -1 else k_raw

        # Clamp k_batch to not exceed the reduced elements per batch and max_k
        if k_batch > num_elem_reduced:
            k_batch = num_elem_reduced

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

        comptime if sampling:
            # Storing the top-K logits in shmem for sampling
            s_id = (idxs_sram + vals_smem_size).bitcast[Int]()
            # The 2* below is for warp align safety
            s_val2 = (s_id + 2 * k_batch).bitcast[Scalar[T]]()

        var s_sum = stack_allocation[
            1, Scalar[T], address_space=AddressSpace.SHARED
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
            var total = _block_reduce_topk[ascending=largest](partial)

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
                        temp_val = temperature.unsafe_value()[batch_id]
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
                # Apply min_p mask: zero out probs below min_p * max_prob.
                # Since s_val2[0] = exp(0/temp) = 1.0 is the max, the
                # threshold in this unnormalized softmax domain is simply
                # min_p_val.
                if min_p:
                    var min_p_val = Scalar[T](
                        min_p.unsafe_value()[batch_id].cast[T]()
                    )
                    if min_p_val > 0:
                        for ki in range(k_batch):
                            if s_val2[ki] < min_p_val:
                                s_sum[0] -= s_val2[ki]
                                s_val2[ki] = Scalar[T](0)

                var top_p_val = Scalar[T](1.0)
                if top_p:
                    top_p_val = top_p.unsafe_value()[batch_id].cast[T]()
                var _top_p = _adjust_top_p[T](
                    top_p_val, s_val2, k_batch, s_sum[0]
                )

                # Use the largest logit's id as the offset for the random number
                # generator, so that we don't use the same random number for every
                # token in the sequence.
                var seed_val = UInt64(0)
                if seed:
                    seed_val = seed.unsafe_value()[batch_id]
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


def _topk_gpu[
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
    MinPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
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
    min_p: Optional[
        TileTensor[DType.float32, MinPLayoutType, ImmutAnyOrigin]
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
        MinPLayoutType: Layout type of the min_p buffer.
        SeedLayoutType: Layout type of the seed buffer.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        max_k: Int
            Largest number of top elements to keep for each batch element.
        input_buf: TileTensor[dtype, [batch_size, N]]
            Input tensor as a device TileTensor.
        device_local_topk_vals: TileTensor[dtype, [batch_size, num_blocks_per_input * max(K)]]
            Temporary buffer for locally reduced top-K values from stage 1.
        device_local_topk_idxs: TileTensor[DType.int, [batch_size, num_blocks_per_input * max(K)]]
            Temporary buffer for locally reduced top-K indices from stage 1.
        out_vals: TileTensor[dtype, [batch_size, max(K)]]
            Output buffer on device for the K largest values.
        out_idxs: TileTensor[DType.int, [batch_size, 1 if sampling else max(K)]]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
        k: Optional TileTensor[DType.int64]
            Device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling for each batch element.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: Optional[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        min_p: Per-row min-p threshold. Tokens with probability below
            ``min_p * max_prob`` are excluded from sampling.
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

    # On Apple GPUs, the no-shmem kernel variants require single-warp
    # blocks. Clamp block_size to WARP_SIZE and recompute blocks.
    var effective_block_size = block_size
    comptime if has_apple_gpu_accelerator():
        effective_block_size = WARP_SIZE

    # Define the number of blocks per grid
    var num_blocks_per_input_: Int = ceildiv(
        N, effective_block_size
    ) if not num_blocks_per_input else num_blocks_per_input.value()
    # Calculate largest num bytes of shmem for each stage
    if effective_block_size % WARP_SIZE != 0:
        # TODO: Need to pad in this case
        raise Error("block_size must be a multiple of WARP_SIZE")

    # Define grid and block dimensions for stage 1
    var grid_dim_stage1 = num_blocks_per_input_ * batch_size
    var block_dim_stage1 = effective_block_size

    # Handle optional k parameter
    var k_ptr: Optional[UnsafePointer[Int64, ImmutAnyOrigin]] = None
    if k:
        k_ptr = rebind[UnsafePointer[Int64, ImmutAnyOrigin]](k.value().ptr)

    # Enqueue the first kernel (stage 1)
    comptime if get_defined_bool[
        "USE_OLD_TOP_K_KERNEL", False
    ]() or _force_old_impl:
        comptime if has_apple_gpu_accelerator():
            # On Apple GPUs, use the no-shmem variant that keeps all
            # data in registers and uses warp-level reduction only.
            comptime kernel_1 = _topk_stage1_old_no_shmem[
                dtype, out_idx_type, largest
            ]
            ctx.enqueue_function[kernel_1](
                k_ptr,
                max_k,
                N,
                num_blocks_per_input_,
                input_buf.to_device_buffer(ctx),
                device_local_topk_vals.to_device_buffer(ctx),
                device_local_topk_idxs.to_device_buffer(ctx),
                grid_dim=grid_dim_stage1,
                block_dim=block_dim_stage1,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
        else:
            var shared_mem_bytes_1 = _get_shmem_size_stg_1[dtype](block_size)
            comptime kernel_1 = _topk_stage1_old[dtype, out_idx_type, largest]
            ctx.enqueue_function[kernel_1](
                k_ptr,
                max_k,
                N,
                num_blocks_per_input_,
                input_buf.to_device_buffer(ctx),
                device_local_topk_vals.to_device_buffer(ctx),
                device_local_topk_idxs.to_device_buffer(ctx),
                grid_dim=grid_dim_stage1,
                block_dim=block_dim_stage1,
                shared_mem_bytes=shared_mem_bytes_1,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
    else:
        var input_buf_tmp = ctx.enqueue_create_buffer[dtype](batch_size * N)
        # Use DMA copy engine instead of kernel-based copy
        ctx.enqueue_copy(input_buf_tmp, input_buf.to_device_buffer(ctx))
        comptime if has_apple_gpu_accelerator():
            comptime kernel_1 = _topk_stage1_no_shmem[
                dtype, out_idx_type, largest
            ]
            ctx.enqueue_function[kernel_1](
                k_ptr,
                max_k,
                N,
                num_blocks_per_input_,
                input_buf_tmp,
                device_local_topk_vals.to_device_buffer(ctx),
                device_local_topk_idxs.to_device_buffer(ctx),
                grid_dim=grid_dim_stage1,
                block_dim=block_dim_stage1,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
        else:
            comptime kernel_1 = _topk_stage1[dtype, out_idx_type, largest]
            ctx.enqueue_function[kernel_1](
                k_ptr,
                max_k,
                N,
                num_blocks_per_input_,
                input_buf_tmp,
                device_local_topk_vals.to_device_buffer(ctx),
                device_local_topk_idxs.to_device_buffer(ctx),
                grid_dim=grid_dim_stage1,
                block_dim=block_dim_stage1,
                attributes=pdl_launch_attributes(PDLLevel.ON),
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
    comptime if has_apple_gpu_accelerator():
        if shared_mem_bytes_2 > _APPLE_STATIC_SHMEM_MAX_BYTES:
            raise Error(
                t"shared memory of {shared_mem_bytes_2} exceeds static"
                t" allocation capacity of {_APPLE_STATIC_SHMEM_MAX_BYTES} for"
                t" the second stage top-k kernel, consider reducing the"
                t" block_size or num_blocks_per_input"
            )

    # Define grid and block dimensions for stage 2
    var grid_dim_stage2 = (
        batch_size  # Single block since num_elements_stage2 is small
    )
    var block_dim_stage2 = block_size

    # Handle optional temperature parameter
    var temp_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
    if temperature:
        temp_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            temperature.value().ptr
        )

    # Handle optional top_p parameter
    var top_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
    if top_p:
        top_p_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            top_p.value().ptr
        )

    # Handle optional min_p parameter
    var min_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
    if min_p:
        min_p_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            min_p.value().ptr
        )

    # Handle optional seed parameter
    var seed_ptr: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]] = None
    if seed:
        seed_ptr = seed.value().ptr

    # Enqueue the second kernel (stage 2)
    comptime kernel_2 = _topk_stage2[dtype, out_idx_type, sampling, largest]
    ctx.enqueue_function[kernel_2](
        k_ptr,
        max_k,
        num_blocks_per_input_,
        device_local_topk_vals.to_device_buffer(ctx),
        device_local_topk_idxs.to_device_buffer(ctx),
        out_vals.to_device_buffer(ctx),
        out_idxs.to_device_buffer(ctx),
        temp_ptr,
        top_p_ptr,
        min_p_ptr,
        seed_ptr,
        grid_dim=grid_dim_stage2,
        block_dim=block_dim_stage2,
        shared_mem_bytes=shared_mem_bytes_2,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@always_inline
def topk_gpu[
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
    MinPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
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
    min_p: Optional[
        TileTensor[DType.float32, MinPLayoutType, ImmutAnyOrigin]
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
        MinPLayoutType: Layout type of the min_p buffer.
        SeedLayoutType: Layout type of the seed buffer.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        max_k: Int
            Largest number of top elements to keep for each batch element.
        input: TileTensor[dtype]
            Input tensor as a device TileTensor.
        out_vals: TileTensor[dtype]
            Output buffer on device for the K largest values.
        out_idxs: TileTensor[DType.int]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
            Last dimension is 1 if sampling is True, otherwise K.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: Optional[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.
        k: Optional TileTensor[DType.int64]
            Device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        min_p: Per-row min-p threshold. Tokens with probability below
            ``min_p * max_prob`` are excluded from sampling.
        seed: The seed to use for the random number generator.
    """
    comptime assert input.rank > 0, "Input rank must be positive"
    var orig_in_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    @parameter
    def trace_information() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", orig_in_shape, dtype),
                    "max_k=" + String(max_k),
                    "sampling=" + String(sampling),
                    "largest=" + String(largest),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_gpu",
        Trace[TraceLevel.OP]._get_detail_str[trace_information](),
        task_id=Int(ctx.id()),
    ):
        var N = orig_in_shape[input.rank - 1]
        var last_idx_dim = 1 if sampling else max_k

        # Clamp max_k
        bound_max_k = 255 if max_k == -1 else max_k

        # heuristic to set block size
        var block_size_: Int
        if input.num_elements() <= 1024 * 64 * 3:
            block_size_ = 256
        elif input.num_elements() <= 32000 * 256:
            block_size_ = 512
        else:
            block_size_ = 1024
        block_size_ = block_size.value() if block_size else block_size_

        # On Apple GPUs, the no-shmem kernel variants require single-warp
        # blocks. Clamp block_size to WARP_SIZE.
        comptime if has_apple_gpu_accelerator():
            block_size_ = min(block_size_, WARP_SIZE)

        # This section handles different input ranks by reshaping to a 2D tensor
        var internal_bs: Int  # Internal batch size
        comptime internal_rank = 2  # We always reshape to 2D for internal processing
        var internal_input: TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[DType.int64, 2].element_types,
                stride_types=DynamicCoord[DType.int64, 2].element_types,
            ],
            input.origin,
            address_space=input.address_space,
        ]
        var internal_out_idxs: TileTensor[
            out_idx_type,
            Layout[
                shape_types=DynamicCoord[DType.int64, 2].element_types,
                stride_types=DynamicCoord[DType.int64, 2].element_types,
            ],
            out_idxs.origin,
            address_space=out_idxs.address_space,
        ]
        var internal_out_vals: TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[DType.int64, 2].element_types,
                stride_types=DynamicCoord[DType.int64, 2].element_types,
            ],
            out_vals.origin,
            address_space=out_vals.address_space,
        ]

        comptime if input.rank == 1:
            # Handle 1D input: treat it as a single batch with one element
            internal_bs = 1
            var internal_in_shape = IndexList[internal_rank](
                1, input.num_elements()
            )
            var internal_out_vals_shape = IndexList[internal_rank](
                1, bound_max_k
            )
            var internal_out_idxs_shape = IndexList[internal_rank](
                1, last_idx_dim
            )
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

            var internal_in_shape = IndexList[internal_rank](
                internal_bs, _last_dim
            )
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
            internal_vals_buf,
            row_major(Coord(internal_cache_shape)),
        )

        # Create temporary buffer for local top-K indices
        var internal_idxs_buf = ctx.enqueue_create_buffer[out_idx_type](
            product(internal_cache_shape)
        )
        var device_local_topk_idxs = TileTensor(
            internal_idxs_buf,
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
            min_p=min_p,
            seed=seed,
        )

        # Clean up buffers
        _ = internal_vals_buf^
        _ = internal_idxs_buf^


def _topk_topp_sampling_fi[
    dtype: DType,
    out_idx_type: DType,
    KLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64]
    ],
    TopPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    MinPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    SeedLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
](
    ctx: DeviceContext,
    max_k: Int,
    min_top_p: Float32,
    input: TileTensor[dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    k: Optional[TileTensor[out_idx_type, KLayoutType, ImmutAnyOrigin]] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
    top_p: Optional[
        TileTensor[DType.float32, TopPLayoutType, ImmutAnyOrigin]
    ] = None,
    min_p: Optional[
        TileTensor[DType.float32, MinPLayoutType, ImmutAnyOrigin]
    ] = None,
    rng_seed: Optional[
        TileTensor[DType.uint64, SeedLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """Top-K + top-P + min-P sampling.

    Applies softmax with per-row temperature scaling, optionally masks
    probabilities below ``min_p * max_prob``, then performs top-k+top-p
    rejection sampling via the dual-pivot algorithm.
    """
    var shape = coord_to_index_list(input.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    # Step 1: softmax with temperature.
    var probs_buf = ctx.enqueue_create_buffer[dtype](batch_size * d)
    var probs = TileTensor(
        probs_buf,
        row_major(Coord(IndexList[2](batch_size, d))),
    )
    softmax_with_temperature(
        ctx,
        input,
        probs,
        temperature_arr=temperature,
    )

    # Step 1b: apply min_p mask (zero probs below min_p * max_prob).
    if min_p:
        comptime MASK_BLOCK_SIZE = 256
        comptime mask_kernel = apply_min_p_mask_kernel[dtype, MASK_BLOCK_SIZE]
        ctx.enqueue_function[mask_kernel](
            probs_buf,
            min_p.value().to_device_buffer(ctx),
            d,
            grid_dim=batch_size,
            block_dim=MASK_BLOCK_SIZE,
        )

    # Step 2: top-k + top-p rejection sampling from probabilities.
    # Reshape out_idxs from [batch, 1] (rank 2) to [batch] (rank 1).
    var out_shape = coord_to_index_list(out_idxs.layout.shape_coord())
    var out_1d = TileTensor(
        out_idxs.ptr,
        row_major(Idx(out_shape[0])),
    )
    topk_topp_sampling_from_prob[dtype, out_idx_type](
        ctx,
        probs,
        out_1d,
        max_k,
        top_p_val=min_top_p,
        top_k_arr=k,
        top_p_arr=top_p,
        rng_seed=rng_seed,
    )

    _ = probs_buf^


@always_inline
def fused_token_sampling_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    KLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[
        RuntimeInt[DType.int64]
    ],
    TopPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
    MinPLayoutType: TensorLayout = RowMajorLayout[RuntimeInt[DType.int64]],
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
    min_p: Optional[
        TileTensor[DType.float32, MinPLayoutType, ImmutAnyOrigin]
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

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    @parameter
    def trace_information() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", input_shape, dtype),
                    "max_k=" + String(max_k),
                    "min_top_p=" + String(min_top_p),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "fused_token_sampling_gpu",
        Trace[TraceLevel.OP]._get_detail_str[trace_information](),
        task_id=Int(ctx.id()),
    ):
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

        comptime assert input.flat_rank == 2

        var vocab_size = Int(input.layout.shape[1]().value())
        var adjusted_max_k = vocab_size if max_k == -1 else max_k

        # softmax with temperature, then top-k+top-p rejection sampling.

        if adjusted_max_k >= 32:
            _topk_topp_sampling_fi[dtype, out_idx_type](
                ctx,
                adjusted_max_k,
                min_top_p,
                input,
                out_idxs,
                k=rebind[
                    Optional[
                        TileTensor[out_idx_type, KLayoutType, ImmutAnyOrigin]
                    ]
                ](k),
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                rng_seed=seed,
            )
            return

        var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
        out_vals_shape[input.rank - 1] = adjusted_max_k
        var out_vals_buf = ctx.enqueue_create_buffer[dtype](
            out_vals_shape.flattened_length()
        )
        var out_vals = TileTensor(
            out_vals_buf,
            row_major(Coord(out_vals_shape)),
        )

        topk_gpu[sampling=True, largest=True](
            ctx,
            adjusted_max_k,
            input,
            out_vals,
            out_idxs,
            k=k,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            block_size=block_size,
            num_blocks_per_input=num_blocks_per_input,
            seed=seed,
        )

        _ = out_vals_buf^


# ===-----------------------------------------------------------------------===#
# Sampling Kernel with the Gumbel-max trick
# ===-----------------------------------------------------------------------===#


@__name(t"apply_gumbel_noise_{dtype}", mangle=True)
def apply_gumbel_noise_kernel[
    dtype: DType,
    OutputLayoutType: TensorLayout,
    InputLayoutType: TensorLayout,
    num_sms: Int,
    num_threads: Int,
](
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InputLayoutType, ImmutAnyOrigin],
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    seed: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]],
):
    comptime EPS = Float32(1e-20)
    comptime LOG2 = Float32(0.6931471806)

    comptime simd_width = simd_width_of[dtype]()
    var N = Int(input.dim(1))
    comptime num_blocks_per_token = 8
    comptime group_size = num_blocks_per_token * num_threads
    comptime num_groups = num_sms // num_blocks_per_token

    var tid = thread_idx.x
    var sm_id = block_idx.x
    var group_id, sm_id_rem = divmod(sm_id, num_blocks_per_token)
    var tid_in_group = tid + sm_id_rem * num_threads

    var num_tokens = input.dim[0]()

    comptime assert (
        simd_width % 4 == 0
    ), "SIMD width must be divisible by 4 to match RNG output size."

    # split workload across blocks
    with PDL():
        if sm_id >= num_groups * num_blocks_per_token:
            return

        for tok_idx in range(group_id, Int(num_tokens), num_groups):
            var temp_val = Float32(1.0)
            if temperature:
                temp_val = temperature.unsafe_value()[tok_idx]

            var seed_val = UInt64(0)
            if seed:
                seed_val = seed.unsafe_value()[tok_idx]

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
def gumbel_sampling_gpu[
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

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    @parameter
    def trace_information() -> String:
        return trace_arg("input", input_shape, dtype)

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "gumbel_sampling_gpu",
        Trace[TraceLevel.OP]._get_detail_str[trace_information](),
        task_id=Int(ctx.id()),
    ):
        # create a buffer to hold the Gumbel noise applied input
        var noised_input_buf = ctx.enqueue_create_buffer[dtype](
            input.num_elements()
        )
        var noised_input = TileTensor(noised_input_buf, input.layout)

        comptime hw_info = ctx.default_device_info
        comptime gumbel_kernel = apply_gumbel_noise_kernel[
            dtype,
            noised_input.LayoutType,
            input.LayoutType,
            hw_info.sm_count,
            hw_info.max_thread_block_size,
        ]

        var temperature_ptr: Optional[
            UnsafePointer[Float32, ImmutAnyOrigin]
        ] = None
        if temperature:
            temperature_ptr = temperature.value().ptr
        var seed_ptr: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]] = None
        if seed:
            seed_ptr = seed.value().ptr

        ctx.enqueue_function[gumbel_kernel](
            noised_input,
            input.as_immut(),
            temperature_ptr,
            seed_ptr,
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

        # Extract argmax after Gumbel noise application.
        var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
        out_vals_shape[input.rank - 1] = 1
        var out_vals_buf = ctx.enqueue_create_buffer[dtype](
            out_vals_shape.flattened_length()
        )
        var out_vals = TileTensor(
            out_vals_buf,
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
