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

from bit import log2_floor
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
from gpu.primitives import block, warp
from gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host.dim import Dim
from gpu.memory import AddressSpace, external_memory
from layout._coord import (
    Coord,
    CoordLike,
    Idx,
    RuntimeInt,
    ComptimeInt,
    coord_to_index_list,
)
from layout._layout import row_major, TensorLayout, Layout
from layout._tile_tensor import TileTensor
from math import ceildiv, gcd, exp
from memory import stack_allocation
from os import Atomic
from random import Random
from sys import align_of, bit_width_of, simd_width_of, size_of


@always_inline
fn get_min_max_value[
    vec_size: Int,
    block_size: Int,
    dtype: DType,
](
    in_data: UnsafePointer[Scalar[dtype]],
    row_idx: Int,
    d: Int,
) -> Tuple[
    Float32, Float32
]:
    """Compute the minimum and maximum values from input data using block reduction.

    Parameters:
        vec_size: Number of elements each thread processes per iteration (vectorization width).
        block_size: Number of threads per block.
        dtype: The dtype of the input data.

    Args:
        in_data: Pointer to input data buffer.
        row_idx: Row index for the current block (for 2D data access).
        d: Total number of elements in the row.

    Returns:
        Tuple containing [min_val, max_val].
    """
    var tx = thread_idx.x

    # Initialize running min/max values across all iterations.
    var max_val = Float32.MIN
    var min_val = Float32.MAX

    var num_iterations = ceildiv(d, block_size * vec_size)
    for i in range(num_iterations):
        var in_data_vec = SIMD[DType.float32, vec_size](0)

        if (i * block_size + Int(tx)) * vec_size < d:
            var offset = (
                row_idx * d + i * block_size * vec_size + Int(tx) * vec_size
            )
            in_data_vec = in_data.load[width=vec_size](offset).cast[
                DType.float32
            ]()

        max_val = max(
            max_val,
            block.max[block_size=block_size, broadcast=True](
                in_data_vec.reduce_max()
            ),
        )

        min_val = min(
            min_val,
            block.min[block_size=block_size, broadcast=True](
                in_data_vec.reduce_min()
            ),
        )

    return Tuple[Float32, Float32](min_val, max_val)


fn TopKMaskLogitsKernel[
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmutOrigin,
    MaskedLogitsLayoutType: TensorLayout,
    masked_logits_origin: MutOrigin,
](
    logits: TileTensor[dtype, LogitsLayoutType, logits_origin],
    masked_logits: TileTensor[
        dtype, MaskedLogitsLayoutType, masked_logits_origin
    ],
    top_k_arr: UnsafePointer[Scalar[out_idx_type], MutExternalOrigin],
    top_k_val: Int,
    d: Int,
):
    var bx = Int(block_idx.x)
    var tx = Int(thread_idx.x)
    var row_idx = bx

    var logits_ptr = logits.ptr + bx * d
    var masked_logits_ptr = masked_logits.ptr + bx * d

    var logits_row = TileTensor(logits_ptr, row_major((Idx[1](), Idx(d))))
    var masked_logits_row = TileTensor(
        masked_logits_ptr, row_major((Idx[1](), Idx(d)))
    )

    var k = top_k_val
    if top_k_arr:
        k = Int(top_k_arr[bx])

    # Initialize pivot to negative infinity.
    var pivot = Float64(Float32.MIN)

    var logits_vec = SIMD[DType.float32, vec_size]()

    if k < d:
        var min_max = get_min_max_value[vec_size, block_size](
            logits.ptr, row_idx, d
        )
        var min_val, max_val = min_max[0], min_max[1]

        # Initialize ternary search bounds.
        var low = Float64(
            min_val - 1 if min_val != Float32.MIN else Float32.MIN
        )
        var high = Float64(max_val)

        while True:
            var pivot_0 = (high + 2 * low) / 3
            var pivot_1 = (2 * high + low) / 3

            var aggregate_gt_pivot_0: Int32 = 0
            var aggregate_gt_pivot_1: Int32 = 0
            var min_gt_low = Float32(high)
            var max_le_high = Float32(low)

            for i in range(ceildiv(d, block_size * vec_size)):
                if (i * block_size + tx) * vec_size < d:
                    logits_vec = logits_row.load[width=vec_size](
                        (
                            Idx[0](),
                            Idx(i * block_size * vec_size + tx * vec_size),
                        ),
                    ).cast[DType.float32]()

                var probs_gt_pivot_0_count = SIMD[DType.int32, vec_size]()
                var probs_gt_pivot_1_count = SIMD[DType.int32, vec_size]()

                @parameter
                for j in range(vec_size):
                    # Calculate the global index for this element in the row.
                    # Will only count if the index is within the valid range [0, d).
                    var idx = (i * block_size + tx) * vec_size + j

                    # Count elements greater than pivot_0 (higher ternary search bound).
                    probs_gt_pivot_0_count[j] = Int32(1) if (
                        Float64(logits_vec[j]) > pivot_0 and idx < d
                    ) else Int32(0)
                    # Count elements greater than pivot_1 (lower ternary search bound).
                    probs_gt_pivot_1_count[j] = Int32(1) if (
                        Float64(logits_vec[j]) > pivot_1 and idx < d
                    ) else Int32(0)

                    # Track the minimum value that's greater than 'low'.
                    # Used to narrow the search range from below.
                    if Float64(logits_vec[j]) > low and idx < d:
                        min_gt_low = min(min_gt_low, logits_vec[j])
                    # Track the maximum value that's less than or equal to 'high'.
                    # Used to narrow the search range from above.
                    if Float64(logits_vec[j]) <= high and idx < d:
                        max_le_high = max(max_le_high, logits_vec[j])

                # Reduce the counts across all threads in the block.
                var thread_count_0 = probs_gt_pivot_0_count.reduce_add()
                var thread_count_1 = probs_gt_pivot_1_count.reduce_add()

                # Sum the counts across all threads in the block.
                aggregate_gt_pivot_0 += block.sum[
                    block_size=block_size, broadcast=True
                ](thread_count_0)
                aggregate_gt_pivot_1 += block.sum[
                    block_size=block_size, broadcast=True
                ](thread_count_1)

            # Find the minimum value that's greater than 'low' across all threads in the block.
            min_gt_low = block.min[block_size=block_size, broadcast=True](
                min_gt_low
            )

            # Find the maximum value that's less than or equal to 'high' across all threads in the block.
            max_le_high = block.max[block_size=block_size, broadcast=True](
                max_le_high
            )

            # Update the search bounds based on the counts and the minimum/maximum values.
            if aggregate_gt_pivot_1 >= Int32(k):
                low = pivot_1
            elif aggregate_gt_pivot_0 >= Int32(k):
                low = pivot_0
                high = min(pivot_1, Float64(max_le_high))
            else:
                high = min(pivot_0, Float64(max_le_high))

            if min_gt_low == max_le_high:
                break

        pivot = low

    for i in range(ceildiv(d, block_size * vec_size)):
        logits_vec = 0
        if (i * block_size + tx) * vec_size < d:
            logits_vec = logits_row.load[width=vec_size](
                (Idx[0](), Idx(i * block_size * vec_size + tx * vec_size))
            ).cast[DType.float32]()

        logits_vec = (logits_vec.cast[DType.float64]().gt(pivot)).select(
            logits_vec, Float32.MIN
        )

        if (i * block_size + tx) * vec_size < d:
            masked_logits_row.store[width=vec_size](
                (Idx[0](), Idx(i * block_size * vec_size + tx * vec_size)),
                logits_vec.cast[dtype](),
            )


fn topk_mask_logits[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types = Variadic.types[RuntimeInt[DType.int64]],
        stride_types = Variadic.types[ComptimeInt[1]],
    ],
](
    ctx: DeviceContext,
    logits: TileTensor[dtype, ...],
    masked_logits: TileTensor[mut=True, dtype, ...],
    top_k_val: Int,
    top_k_arr: Optional[
        TileTensor[out_idx_type, TopKArrLayoutType, MutExternalOrigin]
    ] = None,
) raises:
    comptime assert logits.rank == 2, "logits rank must be 2"
    comptime assert (
        logits.rank == masked_logits.rank
    ), "logits.rank must match masked_logits.rank"

    var shape = coord_to_index_list(logits.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    var out_shape = coord_to_index_list(masked_logits.layout.shape_coord())
    if shape[0] != out_shape[0] or shape[1] != out_shape[1]:
        raise Error("masked_logits shape must match logits shape")

    # Computes optimal vectorization width: find the largest vec_size that divides
    # both max hardware vector size (16 bytes / element size) and dim d.
    var vec_size = gcd(16 // size_of[dtype](), d)

    var top_k_buf: DeviceBuffer[out_idx_type]
    if top_k_arr:
        top_k_buf = top_k_arr.value().to_device_buffer(ctx)
    else:
        top_k_buf = DeviceBuffer[out_idx_type](ctx, {}, 0, owning=False)

    @parameter
    fn launch_kernel[vec_size: Int]() raises:
        comptime kernel = TopKMaskLogitsKernel[
            block_size,
            vec_size,
            dtype,
            out_idx_type,
            LogitsLayoutType = logits.LayoutType,
            logits_origin = ImmutOrigin(logits.origin),
            MaskedLogitsLayoutType = masked_logits.LayoutType,
            masked_logits_origin = masked_logits.origin,
        ]
        ctx.enqueue_function[kernel, kernel](
            logits.as_immut(),
            masked_logits,
            top_k_buf,
            top_k_val,
            d,
            grid_dim=batch_size,
            block_dim=block_size,
            attributes=pdl_launch_attributes(),
        )

    # Runtime dispatch to compile-time parameter.
    @parameter
    for param_vec_size in [16, 8, 4, 2, 1]:
        if vec_size == param_vec_size:
            return launch_kernel[param_vec_size]()


@always_inline
fn device_sampling_from_prob[
    vec_size: Int,
    block_size: Int,
    dtype: DType,
    deterministic: Bool = False,
](
    i: Int,
    d: Int,
    low: Float64,
    u: Float32,
    prob_vec: SIMD[DType.float32, vec_size],
    aggregate: Float32,
    sampled_id_sram: UnsafePointer[
        mut=True, Int, address_space = AddressSpace.SHARED
    ],
    last_valid_id_sram: UnsafePointer[
        mut=True, Int, address_space = AddressSpace.SHARED
    ],
) -> Float32:
    """Device-level sampling from probability distribution with atomic operations.
    """

    var tx = Int(thread_idx.x)

    # Step 1: Filter probabilities based on predicate (prob > low).
    var prob_gt_threshold = SIMD[DType.float32, vec_size]()
    var valid = SIMD[DType.bool, vec_size]()

    @parameter
    for j in range(vec_size):
        var idx = (i * block_size + tx) * vec_size + j
        var passes_pred = prob_vec[j] > Float32(low)
        prob_gt_threshold[j] = prob_vec[j] if passes_pred else 0.0
        valid[j] = passes_pred and (idx < d)

    # Step 2: Block reduce to get sum of filtered probabilities.
    var thread_sum = prob_gt_threshold.reduce_add()

    var aggregate_local = block.sum[
        block_size=block_size,
        broadcast=True,
    ](thread_sum)

    # Step 3: Check if we found the sampled index in this chunk.
    if aggregate + aggregate_local > u:
        # Step 4: Thread-local prefix sum.
        # Intra-SIMD prefix sum using shift operations.
        var local_inclusive_cdf = prob_gt_threshold  # Start with the values

        @parameter
        for i in range(log2_floor(vec_size)):
            # Shift right by 2^i positions (filling with zeros)
            # and add to accumulate prefix sums.
            local_inclusive_cdf += local_inclusive_cdf.shift_right[2**i]()

        # Step 5: Block-level exclusive scan.
        var thread_total = local_inclusive_cdf[vec_size - 1]
        var prefix_from_prev_threads = block.prefix_sum[
            dtype = DType.float32,
            block_size=block_size,
            exclusive=True,
        ](thread_total)

        # Step 6: Compute global inclusive CDF.
        var global_inclusive_cdf = (
            local_inclusive_cdf + prefix_from_prev_threads
        )

        # Step 7: Find first index where cumulative > u using atomic min.
        @parameter
        for j in range(vec_size):
            var idx = (i * block_size + tx) * vec_size + j
            if (global_inclusive_cdf[j] + aggregate > u) and valid[j]:
                # Atomic min to ensure we get the smallest index across all threads.
                Atomic.min(sampled_id_sram.bitcast[Int32](), Int32(idx))
                break

        barrier()

    # Step 8: Update last valid index using atomic max.
    var max_valid_idx = -1

    @parameter
    for j in range(vec_size):
        var idx = (i * block_size + tx) * vec_size + j
        if valid[j]:
            max_valid_idx = idx

    var block_max_valid = block.max[
        block_size=block_size,
        broadcast=False,
    ](Int32(max_valid_idx))

    if tx == 0 and block_max_valid != -1:
        last_valid_id_sram[0] = Int(block_max_valid)

    barrier()

    # Step 9: Update aggregate for next iteration.
    return aggregate + aggregate_local


struct ValueCount[T: DType](Defaultable, TrivialRegisterPassable):
    """A struct that holds a value and a count, used for block reductions.

    This is useful for computing both the sum of values and the count
    of elements that satisfy a condition in a single reduction pass.

    Parameters:
        T: The DType of the value field.
    """

    var value: Scalar[Self.T]
    var count: Int32

    fn __init__(out self, value: Scalar[Self.T], count: Int32):
        # Initialize a ValueCount instance.
        self.value = value
        self.count = count

    fn __init__(out self):
        # Zero-initialize a ValueCount instance.
        self.value = 0
        self.count = 0

    fn __add__(self, other: Self) -> Self:
        # Add two ValueCount instances (element-wise).
        return {self.value + other.value, self.count + other.count}

    fn __iadd__(mut self, other: Self):
        # In-place addition of another ValueCount.
        self.value += other.value
        self.count += other.count


@always_inline
fn _warp_reduce_value_count[T: DType](val: ValueCount[T]) -> ValueCount[T]:
    """Warp-level reduction for ValueCount using shuffle operations.

    Reduces both value and count fields across all lanes in a warp.

    Parameters:
        T: DType of the value field.

    Args:
        val: The ValueCount from this thread's lane.

    Returns:
        ValueCount with both fields reduced across the warp (only valid in lane 0).
    """
    var result = val

    comptime limit = log2_floor(WARP_SIZE)

    # Reduce across warp lanes using shuffle_down.
    @parameter
    for i in reversed(range(limit)):
        comptime offset = 1 << i
        result.value += warp.shuffle_down(result.value, UInt32(offset))
        result.count += warp.shuffle_down(result.count, UInt32(offset))
    return result


@always_inline
fn _block_reduce_value_count[
    T: DType,
    broadcast: Bool = False,
](val: ValueCount[T]) -> ValueCount[T]:
    """Block-level reduction for ValueCount struct.

    Reduces both value and count fields across all threads in a block.

    Parameters:
        T: DType of the value field.
        broadcast: If True, all threads get the reduced result.
                   If False, only thread 0 has the correct result.

    Args:
        val: The ValueCount from this thread.

    Returns:
        ValueCount with both fields reduced across the entire block.
        If broadcast=True, all threads get the same result.
        If broadcast=False, only thread 0 has the valid result.
    """
    comptime MAX_BLOCK_SIZE = 1024
    comptime assert (
        MAX_BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"

    comptime value_width = simd_width_of[Scalar[T]]()
    comptime count_width = simd_width_of[DType.int32]()

    var value_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * value_width,
        Scalar[T],
        address_space = AddressSpace.SHARED,
    ]()
    var count_sram = stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * count_width,
        Int32,
        address_space = AddressSpace.SHARED,
    ]()

    var warp = warp_id()
    comptime num_warps_needed = MAX_BLOCK_SIZE // WARP_SIZE

    var warp_accum = _warp_reduce_value_count(val)

    # Store warp-level results in shared memory (only lane 0 of each warp).
    if lane_id() == 0 and warp < UInt(num_warps_needed):
        value_sram[Int(warp) * value_width] = warp_accum.value
        count_sram[Int(warp) * count_width] = warp_accum.count
    barrier()

    # Each warp has reduced its own ValueCount in smem (value_sram and count_sram).
    # Below we perform block-level reduction (across all warps) to get final result.
    # Only the first N threads from warp 0 will have valid results in the corresponding
    # smem slots above and participate in the final warp-level reduction (e.g. if
    # block_size = 1024 and WARP_SIZE = 32, then only the first 32 threads from warp 0
    # will have valid results).
    var block_accum: ValueCount[T]
    var thread_in_final_warp = thread_idx.x < block_dim.x // UInt(WARP_SIZE)

    if thread_in_final_warp:
        block_accum = {
            value = value_sram[lane_id() * UInt(value_width)],
            count = count_sram[lane_id() * UInt(count_width)],
        }
    else:
        # Initialize unused threads with zeros (identity for sum).
        block_accum = {value = Scalar[T](0), count = 0}

    # Perform final warp-level reduction.
    var result = _warp_reduce_value_count(block_accum)

    @parameter
    if broadcast:
        if thread_idx.x == 0:
            value_sram[0] = result.value
            count_sram[0] = result.count

        barrier()

        result = {
            value = value_sram[0],
            count = count_sram[0],
        }

    return result


fn TopKSamplingFromProbKernel[
    ProbsLayoutType: TensorLayout,
    probs_origin: ImmutOrigin,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    deterministic: Bool,
](
    probs: TileTensor[dtype, ProbsLayoutType, probs_origin],
    output: TileTensor[out_idx_type, OutputLayoutType, output_origin],
    indices: UnsafePointer[Scalar[out_idx_type], MutExternalOrigin],
    top_k_arr: UnsafePointer[Scalar[out_idx_type], MutExternalOrigin],
    top_k_val: Int,
    d: Int,
    rng_seed: UInt64,
    rng_offset: UInt64,
):
    """Kernel for top-k sampling from probability distribution.

    This kernel performs top-k sampling by:
    1. Using ternary search to find a pivot threshold.
    2. Rejecting samples iteratively until acceptance criteria is met.
    3. Sampling an index using uniform random numbers from Random generator.

    Args:
        probs: Input probability distribution [batch_size, d].
        output: Output sampled indices [batch_size].
        indices: Optional row indices for batch indexing [batch_size].
        top_k_arr: Optional per-row top_k values [batch_size].
        top_k_val: Default top_k value if top_k_arr is null.
        d: Vocabulary size.
        rng_seed: Random seed for Random number generator.
        rng_offset: Random offset for Random number generator.
    """
    comptime assert output.flat_rank == 1

    var bx = Int(block_idx.x)
    var tx = Int(thread_idx.x)

    var sampled_id_sram = stack_allocation[
        1, Int, address_space = AddressSpace.SHARED
    ]()
    var last_valid_id_sram = stack_allocation[
        1, Int, address_space = AddressSpace.SHARED
    ]()

    var generator = Random(seed=rng_seed, offset=UInt64(bx) + rng_offset)
    var k = top_k_val
    if top_k_arr:
        k = Int(top_k_arr.load(bx))
    var row_idx = bx
    if indices:
        row_idx = Int(indices.load(bx))

    var probs_ptr = probs.ptr + row_idx * d
    var probs_row = TileTensor(probs_ptr, row_major((Idx[1](), Idx(d))))

    var probs_vec: SIMD[DType.float32, vec_size]
    var aggregate: Float32
    var sampled_id = 0
    var q: Float32 = 1.0
    var low = 0.0
    var high = 1.0
    var round = 0

    while low < high:
        round += 1

        if tx == 0:
            sampled_id_sram[0] = d
            last_valid_id_sram[0] = -1
        barrier()

        var u = generator.step_uniform()[0] * q
        aggregate = 0.0

        for i in range(ceildiv(d, block_size * vec_size)):
            probs_vec = 0
            if (i * block_size + tx) * vec_size < d:
                probs_vec = probs_row.load[width=vec_size](
                    (Idx[0](), Idx((i * block_size + tx) * vec_size))
                ).cast[DType.float32]()

            aggregate = device_sampling_from_prob[
                vec_size, block_size, dtype, deterministic
            ](
                i,
                d,
                low,
                u,
                probs_vec,
                aggregate,
                sampled_id_sram,
                last_valid_id_sram,
            )
            if aggregate > u:
                break

        barrier()

        sampled_id = sampled_id_sram[0]
        if sampled_id == d:
            # This would happen when u is very close to 1 and the
            # sum of probabilities is smaller than u. In this case
            # we use the last valid index as the sampled id.
            sampled_id = last_valid_id_sram[0]

        var pivot_0 = Float64(
            probs_row.load[width=1]((Idx[0](), Idx(sampled_id)))
        )
        var pivot_1 = (pivot_0 + high) / 2.0

        var aggregate_gt_pivot_0 = ValueCount[DType.float32](0.0, 0)
        var aggregate_gt_pivot_1 = ValueCount[DType.float32](0.0, 0)

        for i in range(ceildiv(d, block_size * vec_size)):
            probs_vec = 0
            if (i * block_size + tx) * vec_size < d:
                probs_vec = probs_row.load[width=vec_size](
                    (Idx[0](), Idx((i * block_size + tx) * vec_size))
                ).cast[DType.float32]()

            var probs_gt_pivot_0_values = SIMD[DType.float32, vec_size]()
            var probs_gt_pivot_0_counts = SIMD[DType.int32, vec_size]()
            var probs_gt_pivot_1_values = SIMD[DType.float32, vec_size]()
            var probs_gt_pivot_1_counts = SIMD[DType.int32, vec_size]()

            @parameter
            for j in range(vec_size):
                var idx = (i * block_size + tx) * vec_size + j
                var is_valid = idx < d

                # For pivot_0.
                var gt_pivot_0 = probs_vec[j] > Float32(pivot_0)
                probs_gt_pivot_0_values[j] = probs_vec[j] if gt_pivot_0 else 0.0
                probs_gt_pivot_0_counts[j] = Int32(1) if (
                    gt_pivot_0 and is_valid
                ) else Int32(0)

                # For pivot_1.
                var gt_pivot_1 = probs_vec[j] > Float32(pivot_1)
                probs_gt_pivot_1_values[j] = probs_vec[j] if gt_pivot_1 else 0.0
                probs_gt_pivot_1_counts[j] = Int32(1) if (
                    gt_pivot_1 and is_valid
                ) else Int32(0)

            var thread_value_0 = probs_gt_pivot_0_values.reduce_add()
            var thread_count_0 = probs_gt_pivot_0_counts.reduce_add()
            var thread_value_1 = probs_gt_pivot_1_values.reduce_add()
            var thread_count_1 = probs_gt_pivot_1_counts.reduce_add()

            var thread_vc_0 = ValueCount[DType.float32](
                thread_value_0, thread_count_0
            )
            var thread_vc_1 = ValueCount[DType.float32](
                thread_value_1, thread_count_1
            )

            # Block reduce with broadcast (all threads get the result).
            var block_vc_0 = _block_reduce_value_count[
                DType.float32, broadcast=True
            ](thread_vc_0)
            var block_vc_1 = _block_reduce_value_count[
                DType.float32, broadcast=True
            ](thread_vc_1)

            # Add to running aggregates.
            aggregate_gt_pivot_0 += block_vc_0
            aggregate_gt_pivot_1 += block_vc_1

        if aggregate_gt_pivot_0.count < Int32(k):
            # Case 1: pivot_0 accepted - found acceptable threshold.
            break

        if aggregate_gt_pivot_1.count < Int32(k):
            # Case 2: pivot_0 rejected, pivot_1 accepted.
            # Narrow search to [pivot_0, pivot_1].
            low = pivot_0
            high = pivot_1
            q = aggregate_gt_pivot_0.value
        else:
            # Case 3: both pivots rejected.
            # Search in [pivot_1, high].
            low = pivot_1
            q = aggregate_gt_pivot_1.value

    barrier()

    if tx == 0:
        output[bx] = Scalar[out_idx_type](sampled_id)


fn topk_sampling_from_prob[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types = Variadic.types[RuntimeInt[DType.int64]],
        stride_types = Variadic.types[ComptimeInt[1]],
    ],
    IndicesLayoutType: TensorLayout = Layout[
        shape_types = Variadic.types[RuntimeInt[DType.int64]],
        stride_types = Variadic.types[ComptimeInt[1]],
    ],
](
    ctx: DeviceContext,
    probs: TileTensor[dtype, ...],
    output: TileTensor[mut=True, out_idx_type, ...],
    top_k_val: Int,
    deterministic: Bool = False,
    rng_seed: UInt64 = 0,
    rng_offset: UInt64 = 0,
    indices: Optional[
        TileTensor[out_idx_type, IndicesLayoutType, MutExternalOrigin]
    ] = None,
    top_k_arr: Optional[
        TileTensor[out_idx_type, TopKArrLayoutType, MutExternalOrigin]
    ] = None,
) raises:
    """Top-K sampling from probability distribution.

    Performs stochastic sampling from a probability distribution, considering only
    the top-k most probable tokens. Uses rejection sampling with ternary search
    to efficiently find appropriate samples.

    Args:
        ctx: Device context for kernel execution.
        probs: Input probability distribution [batch_size, d].
        output: Output sampled indices [batch_size].
        top_k_val: Default top-k value (number of top tokens to consider).
        deterministic: Whether to use deterministic sampling.
        rng_seed: Random seed for Random number generator.
        rng_offset: Random offset for Random number generator.
        indices: Optional row indices for batch indexing [batch_size].
        top_k_arr: Optional per-row top-k values [batch_size].

    Raises:
        Error: If tensor ranks or shapes are invalid.
    """

    comptime assert probs.rank == 2, "probs rank must be 2"
    comptime assert output.rank == 1, "output rank must be 1"

    var shape = coord_to_index_list(probs.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    var out_shape = coord_to_index_list(output.layout.shape_coord())
    if out_shape[0] != batch_size:
        raise Error("output batch size must match probs batch size")

    # Computes optimal vectorization width: find the largest vec_size that divides
    # both max hardware vector size (16 bytes / element size) and dim d.
    var vec_size = gcd(16 // size_of[dtype](), d)

    var indices_buf: DeviceBuffer[out_idx_type]
    if indices:
        indices_buf = indices.value().to_device_buffer(ctx)
    else:
        indices_buf = DeviceBuffer[out_idx_type](ctx, {}, 0, owning=False)
    var top_k_buf: DeviceBuffer[out_idx_type]
    if top_k_arr:
        top_k_buf = top_k_arr.value().to_device_buffer(ctx)
    else:
        top_k_buf = DeviceBuffer[out_idx_type](ctx, {}, 0, owning=False)

    @parameter
    fn launch_kernel[vec_size: Int, deterministic: Bool]() raises:
        comptime kernel = TopKSamplingFromProbKernel[
            probs.LayoutType,
            ImmutOrigin(probs.origin),
            output.LayoutType,
            output.origin,
            block_size,
            vec_size,
            dtype,
            out_idx_type,
            deterministic,
        ]
        ctx.enqueue_function[kernel, kernel](
            probs.as_immut(),
            output,
            indices_buf,
            top_k_buf,
            top_k_val,
            d,
            rng_seed,
            rng_offset,
            grid_dim=batch_size,
            block_dim=block_size,
            attributes=pdl_launch_attributes(),
        )

    # Runtime dispatch to compile-time parameter.
    @parameter
    fn dispatch_vec_size[deterministic: Bool]() raises:
        @parameter
        for param_vec_size in [16, 8, 4, 2, 1]:
            if vec_size == param_vec_size:
                return launch_kernel[param_vec_size, deterministic]()

    # Dispatch on deterministic flag.
    if deterministic:
        dispatch_vec_size[True]()
    else:
        dispatch_vec_size[False]()


fn TopKSoftmaxSampleKernel[
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmutOrigin,
    SampledLayoutType: TensorLayout,
    sampled_origin: MutOrigin,
](
    logits: TileTensor[dtype, LogitsLayoutType, logits_origin],
    sampled_indices: TileTensor[
        out_idx_type, SampledLayoutType, sampled_origin
    ],
    top_k_arr: UnsafePointer[Scalar[out_idx_type], MutExternalOrigin],
    top_k_val: Int,
    temperature_val: Float32,
    temperature: UnsafePointer[Float32, MutExternalOrigin],
    seed_val: UInt64,
    seed: UnsafePointer[UInt64, MutExternalOrigin],
    d: Int,
):
    comptime assert sampled_indices.flat_rank == 1

    var bx = Int(block_idx.x)
    var tx = Int(thread_idx.x)
    var row_idx = bx

    var logits_ptr = logits.ptr + bx * d

    var logits_row = TileTensor(logits_ptr, row_major((Idx[1](), Idx(d))))

    var k = top_k_val
    if top_k_arr:
        k = Int(top_k_arr[bx])
    var temp_val = temperature_val
    if temperature:
        temp_val = max(temperature[bx], 1e-6)

    # Allocate shared memory for caching top-k elements.
    # Round up to ensure proper alignment for Int array.
    var k_rounded = ceildiv(k, WARP_SIZE) * WARP_SIZE

    var s_vals = external_memory[
        Float32,
        address_space = AddressSpace.SHARED,
        alignment = align_of[Float32](),
    ]()
    var s_idxs = (s_vals + k_rounded).bitcast[Int]()
    var s_count = stack_allocation[
        1, Int, address_space = AddressSpace.SHARED
    ]()
    if tx == 0:
        s_count[0] = 0

    # PHASE 1: Find pivot (k-th largest) via ternary search.
    var pivot = Float64(Float32.MIN)
    var max_logit: Float32
    var logits_vec = SIMD[DType.float32, vec_size]()

    if k < d:
        var min_max = get_min_max_value[vec_size, block_size](
            logits.ptr, row_idx, d
        )
        var min_val, max_val = min_max[0], min_max[1]

        max_logit = max_val

        # Initialize ternary search bounds.
        var low = Float64(
            min_val - 1 if min_val != Float32.MIN else Float32.MIN
        )
        var high = Float64(max_val)

        while True:
            var pivot_0 = (high + 2 * low) / 3
            var pivot_1 = (2 * high + low) / 3

            var aggregate_gt_pivot_0: Int32 = 0
            var aggregate_gt_pivot_1: Int32 = 0
            var min_gt_low = Float32(high)
            var max_le_high = Float32(low)

            for i in range(ceildiv(d, block_size * vec_size)):
                if (i * block_size + tx) * vec_size < d:
                    logits_vec = logits_row.load[width=vec_size](
                        (
                            Idx[0](),
                            Idx(i * block_size * vec_size + tx * vec_size),
                        )
                    ).cast[DType.float32]()

                var probs_gt_pivot_0_count = SIMD[DType.int32, vec_size]()
                var probs_gt_pivot_1_count = SIMD[DType.int32, vec_size]()

                @parameter
                for j in range(vec_size):
                    var idx = (i * block_size + tx) * vec_size + j

                    probs_gt_pivot_0_count[j] = Int32(1) if (
                        Float64(logits_vec[j]) > pivot_0 and idx < d
                    ) else Int32(0)
                    probs_gt_pivot_1_count[j] = Int32(1) if (
                        Float64(logits_vec[j]) > pivot_1 and idx < d
                    ) else Int32(0)

                    if Float64(logits_vec[j]) > low and idx < d:
                        min_gt_low = min(min_gt_low, logits_vec[j])
                    if Float64(logits_vec[j]) <= high and idx < d:
                        max_le_high = max(max_le_high, logits_vec[j])

                var thread_count_0 = probs_gt_pivot_0_count.reduce_add()
                var thread_count_1 = probs_gt_pivot_1_count.reduce_add()

                aggregate_gt_pivot_0 += block.sum[
                    block_size=block_size, broadcast=True
                ](thread_count_0)
                aggregate_gt_pivot_1 += block.sum[
                    block_size=block_size, broadcast=True
                ](thread_count_1)

            min_gt_low = block.min[block_size=block_size, broadcast=True](
                min_gt_low
            )
            max_le_high = block.max[block_size=block_size, broadcast=True](
                max_le_high
            )

            if aggregate_gt_pivot_1 >= Int32(k):
                low = pivot_1
            elif aggregate_gt_pivot_0 >= Int32(k):
                low = pivot_0
                high = min(pivot_1, Float64(max_le_high))
            else:
                high = min(pivot_0, Float64(max_le_high))

            if min_gt_low == max_le_high:
                break

        pivot = low
    else:
        # If k >= d, include all elements.
        var min_max = get_min_max_value[vec_size, block_size](
            logits.ptr, row_idx, d
        )
        max_logit = min_max[1]

    barrier()

    # PHASE 2: Compute softmax sum and cache top-k elements.

    # All threads cooperatively collect elements > pivot.
    var thread_sum = Float32(0.0)

    # Use atomic counter in shared memory for write position.
    var s_write_idx = stack_allocation[
        1, Int32, address_space = AddressSpace.SHARED
    ]()
    if tx == 0:
        s_write_idx[0] = Int32(0)

    barrier()

    # Each thread processes elements and atomically writes to shared memory.
    for i in range(tx, d, block_size):
        var logit = logits_row.load[width=1]((Idx[0](), Idx(i))).cast[
            DType.float32
        ]()
        if Float64(logit) > pivot:
            var exp_val = exp((logit - max_logit) / temp_val)

            # Atomically get write position and store.
            var pos = Int(Atomic.fetch_add(s_write_idx, Int32(1)))
            if pos < k:
                s_vals[pos] = exp_val
                s_idxs[pos] = i
                thread_sum += exp_val

    var block_sum = block.sum[block_size=block_size, broadcast=True](thread_sum)

    barrier()

    if tx == 0:
        s_count[0] = min(Int(s_write_idx[0]), k)

    barrier()

    # PHASE 3: Sampling.
    if tx == 0:
        var seed_val = seed_val
        if seed:
            seed_val = seed[bx]
        var rng_state = Random(seed=seed_val)
        var rng = rng_state.step_uniform()
        var r = block_sum * rng[0]

        var cached_count = s_count[0]
        for ki in range(cached_count):
            var exp_val = s_vals[ki]
            r -= exp_val
            if r <= 0.0 or ki == cached_count - 1:
                sampled_indices[bx] = Scalar[out_idx_type](s_idxs[ki])
                return


fn topk_softmax_sample[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types = Variadic.types[RuntimeInt[DType.int64]],
        stride_types = Variadic.types[ComptimeInt[1]],
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types = Variadic.types[RuntimeInt[DType.int64]],
        stride_types = Variadic.types[ComptimeInt[1]],
    ],
    SeedLayoutType: TensorLayout = Layout[
        shape_types = Variadic.types[RuntimeInt[DType.int64]],
        stride_types = Variadic.types[ComptimeInt[1]],
    ],
](
    ctx: DeviceContext,
    logits: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    sampled_indices: TileTensor[
        mut=True, out_idx_type, address_space = AddressSpace.GENERIC, ...
    ],
    top_k_val: Int,
    temperature_val: Float32 = 1.0,
    seed_val: UInt64 = 0,
    top_k_arr: Optional[
        TileTensor[out_idx_type, TopKArrLayoutType, MutExternalOrigin]
    ] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, MutExternalOrigin]
    ] = None,
    seed: Optional[
        TileTensor[DType.uint64, SeedLayoutType, MutExternalOrigin]
    ] = None,
) raises:
    """Samples token indices from top-K logits using softmax probabilities.

    This kernel performs single-pass top-K selection and categorical sampling:
    1. Finds the k-th largest logit via ternary search.
    2. Computes softmax over top-K elements and caches them in shared memory.
    3. Samples a single token index from the categorical distribution.

    Parameters:
        dtype: The data type of the input logits tensor.
        out_idx_type: The data type of the output sampled indices.
        block_size: The number of threads per block (default is 1024).
        TopKArrLayoutType: The layout type of the optional top_k_arr tensor.
        TemperatureLayoutType: The layout type of the optional temperature tensor.
        SeedLayoutType: The layout type of the optional seed tensor.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        logits:
            Input logits tensor with shape [batch_size, vocab_size].
        sampled_indices:
            Output buffer for sampled token indices with shape [batch_size].
        top_k_val: Int
            Default number of top elements to sample from for each batch element.
        temperature_val: Float32
            Temperature for softmax scaling (default is 1.0).
        seed_val: UInt64
            Seed for the random number generator (default is 0).
        top_k_arr:
            Optional per-batch top-K values. If provided, overrides top_k_val
            for each batch element.
        temperature:
            Optional per-batch temperature values. If provided, overrides
            temperature_val for each batch element.
        seed:
            Optional per-batch seed values. If provided, overrides seed_val
            for each batch element.
    """
    comptime assert logits.rank == 2, "logits rank must be 2"
    comptime assert sampled_indices.rank == 1, "sampled_indices rank must be 1"

    var shape = coord_to_index_list(logits.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    var out_shape = coord_to_index_list(sampled_indices.layout.shape_coord())
    if shape[0] != out_shape[0]:
        raise Error("sampled_indices shape must be [batch_size]")

    # Computes optimal vectorization width: find the largest vec_size that divides
    # both max hardware vector size (16 bytes / element size) and dim d.
    var vec_size = gcd(16 // size_of[dtype](), d)

    var k_rounded = ceildiv(top_k_val, WARP_SIZE) * WARP_SIZE
    var shared_mem_bytes = k_rounded * (size_of[Float32]() + size_of[Int]())

    var top_k_buf: DeviceBuffer[out_idx_type]
    if top_k_arr:
        top_k_buf = top_k_arr.value().to_device_buffer(ctx)
    else:
        top_k_buf = DeviceBuffer[out_idx_type](ctx, {}, 0, owning=False)
    var temp_buf: DeviceBuffer[DType.float32]
    if temperature:
        temp_buf = temperature.value().to_device_buffer(ctx)
    else:
        temp_buf = DeviceBuffer[DType.float32](ctx, {}, 0, owning=False)
    var seed_buf: DeviceBuffer[DType.uint64]
    if seed:
        seed_buf = seed.value().to_device_buffer(ctx)
    else:
        seed_buf = DeviceBuffer[DType.uint64](ctx, {}, 0, owning=False)

    @parameter
    fn launch_kernel[vec_size: Int]() raises:
        comptime kernel = TopKSoftmaxSampleKernel[
            block_size,
            vec_size,
            dtype,
            out_idx_type,
            LogitsLayoutType = logits.LayoutType,
            logits_origin = ImmutOrigin(logits.origin),
            SampledLayoutType = sampled_indices.LayoutType,
            sampled_origin = sampled_indices.origin,
        ]
        ctx.enqueue_function[kernel, kernel](
            logits.as_immut(),
            sampled_indices,
            top_k_buf,
            top_k_val,
            temperature_val,
            temp_buf,
            seed_val,
            seed_buf,
            d,
            grid_dim=batch_size,
            block_dim=block_size,
            shared_mem_bytes=shared_mem_bytes,
            attributes=pdl_launch_attributes(),
        )

    # Runtime dispatch to compile-time parameter.
    @parameter
    for param_vec_size in [16, 8, 4, 2, 1]:
        if vec_size == param_vec_size:
            return launch_kernel[param_vec_size]()
