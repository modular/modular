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

from bit import log2_floor
from collections import OptionalReg
from gpu import (
    WARP_SIZE,
    barrier,
    block,
    block_dim,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
    warp,
)
from gpu.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext
from gpu.host.dim import Dim
from gpu.random import Random
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
)
from math import ceildiv, gcd
from memory import stack_allocation
from os import Atomic
from sys import simd_width_of, size_of


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
    logits_layout: Layout,
    masked_logits_layout: Layout,
](
    logits: LayoutTensor[dtype, logits_layout, MutAnyOrigin],
    masked_logits: LayoutTensor[
        mut=True, dtype, masked_logits_layout, MutAnyOrigin
    ],
    top_k_arr: UnsafePointer[Scalar[out_idx_type]],
    top_k_val: Int,
    d: Int,
):
    var bx = Int(block_idx.x)
    var tx = Int(thread_idx.x)
    var row_idx = bx

    var logits_ptr = logits.ptr + bx * UInt(d)
    var masked_logits_ptr = masked_logits.ptr + bx * UInt(d)

    alias row_layout = Layout.row_major(1, UNKNOWN_VALUE)
    var logits_row = LayoutTensor[dtype, row_layout, MutAnyOrigin](
        logits_ptr, RuntimeLayout[row_layout]({1, d}, {d, 1})
    )
    var masked_logits_row = LayoutTensor[
        mut=True, dtype, row_layout, MutAnyOrigin
    ](masked_logits_ptr, RuntimeLayout[row_layout]({1, d}, {d, 1}))

    var k = top_k_val
    if top_k_arr:
        k = Int(top_k_arr[bx])

    # Initialize pivot to negative infinity.
    var pivot = Float64(Float32.MIN)

    var logits_vec = SIMD[DType.float32, vec_size]()

    if k < d:
        var min_max = get_min_max_value[vec_size, block_size](
            logits.ptr, Int(row_idx), d
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
                if (i * block_size + Int(tx)) * vec_size < d:
                    logits_vec = logits_row.load[width=vec_size](
                        0, i * block_size * vec_size + Int(tx * UInt(vec_size))
                    ).cast[DType.float32]()

                var probs_gt_pivot_0_count = SIMD[DType.int32, vec_size]()
                var probs_gt_pivot_1_count = SIMD[DType.int32, vec_size]()

                @parameter
                for j in range(vec_size):
                    # Calculate the global index for this element in the row.
                    # Will only count if the index is within the valid range [0, d).
                    var idx = (i * block_size + Int(tx)) * vec_size + j

                    # Count elements greater than pivot_0 (higher ternary search bound).
                    probs_gt_pivot_0_count[j] = 1 if (
                        Float64(logits_vec[j]) > pivot_0 and idx < d
                    ) else 0
                    # Count elements greater than pivot_1 (lower ternary search bound).
                    probs_gt_pivot_1_count[j] = 1 if (
                        Float64(logits_vec[j]) > pivot_1 and idx < d
                    ) else 0

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
            if aggregate_gt_pivot_1 >= k:
                low = pivot_1
            elif aggregate_gt_pivot_0 >= k:
                low = pivot_0
                high = min(pivot_1, Float64(max_le_high))
            else:
                high = min(pivot_0, Float64(max_le_high))

            if min_gt_low == max_le_high:
                break

        pivot = low

    for i in range(ceildiv(d, block_size * vec_size)):
        logits_vec = 0
        if (i * block_size + Int(tx)) * vec_size < d:
            logits_vec = logits_row.load[width=vec_size](
                0, i * block_size * vec_size + Int(tx * UInt(vec_size))
            ).cast[DType.float32]()

        logits_vec = (logits_vec.cast[DType.float64]().gt(pivot)).select(
            logits_vec, Float32.MIN
        )

        if (i * block_size + Int(tx)) * vec_size < d:
            masked_logits_row.store[width=vec_size](
                0,
                i * block_size * vec_size + Int(tx * UInt(vec_size)),
                logits_vec.cast[dtype](),
            )


fn topk_mask_logits[
    dtype: DType, out_idx_type: DType, block_size: Int = 1024
](
    ctx: DeviceContext,
    logits: LayoutTensor[dtype, **_],
    masked_logits: LayoutTensor[mut=True, dtype, **_],
    top_k_val: Int,
    top_k_arr: OptionalReg[
        LayoutTensor[
            out_idx_type, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ] = None,
) raises:
    constrained[logits.rank == 2, "logits rank must be 2"]()
    constrained[
        logits.rank == masked_logits.rank,
        "logits.rank must match masked_logits.rank",
    ]()

    var shape = logits.runtime_layout.shape.value.canonicalize()
    var batch_size = shape[0]
    var d = shape[1]

    var out_shape = masked_logits.runtime_layout.shape.value.canonicalize()
    if shape[0] != out_shape[0] or shape[1] != out_shape[1]:
        raise Error("masked_logits shape must match logits shape")

    # Computes optimal vectorization width: find the largest vec_size that divides
    # both max hardware vector size (16 bytes / element size) and dim d.
    var vec_size = gcd(16 // size_of[dtype](), d)

    @parameter
    fn launch_kernel[vec_size: Int]() raises:
        alias kernel = TopKMaskLogitsKernel[
            block_size,
            vec_size,
            dtype,
            out_idx_type,
            logits.layout,
            masked_logits.layout,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            logits,
            masked_logits,
            top_k_arr.value().to_device_buffer(ctx),
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
    sampled_id_sram: UnsafePointer[Int, address_space = AddressSpace.SHARED],
    last_valid_id_sram: UnsafePointer[Int, address_space = AddressSpace.SHARED],
) -> Float32:
    """Device-level sampling from probability distribution with atomic operations.
    """

    var tx = thread_idx.x

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


@register_passable("trivial")
struct ValueCount[T: DType](Defaultable, ImplicitlyCopyable, Movable):
    """A struct that holds a value and a count, used for block reductions.

    This is useful for computing both the sum of values and the count
    of elements that satisfy a condition in a single reduction pass.

    Parameters:
        T: The DType of the value field.
    """

    var value: Scalar[T]
    var count: Int32

    fn __init__(out self, value: Scalar[T], count: Int32):
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

    alias limit = log2_floor(WARP_SIZE)

    # Reduce across warp lanes using shuffle_down.
    @parameter
    for i in reversed(range(limit)):
        alias offset = 1 << i
        result.value += warp.shuffle_down(result.value, offset)
        result.count += warp.shuffle_down(Int32(result.count), offset)
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
    alias MAX_BLOCK_SIZE = 1024
    constrained[
        MAX_BLOCK_SIZE % WARP_SIZE == 0,
        "block size must be a multiple of the warp size",
    ]()

    alias value_width = simd_width_of[Scalar[T]]()
    alias count_width = simd_width_of[DType.int32]()

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
    alias num_warps_needed = MAX_BLOCK_SIZE // WARP_SIZE

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
    var thread_in_final_warp = thread_idx.x < UInt(
        block_dim.x // UInt(WARP_SIZE)
    )

    if thread_in_final_warp:
        block_accum = {
            value = value_sram[lane_id() * UInt(value_width)],
            count = Int32(count_sram[lane_id() * UInt(count_width)]),
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
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    probs_layout: Layout,
    output_layout: Layout,
    deterministic: Bool,
](
    probs: LayoutTensor[dtype, probs_layout, MutAnyOrigin],
    output: LayoutTensor[mut=True, out_idx_type, output_layout, MutAnyOrigin],
    indices: UnsafePointer[Scalar[out_idx_type]],
    top_k_arr: UnsafePointer[Scalar[out_idx_type]],
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

    alias row_layout = Layout.row_major(1, UNKNOWN_VALUE)
    var probs_ptr = probs.ptr + row_idx * d
    var probs_row = LayoutTensor[dtype, row_layout, MutAnyOrigin](
        probs_ptr, RuntimeLayout[row_layout]({1, d}, {d, 1})
    )

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
                    0, (i * block_size + tx) * vec_size
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

        var pivot_0 = Float64(probs_row.load[width=1](0, sampled_id))
        var pivot_1 = (pivot_0 + high) / 2.0

        var aggregate_gt_pivot_0 = ValueCount[DType.float32](0.0, 0)
        var aggregate_gt_pivot_1 = ValueCount[DType.float32](0.0, 0)

        for i in range(ceildiv(d, block_size * vec_size)):
            probs_vec = 0
            if (i * block_size + tx) * vec_size < d:
                probs_vec = probs_row.load[width=vec_size](
                    0, (i * block_size + tx) * vec_size
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
                probs_gt_pivot_0_counts[j] = 1 if (
                    gt_pivot_0 and is_valid
                ) else 0

                # For pivot_1.
                var gt_pivot_1 = probs_vec[j] > Float32(pivot_1)
                probs_gt_pivot_1_values[j] = probs_vec[j] if gt_pivot_1 else 0.0
                probs_gt_pivot_1_counts[j] = 1 if (
                    gt_pivot_1 and is_valid
                ) else 0

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

        if aggregate_gt_pivot_0.count < k:
            # Case 1: pivot_0 accepted - found acceptable threshold.
            break

        if aggregate_gt_pivot_1.count < k:
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
        output[bx] = sampled_id


fn topk_sampling_from_prob[
    dtype: DType, out_idx_type: DType, block_size: Int = 1024
](
    ctx: DeviceContext,
    probs: LayoutTensor[dtype, **_],
    output: LayoutTensor[mut=True, out_idx_type, **_],
    top_k_val: Int,
    deterministic: Bool = False,
    rng_seed: UInt64 = 0,
    rng_offset: UInt64 = 0,
    indices: OptionalReg[
        LayoutTensor[
            out_idx_type, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
    ] = None,
    top_k_arr: OptionalReg[
        LayoutTensor[
            out_idx_type, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
        ]
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

    constrained[probs.rank == 2, "probs rank must be 2"]()
    constrained[output.rank == 1, "output rank must be 1"]()

    var shape = probs.runtime_layout.shape.value.canonicalize()
    var batch_size = shape[0]
    var d = shape[1]

    var out_shape = output.runtime_layout.shape.value.canonicalize()
    if out_shape[0] != batch_size:
        raise Error("output batch size must match probs batch size")

    # Computes optimal vectorization width: find the largest vec_size that divides
    # both max hardware vector size (16 bytes / element size) and dim d.
    var vec_size = gcd(16 // size_of[dtype](), d)

    @parameter
    fn launch_kernel[vec_size: Int, deterministic: Bool]() raises:
        ctx.enqueue_function[
            TopKSamplingFromProbKernel[
                block_size,
                vec_size,
                dtype,
                out_idx_type,
                probs.layout,
                output.layout,
                deterministic,
            ]
        ](
            probs,
            output,
            indices.value().ptr if indices else UnsafePointer[
                Scalar[out_idx_type]
            ](),
            top_k_arr.value().ptr if top_k_arr else UnsafePointer[
                Scalar[out_idx_type]
            ](),
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
