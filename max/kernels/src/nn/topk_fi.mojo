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
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
)
from math import ceildiv, gcd
from sys import size_of


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
    logits: LayoutTensor[dtype, logits_layout, MutableAnyOrigin],
    masked_logits: LayoutTensor[
        mut=True, dtype, masked_logits_layout, MutableAnyOrigin
    ],
    top_k_arr: UnsafePointer[Scalar[out_idx_type]],
    top_k_val: Int,
    d: Int,
):
    var bx = block_idx.x
    var tx = thread_idx.x
    var row_idx = bx

    var logits_ptr = logits.ptr + bx * d
    var masked_logits_ptr = masked_logits.ptr + bx * d

    alias row_layout = Layout.row_major(1, UNKNOWN_VALUE)
    var logits_row = LayoutTensor[dtype, row_layout, MutableAnyOrigin](
        logits_ptr, RuntimeLayout[row_layout]({1, d}, {d, 1})
    )
    var masked_logits_row = LayoutTensor[
        mut=True, dtype, row_layout, MutableAnyOrigin
    ](masked_logits_ptr, RuntimeLayout[row_layout]({1, d}, {d, 1}))

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
                        0, i * block_size * vec_size + tx * vec_size
                    ).cast[DType.float32]()

                var probs_gt_pivot_0_count = SIMD[DType.int32, vec_size]()
                var probs_gt_pivot_1_count = SIMD[DType.int32, vec_size]()

                @parameter
                for j in range(vec_size):
                    # Calculate the global index for this element in the row.
                    # Will only count if the index is within the valid range [0, d).
                    var idx = (i * block_size + tx) * vec_size + j

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
        if (i * block_size + tx) * vec_size < d:
            logits_vec = logits_row.load[width=vec_size](
                0, i * block_size * vec_size + tx * vec_size
            ).cast[DType.float32]()

        logits_vec = (logits_vec.cast[DType.float64]().gt(pivot)).select(
            logits_vec, Float32.MIN
        )

        if (i * block_size + tx) * vec_size < d:
            masked_logits_row.store[width=vec_size](
                0,
                i * block_size * vec_size + tx * vec_size,
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
            out_idx_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin
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
        ctx.enqueue_function[
            TopKMaskLogitsKernel[
                block_size,
                vec_size,
                dtype,
                out_idx_type,
                logits.layout,
                masked_logits.layout,
            ]
        ](
            logits,
            masked_logits,
            top_k_arr.value().ptr if top_k_arr else UnsafePointer[
                Scalar[out_idx_type]
            ](),
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
