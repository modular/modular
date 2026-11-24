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

from std.math import ceildiv
from std.sys import size_of
from std.sys.info import simd_width_of

from max.gpu import block_dim, block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.sync import AMDScheduleBarrierMask, schedule_group_barrier
from layout import TensorLayout, TileTensor, row_major

from std.utils.index import IndexList


@inline(.always)
def amd_scheduling_hints[
    input_type: DType,
    output_type: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    scheduler_hint: IndexList[3],
]():
    """
    AMD-style scheduling hints for tiled GEMM kernel.

    Parameters:
        input_type: The data type of the input tensors.
        output_type: The data type of the output tensor.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        WM: The warp tile size in the M dimension.
        WN: The warp tile size in the N dimension.
        MMA_M: Tensor core instruction shape in M dimension.
        MMA_N: Tensor core instruction shape in N dimension.
        MMA_K: Tensor core instruction shape in K dimension.
        scheduler_hint: A list of 3 integers indicating the scheduling hints for the kernel.

    This function is used to schedule the kernel using the scheduling hints. Each integer in the
    scheduler_hint describes the number of instructions to be scheduled before the next barrier.
    The scheduler_hint[0] is for MFMA after DS_WRITE.
    The scheduler_hint[1] is for MFMA after DS_READ.
    The scheduler_hint[2] is for MFMA after DS_READ.

    """

    comptime NUM_THREADS = 256
    comptime threads_per_row = BK // simd_width_of[input_type]()
    comptime rows_per_thread_block = NUM_THREADS // threads_per_row
    comptime a_loads_per_thread = BM // rows_per_thread_block
    comptime b_loads_per_thread = BN // rows_per_thread_block
    comptime mmas_per_warp_m = WM // MMA_M
    comptime mmas_per_warp_n = WN // MMA_N
    comptime k_tiles_count = BK // MMA_K

    # scheduler_hint[0] For MFMA after DS_WRITE and VMEM_READ
    # scheduler_hint[1] For MFMA after VMEM_READ
    # scheduler_hint[2] For MFMA after DS_READ

    # Schedule barriers for DS_READ operations
    comptime for i in range(
        (mmas_per_warp_m * k_tiles_count + mmas_per_warp_n * k_tiles_count)
        // k_tiles_count
    ):
        schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
        schedule_group_barrier(
            AMDScheduleBarrierMask.MFMA, Int32(scheduler_hint[2]), 0
        )

    # Schedule barriers for memory load operations
    comptime for i in range(a_loads_per_thread + b_loads_per_thread):
        schedule_group_barrier(AMDScheduleBarrierMask.DS_WRITE, 1, 0)
        schedule_group_barrier(
            AMDScheduleBarrierMask.MFMA, Int32(scheduler_hint[0]), 0
        )
        schedule_group_barrier(AMDScheduleBarrierMask.VMEM_READ, 1, 0)
        schedule_group_barrier(
            AMDScheduleBarrierMask.MFMA, Int32(scheduler_hint[1]), 0
        )

    # Additional DS_READ scheduling for remaining k_tiles
    comptime for i in range(
        (mmas_per_warp_m * k_tiles_count + mmas_per_warp_n * k_tiles_count)
        // k_tiles_count
        * (k_tiles_count - 1)
    ):
        schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
        schedule_group_barrier(
            AMDScheduleBarrierMask.MFMA, Int32(scheduler_hint[2]), 0
        )


@inline(.always)
def compute_relative_error_kernel[
    dtype: DType,
    layout: TensorLayout,
](
    reference: TileTensor[dtype, layout, MutAnyOrigin],
    computed: TileTensor[dtype, layout, MutAnyOrigin],
    output: TileTensor[dtype, layout, MutAnyOrigin],
):
    """
    GPU kernel that computes element-wise relative error between two TileTensors.

    Relative error is computed as: |computed - reference| / max(|reference|, epsilon)
    where epsilon prevents division by zero.

    Parameters:
        dtype: Data type of the tensors (should be floating point).
        layout: Layout of the input tensors.

    Args:
        reference: The reference (ground truth) tensor.
        computed: The computed (test) tensor.
        output: Output tensor to store relative errors (same shape as inputs).
    """
    # Evidence that the (symbolic `TensorLayout`) tensors are flat rank-2, so
    # the 2-index `__setitem__` / `__getitem__` constraints can be proven.
    comptime assert output.flat_rank == 2

    # Get global thread indices
    var idx = global_idx.x
    var idy = global_idx.y

    # Get tensor dimensions (`TileTensor.dim` returns a `Scalar`; the bounds
    # comparison and indexing below want `Int`).
    var rows = Int(reference.dim[0]())
    var cols = Int(reference.dim[1]())

    # Check bounds
    if idx >= rows or idy >= cols:
        return

    # Get values at current position
    var ref_val = reference[idx, idy]
    var comp_val = computed[idx, idy]

    # Compute absolute difference
    var abs_diff = abs(comp_val - ref_val)

    # Compute denominator with epsilon to prevent division by zero
    comptime epsilon = Scalar[dtype](
        1e-8
    ) if dtype == DType.float32 else Scalar[dtype](1e-4)
    var denominator = max(abs(ref_val), epsilon)

    # Compute relative error
    var rel_error = abs_diff / denominator

    # Store result
    output[idx, idy] = rel_error


@inline(.always)
def max_reduce_kernel[
    dtype: DType,
    layout: TensorLayout,
](
    relative_error: TileTensor[dtype, layout, MutAnyOrigin],
    elements_dev: Int32,
    offset_dev: Int32,
    max_idx_dev: Int32,
):
    """
    GPU kernel that computes the maximum relative error in a subset of a tensor.

    Parameters:
        dtype: Data type of the tensor (should be floating point).
        layout: Layout of the input tensor.

    Args:
        relative_error: The relative error tensor to reduce.
        elements_dev: The number of elements per block to reduce.
        offset_dev: The stride/offset for accessing elements.
        max_idx_dev: Maximum valid index to prevent out-of-bounds access.
    """

    var elements = Int(elements_dev)
    var offset = Int(offset_dev)
    var max_idx = Int(max_idx_dev)

    # Get thread and block indices
    var tid = thread_idx.x
    var bid = block_idx.x

    var local_relative_error = relative_error._storage[
        unsafe_offset=offset * elements * bid
    ]

    # Parallel reduction loop: for(int i = elements >> 1; i > 0; i = i >> 1)
    var i = elements >> 1
    while i > 0:
        # Check bounds: threadIdx.x < i && offset * (elements * blockIdx.x + threadIdx.x + i) < maxIdx
        var current_idx = offset * (elements * bid + tid + i)

        if tid < i and current_idx < max_idx:
            var max_val = max(
                local_relative_error[offset * tid],
                local_relative_error[offset * tid + i],
            )
            local_relative_error[tid] = max_val

        barrier()

        # Divide by 2 for next iteration
        i = i >> 1


def compare_equal[
    dtype: DType,
    layout: TensorLayout,
](
    reference: TileTensor[dtype, layout, MutAnyOrigin],
    computed: TileTensor[dtype, layout, MutAnyOrigin],
    print_results: Bool,
) raises:
    """
    Helper function to compare two tensors for equality using relative error.

    Parameters:
        dtype: Data type of the tensors.
        layout: Layout of the input tensor.

    Args:
        reference: The reference (ground truth) tensor.
        computed: The computed (test) tensor.
        print_results: Whether to print the results.
    """

    var gpu_ctx = DeviceContext()

    # `TileTensor.dim` returns a `Scalar`; `Int` is needed for buffer sizes and
    # the host-side loops below.
    var m = Int(reference.dim[0]())
    var n = Int(reference.dim[1]())

    # Allocate a new TileTensor on device with same dtype and shape as reference
    var max_relative_error_buf = gpu_ctx.enqueue_create_buffer[dtype](m * n)
    var max_relative_error = TileTensor(
        max_relative_error_buf, reference.layout
    ).as_unsafe_any_origin()

    # Zero out the memory in the max relative error tensor.
    gpu_ctx.enqueue_memset(
        DeviceBuffer[max_relative_error.dtype](
            gpu_ctx,
            max_relative_error._storage,
            m * n,
            owning=False,
        ),
        0,
    )

    # Compute the relative error between the reference and computed tensors
    comptime rel_error_kernel = compute_relative_error_kernel[dtype, layout]
    gpu_ctx.enqueue_function[rel_error_kernel](
        reference,
        computed,
        max_relative_error,
        grid_dim=(ceildiv(m * n, 1024), 1),
        block_dim=(1024, 1),
    )
    gpu_ctx.synchronize()

    # Find the maximum relative error
    var offset = 1
    var i = m * n

    while i > 1:
        var num_threadblocks = ceildiv(i, 1024)
        var num_elements = i if i < 1024 else 1024

        comptime reduce_kernel = max_reduce_kernel[dtype, layout]
        gpu_ctx.enqueue_function[reduce_kernel](
            max_relative_error,
            Int32(num_elements),
            Int32(offset),
            Int32(m * n),
            grid_dim=(num_threadblocks, 1),
            block_dim=(512, 1),
        )
        gpu_ctx.synchronize()

        i = ceildiv(i, 1024)
        offset = offset * 1024

    # Allocate host buffer to receive the max relative error value
    var host_buffer = gpu_ctx.enqueue_create_host_buffer[dtype](1)

    # Create a device buffer view of just the first element
    var first_element_buf = DeviceBuffer[dtype](
        gpu_ctx, max_relative_error_buf.unsafe_ptr(), 1, owning=False
    )

    # Copy just the first element from device to host
    gpu_ctx.enqueue_copy(host_buffer, first_element_buf)
    gpu_ctx.synchronize()

    # Access the result from the host buffer
    var host_max_relative_error = host_buffer[0]
    print("Maximum relative error:", host_max_relative_error)

    # Print the two tensors if print_results is True
    if print_results:
        var print_tensors = False

        # Copy the tensors to the host
        var reference_host_buf = gpu_ctx.enqueue_create_host_buffer[dtype](
            m * n
        )
        var computed_host_buf = gpu_ctx.enqueue_create_host_buffer[dtype](m * n)

        gpu_ctx.enqueue_copy(
            reference_host_buf,
            reference._storage,
        )
        gpu_ctx.enqueue_copy(
            computed_host_buf,
            computed._storage,
        )

        var diff_buf = gpu_ctx.enqueue_create_host_buffer[dtype](m * n)
        var diff = TileTensor(diff_buf, reference.layout).as_unsafe_any_origin()
        # Evidence that `diff` is flat rank-2 for the 2-index `__setitem__`.
        comptime assert diff.flat_rank == 2
        var max_diff: Float64 = 0.0
        for i in range(m):
            for j in range(n):
                var diff_val = abs(reference[i, j] - computed[i, j])
                diff[i, j] = diff_val
                var diff_f64 = diff_val.cast[.float64]()[0]
                if diff_f64 > max_diff:
                    max_diff = diff_f64
                # diff[i, j] = reference[i, j] - computed[i, j]

        print("Max diff:", max_diff)

        if print_tensors:
            # Print the tensors
            print("Reference tensor:")
            print(reference, sep=", ")
            print("Computed tensor:")
            print(computed, sep=", ")
            print("Diff tensor:")
            print(diff, sep=", ")
