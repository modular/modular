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

from math import ceildiv
from sys import align_of

# AMD Helper functions and structs for Tensor Core MMA operations
from sys.info import simd_width_of

from gpu import barrier, block_dim, block_idx, global_idx, lane_id, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from gpu.memory import AddressSpace
from gpu.sync import AMDScheduleBarrierMask, schedule_group_barrier
from layout import Layout
from layout._utils import make_amd_buffer_resource
from layout.element import Element
from layout.layout_tensor import (
    LayoutTensor,
    ThreadScope,
    copy_dram_to_local,
    copy_local_to_shared,
)
from layout.swizzle import Swizzle
from layout.tensor_core import TiledTensorCore
from memory import Pointer, LegacyUnsafePointer

from utils.index import IndexList


# Function to handle AMD-specific scheduling
@always_inline
# @parameter
fn amd_scheduling_hints[
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


@always_inline("nodebug")
fn copy_local_to_dram_32_32_8[
    dst_thread_layout: Layout,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](dst: LayoutTensor, src: LayoutTensor, dst_base: LayoutTensor):
    # TODO: use copy_local_to_dram instead once fixed. This is a workaround for now.

    var worker_idx = (
        thread_idx.x if thread_scope == ThreadScope.BLOCK else lane_id()
    )
    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var offset = (Int(dst.ptr) - Int(dst_base.ptr)) // size_of[dst.dtype]()
    var buffer = make_amd_buffer_resource(dst_base)
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + Scalar[
        dst.linear_idx_type
    ](offset)
    comptime num_stores_per_thread = dst_fragments.layout.size()

    comptime M = src.layout.shape[0].value()
    comptime N = src.layout.shape[1].value()

    comptime for n in range(N):
        comptime for m in range(M):
            comptime src_idx = 4 * n + 16 * m
            comptime i = 4 * n + m + ((m // 4) * 12)

            comptime dst_static_idx = dst_fragments.layout(i)
            var dst_idx = dst_frag_offset

            comptime if dst_fragments.layout.all_dims_known():
                dst_idx += Scalar[dst.linear_idx_type](dst_static_idx)
            else:
                dst_idx += dst_fragments.runtime_layout(i)

            var src_element = Element[index_type = src.linear_idx_type].load(
                src.ptr + src_idx,
                src.runtime_element_layout,
            )

            comptime element_stride = dst_fragments.element_layout.stride[
                1
            ].value()

            comptime if element_stride == 1:
                buffer.store(
                    Int32(dst_idx),
                    src_element.element_data.cast[dst.dtype](),
                )
            else:
                comptime for i in range(dst_fragments.element_layout.size()):
                    comptime element_offset = dst_fragments.element_layout(i)
                    var src = src_element.element_data[i].cast[dst.dtype]()
                    buffer.store(
                        Int32(
                            dst_idx
                            + Scalar[dst.linear_idx_type](element_offset)
                        ),
                        src,
                    )


struct AMD_MMA[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool,
    k_group_size: Int,
    num_k_tiles: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    simd_width: Int,
    swizzle: Swizzle,
    BK: Int,
    WK: Int,
]:
    comptime type_alignment = align_of[SIMD[Self.in_type, Self.simd_width]]()
    comptime tensor_core_mma = TiledTensorCore[
        Self.out_type,
        Self.in_type,
        Self.shape,
        Self.k_group_size,
        Self.transpose_b,
    ]()

    comptime SharedMemTileType[smem_layout: Layout] = LayoutTensor[
        Self.in_type,
        smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = Self.type_alignment,
    ]

    comptime MMARegTileType[num_mmas: Int] = LayoutTensor[
        Self.in_type,
        Layout.row_major(num_mmas * Self.num_k_tiles, Self.simd_width),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
        alignment = Self.type_alignment,
    ]

    comptime SharedMemWarpTileType[
        warp_rows: Int, smem_layout: Layout
    ] = Self.SharedMemTileType[smem_layout].TileType[warp_rows, Self.WK]


@always_inline
fn mma[
    k_tile_idx: Int,
    swap_a_b: Bool,
    MMAType: type_of(AMD_MMA),
](
    a_tiles: MMATileBuffers[mma_type=MMAType],
    b_tiles: MMATileBuffers[mma_type=MMAType],
    c_reg_tile: LayoutTensor[mut=True, ...],
):
    """
    AMD-style MMA operation wrapper for the AMD_MMA struct.

    Parameters:
        k_tile_idx: The K-dimension tile index.
        swap_a_b: Whether to swap A and B.
        MMAType: The type of MMA operation.

    Args:
        a_tiles: The A-dimension tile buffers.
        b_tiles: The B-dimension tile buffers.
        c_reg_tile: The C-dimension register tile.

    This function is used to perform the MMA operation for the AMD_MMA struct.
    """
    var a_reg_tile = a_tiles.get_reg_tile[k_tile_idx]()
    var b_reg_tile = b_tiles.get_reg_tile[k_tile_idx]()

    a_tiles.mma_type.tensor_core_mma.mma[swap_a_b=swap_a_b](
        a_reg_tile,
        b_reg_tile,
        c_reg_tile,
    )


struct MMATileBuffers[
    tensor_origin: ImmutOrigin,
    //,
    smem_layout: Layout,
    /,
    tensor_type: type_of(LayoutTensor),
    thread_layout: Layout,
    block_rows: Int,
    warp_rows: Int,
    stride: Int,
    num_mmas: Int,
    mma_type: type_of(AMD_MMA),
]:
    """Manages memory for a single matrix (A or B) in GEMM computation.

    This struct encapsulates all memory handling for a matrix, including:
    - Shared memory allocation and tiling
    - Register buffer allocation
    - Data movement between memory levels (DRAM→local→shared)
    """

    # Tensor types for different memory regions

    # Shared memory allocation for matrix data shared across the block
    comptime SharedMemTileType = Self.mma_type.SharedMemTileType[
        Self.smem_layout
    ]
    var shared_mem_tile: Self.SharedMemTileType

    # Tile view optimized for matrix multiplication acceleration (MMA) operations
    var shared_mem_warp_tile: Self.mma_type.SharedMemWarpTileType[
        Self.warp_rows, Self.smem_layout
    ]

    # Buffer for loading data from global memory before transferring to shared memory
    comptime MMARegTileType = Self.mma_type.MMARegTileType[Self.num_mmas]
    var load_reg_tile: Self.MMARegTileType

    # Register-level storage for matrix data during computation
    var mma_reg_tile: Self.MMARegTileType.StaticSplitType[
        Self.mma_type.num_k_tiles
    ]

    # Global memory iterator for input tensor
    comptime iter_type = Self.tensor_type.TileType[
        Self.block_rows, Self.stride
    ].TiledIteratorType[Self.block_rows, Self.mma_type.BK, axis=1]
    var gmem_iter: Self.iter_type

    var global_offset: UInt

    var tensor: Pointer[Self.tensor_type, Self.tensor_origin]

    @always_inline
    fn __init__(
        out self,
        ref[Self.tensor_origin] tensor: Self.tensor_type,
        warp_idx: Int,
        warp_k_idx: Int,
        block_idx: Int,
    ):
        """Initialize memory regions for a matrix based on warp coordinates.

        Args:
            tensor: The tensor to load from global memory.
            warp_idx: The warp index within the computation grid (used for MMA operations).
            warp_k_idx: The warp index within the computation grid (used for MMA operations).
            block_idx: The block index within the computation grid (used for warp tiling).
        """
        self.shared_mem_tile = Self.SharedMemTileType.stack_allocation()
        self.shared_mem_warp_tile = self.shared_mem_tile.tile[
            Self.warp_rows, Self.mma_type.WK
        ](warp_idx, warp_k_idx)
        self.load_reg_tile = Self.MMARegTileType.stack_allocation()
        self.mma_reg_tile = Self.MMARegTileType.stack_allocation().split[
            Self.mma_type.num_k_tiles
        ]()
        self.gmem_iter = tensor.tile[Self.block_rows, Self.stride](
            block_idx, 0
        ).tiled_iterator[Self.block_rows, Self.mma_type.BK, axis=1](0, 0)
        self.global_offset = UInt(Self.stride * (Self.block_rows * block_idx))
        # TODO: remove rebind once MOCO-1905 is fixed
        self.tensor = rebind[Pointer[Self.tensor_type, Self.tensor_origin]](
            Pointer(to=tensor)
        )

    @always_inline
    fn copy_to_shared(self):
        """Copy data from thread-local memory to shared memory.

        Uses structured thread cooperation to efficiently transfer data.
        """
        copy_local_to_shared[
            thread_layout = Self.thread_layout,
            swizzle = Self.mma_type.swizzle,
            thread_scope = ThreadScope.BLOCK,
            row_major=True,
        ](
            self.shared_mem_tile.vectorize[1, Self.mma_type.simd_width](),
            self.load_reg_tile.vectorize[1, Self.mma_type.simd_width](),
        )

    @always_inline
    fn load_from_dram(mut self) -> None:
        """Load data from global memory (DRAM) to thread-local memory."""
        copy_dram_to_local[
            src_thread_layout = Self.thread_layout,
            thread_scope = ThreadScope.BLOCK,
        ](
            self.load_reg_tile.vectorize[1, Self.mma_type.simd_width](),
            self.gmem_iter[].vectorize[1, Self.mma_type.simd_width](),
            self.tensor[],
            self.global_offset,
        )

        self.global_offset += UInt(Self.mma_type.BK)
        self.gmem_iter._incr()

    @always_inline
    fn get_reg_tile[
        k_tile_idx: Int
    ](self) -> Self.MMARegTileType.SplitElementType[Self.mma_type.num_k_tiles]:
        """Get a specific K-dimension tile from the register buffer.

        Parameters:
            k_tile_idx: The K-dimension tile index.

        Returns:
            A tile view for the specified location in the register buffer.
        """
        return self.mma_reg_tile[k_tile_idx]

    @always_inline
    fn load_tile_from_shared[k_tile_idx: Int, is_a: Bool](self):
        comptime if is_a:
            Self.mma_type.tensor_core_mma.mma_op.load_a[
                swizzle = Self.mma_type.swizzle
            ](
                self.shared_mem_warp_tile,
                self.mma_reg_tile[k_tile_idx]
                .tile[Self.num_mmas, Self.mma_type.simd_width](k_tile_idx, 0)
                .vectorize[1, Self.mma_type.simd_width](),
                UInt(k_tile_idx),
            )
        else:
            Self.mma_type.tensor_core_mma.mma_op.load_b[
                swizzle = Self.mma_type.swizzle
            ](
                self.shared_mem_warp_tile,
                self.mma_reg_tile[k_tile_idx]
                .tile[Self.num_mmas, Self.mma_type.simd_width](k_tile_idx, 0)
                .vectorize[1, Self.mma_type.simd_width](),
                UInt(k_tile_idx),
            )


@always_inline
fn compute_relative_error_kernel[
    dtype: DType,
    layout: Layout,
](
    reference: LayoutTensor[dtype, layout, MutAnyOrigin],
    computed: LayoutTensor[dtype, layout, MutAnyOrigin],
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    """
    GPU kernel that computes element-wise relative error between two LayoutTensors.

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
    # Get global thread indices
    var idx = global_idx.x
    var idy = global_idx.y

    # Get tensor dimensions
    var rows = UInt(reference.dim[0]())
    var cols = UInt(reference.dim[1]())

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


@always_inline
fn max_reduce_kernel[
    dtype: DType,
    layout: Layout,
](
    relative_error: LayoutTensor[dtype, layout, MutAnyOrigin],
    elements: Int,
    offset: Int,
    max_idx: Int,
):
    """
    GPU kernel that computes the maximum relative error in a subset of a tensor.

    Parameters:
        dtype: Data type of the tensor (should be floating point).
        layout: Layout of the input tensor.

    Args:
        relative_error: The relative error tensor to reduce.
        elements: The number of elements per block to reduce.
        offset: The stride/offset for accessing elements.
        max_idx: Maximum valid index to prevent out-of-bounds access.
    """

    # Get thread and block indices
    var tid = thread_idx.x
    var bid = block_idx.x

    var local_relative_error = relative_error.ptr[offset * elements * Int(bid)]

    # Parallel reduction loop: for(int i = elements >> 1; i > 0; i = i >> 1)
    var i = elements >> 1
    while i > 0:
        # Check bounds: threadIdx.x < i && offset * (elements * blockIdx.x + threadIdx.x + i) < maxIdx
        var current_idx = offset * (elements * Int(bid) + Int(tid) + i)

        if Int(tid) < i and current_idx < max_idx:
            var max_val = max(
                local_relative_error[offset * Int(tid)],
                local_relative_error[offset * Int(tid) + i],
            )
            local_relative_error[Int(tid)] = max_val

        barrier()

        # Divide by 2 for next iteration
        i = i >> 1


fn compare_equal[
    dtype: DType,
    layout: Layout,
](
    reference: LayoutTensor[dtype, layout, MutAnyOrigin],
    computed: LayoutTensor[dtype, layout, MutAnyOrigin],
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

    var m = reference.dim[0]()
    var n = reference.dim[1]()

    # Allocate a new LayoutTensor on device with same dtype and shape as reference
    var max_relative_error_buf = gpu_ctx.enqueue_create_buffer[dtype](m * n)
    var max_relative_error = LayoutTensor[
        dtype, reference.layout, MutAnyOrigin
    ](max_relative_error_buf.unsafe_ptr())

    # Zero out the memory in the max relative error tensor.
    gpu_ctx.enqueue_memset(
        DeviceBuffer[max_relative_error.dtype](
            gpu_ctx,
            max_relative_error.ptr,
            m * n,
            owning=False,
        ),
        0,
    )

    # Compute the relative error between the reference and computed tensors
    comptime rel_error_kernel = compute_relative_error_kernel[dtype, layout]
    gpu_ctx.enqueue_function_experimental[rel_error_kernel](
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
        gpu_ctx.enqueue_function_experimental[reduce_kernel](
            max_relative_error,
            num_elements,
            offset,
            m * n,
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
    var host_max_relative_error = host_buffer.unsafe_ptr()[0]
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
            reference.ptr,
        )
        gpu_ctx.enqueue_copy(
            computed_host_buf,
            computed.ptr,
        )

        var diff_buf = gpu_ctx.enqueue_create_host_buffer[dtype](m * n)
        var diff = LayoutTensor[dtype, layout, MutAnyOrigin](
            diff_buf.unsafe_ptr()
        )
        var max_diff: Float64 = 0.0
        for i in range(m):
            for j in range(n):
                var diff_val = abs(reference[i, j] - computed[i, j])
                diff[i, j] = diff_val
                var diff_f64 = diff_val.cast[DType.float64]()[0]
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
