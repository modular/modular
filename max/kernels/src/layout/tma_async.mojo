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
"""
Tensor Memory Accelerator (TMA) Asynchronous Operations Module

Provides high-performance abstractions for NVIDIA's Tensor Memory Accelerator (TMA),
enabling efficient asynchronous data movement between global and shared memory in GPU kernels.
It is designed for use with NVIDIA Hopper architecture and newer GPUs that support TMA instructions.

Key Components:
--------------
- `TMATensorTile`: Core struct that encapsulates a TMA descriptor for efficient data transfers
  between global and shared memory with various access patterns and optimizations.

- `SharedMemBarrier`: Synchronization primitive for coordinating asynchronous TMA operations,
  ensuring data transfers complete before dependent operations begin.

- `PipelineState`: Helper struct for managing multi-stage pipeline execution with circular
  buffer semantics, enabling efficient double or triple buffering techniques.

- `create_tma_tile`: Factory functions for creating optimized `TMATensorTile` instances with
  various configurations for different tensor shapes and memory access patterns.
"""

from math import ceildiv, log2
from sys import align_of, llvm_intrinsic, simd_width_of, size_of
from sys._assembly import inlined_assembly

from gpu.host import DeviceBuffer, DeviceContext
from gpu.host._tensormap import (
    SwizzleMode as _SwizzleMode,
    create_tensormap_im2col as _create_tensormap_im2col,
)
from gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    TMADescriptor,
    create_tma_descriptor,
    prefetch_tma_descriptor,
)
from gpu.intrinsics import Scope
from gpu.memory import (
    ReduceOp,
    async_copy,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_reduce,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_shared_cluster_global_im2col,
    cp_async_bulk_tensor_shared_cluster_global_im2col_multicast,
    cp_async_bulk_tensor_shared_cluster_global_multicast,
    CacheEviction,
)
from gpu.sync import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx_relaxed,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._tile_tensor import TileTensor
from layout.int_tuple import product, to_index_list as int_tuple_to_index_list
from layout.runtime_tuple import (
    coalesce_nested_tuple,
    flatten,
    to_index_list as runtime_tuple_to_index_list,
)
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major

from utils.index import Index, IndexList
from builtin.device_passable import DevicePassable
from utils.static_tuple import StaticTuple
from os import abort
from layout.layout_tensor import LayoutTensorIter


# Returns an IntTuple of variadic Int values.
#
fn _to_int_tuple[*vals: Int]() -> IntTuple:
    res = IntTuple()

    comptime num_vals = std.builtin.Variadic.size(vals)

    comptime for i in range(num_vals):
        res.append(vals[i])
    return res


fn _tma_desc_tile_layout[
    dtype: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> Layout:
    comptime assert (
        size_of[dtype]() >= 1
    ), "Don't support sub-byte dtype in TMA yet."

    comptime assert (
        rank == 2 or rank == 3 or rank == 4 or rank == 5
    ), "Only support 2D/3D/4D/5D TMA descriptor for now."

    comptime if rank == 2:
        comptime dim0 = tile_shape[0]
        comptime dim1 = tile_shape[1]

        # TMA copies BM x `swizzle_mode.bytes()` Bytes each time.
        return Layout.row_major(dim0, swizzle_mode.bytes() // size_of[dtype]())

    elif rank == 3:
        comptime dim0 = tile_shape[0]
        comptime dim1 = tile_shape[1]
        comptime dim2 = tile_shape[2]

        return Layout(
            [dim0, dim1, swizzle_mode.bytes() // size_of[dtype]()],
            [1, 1, 1],
        )

    elif rank == 4:
        comptime dim0 = tile_shape[0]
        comptime dim1 = tile_shape[1]
        comptime dim2 = tile_shape[2]
        comptime dim3 = tile_shape[3]

        return Layout(
            [dim0, dim1, dim2, swizzle_mode.bytes() // size_of[dtype]()],
            [1, 1, 1, 1],
        )

    else:  # rank == 5
        comptime dim0 = tile_shape[0]
        comptime dim1 = tile_shape[1]
        comptime dim2 = tile_shape[2]
        comptime dim3 = tile_shape[3]
        comptime dim4 = tile_shape[4]

        return Layout(
            [dim0, dim1, dim2, dim3, swizzle_mode.bytes() // size_of[dtype]()],
            [1, 1, 1, 1, 1],
        )


struct SharedMemBarrier(TrivialRegisterPassable):
    """A hardware-accelerated synchronization primitive for GPU shared memory operations.

    This struct provides a barrier mechanism optimized for coordinating thread execution
    and memory transfers in GPU kernels, particularly for Tensor Memory Accelerator (TMA)
    operations. It enables efficient synchronization between threads and memory operations
    by leveraging hardware-specific barrier instructions.

    Key features:
    - Thread synchronization across thread blocks
    - Memory transfer completion tracking
    - Hardware-accelerated barrier operations
    - Support for phased synchronization

    This barrier is particularly useful for ensuring that shared memory operations
    complete before dependent computations begin, which is critical for maintaining
    data consistency in high-performance GPU kernels.
    """

    var mbar: Int64
    """Shared memory location used for the barrier state.

    This field stores an 8-byte aligned shared memory location that
    maintains the state of the barrier. The memory must be in shared address
    space to be accessible by all threads in a block.
    """

    @always_inline("nodebug")
    fn init[
        o: MutOrigin
    ](ref[o, AddressSpace.SHARED] self, num_threads: Int32 = 1):
        """Initialize the barrier state with the expected number of threads.

        Sets up the barrier to expect arrivals from the specified number of threads
        before it can be satisfied. This is essential for coordinating thread
        synchronization in GPU kernels.

        Args:
            num_threads: Number of threads that must arrive at the barrier
                         before it is satisfied. Defaults to 1.

        Parameters:
            o: Origin of self.
        """
        mbarrier_init(self.unsafe_ptr(), num_threads)

    @always_inline("nodebug")
    fn expect_bytes[
        o: MutOrigin
    ](ref[o, AddressSpace.SHARED] self, bytes: Int32):
        """Configure the barrier to expect a specific number of bytes to be transferred.

        Used with TMA operations to indicate the expected size of data transfer.
        The barrier will be satisfied when the specified number of bytes has been
        transferred, enabling efficient coordination of memory operations.

        Args:
            bytes: Number of bytes expected to be transferred.

        Parameters:
            o: Origin of self.
        """
        mbarrier_arrive_expect_tx_shared(self.unsafe_ptr(), bytes)

    @always_inline
    fn expect_bytes_relaxed[
        o: MutOrigin
    ](ref[o, AddressSpace.SHARED] self, bytes: Int32) -> UInt64:
        """Configure the barrier to expect a specific number of bytes to be transferred.

        Used with TMA operations to indicate the expected size of data transfer.
        The barrier will be satisfied when the specified number of bytes has been
        transferred, enabling efficient coordination of memory operations.

        Args:
            bytes: Number of bytes expected to be transferred.

        Parameters:
            o: Origin of self.

        Returns:
            The state.
        """
        return mbarrier_arrive_expect_tx_relaxed(self.unsafe_ptr(), bytes)

    @always_inline
    fn arrive_and_expect_bytes[
        o: MutOrigin
    ](
        ref[o, AddressSpace.SHARED] self,
        bytes: Int32,
        cta_id: UInt32,
        pred: UInt32,
    ):
        """Configure the barrier to expect a specific number to bytes to be transferred
        at a remote CTA.

         Used with TMA operations to indicate the expected size of data transfer.
         The barrier will be satisfied when the specified number of bytes has been
         transferred at the specified CTA in the cluster.

        Args:
            bytes: Number of bytes expected to be transferred.
            cta_id: The CTA ID in a cluster to configure an arrival.
            pred: Predication on the arrival configuration instruction. Use UInt32 to match `selp.u32` in ptx.

        Parameters:
            o: Origin of self.
        """

        comptime asm = """
        .reg .pred p;
        .reg .b32 remAddr32;
        setp.eq.u32 p, $2, 1;
        @p mapa.shared::cluster.u32  remAddr32, $0, $1;
        @p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], $3;
        """

        inlined_assembly[asm, NoneType, constraints="r,r,r,r"](
            Int32(Int(self.unsafe_ptr())), cta_id, pred, bytes
        )

    @always_inline("nodebug")
    fn wait[
        ticks: Optional[UInt32] = None
    ](ref[AddressSpace.SHARED] self, phase: UInt32 = 0):
        """Wait until the barrier is satisfied.

        Blocks the calling thread until the barrier is satisfied, either by
        the expected number of threads arriving or the expected data transfer
        completing. This method implements an efficient spin-wait mechanism
        optimized for GPU execution.

        Parameters:
            ticks: The number of ticks to wait before timing out in nanoseconds.
                   Defaults to None.

        Args:
            phase: The phase value to check against. Defaults to 0.

        Note:
            Minimizes thread divergence during synchronization by using
            hardware-accelerated barrier instructions.
        """
        # Based on cutlass
        # https://github.com/NVIDIA/cutlass/blob/d1ef0e87f2f3d68cf5ad7472cadc1152a8d3857c/include/cutlass/arch/barrier.h#L408

        comptime wait_asm = (
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1"
            + (" , $2" if ticks else "")
            + ";"
        )
        comptime asm = """{
            .reg .pred P1;
            LAB_WAIT:
            """ + wait_asm + """
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
        }"""

        comptime constraints = "r,r" + (",r" if ticks else "")

        comptime if ticks:
            inlined_assembly[asm, NoneType, constraints=constraints](
                Int32(Int(self.unsafe_ptr())), phase, ticks.value()
            )
        else:
            inlined_assembly[asm, NoneType, constraints=constraints](
                Int32(Int(self.unsafe_ptr())), phase
            )

    @always_inline("nodebug")
    fn wait_acquire[
        scope: Scope
    ](ref[AddressSpace.SHARED] self, phase: UInt32 = 0):
        """Acquire and wait until the barrier is satisfied.

        Blocks the calling thread until the barrier is satisfied, either by
        the expected number of threads arriving or the expected data transfer
        completing. This method implements an efficient spin-wait mechanism
        optimized for GPU execution.

        Parameters:
            scope: The scope of the barrier.

        Args:
            phase: The phase value to check against. Defaults to 0.

        Note:
            Minimizes thread divergence during synchronization by using
            hardware-accelerated barrier instructions.
        """
        # Based on cccl
        # https://github.com/NVIDIA/cccl/blob/ba510b38e01dac5ab9b5faad9b9b1701d60d9980/libcudacxx/include/cuda/__ptx/instructions/generated/mbarrier_try_wait_parity.h#L94

        comptime assert (
            scope == Scope.CLUSTER or scope == Scope.BLOCK
        ), "wait_acquire is only supported for cluster or block/CTA scope."

        comptime asm = (
            """{
            .reg .pred P1;
            LAB_WAIT:
            mbarrier.try_wait.parity.acquire."""
            + scope.mnemonic()
            + """.shared::cta.b64 P1, [$0], $1;
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
            }"""
        )
        inlined_assembly[asm, NoneType, constraints="r,r"](
            Int32(Int(self.unsafe_ptr())), phase
        )

    @always_inline("nodebug")
    fn wait_relaxed[
        scope: Scope
    ](ref[AddressSpace.SHARED] self, phase: UInt32 = 0):
        """Wait until the barrier is satisfied with relaxed ordering.

        Blocks the calling thread until the barrier is satisfied, either by
        the expected number of threads arriving or the expected data transfer
        completing. This method implements an efficient spin-wait mechanism
        optimized for GPU execution.

        Parameters:
            scope: The scope of the barrier.

        Args:
            phase: The phase value to check against. Defaults to 0.

        Note:
            Minimizes thread divergence during synchronization by using
            hardware-accelerated barrier instructions.
        """
        # Based on cccl
        # https://github.com/NVIDIA/cccl/blob/ba510b38e01dac5ab9b5faad9b9b1701d60d9980/libcudacxx/include/cuda/__ptx/instructions/generated/mbarrier_try_wait_parity.h#L104

        comptime assert (
            scope == Scope.CLUSTER or scope == Scope.BLOCK
        ), "wait_relaxed is only supported for cluster or block/CTA scope."

        comptime asm = (
            """{
            .reg .pred P1;
            LAB_WAIT:
            mbarrier.try_wait.parity.relaxed."""
            + scope.mnemonic()
            + """.shared::cta.b64 P1, [$0], $1;
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
            }"""
        )
        inlined_assembly[asm, NoneType, constraints="r,r"](
            Int32(Int(self.unsafe_ptr())), phase
        )

    @always_inline("nodebug")
    fn try_wait(ref[AddressSpace.SHARED] self, phase: UInt32 = 0) -> Bool:
        """Non-blocking check if barrier phase is complete.

        Performs a single non-blocking check to see if the barrier has completed
        the specified phase. Returns immediately with the result without spinning.

        This is useful for implementing the try-acquire pattern where you want to
        overlap barrier checking with other useful work.

        Args:
            phase: The phase parity (0 or 1) to check for. Defaults to 0.

        Returns:
            True if the barrier phase is complete, False otherwise.

        Example:
            ```mojo
            # Try-acquire pattern for pipelined execution
            var ready = barrier.try_wait(phase)
            # Do other work while potentially waiting
            do_useful_work()
            # Now wait conditionally
            if not ready:
                barrier.wait(phase)
            ```
        """
        # PTX: mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [addr], phaseParity;
        return inlined_assembly[
            "mbarrier.try_wait.parity.shared::cta.b64 $0, [$1], $2;",
            Bool,
            constraints="=b,r,r",
        ](Int32(Int(self.unsafe_ptr())), phase)

    @always_inline
    fn unsafe_ptr[
        origin: Origin
    ](
        ref[origin, AddressSpace.SHARED] self,
    ) -> UnsafePointer[
        Int64,
        origin=origin,
        address_space = AddressSpace.SHARED,
    ]:
        """Get an unsafe pointer to the barrier's memory location.

        Provides low-level access to the shared memory location storing the barrier state.
        This method is primarily used internally by other barrier operations that need
        direct access to the underlying memory.

        Parameters:
            origin: Origin of self.

        Returns:
            An unsafe pointer to the barrier's memory location in shared memory,
            properly typed and aligned for barrier operations.
        """
        return UnsafePointer(to=self.mbar).unsafe_origin_cast[origin]()

    @always_inline
    fn arrive_cluster(
        ref[AddressSpace.SHARED] self, cta_id: UInt32, count: UInt32 = 1
    ):
        """Signal arrival at the barrier from a specific CTA (Cooperative Thread Array) in a cluster.

        This method is used in multi-CTA scenarios to coordinate barrier arrivals
        across different CTAs within a cluster. It enables efficient synchronization
        across thread blocks in clustered execution models.

        Args:
            cta_id: The ID of the CTA (Cooperative Thread Array) that is arriving.
            count: The number of arrivals to signal. Defaults to 1.
        """
        comptime asm = """{
            .reg .b32 remAddr32;
            mapa.shared::cluster.u32  remAddr32, $0, $1;
            mbarrier.arrive.shared::cluster.b64  _, [remAddr32], $2;
        }"""
        inlined_assembly[asm, NoneType, constraints="r,r,r"](
            Int32(Int(self.unsafe_ptr())), cta_id, count
        )

    @always_inline("nodebug")
    fn arrive[o: MutOrigin](ref[o, AddressSpace.SHARED] self) -> Int:
        """Signal arrival at the barrier and return the arrival count.

        This method increments the arrival count at the barrier and returns
        the updated count. It's used to track how many threads have reached
        the synchronization point.

        Returns:
            The updated arrival count after this thread's arrival.

        Parameters:
            o: Origin of self.
        """
        return mbarrier_arrive(self.unsafe_ptr())


struct PipelineState[num_stages: Int](Defaultable, TrivialRegisterPassable):
    """Manages state for a multi-stage pipeline with circular buffer semantics.

    PipelineState provides a mechanism for tracking the current stage in a
    multi-stage pipeline, particularly useful for double or triple buffering
    in GPU tensor operations. It maintains an index that cycles through the
    available stages, a phase bit that toggles when the index wraps around,
    and a monotonically increasing count.

    This struct is commonly used with TMA operations to coordinate the use of
    multiple buffers in a pipeline fashion, allowing for overlapping computation
    and data transfer.

    Parameters:
        num_stages: The number of stages in the pipeline (e.g., 2 for double buffering,
                   3 for triple buffering).
    """

    var _index: UInt32
    """The current stage index in the pipeline.

    This field tracks which buffer in the circular pipeline is currently active.
    Values range from 0 to num_stages-1 and wrap around when incremented past
    the last stage.
    """

    var _phase: UInt32
    """The current phase bit of the pipeline.

    This field alternates between 0 and 1 each time the index completes a full cycle.
    It's used to detect when a full pipeline cycle has completed, particularly
    useful for synchronization in producer-consumer scenarios.
    """

    var _count: UInt32
    """A monotonically increasing counter tracking pipeline iterations.

    This counter increments with each pipeline advancement, providing a
    total count of how many times the pipeline has been advanced since
    initialization. Useful for tracking progress and debugging.
    """

    @always_inline
    fn __init__(out self):
        """Initialize a PipelineState with default values.

        Creates a new PipelineState with index 0, phase 0, and count 0.
        """
        self._index = 0
        self._phase = 0
        self._count = 0

    @always_inline
    fn __init__(out self, index: Int, phase: Int, count: Int):
        """Initialize a PipelineState with specific values.

        Creates a new PipelineState with the specified index, phase, and count.

        Args:
            index: The initial stage index.
            phase: The initial phase value (0 or 1).
            count: The initial count value.
        """
        self._index = UInt32(index)
        self._phase = UInt32(phase)
        self._count = UInt32(count)

    @always_inline
    fn index(self) -> UInt32:
        """Get the current stage index.

        Returns:
            The current index value, which ranges from 0 to num_stages-1.
        """
        return self._index

    @always_inline
    fn phase(self) -> UInt32:
        """Get the current phase bit.

        Returns:
            The current phase value (0 or 1), which toggles when the index wraps around.
        """
        return self._phase

    @always_inline
    fn step(mut self):
        """Advance the pipeline state to the next stage.

        Increments the index and count. When the index reaches num_stages,
        it wraps around to 0 and toggles the phase bit.

        This function is used to move to the next buffer in a multi-buffer
        pipeline, implementing circular buffer semantics.
        """

        comptime if Self.num_stages > 1:
            self._index += 1
            self._count += 1
            if self._index == UInt32(Self.num_stages):
                self._index = 0
                self._phase ^= 1

        comptime if Self.num_stages == 1:
            self._count += 1
            self._phase ^= 1

    @always_inline
    fn next(mut self) -> Self:
        """Advance the pipeline state to the next stage and return the new state.

        This function is used to move to the next buffer in a multi-buffer
        pipeline, implementing circular buffer semantics.

        Returns:
            The new pipeline state after advancing to the next stage.
        """
        self.step()
        return self

    @always_inline
    fn __enter__(var self) -> Self:
        """Enter the context manager.

        Returns:
            The pipeline state instance for use in a `with` statement.
        """
        return self


# TMATensorTile is created on the host with specific memory and tile sizes.
# Each TMATensorTile provides an asynchronous load of a specific tile at specified tile coordinates.
#
struct TMATensorTile[
    dtype: DType,
    layout: Layout,
    desc_layout: Layout = layout,
    is_k_major: Bool = True,
](DevicePassable, ImplicitlyCopyable):
    """
    A hardware-accelerated tensor memory access (TMA) tile for efficient asynchronous data movement.

    The TMATensorTile struct provides a high-performance interface for asynchronous data transfers
    between global memory and shared memory in GPU tensor operations. It encapsulates a TMA descriptor
    that defines the memory access pattern and provides methods for various asynchronous operations.

    Parameters:
        dtype: DType
            The data type of the tensor elements.
        layout: Layout
            The layout of the tile in shared memory, typically specified as row_major.
        desc_layout: Layout = layout
            The layout of the descriptor, which can be different from the shared memory layout
            to accommodate hardware requirements like WGMMA.
        is_k_major: Bool = True
            Whether the shared memory is k-major.

    Performance:

        - Hardware-accelerated memory transfers using TMA instructions
        - Supports prefetching of descriptors for latency hiding
        - Enforces 128-byte alignment requirements for optimal memory access
    """

    var descriptor: TMADescriptor
    """The TMA descriptor that defines the memory access pattern.

    This field stores the hardware descriptor that encodes information about:
    - The source tensor's memory layout and dimensions
    - The tile shape and access pattern
    - Swizzling configuration for optimal memory access

    The descriptor is used by the GPU's Tensor Memory Accelerator hardware to
    efficiently transfer data between global and shared memory.
    """

    comptime device_type: AnyType = Self
    """The device-side type representation."""

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.

        Returns:
            This type's name.
        """
        return String(
            "TMATensorTile[dtype = ",
            Self.dtype,
            ", layout = ",
            materialize[Self.layout](),
            ", desc_layout = ",
            materialize[Self.desc_layout](),
            ", is_k_major = ",
            Self.is_k_major,
            "]",
        )

    @always_inline
    @implicit
    fn __init__(out self, descriptor: TMADescriptor):
        """
        Initializes a new TMATensorTile with the provided TMA descriptor.

        Args:
            descriptor: The TMA descriptor that defines the memory access pattern.
        """
        self.descriptor = descriptor

    @always_inline
    fn __init__(out self, *, copy: Self):
        """
        Copy initializes this `TMATensorTile` from another instance.

        Args:
            copy: The other `TMATensorTile` instance to copy from.
        """
        self.descriptor = copy.descriptor

    @always_inline
    fn prefetch_descriptor(self):
        """
        Prefetches the TMA descriptor into cache to reduce latency.

        This method helps hide memory access latency by prefetching the descriptor
        before it's needed for actual data transfers.
        """
        var desc_ptr = UnsafePointer(to=self.descriptor).bitcast[NoneType]()
        prefetch_tma_descriptor(desc_ptr)

    @always_inline
    fn async_copy[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[_, _, address_space = AddressSpace.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to the specified destination in shared memory. The transfer is tracked by the provided memory
        barrier.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 2D coordinates in the source tensor from which to copy data.

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime assert (
            type_of(dst).dtype == Self.dtype
        ), "Input tensor has a different type than the TMA op"

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = product(
            Self.layout.shape[Int(not Self.is_k_major)]
        ) // copy_dim0
        comptime num_copies_dim1 = product(
            Self.layout.shape[Int(Self.is_k_major)]
        ) // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                comptime assert (
                    copy_offset * UInt32(size_of[Self.dtype]())
                ) % 128 == 0, (
                    "copy_offset="
                    + String(copy_offset)
                    + ", size_of[dtype]()="
                    + String(size_of[Self.dtype]())
                    + "\nlayout="
                    + String(Self.layout)
                    + "\ndesc_layout="
                    + String(Self.desc_layout)
                )
                cp_async_bulk_tensor_shared_cluster_global[
                    cta_group=cta_group,
                    eviction_policy=eviction_policy,
                ](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + (j * copy_dim1),
                        coords[1] + (i * copy_dim0),
                    ),
                )

    @always_inline
    fn async_copy[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[
            mut=True,
            dtype = Self.dtype,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """TileTensor overload for 2D async copy from global to shared memory.

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the
                leader CTA needs to be notified upon completion.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 2D coordinates in the source tensor.
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = product(
            Self.layout.shape[Int(not Self.is_k_major)]
        ) // copy_dim0
        comptime num_copies_dim1 = product(
            Self.layout.shape[Int(Self.is_k_major)]
        ) // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )
                cp_async_bulk_tensor_shared_cluster_global[
                    cta_group=cta_group,
                    eviction_policy=eviction_policy,
                ](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + (j * copy_dim1),
                        coords[1] + (i * copy_dim0),
                    ),
                )

    @always_inline("nodebug")
    fn async_copy_3d[
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 3D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to the specified destination in shared memory for 3D tensors. The transfer is tracked by the
        provided memory barrier.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 3D coordinates in the source tensor from which to copy data.

        Parameters:
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_FIRST.

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )

        # This is the layout with which the descs themselves are arranged.
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        )

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        layout_of_descs(IntTuple(m, i, j)) * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global[
                        eviction_policy=eviction_policy
                    ](
                        dst.ptr.mut_cast[True]() + copy_offset,
                        UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + (j * copy_dim2),
                            coords[1] + (i * copy_dim1),
                            coords[2] + (m * copy_dim0),
                        ),
                    )

    @always_inline
    fn async_copy_4d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 4D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to the specified destination in shared memory for 4D tensors. The transfer is tracked by the
        provided memory barrier.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 4D coordinates in the source tensor from which to copy data.

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_dim3 = Self.desc_layout.shape[3].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )
        comptime num_copies_dim3 = ceildiv(
            Self.layout.shape[3].value(), copy_dim3
        )
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2, num_copies_dim3
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2, num_copies_dim3
        )

        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            layout_of_descs(IntTuple(n, m, i, j)) * copy_size
                        )

                        cp_async_bulk_tensor_shared_cluster_global[
                            cta_group=cta_group,
                            eviction_policy=eviction_policy,
                        ](
                            dst.ptr.mut_cast[True]() + copy_offset,
                            UnsafePointer(to=self.descriptor).bitcast[
                                NoneType
                            ](),
                            mem_barrier.unsafe_ptr(),
                            Index(
                                coords[0] + (j * copy_dim3),
                                coords[1] + (i * copy_dim2),
                                coords[2] + (m * copy_dim1),
                                coords[3] + (n * copy_dim0),
                            ),
                        )

    @always_inline
    fn async_copy_4d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[
            mut=True,
            dtype = Self.dtype,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 4D coordinates.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 4D coordinates in the source tensor from which to copy data.
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_dim3 = Self.desc_layout.shape[3].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )
        comptime num_copies_dim3 = ceildiv(
            Self.layout.shape[3].value(), copy_dim3
        )
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2, num_copies_dim3
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2, num_copies_dim3
        )

        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            layout_of_descs(IntTuple(n, m, i, j)) * copy_size
                        )

                        cp_async_bulk_tensor_shared_cluster_global[
                            cta_group=cta_group,
                            eviction_policy=eviction_policy,
                        ](
                            dst.ptr.mut_cast[True]() + copy_offset,
                            UnsafePointer(to=self.descriptor).bitcast[
                                NoneType
                            ](),
                            mem_barrier.unsafe_ptr(),
                            Index(
                                coords[0] + (j * copy_dim3),
                                coords[1] + (i * copy_dim2),
                                coords[2] + (m * copy_dim1),
                                coords[3] + (n * copy_dim0),
                            ),
                        )

    @always_inline
    fn async_copy_5d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 5D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to the specified destination in shared memory for 5D tensors. The transfer is tracked by the
        provided memory barrier.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 5D coordinates in the source tensor from which to copy data.

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_dim3 = Self.desc_layout.shape[3].value()
        comptime copy_dim4 = Self.desc_layout.shape[4].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )
        comptime num_copies_dim3 = ceildiv(
            Self.layout.shape[3].value(), copy_dim3
        )
        comptime num_copies_dim4 = ceildiv(
            Self.layout.shape[4].value(), copy_dim4
        )
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0,
            num_copies_dim1,
            num_copies_dim2,
            num_copies_dim3,
            num_copies_dim4,
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0,
            num_copies_dim1,
            num_copies_dim2,
            num_copies_dim3,
            num_copies_dim4,
        )

        comptime for o in range(num_copies_dim0):
            comptime for n in range(num_copies_dim1):
                comptime for m in range(num_copies_dim2):
                    comptime for i in range(num_copies_dim3):
                        comptime for j in range(num_copies_dim4):
                            comptime copy_offset: UInt32 = UInt32(
                                layout_of_descs(IntTuple(o, n, m, i, j))
                                * copy_size
                            )

                            cp_async_bulk_tensor_shared_cluster_global[
                                cta_group=cta_group,
                                eviction_policy=eviction_policy,
                            ](
                                dst.ptr.mut_cast[True]() + copy_offset,
                                UnsafePointer(to=self.descriptor).bitcast[
                                    NoneType
                                ](),
                                mem_barrier.unsafe_ptr(),
                                Index(
                                    coords[0] + (j * copy_dim4),
                                    coords[1] + (i * copy_dim3),
                                    coords[2] + (m * copy_dim2),
                                    coords[3] + (n * copy_dim1),
                                    coords[4] + (o * copy_dim0),
                                ),
                            )

    @always_inline
    fn async_copy_5d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[
            mut=True,
            dtype = Self.dtype,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 5D coordinates.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 5D coordinates in the source tensor from which to copy data.
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_dim3 = Self.desc_layout.shape[3].value()
        comptime copy_dim4 = Self.desc_layout.shape[4].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )
        comptime num_copies_dim3 = ceildiv(
            Self.layout.shape[3].value(), copy_dim3
        )
        comptime num_copies_dim4 = ceildiv(
            Self.layout.shape[4].value(), copy_dim4
        )
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0,
            num_copies_dim1,
            num_copies_dim2,
            num_copies_dim3,
            num_copies_dim4,
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0,
            num_copies_dim1,
            num_copies_dim2,
            num_copies_dim3,
            num_copies_dim4,
        )

        comptime for o in range(num_copies_dim0):
            comptime for n in range(num_copies_dim1):
                comptime for m in range(num_copies_dim2):
                    comptime for i in range(num_copies_dim3):
                        comptime for j in range(num_copies_dim4):
                            comptime copy_offset: UInt32 = UInt32(
                                layout_of_descs(IntTuple(o, n, m, i, j))
                                * copy_size
                            )

                            cp_async_bulk_tensor_shared_cluster_global[
                                cta_group=cta_group,
                                eviction_policy=eviction_policy,
                            ](
                                dst.ptr.mut_cast[True]() + copy_offset,
                                UnsafePointer(to=self.descriptor).bitcast[
                                    NoneType
                                ](),
                                mem_barrier.unsafe_ptr(),
                                Index(
                                    coords[0] + (j * copy_dim4),
                                    coords[1] + (i * copy_dim3),
                                    coords[2] + (m * copy_dim2),
                                    coords[3] + (n * copy_dim1),
                                    coords[4] + (o * copy_dim0),
                                ),
                            )

    @always_inline("nodebug")
    fn async_copy[
        rank: Int,
        //,
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: StaticTuple[UInt32, rank],
    ):
        """Schedules an asynchronous copy from global memory to shared memory for N-dimensional tensors.

        This is a generic dispatcher that selects the appropriate rank-specific async copy method
        based on the tensor rank. It provides a unified interface for initiating TMA transfers
        across 2D, 3D, 4D, and 5D tensors using `StaticTuple` coordinates.

        Parameters:
            rank: The dimensionality of the tensor (must be 2, 3, 4, or 5).
            cta_group: If set to 2, only the leader CTA needs to be notified upon completion.
                Defaults to 1.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The N-dimensional coordinates in the source tensor from which to copy data,
                provided as a `StaticTuple` of `UInt32` values.

        Constraints:
            - The rank must be 2, 3, 4, or 5.
            - The destination tensor must be 128-byte aligned in shared memory.
        """
        comptime assert rank in (2, 3, 4, 5)

        comptime if rank == 2:
            self.async_copy[eviction_policy=eviction_policy](
                dst, mem_barrier, (Int(coords[0]), Int(coords[1]))
            )
        elif rank == 3:
            self.async_copy_3d[eviction_policy=eviction_policy](
                dst,
                mem_barrier,
                (Int(coords[0]), Int(coords[1]), Int(coords[2])),
            )
        elif rank == 4:
            self.async_copy_4d[eviction_policy=eviction_policy](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                ),
            )
        elif rank == 5:
            self.async_copy_5d[eviction_policy=eviction_policy](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                    Int(coords[4]),
                ),
            )

    @always_inline
    fn async_store[
        rank: Int, //, cta_group: Int = 1
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        coords: StaticTuple[UInt32, rank],
    ):
        """Schedules an asynchronous store from shared memory to global memory for N-dimensional tensors.

        This is a generic dispatcher that selects the appropriate rank-specific async store method
        based on the tensor rank. It provides a unified interface for initiating TMA store operations
        across 2D, 3D, 4D, and 5D tensors using `StaticTuple` coordinates.

        Parameters:
            rank: The dimensionality of the tensor (must be 2, 3, 4, or 5).
            cta_group: CTA group configuration for the store operation. Defaults to 1.

        Args:
            dst: The source tensor in shared memory from which data will be copied to global memory.
                Must be 128-byte aligned.
            coords: The N-dimensional coordinates in the destination global tensor where data
                will be stored, provided as a `StaticTuple` of `UInt32` values.

        Constraints:
            - The rank must be 2, 3, 4, or 5.
            - The source tensor must be 128-byte aligned in shared memory.
        """
        comptime assert rank in (2, 3, 4, 5)

        comptime if rank == 2:
            self.async_store(dst, (UInt(coords[0]), UInt(coords[1])))
        elif rank == 3:
            self.async_store_3d(
                dst,
                (UInt(coords[0]), UInt(coords[1]), UInt(coords[2])),
            )
        elif rank == 4:
            self.async_store_4d(
                dst,
                (
                    UInt(coords[0]),
                    UInt(coords[1]),
                    UInt(coords[2]),
                    UInt(coords[3]),
                ),
            )
        elif rank == 5:
            self.async_store_5d(
                dst,
                (
                    UInt(coords[0]),
                    UInt(coords[1]),
                    UInt(coords[2]),
                    UInt(coords[3]),
                    UInt(coords[4]),
                ),
            )

    @always_inline
    fn async_store[
        rank: Int, //, cta_group: Int = 1
    ](
        self,
        dst: TileTensor[
            dtype = Self.dtype, address_space = AddressSpace.SHARED, ...
        ],
        coords: StaticTuple[UInt32, rank],
    ):
        """Schedules an asynchronous store from shared memory to global memory.

        TileTensor overload of the generic rank-dispatched async_store.
        Dispatches to the rank-specific TileTensor async_store methods.

        Parameters:
            rank: The dimensionality of the tensor (must be 2 or 3).
            cta_group: CTA group configuration. Defaults to 1.

        Args:
            dst: TileTensor in shared memory from which data will be copied.
            coords: The N-dimensional coordinates in the destination tensor.
        """
        comptime assert rank in (2, 3)

        comptime if rank == 2:
            self.async_store(dst, (UInt(coords[0]), UInt(coords[1])))
        elif rank == 3:
            self.async_store_3d(
                dst,
                (UInt(coords[0]), UInt(coords[1]), UInt(coords[2])),
            )

    @always_inline
    fn async_multicast_load[
        cta_group: Int = 1
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous multicast load from global memory to multiple shared memory locations.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to multiple destination locations in shared memory across different CTAs (Cooperative Thread Arrays)
        as specified by the multicast mask.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: LayoutTensor
                The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: SharedMemBarrierArray
                The memory barrier used to track and synchronize the asynchronous transfer.
            coords: Tuple[UInt, UInt]
                The 2D coordinates in the source tensor from which to copy data.
            multicast_mask: UInt16
                A bit mask specifying which CTAs should receive the data.

        Constraints:
            The destination tensor must be 128-byte aligned in shared memory.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = Self.layout.shape[0].value() // copy_dim0
        comptime num_copies_dim1 = Self.layout.shape[1].value() // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                cp_async_bulk_tensor_shared_cluster_global_multicast[
                    cta_group=cta_group
                ](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + UInt(j * copy_dim1),
                        coords[1] + UInt(i * copy_dim0),
                    ),
                    multicast_mask,
                )

    @always_inline
    fn async_multicast_load[
        cta_group: Int = 1,
    ](
        self,
        dst: TileTensor[
            mut=True,
            dtype = Self.dtype,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous 2D multicast load from global to shared memory.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 2D coordinates in the source tensor from which to copy.
            multicast_mask: Bit mask specifying which CTAs should receive the data.
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = Self.layout.shape[0].value() // copy_dim0
        comptime num_copies_dim1 = Self.layout.shape[1].value() // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                cp_async_bulk_tensor_shared_cluster_global_multicast[
                    cta_group=cta_group
                ](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + UInt(j * copy_dim1),
                        coords[1] + UInt(i * copy_dim0),
                    ),
                    multicast_mask,
                )

    @always_inline
    fn async_multicast_load_3d[
        cta_group: Int = 1
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt, UInt],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous 3D multicast load from global memory to multiple shared memory locations.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to multiple destination locations in shared memory across different CTAs (Cooperative Thread Arrays)
        as specified by the multicast mask.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: LayoutTensor
                The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: SharedMemBarrierArray
                The memory barrier used to track and synchronize the asynchronous transfer.
            coords: Tuple[UInt, UInt, UInt]
                The 2D coordinates in the source tensor from which to copy data.
            multicast_mask: UInt16
                A bit mask specifying which CTAs should receive the data.

        Constraints:
            The destination tensor must be 128-byte aligned in shared memory.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )

        # This is the layout with which the descs themselves are arranged.
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        )

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        layout_of_descs(IntTuple(m, i, j)) * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global_multicast[
                        cta_group=cta_group
                    ](
                        dst.ptr.mut_cast[True]() + copy_offset,
                        UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + UInt(j * copy_dim2),
                            coords[1] + UInt(i * copy_dim1),
                            coords[2] + UInt(m * copy_dim0),
                        ),
                        multicast_mask,
                    )

    @always_inline
    fn async_multicast_load_3d[
        cta_group: Int = 1,
    ](
        self,
        dst: TileTensor[
            mut=True,
            dtype = Self.dtype,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt, UInt],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous 3D multicast load from global to shared memory.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 3D coordinates in the source tensor from which to copy.
            multicast_mask: Bit mask specifying which CTAs should receive the data.
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )

        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        )

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        layout_of_descs(IntTuple(m, i, j)) * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global_multicast[
                        cta_group=cta_group
                    ](
                        dst.ptr.mut_cast[True]() + copy_offset,
                        UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + UInt(j * copy_dim2),
                            coords[1] + UInt(i * copy_dim1),
                            coords[2] + UInt(m * copy_dim0),
                        ),
                        multicast_mask,
                    )

    @always_inline
    fn async_multicast_load_partitioned[
        tma_rows: Int,
        tma_load_size: Int,
    ](
        self,
        dst: LayoutTensor[
            Self.dtype,
            _,
            address_space = AddressSpace.SHARED,
            alignment=128,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        rank: UInt,
        coords: Tuple[UInt, UInt],
        multicast_mask: UInt16,
    ):
        """
        Performs a partitioned multicast load where each rank loads a distinct slice of data.

        This method is designed for clustered execution where different ranks (CTAs) load
        different, contiguous slices of the source tensor. Each rank's slice is offset
        by `rank * tma_rows` in the second dimension and stored at offset `rank * tma_load_size`
        in shared memory.

        Parameters:
            tma_rows: The number of rows each rank is responsible for loading.
            tma_load_size: The size in elements of each rank's slice in shared memory.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            rank: The rank ID (0-based) that determines which slice to load.
            coords: The base 2D coordinates in the source tensor from which to copy data.
                   The second coordinate will be offset by `rank * tma_rows`.
            multicast_mask: A bit mask specifying which CTAs should receive the data.

        Note:
            This is typically used in matrix multiplication kernels where the input matrices
            are partitioned across multiple CTAs for parallel processing.
        """
        var dst_slice = LayoutTensor[
            Self.dtype,
            dst.layout,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ](dst.ptr + rank * UInt(tma_load_size))

        self.async_multicast_load(
            dst_slice,
            mem_barrier,
            (coords[0], coords[1] + rank * UInt(tma_rows)),
            multicast_mask,
        )

    @always_inline
    fn async_store(
        self,
        src: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        coords: Tuple[UInt, UInt],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory.

        This method initiates a hardware-accelerated asynchronous transfer of data from shared memory
        to global memory at the specified coordinates.

        Args:
            src: LayoutTensor
                The source tensor in shared memory from which data will be copied.
                Must be 128-byte aligned.
            coords: Tuple[UInt, UInt]
                The 2D coordinates in the destination tensor where data will be stored.

        Constraints:
            The source tensor must be 128-byte aligned in shared memory.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = product(
            Self.layout.shape[Int(not Self.is_k_major)]
        ) // copy_dim0
        comptime num_copies_dim1 = product(
            Self.layout.shape[Int(Self.is_k_major)]
        ) // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                cp_async_bulk_tensor_global_shared_cta(
                    src.ptr + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    Index(
                        coords[0] + UInt(j * copy_dim1),
                        coords[1] + UInt(i * copy_dim0),
                    ),
                )

    @always_inline
    fn async_store(
        self,
        src: TileTensor[
            dtype = Self.dtype, address_space = AddressSpace.SHARED, ...
        ],
        coords: Tuple[UInt, UInt],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Args:
            src: TileTensor in shared memory from which data will be copied.
            coords: The 2D coordinates in the destination tensor where data will be stored.
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = product(
            Self.layout.shape[Int(not Self.is_k_major)]
        ) // copy_dim0
        comptime num_copies_dim1 = product(
            Self.layout.shape[Int(Self.is_k_major)]
        ) // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                cp_async_bulk_tensor_global_shared_cta(
                    src.ptr + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    Index(
                        coords[0] + UInt(j * copy_dim1),
                        coords[1] + UInt(i * copy_dim0),
                    ),
                )

    @always_inline
    fn async_store_3d(
        self,
        src: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        coords: Tuple[UInt, UInt, UInt],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at specified 3D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from shared memory
        to the specified destination in global memory for 3D tensors.

        Args:
            src: The source tensor in shared memory from which data will be copied.
                 Must be 128-byte aligned.
            coords: The 3D coordinates in the destination tensor where data will be stored.

        Constraints:

            - The source tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )

        # This is the layout with which the descs themselves are arranged.
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        )

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        layout_of_descs(IntTuple(m, i, j)) * copy_size
                    )

                    cp_async_bulk_tensor_global_shared_cta(
                        src.ptr + copy_offset,
                        UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                        Index(
                            coords[0] + UInt(j * copy_dim2),
                            coords[1] + UInt(i * copy_dim1),
                            coords[2] + UInt(m * copy_dim0),
                        ),
                    )

    @always_inline
    fn async_store_3d(
        self,
        src: TileTensor[
            dtype = Self.dtype, address_space = AddressSpace.SHARED, ...
        ],
        coords: Tuple[UInt, UInt, UInt],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at 3D coordinates.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Args:
            src: TileTensor in shared memory from which data will be copied.
            coords: The 3D coordinates in the destination tensor.
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )

        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2
        )

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        layout_of_descs(IntTuple(m, i, j)) * copy_size
                    )

                    cp_async_bulk_tensor_global_shared_cta(
                        src.ptr + copy_offset,
                        UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                        Index(
                            coords[0] + UInt(j * copy_dim2),
                            coords[1] + UInt(i * copy_dim1),
                            coords[2] + UInt(m * copy_dim0),
                        ),
                    )

    @always_inline
    fn async_store_4d(
        self,
        src: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        coords: Tuple[UInt, UInt, UInt, UInt],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at specified 4D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from shared memory
        to the specified destination in global memory for 4D tensors.

        Args:
            src: The source tensor in shared memory from which data will be copied.
                 Must be 128-byte aligned.
            coords: The 4D coordinates in the destination tensor where data will be stored.

        Constraints:

            - The source tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_dim3 = Self.desc_layout.shape[3].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )
        comptime num_copies_dim3 = ceildiv(
            Self.layout.shape[3].value(), copy_dim3
        )
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2, num_copies_dim3
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0, num_copies_dim1, num_copies_dim2, num_copies_dim3
        )

        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            layout_of_descs(IntTuple(n, m, i, j)) * copy_size
                        )

                        cp_async_bulk_tensor_global_shared_cta(
                            src.ptr + copy_offset,
                            UnsafePointer(to=self.descriptor).bitcast[
                                NoneType
                            ](),
                            Index(
                                coords[0] + UInt(j * copy_dim3),
                                coords[1] + UInt(i * copy_dim2),
                                coords[2] + UInt(m * copy_dim1),
                                coords[3] + UInt(n * copy_dim0),
                            ),
                        )

    @always_inline
    fn async_store_5d(
        self,
        src: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        coords: Tuple[UInt, UInt, UInt, UInt, UInt],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at specified 5D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from shared memory
        to the specified destination in global memory for 5D tensors.

        Args:
            src: The source tensor in shared memory from which data will be copied.
                 Must be 128-byte aligned.
            coords: The 5D coordinates in the destination tensor where data will be stored.

        Constraints:

            - The source tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_dim2 = Self.desc_layout.shape[2].value()
        comptime copy_dim3 = Self.desc_layout.shape[3].value()
        comptime copy_dim4 = Self.desc_layout.shape[4].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = ceildiv(
            Self.layout.shape[0].value(), copy_dim0
        )
        comptime num_copies_dim1 = ceildiv(
            Self.layout.shape[1].value(), copy_dim1
        )
        comptime num_copies_dim2 = ceildiv(
            Self.layout.shape[2].value(), copy_dim2
        )
        comptime num_copies_dim3 = ceildiv(
            Self.layout.shape[3].value(), copy_dim3
        )
        comptime num_copies_dim4 = ceildiv(
            Self.layout.shape[4].value(), copy_dim4
        )
        comptime layout_of_descs = Layout.col_major(
            num_copies_dim0,
            num_copies_dim1,
            num_copies_dim2,
            num_copies_dim3,
            num_copies_dim4,
        ) if Self.is_k_major else Layout.row_major(
            num_copies_dim0,
            num_copies_dim1,
            num_copies_dim2,
            num_copies_dim3,
            num_copies_dim4,
        )

        comptime for o in range(num_copies_dim0):
            comptime for n in range(num_copies_dim1):
                comptime for m in range(num_copies_dim2):
                    comptime for i in range(num_copies_dim3):
                        comptime for j in range(num_copies_dim4):
                            comptime copy_offset: UInt32 = UInt32(
                                layout_of_descs(IntTuple(o, n, m, i, j))
                                * copy_size
                            )

                            cp_async_bulk_tensor_global_shared_cta(
                                src.ptr + copy_offset,
                                UnsafePointer(to=self.descriptor).bitcast[
                                    NoneType
                                ](),
                                Index(
                                    coords[0] + UInt(j * copy_dim4),
                                    coords[1] + UInt(i * copy_dim3),
                                    coords[2] + UInt(m * copy_dim2),
                                    coords[3] + UInt(n * copy_dim1),
                                    coords[4] + UInt(o * copy_dim0),
                                ),
                            )

    @always_inline
    fn async_reduce[
        reduction_kind: ReduceOp
    ](
        self,
        src: LayoutTensor[
            Self.dtype, Self.layout, address_space = AddressSpace.SHARED, ...
        ],
        coords: Tuple[UInt, UInt],
    ):
        """
        Schedules an asynchronous reduction operation from shared memory to global memory.

        This method initiates a hardware-accelerated asynchronous reduction operation that combines
        data from shared memory with data in global memory using the specified reduction operation.
        The reduction is performed element-wise at the specified coordinates in the global tensor.

        Parameters:
            reduction_kind: The type of reduction operation to perform (e.g., ADD, MIN, MAX).
                           This determines how values are combined during the reduction.

        Args:
            src: The source tensor in shared memory containing the data to be reduced.
                 Must be 128-byte aligned.
            coords: The 2D coordinates in the destination tensor where the reduction will be applied.

        Constraints:
            The source tensor must be 128-byte aligned in shared memory.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"
        cp_async_bulk_tensor_reduce[reduction_kind=reduction_kind](
            src.ptr,
            UnsafePointer(to=self.descriptor).bitcast[NoneType](),
            Index(coords[0], coords[1]),
        )

    @always_inline
    fn commit_group(self):
        """
        Commits all prior initiated but uncommitted TMA instructions into a group.

        This function behaves the same as `cp_async_bulk_commit_group`, which creates
        a synchronization point for bulk TMA transfer.
        """
        cp_async_bulk_commit_group()

    @always_inline
    fn wait_group[n: Int = 0](self):
        """
        Wait for the completion of asynchronous copy until a specified number of groups are waiting.

        This function behaves the same as `cp_async_bulk_wait_group`, which causes the executing
        thread to wait until a specified number of the most recent TMA copy are pending.

        Parameters:
            n: The number of pending groups left.
        """
        cp_async_bulk_wait_group[Int32(n)]()

    @always_inline
    fn smem_tensormap_init(
        self,
        smem_tma_descriptor_ptr: UnsafePointer[
            TMADescriptor, _, address_space = AddressSpace.SHARED
        ],
    ):
        """
        Initializes a TMA descriptor in shared memory from this tensor tile's descriptor.

        This method copies the TMA descriptor from global memory to shared memory, allowing
        for faster access during kernel execution. The descriptor is copied in 16-byte chunks
        using asynchronous copy operations for efficiency.

        Args:
            smem_tma_descriptor_ptr: Pointer to the location in shared memory where the
                                    descriptor will be stored. Must be properly aligned.

        Note:

            - Only one thread should call this method to avoid race conditions
            - The descriptor is copied in 8 chunks of 16 bytes each (total 128 bytes)
        """
        # NOTE: Only one thread should call this

        var src_desc = (
            UnsafePointer(to=self.descriptor)
            .bitcast[UInt8]()
            .address_space_cast[AddressSpace.GLOBAL]()
        )
        var dst_desc = smem_tma_descriptor_ptr.bitcast[UInt8]().unsafe_mut_cast[
            True
        ]()

        comptime simd_width = simd_width_of[DType.uint8]()
        comptime src_align = align_of[SIMD[DType.uint8, simd_width]]()
        comptime dst_align = align_of[SIMD[DType.uint8, simd_width]]()

        comptime descriptor_bytes = 128

        comptime for src_idx in range(descriptor_bytes // simd_width):
            var src_vec = (src_desc).load[
                width=simd_width, alignment=src_align
            ](src_idx * simd_width)
            dst_desc.store[alignment=dst_align](src_idx * simd_width, src_vec)

    @always_inline
    fn replace_tensormap_global_address_in_gmem[
        _dtype: DType,
    ](self, src_ptr: UnsafePointer[Scalar[_dtype], _],):
        """
        Replaces the global memory address in the TMA descriptor stored in global memory.

        This method allows dynamically changing the source tensor for TMA operations without
        recreating the entire descriptor, which is useful for reusing descriptors with different
        data sources. The operation modifies the descriptor in global memory directly.


        Parameters:
            _dtype: The data type of the new source tensor.

        Args:
            src_ptr: The new source tensor whose address will replace the current one in the descriptor.
                    Must have compatible layout with the original tensor.

        Note:
            A memory fence may be required after this operation to ensure visibility
            of the changes to other threads.
        """

        comptime assert src_ptr.address_space in (
            AddressSpace.GENERIC,
            AddressSpace.GLOBAL,
        ), "src address space must be GENERIC or GLOBAL."

        var desc_ptr = UnsafePointer(to=self.descriptor).bitcast[NoneType]()

        inlined_assembly[
            "tensormap.replace.tile.global_address.global.b1024.b64 [$0], $1;",
            NoneType,
            constraints="l,l",
            has_side_effect=True,
        ](desc_ptr, src_ptr.bitcast[NoneType]())

    @always_inline
    fn tensormap_fence_acquire(self):
        """
        Establishes a memory fence for TMA operations with acquire semantics.

        This method ensures proper ordering of memory operations by creating a barrier
        that prevents subsequent TMA operations from executing before prior operations
        have completed. It is particularly important when reading from a descriptor
        that might have been modified by other threads or processes.

        The acquire semantics ensure that all memory operations after this fence
        will observe any modifications made to the descriptor before the fence.

        Notes:

            - The entire warp must call this function as the instruction is warp-aligned.
            - Typically used in pairs with `tensormap_fence_release` for proper synchronization.
        """
        # NOTE: Entire warp must call this function as the instruction is aligned
        llvm_intrinsic[
            "llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu", NoneType
        ](
            UnsafePointer(to=self.descriptor).bitcast[NoneType](),
            Int32(128),
        )

    @always_inline
    fn tensormap_fence_release(self):
        """
        Establishes a memory fence for TMA operations with release semantics.

        This method ensures proper ordering of memory operations by creating a barrier
        that ensures all prior memory operations are visible before subsequent operations
        can proceed. It is particularly important when modifying a TMA descriptor in
        global memory that might be read by other threads or processes.

        The release semantics ensure that all memory operations before this fence
        will be visible to any thread that observes operations after the fence.

        Notes:

            - Typically used after modifying a tensormap descriptor in global memory.
            - Often paired with `tensormap_fence_acquire` for proper synchronization.
        """
        # This fence is needed when modifying tensormap directly in GMEM
        llvm_intrinsic[
            "llvm.nvvm.fence.proxy.tensormap_generic.release.gpu", NoneType
        ]()

    @always_inline
    fn replace_tensormap_global_address_in_shared_mem[
        _dtype: DType,
    ](
        self,
        smem_tma_descriptor_ptr: UnsafePointer[
            TMADescriptor,
            _,
            address_space = AddressSpace.SHARED,
        ],
        src_ptr: UnsafePointer[Scalar[_dtype], _],
    ):
        """
        Replaces the global memory address in the TMA descriptor stored in shared memory.

        This method allows dynamically changing the source tensor for TMA operations without
        recreating the entire descriptor, which is useful for reusing descriptors with different
        data sources. The operation modifies a descriptor that has been previously copied to
        shared memory.


        Parameters:
            _dtype: The data type of the new source tensor.

        Args:
            smem_tma_descriptor_ptr: Pointer to the TMA descriptor in shared memory that will be modified.
            src_ptr: The new source tensor whose address will replace the current one in the descriptor.

        Notes:

            - Only one thread should call this method to avoid race conditions.
            - A memory fence may be required after this operation to ensure visibility
              of the changes to other threads.
            - Typically used with descriptors previously initialized with `smem_tensormap_init`.
        """

        comptime assert src_ptr.address_space in (
            AddressSpace.GENERIC,
            AddressSpace.GLOBAL,
        ), "src address space must be GENERIC or GLOBAL."

        # NOTE: Only one thread should call this
        inlined_assembly[
            (
                "tensormap.replace.tile.global_address.shared::cta.b1024.b64"
                " [$0], $1;"
            ),
            NoneType,
            constraints="r,l",
            has_side_effect=True,
        ](
            smem_tma_descriptor_ptr.bitcast[NoneType](),
            src_ptr.bitcast[NoneType](),
        )

    @always_inline
    fn tensormap_cp_fence_release(
        self,
        smem_tma_descriptor_ptr: UnsafePointer[
            TMADescriptor, address_space = AddressSpace.SHARED
        ],
    ):
        """
        Establishes a memory fence for TMA operations with release semantics for shared memory descriptors.

        This method ensures proper ordering of memory operations by creating a barrier
        that ensures all prior memory operations are visible before subsequent operations
        can proceed. It is specifically designed for synchronizing between global memory and
        shared memory TMA descriptors.

        The release semantics ensure that all memory operations before this fence
        will be visible to any thread that observes operations after the fence.

        Args:
            smem_tma_descriptor_ptr: Pointer to the TMA descriptor in shared memory that
                                    is being synchronized with the global memory descriptor.

        Notes:

            - The entire warp must call this function as the instruction is warp-aligned
            - Typically used after modifying a tensormap descriptor in shared memory
            - More specialized than the general `tensormap_fence_release` for cross-memory space synchronization
        """
        # This fence is needed when modifying tensormap directly in SMEM
        # NOTE: Entire warp must call this function as the instruction is aligned
        var gmem_tma_descriptor_ptr = UnsafePointer(to=self.descriptor).bitcast[
            NoneType
        ]()

        inlined_assembly[
            (
                "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"
                " [$0], [$1], 128;"
            ),
            NoneType,
            constraints="l,r",
            has_side_effect=True,
        ](gmem_tma_descriptor_ptr, smem_tma_descriptor_ptr.bitcast[NoneType]())

    @always_inline
    fn replace_tensormap_global_dim_strides_in_shared_mem[
        _dtype: DType,
        only_update_dim_0: Bool,
        /,
        *,
        rank: Int,
    ](
        self,
        smem_tma_descriptor_ptr: UnsafePointer[
            TMADescriptor, address_space = AddressSpace.SHARED, ...
        ],
        gmem_dims: IndexList[rank],
        gmem_strides: IndexList[rank],
    ):
        """
        Replaces dimensions and strides in a TMA descriptor stored in shared memory.
        Note: This function is only supported for CUDA versions >= 12.5.

        This function allows dynamically modifying the dimensions and strides of a TMA
        descriptor that has been previously initialized in shared memory. If only the first dimension (dim 0) is updated, then updating strides can be skipped.

        Parameters:
            _dtype: The data type of the new source tensor.
            only_update_dim_0: If true, only the first dimension (dim 0) is updated with updating strides.
            rank: The rank of the tensor.

        Args:
            smem_tma_descriptor_ptr: Pointer to the TMA descriptor in shared memory that will be modified.
            gmem_dims: The global dimensions of the tensor to be updated.
            gmem_strides: The global strides of the tensor to be updated.

        Notes:
            - Only one thread should call this method to avoid race conditions.
            - A memory fence may be required after this operation to ensure visibility
            of the changes to other threads.
        """

        var desc_ptr = smem_tma_descriptor_ptr.bitcast[UInt64]()

        comptime if only_update_dim_0:
            comptime temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
                rank - 1
            ) + ", $1;"
            inlined_assembly[
                temp,
                NoneType,
                constraints="l,r",
                has_side_effect=True,
            ](desc_ptr, gmem_dims[0])

        else:
            # Replace dimensions
            comptime for i in range(rank):
                comptime temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
                    i
                ) + ", $1;"
                inlined_assembly[
                    temp,
                    NoneType,
                    constraints="l,r",
                    has_side_effect=True,
                ](desc_ptr, gmem_dims[rank - i - 1])

            # Replace strides - note: stride for innermost dimension is implicitly 1
            # For CUDA versions >= 12.5, we use the full stride value. Note that this is not true for all CUDA versions and strides shound be left shifted by 4 for CUDA versions < 12.5
            comptime for i in range(1, rank):
                comptime temp = "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [$0], " + String(
                    i - 1
                ) + ", $1;"
                inlined_assembly[
                    temp,
                    NoneType,
                    constraints="l,l",
                    has_side_effect=True,
                ](desc_ptr, gmem_strides[rank - i - 1] * size_of[Self.dtype]())

    @always_inline
    fn replace_tensormap_global_dim_strides_in_shared_mem[
        _dtype: DType,
        tensor_rank: Int,
        dim_idx: Int,
    ](
        self,
        smem_tma_descriptor_ptr: UnsafePointer[
            TMADescriptor, address_space = AddressSpace.SHARED, ...
        ],
        dim_value: UInt32,
        dim_stride: Optional[UInt64] = None,
    ):
        """
        Replaces dimensions and strides in a TMA descriptor stored in shared memory.
        Note: This function is only supported for CUDA versions >= 12.5.
        This function allows dynamically modifying the dimensions and strides of a TMA
        descriptor that has been previously initialized in shared memory. If only the first dimension is updated, then updating strides can be skipped.

        Parameters:
            _dtype: The data type of the source tensor in GMEM.
            tensor_rank: The rank of the source tensor in GMEM.
            dim_idx: The index of the dimension to be updated in the TMA descriptor with the provided dimension and stride values at runtime.

        Args:
            smem_tma_descriptor_ptr: Pointer to the TMA descriptor in shared memory that will be modified.
            dim_value: The new dimension value to be set.
            dim_stride: The new stride value to be set.

        Notes:
            - Only one thread should call this method to avoid race conditions.
            - A memory fence may be required after this operation to ensure visibility
            of the changes to other threads.
        """

        var desc_ptr = smem_tma_descriptor_ptr.bitcast[UInt64]()

        # Replace dimensions

        comptime temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
            tensor_rank - dim_idx - 1
        ) + ", $1;"
        inlined_assembly[
            temp,
            NoneType,
            constraints="l,r",
            has_side_effect=True,
        ](desc_ptr, dim_value)

        # Replace strides - note: stride for innermost dimension is implicitly 1
        # For CUDA versions >= 12.5, we use the full stride value. Note that this is not true for all CUDA versions and strides shound be left shifted by 4 for CUDA versions < 12.5
        comptime if dim_idx > 0:
            debug_assert(
                dim_stride is not None,
                " dim_stride must be provided if dim_idx > 0",
            )
            comptime temp = "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [$0], " + String(
                tensor_rank - dim_idx - 1
            ) + ", $1;"
            inlined_assembly[
                temp,
                NoneType,
                constraints="l,l",
                has_side_effect=True,
            ](desc_ptr, dim_stride)


@always_inline
def create_tma_tile[
    *tile_sizes: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext, tensor: LayoutTensor) -> TMATensorTile[
    tensor.dtype,
    Layout.row_major(_to_int_tuple[*tile_sizes]()),
]:
    """
    Creates a `TMATensorTile` with specified tile dimensions and swizzle mode.

    This function creates a hardware-accelerated Tensor Memory Access (TMA) descriptor
    for efficient asynchronous data transfers between global memory and shared memory.
    It configures the tile dimensions and memory access patterns based on the provided
    parameters.

    Parameters:
        tile_sizes: The dimensions of the tile to be transferred. For 2D tensors, this should be
            [height, width]. The dimensions determine the shape of data transferred in each
            TMA operation.
        swizzle_mode:
            The swizzling mode to use for memory access optimization. Swizzling can improve
            memory access patterns for specific hardware configurations.

    Args:
        ctx:
            The CUDA device context used to create the TMA descriptor.
        tensor:
            The source tensor from which data will be transferred. This defines the
            global memory layout and data type.

    Returns:
        A `TMATensorTile` configured with the specified tile dimensions and swizzle mode,
        ready for use in asynchronous data transfer operations.

    Constraints:

        - The last dimension's size in bytes must not exceed the swizzle mode's byte limit
          (32B for SWIZZLE_32B, 64B for SWIZZLE_64B, 128B for SWIZZLE_128B).
        - Only supports 2D tensors in this overload.
    """
    # the last dimension of smem shape has to be smaller or equals to the
    # swizzle bytes.
    comptime swizzle_rows_bytes = tile_sizes[tensor.rank - 1] * size_of[
        tensor.dtype
    ]()

    comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        comptime assert swizzle_rows_bytes <= swizzle_mode.bytes(), (
            "Current swizzle bytes is "
            + String(swizzle_rows_bytes)
            + " which exceeds "
            + String(swizzle_mode.bytes())
            + "B swizzle requirement."
        )

    return create_tma_descriptor[tensor.dtype, 2, swizzle_mode](
        DeviceBuffer(
            ctx,
            tensor.ptr.unsafe_mut_cast[True]().address_space_cast[
                AddressSpace.GENERIC
            ](),
            1,
            owning=False,
        ),
        (tensor.dim(0), tensor.dim(1)),
        (tensor.stride(0), tensor.stride(1)),
        (tile_sizes[0], tile_sizes[1]),
    )


@always_inline
def _create_tma_descriptor_helper[
    dtype: DType,
    rank: Int,
    //,
    desc_index_list: IndexList[rank],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext, tensor: LayoutTensor[dtype, ...]) -> TMADescriptor:
    """
    Helper function to create a TMA descriptor from a global memory layout tensor.

    This internal function creates a hardware-accelerated Tensor Memory Access (TMA) descriptor
    for efficient asynchronous data transfers between global memory and shared memory.
    It validates the tensor rank, flattens the layout shape and strides, and ensures
    swizzle mode compatibility with the tile dimensions.

    Parameters:
        dtype: The data type of the tensor elements.
        rank: The rank (number of dimensions) of the tensor.
        desc_index_list:
            The dimensions of the tile descriptor in each dimension. This defines the shape
            of data transferred in each TMA operation.
        swizzle_mode:
            The swizzling mode to use for memory access optimization. Swizzling can improve
            memory access patterns for specific hardware configurations. Defaults to SWIZZLE_NONE.

    Args:
        ctx:
            The CUDA device context used to create the TMA descriptor.
        tensor:
            The source layout tensor from which data will be transferred. This defines the
            global memory layout and data type.

    Returns:
        A `TMADescriptor` configured with the specified tile dimensions and swizzle mode,
        ready for use in asynchronous data transfer operations.

    Constraints:
        - The tensor rank must match the specified rank parameter.
        - When swizzling is enabled, the last dimension's size in bytes (calculated as
          `desc_index_list[rank-1] * sizeof(dtype)`) must not exceed the swizzle mode's
          byte limit (32B for SWIZZLE_32B, 64B for SWIZZLE_64B, 128B for SWIZZLE_128B).
    """

    comptime assert rank == tensor.rank, "Rank mismatch"

    var global_shape = coalesce_nested_tuple(tensor.runtime_layout.shape)
    var global_strides = coalesce_nested_tuple(tensor.runtime_layout.stride)

    comptime swizzle_rows_bytes = desc_index_list[rank - 1] * size_of[
        tensor.dtype
    ]()

    var global_shape_list = runtime_tuple_to_index_list[rank](global_shape)
    var global_strides_list = runtime_tuple_to_index_list[rank](global_strides)

    comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        comptime assert swizzle_rows_bytes <= swizzle_mode.bytes(), (
            "Current swizzle bytes is "
            + String(swizzle_rows_bytes)
            + " which exceeds "
            + String(swizzle_mode.bytes())
            + "B swizzle requirement."
        )

    return create_tma_descriptor[tensor.dtype, rank, swizzle_mode](
        DeviceBuffer(
            ctx,
            tensor.ptr.mut_cast[True]().address_space_cast[
                AddressSpace.GENERIC
            ](),
            1,
            owning=False,
        ),
        global_shape_list,
        global_strides_list,
        desc_index_list,
    )


@always_inline
def create_tensor_tile[
    dtype: DType,
    rank: Int,
    //,
    tile_shape: IndexList[rank],
    /,
    k_major_tma: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1]),
    __desc_layout: Layout = _tma_desc_tile_layout[
        dtype, rank, tile_shape, swizzle_mode
    ](),
](ctx: DeviceContext, tensor: LayoutTensor[dtype, ...]) -> TMATensorTile[
    dtype,
    __tile_layout,
    __desc_layout,
    is_k_major=k_major_tma,
]:
    """
    Creates a `TMATensorTile` with advanced configuration options for 2D, 3D, 4D, or 5D tensors.

    This overload provides more control over the TMA descriptor creation, allowing
    specification of data type, rank, and layout orientation. It supports 2D, 3D, 4D, and 5D
    tensors and provides fine-grained control over the memory access patterns.

    Parameters:
        dtype: DType
            The data type of the tensor elements.
        rank: Int
            The dimensionality of the tensor (must be 2, 3, 4, or 5).
        tile_shape: IndexList[rank]
            The shape of the tile to be transferred.
        k_major_tma: Bool = True
            Whether the tma should copy desc into shared memory following a
            column-major (if `True`) or row-major (if `False`) pattern.
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE
            The swizzling mode to use for memory access optimization.
        __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1])
            Internal parameter for the tile layout in shared memory.
        __desc_layout: Layout = _tma_desc_tile_layout[...]
            Internal parameter for the descriptor layout, which may differ from the
            tile layout to accommodate hardware requirements.

    Args:
        ctx: DeviceContext
            The CUDA device context used to create the TMA descriptor.
        tensor: LayoutTensor[dtype, ...]
            The source tensor from which data will be transferred. This defines the
            global memory layout and must match the specified data type.

    Returns:
        A `TMATensorTile` configured with the specified parameters, ready for use in
        asynchronous data transfer operations.

    Constraints:

        - Only supports 2D, 3D, 4D, and 5D tensors (rank must be 2, 3, 4, or 5).
        - For non-SWIZZLE_NONE modes, the K dimension size in bytes must be a multiple
          of the swizzle mode's byte size.
        - For MN-major layout, only SWIZZLE_128B is supported.
        - For 3D, 4D, and 5D tensors, only K-major layout is supported.
    """
    # Current impl limitations
    comptime assert (
        rank == 2 or rank == 3 or rank == 4 or rank == 5
    ), "Only support 2D/3D/4D/5D TMA"

    comptime desc_bytes_size = __desc_layout.size() * size_of[dtype]()
    comptime layout_size = __tile_layout.size() * size_of[dtype]()

    comptime if desc_bytes_size < layout_size:
        # When we do multiple TMA copy, every address has to be align to 128.
        comptime assert desc_bytes_size % 128 == 0, (
            "desc layout byte size has to be  align to 128 bytes for"
            " multiple TMA copies. desc_layout: "
            + String(__desc_layout.shape[0].value())
            + " "
            + String(__desc_layout.shape[1].value())
            + " tile_layout: "
            + String(__tile_layout.shape[0].value())
            + " "
            + String(__tile_layout.shape[1].value())
        )

    comptime if rank == 2:
        comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            comptime assert (
                tile_shape[1] * size_of[dtype]()
            ) % swizzle_mode.bytes() == 0, (
                String(swizzle_mode)
                + " mode requires K dim multiple of "
                + String(swizzle_mode.bytes())
                + "B. K dim is now "
                + String(tile_shape[1] * size_of[dtype]())
                + " bytes."
            )

        return create_tma_descriptor[dtype, 2, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
                1,
                owning=False,
            ),
            (tensor.dim(0), tensor.dim(1)),
            (tensor.stride(0), tensor.stride(1)),
            (__desc_layout.shape[0].value(), __desc_layout.shape[1].value()),
        )

    elif rank == 3:
        comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            comptime assert (
                tile_shape[2] * size_of[dtype]()
            ) % swizzle_mode.bytes() == 0, (
                String(swizzle_mode)
                + " mode requires K dim multiple of "
                + String(swizzle_mode.bytes())
                + "B. K dim is now "
                + String(tile_shape[2] * size_of[dtype]())
                + "bytes."
            )

        return create_tma_descriptor[dtype, 3, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[3](tensor.dim(0), tensor.dim(1), tensor.dim(2)),
            IndexList[3](tensor.stride(0), tensor.stride(1), tensor.stride(2)),
            IndexList[3](
                __desc_layout.shape[0].value(),
                __desc_layout.shape[1].value(),
                __desc_layout.shape[2].value(),
            ),
        )

    elif rank == 4:
        comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            comptime assert (
                tile_shape[3] * size_of[dtype]()
            ) % swizzle_mode.bytes() == 0, (
                String(swizzle_mode)
                + " mode requires K dim multiple of "
                + String(swizzle_mode.bytes())
                + "B. K dim is now "
                + String(tile_shape[3] * size_of[dtype]())
                + "bytes."
            )

        return create_tma_descriptor[dtype, 4, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[4](
                tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3)
            ),
            IndexList[4](
                tensor.stride(0),
                tensor.stride(1),
                tensor.stride(2),
                tensor.stride(3),
            ),
            IndexList[4](
                __desc_layout.shape[0].value(),
                __desc_layout.shape[1].value(),
                __desc_layout.shape[2].value(),
                __desc_layout.shape[3].value(),
            ),
        )

    else:  # rank == 5
        comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            comptime assert (
                tile_shape[4] * size_of[dtype]()
            ) % swizzle_mode.bytes() == 0, (
                String(swizzle_mode)
                + " mode requires K dim multiple of "
                + String(swizzle_mode.bytes())
                + "B. K dim is now "
                + String(tile_shape[4] * size_of[dtype]())
                + "bytes."
            )

        return create_tma_descriptor[dtype, 5, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[5](
                tensor.dim(0),
                tensor.dim(1),
                tensor.dim(2),
                tensor.dim(3),
                tensor.dim(4),
            ),
            IndexList[5](
                tensor.stride(0),
                tensor.stride(1),
                tensor.stride(2),
                tensor.stride(3),
                tensor.stride(4),
            ),
            IndexList[5](
                __desc_layout.shape[0].value(),
                __desc_layout.shape[1].value(),
                __desc_layout.shape[2].value(),
                __desc_layout.shape[3].value(),
                __desc_layout.shape[4].value(),
            ),
        )


@always_inline
def create_tensor_tile[
    dtype: DType,
    rank: Int,
    //,
    tile_shape: IndexList[rank],
    /,
    k_major_tma: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1]),
    __desc_layout: Layout = _tma_desc_tile_layout[
        dtype, rank, tile_shape, swizzle_mode
    ](),
](ctx: DeviceContext, tensor: TileTensor[dtype, ...]) -> TMATensorTile[
    dtype,
    __tile_layout,
    __desc_layout,
    is_k_major=k_major_tma,
]:
    """
    Creates a `TMATensorTile` from a TileTensor.

    This overload accepts a TileTensor instead of LayoutTensor, enabling use
    with the new coordinate-based tensor abstraction.

    Parameters:
        dtype: The data type of the tensor elements.
        rank: The dimensionality of the tensor (must be 2, 3, 4, or 5).
        tile_shape: The shape of the tile to be transferred.
        k_major_tma: Whether the TMA should use column-major pattern.
        swizzle_mode: The swizzling mode for memory access optimization.
        __tile_layout: Internal parameter for the tile layout.
        __desc_layout: Internal parameter for the descriptor layout.

    Args:
        ctx: The CUDA device context.
        tensor: The source TileTensor.

    Returns:
        A `TMATensorTile` configured for the given tensor.
    """
    comptime assert rank in (2, 3, 4, 5), "Only support 2D/3D/4D/5D TMA"

    comptime desc_bytes_size = __desc_layout.size() * size_of[dtype]()
    comptime layout_size = __tile_layout.size() * size_of[dtype]()

    comptime if desc_bytes_size < layout_size:
        comptime assert desc_bytes_size % 128 == 0, (
            "desc layout byte size has to be align to 128 bytes for"
            " multiple TMA copies."
        )

    # Swizzle constraint applies to all ranks - check once here
    comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        comptime assert (
            tile_shape[rank - 1] * size_of[dtype]()
        ) % swizzle_mode.bytes() == 0, (
            String(swizzle_mode)
            + " mode requires K dim multiple of "
            + String(swizzle_mode.bytes())
            + "B."
        )

    comptime if rank == 2:
        return create_tma_descriptor[dtype, 2, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
                1,
                owning=False,
            ),
            (
                tensor.layout.shape[0]().value(),
                tensor.layout.shape[1]().value(),
            ),
            (
                tensor.layout.stride[0]().value(),
                tensor.layout.stride[1]().value(),
            ),
            (__desc_layout.shape[0].value(), __desc_layout.shape[1].value()),
        )

    elif rank == 3:
        return create_tma_descriptor[dtype, 3, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[3](
                tensor.layout.shape[0]().value(),
                tensor.layout.shape[1]().value(),
                tensor.layout.shape[2]().value(),
            ),
            IndexList[3](
                tensor.layout.stride[0]().value(),
                tensor.layout.stride[1]().value(),
                tensor.layout.stride[2]().value(),
            ),
            IndexList[3](
                __desc_layout.shape[0].value(),
                __desc_layout.shape[1].value(),
                __desc_layout.shape[2].value(),
            ),
        )

    elif rank == 4:
        return create_tma_descriptor[dtype, 4, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[4](
                tensor.layout.shape[0]().value(),
                tensor.layout.shape[1]().value(),
                tensor.layout.shape[2]().value(),
                tensor.layout.shape[3]().value(),
            ),
            IndexList[4](
                tensor.layout.stride[0]().value(),
                tensor.layout.stride[1]().value(),
                tensor.layout.stride[2]().value(),
                tensor.layout.stride[3]().value(),
            ),
            IndexList[4](
                __desc_layout.shape[0].value(),
                __desc_layout.shape[1].value(),
                __desc_layout.shape[2].value(),
                __desc_layout.shape[3].value(),
            ),
        )

    else:  # rank == 5
        return create_tma_descriptor[dtype, 5, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[5](
                tensor.layout.shape[0]().value(),
                tensor.layout.shape[1]().value(),
                tensor.layout.shape[2]().value(),
                tensor.layout.shape[3]().value(),
                tensor.layout.shape[4]().value(),
            ),
            IndexList[5](
                tensor.layout.stride[0]().value(),
                tensor.layout.stride[1]().value(),
                tensor.layout.stride[2]().value(),
                tensor.layout.stride[3]().value(),
                tensor.layout.stride[4]().value(),
            ),
            IndexList[5](
                __desc_layout.shape[0].value(),
                __desc_layout.shape[1].value(),
                __desc_layout.shape[2].value(),
                __desc_layout.shape[3].value(),
                __desc_layout.shape[4].value(),
            ),
        )


fn _split_last_layout[
    rank: Int, //, dtype: DType
](
    tile_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
    *,
    pad: Bool,
) -> Layout:
    """
    If no padding is needed, split the last dimension so we can index
    with `0`, `1`,... instead of `0`, `swizzle_bytes()//size_of[dtype]()`,...
    """
    final_dim = tile_shape[rank - 1]
    swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()
    num_tma = ceildiv(final_dim, swizzle_granularity)
    if pad:
        var padded_shape: IndexList[rank] = {}
        for i in range(rank - 1):
            padded_shape[i] = tile_shape[i]
        padded_shape[rank - 1] = num_tma * swizzle_granularity
        return Layout.row_major(padded_shape)
    else:
        return Layout.row_major(tile_shape)


fn _ragged_fill_tile[
    rank: Int
](axis0: Int, dim0: Int, final: Int) -> IndexList[rank]:
    var desc_shape: IndexList[rank] = {}
    for i in range(rank - 1):
        if i == axis0:
            desc_shape[i] = dim0
        else:
            desc_shape[i] = 1
    desc_shape[rank - 1] = final
    return desc_shape


fn _ragged_desc_layout[
    rank: Int, //, dtype: DType
](tile_shape: IndexList[rank], swizzle_mode: TensorMapSwizzle,) -> Layout:
    swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()
    var axis0: Int = -1
    var dim0: Int = 1
    for i in range(rank - 1):
        tsi = tile_shape[i]
        if tsi != 1:
            if axis0 == -1:
                axis0 = i
                dim0 = tsi
            else:
                abort("Found multiple leading smem shapes with a non-1 axis.")

    return Layout.row_major(
        _ragged_fill_tile[rank](axis0, dim0, swizzle_granularity)
    )


comptime SplitLastDimTMATensorTile[
    rank: Int,
    //,
    dtype: DType,
    smem_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
] = TMATensorTile[
    dtype,
    _split_last_layout[dtype](smem_shape, swizzle_mode, pad=True),
    _ragged_desc_layout[dtype](smem_shape, swizzle_mode),
]
"""A specialized TMA tensor tile type alias that handles layouts where the last
dimension is split based on swizzle granularity for optimal memory access patterns.
The current behavior is to not actually split the last dimension.

Parameters:
    rank: The number of dimensions of the tensor.
    dtype: The data type of the tensor elements.
    smem_shape: The shape of the tile in shared memory. The last dimension will be
        padded if necessary to align with the swizzle granularity.
    swizzle_mode: The swizzling mode for memory access optimization. Determines
        the granularity at which the last dimension is split or padded.
"""


fn _tile_shape[smem_layout: Layout]() -> IndexList[len(smem_layout)]:
    comptime rank = len(smem_layout)
    var shape: IndexList[len(smem_layout)] = {}
    for r in range(rank):
        shape[r] = smem_layout.shape[r].value()
    return shape


@always_inline
fn _split_tma_gmem_tensor[
    dtype: DType,
    rank: Int,
    //,
    shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
](
    ptr: UnsafePointer[Scalar[dtype]],
    dim0: Int,
    out ret: LayoutTensor[
        dtype,
        _split_last_layout[dtype](shape, swizzle_mode, pad=False),
        ptr.origin,
    ],
):
    comptime split_rank = len(flatten(ret.layout.shape))
    var runtime_shape: IndexList[split_rank] = {}
    runtime_shape[0] = dim0

    comptime for i in range(1, split_rank):
        comptime dim_i: Int = ret.layout.shape[i].value()
        runtime_shape[i] = dim_i
    ret = {ptr, RuntimeLayout[ret.layout].row_major(runtime_shape)}


@always_inline
fn _split_tma_gmem_tensor[
    dtype: DType,
    rank: Int,
    //,
    shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
](
    ptr: UnsafePointer[Scalar[dtype]],
    dim0: Int,
    dim1: Int,
    out ret: LayoutTensor[
        dtype,
        _split_last_layout[dtype](shape, swizzle_mode, pad=False),
        ptr.origin,
    ],
):
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()
    var runtime_shape: IndexList[rank] = {}
    runtime_shape[0] = dim0
    runtime_shape[1] = dim1

    comptime for i in range(2, rank):
        runtime_shape[i] = shape[i]

    comptime assert rank == len(flatten(ret.layout.shape)), (
        "rank = " + String(rank) + "\nlayout = " + String(ret.layout)
    )
    ret = {ptr, RuntimeLayout[ret.layout].row_major(runtime_shape)}


fn create_split_tma[
    rank: Int,
    dtype: DType,
    //,
    smem_shape: IndexList[rank],
    gmem_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype]],
    runtime_dim0: Int,
    out res: SplitLastDimTMATensorTile[
        dtype,
        smem_shape,
        swizzle_mode,
    ],
) raises:
    """Creates a TMA tensor tile assuming that the first dimension in global memory has `UNKNOWN_VALUE`.

    This function creates a `TMATensorTile` that optionally splits the last dimension
    of the tensor into multiples of swizzle granularity. This functionality is currently
    disabled because it was not found to improve performance.

    Parameters:
        rank: The number of dimensions of the tensor.
        dtype: The data type of the tensor elements.
        smem_shape: The shape of the tile in shared memory.
        gmem_shape: The shape of the global memory tensor.
        swizzle_mode: The swizzling mode for memory access optimization.

    Args:
        ctx: The CUDA device context used to create the TMA descriptor.
        ptr: Pointer to the global memory tensor data.
        runtime_dim0: The runtime size of the first dimension of the global tensor.

    Returns:
        The resulting TMA tensor tile with split layout.

    Raises:
        If TMA descriptor creation fails.
    """
    var tensor = _split_tma_gmem_tensor[gmem_shape, swizzle_mode](
        ptr, runtime_dim0
    )
    res = create_tensor_tile[
        _tile_shape[res.layout](),
        # k_major_tma=is_k_major,
        swizzle_mode=swizzle_mode,
        __tile_layout = res.layout,
        __desc_layout = res.desc_layout,
    ](ctx, tensor)


fn create_split_tma[
    rank: Int,
    dtype: DType,
    //,
    smem_shape: IndexList[rank],
    gmem_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    runtime_dim0: Int,
    runtime_dim1: Int,
    out res: SplitLastDimTMATensorTile[
        dtype,
        smem_shape,
        swizzle_mode,
    ],
) raises:
    """Creates a TMA tensor tile assuming that the first two dimensions in global memory has `UNKNOWN_VALUE`.

    This function creates a `TMATensorTile` that optionally splits the last dimension
    of the tensor into multiples of swizzle granularity. This functionality is currently
    disabled because it was not found to improve performance.

    Parameters:
        rank: The number of dimensions of the tensor.
        dtype: The data type of the tensor elements.
        smem_shape: The shape of the tile in shared memory.
        gmem_shape: The shape of the global memory tensor.
        swizzle_mode: The swizzling mode for memory access optimization.

    Args:
        ctx: The CUDA device context used to create the TMA descriptor.
        ptr: Pointer to the global memory tensor data.
        runtime_dim0: The runtime size of the first dimension of the global tensor.
        runtime_dim1: The runtime size of the second dimension of the global tensor.

    Returns:
        The resulting TMA tensor tile with split layout.

    Raises:
        If TMA descriptor creation fails.
    """
    var tensor = _split_tma_gmem_tensor[gmem_shape, swizzle_mode](
        ptr, runtime_dim0, runtime_dim1
    )
    res = create_tensor_tile[
        _tile_shape[res.layout](),
        swizzle_mode=swizzle_mode,
        __tile_layout = res.layout,
        __desc_layout = res.desc_layout,
    ](ctx, tensor)


@always_inline
def create_tma_tile_template[
    dtype: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    /,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1]),
    __desc_layout: Layout = _tma_desc_tile_layout[
        dtype, rank, tile_shape, swizzle_mode
    ](),
]() -> TMATensorTile[dtype, __tile_layout, __desc_layout]:
    """
    Same as create_tma_tile expect the descriptor is only a placeholder or a template for later replacement.

    specification of data type, rank, and layout orientation. It supports both 2D and 3D
    tensors and provides fine-grained control over the memory access patterns.

    Parameters:
        dtype: DType
            The data type of the tensor elements.
        rank: Int
            The dimensionality of the tensor (must be 2 or 3).
        tile_shape: IndexList[rank]
            The shape of the tile to be transferred.
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE
            The swizzling mode to use for memory access optimization.
        __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1])
            Internal parameter for the tile layout in shared memory.
        __desc_layout: Layout = _tma_desc_tile_layout[...]
            Internal parameter for the descriptor layout, which may differ from the
            tile layout to accommodate hardware requirements.

    Returns:
        A `TMATensorTile` configured with the specified parameters, ready for use in
        asynchronous data transfer operations.

    Constraints:

        - Only supports 2D and 3D tensors (rank must be 2 or 3).
        - For non-SWIZZLE_NONE modes, the K dimension size in bytes must be a multiple
          of the swizzle mode's byte size.
        - For MN-major layout, only SWIZZLE_128B is supported.
        - For 3D tensors, only K-major layout is supported.
    """

    return TMATensorTile[dtype, __tile_layout, __desc_layout](TMADescriptor())


struct TMATensorTileArray[
    num_of_tensormaps: Int,
    dtype: DType,
    cta_tile_layout: Layout,
    desc_layout: Layout,
](DevicePassable, TrivialRegisterPassable):
    """An array of TMA descripotr.

    Parameters:
        num_of_tensormaps: Int
            The number of TMA descriptors aka tensor map.
        dtype: DType
            The data type of the tensor elements.
        cta_tile_layout: Layout
            The layout of the tile in shared memory, typically specified as row_major.
        desc_layout: Layout
            The layout of the descriptor, which can be different from the shared memory layout
            to accommodate hardware requirements like WGMMA.
    """

    var tensormaps_ptr: UnsafePointer[UInt8, MutAnyOrigin]
    """A static tuple of pointers to TMA descriptors.

    This field stores an array of pointers to `TMATensorTile` instances, where each pointer
    references a TMA descriptor in device memory. The array has a fixed size determined by
    the num_of_tensormaps parameter.

    The TMA descriptors are used by the GPU hardware to efficiently transfer data between
    global and shared memory with specific memory access patterns defined by the layouts.
    """

    comptime descriptor_bytes = 128
    """Size of the TMA descriptor in bytes.

    This is a constant value that represents the size of the TMA descriptor in bytes.
    It is used to calculate the offset of the TMA descriptor in the device memory.
    """

    comptime device_type: AnyType = Self
    """The device-side type representation."""

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.

        Returns:
            This type's name.
        """
        return String(
            "TMATensorTileArray[num_of_tensormaps = ",
            Self.num_of_tensormaps,
            ", dtype = ",
            Self.dtype,
            ", cta_tile_layout = ",
            materialize[Self.cta_tile_layout](),
            ", desc_layout = ",
            materialize[Self.desc_layout](),
            "]",
        )

    @always_inline
    fn __init__(
        out self,
        tensormaps_device: DeviceBuffer[DType.uint8],
    ):
        """
        Initializes a new TMATensorTileArray.

        Args:
            tensormaps_device: Device buffer to store TMA descriptors.
        """

        self.tensormaps_ptr = tensormaps_device.unsafe_ptr()

    @always_inline
    fn __getitem__(
        self, index: Int
    ) -> UnsafePointer[
        TMATensorTile[Self.dtype, Self.cta_tile_layout, Self.desc_layout],
        MutAnyOrigin,
    ]:
        """
        Retrieve a TMA descriptor.

        Args:
            index: Index of the TMA descriptor.

        Returns:
            `UnsafePointer` to the `TMATensorTile` at the specified index.
        """
        return (self.tensormaps_ptr + index * self.descriptor_bytes).bitcast[
            TMATensorTile[Self.dtype, Self.cta_tile_layout, Self.desc_layout]
        ]()


struct RaggedTMA3DTile[
    dtype: DType, swizzle_mode: TensorMapSwizzle, BM: Int, BN: Int
](DevicePassable, ImplicitlyCopyable):
    """
    Creates a TMA descriptor for loading/storing from ragged 3D arrays with a
    ragged leading dimension. This loads 2D tiles, indexing into the middle dim.
    When using this loads, it is essential that at least `BM * stride` space
    has been allocated in front of the gmem pointer, otherwise
    `CUDA_ERROR_ILLEGAL_ADDRESS` may result.

    Parameters:
        dtype: The data type of the tensor.
        swizzle_mode: The swizzling mode to use for memory access.
        BM: The number of rows of the corresponding 2D shared memory tile.
        BN: The number of columns of the corresponding 2D shared memory tile.
    """

    var descriptor: TMADescriptor
    """The TMA descriptor that will be used to store the ragged tensor."""

    comptime device_type: AnyType = Self
    """The device-side type representation."""

    comptime swizzle_granularity = Self.swizzle_mode.bytes() // size_of[
        Self.dtype
    ]()
    """The number of columns that must be copied at a time due to the swizzle size."""

    comptime layout: Layout = tile_layout_k_major[
        Self.dtype, Self.BM, Self.BN, Self.swizzle_mode
    ]()
    """The unswizzled-smem layout copied to/from by this tma op."""

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Returns a string representation of the RaggedTMA3DTile type.

        Returns:
            A string containing the type name with all template parameters.
        """
        return String(
            "RaggedTMA3DTile[dtype = ",
            Self.dtype,
            ", BM = ",
            Self.BM,
            ", BN = ",
            Self.BN,
            ", swizzle_mode = ",
            Self.swizzle_mode,
        )

    @always_inline
    @implicit
    fn __init__(out self, descriptor: TMADescriptor):
        """
        Initializes a new TMATensorTile with the provided TMA descriptor.

        Args:
            descriptor: The TMA descriptor that defines the memory access pattern.
        """
        self.descriptor = descriptor

    @staticmethod
    @always_inline
    fn create[
        *,
        depth: Int = Self.BN,
    ](
        ctx: DeviceContext,
        ptr: UnsafePointer[Scalar[Self.dtype]],
        *,
        rows: Int,
        middle_dim: Int,
    ) raises -> Self:
        """
        Create a RaggedTMA3DTile.

        Parameters:
            depth: The size of the inner-most, contiguous, dimension.

        Args:
            ctx: The device context used to create the TMA descriptors.
            ptr: The global memory pointer.
            rows: The size of the ragged dimension.
            middle_dim: The size of the middle dimension.

        Returns:
            A RaggedTMA3DTile corresponding to the gmem.

        Raises:
            If TMA descriptor creation fails.
        """
        stride = middle_dim * depth
        return create_tma_descriptor[Self.dtype, 4, Self.swizzle_mode](
            DeviceBuffer(
                ctx,
                ptr - stride * Self.BM,
                1,
                owning=False,
            ),
            IndexList[4](rows + 1, middle_dim, Self.BM, depth),
            IndexList[4](stride, depth, stride, 1),
            IndexList[4](1, 1, Self.BM, Self.swizzle_granularity),
        )

    @always_inline
    fn __init__(out self, *, copy: Self):
        """
        Copy initializes this `RaggedTMA3DTile` from another instance.

        Args:
            copy: The other `RaggedTMA3DTile` instance to copy from.
        """
        self.descriptor = copy.descriptor

    @always_inline("nodebug")
    fn async_copy_to[
        cta_group: Int = 1
    ](
        self,
        dst: UnsafePointer[
            Scalar[Self.dtype], address_space = AddressSpace.SHARED
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        *,
        ragged_idx: UInt32,
        dynamic_dim: UInt32,
        middle_idx: UInt32,
    ):
        """
        Copy from the `RaggedTMA3DTile` source to the smem destination.

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the leader CTA needs
                       to be notified upon completion.

        Args:
            dst: The destination shared memory pointer to which we copy memory.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            ragged_idx: Index into the ragged dimension.
            dynamic_dim: Number of rows to copy.
            middle_idx: Index into the middle (generally head) dimension.

        """

        var offset_ragged_idx: UInt = UInt(ragged_idx + dynamic_dim)
        var box_idx: UInt = UInt(UInt32(Self.BM) - dynamic_dim)

        comptime for col in range(ceildiv(Self.BN, Self.swizzle_granularity)):
            comptime copy_offset = col * Self.BM * Self.swizzle_granularity

            cp_async_bulk_tensor_shared_cluster_global[cta_group=cta_group](
                dst.mut_cast[True]() + copy_offset,
                UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                mem_barrier.unsafe_ptr(),
                Index(
                    UInt(col * Self.swizzle_granularity),
                    box_idx,
                    UInt(middle_idx),
                    offset_ragged_idx,
                ),
            )

    @always_inline
    fn async_copy_from_col[
        col: Int,
        eviction_policy: CacheEviction = CacheEviction.EVICT_FIRST,
    ](
        self,
        src: UnsafePointer[
            Scalar[Self.dtype], address_space = AddressSpace.SHARED
        ],
        *,
        ragged_idx: UInt32,
        dynamic_dim: UInt32,
        middle_idx: UInt32,
    ):
        """Copy a single swizzle_granularity-wide column chunk from smem to
        gmem.

        Parameters:
            col: Which column chunk (0-indexed, each chunk is
                swizzle_granularity columns).
            eviction_policy: Optional cache eviction policy that controls how
                the data is handled in the cache hierarchy. Defaults to
                EVICT_FIRST.

        Args:
            src: Source shared memory pointer (base of the full tile).
            ragged_idx: Index into the ragged dimension.
            dynamic_dim: Number of rows to copy.
            middle_idx: Index into the middle (generally head) dimension.
        """
        var offset_ragged_idx: UInt = UInt(ragged_idx + dynamic_dim)
        var box_idx: UInt = UInt(UInt32(Self.BM) - dynamic_dim)
        comptime copy_offset = col * Self.BM * Self.swizzle_granularity

        cp_async_bulk_tensor_global_shared_cta[eviction_policy=eviction_policy](
            src + copy_offset,
            UnsafePointer(to=self.descriptor).bitcast[NoneType](),
            Index(
                UInt(col * Self.swizzle_granularity),
                box_idx,
                UInt(middle_idx),
                offset_ragged_idx,
            ),
        )

    @always_inline
    fn async_copy_from[
        eviction_policy: CacheEviction = CacheEviction.EVICT_FIRST,
    ](
        self,
        src: UnsafePointer[
            Scalar[Self.dtype], address_space = AddressSpace.SHARED
        ],
        *,
        ragged_idx: UInt32,
        dynamic_dim: UInt32,
        middle_idx: UInt32,
    ):
        """
        Copy from the smem source to the `RaggedTMA3DTile` destination.

        Args:
            src: The source shared memory pointer from which we copy memory.
            ragged_idx: Index into the ragged dimension.
            dynamic_dim: Number of rows to copy.
            middle_idx: Index into the middle (generally head) dimension.

        Parameters:
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_FIRST.
        """

        comptime for col in range(ceildiv(Self.BN, Self.swizzle_granularity)):
            self.async_copy_from_col[col, eviction_policy](
                src,
                ragged_idx=ragged_idx,
                dynamic_dim=dynamic_dim,
                middle_idx=middle_idx,
            )

    @always_inline
    fn prefetch_descriptor(self):
        """
        Prefetches the TMA descriptor into cache.
        """

        prefetch_tma_descriptor(
            UnsafePointer(to=self.descriptor).bitcast[NoneType]()
        )


struct RaggedTensorMap[
    descriptor_rank: Int,
    //,
    dtype: DType,
    descriptor_shape: IndexList[descriptor_rank],
    remaining_global_dim_rank: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](DevicePassable, ImplicitlyCopyable):

    """
    Creates a TMA descriptor that can handle stores with varying lengths. This struct is mainly used
    for MHA, where sequence lengths may vary between sample.

    This struct only supports one dimension being ragged. The continous dimension (where stride is 1) cannot be ragged.

    Parameters:
        descriptor_rank:
            The rank of the descriptor shape (inferred).
        dtype:
            The data type of the tensor.
        descriptor_shape:
            The shape of the shared memory descriptor.
        remaining_global_dim_rank:
            The rank of the remaining global tensor dimensions.
        swizzle_mode:
            The swizzling mode to use for memory access optimization. Swizzling can improve
            memory access patterns for specific hardware configurations. Defaults to SWIZZLE_NONE.

    """

    var descriptor: TMADescriptor
    """The TMA descriptor that will be used to store the ragged tensor."""
    var max_length: Int
    """The maximum length present in the sequences of the ragged tensor."""
    var global_shape: IndexList[Self.global_rank]
    """The shape of the global tensor."""
    var global_stride: IndexList[Self.global_rank]
    """The stride of the global tensor."""

    comptime global_rank = Self.remaining_global_dim_rank + 3
    """The rank of the global tensor."""

    @staticmethod
    fn _descriptor_shape() -> IndexList[Self.descriptor_rank + 1]:
        """
        Constructs a descriptor shape that can handle one ragged dimension for loads.

        Returns:
            A descriptor shape.
        """

        var idx_list = IndexList[Self.descriptor_rank + 1](fill=0)
        idx_list[0] = 1

        comptime for idx in range(Self.descriptor_rank):
            idx_list[idx + 1] = Self.descriptor_shape[idx]

        return idx_list

    @staticmethod
    @always_inline
    fn _get_layout() -> Layout:
        var layout = Layout(
            IntTuple(num_elems=Self.global_rank),
            IntTuple(num_elems=Self.global_rank),
        )

        comptime for idx in range(Self.global_rank):
            layout.shape.replace_entry(idx, int_value=UNKNOWN_VALUE)
            layout.stride.replace_entry(idx, int_value=UNKNOWN_VALUE)

        return layout^

    comptime device_type: AnyType = Self
    """The TensorMapDescriptorArray type."""

    comptime ragged_descriptor_shape = Self._descriptor_shape()
    """The shape of the descriptor that will tile and load from shared -> global memory."""

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """
        Copies this descriptor array to device memory.

        Args:
            target: Opaque pointer to the target device memory location.
        """
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Returns a string representation of the TensorMapDescriptorArray type.

        Returns:
            A string containing the type name with all template parameters.
        """
        return String(
            "RaggedTensorMap[rank = ",
            Self.descriptor_rank,
            ", dtype = ",
            Self.dtype,
            ", descriptor_shape = ",
            Self.ragged_descriptor_shape,
            ", swizzle_mode = ",
            Self.swizzle_mode,
            ", max_descriptor_length = ",
            "]",
        )

    @staticmethod
    @always_inline
    fn _create_global_stride(
        ragged_stride: Int,
        remaining_global_stride: IndexList[Self.remaining_global_dim_rank],
    ) -> IndexList[Self.global_rank]:
        var global_stride = IndexList[Self.global_rank](fill=0)
        global_stride[0] = ragged_stride
        global_stride[Self.global_rank - 2] = ragged_stride
        global_stride[Self.global_rank - 1] = 1

        comptime for idx in range(1, 1 + Self.remaining_global_dim_rank):
            global_stride[idx] = remaining_global_stride[idx - 1]

        return global_stride

    @staticmethod
    @always_inline
    fn _create_global_shape(
        cumulative_length: Int,
        max_length: Int,
        global_last_dim: Int,
        remaining_global_shape: IndexList[Self.remaining_global_dim_rank],
    ) -> IndexList[Self.global_rank]:
        var global_shape = IndexList[Self.global_rank](fill=0)
        global_shape[0] = cumulative_length

        comptime for idx in range(1, 1 + Self.remaining_global_dim_rank):
            global_shape[idx] = remaining_global_shape[idx - 1]

        global_shape[Self.global_rank - 2] = max_length
        global_shape[Self.global_rank - 1] = global_last_dim

        return global_shape

    fn __init__(
        out self,
        ctx: DeviceContext,
        global_ptr: UnsafePointer[Scalar[Self.dtype]],
        max_length: Int,
        ragged_stride: Int,
        batch_size: Int,
        global_last_dim: Int,
        remaining_global_dims: IndexList[Self.remaining_global_dim_rank],
        remaining_global_stride: IndexList[Self.remaining_global_dim_rank],
    ) raises:
        """
        Initializes a TensorMapDescriptorArray with descriptors for all power-of-2 lengths.

        This constructor creates a complete set of TMA descriptors, one for each power of 2
        from 1 up to max_descriptor_length. Each descriptor is configured to handle a different
        first dimension size (1, 2, 4, 8, ..., max_descriptor_length) while maintaining the
        same remaining tile shape specified by desc_remaining_tile_shape.

        Raises:
            If the operation fails.

        Args:
            ctx:
                The device context used to create the TMA descriptors.
            global_ptr:
                The source tensor in global memory that will be accessed using the descriptors.
            max_length:
                The maximum length present in the sequences of the ragged tensor.
            ragged_stride:
                The stride of the ragged dimension in the global tensor.
            batch_size:
                The total number of sequences in the ragged tensor.
            global_last_dim:
                The last dimension of the global tensor.
            remaining_global_dims:
                The dimensions of the remaining global tensor.
            remaining_global_stride:
                The stride of the remaining global tensor.
        Constraints:
            - max_descriptor_length must be a power of two.
            - max_descriptor_length must be less than or equal to 256.
        """

        comptime assert (
            Self.global_rank >= 2
        ), "global_rank must be at least 2 with one ragged dimension"

        var cumulative_length = (batch_size + 1) * max_length

        var global_shape = Self._create_global_shape(
            cumulative_length,
            max_length,
            global_last_dim,
            remaining_global_dims,
        )

        var global_stride = Self._create_global_stride(
            ragged_stride, remaining_global_stride
        )

        comptime global_layout = Self._get_layout()

        var global_runtime_layout = RuntimeLayout[global_layout](
            global_shape, global_stride
        )

        comptime GlobalTensorType = LayoutTensor[
            Self.dtype,
            global_layout,
            MutAnyOrigin,
        ]

        var decremented_ptr = global_ptr - (ragged_stride * max_length)
        var global_tensor = GlobalTensorType(
            decremented_ptr.unsafe_mut_cast[True]().unsafe_origin_cast[
                MutAnyOrigin
            ](),
            global_runtime_layout,
        )

        self.descriptor = _create_tma_descriptor_helper[
            Self.ragged_descriptor_shape, Self.swizzle_mode
        ](
            ctx,
            global_tensor,
        )

        self.max_length = max_length
        self.global_shape = global_shape
        self.global_stride = global_stride

    @always_inline
    fn _get_descriptor_ptr(self) -> UnsafePointer[NoneType, MutAnyOrigin]:
        return (
            UnsafePointer(to=self.descriptor)
            .bitcast[NoneType]()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

    @always_inline
    fn store_ragged_tile[
        rank: Int,
        //,
        using_max_descriptor_size: Bool = False,
    ](
        self,
        coordinates: IndexList[rank],
        preceding_cumulative_length: Int,
        store_length: Int,
        mut tile_iterator: LayoutTensorIter[
            Self.dtype,
            _,
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
            ...,
        ],
    ):
        """
        Stores a ragged tile from shared memory to global memory.

        Parameters:
            rank:
                The rank of the coordinates.
            using_max_descriptor_size:
                If True, optimizes the store around the max descriptor size.

        Args:
            coordinates:
                The starting coordinates of all dimensions except the ragged dimension.
            preceding_cumulative_length:
                The cumulative length of the preceding sequences.
            store_length:
                The length of the current sequence to be stored.
            tile_iterator:
                The iterator over the tile in shared memory.
        """

        comptime assert rank == Self.global_rank

        # Assume we have the folowing ragged tensor:

        # It has 16 heads, head depth of 128, and 4 sequences of length
        # [43, 32, 10, 64]

        # The overall shape will look like this with ? representing the 4 sequences:
        # [?, 16, 128]

        # When creating the TMA descriptor you pass in several values: max_length, ragged_stride,
        # batch_size, global_last_dim, remaining_global_dims, remaining_global_stride

        # In our case:

        # max_length = 64 (the max length of the sequences)
        # ragged_stride = 2048 (heads x head depth)
        # batch_size = 4 (the number of sequence batches)
        # global_last_dim = 128 (the last dimension of the global tensor, the head depth)
        # remaining_global_dims = [16] (the only value not supplied, the head dimension)
        # remaining_global_stride = [128] (the stride of the head dimension)

        # We also compute values such as the cummulative length using this formula:
        # cummulative_length = (batch_size + 1) * max_length = (4 + 1) * 64 = 320

        # With these values we create our descriptor with an artificial layout of:

        # (cumulative_length, remaining_global_dims..., max_length, global_last_dim) : (ragged_stride, remaining_global_stride..., max_length, global_last_dim)
        # (320, 16, 64, 128) : (2048, 128, 2048, 1)

        # (internally this layout gets reversed when passed into the descriptor)

        # Now lets say we have a descriptor of shape (1, 1, 24, 64), the 24 tells us that we
        # want to store 24 sequences at once and 64 tells us we want to store half the depth.

        # Now lets say we want to store the first depth chunk (64) of the first batch (43) at head 7.

        # We would need to do a total of 2 stores, with the global coordinates naively being:
        # [(0, 7, 0, 0), (24, 7, 0, 0)] || [(0, 7, 0, 0), (0, 7, 24, 0)]

        # Both cases will cause spillage since 24 * 2 = 48.

        # Instead we will utilize the cumulative_length dimension and max_length dimension to mask the out of bounds
        # segments in each ragged store.

        # One prerequsite for this to work is that the starting pointer must be negatively offset by ragged_stride * max_length.
        # Which in our case is 2048 * 64 or 64 sequences.

        # Now to get bounds checked store we set the
        # cumulative_length dimension to the cumulative length of the preceding sequences + this sequence's length.
        # And we set the max_length dimension to the max_length - this sequence's length.

        # This would make our new coordinate starting global coordinates: [(43, 7, 21, 0), (43, 7, 45, 0)]

        # When adding 43 + 21 we get a starting offset of 64, which is how much we offset our orignal pointer. This gives
        # us the correct starting offset for our store. Finally our max_length dimension is set to start at 21. It is hardbounded
        # by 64 (the max length) so this ensure that anything we load past 64 will be masked out. So when we end at 68 for the second store,
        # the last 5 sequences will be masked out.

        # Now lets say we want to try and store the second sequence (32)

        # Our new coordinates would be: [(75, 7, 32, 0), (75, 7, 56, 0)]

        # starting us at (75 + 32) - 64 = 43, and allowing us to only load 32 sequences

        comptime if using_max_descriptor_size:
            # if the max length is the same as the descriptor size we dont need to do
            # multiple stores and generate multiple coords so we can avoid unnessecary
            # branching in this case.
            var cummulative_length = preceding_cumulative_length + store_length

            var adjusted_coordinates = coordinates
            adjusted_coordinates[Self.global_rank - 1] = cummulative_length
            adjusted_coordinates[1] = self.max_length - store_length

            cp_async_bulk_tensor_global_shared_cta(
                tile_iterator[].ptr,
                self._get_descriptor_ptr(),
                adjusted_coordinates,
            )
        else:
            comptime descriptor_load_length = Self.ragged_descriptor_shape[
                Self.global_rank - 2
            ]

            var descriptor_iters = ceildiv(store_length, descriptor_load_length)

            var cummulative_length = preceding_cumulative_length + store_length

            var adjusted_coordinates = coordinates
            adjusted_coordinates[Self.global_rank - 1] = cummulative_length

            for i in range(descriptor_iters):
                var max_length_offset = (
                    self.max_length
                    - store_length
                    + (i * descriptor_load_length)
                )
                adjusted_coordinates[1] = max_length_offset

                cp_async_bulk_tensor_global_shared_cta(
                    tile_iterator[].ptr,
                    self._get_descriptor_ptr(),
                    adjusted_coordinates,
                )

                tile_iterator._incr()

    @always_inline
    fn prefetch_descriptor(self):
        """
        Prefetches the TMA descriptor into cache.
        """

        prefetch_tma_descriptor(self._get_descriptor_ptr())


struct TMATensorTileIm2col[
    dtype: DType,
    layout: Layout,
    desc_layout: Layout = layout,
](DevicePassable, ImplicitlyCopyable):
    """TMA tensor tile with im2col coordinate transformation for convolution.

    This struct enables hardware-accelerated im2col transformation during TMA loads,
    used for implicit GEMM convolution. The TMA descriptor encodes the convolution
    geometry (padding, stride, dilation) and performs coordinate transformation
    on-the-fly.

    The coordinate system uses GEMM-style 2D coordinates:
    - coords[0]: K coordinate (indexes into R * S * C reduction dimension)
    - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

    Internally:
    - K is decomposed into (c, r, s) where K = r*S*C + s*C + c (filter-first, channel-last for NHWC)
    - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
    - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to the
      PTX im2col instruction.

    Parameters:
        dtype: The data type of tensor elements.
        layout: The layout of the tile in shared memory.
        desc_layout: The layout of the descriptor (may differ for WGMMA compatibility).
    """

    var descriptor: TMADescriptor
    """The TMA descriptor encoding im2col transformation parameters."""

    var out_height: UInt32
    """Output height (H_out) for M coordinate decomposition."""

    var out_width: UInt32
    """Output width (W_out) for M coordinate decomposition."""

    var filter_h: UInt32
    """Filter height (R) for K coordinate decomposition."""

    var filter_w: UInt32
    """Filter width (S) for K coordinate decomposition."""

    var in_channels: UInt32
    """Input channels (C) for K coordinate decomposition."""

    var lower_corner_h: Int32
    """Lower corner offset for height (H dimension) - matches CUTLASS ArithmeticTupleIterator pattern."""

    var lower_corner_w: Int32
    """Lower corner offset for width (W dimension) - matches CUTLASS ArithmeticTupleIterator pattern."""

    comptime device_type: AnyType = Self
    """The device-side type representation."""

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """Gets this type's name for error messages.

        Returns:
            This type's name.
        """
        return String(
            "TMATensorTileIm2col[dtype = ",
            Self.dtype,
            ", layout = ",
            materialize[Self.layout](),
            ", desc_layout = ",
            materialize[Self.desc_layout](),
            "]",
        )

    @always_inline
    fn __init__(
        out self,
        descriptor: TMADescriptor,
        out_height: UInt32,
        out_width: UInt32,
        filter_h: UInt32,
        filter_w: UInt32,
        in_channels: UInt32,
        lower_corner_h: Int32 = 0,
        lower_corner_w: Int32 = 0,
    ):
        """Initializes with the provided TMA im2col descriptor and dimensions.

        Args:
            descriptor: The TMA descriptor that encodes im2col transformation.
            out_height: Output height (H_out) for M coordinate decomposition.
            out_width: Output width (W_out) for M coordinate decomposition.
            filter_h: Filter height (R) for K coordinate decomposition.
            filter_w: Filter width (S) for K coordinate decomposition.
            in_channels: Input channels (C) for K coordinate decomposition.
            lower_corner_h: Lower corner offset for H dimension (matches CUTLASS pattern).
            lower_corner_w: Lower corner offset for W dimension (matches CUTLASS pattern).
        """
        self.descriptor = descriptor
        self.out_height = out_height
        self.out_width = out_width
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.in_channels = in_channels
        self.lower_corner_h = lower_corner_h
        self.lower_corner_w = lower_corner_w

    @always_inline
    fn __init__(out self, *, copy: Self):
        """Copy initializes from another instance.

        Args:
            copy: The other instance to copy from.
        """
        self.descriptor = copy.descriptor
        self.out_height = copy.out_height
        self.out_width = copy.out_width
        self.filter_h = copy.filter_h
        self.filter_w = copy.filter_w
        self.in_channels = copy.in_channels
        self.lower_corner_h = copy.lower_corner_h
        self.lower_corner_w = copy.lower_corner_w

    @always_inline
    fn prefetch_descriptor(self):
        """Prefetches the TMA descriptor into cache."""
        var desc_ptr = UnsafePointer(to=self.descriptor).bitcast[NoneType]()
        prefetch_tma_descriptor(desc_ptr)

    @always_inline
    fn async_copy[
        cta_group: Int = 1,  # Use SM90-style TMA for cluster 1x1x1
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt],
    ):
        """Schedules an asynchronous im2col TMA load.

        Uses 2D GEMM-style coordinates:
        - coords[0]: K coordinate (indexes into C * R * S reduction dimension)
        - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

        Internally:
        - K is decomposed into (c, r, s) where K = c*R*S + r*S + s
        - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
        - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to
          the PTX im2col instruction.

        Note: The cta_group parameter defaults to 2 because SM100/Blackwell
        im2col TMA with padding (negative corners) requires the cta_group::2
        PTX format. This is consistent with CUTLASS which only provides
        SM100_TMA_2SM_LOAD_IM2COL (no cta_group::1 variant for im2col).

        Parameters:
            cta_group: CTA group size for TMA operations.
            eviction_policy: Cache eviction policy for the TMA load.

        Args:
            dst: Destination tensor in shared memory.
            mem_barrier: Memory barrier for synchronization.
            coords: GEMM coordinates (k_coord, m_coord).
        """
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = Self.layout.shape[0].value() // copy_dim0
        comptime num_copies_dim1 = Self.layout.shape[1].value() // copy_dim1

        # Precompute spatial size for M decomposition
        var hw = UInt(self.out_height) * UInt(self.out_width)
        var out_w = UInt(self.out_width)

        # Precompute filter window size for K decomposition
        # K = r * S * C + s * C + c (filter-first, channel-last ordering for NHWC)
        var num_channels = UInt(self.in_channels)
        var filter_w = UInt(self.filter_w)

        # OPTIMIZATION: Hoist K decomposition outside loop (constant when j=0).
        # For typical configs (num_copies_dim1=1), K coords don't change within tile.
        var k_coord = coords[0]
        var filter_idx, c = divmod(k_coord, num_channels)
        var r, s = divmod(filter_idx, filter_w)

        # Initial M decomposition (done once, then use iterator)
        var m_coord_init = coords[1]
        var n, m_remainder = divmod(m_coord_init, hw)
        var h_out, w_out = divmod(m_remainder, out_w)

        # Pre-add lower_corner offset
        var h = Int(h_out) + Int(self.lower_corner_h)
        var w = Int(w_out) + Int(self.lower_corner_w)

        # Cache bounds for iterator wraparound
        var out_w_int = Int(out_w)
        var out_h_int = Int(self.out_height)
        var lower_h = Int(self.lower_corner_h)
        var lower_w = Int(self.lower_corner_w)

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                # K recomputation only needed when j > 0 (rare in practice)
                comptime if j > 0:
                    k_coord = coords[0] + UInt(j * copy_dim1)
                    filter_idx, c = divmod(k_coord, num_channels)
                    r, s = divmod(filter_idx, filter_w)

                # Pass 4D coords (c, w, h, n) and filter offsets (s, r) to im2col PTX
                cp_async_bulk_tensor_shared_cluster_global_im2col[
                    cta_group=cta_group,
                ](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(Int(c), w, h, Int(n)),
                    Index(Int(s), Int(r)),
                )

            # Iterator pattern: advance M by copy_dim0 using addition (not division)
            # This avoids 4 divisions per sub-tile, reducing from O(n*8) to O(8+n*3)
            w += copy_dim0
            if w >= out_w_int + lower_w:
                w -= out_w_int
                h += 1
                if h >= out_h_int + lower_h:
                    h -= out_h_int
                    n += 1

    @always_inline
    fn async_multicast_load[
        cta_group: Int = 1,  # Use SM90-style TMA for cluster 1x1x1
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, ...
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt],
        multicast_mask: UInt16,
    ):
        """Schedules an asynchronous im2col TMA load with multicast.

        Uses 2D GEMM-style coordinates:
        - coords[0]: K coordinate (indexes into C * R * S reduction dimension)
        - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

        Internally:
        - K is decomposed into (c, r, s) where K = c*R*S + r*S + s
        - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
        - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to
          the PTX im2col instruction with multicast.

        Note: The cta_group parameter defaults to 2 because SM100/Blackwell
        im2col TMA with padding (negative corners) requires the cta_group::2
        PTX format. This is consistent with CUTLASS which only provides
        SM100_TMA_2SM_LOAD_IM2COL_MULTICAST (no cta_group::1 variant).

        Parameters:
            cta_group: CTA group size for TMA operations.
            eviction_policy: Cache eviction policy for the TMA load.

        Args:
            dst: Destination tensor in shared memory.
            mem_barrier: Memory barrier for synchronization.
            coords: GEMM coordinates (k_coord, m_coord).
            multicast_mask: Bitmask specifying target CTAs for multicast.
        """
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = Self.layout.shape[0].value() // copy_dim0
        comptime num_copies_dim1 = Self.layout.shape[1].value() // copy_dim1

        # Precompute spatial size for M decomposition
        var hw = UInt(self.out_height) * UInt(self.out_width)
        var out_w = UInt(self.out_width)

        # Precompute filter window size for K decomposition
        # K = r * S * C + s * C + c (filter-first, channel-last ordering for NHWC)
        var num_channels = UInt(self.in_channels)
        var filter_w = UInt(self.filter_w)

        # OPTIMIZATION: Hoist K decomposition outside loop (constant when j=0).
        var k_coord = coords[0]
        var filter_idx, c = divmod(k_coord, num_channels)
        var r, s = divmod(filter_idx, filter_w)

        # Initial M decomposition (done once, then use iterator)
        var m_coord_init = coords[1]
        var n, m_remainder = divmod(m_coord_init, hw)
        var h_out, w_out = divmod(m_remainder, out_w)

        # Pre-add lower_corner offset
        var h = Int(h_out) + Int(self.lower_corner_h)
        var w = Int(w_out) + Int(self.lower_corner_w)

        # Cache bounds for iterator wraparound
        var out_w_int = Int(out_w)
        var out_h_int = Int(self.out_height)
        var lower_h = Int(self.lower_corner_h)
        var lower_w = Int(self.lower_corner_w)

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                # K recomputation only needed when j > 0
                comptime if j > 0:
                    k_coord = coords[0] + UInt(j * copy_dim1)
                    filter_idx, c = divmod(k_coord, num_channels)
                    r, s = divmod(filter_idx, filter_w)

                # Pass 4D coords (c, w, h, n) and filter offsets (s, r) to im2col PTX
                cp_async_bulk_tensor_shared_cluster_global_im2col_multicast[
                    cta_group=cta_group,
                ](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(Int(c), w, h, Int(n)),
                    Index(Int(s), Int(r)),
                    multicast_mask,
                )

            # Iterator pattern: advance M by copy_dim0 using addition
            w += copy_dim0
            if w >= out_w_int + lower_w:
                w -= out_w_int
                h += 1
                if h >= out_h_int + lower_h:
                    h -= out_h_int
                    n += 1

    @always_inline
    fn async_copy[
        cta_group: Int = 1,  # Use SM90-style TMA for cluster 1x1x1
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[
            mut=True,
            dtype = Self.dtype,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt],
    ):
        """Schedules an asynchronous im2col TMA load.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Uses 2D GEMM-style coordinates:
        - coords[0]: K coordinate (indexes into C * R * S reduction dimension)
        - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

        Internally:
        - K is decomposed into (c, r, s) where K = c*R*S + r*S + s
        - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
        - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to
          the PTX im2col instruction.

        Note: Uses cta_group=1 (SM90-style TMA) for single-CTA clusters.

        Parameters:
            cta_group: CTA group size for TMA operations.
            eviction_policy: Cache eviction policy for the TMA load.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: Memory barrier for synchronization.
            coords: GEMM coordinates (k_coord, m_coord).
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = Self.layout.shape[0].value() // copy_dim0
        comptime num_copies_dim1 = Self.layout.shape[1].value() // copy_dim1

        # Precompute spatial size for M decomposition
        var hw = UInt(self.out_height) * UInt(self.out_width)
        var out_w = UInt(self.out_width)

        # Precompute filter window size for K decomposition
        # K = r * S * C + s * C + c (filter-first, channel-last ordering for NHWC)
        var num_channels = UInt(self.in_channels)
        var filter_w = UInt(self.filter_w)

        # OPTIMIZATION: Hoist K decomposition outside loop (constant when j=0).
        var k_coord = coords[0]
        var filter_idx, c = divmod(k_coord, num_channels)
        var r, s = divmod(filter_idx, filter_w)

        # Initial M decomposition (done once, then use iterator)
        var m_coord_init = coords[1]
        var n, m_remainder = divmod(m_coord_init, hw)
        var h_out, w_out = divmod(m_remainder, out_w)

        # Pre-add lower_corner offset
        var h = Int(h_out) + Int(self.lower_corner_h)
        var w = Int(w_out) + Int(self.lower_corner_w)

        # Cache bounds for iterator wraparound
        var out_w_int = Int(out_w)
        var out_h_int = Int(self.out_height)
        var lower_h = Int(self.lower_corner_h)
        var lower_w = Int(self.lower_corner_w)

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                # K recomputation only needed when j > 0
                comptime if j > 0:
                    k_coord = coords[0] + UInt(j * copy_dim1)
                    filter_idx, c = divmod(k_coord, num_channels)
                    r, s = divmod(filter_idx, filter_w)

                # Pass 4D coords (c, w, h, n) and filter offsets (s, r) to im2col PTX
                cp_async_bulk_tensor_shared_cluster_global_im2col[
                    cta_group=cta_group,
                ](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(Int(c), w, h, Int(n)),
                    Index(Int(s), Int(r)),
                )

            # Iterator pattern: advance M by copy_dim0 using addition
            w += copy_dim0
            if w >= out_w_int + lower_w:
                w -= out_w_int
                h += 1
                if h >= out_h_int + lower_h:
                    h -= out_h_int
                    n += 1

    @always_inline
    fn async_multicast_load[
        cta_group: Int = 1,  # Use SM90-style TMA for cluster 1x1x1
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[
            mut=True,
            dtype = Self.dtype,
            address_space = AddressSpace.SHARED,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt],
        multicast_mask: UInt16,
    ):
        """Schedules an asynchronous im2col TMA load with multicast.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Uses 2D GEMM-style coordinates:
        - coords[0]: K coordinate (indexes into C * R * S reduction dimension)
        - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

        Internally:
        - K is decomposed into (c, r, s) where K = c*R*S + r*S + s
        - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
        - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to
          the PTX im2col instruction with multicast.

        Note: Uses cta_group=1 (SM90-style TMA) for single-CTA clusters.

        Parameters:
            cta_group: CTA group size for TMA operations.
            eviction_policy: Cache eviction policy for the TMA load.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: Memory barrier for synchronization.
            coords: GEMM coordinates (k_coord, m_coord).
            multicast_mask: Bitmask specifying target CTAs for multicast.
        """
        comptime copy_dim0 = Self.desc_layout.shape[0].value()
        comptime copy_dim1 = Self.desc_layout.shape[1].value()
        comptime copy_size = Self.desc_layout.size()
        comptime num_copies_dim0 = Self.layout.shape[0].value() // copy_dim0
        comptime num_copies_dim1 = Self.layout.shape[1].value() // copy_dim1

        # Precompute spatial size for M decomposition
        var hw = UInt(self.out_height) * UInt(self.out_width)
        var out_w = UInt(self.out_width)

        # Precompute filter window size for K decomposition
        # K = r * S * C + s * C + c (filter-first, channel-last ordering for NHWC)
        var num_channels = UInt(self.in_channels)
        var filter_w = UInt(self.filter_w)

        # OPTIMIZATION: Hoist K decomposition outside loop (constant when j=0).
        var k_coord = coords[0]
        var filter_idx, c = divmod(k_coord, num_channels)
        var r, s = divmod(filter_idx, filter_w)

        # Initial M decomposition (done once, then use iterator)
        var m_coord_init = coords[1]
        var n, m_remainder = divmod(m_coord_init, hw)
        var h_out, w_out = divmod(m_remainder, out_w)

        # Pre-add lower_corner offset
        var h = Int(h_out) + Int(self.lower_corner_h)
        var w = Int(w_out) + Int(self.lower_corner_w)

        # Cache bounds for iterator wraparound
        var out_w_int = Int(out_w)
        var out_h_int = Int(self.out_height)
        var lower_h = Int(self.lower_corner_h)
        var lower_w = Int(self.lower_corner_w)

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                # K recomputation only needed when j > 0
                comptime if j > 0:
                    k_coord = coords[0] + UInt(j * copy_dim1)
                    filter_idx, c = divmod(k_coord, num_channels)
                    r, s = divmod(filter_idx, filter_w)

                # Pass 4D coords (c, w, h, n) and filter offsets (s, r) to im2col PTX
                cp_async_bulk_tensor_shared_cluster_global_im2col_multicast[
                    cta_group=cta_group,
                ](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(Int(c), w, h, Int(n)),
                    Index(Int(s), Int(r)),
                    multicast_mask,
                )

            # Iterator pattern: advance M by copy_dim0 using addition
            w += copy_dim0
            if w >= out_w_int + lower_w:
                w -= out_w_int
                h += 1
                if h >= out_h_int + lower_h:
                    h -= out_h_int
                    n += 1


@always_inline
fn _im2col_desc_tile_layout[
    dtype: DType,
    tile_shape: IndexList[2],
    swizzle_mode: TensorMapSwizzle,
]() -> Layout:
    """Compute the TMA descriptor layout for im2col.

    For im2col TMA, each transaction loads multiple output pixels with multiple channels.
    Following CUTLASS's approach (copy_traits_sm90_im2col.hpp:650-651):
    - channels_per_pixel = min(K_tile, swizzle_width) (contiguous channels)
    - pixels_per_column = computed from tile shape and TMA box constraints

    The TMA im2col box is constrained by hardware limits. The maximum box size
    is typically 256 elements for im2col TMA. This function computes the largest
    box that fits within these constraints.

    The descriptor layout is row_major(pixels_per_column, channels_per_pixel).
    """
    # Swizzle width in elements (bytes / element_size)
    comptime swizzle_bytes = (
        16 if swizzle_mode
        == TensorMapSwizzle.SWIZZLE_NONE else (
            32 if swizzle_mode
            == TensorMapSwizzle.SWIZZLE_32B else (
                64 if swizzle_mode == TensorMapSwizzle.SWIZZLE_64B else 128
            )
        )
    )
    comptime element_size = size_of[dtype]()
    comptime swizzle_width = swizzle_bytes // element_size

    # Channels per pixel is the minimum of K_tile and swizzle width
    comptime k_tile = tile_shape[1]
    comptime channels_per_pixel = swizzle_width if swizzle_width < k_tile else k_tile

    # Maximum TMA im2col box size in elements (hardware constraint)
    # Based on CUDA TMA documentation and CUTLASS patterns, 256 elements is
    # a safe limit for im2col TMA transactions.
    comptime max_tma_box_elements = 256

    # Compute pixels_per_column from tile shape and TMA constraints
    # pixels = min(M_tile, max_box_elements / channels_per_pixel)
    comptime m_tile = tile_shape[0]
    comptime max_pixels_from_box = max_tma_box_elements // channels_per_pixel
    comptime pixels_per_column = (
        m_tile if m_tile < max_pixels_from_box else max_pixels_from_box
    )

    return Layout.row_major(pixels_per_column, channels_per_pixel)


@always_inline
fn create_tensor_tile_im2col[
    dtype: DType,
    tile_shape: IndexList[2],  # [M_tile, K_tile] = [pixels, channels]
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1]),
    __desc_layout: Layout = _im2col_desc_tile_layout[
        dtype, tile_shape, swizzle_mode
    ](),
](
    ctx: DeviceContext,
    tensor: LayoutTensor[dtype, ...],  # 4D NHWC tensor
    lower_corner_h: Int,
    lower_corner_w: Int,
    upper_corner_h: Int,
    upper_corner_w: Int,
    out_height: Int,
    out_width: Int,
    filter_h: Int,
    filter_w: Int,
) raises -> TMATensorTileIm2col[dtype, __tile_layout, __desc_layout]:
    """Creates a TMA tensor tile with im2col transformation for 2D convolution.

    This factory function creates a TMA descriptor that performs hardware
    im2col transformation during loads. The descriptor encodes the convolution
    geometry and the TMA hardware computes addresses on-the-fly.

    For im2col TMA, each transaction loads one output pixel with multiple channels.
    This follows CUTLASS's approach where:
    - pixels_per_column = 1 (one pixel per TMA transaction)
    - channels_per_pixel = min(K_tile, swizzle_width) (contiguous channels)

    Parameters:
        dtype: The data type of tensor elements.
        tile_shape: Shape `[M_tile, K_tile]` for the GEMM tile.
            - M_tile: Number of output pixels (batch * H_out * W_out slice).
            - K_tile: Number of channels (C_in * R * S slice for filter).
        swizzle_mode: Memory swizzling pattern.
        __tile_layout: Internal layout parameter (full tile shape).
        __desc_layout: Internal descriptor layout parameter (TMA box shape).

    Args:
        ctx: The CUDA device context.
        tensor: The 4D activation tensor in NHWC layout.
        lower_corner_h: Lower corner offset for height (negative for padding).
        lower_corner_w: Lower corner offset for width (negative for padding).
        upper_corner_h: Upper corner offset for height.
        upper_corner_w: Upper corner offset for width.
        out_height: Output height (H_out) for M coordinate decomposition.
        out_width: Output width (W_out) for M coordinate decomposition.
        filter_h: Filter height (R) for K coordinate decomposition.
        filter_w: Filter width (S) for K coordinate decomposition.

    Returns:
        A TMATensorTileIm2col configured for im2col loads.

    Raises:
        Error if TMA descriptor creation fails.

    Note:
        For stride=1, dilation=1 convolution with padding (following CUTLASS convention):
        - lower_corner_h = -pad_h
        - lower_corner_w = -pad_w
        - upper_corner_h = pad_h - (filter_h - 1)
        - upper_corner_w = pad_w - (filter_w - 1)

        The filter offsets passed to the PTX instruction range from 0 to (filter_size - 1)
        and are added to lower_corner to compute actual input coordinates.
    """
    comptime assert tensor.rank == 4, "Im2col TMA requires 4D NHWC tensor"

    # Extract tensor dimensions
    var batch = tensor.dim(0)
    var height = tensor.dim(1)
    var width = tensor.dim(2)
    var channels = tensor.dim(3)

    # Create device buffer wrapper
    var global_buf = DeviceBuffer(
        ctx,
        tensor.ptr.mut_cast[True]().address_space_cast[AddressSpace.GENERIC](),
        1,
        owning=False,
    )

    # Global shape in NHWC order
    var global_shape = IndexList[4](batch, height, width, channels)

    # Compute row-major strides for NHWC layout
    # When tensor.stride() returns -1 (unknown at compile-time), compute from dims
    # For row-major NHWC: stride(i) = product of all dims after i
    var stride_n = height * width * channels  # batch stride
    var stride_h = width * channels  # height stride
    var stride_w = channels  # width stride
    var stride_c = 1  # channel stride (innermost)

    var global_strides = IndexList[4](stride_n, stride_h, stride_w, stride_c)

    # Spatial corners (H, W order)
    var lower_corner = IndexList[2](lower_corner_h, lower_corner_w)
    var upper_corner = IndexList[2](upper_corner_h, upper_corner_w)

    # Tile dimensions for TMA im2col box (from desc_layout)
    # desc_layout is row_major(pixels_per_column, channels_per_pixel)
    comptime pixels_per_column = __desc_layout.shape[0].value()
    comptime channels_per_pixel = __desc_layout.shape[1].value()

    # Convert TensorMapSwizzle to SwizzleMode (same underlying values)
    var swizzle = _SwizzleMode(Int32(Int(swizzle_mode)))

    var tensormap = _create_tensormap_im2col[dtype, 4, 2](
        global_buf,
        global_shape,
        global_strides,
        lower_corner,
        upper_corner,
        channels_per_pixel,
        pixels_per_column,
        swizzle,
    )

    # Convert TensorMap to TMADescriptor (both are 128-byte aligned, same layout)
    var descriptor = TMADescriptor()
    descriptor.data = tensormap.data

    return TMATensorTileIm2col[dtype, __tile_layout, __desc_layout](
        descriptor,
        UInt32(out_height),
        UInt32(out_width),
        UInt32(filter_h),
        UInt32(filter_w),
        UInt32(channels),  # in_channels from the NHWC tensor
        Int32(lower_corner_h),  # CUTLASS ArithmeticTupleIterator pattern
        Int32(lower_corner_w),
    )
