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

from memory import (
    LegacyOpaquePointer as OpaquePointer,
    LegacyUnsafePointer as UnsafePointer,
)
from sys import align_of, llvm_intrinsic, simd_width_of, size_of
from sys._assembly import inlined_assembly

from gpu.host import DeviceBuffer, DeviceContext
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
    cp_async_bulk_tensor_shared_cluster_global_multicast,
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
from layout.int_tuple import product, to_index_list
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major

from utils.index import Index, IndexList
from builtin.device_passable import DevicePassable
from math import log2


# Returns an IntTuple of variadic Int values.
#
fn _to_int_tuple[*vals: Int]() -> IntTuple:
    res = IntTuple()

    alias num_vals = stdlib.builtin.variadic_size(vals)

    @parameter
    for i in range(num_vals):
        res.append(vals[i])
    return res


fn _tma_desc_tile_layout[
    dtype: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    is_k_major: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> Layout:
    constrained[
        size_of[dtype]() >= 1, "Don't support sub-byte dtype in TMA yet."
    ]()

    constrained[
        rank == 2 or rank == 3, "Only support 2D/3D TMA descriptor for now."
    ]()

    @parameter
    if rank == 2:
        alias dim0 = tile_shape[0]
        alias dim1 = tile_shape[1]

        @parameter
        if is_k_major:
            # TMA copies BM x `swizzle_mode.bytes()` Bytes each time.
            return Layout.row_major(
                dim0, swizzle_mode.bytes() // size_of[dtype]()
            )
        else:
            constrained[
                swizzle_mode == TensorMapSwizzle.SWIZZLE_128B,
                "Only support 128B swizzle for mn-major.",
            ]()

            alias swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()

            @parameter
            if dim1 == swizzle_granularity:
                return Layout.row_major(dim0, swizzle_granularity)
            else:
                alias core_matrix_num_rows = 8
                return Layout.row_major(
                    core_matrix_num_rows, swizzle_granularity
                )

    else:
        alias dim0 = tile_shape[0]
        alias dim1 = tile_shape[1]
        alias dim2 = tile_shape[2]

        constrained[is_k_major, "Only K-Major is supported!"]()

        return Layout(
            [dim0, dim1, swizzle_mode.bytes() // size_of[dtype]()],
            [1, 1, 1],
        )


@register_passable("trivial")
struct SharedMemBarrier(ImplicitlyCopyable, Movable):
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
    fn init(ref [AddressSpace.SHARED]self, num_threads: Int32 = 1):
        """Initialize the barrier state with the expected number of threads.

        Sets up the barrier to expect arrivals from the specified number of threads
        before it can be satisfied. This is essential for coordinating thread
        synchronization in GPU kernels.

        Args:
            num_threads: Number of threads that must arrive at the barrier
                         before it is satisfied. Defaults to 1.
        """
        mbarrier_init(self.unsafe_ptr(), num_threads)

    @always_inline
    fn expect_bytes(ref [AddressSpace.SHARED]self, bytes: Int32):
        """Configure the barrier to expect a specific number of bytes to be transferred.

        Used with TMA operations to indicate the expected size of data transfer.
        The barrier will be satisfied when the specified number of bytes has been
        transferred, enabling efficient coordination of memory operations.

        Args:
            bytes: Number of bytes expected to be transferred.
        """
        mbarrier_arrive_expect_tx_shared(self.unsafe_ptr(), bytes)

    @always_inline
    fn expect_bytes_relaxed(
        ref [AddressSpace.SHARED]self, bytes: Int32
    ) -> UInt64:
        """Configure the barrier to expect a specific number of bytes to be transferred.

        Used with TMA operations to indicate the expected size of data transfer.
        The barrier will be satisfied when the specified number of bytes has been
        transferred, enabling efficient coordination of memory operations.

        Args:
            bytes: Number of bytes expected to be transferred.

        Returns:
            The state.
        """
        return mbarrier_arrive_expect_tx_relaxed(self.unsafe_ptr(), bytes)

    @always_inline
    fn arrive_and_expect_bytes(
        ref [AddressSpace.SHARED]self,
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
        """

        alias asm = """
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
    ](ref [AddressSpace.SHARED]self, phase: UInt32 = 0):
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

        alias wait_asm = (
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1"
            + (" , $2" if ticks else "")
            + ";"
        )
        alias asm = """{
            .reg .pred P1;
            LAB_WAIT:
            """ + wait_asm + """
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
        }"""

        alias constraints = "r,r" + (",r" if ticks else "")

        @parameter
        if ticks:
            inlined_assembly[asm, NoneType, constraints=constraints](
                Int32(Int(self.unsafe_ptr())), phase, UInt32(ticks.value())
            )
        else:
            inlined_assembly[asm, NoneType, constraints=constraints](
                Int32(Int(self.unsafe_ptr())), phase
            )

    @always_inline("nodebug")
    fn wait_acquire[
        scope: Scope
    ](ref [AddressSpace.SHARED]self, phase: UInt32 = 0):
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

        constrained[
            scope == Scope.CLUSTER or scope == Scope.BLOCK,
            "wait_acquire is only supported for cluster or block/CTA scope.",
        ]()

        alias asm = (
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
    ](ref [AddressSpace.SHARED]self, phase: UInt32 = 0):
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

        constrained[
            scope == Scope.CLUSTER or scope == Scope.BLOCK,
            "wait_relaxed is only supported for cluster or block/CTA scope.",
        ]()

        alias asm = (
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

    @always_inline
    fn unsafe_ptr[
        mut: Bool, //,
        origin: Origin[mut],
    ](
        ref [origin, AddressSpace.SHARED]self,
    ) -> UnsafePointer[
        Int64,
        address_space = AddressSpace.SHARED,
        mut=mut,
        origin=origin,
    ]:
        """Get an unsafe pointer to the barrier's memory location.

        Provides low-level access to the shared memory location storing the barrier state.
        This method is primarily used internally by other barrier operations that need
        direct access to the underlying memory.

        Parameters:
            mut: Mutability of self.
            origin: Origin of self.

        Returns:
            An unsafe pointer to the barrier's memory location in shared memory,
            properly typed and aligned for barrier operations.
        """
        return (
            UnsafePointer(to=self.mbar)
            .mut_cast[mut]()
            .unsafe_origin_cast[origin]()
        )

    @always_inline
    fn arrive_cluster(
        ref [AddressSpace.SHARED]self, cta_id: UInt32, count: UInt32 = 1
    ):
        """Signal arrival at the barrier from a specific CTA (Cooperative Thread Array) in a cluster.

        This method is used in multi-CTA scenarios to coordinate barrier arrivals
        across different CTAs within a cluster. It enables efficient synchronization
        across thread blocks in clustered execution models.

        Args:
            cta_id: The ID of the CTA (Cooperative Thread Array) that is arriving.
            count: The number of arrivals to signal. Defaults to 1.
        """
        alias asm = """{
            .reg .b32 remAddr32;
            mapa.shared::cluster.u32  remAddr32, $0, $1;
            mbarrier.arrive.shared::cluster.b64  _, [remAddr32], $2;
        }"""
        inlined_assembly[asm, NoneType, constraints="r,r,r"](
            Int32(Int(self.unsafe_ptr())), cta_id, count
        )

    @always_inline("nodebug")
    fn arrive(ref [AddressSpace.SHARED]self) -> Int:
        """Signal arrival at the barrier and return the arrival count.

        This method increments the arrival count at the barrier and returns
        the updated count. It's used to track how many threads have reached
        the synchronization point.

        Returns:
            The updated arrival count after this thread's arrival.
        """
        return mbarrier_arrive(self.unsafe_ptr())


@register_passable("trivial")
struct PipelineState[num_stages: Int](Defaultable, ImplicitlyCopyable, Movable):
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
        self._index = index
        self._phase = phase
        self._count = count

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

        @parameter
        if Self.num_stages > 1:
            self._index += 1
            self._count += 1
            if self._index == Self.num_stages:
                self._index = 0
                self._phase ^= 1

        @parameter
        if Self.num_stages == 1:
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
](DevicePassable, ImplicitlyCopyable, Movable):
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

    alias device_type: AnyType = Self

    fn _to_device_type(self, target: OpaquePointer):
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
            Self.layout,
            ", desc_layout = ",
            Self.desc_layout,
            ", is_k_major = ",
            Self.is_k_major,
            "]",
        )

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Gets device_type's name, for use in error messages when handing arguments
        to kernels.

        Returns:
            This type's name.
        """
        return Self.get_type_name()

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
    fn __copyinit__(out self, other: Self):
        """
        Copy initializes this `TMATensorTile` from another instance.

        Args:
            other: The other `TMATensorTile` instance to copy from.
        """
        self.descriptor = other.descriptor

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
        cta_group: Int = 1
    ](
        self,
        dst: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_],
        ref [AddressSpace.SHARED]mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt],
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
        constrained[
            type_of(dst).alignment % 128 == 0,
            "TMA requires 128B alignment in shared memory",
        ]()

        constrained[
            type_of(dst).dtype == Self.dtype,
            "Input tensor has a different type than the TMA op",
        ]()

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        alias copy_dim0 = Self.desc_layout.shape[0].value()
        alias copy_dim1 = Self.desc_layout.shape[1].value()
        alias copy_size = Self.desc_layout.size()
        alias num_copies_dim0 = product(
            Self.layout.shape[Int(not Self.is_k_major)]
        ) // copy_dim0
        alias num_copies_dim1 = product(
            Self.layout.shape[Int(Self.is_k_major)]
        ) // copy_dim1

        @parameter
        for i in range(num_copies_dim0):

            @parameter
            for j in range(num_copies_dim1):
                alias copy_offset = (i * num_copies_dim1 + j) * copy_size

                constrained[
                    (copy_offset * size_of[Self.dtype]()) % 128 == 0,
                    "copy_offset="
                    + String(copy_offset)
                    + ", size_of[dtype]()="
                    + String(size_of[Self.dtype]())
                    + "\nlayout="
                    + String(Self.layout)
                    + "\ndesc_layout="
                    + String(Self.desc_layout),
                ]()
                cp_async_bulk_tensor_shared_cluster_global[cta_group=cta_group](
                    dst.ptr.mut_cast[True]() + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + UInt(j * copy_dim1),
                        coords[1] + UInt(i * copy_dim0),
                    ),
                )

    @always_inline
    fn async_copy_3d(
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        ref [AddressSpace.SHARED]mem_barrier: SharedMemBarrier,
        coords: Tuple[UInt, UInt, UInt],
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

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        constrained[
            type_of(dst).alignment % 128 == 0,
            "TMA requires 128B alignment in shared memory",
        ]()

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        alias copy_dim0 = Self.desc_layout.shape[0].value()
        alias copy_dim1 = Self.desc_layout.shape[1].value()
        alias copy_dim2 = Self.desc_layout.shape[2].value()
        alias copy_size = Self.desc_layout.size()
        alias num_copies_dim0 = Self.layout.shape[0].value() // copy_dim0
        alias num_copies_dim1 = Self.layout.shape[1].value() // copy_dim1
        alias num_copies_dim2 = Self.layout.shape[2].value() // copy_dim2

        @parameter
        for m in range(num_copies_dim0):

            @parameter
            for i in range(num_copies_dim1):

                @parameter
                for j in range(num_copies_dim2):
                    alias copy_offset = m * (
                        num_copies_dim1 * num_copies_dim2
                    ) + (i * num_copies_dim2 + j) * copy_size

                    cp_async_bulk_tensor_shared_cluster_global(
                        dst.ptr.mut_cast[True]() + copy_offset,
                        UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + UInt(j * copy_dim2),
                            coords[1] + UInt(i * copy_dim1),
                            coords[2] + UInt(m * copy_dim0),
                        ),
                    )

    @always_inline
    fn async_multicast_load[
        cta_group: Int = 1
    ](
        self,
        dst: LayoutTensor[
            Self.dtype, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        ref [AddressSpace.SHARED]mem_barrier: SharedMemBarrier,
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
        constrained[
            type_of(dst).alignment % 128 == 0,
            "TMA requires 128B alignment in shared memory",
        ]()

        alias copy_dim0 = Self.desc_layout.shape[0].value()
        alias copy_dim1 = Self.desc_layout.shape[1].value()
        alias copy_size = Self.desc_layout.size()
        alias num_copies_dim0 = Self.layout.shape[0].value() // copy_dim0
        alias num_copies_dim1 = Self.layout.shape[1].value() // copy_dim1

        @parameter
        for i in range(num_copies_dim0):

            @parameter
            for j in range(num_copies_dim1):
                alias copy_offset = (i * num_copies_dim1 + j) * copy_size

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
            *_, **_,
        ],
        ref [AddressSpace.SHARED]mem_barrier: SharedMemBarrier,
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
            Self.dtype, _, address_space = AddressSpace.SHARED, **_
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
        constrained[
            type_of(src).alignment % 128 == 0,
            "TMA requires 128B alignment in shared memory",
        ]()

        alias copy_dim0 = Self.desc_layout.shape[0].value()
        alias copy_dim1 = Self.desc_layout.shape[1].value()
        alias copy_size = Self.desc_layout.size()
        alias num_copies_dim0 = product(
            Self.layout.shape[Int(not Self.is_k_major)]
        ) // copy_dim0
        alias num_copies_dim1 = product(
            Self.layout.shape[Int(Self.is_k_major)]
        ) // copy_dim1

        @parameter
        for i in range(num_copies_dim0):

            @parameter
            for j in range(num_copies_dim1):
                alias copy_offset = (i * num_copies_dim1 + j) * copy_size

                cp_async_bulk_tensor_global_shared_cta(
                    src.ptr + copy_offset,
                    UnsafePointer(to=self.descriptor).bitcast[NoneType](),
                    Index(
                        coords[0] + UInt(j * copy_dim1),
                        coords[1] + UInt(i * copy_dim0),
                    ),
                )

    @always_inline
    fn async_reduce[
        reduction_kind: ReduceOp
    ](
        self,
        src: LayoutTensor[
            Self.dtype, Self.layout, address_space = AddressSpace.SHARED, **_
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
        constrained[
            type_of(src).alignment % 128 == 0,
            "TMA requires 128B alignment in shared memory",
        ]()
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
        cp_async_bulk_wait_group[n]()

    @always_inline
    fn smem_tensormap_init(
        self,
        smem_tma_descriptor_ptr: UnsafePointer[
            TMADescriptor, address_space = AddressSpace.SHARED
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
        var dst_desc = smem_tma_descriptor_ptr.bitcast[UInt8]()

        alias simd_width = simd_width_of[DType.uint8]()
        alias src_align = align_of[SIMD[DType.uint8, simd_width]]()
        alias dst_align = align_of[SIMD[DType.uint8, simd_width]]()

        alias descriptor_bytes = 128

        @parameter
        for src_idx in range(descriptor_bytes // simd_width):
            var src_vec = (src_desc).load[
                width=simd_width, alignment=src_align
            ](src_idx * simd_width)
            dst_desc.store[alignment=dst_align](src_idx * simd_width, src_vec)

    @always_inline
    fn replace_tensormap_global_address_in_gmem[
        _dtype: DType,
    ](self, src_ptr: UnsafePointer[Scalar[_dtype],],):
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

        constrained[
            src_ptr.address_space
            in (AddressSpace.GENERIC, AddressSpace.GLOBAL),
            "src address space must be GENERIC or GLOBAL.",
        ]()

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
            TMADescriptor, address_space = AddressSpace.SHARED, **_
        ],
        src_ptr: UnsafePointer[Scalar[_dtype],],
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

        constrained[
            src_ptr.address_space
            in (AddressSpace.GENERIC, AddressSpace.GLOBAL),
            "src address space must be GENERIC or GLOBAL.",
        ]()

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
            TMADescriptor, address_space = AddressSpace.SHARED, **_
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

        @parameter
        if only_update_dim_0:
            alias temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
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
            @parameter
            for i in range(rank):
                alias temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
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
            @parameter
            for i in range(1, rank):
                alias temp = "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [$0], " + String(
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
            TMADescriptor, address_space = AddressSpace.SHARED, **_
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

        alias temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
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
        @parameter
        if dim_idx > 0:
            debug_assert(
                dim_stride is not None,
                " dim_stride must be provided if dim_idx > 0",
            )
            alias temp = "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [$0], " + String(
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
    alias swizzle_rows_bytes = tile_sizes[tensor.rank - 1] * size_of[
        tensor.dtype
    ]()

    @parameter
    if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        constrained[
            swizzle_rows_bytes <= swizzle_mode.bytes(),
            "Current swizzle bytes is ",
            String(swizzle_rows_bytes),
            " which exceeds ",
            String(swizzle_mode.bytes()),
            "B swizzle requirement.",
        ]()

    return create_tma_descriptor[tensor.dtype, 2, swizzle_mode](
        DeviceBuffer(
            ctx,
            tensor.ptr.mut_cast[True]().address_space_cast[
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
    rank: Int, //,
    desc_index_list: IndexList[rank],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext, tensor: LayoutTensor[dtype, *_, **_]) -> TMADescriptor:
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

    constrained[rank == tensor.rank, "Rank mismatch"]()

    alias global_shape = tensor.layout.shape.product_flatten()
    alias global_strides = tensor.layout.stride.product_flatten()

    alias swizzle_rows_bytes = desc_index_list[rank - 1] * size_of[
        tensor.dtype
    ]()

    alias global_shape_list = to_index_list[rank](global_shape)
    alias global_strides_list = to_index_list[rank](global_strides)

    @parameter
    if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        constrained[
            swizzle_rows_bytes <= swizzle_mode.bytes(),
            "Current swizzle bytes is ",
            String(swizzle_rows_bytes),
            " which exceeds ",
            String(swizzle_mode.bytes()),
            "B swizzle requirement.",
        ]()

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
def create_tma_tile[
    dtype: DType,
    rank: Int, //,
    tile_shape: IndexList[rank],
    /,
    is_k_major: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1]),
    __desc_layout: Layout = _tma_desc_tile_layout[
        dtype, rank, tile_shape, is_k_major, swizzle_mode
    ](),
](ctx: DeviceContext, tensor: LayoutTensor[dtype, *_, **_]) -> TMATensorTile[
    dtype, __tile_layout, __desc_layout
]:
    """
    Creates a `TMATensorTile` with advanced configuration options for 2D or 3D tensors.

    This overload provides more control over the TMA descriptor creation, allowing
    specification of data type, rank, and layout orientation. It supports both 2D and 3D
    tensors and provides fine-grained control over the memory access patterns.

    Parameters:
        dtype: DType
            The data type of the tensor elements.
        rank: Int
            The dimensionality of the tensor (must be 2 or 3).
        tile_shape: IndexList[rank]
            The shape of the tile to be transferred.
        is_k_major: Bool = True
            Whether the tensor layout is K-major (True) or MN-major (False).
            K-major is typically used for weight matrices, while MN-major is used for
            activation matrices in matrix multiplication operations.
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
        tensor: LayoutTensor[dtype, *_, **_]
            The source tensor from which data will be transferred. This defines the
            global memory layout and must match the specified data type.

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
    # Current impl limitations
    constrained[rank == 2 or rank == 3, "Only support 2D/3D TMA"]()

    alias desc_bytes_size = __desc_layout.size() * size_of[dtype]()
    alias layout_size = __tile_layout.size() * size_of[dtype]()

    @parameter
    if desc_bytes_size < layout_size:
        # When we do multiple TMA copy, every address has to be align to 128.
        constrained[
            desc_bytes_size % 128 == 0,
            (
                "desc layout byte size has to be  align to 128 bytes for"
                " multiple TMA copies. desc_layout: "
                + String(__desc_layout.shape[0].value())
                + " "
                + String(__desc_layout.shape[1].value())
                + " tile_layout: "
                + String(__tile_layout.shape[0].value())
                + " "
                + String(__tile_layout.shape[1].value())
            ),
        ]()

    @parameter
    if rank == 2:

        @parameter
        if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            constrained[
                (tile_shape[1] * size_of[dtype]()) % swizzle_mode.bytes() == 0,
                String(swizzle_mode),
                " mode requires K dim multiple of ",
                String(swizzle_mode.bytes()),
                "B. K dim is now ",
                String(tile_shape[1] * size_of[dtype]()),
                " bytes.",
            ]()

        return create_tma_descriptor[dtype, 2, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.mut_cast[True]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
                1,
                owning=False,
            ),
            (tensor.dim(0), tensor.dim(1)),
            (tensor.stride(0), tensor.stride(1)),
            (__desc_layout.shape[0].value(), __desc_layout.shape[1].value()),
        )

    else:

        @parameter
        if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            constrained[
                (tile_shape[2] * size_of[dtype]()) % swizzle_mode.bytes() == 0,
                String(swizzle_mode),
                " mode requires K dim multiple of ",
                String(swizzle_mode.bytes()),
                "B. K dim is now ",
                String(tile_shape[2] * size_of[dtype]()),
                "bytes.",
            ]()

        return create_tma_descriptor[dtype, 3, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.mut_cast[True]().address_space_cast[
                    AddressSpace.GENERIC
                ](),
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


alias TMANestedTensorTile[
    dtype: DType,
    tile_m: Int,
    tile_n: Int,
    swizzle_mode: TensorMapSwizzle,
    is_k_major: Bool,
] = TMATensorTile[
    dtype,
    tile_layout_k_major[
        dtype, tile_m, tile_n, swizzle_mode=swizzle_mode
    ]() if is_k_major else tile_layout_mn_major[
        dtype, tile_n, tile_m, swizzle_mode=swizzle_mode
    ](),
    _tma_desc_tile_layout[
        dtype, 2, IndexList[2](tile_m, tile_n), is_k_major, swizzle_mode
    ](),
    is_k_major,
]


fn create_nested_tma_tile[
    dtype: DType, //,
    tile_m: Int,
    tile_n: Int,
    swizzle_mode: TensorMapSwizzle,
    *,
    is_k_major: Bool,
](
    ctx: DeviceContext,
    tensor: LayoutTensor[dtype, *_, **_],
    out res: TMANestedTensorTile[
        dtype, tile_m, tile_n, swizzle_mode, is_k_major
    ],
) raises:
    """
    Creates a rank 2 `TMATensorTile` with a nested layout using
    `tile_layout_k_major` is `is_k_major` or `tile_layout_mn_major` otherwise.

    Parameters:
        dtype: DType
            The data type of the tensor elements.
        tile_m: The number of rows of a global memory tile.
        tile_n: The number of columns of a global memory tile.
        swizzle_mode: The swizzle_mode used by the TMA operation.
        is_k_major: Whether the shared memory is to be k-major
            or mn-major. If mn-major, it is transposed.

    Args:
        ctx: DeviceContext
            The CUDA device context used to create the TMA descriptor.
        tensor: LayoutTensor[type, *_, **_]
            The source tensor from which data will be transferred. This defines the
            global memory layout and must match the specified data type.

    Returns:
        The `TMATensorTile` configured with the specified tile dimensions and
        swizzle mode, ready for use in asynchronous data transfer operations.

    Raises:
        If there was an error creating the underlying TMADescriptor.
    """
    alias ResultType = type_of(res)
    alias desc_layout = ResultType.desc_layout
    alias desc_bytes_size = desc_layout.size() * size_of[dtype]()
    alias layout_size = ResultType.layout.size() * size_of[dtype]()

    # When we do multiple TMA copy, every address has to be align to 128.
    constrained[
        desc_bytes_size == layout_size or desc_bytes_size % 128 == 0,
        (
            "desc layout byte size has to be  align to 128 bytes for"
            " multiple TMA copies. desc_layout: "
            + String(desc_layout.shape[0].value())
            + " "
            + String(desc_layout.shape[1].value())
            + " tile_layout: "
            + String(tile_m)
            + " "
            + String(tile_n)
        ),
    ]()

    @parameter
    if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        constrained[
            (tile_n * size_of[dtype]()) % swizzle_mode.bytes() == 0,
            String(swizzle_mode),
            " mode requires K dim multiple of ",
            String(swizzle_mode.bytes()),
            "B. K dim is now ",
            String(tile_n * size_of[dtype]()),
            " bytes. tile_m: " + String(tile_m) + " tile_n: " + String(tile_n),
        ]()

    res = create_tma_descriptor[dtype, 2, swizzle_mode](
        DeviceBuffer(
            ctx,
            tensor.ptr.mut_cast[True]().address_space_cast[
                AddressSpace.GENERIC
            ](),
            1,
            owning=False,
        ),
        IndexList[2](tensor.dim(0), tensor.dim(1)),
        IndexList[2](tensor.stride(0), tensor.stride(1)),
        IndexList[2](
            desc_layout.shape[0].value(),
            desc_layout.shape[1].value(),
        ),
    )


@always_inline
def create_tma_tile_template[
    dtype: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    /,
    is_k_major: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1]),
    __desc_layout: Layout = _tma_desc_tile_layout[
        dtype, rank, tile_shape, is_k_major, swizzle_mode
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
        is_k_major: Bool = True
            Whether the tensor layout is K-major (True) or MN-major (False).
            K-major is typically used for weight matrices, while MN-major is used for
            activation matrices in matrix multiplication operations.
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


@register_passable("trivial")
struct TMATensorTileArray[
    num_of_tensormaps: Int,
    dtype: DType,
    cta_tile_layout: Layout,
    desc_layout: Layout,
](DevicePassable, ImplicitlyCopyable, Movable):
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

    var tensormaps_ptr: UnsafePointer[UInt8]
    """A static tuple of pointers to TMA descriptors.

    This field stores an array of pointers to `TMATensorTile` instances, where each pointer
    references a TMA descriptor in device memory. The array has a fixed size determined by
    the num_of_tensormaps parameter.

    The TMA descriptors are used by the GPU hardware to efficiently transfer data between
    global and shared memory with specific memory access patterns defined by the layouts.
    """

    alias descriptor_bytes = 128
    """Size of the TMA descriptor in bytes.

    This is a constant value that represents the size of the TMA descriptor in bytes.
    It is used to calculate the offset of the TMA descriptor in the device memory.
    """

    alias device_type: AnyType = Self

    fn _to_device_type(self, target: OpaquePointer):
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
            Self.cta_tile_layout,
            ", desc_layout = ",
            Self.desc_layout,
            "]",
        )

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Gets device_type's name, for use in error messages when handing arguments
        to kernels.

        Returns:
            This type's name.
        """
        return Self.get_type_name()

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
        self,
        index: Int,
        out result: UnsafePointer[
            TMATensorTile[Self.dtype, Self.cta_tile_layout, Self.desc_layout]
        ],
    ):
        """
        Retrieve a TMA descriptor.

        Args:
            index: Index of the TMA descriptor.

        Returns:
            `UnsafePointer` to the `TMATensorTile` at the specified index.
        """
        return UnsafePointer[UInt8](
            self.tensormaps_ptr + index * self.descriptor_bytes
        ).bitcast[
            TMATensorTile[Self.dtype, Self.cta_tile_layout, Self.desc_layout]
        ]()


struct TensorMapArray[
    rank: Int, //,
    dtype: DType,
    desc_remaining_tile_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    max_descriptor_length: Int = 256,
](DevicePassable, ImplicitlyCopyable):
    """
    An array of TMA descriptors for efficient multi-descriptor management.

    This struct maintains multiple TMA (Tensor Memory Access) descriptors organized in a
    power-of-2 indexed array structure. It enables efficient selection and use of different
    descriptor configurations at runtime, which is particularly useful for handling varying
    tensor dimensions or batch sizes in GPU operations.

    The array uses a logarithmic indexing scheme where descriptors are stored at positions
    corresponding to powers of 2 (1, 2, 4, 8, 16, ..., up to max_descriptor_length). This
    allows for efficient lookup and memory usage while supporting a wide range of descriptor
    configurations.

    Parameters:
        rank:
            The rank (number of dimensions) of the tensors that will be accessed.
            Currently supports rank 1 or 2.
        dtype:
            The data type of the tensor elements that will be transferred.
        desc_remaining_tile_shape:
            All dims of the descriptor shape except the first dimension.
        swizzle_mode:
            The swizzling mode to use for memory access optimization. Swizzling can improve
            memory access patterns for specific hardware configurations. Defaults to SWIZZLE_NONE.
        max_descriptor_length:
            The maximum first dimension size supported by the descriptor array. The array
            will contain descriptors for all powers of 2 up to this value. Defaults to 256.

    Constraints:
        - The rank must be 1 or 2 for descriptor shape construction.
        - When swizzling is enabled, tile dimensions must comply with swizzle mode byte limits.
        - max_descriptor_length should be a reasonable power of 2 to optimize memory usage.
    """

    alias arr_size = Int(log2(Float32(Self.max_descriptor_length))) + 1
    """How many descriptors are in the array."""

    @staticmethod
    fn _desc_length_list() -> IndexList[Self.arr_size]:
        """
        Constructs a list of descriptor lengths in ascending order.

        Returns:
            A list of descriptor lengths in ascending order.
        """

        var res = IndexList[Self.arr_size](fill=0)

        for i in range(Self.arr_size):
            res[i] = 2**i

        return res

    alias desc_length_list = Self._desc_length_list()
    """The list of descriptor lengths in ascending order."""

    alias desc_length_list_reverse = Self.desc_length_list.reverse()
    """The list of descriptor lengths in descending order."""

    var descriptor_array: InlineArray[TMADescriptor, Self.arr_size]
    """The array of TMA descriptors."""

    @staticmethod
    fn _descriptor_shape[first_dim: Int]() -> IntTuple:
        """
        Constructs a descriptor shape from a first dimension.

        Parameters:
            first_dim:
                The first dimension of the descriptor shape.

        Returns:
            A descriptor shape.
        """

        constrained[
            Self.rank == 2 or Self.rank == 1,
            "we can only construct 2D or 3D descriptor shapes",
        ]()
        var tup = IntTuple(num_elems=Self.rank + 1)
        tup.replace_entry(0, int_value=first_dim)

        for i in range(Self.rank):
            tup.replace_entry(
                i + 1, int_value=Self.desc_remaining_tile_shape[i]
            )

        return tup

    alias device_type: AnyType = Self
    """The TensorMapDescriptorArray type"""

    fn _to_device_type(self, target: OpaquePointer):
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
            "TensorMapDescriptorArray[rank = ",
            Self.rank,
            ", dtype = ",
            Self.dtype,
            ", desc_remaining_tile_shape = ",
            Self.desc_remaining_tile_shape,
            ", swizzle_mode = ",
            Self.swizzle_mode,
            ", max_descriptor_length = ",
            Self.max_descriptor_length,
            "]",
        )

    @staticmethod
    fn calculate_dim_repeat(sequence_length: Int) -> Int:
        """
        Returns the number of times the descriptor length fits into the sequence length.

        Args:
            sequence_length:
                The length of the sequence to be transferred.

        Returns:
            The number of times the 1 dim descriptor fits into the sequence length.
        """
        alias desc_shape_1 = Self._descriptor_shape[1]()
        alias desc_size_1 = product(desc_shape_1)

        return sequence_length // desc_size_1

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Returns the device type name for this descriptor array.

        Returns:
            A string containing the type name with all template parameters.
        """
        return Self.get_type_name()

    fn __init__(
        out self,
        ctx: DeviceContext,
        global_tensor: LayoutTensor[Self.dtype, *_, **_],
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
            global_tensor:
                The source tensor in global memory that will be accessed using these descriptors.
                This defines the global memory layout and data type.

        Constraints:
            - max_descriptor_length must be a power of two.
            - max_descriptor_length must be less than or equal to 256.
        """

        constrained[
            Self.max_descriptor_length.is_power_of_two(),
            "max_descriptor_length must be a power of two",
        ]()
        constrained[
            Self.max_descriptor_length <= 256,
            "max_descriptor_length must be less than or equal to 256",
        ]()

        self.descriptor_array = InlineArray[TMADescriptor, Self.arr_size](
            fill=TMADescriptor()
        )

        @parameter
        for i in range(Self.arr_size):
            alias desc_length = Self.desc_length_list[i]
            alias desc_tile_tuple = Self._descriptor_shape[desc_length]()
            alias desc_tile_shape = to_index_list[len(desc_tile_tuple)](
                desc_tile_tuple
            )

            self.descriptor_array[i] = _create_tma_descriptor_helper[
                desc_tile_shape, Self.swizzle_mode
            ](ctx, global_tensor)

    fn _get_descriptor_ptr[desc_index: Int](self) -> UnsafePointer[NoneType]:
        """
        Returns a pointer to the descriptor for a specific first dimension size.

        This method computes the logarithmic index for the given descriptor index
        and returns a pointer to the corresponding TMA descriptor in the array.

        Parameters:
            desc_index:
                The first dimension size for which to retrieve the descriptor.
                Must be a power of 2 between 1 and max_descriptor_length.

        Returns:
            An unsafe pointer to the TMA descriptor for the specified dimension size.
        """
        alias idx = Int(log2(Float32(desc_index)))
        var ptr = self.descriptor_array.unsafe_ptr() + idx
        return ptr.bitcast[NoneType]()

    fn store_ragged_tile(
        self,
        rows_to_copy: Int,
        start_coord: IndexList[Self.rank + 1],
        src: UnsafePointer[
            Scalar[Self.dtype], *_, address_space = AddressSpace.SHARED, **_
        ],
    ):
        """
        Stores a ragged tile from shared memory to global memory using multiple TMA descriptors.

        This method efficiently handles non-power-of-2 row counts by decomposing the transfer
        into multiple operations using the largest possible descriptors. It uses a greedy algorithm
        to select descriptors in descending order (largest first), minimizing the number of
        individual TMA operations required.

        For example, transferring 13 rows would use descriptors for 8 + 4 + 1 rows.

        Args:
            rows_to_copy:
                The total number of rows to transfer from shared memory to global memory.
            start_coord:
                The starting coordinate in global memory where the transfer should begin.
                Must have rank+1 dimensions (includes the batch/first dimension).
            src:
                The source pointer in shared memory containing the data to be transferred.
                Must be in the SHARED address space.
        """

        var row_chunk_size = rows_to_copy
        var current_coord = start_coord

        var src_ptr = src

        # NOTE: would it be more efficent to use break, instead of unrolled loop?
        @parameter
        for i in range(len(Self.desc_length_list)):
            alias desc_length = Self.desc_length_list_reverse[i]

            if desc_length <= row_chunk_size:
                var load_count = row_chunk_size // desc_length
                row_chunk_size = row_chunk_size % desc_length

                current_coord = self._batched_async_store[desc_length](
                    src_ptr, load_count, current_coord
                )

                alias desc_shape = Self._descriptor_shape[desc_length]()
                alias copy_size = product(desc_shape)

                src_ptr += copy_size * load_count

    @always_inline
    fn _batched_async_store[
        desc_length: Int
    ](
        self,
        src: UnsafePointer[
            Scalar[Self.dtype], *_, address_space = AddressSpace.SHARED, **_
        ],
        num_copies: Int,
        coords: IndexList[Self.rank + 1],
    ) -> IndexList[Self.rank + 1]:
        """
        Performs batched asynchronous stores using a specific descriptor size.

        This internal method executes multiple TMA store operations using the same descriptor,
        incrementing the destination coordinates for each successive copy. It's used by
        `store_ragged_tile` to efficiently transfer contiguous chunks of data.

        Parameters:
            desc_length:
                The first dimension size of the descriptor to use for the transfers.
                Must be a power of 2 and correspond to an available descriptor in the array.

        Args:
            src:
                Pointer to the source data in shared memory. The method will advance this
                pointer by `copy_size` for each successive copy.
            num_copies:
                The number of times to repeat the TMA store operation with this descriptor.
            coords:
                The starting coordinates in global memory for the first copy.

        Returns:
            The updated coordinates after all copies have been initiated, pointing to the
            position immediately after the last transferred data.
        """

        alias desc_shape = Self._descriptor_shape[desc_length]()
        alias copy_size = product(desc_shape)
        alias desc_0 = product(desc_shape[0])

        var row_increment = IndexList[Self.rank + 1](fill=0)
        var reflected_position = Self.rank

        for i in range(num_copies):
            var copy_offset = i * copy_size
            var current_coords = coords + row_increment

            cp_async_bulk_tensor_global_shared_cta(
                src + copy_offset,
                self._get_descriptor_ptr[desc_length](),
                current_coords,
            )

            row_increment[reflected_position] = (i + 1) * desc_0

        return coords + row_increment
