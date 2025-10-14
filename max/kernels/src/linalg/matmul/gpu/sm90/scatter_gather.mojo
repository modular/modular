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

"""ScatterGather module for efficient tile loading in GPU matrix multiplication.

This module provides utilities for loading matrix tiles from global memory to
shared memory using two different mechanisms:

1. TMA (Tensor Memory Accelerator): Hardware-accelerated loads that can efficiently
   transfer 2D tiles with multicast support for multi-block clusters.

2. cp.async: Software-based asynchronous copy instructions with manual bounds
   checking and swizzling for optimal shared memory access patterns.

The ScatterGather struct abstracts these loading mechanisms to provide a unified
interface for the matmul kernel's producer threads.
"""
from layout.tma_async import TMATensorTile
from layout.layout_tensor import LayoutTensor
from gpu.memory import (
    AddressSpace,
    async_copy,
)
from ....structuring import SharedMemBarrier
from layout.swizzle import make_swizzle
from gpu.id import thread_idx
from sys import simd_width_of
from gpu.host._nvidia_cuda import TensorMapSwizzle
from layout.layout import coalesce


@register_passable("trivial")
struct ScatterGather:
    """Utilities for efficient tile loading from global to shared memory.

    This struct provides static methods for loading matrix tiles using either
    TMA (Tensor Memory Accelerator) or cp.async instructions, depending on the
    hardware capabilities and memory alignment requirements.
    """

    @staticmethod
    @always_inline
    fn load_tile[
        dtype: DType,
        tile_layout: Layout,
        desc_layout: Layout,
        dst_layout: Layout, //,
        cluster_size: Int,
        use_partitioned_multicast: Bool,
    ](
        tma_op: TMATensorTile[dtype, tile_layout, desc_layout],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            address_space = AddressSpace.SHARED,
            alignment=128,
            *_, **_,
        ],
        ref [AddressSpace.SHARED]mem_barrier: SharedMemBarrier,
        rank: UInt,
        coords: Tuple[UInt, UInt],
        multicast_mask: UInt16,
    ):
        """Load a tile using TMA (Tensor Memory Accelerator) with optional multicast.

        This method uses hardware-accelerated TMA instructions to efficiently load
        2D tiles from global memory to shared memory. It supports multicasting the
        load across multiple thread blocks in a cluster for improved efficiency.

        Template Parameters:
            dtype: Data type of the tile elements.
            tile_layout: Layout of the source tile in global memory.
            desc_layout: Layout described by the TMA descriptor.
            dst_layout: Layout of the destination tile in shared memory.
            cluster_size: Number of blocks in the cluster (1 for single-block).
            use_partitioned_multicast: Whether to use partitioned multicast mode.

        Args:
            tma_op: TMA descriptor for the tile operation.
            dst: Destination tensor in shared memory.
            mem_barrier: Barrier for synchronizing the TMA operation.
            rank: Rank of this block within the cluster.
            coords: (row, col) coordinates of the tile to load.
            multicast_mask: Bitmask indicating which blocks receive the data.
        """
        alias tma_load_size = desc_layout.size()
        alias tma_rows = desc_layout.shape[0].value()

        @parameter
        if cluster_size > 1:
            # Multi-block cluster: Use multicast to share data across blocks

            @parameter
            if use_partitioned_multicast:
                # Partitioned multicast: Each block loads a portion of the tile
                # This is more efficient for large tiles as it distributes the load
                tma_op.async_multicast_load_partitioned[
                    tma_rows, tma_load_size
                ](
                    dst,
                    mem_barrier,
                    rank,
                    coords,
                    multicast_mask,
                )

            else:
                # Standard multicast: Only rank 0 loads and broadcasts to others
                # This is simpler but can create a bottleneck for large tiles
                if rank == 0:
                    tma_op.async_multicast_load(
                        dst,
                        mem_barrier,
                        coords,
                        multicast_mask,
                    )

        else:
            # Single block: Direct TMA copy without multicast overhead
            tma_op.async_copy(
                dst,
                mem_barrier,
                coords,
            )

    @staticmethod
    @always_inline("nodebug")
    fn load_tile[
        dtype: DType,
        src_layout: Layout,
        dst_layout: Layout, //,
        thread_layout: Layout,
        swizzle_mode: TensorMapSwizzle,
        vector_size: Int,
    ](
        src: LayoutTensor[
            dtype,
            src_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.GENERIC,
            *_, **_,
        ],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
            *_, **_,
        ],
        tile_idx_m: Int,
        tile_idx_n: Int,
    ):
        """Load a tile using cp.async instructions with bounds checking.

        This method uses cp.async instructions for asynchronous memory copies when
        TMA is not available or suitable. It handles unaligned memory access and
        performs bounds checking to ensure safe operation.

        Template Parameters:
            dtype: Data type of the tile elements.
            src_layout: Layout of the source tile in global memory.
            dst_layout: Layout of the destination tile in shared memory.
            thread_layout: Layout describing how threads map to data.
            swizzle_mode: Swizzling pattern for optimal shared memory access.
            vector_size: Number of elements to load per thread at once.

        Args:
            src: Source tensor in global memory.
            dst: Destination tensor in shared memory.
            tile_idx_m: Row index of the tile to load.
            tile_idx_n: Column index of the tile to load.
        """
        # Coalesce the destination layout for optimal memory access patterns
        alias coalesced_dst_layout = coalesce(dst_layout)
        alias BM = coalesced_dst_layout.shape[0].value()
        alias BN = coalesced_dst_layout.shape[1].value()

        # Extract the requested tile from global memory and vectorize it
        var a_gmem_tile = src.tile[BM, BN](
            tile_idx_m,
            tile_idx_n,
        ).vectorize[1, vector_size]()

        # Perform the async copy with bounds checking and swizzling
        Self.async_copy_with_bound_check[
            thread_layout,
            swizzle_mode,
        ](a_gmem_tile, dst.vectorize[1, vector_size]())

    @staticmethod
    @always_inline
    fn async_copy_with_bound_check[
        dtype: DType,
        src_layout: Layout,
        dst_layout: Layout, //,
        thread_layout: Layout,
        swizzle_mode: TensorMapSwizzle,
    ](
        src: LayoutTensor[
            dtype,
            src_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.GENERIC,
            *_, **_,
        ],
        dst: LayoutTensor[
            dtype,
            dst_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            *_, **_,
        ],
    ):
        """Helper function for cp.async with boundary checking.

        This method performs element-wise async copies with per-element boundary
        checking. Out-of-bounds accesses are automatically zero-filled, ensuring
        safe operation near matrix edges.

        The method also handles shared memory swizzling to avoid bank conflicts
        and maximize memory bandwidth utilization.

        Template Parameters:
            dtype: Data type of the elements.
            src_layout: Layout of the source tile.
            dst_layout: Layout of the destination tile.
            thread_layout: Thread arrangement for distributed copying.
            swizzle_mode: Swizzling pattern for bank conflict avoidance.

        Args:
            src: Source tensor fragment in global memory.
            dst: Destination tensor fragment in shared memory.
        """
        constrained[
            src.layout.rank() == 2, "Global memory tile must be rank 2."
        ]()

        constrained[
            src_layout.shape == dst_layout.shape,
            "Global memory tile must match source layout: "
            + String(src_layout)
            + " != "
            + String(dst_layout),
        ]()

        # Validate swizzle pattern alignment with tile dimensions
        alias src_shape1 = src.layout.shape[1].value()
        alias swizzle_bytes = swizzle_mode.bytes()
        constrained[
            src_shape1 * src.element_size * size_of[src.dtype]()
            == swizzle_bytes,
            String(
                "Global memory tile shape-1 ",
                src_shape1 * src.element_size,
                "must match swizzle bytes.",
                swizzle_bytes,
            ),
        ]()

        # Distribute work across threads according to thread_layout
        var src_frag = src.distribute[thread_layout](thread_idx.x)
        var dst_frag = dst.distribute[thread_layout](thread_idx.x)

        # Source matrix bounds for boundary checking
        alias src_stride0 = src.layout.stride[0].value()
        var src_bound0 = Int32(src.runtime_layout.shape.value[0])
        var src_bound1 = (
            Int32(src.runtime_layout.shape.value[1]) * dst.element_size
        )

        # Calculate base coordinates for this thread's destination fragment
        var dst_frag_offset = dst_frag.distance(dst.ptr)
        alias dst_stride0 = dst.layout.stride[0].value()
        var dst_frag_base_coord0 = Int32(dst_frag_offset // dst_stride0)
        var dst_frag_base_coord1 = Int32(dst_frag_offset % dst_stride0)

        # Create swizzle pattern to avoid shared memory bank conflicts
        alias swizzle = make_swizzle[
            8,
            Int(swizzle_bytes // size_of[dst.dtype]()),
            Int(simd_width_of[dst.dtype]()),
        ]()

        alias num_vecs = dst_frag.layout.size()

        # Process each vector element assigned to this thread
        @parameter
        for i in range(num_vecs):
            # Apply swizzling to the destination index to avoid bank conflicts
            alias dst_idx = dst_frag.layout(i)
            alias dst_idx_base = dst_idx % swizzle.size()
            alias dst_idx_diff = dst_idx - dst_idx_base
            var dst_swizzled_idx = Int32(
                swizzle(dst_frag_offset + dst_idx_base) + dst_idx_diff
            )
            var dst_ptr = dst.ptr + Int(dst_swizzled_idx)

            # Calculate the 2D coordinates for this element
            # TODO: we should be able to use idx2crd for this.
            alias dst_shifted_coord0 = dst_idx // dst_stride0
            alias dst_shifted_coord1 = dst_idx % dst_stride0
            var dst_coord0 = dst_shifted_coord0 + dst_frag_base_coord0
            var dst_coord1 = dst_shifted_coord1 + dst_frag_base_coord1

            alias size_bytes = dst.element_size * size_of[dst.dtype]()

            # Calculate source pointer based on 2D coordinates
            var src_ptr = (
                src.ptr.address_space_cast[AddressSpace.GLOBAL]()
                + dst_coord1
                + dst_coord0 * src_stride0
            )

            # Perform boundary check and issue appropriate async copy
            if dst_coord0 < src_bound0 and dst_coord1 < src_bound1:
                # In-bounds: copy actual data
                async_copy[
                    size_bytes,
                    bypass_L1_16B=False,
                    fill = Scalar[dst.dtype](0),
                ](src_ptr, dst_ptr, src_size=size_bytes)
            else:
                # Out-of-bounds: zero-fill by setting src_size=0
                async_copy[
                    size_bytes, bypass_L1_16B=False, fill = Scalar[dst.dtype](0)
                ](src_ptr, dst_ptr, src_size=0)
