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
"""RDNA-specific buffer implementations for Wave32 WMMA attention.

This module provides buffer management optimized for AMD RDNA consumer GPUs
(Radeon RX 7000/8000 series, gfx11xx/gfx12xx) using Wave32 execution.

Key differences from CDNA buffers:
- Wave size: 32 lanes (vs 64 for CDNA)
- MMA shape: 16x16x16 only (vs multiple shapes for CDNA)
- Fragment sizes: A/B = 16 elements (full K dimension), C/D = 8 elements per lane
- Wave-cooperative mode: lanes 0-15 provide unique data, lanes 16-31 replicate
- Memory access patterns optimized for 16-lane unique data distribution
"""

from collections import OptionalReg
from math import ceildiv, recip
from sys import simd_width_of

from gpu import barrier, lane_id
from gpu import warp_id as get_warp_id
from layout import Layout, LayoutTensor
from layout._utils import idx2crd, make_amd_buffer_resource
from layout.layout import blocked_product
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    copy_dram_to_local,
    copy_local_to_shared,
)
from layout.swizzle import Swizzle
from layout.tensor_core import TiledTensorCore
from memory.pointer import AddressSpace as BaseAddressSpace

from utils import IndexList

from .buffers import KVBuffer, RegisterBuffer, RegisterMMABuffer
from .utils import (
    LocalLayoutTensor,
    SharedLayoutTensor,
    pad,
)

# RDNA-specific constants
comptime RDNA_WARP_SIZE = 32
comptime RDNA_MMA_M = 16
comptime RDNA_MMA_N = 16
comptime RDNA_MMA_K = 16
# For RDNA 16x16x16 WMMA with Wave32:
# A/B fragments = 16 elements per lane (full K dimension)
#   - RDNA WMMA uses wave-cooperative mode
#   - Lanes 0-15 provide unique data, lanes 16-31 replicate
#   - Each lane holds 16 elements (full K dimension)
# C/D fragments = (M*N)/WARP = 256/32 = 8 elements per lane
comptime RDNA_AB_FRAG_SIZE = 16
comptime RDNA_CD_FRAG_SIZE = 8


@always_inline
fn get_rdna_fragment_layout() -> Layout:
    """Get the fragment layout for RDNA WMMA output fragments.

    RDNA uses Wave32 (32 lanes) with 16x16 output tiles, so each lane
    holds 256/32 = 8 fp32 accumulator elements.

    RDNA WMMA C/D register mapping: lane l, elem v -> D[row=v*2+l//16, col=l%16].
    All 8 elements in a lane share the same column (l%16) but have different
    interleaved rows.

    Returns:
        Layout for 1 seq row x 8 key columns per lane.
    """
    return Layout.row_major(1, 8)


@always_inline
fn get_rdna_warp_layout() -> Layout:
    """Get the warp thread layout for RDNA WMMA operations.

    For RDNA Wave32 with 16x16 tiles, using col_major(16, 2):
    - Lane i: row = i % 16, col = i // 16
    - Groups lanes (0, 16), (1, 17), ..., (15, 31) for cross-group reduction
    - Matches C/D mapping where lanes 0-15 hold even rows, 16-31 hold odd rows.

    Returns:
        Layout for col_major(16, 2) thread distribution.
    """
    return Layout.col_major(16, 2)


@always_inline
fn get_rdna_warp_coords[BN: Int, WN: Int]() -> IndexList[2]:
    """Get warp coordinates for RDNA Wave32."""
    comptime num_warps_n = BN // WN
    var warp_row = get_warp_id() // UInt(num_warps_n)
    var warp_col = get_warp_id() % UInt(num_warps_n)
    return IndexList[2](Int(warp_row), Int(warp_col))


struct KBufferRDNA[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool,
    layout_int_type: DType,
    linear_idx_type: DType,
    //,
    tensor_core_mma: TiledTensorCore,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
](KVBuffer):
    """RDNA-specific K buffer for Wave32 WMMA attention."""

    comptime _dtype = Self.dtype
    comptime _num_stages = Self.num_stages
    comptime MMA_M = RDNA_MMA_M
    comptime MMA_N = RDNA_MMA_N
    comptime MMA_K = RDNA_MMA_K
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_mmas = ceildiv(Self.WN, Self.MMA_N)

    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    comptime simd_width = simd_width_of[Self.dtype]()

    comptime num_repeats = Self.BK // Self.simd_width

    # Shared memory layout for RDNA
    comptime base_layout = Layout.row_major(Self.BN, Self.simd_width)
    comptime tiler_layout = Layout.row_major(1, Self.num_repeats)
    comptime smem_layout = blocked_product(
        Self.base_layout,
        Self.tiler_layout,
        coalesce_output=True,
    )

    # Thread layout for Wave32 - adjusted for 32-thread waves
    # With num_threads=128 and 4 warps of 32 threads each
    comptime thread_layout = Layout.row_major(Self.num_threads // 4, 4)

    comptime LoadTileType = LocalLayoutTensor[
        Self.dtype,
        Layout.row_major(
            Self.num_stages * Self.num_mmas * Self.num_k_tiles,
            Self.simd_width,
        ),
    ]
    var load_tile: Self.LoadTileType

    # MMA tile must have B fragment size = 16 for RDNA WMMA
    comptime mma_tile_layout = Layout.row_major(
        Self.num_mmas, RDNA_AB_FRAG_SIZE
    )

    comptime MMATileType = LocalLayoutTensor[
        Self.dtype,
        Self.mma_tile_layout,
    ]
    var mma_tile: Self.MMATileType

    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    comptime SharedIterType = LayoutTensorIter[
        Self.dtype,
        Self.smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    comptime SharedTileType = Self.SharedIterType.LayoutTensorType
    comptime SharedWarpTileType = Self.SharedTileType.TileType[
        Self.wtile_dim0, Self.wtile_dim1
    ]

    var bounds: Int
    var load_tile_id: Int

    comptime GlobalTensorType = LayoutTensor[
        Self.dtype,
        Self.layout,
        Self.origin,
        address_space = Self.address_space,
        alignment = Self.alignment,
        masked = Self.masked,
        layout_int_type = Self.layout_int_type,
        linear_idx_type = Self.linear_idx_type,
    ]

    comptime GlobalTiledIteratorType = Self.GlobalTensorType.TiledIteratorType[
        Self.BN,
        Self.BK,
        axis=1,
    ]

    var global_iterator: Self.GlobalTiledIteratorType

    @always_inline
    fn __init__(
        out self,
        global_tile: Self.GlobalTensorType,
        num_b_rows: OptionalReg[Int],
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        self.load_tile = type_of(self.load_tile).stack_allocation()
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_iter = type_of(self.smem_iter)(shared_ptr, 0)
        comptime stride = Self.GlobalTiledIteratorType.layout.stride[0].value()
        self.bounds = num_b_rows.value() * stride if num_b_rows else Int.MAX
        self.global_iterator = global_tile.tiled_iterator[
            Self.BN,
            Self.BK,
            axis=1,
        ](0, 0)
        self.load_tile_id = 0

    @always_inline
    @staticmethod
    fn get_dtype() -> DType:
        return Self._dtype

    @always_inline
    fn load_from_dram(
        mut self,
    ):
        copy_dram_to_local[src_thread_layout = Self.thread_layout,](
            self.load_tile.split[Self.num_stages]()[
                self.load_tile_id
            ].vectorize[1, Self.simd_width](),
            self.global_iterator,
            UInt32(self.bounds),
        )
        self.global_iterator._incr()
        self.load_tile_id = (self.load_tile_id + 1) % Self.num_stages

    @always_inline
    fn get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    fn copy_to_shared[
        tile_id: Int = 0
    ](self,):
        var smem_tile = self.smem_iter.next_unsafe(0)[]
        var load_tile_slice = self.load_tile.split[Self.num_stages]()[tile_id]

        copy_local_to_shared[
            thread_layout = Self.thread_layout,
            swizzle = Self.swizzle,
            row_major=True,
        ](
            smem_tile.vectorize[1, Self.simd_width](),
            load_tile_slice.vectorize[1, Self.simd_width](),
        )

    @always_inline
    fn load_from_shared[
        k_mma: Int,
    ](self):
        var warp_col = get_rdna_warp_coords[Self.BN, Self.WN]()[1]
        var smem_tile = self.smem_iter.next_unsafe(0)[]

        var warp_tile = smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
            warp_col, 0
        )

        # RDNA WMMA A register: a_frag[v] = A[lane%16, v]
        # (lane selects the ROW, element selects the COLUMN)
        # K goes to hardware A (via swap_a_b). We compute P^T = K * Q^T.
        # Hardware A maps: lane = key (row of K), element = depth (column of K).
        # So: a_frag[v] = K[key=lane, depth=v].
        var lane = Int(lane_id() % UInt(16))

        @parameter
        for m in range(Self.num_mmas):

            @parameter
            for k in range(RDNA_AB_FRAG_SIZE):
                self.mma_tile[m, k] = warp_tile[
                    m * Self.MMA_N + lane, k_mma * Self.MMA_K + k
                ]


struct VBufferRDNA[
    dtype: DType,
    layout: Layout,
    address_space: BaseAddressSpace,
    alignment: Int,
    origin: Origin,
    masked: Bool,
    layout_int_type: DType,
    linear_idx_type: DType,
    //,
    tensor_core_mma: TiledTensorCore,
    BN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    num_warps_n: Int = 1,
](KVBuffer):
    """RDNA-specific V buffer with transpose loads for Wave32 WMMA."""

    comptime _dtype = Self.dtype
    comptime _num_stages = Self.num_stages
    comptime simd_width = simd_width_of[Self.dtype]()
    comptime num_repeats = Self.BK // Self.simd_width

    # V Buffer shared memory layout with padding for bank conflict avoidance
    comptime base_layout = Layout.row_major(
        Self.pad[Self.depth](),
        Self.simd_width,
    )
    comptime tiler_layout = Layout.row_major(1, Self.num_repeats)
    comptime smem_layout = blocked_product(
        Self.base_layout,
        Self.tiler_layout,
        coalesce_output=True,
    )

    comptime MMA_M = RDNA_MMA_M
    comptime MMA_K = RDNA_MMA_K
    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    # This is the total number of depth tiles across all warps
    comptime num_depth_tiles = Self.depth // Self.MMA_M
    # This is the number of depth tiles for THIS warp's output portion
    comptime warp_depth_tiles = Self.depth // Self.num_warps_n // Self.MMA_M

    comptime depth_tile_size = min(Self.depth, 128)

    # RDNA-adjusted load width and loads per thread
    comptime load_width = 4 if Self.depth == 64 else Self.simd_width

    # For RDNA with Wave32: adjust loads_per_thread calculation
    comptime loads_per_thread_per_depth_tile = (
        Self.depth_tile_size * Self.BK
    ) // (Self.load_width * Self.num_threads)

    comptime LoadTileType = LocalLayoutTensor[
        Self.dtype,
        Layout.row_major(
            (
                Self.loads_per_thread_per_depth_tile
                * (Self.depth // Self.depth_tile_size)
            )
            * Self.num_stages,
            Self.load_width,
        ),
    ]

    var load_tile: Self.LoadTileType

    # MMA tile for RDNA: each warp processes warp_depth_tiles of the output
    comptime mma_tile_layout = Layout.row_major(
        Self.warp_depth_tiles, RDNA_AB_FRAG_SIZE
    )

    comptime MMATileType = LocalLayoutTensor[
        Self.dtype,
        Self.mma_tile_layout,
    ]

    var mma_tile: Self.MMATileType

    comptime SharedIterType = LayoutTensorIter[
        Self.dtype,
        Self.smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        circular=True,
    ]

    var smem_iter: Self.SharedIterType

    comptime SharedTileType = Self.SharedIterType.LayoutTensorType

    comptime GlobalTensorType = LayoutTensor[
        Self.dtype,
        Self.layout,
        Self.origin,
        address_space = Self.address_space,
        alignment = Self.alignment,
        masked = Self.masked,
        layout_int_type = Self.layout_int_type,
        linear_idx_type = Self.linear_idx_type,
    ]

    comptime GlobalTiledIteratorType = Self.GlobalTensorType.TiledIteratorType[
        Self.BK,
        Self.depth,
        axis=0,
    ]

    var global_iterator: Self.GlobalTiledIteratorType
    var global_base_tile: Self.GlobalTensorType
    var current_stage: Int
    var remaining_rows: Int

    @always_inline
    fn __init__(
        out self,
        global_tile: Self.GlobalTensorType,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
        total_rows: OptionalReg[Int] = None,
    ):
        constrained[
            Self.depth in (64, 128, 256), "depth must be 64, 128, or 256"
        ]()

        self.global_base_tile = global_tile
        self.global_iterator = global_tile.tiled_iterator[
            Self.BK, Self.depth, axis=0
        ](0, 0)

        self.load_tile = type_of(self.load_tile).stack_allocation()
        self.mma_tile = type_of(self.mma_tile).stack_allocation()
        self.smem_iter = type_of(self.smem_iter)(shared_ptr, 0)
        self.current_stage = 0
        self.remaining_rows = total_rows.value() if total_rows else Int.MAX

    @always_inline
    @staticmethod
    fn get_dtype() -> DType:
        return Self._dtype

    @always_inline
    @staticmethod
    fn pad[dim: Int]() -> Int:
        return pad[Self.dtype, Self.depth, dim]()

    @always_inline
    fn load_from_dram(
        mut self,
    ):
        """Load V tile from global memory using Wave32-aware pattern.

        Thread decomposition adapts to the number of warps:
        - threads_per_row = WARP_SIZE / rows_per_warp
        - Each thread loads depth_per_thread elements from one row
        - All BK rows are covered across all warps
        """
        var global_tile = self.global_iterator[]
        var warp_id = get_warp_id()

        var load_tile = self.load_tile.split[Self.num_stages]()[
            self.current_stage
        ]

        # Adaptive thread decomposition: distribute threads across BK rows
        # and depth columns based on actual warp count.
        comptime num_warps = Self.num_threads // RDNA_WARP_SIZE
        comptime rows_per_warp = Self.BK // num_warps
        comptime threads_per_row = RDNA_WARP_SIZE // rows_per_warp
        comptime depth_per_thread = Self.depth_tile_size // threads_per_row

        var lane = lane_id()
        var thread_row = Int(lane) // threads_per_row
        var thread_col = Int(lane) % threads_per_row

        var src_row = Int(warp_id) * rows_per_warp + thread_row
        var stride = global_tile.stride[0]()

        var tile_valid_rows = min(Self.BK, self.remaining_rows)

        @parameter
        for depth_idx in range(Self.depth // Self.depth_tile_size):

            @parameter
            for i in range(Self.loads_per_thread_per_depth_tile):
                var dst_idx = (
                    i + depth_idx * Self.loads_per_thread_per_depth_tile
                )
                if src_row < tile_valid_rows:
                    var src_col = (
                        thread_col * depth_per_thread + i * Self.load_width
                    )
                    src_col += depth_idx * Self.depth_tile_size

                    var offset = src_row * stride + src_col
                    var data = (global_tile.ptr + offset).load[
                        width = Self.load_width
                    ]()

                    @parameter
                    for j in range(Self.load_width):
                        load_tile[dst_idx, j] = data[j]
                else:

                    @parameter
                    for j in range(Self.load_width):
                        load_tile[dst_idx, j] = 0

        self.remaining_rows -= Self.BK
        self.current_stage = (self.current_stage + 1) % Self.num_stages
        self.global_iterator._incr()

    @always_inline
    fn get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    fn copy_to_shared[
        tile_id: Int = 0
    ](self,):
        """Copy V tile to shared memory with transpose for RDNA.

        V is stored transposed in shared memory: (depth, BK).
        Thread decomposition matches load_from_dram so each thread writes
        the data it loaded to the correct transposed position.
        """
        var warp_id = get_warp_id()
        var lane = lane_id()

        # Match the adaptive thread decomposition from load_from_dram
        comptime num_warps = Self.num_threads // RDNA_WARP_SIZE
        comptime rows_per_warp = Self.BK // num_warps
        comptime threads_per_row = RDNA_WARP_SIZE // rows_per_warp
        comptime depth_per_thread = Self.depth_tile_size // threads_per_row

        var thread_row = Int(lane) // threads_per_row
        var thread_col = Int(lane) % threads_per_row

        var smem_iter_tensor = self.smem_iter.next_unsafe(0)[]
        var load_tile = self.load_tile.split[Self.num_stages]()[tile_id]

        # In shared memory, V is transposed: smem[depth_pos, seq_pos]
        # smem_col_abs = V's row index (sequence position in BK dimension)
        var smem_col_abs = Int(warp_id) * rows_per_warp + thread_row
        var smem_chunk = smem_col_abs // Self.simd_width
        var smem_col = smem_col_abs % Self.simd_width

        @parameter
        for depth_idx in range(Self.depth // Self.depth_tile_size):
            var smem_tile = smem_iter_tensor.tile[
                Self.pad[Self.depth](),
                Self.simd_width,
            ](0, smem_chunk).tile[
                Self.pad[Self.depth_tile_size](),
                Self.simd_width,
            ](
                depth_idx, 0
            )

            @parameter
            for i in range(Self.loads_per_thread_per_depth_tile):
                var smem_row_base = (
                    thread_col * depth_per_thread + i * Self.load_width
                )

                @parameter
                for j in range(Self.load_width):
                    var smem_row = smem_row_base + j
                    var val = load_tile[
                        i + depth_idx * Self.loads_per_thread_per_depth_tile, j
                    ]
                    smem_tile[smem_row, smem_col] = val

    @always_inline
    fn load_from_shared[
        k_mma: Int,
    ](self):
        """Load MMA fragments from shared memory for RDNA Wave32.

        RDNA WMMA A register: a_frag[v] = A[lane%16, v]
        (lane selects ROW, element selects COLUMN).
        V goes to hardware A (via swap_a_b). For PV: D = V * P.
        Hardware A maps: lane = depth (row of V^T), element = key (column of V^T).
        So: a_frag[v] = V^T[depth=lane, key=v].
        V^T is stored in shared memory as [depth, key] with key split into
        blocks of simd_width=8: block0 has keys 0..7, block1 has keys 8..15.
        """
        var smem_iter_tensor = self.smem_iter.next_unsafe(0)[]

        var lane = lane_id() % UInt(16)

        # Each warp reads its portion of the depth dimension
        var warp_n_idx = Int(get_warp_id()) % Self.num_warps_n
        var depth_offset = warp_n_idx * Self.warp_depth_tiles * Self.MMA_M

        @parameter
        for depth_idx in range(Self.warp_depth_tiles):
            var smem_block0 = smem_iter_tensor.tile[
                Self.pad[Self.depth](), Self.simd_width
            ](0, k_mma * 2)
            var smem_block1 = smem_iter_tensor.tile[
                Self.pad[Self.depth](), Self.simd_width
            ](0, k_mma * 2 + 1)

            # Compute the logical depth position, then remap to the padded
            # shared memory row.  copy_to_shared writes in sub-tiles of
            # pad(depth_tile_size) rows each, so positions in the second
            # (and subsequent) depth tiles are offset by the padding.
            var global_depth_pos = (
                depth_offset + depth_idx * Self.MMA_M + Int(lane)
            )
            var dtile_idx = global_depth_pos // Self.depth_tile_size
            var pos_in_dtile = global_depth_pos % Self.depth_tile_size
            var depth_row = (
                dtile_idx * Self.pad[Self.depth_tile_size]() + pos_in_dtile
            )

            @parameter
            for j in range(RDNA_AB_FRAG_SIZE):
                # Element j selects key position (column of hardware A)
                # Keys 0..simd_width-1 are in block0, simd_width..15 in block1
                @parameter
                if j < Self.simd_width:
                    self.mma_tile[depth_idx, j] = smem_block0[depth_row, j]
                else:
                    self.mma_tile[depth_idx, j] = smem_block1[
                        depth_row, j - Self.simd_width
                    ]


struct QRegisterBufferRDNA[
    dtype: DType,
    mma_shape: IndexList[3],
    k_group_size: Int,
    WM: Int,
    WN: Int,
    BN: Int,
    BK: Int,
    depth: Int,
    thread_layout: Layout,
](RegisterMMABuffer):
    """RDNA-specific Q register buffer for Wave32 WMMA attention."""

    comptime reg_dtype = Self.dtype
    comptime mma_dtype = Self.dtype
    comptime simd_width = simd_width_of[Self.dtype]()
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(Self.WM, Self.MMA_M)
    comptime num_k_tiles = ceildiv(Self.BK, Self.MMA_K * Self.k_group_size)

    comptime rdna_frag_size = RDNA_AB_FRAG_SIZE

    comptime num_tiles = Self.depth // Self.BK
    comptime reg_tile_layout = Layout.row_major(
        Self.num_mmas * Self.num_k_tiles * Self.num_tiles, Self.rdna_frag_size
    )
    comptime RegisterTileType = LocalLayoutTensor[
        Self.dtype,
        Self.reg_tile_layout,
    ]

    comptime MMATileType = Self.RegisterTileType.SplitElementType[
        Self.num_tiles
    ].SplitElementType[Self.num_k_tiles]
    comptime mma_tile_layout = Self.MMATileType.layout

    var reg_tile: Self.RegisterTileType

    comptime TiledIteratorType = Self.RegisterTileType.TiledIteratorType[
        Self.num_mmas * Self.num_k_tiles, Self.rdna_frag_size, axis=0
    ]

    @staticmethod
    @always_inline
    fn get_dtype() -> DType:
        return Self.reg_dtype

    @always_inline
    fn __init__(out self, tensor: LayoutTensor[Self.dtype, **_]):
        self.reg_tile = type_of(self.reg_tile).stack_allocation()

        var warp_row = get_rdna_warp_coords[Self.BN, Self.WN]()[0]

        var warp_base_row = warp_row * Self.WM
        var warp_tensor = tensor.tile[Self.WM, Self.depth](warp_row, 0)

        var lane = lane_id()
        # RDNA WMMA B register: b_frag[v] = B[v, lane%16]
        # With swap_a_b, Q goes to hardware B. We need B = Q^T[depth, seq].
        # B[k=element_v, n=lane%16] → element = depth (row of Q^T), lane = seq (col of Q^T).
        # So: b_frag[v] = Q^T[v, lane%16] = Q[lane%16, v] = Q[seq=lane, depth=v].
        # Each lane loads one row of Q (its seq position) with elements = depth values.
        var row_in_first_mma = Int(lane % UInt(16))

        var warp_offset = Int(warp_base_row) * tensor.stride[0]()
        var valid_rows = Int(
            min(Int32(Self.WM), Int32(tensor.dim[0]()) - Int32(warp_base_row))
        )

        @parameter
        for tile_idx in range(Self.num_tiles):
            var depth_offset = tile_idx * Self.BK

            @parameter
            for k_mma in range(Self.num_k_tiles):
                var k_offset = depth_offset + k_mma * Self.MMA_K

                @parameter
                for m_mma in range(Self.num_mmas):
                    var row = row_in_first_mma + m_mma * 16
                    var row_offset = row * tensor.stride[0]()

                    var frag_idx = (
                        tile_idx * Self.num_k_tiles * Self.num_mmas
                        + k_mma * Self.num_mmas
                        + m_mma
                    )

                    if row < valid_rows:
                        var offset0 = warp_offset + row_offset + k_offset
                        var data0 = (tensor.ptr + offset0).load[
                            width = Self.simd_width
                        ]()

                        var offset1 = offset0 + Self.simd_width
                        var data1 = (tensor.ptr + offset1).load[
                            width = Self.simd_width
                        ]()

                        @parameter
                        for j in range(Self.simd_width):
                            self.reg_tile[frag_idx, j] = data0[j]
                            self.reg_tile[
                                frag_idx, Self.simd_width + j
                            ] = data1[j]
                    else:

                        @parameter
                        for j in range(Self.simd_width):
                            self.reg_tile[frag_idx, j] = 0
                            self.reg_tile[frag_idx, Self.simd_width + j] = 0

    @always_inline
    fn get_iter(self) -> Self.TiledIteratorType:
        return self.reg_tile.tiled_iterator[
            Self.num_mmas * Self.num_k_tiles, Self.rdna_frag_size, axis=0
        ]()

    @always_inline
    fn get_mma_tile[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        return self.reg_tile.split[Self.num_tiles]()[tile_idx].split[
            Self.num_k_tiles
        ]()[k_idx]

    @always_inline
    fn get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return self.reg_tile

    @always_inline
    fn zero(self):
        _ = self.reg_tile.fill(0)


struct OutputRegisterBufferRDNA[
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
](RegisterBuffer):
    """RDNA-specific output register buffer for Wave32 WMMA."""

    comptime reg_dtype = Self.dtype
    comptime output_frag_size = RDNA_CD_FRAG_SIZE

    comptime reg_tile_layout = Layout.row_major(
        Self.num_n_mmas * Self.num_m_mmas, Self.output_frag_size
    )
    comptime RegisterTileType = LocalLayoutTensor[
        Self.dtype,
        Self.reg_tile_layout,
    ]

    var reg_tile: Self.RegisterTileType

    @always_inline
    fn __init__(out self):
        self.reg_tile = Self.RegisterTileType.stack_allocation()

    @staticmethod
    @always_inline
    fn get_dtype() -> DType:
        return Self.reg_dtype

    @always_inline
    fn vectorize(
        self,
    ) -> Self.RegisterTileType.VectorizedType[1, Self.output_frag_size]:
        return self.reg_tile.vectorize[1, Self.output_frag_size]()

    @always_inline
    fn apply_softmax_denominator(self, rowsum: LayoutTensor[Self.dtype, **_]):
        """Apply softmax denominator normalization to output accumulator."""

        @parameter
        for m_mma in range(Self.num_m_mmas):
            var rowsum_inv = recip(rowsum[m_mma, 0])

            @parameter
            for n_mma in range(Self.num_n_mmas):

                @parameter
                for i in range(Self.output_frag_size):
                    self.reg_tile[n_mma * Self.num_m_mmas + m_mma, i] *= rebind[
                        Self.RegisterTileType.element_type
                    ](rowsum_inv)

    @always_inline
    fn zero(self):
        _ = self.reg_tile.fill(0)

    @always_inline
    fn get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return self.reg_tile


struct PRegisterBufferRDNA[
    accum_type_: DType,
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    mma_shape: IndexList[3],
    k_group_size: Int,
](RegisterMMABuffer):
    """RDNA-specific P register buffer for Wave32 WMMA attention."""

    comptime reg_dtype = Self.accum_type_
    comptime mma_dtype = Self.dtype
    comptime output_frag_size = RDNA_CD_FRAG_SIZE
    comptime mma_frag_size = RDNA_AB_FRAG_SIZE

    comptime mma_tile_layout = Layout.row_major(
        Self.num_m_mmas, Self.mma_frag_size
    )
    comptime reg_tile_layout = Layout.row_major(
        Self.num_n_mmas * Self.num_m_mmas, Self.output_frag_size
    )

    comptime RegisterTileType = LocalLayoutTensor[
        Self.accum_type_,
        Self.reg_tile_layout,
    ]

    comptime MMATileType = LocalLayoutTensor[
        Self.mma_dtype,
        Self.mma_tile_layout,
    ]

    var reg_tile: Self.RegisterTileType

    comptime chunk_shared_memory_layout = Layout.row_major(Self.BK, Self.BM)

    comptime ChunkSharedMemoryTileType = SharedLayoutTensor[
        Self.dtype,
        Self.chunk_shared_memory_layout,
    ]

    var shared_memory_ptr: UnsafePointer[
        Scalar[Self.dtype], MutAnyOrigin, address_space = AddressSpace.SHARED
    ]

    @always_inline
    fn __init__(
        out self,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ],
    ):
        self.reg_tile = Self.RegisterTileType.stack_allocation()
        self.shared_memory_ptr = shared_ptr

    @always_inline
    fn get_mma_tile[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        """Get MMA tile by loading from shared memory.

        RDNA WMMA B register: b_frag[v] = B[v, lane%16].
        With swap_a_b, P goes to hardware B. We need B = P^T[key, seq].
        B[k=element_v, n=lane%16] → element = key (row), lane = seq (col).
        So: b_frag[v] = P^T[key=v, seq=lane] = P_shared[key=v, seq=lane].
        """
        var mma_reg_tile = Self.MMATileType.stack_allocation()
        var warp_row = get_rdna_warp_coords[Self.BN, Self.WN]()[0]

        var warp_base_seq = Int(warp_row) * Self.WM
        var k_key_base = k_idx * Self.mma_shape[2]

        var lane = Int(lane_id() % UInt(16))

        @parameter
        for m_mma in range(Self.num_m_mmas):
            var seq = warp_base_seq + m_mma * Self.mma_shape[0] + lane

            @parameter
            for k in range(Self.mma_shape[2]):
                var key = k_key_base + k
                var smem_offset = key * Self.BM + seq
                mma_reg_tile[m_mma, k] = self.shared_memory_ptr[smem_offset]

        return mma_reg_tile

    @staticmethod
    @always_inline
    fn get_dtype() -> DType:
        return Self.mma_dtype

    @always_inline
    fn vectorize(
        self,
    ) -> Self.RegisterTileType.VectorizedType[1, Self.output_frag_size]:
        return self.reg_tile.vectorize[1, Self.output_frag_size]()

    @always_inline
    fn zero(self):
        _ = self.reg_tile.fill(0)

    @always_inline
    fn get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return self.reg_tile

    @always_inline
    fn copy_to_shared[chunk_idx: Int](self):
        """Copy one BK chunk of P register tile to shared memory using RDNA layouts.

        Each chunk corresponds to BK=32 keys. With 2 warps each handling WN=32
        keys, chunk 0 = warp 0's data, chunk 1 = warp 1's data. Only the owning
        warp writes, using warp-local tile indices to avoid OOB register access.
        """
        comptime reg_per_thread = Self.output_frag_size
        comptime num_warps_n = Self.BN // Self.WN
        comptime chunks_per_warp = Self.WN // Self.BK
        comptime owning_warp = chunk_idx // chunks_per_warp

        var warp_coords = get_rdna_warp_coords[Self.BN, Self.WN]()
        var warp_row = warp_coords[0]
        var warp_n_idx = warp_coords[1]

        # Only the warp that owns this chunk writes P data
        if Int(warp_n_idx) != owning_warp:
            return

        # Map to warp-local tile indices (always 0-based, avoids OOB)
        comptime local_chunk = chunk_idx % chunks_per_warp
        comptime num_n_mmas_per_bk = Self.BK // Self.mma_shape[1]
        comptime chunk_n_start = local_chunk * num_n_mmas_per_bk

        var warp_base_seq = Int(warp_row) * Self.WM

        var lane = Int(lane_id())
        var lane_seq_offset = lane % 16
        var lane_key_group = lane // 16

        var reg_ptr = self.reg_tile.ptr

        @parameter
        for m_mma in range(Self.num_m_mmas):

            @parameter
            for n_mma_local in range(num_n_mmas_per_bk):
                comptime n_mma = chunk_n_start + n_mma_local

                var mma_seq_base = warp_base_seq + m_mma * Self.mma_shape[0]
                var mma_key_base = n_mma_local * Self.mma_shape[1]

                comptime p_reg_tile_idx = n_mma * Self.num_m_mmas + m_mma
                var reg_base_offset = p_reg_tile_idx * Self.output_frag_size

                var global_seq = mma_seq_base + lane_seq_offset

                @parameter
                for elem in range(reg_per_thread):
                    # RDNA WMMA C/D mapping: lane l, elem v → D[row=v*2+l//16, col=l%16]
                    # P^T[key, seq]: key = elem*2 + lane_key_group (interleaved)
                    var key_in_mma = elem * 2 + lane_key_group
                    var global_key = mma_key_base + key_in_mma

                    var smem_offset = global_key * Self.BM + global_seq
                    var fp32_val = reg_ptr[reg_base_offset + elem]
                    self.shared_memory_ptr[smem_offset] = fp32_val.cast[
                        Self.dtype
                    ]()
