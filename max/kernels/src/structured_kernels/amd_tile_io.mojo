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
"""TileTensor data movement and AMD GPU hardware operations.

Provides reusable building blocks for TileTensor-based DMA, LDS reads,
and MMA operand loads on AMD CDNA GPUs (gfx950+).

Functions:
    ds_read_tr16_b64_row    - 4x16 transposed LDS read (raw rocdl intrinsic).
    ds_read_tr16_b64_warp   - Warp-level transposed LDS read.
    tt_load_b_tr            - Transposed B operand load (split into halves).
    tt_load_b_tile          - Single MMA tile load from SMEM with swizzle.
    tt_load_b               - Full B operand load from SMEM warp tile.
    tt_copy_dram_to_sram_lds - Buffer-load-to-LDS DMA with TileTensor.

Structs:
    LdsTileLoader - AMD buffer-load-to-LDS with pre-built buffer resource.
    RegTileLoader - AMD buffer-resource load from DRAM to registers.
    RegTileWriter - AMD buffer-resource store from registers to DRAM.

Free functions:
    tt_copy_local_to_shared - Register-to-SMEM copy (TileTensor, col-major src).
"""

from std.sys import simd_width_of, size_of
from std.gpu import lane_id, thread_idx, WARP_SIZE
from std.gpu._utils import to_i32, to_i64
from std.gpu.intrinsics import AMDBufferResource
from std.memory import AddressSpace
from std.memory.unsafe import bitcast
from std.math.uutils import umod, ufloordiv
from std.sys.intrinsics import readfirstlane
from std.utils import IndexList
from layout import TileTensor
from layout._utils import make_amd_buffer_resource
from layout.tile_layout import Layout, row_major, col_major
from layout.swizzle import Swizzle
from std.itertools import product


comptime _alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
comptime _no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`


# ===----------------------------------------------------------------------=== #
# LDS transposed reads
# ===----------------------------------------------------------------------=== #


@always_inline
def ds_read_tr16_b64_row(
    tile: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> SIMD[tile.dtype, 4]:
    """4x16 transposed LDS read via rocdl.ds.read.tr16.b64.

    Each 16-lane "row" loads a 4x16 tile, with per-lane exchange so each
    lane gets a column of the tile as SIMD[dtype, 4].

    Args:
        tile: A 4x16 TileTensor in shared memory (2-byte element type).

    Returns:
        A SIMD[dtype, 4] vector with one column of the transposed tile.
    """
    comptime assert size_of[tile.dtype]() == 2
    comptime assert type_of(tile).static_shape[0] == 4
    comptime assert type_of(tile).static_shape[1] == 16

    comptime thread_layout = row_major[4, 4]()
    var lane_in_row = umod(lane_id(), 16)
    var dist_result = tile.vectorize[1, 4]().distribute_with_offset[
        thread_layout
    ](lane_in_row)
    var offset = dist_result[2]
    var ptr = tile.ptr + offset

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](ptr)

    var llvm_res = __mlir_op.`rocdl.ds.read.tr16.b64`[
        _type=__mlir_type.`vector<4 x bf16>`,
        noalias_scopes=_alias_scope_attr,
        alias_scopes=_no_alias_scope_attr,
    ](shared_ptr3)

    return rebind[SIMD[tile.dtype, 4]](
        __mlir_op.`pop.cast_from_builtin`[_type=SIMD[tile.dtype, 4]._mlir_type](
            llvm_res
        )
    )


@always_inline
def ds_read_tr16_b64_warp[
    mma_shape: IndexList[3],
](
    tile: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> SIMD[
    tile.dtype, 4
]:
    """Warp-level transposed LDS read distributing across 16-lane rows.

    For 32x32x16 MMA: 2x2 row distribution over 8x32 tile.
    For 16x16x32 MMA: 4x1 row distribution over 16x16 tile.

    Parameters:
        mma_shape: MMA instruction shape (M, N, K).

    Args:
        tile: A TileTensor in shared memory sized for the MMA shape.

    Returns:
        A SIMD[dtype, 4] vector with transposed data for one lane.
    """
    comptime row_dim0 = 2 if mma_shape[0] == 32 else 4
    comptime row_dim1 = 2 if mma_shape[0] == 32 else 1

    comptime assert tile.dtype == DType.bfloat16
    comptime assert type_of(tile).static_shape[0] == row_dim0 * 4
    comptime assert type_of(tile).static_shape[1] == row_dim1 * 16

    var row_idx = ufloordiv(lane_id(), 16)
    var coord0 = row_idx // row_dim1
    var coord1 = row_idx % row_dim1
    var shared_b_tile = tile.tile[4, 16](coord0, coord1)
    return ds_read_tr16_b64_row(shared_b_tile)


# ===----------------------------------------------------------------------=== #
# MMA operand loads from SMEM
# ===----------------------------------------------------------------------=== #


@always_inline
def tt_load_b_tr[
    mma_shape: IndexList[3],
](
    tile: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> SIMD[
    tile.dtype, 8
]:
    """Transposed B operand load for double-rate MFMA shapes.

    Splits the tile along the K dimension into two halves and
    concatenates the results.

    Parameters:
        mma_shape: MMA instruction shape (M, N, K).

    Args:
        tile: A MMA_K x MMA_N TileTensor in shared memory.

    Returns:
        A SIMD[dtype, 8] vector with both halves concatenated.
    """
    comptime assert mma_shape in (
        IndexList[3](32, 32, 16),
        IndexList[3](16, 16, 32),
    )
    comptime assert tile.dtype == DType.bfloat16
    comptime MMA_K = mma_shape[2]
    comptime MMA_N = mma_shape[1]
    comptime half_k = MMA_K // 2
    comptime assert type_of(tile).static_shape[0] == MMA_K
    comptime assert type_of(tile).static_shape[1] == MMA_N

    var part_1 = ds_read_tr16_b64_warp[mma_shape](
        tile.tile[half_k, MMA_N](0, 0)
    )
    var part_2 = ds_read_tr16_b64_warp[mma_shape](
        tile.tile[half_k, MMA_N](1, 0)
    )
    return part_1.join(part_2)


@always_inline
def tt_load_b_tile[
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle],
    k_tile_idx: Int,
](
    src: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> SIMD[
    src.dtype, simd_width_of[src.dtype]()
]:
    """Single MMA tile load from SMEM with optional swizzle.

    Loads one MMA_M x MMA_K sub-tile at column k_tile_idx, distributes
    across lanes, applies swizzle, and reads via llvm.load.

    Parameters:
        mma_shape: MMA instruction shape (M, N, K).
        swizzle: Optional swizzle for bank conflict reduction.
        k_tile_idx: Column tile index within the warp tile.

    Args:
        src: An MMA_M-row TileTensor in shared memory.

    Returns:
        A SIMD vector with the loaded MMA fragment.
    """
    comptime MMA_M = mma_shape[0]
    comptime MMA_K = mma_shape[2]
    comptime assert type_of(src).static_shape[0] == MMA_M
    comptime simd_width = simd_width_of[src.dtype]()

    comptime assert mma_shape[0] == 32, "tt_load_b_tile only supports MMA_M=32"
    var sub_tile = src.tile[MMA_M, MMA_K](0, k_tile_idx)
    comptime thread_layout = col_major[32, 2]()

    var dist = sub_tile.vectorize[1, simd_width]().distribute[thread_layout](
        lane_id()
    )
    comptime idx_type = src.linear_idx_type
    var offset = Scalar[idx_type](Int(dist.ptr) - Int(src.ptr)) // Scalar[
        idx_type
    ](size_of[src.dtype]())

    comptime if swizzle:
        offset = swizzle.value()(
            offset // Scalar[idx_type](simd_width)
        ) * Scalar[idx_type](simd_width)

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](src.ptr + offset)

    var llvm_res = __mlir_op.`llvm.load`[
        _type=__mlir_type.`vector<8 x bf16>`,
        alignment=to_i64(16),
        noalias_scopes=_alias_scope_attr,
        alias_scopes=_no_alias_scope_attr,
    ](shared_ptr3)

    return rebind[SIMD[src.dtype, simd_width]](
        __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[src.dtype, simd_width]._mlir_type
        ](llvm_res)
    )


@always_inline
def tt_load_b[
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle],
    num_mmas: Int,
    simd_width: Int,
](
    src: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
) -> InlineArray[SIMD[src.dtype, simd_width], num_mmas]:
    """Full B operand load from a SMEM warp tile.

    Loads all MMA tiles from a WN x BK SMEM warp tile and returns them
    as an InlineArray of SIMD fragments (one per MMA tile).

    Parameters:
        mma_shape: MMA instruction shape (M, N, K).
        swizzle: Optional swizzle for bank conflict reduction.
        num_mmas: Number of MMA tiles to load.
        simd_width: SIMD vector width for the element type.

    Args:
        src: A WN x BK TileTensor in shared memory.

    Returns:
        An InlineArray of SIMD fragments, one per MMA tile.
    """
    comptime MMA_M = mma_shape[0]
    comptime MMA_K = mma_shape[2]
    comptime M = type_of(src).static_shape[0] // MMA_M
    comptime N = type_of(src).static_shape[1] // MMA_K

    var result = InlineArray[SIMD[src.dtype, simd_width], num_mmas](
        uninitialized=True
    )
    comptime for i in range(M):
        comptime for j in range(N):
            result[Int(i) + Int(j) * M] = rebind[SIMD[src.dtype, simd_width]](
                tt_load_b_tile[mma_shape, swizzle, Int(j)](
                    src.tile[MMA_M, type_of(src).static_shape[1]](Int(i), 0)
                )
            )
    return result


# ===----------------------------------------------------------------------=== #
# DRAM -> LDS DMA
# ===----------------------------------------------------------------------=== #


@always_inline
def tt_copy_dram_to_sram_lds[
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](
    dst: TileTensor[_, _, address_space=AddressSpace.SHARED, ...],
    src: TileTensor,
    lds_base_ptr: UInt32,
    bc: AMDBufferResource,
):
    """DMA from DRAM to LDS with TileTensor for both dst and src.

    Uses rocdl.raw.ptr.buffer.load.lds to transfer data directly from
    DRAM to LDS. Scalar offsets are relative to bc's base pointer.

    Parameters:
        swizzle: Optional swizzle for bank conflict reduction.

    Args:
        dst: Destination TileTensor in shared memory.
        src: Source TileTensor in global memory (warp sub-tile).
        lds_base_ptr: Base LDS pointer for the warp.
        bc: AMD buffer resource descriptor for the source.
    """
    comptime thread_layout = row_major[16, 4]()
    var worker_idx = lane_id()

    var dram_base = bc.get_base_ptr()

    comptime M = type_of(src).static_shape[0]
    comptime N = type_of(src).static_shape[1]
    comptime BM = 32
    comptime BN = 32
    comptime BM_SUB = 16

    comptime dst_stride0 = type_of(dst).static_stride[0]
    comptime dst_stride1 = type_of(dst).static_stride[1]
    comptime assert dst_stride1 == 1
    comptime assert dst_stride0 == 32

    comptime aux = 0

    var lds_ptr = lds_base_ptr

    comptime for n_tile, m_tile, m_sub_tile in product(
        range(N // BN), range(M // BM), range(BM // BM_SUB)
    ):
        var dst_partitions = dst.tile[BM, BN](m_tile, n_tile).tile[BM_SUB, BN](
            m_sub_tile, 0
        )
        var src_partitions = src.tile[BM, BN](m_tile, n_tile).tile[BM_SUB, BN](
            m_sub_tile, 0
        )
        var worker_idx_with_offset = worker_idx + m_sub_tile * WARP_SIZE
        var src_dist = src_partitions.vectorize[
            1, simd_width_of[src.dtype]()
        ]().distribute[thread_layout](
            umod(
                swizzle.value()(
                    worker_idx_with_offset
                ) if swizzle else worker_idx_with_offset,
                WARP_SIZE,
            )
        )
        comptime dtype = src.dtype
        var dst_ptr = dst_partitions.ptr.address_space_cast[
            AddressSpace.SHARED
        ]()

        var desc_ptr_ = UnsafePointer[
            Scalar[DType.bfloat16],
            MutAnyOrigin,
            address_space=AddressSpace.BUFFER_RESOURCE,
        ](_unsafe_null=())

        var ptr_to_ptr = UnsafePointer(to=desc_ptr_)
        var ptr_to_simd = UnsafePointer(to=bc.desc)
        ptr_to_ptr[0] = ptr_to_simd.bitcast[
            UnsafePointer[
                Scalar[DType.bfloat16],
                MutAnyOrigin,
                address_space=AddressSpace.BUFFER_RESOURCE,
            ]
        ]()[0]
        var desc_ptr_llvm = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_type.`!llvm.ptr<8>`
        ](desc_ptr_)

        var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_type.`!llvm.ptr<3>`
        ](dst_ptr)

        comptime num_bytes_per_lane = size_of[dtype]() * simd_width_of[dtype]()
        var vector_offset_bytes = Int(src_dist.ptr) - Int(src_partitions.ptr)
        var scalar_offset_bytes = Int(src_partitions.ptr) - dram_base

        __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
            alias_scopes=_alias_scope_attr,
            _type=None,
        ](
            desc_ptr_llvm,
            shared_ptr3,
            to_i32(Int32(num_bytes_per_lane)),
            to_i32(Int32(vector_offset_bytes)),
            to_i32(Int32(scalar_offset_bytes)),
            to_i32(0),
            to_i32(aux),
        )
        comptime num_bytes_per_warp = UInt32(
            thread_layout.size() * num_bytes_per_lane
        )
        lds_ptr += num_bytes_per_warp


# ===----------------------------------------------------------------------=== #
# Blocked SMEM navigation
# ===----------------------------------------------------------------------=== #


@always_inline
def smem_subtile[
    tile_rows: Int,
    tile_cols: Int,
    BN: Int,
    BK: Int,
    dtype: DType,
](
    smem_ptr: UnsafePointer[
        Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    tile_row: Int,
    tile_col: Int,
) -> TileTensor[
    dtype,
    type_of(row_major[tile_rows, tile_cols]()),
    MutAnyOrigin,
    address_space=AddressSpace.SHARED,
]:
    """Creates a flat TileTensor sub-view of a blocked SMEM layout.

    The blocked layout has num_repeats contiguous BN x BK blocks. This
    function computes the physical offset for a block-aligned tile and
    returns a row-major TileTensor view with strides (tile_cols, 1).

    Correct only when tile_cols == BK (tiles don't cross block boundaries
    in the column dimension).

    Parameters:
        tile_rows: Height of the sub-tile.
        tile_cols: Width of the sub-tile (must equal BK for block alignment).
        BN: Number of rows per block (full block height).
        BK: Number of columns per block (full block width).
        dtype: Element data type.

    Args:
        smem_ptr: Base pointer to the SMEM allocation.
        tile_row: Tile row index (0-based, in units of tile_rows).
        tile_col: Tile column index (0-based, in units of tile_cols).

    Returns:
        A TileTensor view into the specified sub-tile region.
    """
    comptime block_size = BN * BK
    var offset = tile_row * tile_rows * BK + tile_col * block_size
    return TileTensor[
        dtype,
        type_of(row_major[tile_rows, tile_cols]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ](smem_ptr + offset, row_major[tile_rows, tile_cols]())


@always_inline
def smem_mma_subtile[
    mma_rows: Int,
    mma_cols: Int,
    BN: Int,
    BK: Int,
    dtype: DType,
](
    smem_ptr: UnsafePointer[
        Scalar[dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    bk_tile: Int,
    k_sub: Int,
    mma_idx: Int,
) -> TileTensor[
    dtype,
    type_of(row_major[mma_rows, mma_cols]()),
    MutAnyOrigin,
    address_space=AddressSpace.SHARED,
]:
    """Creates a flat TileTensor for an MMA-sized sub-tile in blocked SMEM.

    Used by the non-transposed (V buffer) load_from_shared path. The V
    buffer's SMEM has shape (BN, depth) with blocked layout
    (num_repeats x BN x BK blocks). Each MMA tile is mma_rows x mma_cols
    within one block.

    Parameters:
        mma_rows: MMA tile height (e.g., MMA_K=16).
        mma_cols: MMA tile width (e.g., MMA_M=32).
        BN: Block height.
        BK: Block width.
        dtype: Element data type.

    Args:
        smem_ptr: Base pointer to the SMEM allocation for this buffer stage.
        bk_tile: Which BK-tall row group (0..depth/BK-1).
        k_sub: Which MMA_K sub-row within the BK group (0..BK/MMA_K-1).
        mma_idx: Linear MMA tile index across the full depth dimension.

    Returns:
        A TileTensor view into the MMA-sized sub-tile.
    """
    comptime block_size = BN * BK
    comptime tiles_per_block = BK // mma_cols
    var block_idx = mma_idx // tiles_per_block
    var col_in_block = (mma_idx % tiles_per_block) * mma_cols
    var offset = (
        bk_tile * BK * BK
        + k_sub * mma_rows * BK
        + block_idx * block_size
        + col_in_block
    )
    return TileTensor[
        dtype,
        type_of(row_major[mma_rows, mma_cols]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ](smem_ptr + offset, row_major[mma_rows, mma_cols]())


# ===----------------------------------------------------------------------=== #
# LdsTileLoader
# ===----------------------------------------------------------------------=== #


struct LdsTileLoader[
    dtype: DType,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](TrivialRegisterPassable):
    """AMD buffer-load-to-LDS tile loader with pre-built buffer resource.

    Constructs the AMDBufferResource once from a DRAM tile (which may
    carry RuntimeInt valid_rows for bounds clamping via MixedLayout).
    Each load() call reuses the descriptor — one shared bc per tile,
    zero per-warp overhead for buffer resource construction.

    The SRD bounds are computed by make_amd_buffer_resource, which calls
    _get_bounds on the TileTensor. For tiles with RuntimeInt dim[0],
    this computes (valid_rows-1)*stride0 + (dim1-1)*stride1 + 1. The
    hardware clamps OOB reads to zero.

    Parameters:
        dtype: Element data type.
        swizzle: Optional swizzle for bank conflict reduction.
    """

    var bc: AMDBufferResource
    """The 128-bit buffer resource descriptor for DRAM access."""

    @always_inline
    def __init__(out self, gmem_tile: TileTensor[Self.dtype, ...]):
        """Create a loader from a DRAM tile.

        The tile's layout carries the valid row count (via RuntimeInt
        dim[0] in MixedLayout). make_amd_buffer_resource reads that
        dimension to compute the SRD size.

        Args:
            gmem_tile: The full DRAM tile from KVCacheIterator.
        """
        self.bc = make_amd_buffer_resource(gmem_tile)

    @always_inline
    def load(
        self,
        dst: TileTensor[
            Self.dtype, _, _, address_space=AddressSpace.SHARED, ...
        ],
        src: TileTensor[Self.dtype, ...],
        lds_base_ptr: UInt32,
    ):
        """Load a warp sub-tile from DRAM to LDS.

        The src tile should be a warp-sized sub-tile of the original DRAM
        tile. Offsets are computed relative to the bc's base pointer, so
        the src pointer must be within the original tile's address range.

        Args:
            dst: Destination TileTensor in shared memory.
            src: Source TileTensor in global memory (warp sub-tile).
            lds_base_ptr: Base LDS pointer for the warp.
        """
        tt_copy_dram_to_sram_lds[Self.swizzle](dst, src, lds_base_ptr, self.bc)


# ===----------------------------------------------------------------------=== #
# RegTileLoader
# ===----------------------------------------------------------------------=== #


struct RegTileLoader[
    dtype: DType,
    thread_rows: Int,
    thread_cols: Int,
    num_threads: Int = thread_rows * thread_cols,
    warp_scope: Bool = False,
    col_major_threads: Bool = False,
](TrivialRegisterPassable):
    """AMD buffer-resource load from DRAM to registers.

    Pre-builds the AMDBufferResource from a DRAM TileTensor once.
    Each load() call distributes a source tile across threads and
    issues buffer_load intrinsics to fill a LOCAL register TileTensor.

    Mirrors RegTileWriter (which handles registers -> DRAM stores).
    Uses the same distribute_with_offset + AMDBufferResource pattern.

    The dst register tile uses col-major element ordering, matching
    the LayoutTensor copy_dram_to_local convention. This ensures
    compatibility with copy_local_to_shared (row_major=True path),
    which reads registers in the same col-major order.

    Parameters:
        dtype: Element data type.
        thread_rows: Number of rows in the thread distribution.
        thread_cols: Number of columns in the thread distribution.
        num_threads: Total threads; threads beyond thread_rows * thread_cols
            are idle.
        warp_scope: If True, uses lane_id() as worker index (warp scope).
            If False, uses thread_idx.x (block scope).
        col_major_threads: If True, uses col-major thread distribution.
            If False (default), uses row-major. Col-major matches the
            warp_layout used by Q loading in MHA attention.
    """

    var bc: AMDBufferResource
    """The 128-bit buffer resource descriptor for DRAM loads."""
    var base_ptr_as_int: Int
    """Integer address of the DRAM tile base pointer."""

    @always_inline
    def __init__(out self, gmem_tile: TileTensor[Self.dtype, ...]):
        """Creates a loader from a DRAM tile.

        The TileTensor may carry RuntimeInt for any masked dimension
        (e.g. valid_rows in MixedLayout) so that make_amd_buffer_resource
        computes correct OOB clamping bounds.

        Args:
            gmem_tile: The DRAM tile as TileTensor.
        """
        self.bc = make_amd_buffer_resource(gmem_tile)
        self.base_ptr_as_int = Int(gmem_tile.ptr)

    @always_inline
    def load(
        self,
        dst: TileTensor[
            mut=True, Self.dtype, _, _, address_space=AddressSpace.LOCAL, ...
        ],
        src: TileTensor[Self.dtype, ...],
    ):
        """Loads DRAM tile data into a LOCAL register tile.

        Distributes the src DRAM tile across threads using the configured
        thread layout (row-major by default, col-major if col_major_threads
        is True), then each thread issues buffer_load intrinsics to fill
        its portion of the dst register tile. OOB reads are hardware-clamped
        to zero via the AMDBufferResource bounds.

        Row-major iteration over src fragments for L1 cache locality,
        col-major storage into dst for copy_local_to_shared compatibility.

        Args:
            dst: Destination register TileTensor (LOCAL address space).
            src: Source DRAM TileTensor (vectorized). Must be within the
                address range of the tile used to construct this loader.
        """
        comptime if Self.col_major_threads:
            self._load_with_layout[
                col_major[Self.thread_rows, Self.thread_cols]()
            ](dst, src)
        else:
            self._load_with_layout[
                row_major[Self.thread_rows, Self.thread_cols]()
            ](dst, src)

    @always_inline
    def _load_with_layout[
        thread_layout: Layout,
    ](
        self,
        dst: TileTensor[
            mut=True, Self.dtype, _, _, address_space=AddressSpace.LOCAL, ...
        ],
        src: TileTensor[Self.dtype, ...],
    ):
        """Inner load implementation parameterized by thread layout.

        Separated from load() because row_major and col_major return
        different Layout type parameterizations that cannot be held in
        a single comptime field or ternary expression.
        """
        comptime elem_size = type_of(src).element_size
        comptime num_busy_threads = thread_layout.size()

        var worker_idx = Int(lane_id()) if Self.warp_scope else thread_idx.x

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var dist_result = src.distribute_with_offset[thread_layout](worker_idx)
        var src_dist = dist_result[0]

        # Base offset: thread fragment position relative to buffer base,
        # in element units.
        var base_offset = Int32(
            (Int(src_dist.ptr) - self.base_ptr_as_int) // size_of[Self.dtype]()
        )

        # Distributed fragment shape and strides (all comptime).
        comptime M = type_of(src_dist).static_shape[0]
        comptime N = type_of(src_dist).static_shape[1]
        comptime src_stride0 = type_of(src_dist).static_stride[0]
        comptime src_stride1 = type_of(src_dist).static_stride[1]

        # Row-major src iteration, col-major dst storage.
        comptime for i in range(M):
            comptime for j in range(N):
                comptime src_elem_offset = i * src_stride0 + j * src_stride1
                comptime dst_idx = i + j * M  # col_major(M, N)([i, j])

                dst.ptr.store[width=elem_size](
                    dst_idx * elem_size,
                    self.bc.load[Self.dtype, elem_size](
                        base_offset,
                        scalar_offset=Int32(src_elem_offset),
                    ),
                )


# ===----------------------------------------------------------------------=== #
# tt_copy_local_to_shared
# ===----------------------------------------------------------------------=== #


@always_inline
def tt_copy_local_to_shared[
    thread_rows: Int,
    thread_cols: Int,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_rows * thread_cols,
](
    dst: TileTensor[mut=True, _, _, _, address_space=AddressSpace.SHARED, ...],
    src: TileTensor[_, _, _, address_space=AddressSpace.LOCAL, ...],
):
    """Copy register data to SMEM, distributed across threads.

    Mirrors LayoutTensor copy_local_to_shared[row_major=True] but operates
    on TileTensors. Reads src registers in col-major element order to match
    the storage convention of RegTileLoader.

    Parameters:
        thread_rows: Number of rows in the row-major thread distribution.
        thread_cols: Number of columns in the row-major thread distribution.
        swizzle: Optional swizzle to reduce SMEM bank conflicts.
        num_threads: Total threads; threads beyond thread_rows*thread_cols
            are idle.

    Args:
        dst: Destination TileTensor in shared memory.
        src: Source TileTensor in local (register) memory.
    """
    comptime thread_layout = row_major[thread_rows, thread_cols]()
    comptime num_busy_threads = thread_layout.size()
    comptime elem_size = type_of(dst).element_size

    var worker_idx = Int(thread_idx.x)

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var dist_result = dst.distribute_with_offset[thread_layout, swizzle](
        worker_idx
    )
    var dst_dist = dist_result[0]

    comptime M = type_of(dst_dist).static_shape[0]
    comptime N = type_of(dst_dist).static_shape[1]
    comptime dst_stride0 = type_of(dst_dist).static_stride[0]
    comptime dst_stride1 = type_of(dst_dist).static_stride[1]

    # Col-major iteration order to match RegTileLoader's storage convention.
    comptime for i in range(M):
        comptime for j in range(N):
            comptime src_idx = i + j * M  # col_major(M, N)([i, j])
            comptime dst_elem_offset = i * dst_stride0 + j * dst_stride1

            comptime DstVec = SIMD[type_of(dst_dist).dtype, elem_size]
            dst_dist.ptr.store[width=elem_size](
                dst_elem_offset,
                rebind[DstVec](
                    src.ptr.load[width=elem_size](src_idx * elem_size)
                ),
            )


# ===----------------------------------------------------------------------=== #
# RegTileWriter
# ===----------------------------------------------------------------------=== #


struct RegTileWriter[
    dtype: DType,
    thread_rows: Int,
    thread_cols: Int,
](TrivialRegisterPassable):
    """AMD buffer-resource store for writing register tiles to DRAM.

    Pre-builds the AMDBufferResource from the full DRAM output tile once.
    Each store call writes a warp sub-tile's worth of register data
    to DRAM via the pre-built descriptor.

    Pure TileTensor implementation — uses TileTensor distribute_with_offset
    directly (no LayoutTensor conversion). The distribute operation divides
    shape by thread_shape and multiplies strides by thread_shape, producing
    identical offsets to LayoutTensor's zipped_divide for flat 2D layouts.

    Two store methods are provided:
    - store: Generic path using src layout indexing (any MMA shape).
    - store_mfma32: 32x32 MMA path with hardware-specific register remapping.

    Parameters:
        dtype: Element data type for DRAM destination.
        thread_rows: Number of rows in the col-major thread distribution.
        thread_cols: Number of columns in the col-major thread distribution.
    """

    comptime thread_layout = col_major[Self.thread_rows, Self.thread_cols]()

    var bc: AMDBufferResource
    """The 128-bit buffer resource descriptor for DRAM stores."""
    var base_ptr_as_int: Int
    """Integer address of the full DRAM tile base pointer."""

    @always_inline
    def __init__(out self, dst_base: TileTensor):
        """Create a writer from the full DRAM output tile.

        The TileTensor must carry RuntimeInt for any masked dimension
        (e.g. valid_rows) so that make_amd_buffer_resource computes
        correct OOB clamping bounds.

        Args:
            dst_base: The full DRAM output tile as TileTensor.
        """
        self.bc = make_amd_buffer_resource(dst_base)
        self.base_ptr_as_int = Int(dst_base.ptr)

    @always_inline
    def store(
        self,
        dst_warp_tile: TileTensor[Self.dtype, ...],
        src_tile: TileTensor[_, _, _, address_space=AddressSpace.LOCAL, ...],
    ):
        """Write register tile data to a DRAM warp sub-tile.

        Generic path that uses the source layout's own index mapping,
        suitable for any MMA shape. Source elements are cast to Self.dtype
        before storing.

        Args:
            dst_warp_tile: Vectorized DRAM warp sub-tile.
            src_tile: Register TileTensor with MMA output data.
        """
        comptime elem_size = type_of(dst_warp_tile).element_size

        # Distribute dst among threads via TileTensor directly.
        var dist_result = dst_warp_tile.distribute_with_offset[
            Self.thread_layout
        ](lane_id())
        var dst_dist = dist_result[0]

        # Base offset: warp tile position + thread offset, in scalar units.
        var base_offset = Int32(
            (Int(dst_dist.ptr) - self.base_ptr_as_int) // size_of[Self.dtype]()
        )

        # Distributed tile shape and strides for iteration.
        comptime dst_shape0 = type_of(dst_dist).static_shape[0]
        comptime dst_shape1 = type_of(dst_dist).static_shape[1]
        comptime dst_stride0 = type_of(dst_dist).static_stride[0]
        comptime dst_stride1 = type_of(dst_dist).static_stride[1]

        # Src strides for direct pointer loads.
        comptime src_stride0 = type_of(src_tile).static_stride[0]
        comptime src_cols = type_of(src_tile).static_shape[1] // elem_size

        comptime for i in range(dst_shape0 * dst_shape1):
            # Src offset: row-major with elem_size grouping.
            comptime sr, sc = divmod(i, src_cols)
            comptime src_offset = sr * src_stride0 + sc * elem_size

            # Dst offset within distributed fragment.
            comptime dr, dc = divmod(i, dst_shape1)
            comptime dst_elem_offset = dr * dst_stride0 + dc * dst_stride1

            var data = src_tile.ptr.load[width=elem_size](src_offset)
            self.bc.store(
                base_offset + Int32(dst_elem_offset),
                data.cast[Self.dtype](),
            )

    @always_inline
    def store_mfma32(
        self,
        dst_warp_tile: TileTensor[Self.dtype, ...],
        src_tile: TileTensor[_, _, _, address_space=AddressSpace.LOCAL, ...],
    ):
        """Write register tile data with 32x32 MFMA index remapping.

        Uses the 32x32 MMA hardware-specific register layout where
        src[4*n + 16*m] maps to dst fragment [4*m + n]. Source data
        is read via direct pointer loads with the permuted offsets.

        Args:
            dst_warp_tile: Vectorized DRAM warp sub-tile.
            src_tile: Register TileTensor with 32x32 MMA output data.
        """
        comptime elem_size = type_of(dst_warp_tile).element_size

        # Distribute dst among threads via TileTensor directly.
        var dist_result = dst_warp_tile.distribute_with_offset[
            Self.thread_layout
        ](lane_id())
        var dst_dist = dist_result[0]

        # Base offset: warp tile position + thread offset, in scalar units.
        var base_offset = Int32(
            (Int(dst_dist.ptr) - self.base_ptr_as_int) // size_of[Self.dtype]()
        )

        # Distributed tile strides for iteration.
        comptime dst_stride0 = type_of(dst_dist).static_stride[0]
        comptime dst_stride1 = type_of(dst_dist).static_stride[1]

        # M, N for the 32x32 MFMA register iteration.
        comptime M = type_of(src_tile).static_shape[0]
        comptime N = type_of(src_tile).static_shape[1] // elem_size

        comptime for n in range(N):
            comptime for m in range(M):
                # MFMA 32x32 hardware register permutation.
                comptime src_scalar_offset = 4 * n + 16 * m
                comptime i = 4 * m + n

                # Dst offset within distributed fragment.
                comptime dr, dc = divmod(i, type_of(dst_dist).static_shape[1])
                comptime dst_elem_offset = dr * dst_stride0 + dc * dst_stride1

                var data = src_tile.ptr.load[width=elem_size](src_scalar_offset)
                self.bc.store(
                    base_offset + Int32(dst_elem_offset),
                    data.cast[Self.dtype](),
                )
