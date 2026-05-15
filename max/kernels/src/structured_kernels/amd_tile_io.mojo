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

Low-level LDS read primitives:
    ds_read_tr16_b64_row  - 4x16 transposed LDS read (raw rocdl intrinsic).
    ds_read_tr16_b64_warp - Warp-level transposed LDS read.
    load_lds_fragment     - Generic MFMA-fragment LDS read with swizzle.

DRAM→LDS cooperative DMA loaders (expert objects, structurally composed):
    TileLoaderLDS     - Warp-group cooperative, coord-indexed tile iteration
                        (half-tile BK-wide steps, per-iter swizzle). Matmul's
                        pattern. Uses stdlib `AMDBufferResource.load_to_lds`.
    SubTileLoaderLDS  - Single sub-tile DMA, TileTensor-indexed. Attention's
                        pattern. Uses `rocdl.raw.ptr.buffer.load.lds` with
                        the `amdgpu.AsyncCopies` alias scope so consumers
                        carrying `noalias_scopes=_alias_scope_attr` can skip
                        `s_waitcnt vmcnt(0)` (PR #74537).

SMEM→register MMA-fragment loader (expert object, static methods):
    TiledMmaLoader   - Sibling to `TiledMmaOp`. Parameterized by operand
                       dtype, MMA shape, and optional swizzle. Static
                       `load_b`, `load_b_tr`, `load_v_fp8_strip` methods
                       cover the B-operand and V-operand MFMA-fragment
                       load patterns (attention's QK / PV matmuls).

DRAM↔register loaders:
    RegTileLoader   - AMD buffer-resource load from DRAM to registers.
    RegTileWriter   - AMD buffer-resource store from registers to DRAM.
                      Buffer-resource OOB clamping handles the M
                      boundary cleanly but cannot distinguish row/col
                      straddle, so use this only when N is BN-aligned
                      and no fused lambda is needed.
    RegTileEpilogue - Per-lane epilogue writer with optional fused
                      elementwise lambda. Caller passes
                      (m_global, n_global) per call; the writer
                      handles the fully-in-bounds chunk store, the
                      partial-chunk-straddling-N per-element fallback,
                      and the lambda dispatch. Use this for any kernel
                      that needs to support N-misaligned shapes or a
                      fused epilogue.

Register→LDS writer (expert object, static methods):
    RegTileWriterLDS - Sibling to `RegTileLoader` / `RegTileWriter`.
        Stateless; parameterized by `thread_layout + swizzle`.
        `.copy` handles plain SMEM; `.copy_blocked[block_cols]` handles
        the `blocked_product`-mismatched-layout case.

SMEM layout helpers:
    smem_subtile / smem_mma_subtile / smem_mma_subtile_offset - blocked
        SMEM navigation (TileTensor views + offset math).
"""

from std.sys import align_of, simd_width_of, size_of
from std.gpu import lane_id, thread_idx, WARP_SIZE
from std.gpu.intrinsics import ds_read_tr8_b64
from std.gpu._utils import to_i32, to_i64
from std.gpu.intrinsics import AMDBufferResource
from std.memory import AddressSpace
from std.memory.unsafe import bitcast
from std.math import min
from std.math.uutils import umod, ufloordiv
from std.sys.intrinsics import readfirstlane
from std.utils import IndexList
from layout import Coord, Idx, TileTensor, TensorLayout
from layout._utils import make_amd_buffer_resource
from layout.tile_layout import Layout, row_major, col_major
from layout.swizzle import Swizzle
from std.itertools import product


comptime elementwise_epilogue_type = def[
    dtype: DType, width: SIMDSize, *, alignment: Int = 1
](IndexList[2], SIMD[dtype, width]) capturing -> None
"""Type alias for a fused elementwise epilogue lambda.

Local re-declaration of `linalg.utils.elementwise_epilogue_type`.
`structured_kernels` is a *dependency* of `linalg`, so we cannot
import the canonical definition without creating a cyclic bazel dep.
Mojo function-pointer types are structural, so this duplicate alias
is interchangeable with the canonical one at every call site that
hands a lambda across the package boundary.
"""


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


struct TiledMmaLoader[
    in_type: DType,
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
]:
    """SMEM→register loader expert for MFMA operand fragments.

    Sibling to `TiledMmaOp` (static MFMA compute). Stateless — all
    methods are `@staticmethod`. Parameterized by operand dtype, MMA
    instruction shape, and optional vector-space swizzle. Reusable
    wherever a kernel issues MMA-tile-shaped SMEM reads (attention's
    QK / PV matmuls, potential future matmul variants).

    Static methods:

    - `load_b`: full B-operand load from a warp-sized SMEM tile.
      M-outer iteration; handles BF16 single-load and FP8 two-half-load
      (`num_packs` branch) with optional vector-space swizzle.
    - `load_b_tr`: transposed single-MMA-tile load via
      `ds_read_tr16_b64_warp` halves + join (BF16 double-rate shapes:
      32x32x16 and 16x16x32).
    - `load_v_fp8_strip`: FP8 V-operand per-strip load via
      `ds_read_tr8_b64` with paired-lane addressing (16x16x128 FP8 PV
      matmul).

    `_load_b_tile` is a private helper used by `load_b`.

    Parameters:
        in_type: Operand element type.
        mma_shape: MMA instruction shape [M, N, K].
        swizzle: Optional vector-space swizzle for `load_b`.
    """

    @staticmethod
    @always_inline
    def load_b[
        num_mmas: Int,
        simd_width: Int,
    ](
        src: TileTensor[
            Self.in_type, _, address_space=AddressSpace.SHARED, ...
        ],
    ) -> InlineArray[SIMD[Self.in_type, simd_width], num_mmas]:
        """Full B operand load from a SMEM warp tile.

        Loads all MMA tiles from a WN x BK SMEM warp tile and returns
        them as an InlineArray of SIMD fragments (one per MMA tile).

        Parameters:
            num_mmas: Number of MMA tiles to load.
            simd_width: SIMD vector width for the element type.

        Args:
            src: A WN x BK TileTensor in shared memory.

        Returns:
            An InlineArray of SIMD fragments, one per MMA tile.
        """
        comptime MMA_M = Self.mma_shape[0]
        comptime MMA_K = Self.mma_shape[2]
        comptime BK = type_of(src).static_shape[1]
        comptime M = type_of(src).static_shape[0] // MMA_M
        comptime N = BK // MMA_K
        comptime load_width = simd_width_of[Self.in_type]()
        comptime mma_frag_width = (MMA_M * MMA_K) // WARP_SIZE
        comptime num_packs = mma_frag_width // load_width
        comptime assert num_packs == 1 or num_packs == 2

        var result = InlineArray[SIMD[Self.in_type, simd_width], num_mmas](
            uninitialized=True
        )
        comptime for i in range(M):
            comptime for j in range(N):
                var src_row = src.tile[MMA_M, BK](Int(i), 0)
                comptime if num_packs == 1:
                    # BF16: single load covers the full fragment.
                    result[Int(i) + Int(j) * M] = rebind[
                        SIMD[Self.in_type, simd_width]
                    ](Self._load_b_tile[Self.mma_shape, Int(j)](src_row))
                else:
                    # FP8: MMA K (128) = 2 * SMEM load width (64 elements
                    # per 16-element lane SIMD). Load two [MMA_M, MMA_K/2]
                    # halves and join so the K-dim permutation matches Q.
                    comptime half_k_shape = IndexList[3](
                        MMA_M, Self.mma_shape[1], MMA_K // 2
                    )
                    var lo = Self._load_b_tile[half_k_shape, Int(j) * 2](
                        src_row
                    )
                    var hi = Self._load_b_tile[half_k_shape, Int(j) * 2 + 1](
                        src_row
                    )
                    result[Int(i) + Int(j) * M] = rebind[
                        SIMD[Self.in_type, simd_width]
                    ](lo.join(hi))
        return result

    @staticmethod
    @always_inline
    def load_b_tr(
        tile: TileTensor[
            Self.in_type, _, address_space=AddressSpace.SHARED, ...
        ],
    ) -> SIMD[Self.in_type, 8]:
        """Transposed B operand load for double-rate MFMA shapes.

        Splits the tile along the K dimension into two halves and
        concatenates the results.

        Args:
            tile: A MMA_K x MMA_N TileTensor in shared memory.

        Returns:
            A SIMD[in_type, 8] vector with both halves concatenated.
        """
        comptime assert Self.mma_shape in (
            IndexList[3](32, 32, 16),
            IndexList[3](16, 16, 32),
        )
        comptime assert Self.in_type == DType.bfloat16
        comptime MMA_K = Self.mma_shape[2]
        comptime MMA_N = Self.mma_shape[1]
        comptime half_k = MMA_K // 2
        comptime assert type_of(tile).static_shape[0] == MMA_K
        comptime assert type_of(tile).static_shape[1] == MMA_N

        var part_1 = ds_read_tr16_b64_warp[Self.mma_shape](
            tile.tile[half_k, MMA_N](0, 0)
        )
        var part_2 = ds_read_tr16_b64_warp[Self.mma_shape](
            tile.tile[half_k, MMA_N](1, 0)
        )
        return part_1.join(part_2)

    @staticmethod
    @always_inline
    def load_v_fp8_strip[
        BN: Int,
        BK: Int,
        bk_tile: Int,
        dt: Int,
    ](
        v_base: UnsafePointer[
            Scalar[Self.in_type],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        rel_key: Int,
        hw_key_shift: Int,
        depth_base: Int,
    ) -> SIMD[Self.in_type, 32]:
        """FP8 V per-strip `ds_read_tr8_b64` load for one (bk_tile, dt).

        Paired-lane addressing: issues 4 `ds_read_tr8_b64` calls (at
        `key_base = 0, 16, 32, 48`) and joins the results into one
        32-element SIMD matching the MFMA C-output column pattern for
        the 16x16x128 FP8 V operand in the PV matmul.

        Caller is responsible for precomputing the per-lane coords
        (`rel_key`, `hw_key_shift`, `depth_base`) ONCE before the outer
        (bk_tile, dt) loop — they're lane-only, not
        (bk_tile, dt)-dependent, so hoisting saves redundant address
        math per iteration.

        Parameters:
            BN: V block height in elements.
            BK: V block width in elements.
            bk_tile: Which BK-tall row strip (0..num_k_tiles - 1).
            dt: Which depth-tile within the strip (0..depth/MMA_M - 1).

        Args:
            v_base: Pointer to V SMEM stage base (block 0 of
                `num_repeats`).
            rel_key: Per-lane relative key index within the 16-lane row.
            hw_key_shift: +4 for lanes in hw1, +0 for hw0.
            depth_base: Per-lane depth sub-range offset
                (0 or 8 or 16 or 24).

        Returns:
            `SIMD[in_type, 32]` for this lane's (bk_tile, dt) strip.
        """
        comptime MMA_M = Self.mma_shape[0]
        comptime row_offset = bk_tile * 64
        comptime depth_offset = dt * MMA_M
        comptime blk = depth_offset // BK
        comptime d_in_blk = depth_offset % BK
        var block_base = v_base + blk * BN * BK

        # Match K's swizzle on V reads.  In `mla_kv_alias` mode K and V
        # share one SMEM image: K writes via `SubTileLoaderLDS.load`,
        # which permutes write positions by `swizzle(vec_idx) * simd_w`
        # where `vec_idx = byte_offset / simd_w` (16B for FP8) and
        # `swizzle` XORs bits 4..6 of `vec_idx` into bits 0..2.  To read
        # the same element back, V must apply the same permutation.
        #
        # An element at logical `byte_offset` was written to
        #     swizzle(byte_offset / simd_w) * simd_w + (byte_offset % simd_w)
        # because the swizzle operates only on the vec-aligned part —
        # the low `log2(simd_w)` bits (here 0..3) are the within-vec
        # byte position, which is the same for writer and reader.
        # Paired V lanes use `depth_base ∈ {0, 8, 16, 24}`, i.e. an
        # 8-byte sub-vec offset, so preserving the low 4 bits is
        # required for the 8-byte `ds_read_tr8_b64` to land correctly.
        comptime simd_w = simd_width_of[Self.in_type]()

        @always_inline
        @parameter
        def _load_keys[key_base: Int]() -> SIMD[Self.in_type, 8]:
            var key = row_offset + key_base + rel_key + hw_key_shift
            var byte_offset = key * BK + d_in_blk + depth_base
            comptime if Self.swizzle:
                # Decompose byte_offset = vec_idx * simd_w + sub_vec,
                # permute vec_idx, then reassemble.
                var vec_idx = byte_offset // simd_w
                var sub_vec = byte_offset & (simd_w - 1)
                var swizzled_vec = Self.swizzle.value()(vec_idx)
                var addr = block_base + swizzled_vec * simd_w + sub_vec
                return ds_read_tr8_b64(addr)
            else:
                return ds_read_tr8_b64(block_base + byte_offset)

        var r0 = _load_keys[0]()
        var r1 = _load_keys[16]()
        var r2 = _load_keys[32]()
        var r3 = _load_keys[48]()
        return r0.join(r1).join(r2.join(r3))

    @staticmethod
    @always_inline
    def _load_b_tile[
        tile_mma_shape: IndexList[3],
        k_tile_idx: Int,
    ](
        src: TileTensor[
            Self.in_type, _, address_space=AddressSpace.SHARED, ...
        ],
    ) -> SIMD[Self.in_type, simd_width_of[Self.in_type]()]:
        """Private helper for `load_b`: single MMA sub-tile load.

        Takes the MMA shape as a parameter (rather than reading Self.mma_shape)
        because `load_b` may need to issue loads with a half-K MMA shape on
        the FP8 num_packs==2 path, distinct from the struct's MMA shape.
        """
        comptime MMA_M = tile_mma_shape[0]
        comptime MMA_K = tile_mma_shape[2]
        comptime assert type_of(src).static_shape[0] == MMA_M
        comptime simd_width = simd_width_of[Self.in_type]()
        comptime assert MMA_M == 32 or MMA_M == 16

        # Pick up the sub-tile's element offset from `tile_with_offset`,
        # then add the lane's per-element offset from
        # `distribute_with_offset`. Both offsets are already in
        # `Self.in_type` scalar elements, so no ptr-to-int byte math is
        # needed.
        var sub_tile_res = src.tile_with_offset[MMA_M, MMA_K](
            Coord(Idx(0), Idx(k_tile_idx))
        )
        var sub_tile = sub_tile_res[0]
        comptime idx_type = src.linear_idx_type
        var sub_offset = Scalar[idx_type](sub_tile_res[2])

        # BF16: col_major(32, 2) — 32 lanes along M, 2 groups along K.
        # FP8:  col_major(16, 4) — 16 lanes along M, 4 groups along K.
        var offset: Scalar[idx_type]
        comptime if MMA_M == 32:
            comptime thread_layout = col_major[32, 2]()
            var dist_res = sub_tile.vectorize[
                1, simd_width
            ]().distribute_with_offset[thread_layout](lane_id())
            offset = sub_offset + Scalar[idx_type](dist_res[2])
        else:
            comptime thread_layout = col_major[16, 4]()
            var dist_res = sub_tile.vectorize[
                1, simd_width
            ]().distribute_with_offset[thread_layout](lane_id())
            offset = sub_offset + Scalar[idx_type](dist_res[2])

        comptime if Self.swizzle:
            offset = Self.swizzle.value()(
                offset // Scalar[idx_type](simd_width)
            ) * Scalar[idx_type](simd_width)

        return _load_from_lds[width=simd_width](src.ptr + offset)


# ===----------------------------------------------------------------------=== #
# _load_from_lds: Alias-scoped LDS load
# ===----------------------------------------------------------------------=== #


@always_inline
def _load_from_lds[
    dtype: DType,
    //,
    width: Int = 1,
](
    shared_ptr: UnsafePointer[
        Scalar[dtype], _, address_space=AddressSpace.SHARED
    ],
) -> SIMD[dtype, width]:
    """Alias-scoped LDS load via LLVM intrinsic with noalias annotations."""
    comptime alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
    comptime no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](shared_ptr)

    comptime load_bytes = width * size_of[dtype]()
    comptime alignment = min(load_bytes, 16)

    comptime if dtype == DType.bfloat16 and width == 4:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<4 x bf16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[DType.bfloat16, 4]._mlir_type
            ](llvm_res)
        )
    elif dtype == DType.bfloat16 and width == 8:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x bf16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[DType.bfloat16, 8]._mlir_type
            ](llvm_res)
        )
    elif dtype.is_float8() and width == 8:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 8]._mlir_type
        ](llvm_res)
        return bitcast[dtype, width](
            rebind[SIMD[DType.uint8, width]](uint8_vec)
        )
    elif dtype.is_float8() and width == 16:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res)
        return bitcast[dtype, width](
            rebind[SIMD[DType.uint8, width]](uint8_vec)
        )
    elif dtype.is_float8() and width == 32:
        var llvm_res0 = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var shared_ptr_offset = shared_ptr + 16
        var shared_ptr3_hi = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_type.`!llvm.ptr<3>`
        ](shared_ptr_offset)
        var llvm_res1 = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3_hi)
        var uint8_vec0 = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res0)
        var uint8_vec1 = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[DType.uint8, 16]._mlir_type
        ](llvm_res1)
        var uint8_vec = rebind[SIMD[DType.uint8, 16]](uint8_vec0).join(
            rebind[SIMD[DType.uint8, 16]](uint8_vec1)
        )
        return bitcast[dtype, width](uint8_vec)
    else:
        comptime assert False, "Unsupported dtype/width for _load_from_lds"


# ===----------------------------------------------------------------------=== #
# load_lds_fragment: MMA LDS→register load (element-granularity swizzle)
# ===----------------------------------------------------------------------=== #


@always_inline
def load_lds_fragment[
    smem_layout: TensorLayout,
    reg_layout: TensorLayout,
    //,
    MMA_K: Int,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](
    smem_tile: SMemTile[mut=False, _, smem_layout, _],
    reg_tile: RegTile[mut=True, smem_tile.dtype, reg_layout, _],
):
    """Load MMA fragments from SMEM to registers using hardware access pattern.

    Dimensions are derived from the tile layouts:
        - num_mmas = reg rows, MMA_M = smem rows / num_mmas
        - lds_frag_width = MMA_M * MMA_K / WARP_SIZE
        - lds_row_stride: MMA_K (BF16 dense), smem stride (FP8 or strided)
        - num_iterations = reg flat elements / lds_frag_width

    Parameters:
        smem_layout: Inferred layout of the SMEM source tile.
        reg_layout: Inferred layout of the register destination tile.
        MMA_K: MMA K dimension (hardware instruction width).
        swizzle: Optional element-space swizzle.

    Args:
        smem_tile: Source [num_mmas * MMA_M, K] in SHARED.
        reg_tile: Destination [num_mmas, K_frags * frag_width] in LOCAL.
    """
    comptime dtype = smem_tile.dtype
    comptime smem_rows = smem_layout.static_shape[0]
    comptime smem_cols = smem_layout.static_shape[1]
    comptime smem_stride = smem_layout.static_stride[0]
    comptime num_mmas = reg_layout.static_shape[0]
    comptime reg_cols = reg_layout.static_shape[1]
    comptime reg_stride = reg_layout.static_stride[0]
    comptime MMA_M = smem_rows // num_mmas
    comptime mma_frag_width = MMA_M * MMA_K // WARP_SIZE
    comptime use_fp8_split = (
        dtype.is_float8() and MMA_M == 16 and MMA_K == 128
    )
    comptime lds_frag_width = 16 if use_fp8_split else mma_frag_width
    comptime num_iterations = (num_mmas * reg_cols) // lds_frag_width

    # SMEM row stride: when the smem tile is a narrow sub-tile of a wider
    # allocation (stride > cols), use the physical stride. Otherwise use the
    # MMA-native stride: smem_cols for FP8 (contiguous BK), MMA_K for BF16
    # (mma_access_layout packs MMA_K elements per logical row).
    comptime smem_is_subtile = smem_stride > smem_cols
    comptime lds_row_stride = (
        smem_stride if smem_is_subtile else (
            smem_cols if dtype.is_float8() else MMA_K
        )
    )

    # Register stride: strided sub-tile (stride > cols) spaces fragments at
    # row stride; dense tile packs fragments contiguously.
    comptime _reg_stride = (
        reg_stride if reg_stride > reg_cols else lds_frag_width
    )

    var smem_ptr = smem_tile.ptr
    var reg_ptr = reg_tile.ptr
    comptime FragElement = SIMD[dtype, lds_frag_width]

    comptime col_groups = WARP_SIZE // MMA_M
    var lane = lane_id()
    var lane_offset = (
        Int(lane % MMA_M) * lds_row_stride + Int(lane // MMA_M) * lds_frag_width
    )

    comptime elements_per_iter = col_groups * lds_frag_width
    comptime use_split_k = lds_row_stride > elements_per_iter

    comptime if use_split_k:
        comptime k_splits = lds_row_stride // elements_per_iter
        comptime m_positions = num_iterations // k_splits
        comptime k_stride = elements_per_iter
        comptime m_stride = lds_row_stride * MMA_M

        comptime for m_idx in range(m_positions):
            comptime for k_idx in range(k_splits):
                var iter_base = m_idx * m_stride + k_idx * k_stride
                var full_offset = iter_base + lane_offset

                comptime if swizzle:
                    full_offset = swizzle.value()(full_offset)

                comptime frag_idx = m_idx * k_splits + k_idx
                reg_ptr.store[width=lds_frag_width](
                    frag_idx * _reg_stride,
                    rebind[FragElement](
                        _load_from_lds[width=lds_frag_width](
                            smem_ptr + full_offset
                        )
                    ),
                )
    else:
        comptime for i in range(num_iterations):
            var iter_base = i * WARP_SIZE * lds_frag_width
            var full_offset = iter_base + lane_offset

            comptime if swizzle:
                full_offset = swizzle.value()(full_offset)

            reg_ptr.store[width=lds_frag_width](
                i * _reg_stride,
                rebind[FragElement](
                    _load_from_lds[width=lds_frag_width](smem_ptr + full_offset)
                ),
            )


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
def smem_mma_subtile_offset[
    mma_rows: Int,
    mma_cols: Int,
    BN: Int,
    BK: Int,
](bk_tile: Int, k_sub: Int, mma_idx: Int) -> Int:
    """Element offset of an MMA sub-tile within a blocked (BN x BK) SMEM region.

    The physical SMEM layout is `num_repeats` contiguous `BN x BK`
    blocks. This helper computes the scalar-element offset (from the
    stage base) of the MMA sub-tile at `(bk_tile, k_sub, mma_idx)`.

    The offset is layout-agnostic — callers that need a `TileTensor`
    view pair it with whatever within-tile stride their load pattern
    requires (see `smem_mma_subtile` for the default row-major form).
    """
    comptime block_size = BN * BK
    comptime tiles_per_block = BK // mma_cols
    var block_idx = mma_idx // tiles_per_block
    var col_in_block = (mma_idx % tiles_per_block) * mma_cols
    return (
        bk_tile * BK * BK
        + k_sub * mma_rows * BK
        + block_idx * block_size
        + col_in_block
    )


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
    within one block. The returned TileTensor uses plain
    `row_major[mma_rows, mma_cols]` strides — only correct when the
    physical row stride equals `mma_cols`. For `mma_cols < BK`, callers
    must pair `smem_mma_subtile_offset` with an explicit-stride layout
    (e.g. `MixedLayout((mma_rows, mma_cols), (BK, 1))`).

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
    var offset = smem_mma_subtile_offset[mma_rows, mma_cols, BN, BK](
        bk_tile, k_sub, mma_idx
    )
    return TileTensor[
        dtype,
        type_of(row_major[mma_rows, mma_cols]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ](smem_ptr + offset, row_major[mma_rows, mma_cols]())


# ===----------------------------------------------------------------------=== #
# TileLoaderLDS: Cooperative Global→LDS loader
# ===----------------------------------------------------------------------=== #


struct TileLoaderLDS[
    dtype: DType,
    tile_rows: Int,
    tile_cols: Int,
    stride: Int,
    num_loading_warps: Int,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
    load_width: Int = simd_width_of[dtype](),
    use_full_tile_width: Bool = False,
](TrivialRegisterPassable):
    """DRAM→LDS DMA expert for warp-group cooperative coord-indexed loads.

    Sibling of `SubTileLoaderLDS` (single-sub-tile TileTensor-indexed).
    This one coordinates a warp group (typically 8 warps) to cooperatively
    fill a half-tile via coord-indexed iteration: `load_tile(dst, src_row,
    src_col)` steps through `num_iterations` BK-wide rows, optionally
    applying a per-iteration byte-space swizzle for LDS bank-conflict
    avoidance. Matmul's DRAM→LDS pattern (ping-pong, etc.).

    Uses stdlib `AMDBufferResource.load_to_lds` directly — no alias scope
    attached. Matmul's scheduling uses `s_sched_group_barrier` hints,
    which don't qualify as the runtime fence required by the
    `SIInsertWaitcnts` vmcnt-relaxation contract; attaching the scope
    would miscompile (see the `async_copies` docstring on
    `load_to_lds`). For attention patterns that DO satisfy the contract
    via explicit `s_waitcnt vmcnt(0) + s_barrier` fences, use
    `SubTileLoaderLDS` instead.

    Parameters:
        dtype: Element data type.
        tile_rows: Height of each half-tile to load.
        tile_cols: Width (K dimension) of each half-tile.
        stride: Row stride of the source GMEM tensor.
        num_loading_warps: Warps cooperating on each load (typically 8).
        swizzle: Optional byte-space swizzle for LDS bank conflicts.
        load_width: Elements per load (SIMD width).
        use_full_tile_width: FP8 row-major mode.
    """

    comptime subtile_cols = Self.tile_cols if Self.use_full_tile_width else 32
    comptime threads_per_row = Self.subtile_cols // Self.load_width
    comptime thread_rows = WARP_SIZE // Self.threads_per_row

    comptime num_warp_cols = Self.tile_cols // Self.subtile_cols
    comptime num_warp_rows = Self.num_loading_warps // Self.num_warp_cols

    comptime elements_per_warp = WARP_SIZE * Self.load_width
    comptime rows_per_warp = Self.elements_per_warp // Self.tile_cols

    comptime loading_threads = Self.num_loading_warps * WARP_SIZE
    comptime rows_per_iteration = Self.loading_threads // (
        Self.tile_cols // Self.load_width
    )
    comptime num_iterations = Self.tile_rows // Self.rows_per_iteration

    comptime warp_subtile_bytes = Self.rows_per_warp * Self.tile_cols * size_of[
        Self.dtype
    ]()
    comptime lane_load_bytes = Self.load_width * size_of[Self.dtype]()
    comptime row_bytes = Self.tile_cols * size_of[Self.dtype]()

    comptime _needs_per_iter_swizzle = Bool(
        Self.swizzle
    ) and Self.use_full_tile_width

    var buffer: AMDBufferResource
    var thread_row: Int
    var thread_col: Int
    var warp_id: Int
    var lane_id: Int

    @always_inline
    def __init__(
        out self,
        src: GMemTile[Self.dtype, _, _],
        warp_id: Int,
        lane_id: Int,
    ):
        """Build from a GMEM tile (block-level A or B tile)."""
        self.buffer = make_amd_buffer_resource(src)
        self.warp_id = warp_id
        self.lane_id = lane_id

        var effective_lane = lane_id

        comptime if Self.swizzle and not Self._needs_per_iter_swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        var warp_row, warp_col = divmod(warp_id, Self.num_warp_cols)
        var subtile_row, subtile_col_idx = divmod(
            effective_lane, Self.threads_per_row
        )
        var subtile_col = subtile_col_idx * Self.load_width

        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

    @always_inline
    def load_tile(
        self,
        dst: SMemTile[Self.dtype, _, _],
        src_row: Int,
        src_col: Int,
    ):
        """Load from GMEM at (src_row, src_col) into SMEM dst via load_to_lds.

        Args:
            dst: Destination TileTensor in SHARED (half-tile sized).
            src_row: Row offset in the block's GMEM tile.
            src_col: Column (K) offset.
        """
        comptime SmemPtr = UnsafePointer[
            Scalar[Self.dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]

        comptime if Self._needs_per_iter_swizzle:
            var lane_byte = self.lane_id * Self.lane_load_bytes

            comptime for i in range(Self.num_iterations):
                var tile_idx = i * Self.num_loading_warps + self.warp_id
                var warp_tile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                    tile_idx, 0
                )
                var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                var full_byte = tile_idx * Self.warp_subtile_bytes + lane_byte
                var swizzled_byte = Self.swizzle.value()(full_byte)

                var swizzled_row = swizzled_byte // Self.row_bytes
                var swizzled_col = (swizzled_byte % Self.row_bytes) // size_of[
                    Self.dtype
                ]()

                var lane_offset = swizzled_col + swizzled_row * Self.stride
                var uniform_offset = src_col + src_row * Self.stride

                self.buffer.load_to_lds[width=Self.load_width](
                    Int32(lane_offset),
                    smem_ptr,
                    scalar_offset=Int32(uniform_offset),
                )
        else:
            var lane_offset = self.thread_col + self.thread_row * Self.stride

            comptime for i in range(Self.num_iterations):
                var tile_idx = i * Self.num_loading_warps + self.warp_id
                var warp_tile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                    tile_idx, 0
                )
                var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                var tile_row = src_row + i * Self.rows_per_iteration
                var uniform_offset = src_col + tile_row * Self.stride

                self.buffer.load_to_lds[width=Self.load_width](
                    Int32(lane_offset),
                    smem_ptr,
                    scalar_offset=Int32(uniform_offset),
                )


# ===----------------------------------------------------------------------=== #
# SubTileLoaderLDS
# ===----------------------------------------------------------------------=== #


struct SubTileLoaderLDS[
    dtype: DType,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](TrivialRegisterPassable):
    """DRAM→LDS DMA expert for single-sub-tile TileTensor-indexed loads.

    Sibling of `TileLoaderLDS` (warp-group cooperative coord-indexed).
    This one issues one `buffer_load_*_lds` burst per `.load()` call for
    a single source sub-tile. Attention's KV-cache warp DMA pattern:
    each warp claims a `(warp_tile_rows, BK)` slice of a
    `(BN, K-span)` DRAM tile and streams it into its SMEM lane.

    The AMD `buffer_load_*_lds` intrinsic is emitted with the
    `amdgpu.AsyncCopies` alias scope via `rocdl.raw.ptr.buffer.load.lds`
    so consumer-side LDS reads tagged with
    `noalias_scopes=_alias_scope_attr` (see `ds_read_tr*` at lines 96,
    419-480) can skip `s_waitcnt vmcnt(0)` — LLVM PR #74537's
    `SIInsertWaitcnts` vmcnt-relaxation handshake. Safe because
    attention kernels also maintain an explicit
    `s_waitcnt vmcnt(0) + s_barrier` fence at DMA/compute boundaries.

    Why not `stdlib load_to_lds[async_copies=True]`: stdlib's
    `async_copies=True` attaches its OWN `alias_scope` MLIR attribute
    which is textually identical to `_alias_scope_attr` but an
    MLIR-distinct object; `ScopedNoAliasAA` matches by identity, so
    the DMA and LDS-consumer scopes don't match and the relaxation is
    silently disabled (MLA regresses 0.76 abs at output[0,0,0,0] — the
    same signature as `b7b68a00290`). Keeping the intrinsic emission
    local to this file so producer + consumer share the exact same
    `_alias_scope_attr` object. If stdlib ever exports the scope as a
    shareable symbol, collapse this body to `bc.load_to_lds[
    async_copies=True]`.

    Constructs the `AMDBufferResource` once from a DRAM tile (which may
    carry `RuntimeInt` valid_rows for bounds clamping via `MixedLayout`).
    Each `load()` call reuses the descriptor — one shared bc per tile,
    zero per-warp overhead for buffer resource construction. SRD bounds
    computed by `make_amd_buffer_resource` via `_get_bounds`; hardware
    clamps OOB reads to zero.

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
    ):
        """Load a warp sub-tile from DRAM to LDS.

        The src tile should be a warp-sized sub-tile of the original DRAM
        tile. Offsets are computed relative to the bc's base pointer, so
        the src pointer must be within the original tile's address range.

        Args:
            dst: Destination TileTensor in shared memory.
            src: Source TileTensor in global memory (warp sub-tile).
        """
        comptime M = type_of(src).static_shape[0]
        comptime N = type_of(src).static_shape[1]
        comptime BM = 32
        comptime BN = N
        # Adapt thread layout to keep bytes/lane ≤ 16
        # (buffer_load_dwordx4_lds limit).  bf16 BK=64 and fp8 BK=128
        # would load 32 B/lane with the base 16×4 layout, so widen to
        # 8×8.
        comptime raw_load_bytes = (N * size_of[Self.dtype]()) // 4
        comptime thread_rows = 16 if raw_load_bytes <= 16 else 8
        comptime thread_cols = 4 if raw_load_bytes <= 16 else 8
        comptime thread_layout = row_major[thread_rows, thread_cols]()
        comptime load_width = BN // thread_cols
        comptime BM_SUB = thread_rows

        var worker_idx = lane_id()
        var dram_base = self.bc.get_base_ptr()

        comptime dst_stride0 = type_of(dst).static_stride[0]
        comptime dst_stride1 = type_of(dst).static_stride[1]
        comptime assert dst_stride1 == 1
        comptime assert dst_stride0 == BN

        comptime aux = 0

        comptime for n_tile, m_tile, m_sub_tile in product(
            range(N // BN), range(M // BM), range(BM // BM_SUB)
        ):
            var dst_partitions = dst.tile[BM, BN](m_tile, n_tile).tile[
                BM_SUB, BN
            ](m_sub_tile, 0)
            var src_partitions = src.tile[BM, BN](m_tile, n_tile).tile[
                BM_SUB, BN
            ](m_sub_tile, 0)
            var worker_idx_with_offset = worker_idx + m_sub_tile * WARP_SIZE
            var src_dist = src_partitions.vectorize[1, load_width]().distribute[
                thread_layout
            ](
                umod(
                    Self.swizzle.value()(
                        worker_idx_with_offset
                    ) if Self.swizzle else worker_idx_with_offset,
                    WARP_SIZE,
                )
            )
            var dst_ptr = dst_partitions.ptr.address_space_cast[
                AddressSpace.SHARED
            ]()

            var desc_ptr_ = UnsafePointer[
                Scalar[DType.bfloat16],
                MutAnyOrigin,
                address_space=AddressSpace.BUFFER_RESOURCE,
            ].unsafe_dangling()
            var ptr_to_ptr = UnsafePointer(to=desc_ptr_)
            var ptr_to_simd = UnsafePointer(to=self.bc.desc)
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

            comptime num_bytes_per_lane = size_of[Self.dtype]() * load_width
            var vector_offset_bytes = Int(src_dist.ptr) - Int(
                src_partitions.ptr
            )
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


# ===----------------------------------------------------------------------=== #
# RegTileLoader
# ===----------------------------------------------------------------------=== #


struct RegTileLoader[
    dtype: DType,
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    warp_scope: Bool = False,
](TrivialRegisterPassable):
    """AMD buffer-resource load from DRAM to registers.

    Pre-builds the AMDBufferResource from a DRAM TileTensor once.
    Each load() call distributes a source tile across threads and
    issues buffer_load intrinsics to fill a LOCAL register TileTensor.

    The dst register tile uses row-major element ordering (per-thread
    (M, N) fragment stored with strides (N, 1)) so that dst row i is
    m_mma=i's contiguous fragment. `RegTileWriterLDS.copy` reads in the
    same row-major order; the two are paired and must agree.

    Parameters:
        dtype: Element data type.
        thread_layout: Thread distribution layout (e.g. row_major[r, c]()
            or col_major[r, c]()).
        num_threads: Total threads in the block. When the block has more
            threads than thread_layout.size(), extra threads are idled.
            Only needed when the block size differs from the layout size
            (e.g. attention uses a warp-sized layout within a larger block).
            Defaults to thread_layout.size().
        warp_scope: If True, uses lane_id() as worker index (warp scope).
            If False, uses thread_idx.x (block scope).
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
    def __init__(
        out self,
        gmem_tile: TileTensor[Self.dtype, ...],
        *,
        bounds_from: TileTensor[Self.dtype, ...],
    ):
        """Creates a loader with OOB bounds from a full (pre-tiled) tensor.

        TileTensor.tile produces compile-time shapes that are never clipped
        to the actual tensor extent. This overload derives the buffer
        resource clamping range from bounds_from (which carries runtime
        dimensions), so OOB loads return zero for partial edge blocks.

        Args:
            gmem_tile: Block-row tile (provides base pointer for loads).
            bounds_from: Full tensor with runtime dims for OOB bounds.
        """
        from layout._utils import _get_bounds

        var off = (Int(gmem_tile.ptr) - Int(bounds_from.ptr)) // size_of[
            Self.dtype
        ]()
        self.bc = AMDBufferResource(
            readfirstlane(gmem_tile.ptr),
            readfirstlane(_get_bounds(bounds_from) - off),
        )
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

        Distributes src across threads, reads row-major from DRAM,
        stores row-major into dst (matched by `RegTileWriterLDS.copy`).

        Args:
            dst: Destination register TileTensor (LOCAL address space).
            src: Source DRAM TileTensor (vectorized).
        """
        comptime _DistType = type_of(
            src.distribute_with_offset[Self.thread_layout](0)[0]
        )
        comptime M = _DistType.static_shape[0]
        comptime N = _DistType.static_shape[1]

        _buffer_load_impl[
            Self.thread_layout,
            Self.num_threads,
            Self.warp_scope,
        ](
            dst,
            src,
            self.bc,
            self.base_ptr_as_int,
            dst_layout=row_major[M, N](),
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
    Each `store()` call writes a warp sub-tile's worth of register data
    to DRAM via the pre-built descriptor; OOB lanes (past the recorded
    byte bound) are silently dropped by the hardware clamp.

    Pure TileTensor implementation — uses TileTensor distribute_with_offset
    directly (no LayoutTensor conversion). The distribute operation divides
    shape by thread_shape and multiplies strides by thread_shape, producing
    identical offsets to LayoutTensor's zipped_divide for flat 2D layouts.

    A single `store[mfma32: Bool = False]` method handles both:
    - mfma32=False: Generic path using the src tile's own layout indexing
      (any MMA shape).
    - mfma32=True: 32×32 MFMA path with hardware-specific register
      permutation (`src[4*n + 16*m]` → fragment position `4*m + n`).

    See `RegTileWriterLDS.copy` for the matching row-major register reader
    used in DRAM→reg→SMEM pipelines.

    The buffer-resource OOB clamp bounds the store by the destination
    tensor's TOTAL byte extent, not by a per-row column extent — so a
    SIMD chunk that straddles an N boundary (last column block when
    `N % BN != 0`) will spill into the next row of the same buffer
    instead of being clipped. Use `RegTileEpilogue` instead for kernels
    that need to support N-misaligned shapes (or a fused lambda).

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
    def store[
        mfma32: Bool = False
    ](
        self,
        dst_warp_tile: TileTensor[Self.dtype, ...],
        src_tile: TileTensor[_, _, _, address_space=AddressSpace.LOCAL, ...],
    ):
        """Write register tile data to a DRAM warp sub-tile.

        The distribute + base-offset prologue is identical across MMA
        shapes; only the `(iteration index i, src scalar offset)` pair
        differs:

        - `mfma32=False`: iterate `i in range(dst_shape0 * dst_shape1)`,
          read src at `(i // src_cols) * src_stride0 + (i % src_cols) *
          elem_size` (source's natural row-major layout).
        - `mfma32=True`: iterate `(m, n)` over `(src_shape0, src_shape1 /
          elem_size)`, read src at `4*n + 16*m` (32×32 MFMA register
          permutation) and map to fragment position `i = 4*m + n`.

        Parameters:
            mfma32: Select the 32×32 MFMA register permutation instead of
                the src tile's natural layout.

        Args:
            dst_warp_tile: Vectorized DRAM warp sub-tile.
            src_tile: Register TileTensor with MMA output data.
        """
        comptime elem_size = type_of(dst_warp_tile).element_size

        # Distribute dst among threads and compute the per-lane base
        # offset (in scalar units).
        var dist_result = dst_warp_tile.distribute_with_offset[
            Self.thread_layout
        ](lane_id())
        var dst_dist = dist_result[0]
        var lane_offset = (Int(dst_dist.ptr) - self.base_ptr_as_int) // size_of[
            Self.dtype
        ]()
        var base_offset = Int32(lane_offset)

        comptime dst_shape1 = type_of(dst_dist).static_shape[1]
        comptime dst_stride0 = type_of(dst_dist).static_stride[0]
        comptime dst_stride1 = type_of(dst_dist).static_stride[1]

        comptime if mfma32:
            # 32×32 MFMA hardware register permutation.
            comptime M = type_of(src_tile).static_shape[0]
            comptime N = type_of(src_tile).static_shape[1] // elem_size
            comptime for n in range(N):
                comptime for m in range(M):
                    comptime src_offset = 4 * n + 16 * m
                    comptime i = 4 * m + n
                    comptime dr, dc = divmod(i, dst_shape1)
                    comptime dst_elem_offset = (
                        dr * dst_stride0 + dc * dst_stride1
                    )
                    var data = src_tile.raw_load[width=elem_size](src_offset)
                    self.bc.store(
                        base_offset + Int32(dst_elem_offset),
                        data.cast[Self.dtype](),
                    )
        else:
            # Generic path: src uses its own row-major layout.
            comptime dst_shape0 = type_of(dst_dist).static_shape[0]
            comptime src_stride0 = type_of(src_tile).static_stride[0]
            comptime src_cols = type_of(src_tile).static_shape[1] // elem_size
            comptime for i in range(dst_shape0 * dst_shape1):
                comptime sr, sc = divmod(i, src_cols)
                comptime src_offset = sr * src_stride0 + sc * elem_size
                comptime dr, dc = divmod(i, dst_shape1)
                comptime dst_elem_offset = (dr * dst_stride0 + dc * dst_stride1)
                var data = src_tile.raw_load[width=elem_size](src_offset)
                self.bc.store(
                    base_offset + Int32(dst_elem_offset),
                    data.cast[Self.dtype](),
                )


# ===----------------------------------------------------------------------=== #
# RegTileEpilogue
# ===----------------------------------------------------------------------=== #


struct RegTileEpilogue[
    c_type: DType,
    chunk_width: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](TrivialRegisterPassable):
    """Per-lane MFMA epilogue writer with optional fused elementwise lambda.

    Encapsulates the per-lane `(m_global, n_global) → store / lambda`
    handoff at the end of an AMD matmul kernel. Each `store()` call
    writes one SIMD chunk of `chunk_width` columns at a single row —
    the natural shape of an AMD MFMA output fragment for one lane.

    Per-lane bound handling:
    - In-bounds chunk (`n + chunk_width <= n_total`): one SIMD store
      or one lambda call.
    - Partial chunk (`n < n_total < n + chunk_width`): per-element
      fallback. The SIMD-of-`chunk_width` store would otherwise spill
      into the next row of the buffer (where stride==N), so we degrade
      to up to `chunk_width` scalar stores or scalar lambda calls,
      each gated on `col < n_total`. This is what makes the writer
      correct for N-misaligned outputs.
    - Fully OOB column (`n >= n_total`): skip silently.

    The caller is responsible for the M bound check before calling
    `store()` — a split-K matmul kernel passes a workspace row that
    differs from the logical output row, so the writer cannot derive
    a single M bound that applies to both DRAM and lambda modes.

    With `elementwise_lambda_fn=None` writes go to DRAM at
    `c_ptr + m * row_stride + n` directly (no buffer-resource clamp;
    the partial-chunk fallback gates by `n_total` explicitly). With a
    lambda set the lambda receives global `(m, n)` and the SIMD chunk;
    DRAM is left untouched. Lambda mode therefore requires the caller
    to pass `m` as the LOGICAL output row — incompatible with a
    per-split workspace write. Kernels that use both split-K and a
    fused lambda should not set the lambda on the per-split matmul
    kernel; instead run a non-fused split-K and apply the lambda in
    the reduce kernel that consumes the partials.

    Parameters:
        c_type: Output element type.
        chunk_width: Number of contiguous columns per lane per call.
            For 16x16x* MFMA on AMD this is `MMA_M * MMA_N // WARP_SIZE
            = 4`. For 32x32x* MFMA the natural per-lane fragment is 16
            elements but they are spread across non-contiguous columns,
            so callers should fan out into per-element calls
            (`chunk_width = 1`) instead.
        elementwise_lambda_fn: Optional fused epilogue.
    """

    var c_ptr_as_int: Int
    """Integer address of the destination's base pointer. Stored as
    Int rather than `UnsafePointer` because the dst tile's origin
    may be any mutable origin and the writer is reused across
    kernels with different origin types."""

    var row_stride: Int
    """Element stride between consecutive rows of dst."""

    var n_total: Int
    """N dimension of the output, used for the chunk-boundary
    detection and the per-element OOB gate."""

    @always_inline
    def __init__(out self, dst: TileTensor[mut=True, Self.c_type, ...]):
        """Build from the (mutable) destination DRAM tile.

        For non-split-K: `dst` is the logical output tensor; `m` in
        subsequent `store()` calls is the logical output row, which
        is also the DRAM row.

        For split-K matmul kernels: `dst` is the
        `(num_splits * M, N)` workspace; `m` in `store()` is the
        workspace row (`split_id * M + pid_m * BM + ...`). Callers
        must keep `elementwise_lambda_fn` unset in that case — see
        the struct doc.

        Args:
            dst: Destination DRAM tile (must be mutable).
        """
        self.c_ptr_as_int = Int(dst.ptr)
        self.row_stride = Int(dst.layout.stride[0]().value())
        self.n_total = Int(dst.dim[1]())

    @always_inline
    def _ptr(self) -> UnsafePointer[Scalar[Self.c_type], MutAnyOrigin]:
        return UnsafePointer[Scalar[Self.c_type], MutAnyOrigin](
            unsafe_from_address=self.c_ptr_as_int
        )

    @always_inline
    def store(
        self,
        v: SIMD[Self.c_type, Self.chunk_width],
        *,
        m: Int,
        n: Int,
    ):
        """Write a SIMD chunk at `(m, n)` of dst.

        The caller has already checked the M bound. If the chunk
        straddles `n_total` (a partial block at the column boundary)
        the writer falls back to per-element stores or lambda calls.

        Args:
            v: SIMD value to write (already cast to `Self.c_type`).
            m: Destination row (DRAM row for split-K workspace, or
                logical output row for non-split-K / reduce-kernel
                lambda mode).
            n: Destination starting column. Caller has typically
                offset by `lane_group * chunk_width` already.
        """
        if n + Self.chunk_width <= self.n_total:
            comptime if Bool(Self.elementwise_lambda_fn):
                comptime epilogue_fn = Self.elementwise_lambda_fn.value()
                epilogue_fn[
                    alignment=align_of[SIMD[Self.c_type, Self.chunk_width]]()
                ](IndexList[2](m, n), v)
            else:
                self._ptr().store[
                    alignment=align_of[SIMD[Self.c_type, Self.chunk_width]]()
                ](m * self.row_stride + n, v)
        elif n < self.n_total:
            for e in range(Self.chunk_width):
                var col = n + e
                if col < self.n_total:
                    comptime if Bool(Self.elementwise_lambda_fn):
                        comptime epilogue_fn = Self.elementwise_lambda_fn.value()
                        epilogue_fn[alignment=align_of[Scalar[Self.c_type]]()](
                            IndexList[2](m, col),
                            SIMD[Self.c_type, 1](v[e]),
                        )
                    else:
                        self._ptr()[m * self.row_stride + col] = v[e]


@always_inline
def _buffer_load_impl[
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    warp_scope: Bool = False,
](
    dst: TileTensor[mut=True, _, _, _, address_space=AddressSpace.LOCAL, ...],
    src: TileTensor[dst.dtype, ...],
    bc: AMDBufferResource,
    base_ptr_as_int: Int,
    dst_layout: Layout,
):
    """Load DRAM data into registers with a caller-specified storage layout.

    Distributes src across threads via thread_layout, reads row-major from
    DRAM (for cache locality), stores into dst using dst_layout strides.
    The dst_layout controls how the M x N per-thread fragment is packed
    into registers; `RegTileLoader` pairs row_major dst with
    `RegTileWriterLDS.copy`'s row-major reads.

    Parameters:
        thread_layout: Thread distribution layout (row_major or col_major).
        num_threads: Total threads; threads beyond layout size are idle.
        warp_scope: If True, uses lane_id() (warp scope).

    Args:
        dst: Destination register tile (LOCAL).
        src: Source DRAM tile (vectorized).
        bc: AMD buffer resource descriptor with OOB bounds.
        base_ptr_as_int: Integer address of the DRAM tile base pointer.
        dst_layout: Layout controlling register storage order. Shape must
            match the per-thread fragment dimensions (M, N).
    """
    var worker_idx = Int(lane_id()) if warp_scope else Int(thread_idx.x)

    comptime if num_threads > thread_layout.size():
        if worker_idx >= thread_layout.size():
            return

    var dist = src.distribute_with_offset[thread_layout](worker_idx)[0]
    var base_offset = Int32(
        (Int(dist.ptr) - base_ptr_as_int) // size_of[dst.dtype]()
    )

    comptime elem_size = type_of(src).element_size
    comptime M = type_of(dist).static_shape[0]
    comptime N = type_of(dist).static_shape[1]
    comptime src_s0 = type_of(dist).static_stride[0]
    comptime src_s1 = type_of(dist).static_stride[1]
    comptime dst_s0 = dst_layout.static_stride[0]
    comptime dst_s1 = dst_layout.static_stride[1]

    comptime for i in range(M):
        comptime for j in range(N):
            dst.raw_store[width=elem_size](
                (i * dst_s0 + j * dst_s1) * elem_size,
                bc.load[dst.dtype, elem_size](
                    base_offset,
                    scalar_offset=Int32(i * src_s0 + j * src_s1),
                ),
            )


# ===----------------------------------------------------------------------=== #
# RegTileWriterLDS
# ===----------------------------------------------------------------------=== #


struct RegTileWriterLDS[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
]:
    """Stateless register→LDS copy expert.

    Sibling to `RegTileLoader` / `RegTileWriter` (DRAM↔reg) and
    `TileLoaderLDS` / `SubTileLoaderLDS` (DRAM→LDS). Writes register
    tiles to shared memory via thread-distributed element stores.

    Parameters:
        thread_layout: Thread distribution layout across the tile.
        swizzle: Optional SMEM swizzle for bank-conflict avoidance.
        num_threads: Number of threads to participate (threads past
            `thread_layout.size()` early-exit).

    Static methods:
        copy         - Standard plain-SMEM write (rank-2 or rank-3
                       distributed layouts); reads src in row-major
                       order to match `RegTileLoader`'s storage.
        copy_blocked - `blocked_product` SMEM write with its own
                       `block_cols` param. Used when thread_layout and
                       SMEM layout have mismatched blocked structure
                       that `distribute_with_offset` can't resolve.
    """

    @staticmethod
    @always_inline
    def copy(
        dst: TileTensor[
            mut=True, _, _, _, address_space=AddressSpace.SHARED, ...
        ],
        src: TileTensor[_, _, _, address_space=AddressSpace.LOCAL, ...],
    ):
        """Copy register data to SMEM, distributed across threads.

        Reads src registers in row-major element order to match the
        storage convention of `RegTileLoader`. Supports both flat
        (rank 2) and hierarchical (rank 3) distributed layouts.

        Args:
            dst: Destination TileTensor in shared memory.
            src: Source TileTensor in local (register) memory.
        """
        comptime num_busy_threads = Self.thread_layout.size()
        comptime elem_size = type_of(dst).element_size

        var worker_idx = Int(thread_idx.x)

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var dist_result = dst.distribute_with_offset[
            Self.thread_layout, Self.swizzle
        ](worker_idx)
        var dst_dist = dist_result[0]

        comptime dist_type = type_of(dst_dist)
        comptime DstVec = SIMD[dist_type.dtype, elem_size]
        comptime rank = dist_type.LayoutType.flat_rank

        # Row-major iteration order matches RegTileLoader's storage.
        # Handles both flat (rank 2) and hierarchical Coord layouts (rank 3).
        comptime if rank == 2:
            comptime R0 = dist_type.static_shape[0]
            comptime R1 = dist_type.static_shape[1]
            comptime s0 = dist_type.static_stride[0]
            comptime s1 = dist_type.static_stride[1]
            comptime for i in range(R0):
                comptime for j in range(R1):
                    comptime src_idx = i * R1 + j
                    comptime dst_off = i * s0 + j * s1
                    dst_dist.raw_store[width=elem_size](
                        dst_off,
                        rebind[DstVec](
                            src.raw_load[width=elem_size](src_idx * elem_size)
                        ),
                    )
        elif rank == 3:
            comptime R0 = dist_type.static_shape[0]
            comptime R1 = dist_type.static_shape[1]
            comptime R2 = dist_type.static_shape[2]
            comptime s0 = dist_type.static_stride[0]
            comptime s1 = dist_type.static_stride[1]
            comptime s2 = dist_type.static_stride[2]
            comptime for i in range(R0):
                comptime for j in range(R1):
                    comptime for k in range(R2):
                        comptime src_idx = i * R1 * R2 + j * R2 + k
                        comptime dst_off = i * s0 + j * s1 + k * s2
                        dst_dist.raw_store[width=elem_size](
                            dst_off,
                            rebind[DstVec](
                                src.raw_load[width=elem_size](
                                    src_idx * elem_size
                                )
                            ),
                        )
        else:
            comptime assert (
                False
            ), "RegTileWriterLDS.copy: unsupported flat_rank"

    @staticmethod
    @always_inline
    def copy_blocked[
        block_cols: Int,
    ](dst: SMemTile[mut=True, _, _, _], src: RegTile[dst.dtype, _, _]):
        """Copy register tile to blocked_product SMEM layout.

        Handles structural mismatches between `thread_layout` and SMEM
        layout by computing per-element SMEM offsets using the
        `blocked_product` formula. Reads registers sequentially as
        `simd_width`-wide vectors; this is invariant to col- vs row-major
        flat ordering when each per-thread row equals one SIMD vector.

        The SMEM layout is `blocked_product` with blocks of
        `dst.shape[0] x block_cols`. `thread_layout` distributes a 2D
        grid of `(data_rows, data_cols/simd_width)` vector positions
        across threads.

        Parameters:
            block_cols: Cols per SMEM block in `blocked_product` layout.

        Args:
            dst: Destination `[block_rows, data_cols]` in SHARED.
            src: Source register tile in LOCAL (row-major elements).
        """
        comptime block_rows = type_of(dst).static_shape[0]
        comptime data_cols = type_of(dst).static_shape[1]
        comptime simd_width = simd_width_of[dst.dtype]()

        var worker_idx = Int(thread_idx.x)

        comptime if Self.num_threads > Self.thread_layout.size():
            if worker_idx >= Self.thread_layout.size():
                return

        # Thread grid dimensions (flat rows/cols in the thread grid).
        # Cols are in vector units (each = simd_width elements).
        comptime tgr = Self.thread_layout.static_shape[0] * (
            1 if Self.thread_layout.flat_rank
            == 2 else Self.thread_layout.static_shape[1]
        )
        comptime tgc = (
            Self.thread_layout.static_shape[1] if Self.thread_layout.flat_rank
            == 2 else Self.thread_layout.static_shape[2]
            * Self.thread_layout.static_shape[3]
        )

        # Number of vector positions per thread in each dimension.
        comptime data_vcols = data_cols // simd_width
        comptime vectors_per_thread = (block_rows * data_vcols) // (tgr * tgc)
        comptime cols_per_blk = block_cols // simd_width

        # Distribute thread ID → (row, vcol) using UInt32 bitwise ops.
        # All grid dimensions are power-of-2 so divmod compiles to shift/mask.
        var tid = UInt32(thread_idx.x)
        var base_row = tid // UInt32(tgc)
        var base_vcol = tid % UInt32(tgc)

        # blocked_product base address: compute within-block super-element
        # index, apply swizzle once, then use compile-time row deltas for
        # subsequent stores.  Inter-row stride bits are above the swizzle
        # range, so swz(base + delta) == swz(base) + delta.
        var blk = base_vcol // UInt32(cols_per_blk)
        var col_in_blk = base_vcol % UInt32(cols_per_blk)
        var local_idx = base_row * UInt32(cols_per_blk) + col_in_blk
        comptime if Self.swizzle:
            comptime swizzle_fn = Self.swizzle.value()
            local_idx = UInt32(swizzle_fn(Int(local_idx)))

        var base_offset = blk * UInt32(
            block_rows * block_cols
        ) + local_idx * UInt32(simd_width)

        # Compile-time vector stride (above swizzle range for typical configs).
        comptime row_delta = tgr * cols_per_blk * simd_width

        comptime for v in range(vectors_per_thread):
            dst.raw_store[width=simd_width](
                Int(base_offset + UInt32(v * row_delta)),
                src.raw_load[width=simd_width](v * simd_width),
            )


# ===----------------------------------------------------------------------=== #
# Tile type aliases
# ===----------------------------------------------------------------------=== #


comptime GMemTile[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
] = TileTensor[dtype, LayoutType, origin]
"""Global memory tile. Alias for TileTensor in default (GENERIC) address space."""


comptime SMemTile[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
] = TileTensor[dtype, LayoutType, origin, address_space=AddressSpace.SHARED]
"""Shared memory tile. Alias for TileTensor in SHARED address space."""


comptime RegTile[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
] = TileTensor[dtype, LayoutType, origin, address_space=AddressSpace.LOCAL]
"""Register tile. Alias for TileTensor in LOCAL address space."""
