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

"""Provides shared utilities for SM100 MLA decode attention kernels.

Defines TMA tile helpers, pipeline producer/consumer structs, MMA tensor
accumulator descriptors, and the common softmax/correction/store logic reused
across the BF16, native FP8, and per-token-scale decode backends.
"""

from std.collections import OptionalReg
from std.math import exp2, recip, align_up, log2, ceildiv
from std.math.constants import log2e
from std.sys import get_defined_int, size_of, align_of, _RegisterPackType
from max.gpu import thread_idx, block_idx, block_dim, warp_id
from max.gpu.sync import barrier
from max.gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    create_tma_descriptor,
)
from max.gpu.host.info import B200
from max.gpu.memory import fence_async_view_proxy, external_memory
from max.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAInsDescriptor,
    UMMAKind,
    mma_arrive_multicast,
)

from max.gpu.sync import named_barrier, syncwarp
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_st,
    tcgen05_store_wait,
)
from max.gpu.primitives.warp import shuffle_xor, vote
from max.gpu.compute.mma import _str_iota
from max.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from layout import TileTensor, row_major, Idx, stack_allocation, coord
from layout.layout import row_major as tt_row_major
from layout.swizzle import make_ldmatrix_swizzle
from layout.tma_async import (
    create_tensor_tile,
    PipelineState,
    _default_desc_shape,
    TMATensorTile,
)
from std.memory import bitcast
from nn.attention.gpu.nvidia.common import (
    NullPointer,
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask, MASK_VALUE
from nn.attention.mha_utils import null_pointer, unread_pointer
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.mla_common import (
    SparseRowLayout,
    _sparse_row_packed,
    assert_sparse_row_packed,
)
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple
from layout import Coord
from linalg.arch.sm100.mma import smem_descriptor

from nn.attention.gpu.nvidia.sm100.attention import SM100_RESERVED_SMEM_BYTES
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    LocalTensor,
    SharedMemPointer,
    MBarType,
    elect_mma_arrive,
    ProducerPipeline,
    ConsumerPipeline,
    MBarPipeline,
    sub_ftz,
    bulk_mma,
    bulk_mma_ws,
    bulk_mma_ws_ts,
    st_shared_v4_b32,
    opaque_nonneg_i32,
    opaque_u32,
    llvm_opaque_tid,
)
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
)
from nn.attention.gpu.nvidia.common import KVTMATile
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys._assembly import inlined_assembly
from nn.attention.gpu.nvidia.sm100.mla_stamp import (
    stamp_begin,
    stamp_finish,
    stamp_tile,
)


@inline(.always)
def _sparse_physical_sentinel_lane[
    role_local_lane: Bool = False
](raw_row: MutPointer[Int32, MutAnyOrigin], readable: Int) -> Int32:
    # One slot per lane; the warp's ballot turns the warp's slots into the
    # tile's validity word. Slots past the row's readable tail read nothing.
    # Callers must have one `col_base` per warp (Layout E's softmax mapping),
    # so every lane of the ballot covers a different slot of the same tile.
    var lane: Int
    comptime if role_local_lane:
        lane = Int(llvm_opaque_tid()) & 31
    else:
        lane = Int(thread_idx.x) & 31
    if lane < readable:
        return raw_row[lane]
    return Int32(-1)


@inline(.always)
def _sparse_physical_sentinel_lane_hi[
    role_local_lane: Bool = False
](raw_row: MutPointer[Int32, MutAnyOrigin], readable: Int) -> Int32:
    # The second 32-slot word of a 64-column thread window: slot `lane + 32`
    # of the same row, so the window takes two warp ballots.
    var lane: Int
    comptime if role_local_lane:
        lane = Int(llvm_opaque_tid()) & 31
    else:
        lane = Int(thread_idx.x) & 31
    if lane + 32 < readable:
        return raw_row[lane + 32]
    return Int32(-1)


@inline(.always)
def _sparse_physical_sentinel_live_pair(raw_lo: Int32, raw_hi: Int32) -> Bool:
    # A thread window wider than 32 slots needs one ballot per 32 slots; the
    # tile's mask is the identity only when every word is all live.
    comptime all_live = UInt32((UInt64(1) << UInt64(32)) - UInt64(1))
    return (
        vote[.uint32](raw_lo >= 0) == all_live
        and vote[.uint32](raw_hi >= 0) == all_live
    )


@inline(.always)
def _sparse_physical_sentinel_live_window[
    cols: Int
](raw_lane: Int32, readable: Int, shift: Int) -> Bool:
    # The same all-live test for a window narrower than a warp's 32 ballot
    # slots: a Layout G thread owns the 16 slots at `shift` and a half-row
    # thread the 8 there, while the ballot covers the whole word from the
    # warp's aligned base. Slots outside the window belong to another thread's
    # window, so their state says nothing about this one.
    comptime mask = UInt32((UInt64(1) << UInt64(cols)) - UInt64(1))
    var live = (vote[.uint32](raw_lane >= 0) >> UInt32(shift)) & mask
    if readable >= shift + cols:
        return live == mask
    return False


@inline(.always)
def _sparse_physical_sentinel_live_word[
    cols: Int
](raw_lane: Int32, readable: Int) -> Bool:
    # Every one of this thread's `cols` slots holds a live physical row: the
    # tile is fully visible, so its mask is the identity. A slot at or past
    # `readable` lies past the row's valid length, and a negative row is
    # padding the gather clamped to an in-range duplicate; either makes the
    # mask non-identity.
    comptime all_live = UInt32((UInt64(1) << UInt64(cols)) - UInt64(1))
    var live = vote[.uint32](raw_lane >= 0) == all_live
    if readable >= cols:
        return live
    return False


@inline(.always)
def _sparse_physical_sentinel_bits[
    use_ballot: Bool, role_local_lane: Bool = False
](raw_row: MutPointer[Int32, MutAnyOrigin], readable: Int) -> UInt32:
    # Requires 0 <= readable <= 32; ballot callers pass one row and bound
    # uniformly across a full warp, including lanes beyond the readable tail.
    comptime if use_ballot:
        var lane: Int
        comptime if role_local_lane:
            lane = Int(llvm_opaque_tid()) & 31
        else:
            lane = Int(thread_idx.x) & 31
        var raw: Int32 = -1
        if lane < readable:
            raw = raw_row[lane]
        return vote[DType.uint32](raw >= 0)
    else:
        comptime batch = 8
        var valid_bits: UInt32 = 0
        var slot = 0
        while slot + batch <= readable:
            var raw = raw_row.unsafe_load[width=batch, alignment=4](slot)
            var valid = raw.ge(SIMD[DType.int32, batch](0))
            comptime for j in range(batch):
                if valid[j]:
                    valid_bits |= UInt32(1) << UInt32(slot + j)
            slot += batch
        while slot < readable:
            if raw_row[slot] >= 0:
                valid_bits |= UInt32(1) << UInt32(slot)
            slot += 1
        return valid_bits


# ------------------------------------------------------------------------------
# Helper functions for MLA decoding TMA tiles
# ------------------------------------------------------------------------------

comptime QOTMATile[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
] = TMATensorTile[
    dtype,
    coord[BM, BK],
    _default_desc_shape[dtype, coord[BM, BK], swizzle_mode](),
    is_k_major=True,
]


@inline(.always)
def tma_tile_qo[
    dtype: DType,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    BK: Int,
    depth: Int,
](
    ctx: DeviceContext,
    ptr: MutPointer[Scalar[dtype], _],
    rows: Int,
    out res: QOTMATile[dtype, BM, BK, swizzle_mode],
) raises:
    """Creates a TMA descriptor for the Q or output tensor used in MLA decode.

    Parameters:
        dtype: Element type of the Q or output tensor (inferred).
        swizzle_mode: TMA swizzle mode applied to the descriptor.
        BM: Tile height in rows for each TMA copy.
        BK: Tile width in columns for each TMA copy.
        depth: Column count of the full Q or output tensor.

    Args:
        ctx: Device context used to create the TMA descriptor.
        ptr: Base pointer of the Q or output tensor in device memory.
        rows: Number of rows in the full Q or output tensor.
    """
    var rt_layout = row_major((Int64(rows), Idx[depth]))
    var tensor = TileTensor(ptr, rt_layout)

    res = rebind[QOTMATile[dtype, BM, BK, swizzle_mode]](
        create_tensor_tile[
            coord[BM, BK],
            swizzle_mode=swizzle_mode,
        ](ctx, tensor)
    )


# Output TMA tile whose stored row count is chosen per copy. The row axis is
# its own descriptor dim of extent BM, so a row coordinate of
# BM - rows_to_store leaves rows_to_store box rows in bounds and the TMA
# masks the rest. tma_tile_o builds the descriptor and store_row_coords the
# coordinate.
# Similar ragged masking technique as RaggedTMA3DTile in
# layout/tma_async.mojo; consider merging the two in the future.
comptime ORaggedTMATile[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
] = TMATensorTile[
    dtype,
    coord[1, BM, BK],
    _default_desc_shape[dtype, coord[1, BM, BK], swizzle_mode](),
    is_k_major=True,
]


@inline(.always)
def tma_tile_o[
    dtype: DType,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    BK: Int,
    depth: Int,
](
    ctx: DeviceContext,
    ptr: MutPointer[Scalar[dtype], _],
    rows: Int,
    out res: ORaggedTMATile[dtype, BM, BK, swizzle_mode],
) raises:
    """Creates the MLA decode output TMA descriptor.

    The row axis is its own descriptor dim of extent BM, reached through a
    separate outer dim with the same row stride. This makes the stored row
    count a per-copy coordinate, built by store_row_coords. Box row r lands
    on tensor row row + r, and box rows at or past rows_to_store fall
    outside the extent and are dropped. The descriptor base sits BM rows
    before ptr, but masked rows are never written, so memory before ptr is
    never touched.

    Parameters:
        dtype: Element type of the output tensor (inferred).
        swizzle_mode: TMA swizzle mode applied to the descriptor.
        BM: Row extent of the box.
        BK: Tile width in columns for each TMA copy.
        depth: Column count of the full output tensor.

    Args:
        ctx: Device context used to create the TMA descriptor.
        ptr: Base pointer of the output tensor in device memory.
        rows: Number of rows in the full output tensor.
    """
    # Outer coordinates run up to rows, so the extent is one past that. Its
    # box is 1, so it never masks.
    res = create_tma_descriptor[dtype, swizzle_mode=swizzle_mode](
        DeviceBuffer(ctx, ptr.unsafe_offset(-depth * BM), 1, owning=False),
        (rows + 1, Idx[BM], depth),
        (depth, depth, Idx[1]),
        _default_desc_shape[dtype, coord[1, BM, BK], swizzle_mode](),
    )


@inline(.always)
def store_row_coords[
    BM: Int
](col: Int, row: Int, rows_to_store: Int) -> Tuple[Int, Int, Int]:
    """Builds the ORaggedTMATile store coordinate for a partial row count.

    Args:
        col: Column offset within the output row.
        row: First output tensor row the copy writes.
        rows_to_store: How many rows of the box are real, at most BM.

    Returns:
        Coordinates for async_store_3d, ordered innermost dim first.
    """
    return (col, BM - rows_to_store, row + rows_to_store)


# Per-token scales TMA tile: loads BN_QK contiguous float32 values via TMA.
# Scales are treated as a flat 1D array indexed by row_idx (same paging
# as the KV cache blocks). The TMA uses a [1, total_elements] 2D layout
# so the inner dimension (total_elements * 4 bytes) exceeds the TMA minimum
# of 32 bytes, with tile shape [1, BN_QK] and SWIZZLE_NONE.
#
# We set desc_shape = tile_shape (no sub-tiling) so that desc_bytes ==
# tile_bytes and the 128-byte alignment constraint for multi-copy TMA is
# not triggered. With BN_QK=64, tile_bytes = 256 which is already 128-aligned.
comptime ScalesTMATile[BN_QK: Int] = TMATensorTile[
    DType.float32,
    coord[1, BN_QK],
    is_k_major=True,
]


@inline(.always)
def tma_tile_scales[
    BN_QK: Int,
](
    ctx: DeviceContext,
    ptr: MutPointer[Float32, _],
    total_elements: Int,
    out res: ScalesTMATile[BN_QK],
) raises:
    """Create a TMA descriptor for per-token float32 scales.

    The scales are a flat array of float32 values indexed by the same
    row_idx as the KV cache blocks. We create a 2D TMA with shape
    [1, total_elements] and tile [1, BN_QK] so that each async_copy loads
    BN_QK contiguous float32 values (BN_QK * 4 bytes) starting at the
    specified column offset.

    Parameters:
        BN_QK: Tile width in float32 values loaded by each async_copy,
            equal to the KV cache tile width.

    Args:
        ctx: Device context used to create the TMA descriptor.
        ptr: Base pointer of the flat float32 scales array in device
            memory, indexed by the same row index as the KV cache blocks.
        total_elements: Total number of float32 scales in the array,
            used as the inner (column) dimension of the 2D TMA descriptor.
    """
    var rt_layout = row_major((Idx[1], total_elements))
    var tensor = TileTensor(ptr, rt_layout)
    res = rebind[ScalesTMATile[BN_QK]](
        create_tensor_tile[
            coord[1, BN_QK],
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            __desc_shape=coord[1, BN_QK],
        ](ctx, tensor)
    )


# ------------------------------------------------------------------------------
# Helper functions for MLA decoding pack
# ------------------------------------------------------------------------------


struct _MLADecodePackLegacyABI[
    ValidLengthType: OptionalPointer,
    MaskType: MHAMask,
    SplitAccumType: OptionalPointer,
](Copyable, TrivialRegisterPassable):
    var mask: Self.MaskType
    var valid_length: Self.ValidLengthType
    var lse_accum_split_ptr: Self.SplitAccumType
    var num_partitions: Int64


struct MLA_Decode_Pack[
    ValidLengthType: OptionalPointer,
    MaskType: MHAMask,
    SplitAccumType: OptionalPointer,
    prepared_bounds: Bool = False,
](Copyable, DevicePassable, TrivialRegisterPassable):
    """Bundles the mask, valid-length, and split-K accumulator pointers passed to decode kernels.

    Parameters:
        ValidLengthType: `OptionalPointer` type wrapping the per-batch
            valid-sequence-length tensor (may be `Null` when unused).
        MaskType: `MHAMask` type applied to the attention scores.
        SplitAccumType: `OptionalPointer` type wrapping the split-K LSE
            accumulator buffer (may be `Null` when split-K is unused).
        prepared_bounds: Whether the pack carries the two prepared row bounds
            (defaults to `False`).
    """

    var mask: Self.MaskType
    var valid_length: Self.ValidLengthType
    var lse_accum_split_ptr: Self.SplitAccumType
    var num_partitions: Int64
    comptime BoundsType = StaticTuple[Int64, 2 if Self.prepared_bounds else 0]
    var bounds: Self.BoundsType
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "Pack"

    @staticmethod
    def get_device_type_name() -> String:
        return Self.get_type_name()

    @inline(.always)
    def __init__(
        out self,
        mask: Self.MaskType,
        valid_length: Self.ValidLengthType,
        lse_accum_split_ptr: Self.SplitAccumType,
        num_partitions: Int,
        batch_size: Int = 0,
        q_max_seq_len: Int = 0,
    ):
        comptime if not Self.prepared_bounds:
            comptime Legacy = _MLADecodePackLegacyABI[
                Self.ValidLengthType, Self.MaskType, Self.SplitAccumType
            ]
            comptime assert size_of[Self]() == size_of[Legacy]()
            comptime assert align_of[Self]() == align_of[Legacy]()
        self.mask = mask
        self.valid_length = valid_length
        self.lse_accum_split_ptr = lse_accum_split_ptr
        self.num_partitions = Int64(num_partitions)
        self.bounds = Self.BoundsType()
        comptime if Self.prepared_bounds:
            var values = StaticTuple[Int64, 2](
                Int64(batch_size), Int64(q_max_seq_len)
            )
            self.bounds = rebind[Self.BoundsType](values)

    @inline(.always)
    def batch_size(self) -> Int:
        comptime assert Self.prepared_bounds
        return Int(rebind[StaticTuple[Int64, 2]](self.bounds)[0])

    @inline(.always)
    def q_max_seq_len(self) -> Int:
        comptime assert Self.prepared_bounds
        return Int(rebind[StaticTuple[Int64, 2]](self.bounds)[1])


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------


@inline(.always)
def num_matrix_view_rows_decode[
    dtype: DType,
    //,
](q: TileTensor[dtype, ...]) -> Int:
    """TileTensor overload of `num_matrix_view_rows_decode`.

    Parameters:
        dtype: Element type of the query tile tensor (inferred).

    Args:
        q: Query tile tensor whose leading `rank - 1` dimensions are
            multiplied to compute the total matrix view row count.
    """
    # q and output are (batch x seq_len x num_heads , depth)
    # output when split-k is used are (split_k x batch x seq_len x num_heads , depth)
    var num_rows = Int(q.dim[0]())

    comptime for i in range(1, q.rank - 1):
        num_rows *= Int(q.dim[i]())
    return num_rows


# SharedMemPointer and MBarType are imported from
# nn.attention.gpu.nvidia.sm100.attention_utils (centralized shared memory type aliases).


# With fp8 P, P = exp2(s - m) is converted to e4m3 before the P@V MMA.
# e4m3 rounds values below 2^-10 to zero while the fp32 row sum still
# includes them, so the output is too small. P is scaled by
# 2^fp8_p_bias before the conversion. The lazy rescale lets P reach
# 2^-threshold, so fp8 caps the threshold at -2 and uses a scale of
# 2^(8 + threshold): P stays at or below 2^8, under the e4m3 maximum of
# 448, and values down to 2^-16 of the row max are kept. The row sum
# carries the same factor, which cancels in O / l; the LSE and the
# attention-sink term subtract it.
comptime _FP8_P_MAX_RESCALE_THRESHOLD: Float32 = -2.0

# Half-row softmax (`MLA_SM100_Decode_Config.half_rows`) negative controls,
# test-only (`MLA_HALF_ROWS_BREAK`): 1 skips the pair max; 3 writes both
# halves' P at half 0's columns; 5 hands each half the other half's ballot
# bits; 6 sends every split to split 0's keys; 7 shortens every split by one
# page.
comptime _MLA_HALF_ROWS_BREAK = get_defined_int["MLA_HALF_ROWS_BREAK", 0]()

# Split-K decodes launch with PDL (OVERLAP_AT_END), so they may start before
# the kernel that wrote their inputs, or still reads their workspaces, has
# finished. 0 is the old order, kept only as the race test's control.
comptime _MLA_DECODE_PDL_WAIT = get_defined_int["MLA_DECODE_PDL_WAIT", 1]()
# Nonzero is main's policy (#106899): PDL for every nn kernel launch on
# NVIDIA, not only split-K. Each decode launch attaches the PDL attribute,
# and each kernel waits on its predecessor at entry and releases its
# dependents at the end of its work, before the TMEM dealloc tail. Main's
# scope form (`with PDL[overlap_at_beginning=True]()`) releases at entry
# instead; on this kernel's split cells that costs 7.5% (combine CTAs go
# resident for the whole decode and contend its SMs, forced-np8 B200
# screening), so the release keeps the measured tail placement while the
# attribute and the wait go unconditional. Early-exit paths that predate the
# gate still release at grid completion, which is safe, only later. 0
# restores the split-K-only gate; the package reads this at precompile, so
# flipping it is a source edit, not a `-D`.
comptime _MLA_PDL_ALWAYS = get_defined_int["MLA_PDL_ALWAYS", 1]()
# Test-only (`mla_decode_combine.mojo`): 2 drops the non-split sparse decode's
# ragged exit.
comptime _MLA_RAGGED_EXIT_BREAK = get_defined_int[
    "MLA_COMBINE_RAGGED_EXIT_BREAK", 0
]()
# Test-only (`MLA_CORR_BREAK`): 1 skips the correction's O-rescale TMEM walk
# (the barriers stay, so the pipeline runs; results are wrong by the skipped
# rescale), which is the fault `_break_corr` plants for the head-padding cells
# that need two key tiles.
comptime _MLA_CORR_BREAK = get_defined_int["MLA_CORR_BREAK", 0]()
# Layout E FP8 direct gather (default; `-D MLA_SPARSE_DIRECT_GATHER_E=0`
# restores the linear ring plus re-swizzle): the unified sparse kernel's
# Layout E form gathers K/V straight into its SW128 ring, the routed Layout G
# front end. The FP8 operand keeps its separate P ring; `p_in_kv` stays 2-byte
# only, because the softmax's FP8 P write has no rope-block path.
comptime _MLA_SPARSE_DIRECT_GATHER_E = get_defined_int[
    "MLA_SPARSE_DIRECT_GATHER_E", 1
]()
# CTA pair at 128 heads (the default; `-D MLA_SPARSE_CG2=0` is the escape
# hatch to the unpaired direct kernel): the unified sparse kernel's FP8
# Layout E form runs as a 2-CTA cluster. The pair issues one `cta_group::2`
# QK MMA over both CTAs' 64-head Q tiles and both CTAs' 64-key K halves,
# and splits each PV atom on N so each CTA supplies the half of V's depth
# columns it owns. Each CTA gathers its key half of K into its Z ring and
# all pair keys of its V depth slice into its own V ring, with Z one tile
# ahead of V, relaxed cluster syncs, a staged producer tail and the
# softmax's unmasked fast path.
# `MLA_SPARSE_CG2`, `MLA_SPARSE_CG2_ROUTE` and `MLA_SPARSE_CG2_BF16` also set
# the 128-head sparse graph capture key (`compute_mla_dispatch_key_runtime`),
# which runs in the `_kv_cache_ops` Python extension and reads the values that
# extension was built with, not the kernels'. Build both with the same values:
# a knob set on the kernels alone still runs correctly, but its captures are
# keyed on the other launch and replay the wrong split or kernel.
comptime _MLA_SPARSE_BN_QK = get_defined_int["MLA_SPARSE_BN_QK", 0]()
comptime _MLA_SPARSE_CG2 = get_defined_int["MLA_SPARSE_CG2", 1]()
# Short walks leave the pair (`MLA_SPARSE_CG2_ROUTE`, on by default): a
# launch whose critical CTA walks at most `_SPARSE_PAIR_MAX_DIRECT_WALK` keys
# takes the unpaired direct kernel, and one with at most
# `_SPARSE_PAIR_NO_SPLIT_KEYS` keys per token does not split (see
# mla_decode_dispatch.mojo). 0 keeps every pair-eligible launch on the pair
# at the split the policy would otherwise pick; the pair's A/B and
# verification arms set it so their short cells still reach the pair.
comptime _MLA_SPARSE_CG2_ROUTE = get_defined_int[
    "MLA_SPARSE_CG2_ROUTE", 1 if _MLA_SPARSE_CG2 != 0 else 0
]()
# The BF16 h=128 pair (the default; `-D MLA_SPARSE_CG2_BF16=0` is the BF16
# off switch): BF16 Q over a BF16 cache takes the same V-gather pair at a
# 64-key pair tile (32 keys per CTA), the largest whose Z, V and P rings fit
# two stages beside a 2-byte Q. P keeps its own ring, written as BF16.
# Requires `MLA_SPARSE_CG2`.
comptime _MLA_SPARSE_CG2_BF16 = get_defined_int["MLA_SPARSE_CG2_BF16", 1]()
# NVFP4 sparse decode A/B arms. `MLA_NVFP4_DEQUANT` picks the dequant: 1
# (the default) converts every element through the hardware e2m1 / e4m3
# converts, 0 builds each group's e4m3 table in f32, 2 builds the table
# from f16 products. All three are bit-identical.
comptime _MLA_NVFP4_DEQUANT = get_defined_int["MLA_NVFP4_DEQUANT", 1]()
# WG3 loads every packed chunk of its row before it converts or stores any,
# so no store sits between two loads (`MLA_NVFP4_BATCH=0` interleaves them).
comptime _MLA_NVFP4_BATCH = get_defined_int["MLA_NVFP4_BATCH", 1]()
# NVFP4 sparse decode (`MLA_NVFP4_PACKED_LINEAR`, on by default): the
# linear front end's gather slot holds only the tile's packed rows and
# scales (22,528 B at a 576-deep row) instead of a full FP8 stage
# (36,864 B), freeing 14,336 B per slot. At Layout E that is what fits the
# form in the carveout at 64 and 128 heads -- full-size slots overrun it by
# 28 B, so those head counts compiled no decode form at all -- and at
# Layout G the config spends the freed SMEM on a third gather stage.
# `-D MLA_NVFP4_PACKED_LINEAR=0` restores the full-size slot everywhere:
# the Layout E carve fails again, and Layout G returns to its measured 2
# gather / 3 MMA stages. The form reads each selected slot through the page
# table one row at a time, so a selection spanning many pages is its normal
# case. Prefill reaches this form only through
# `mla_sparse_prefill_on_decode_kernel`.
comptime _MLA_NVFP4_PACKED_LINEAR = get_defined_int[
    "MLA_NVFP4_PACKED_LINEAR", 1
]()
# `MLA_Q_TMEM=1`: the 64-key linear front end keeps all of Q in TMEM (the
# QK MMA's TS form), so Q's SMEM lands in, and then becomes, one more SW64
# MMA stage.
comptime _MLA_Q_TMEM = get_defined_int["MLA_Q_TMEM", 0]()
# Test-only negative controls for the pair (`MLA_CG2_SLICE_BREAK`): 1
# gathers the leader's V depth slice into the follower's V ring; 2 gathers
# the leader's key half into the follower's Z slot. Both move the same
# bytes, so the pipeline runs and only the results are wrong.
comptime _MLA_CG2_SLICE_BREAK = get_defined_int["MLA_CG2_SLICE_BREAK", 0]()
# The pair's index ring depth. The pair holds an index slot until V's
# issue has read its all-key quads, and that issue follows the next Z
# issue in the same lane, so at the K/V ring's two stages the index refill
# (about 700 cycles) lands on the K recurrence every tile; four stages
# take it off. A build that needs SMEM back for `MLA_STAMP`'s reserve runs
# three (`-D MLA_SPARSE_PAIR_IDX_STAGES=3`).
comptime _SPARSE_PAIR_IDX_STAGES = get_defined_int[
    "MLA_SPARSE_PAIR_IDX_STAGES", 4
]()
# The pair's mask-free softmax fast path, extended to the masked
# (`skip_masked_tiles`) arm: a tile whose every column carries a live
# gathered index has an identity mask, so it takes the same mask-free max as
# an unmasked tile instead of the generic elementwise mask. 0 restores the
# generic mask on every tile of that arm.
comptime _MLA_SMX_SENTINEL_FAST = get_defined_int["MLA_SMX_SENTINEL_FAST", 1]()
# Gathered-row sentinel check: 1 (default) has each lane load one index and
# the warp ballot the live bits; 0 has every thread scan the whole row.
comptime _MLA_SPARSE_SENTINEL_BALLOT = get_defined_int[
    "MLA_SPARSE_SENTINEL_BALLOT", 1
]()
# The unpaired forms' mask-free softmax max: every form takes it by default
# behind a per-head-count mask, and a form only gains the masked arm's
# sentinel check where the ballot window matches the thread's columns.
# `MLA_SMX_UNPAIRED` overrides both masks: 1 on for every unpaired form, 0 off
# for every unpaired form, -1 (default) reads the two masks.
comptime _MLA_SMX_UNPAIRED = get_defined_int["MLA_SMX_UNPAIRED", -1]()
# Bit per head count: 8, 16, 32, 64, 128 (see `_mla_smx_head_bit`). The
# plain arm wins at h=32 and h=128 and loses about 0.6% on prompt-shaped cells
# at h=8, h=16 and h=64, so those keep the generic mask. The masked arm wins
# at every head count: 1-6% at h=128 and h=8/16/32, and at h=64 it ties on
# long cells and takes 2-8% off short and prompt-shaped ones.
comptime _MLA_SMX_UNPAIRED_HEADS = get_defined_int[
    "MLA_SMX_UNPAIRED_HEADS", 20
]()
comptime _MLA_SMX_UNPAIRED_MASK_HEADS = get_defined_int[
    "MLA_SMX_UNPAIRED_MASK_HEADS", 31
]()
# The FP8 cache's sets, measured separately: the masked arm is worth about
# 12% at h=64, where the head mask is on. The plain arm has no FP8
# measurement, so nothing is on there.
comptime _MLA_SMX_UNPAIRED_FP8_HEADS = 0
comptime _MLA_SMX_UNPAIRED_FP8_MASK_HEADS = 8
# Layout G and the half-row softmax give a thread 16 or 8 of a warp's 32
# ballot slots, so the all-live word needs the shifted window
# `_sparse_physical_sentinel_live_window`; 0 leaves their masked arm on the
# generic mask.
comptime _MLA_SMX_WINDOW_SENTINEL = get_defined_int[
    "MLA_SMX_WINDOW_SENTINEL", 1
]()
# Debug-only (`MLA_SPARSE_SENTINEL_TRAP=1`): re-enable the gathered-row
# live-index check on the mask-free path and fail a tile the per-thread bound
# admitted but the check rejects. Never a default: the bound is meant to
# subsume the check, and this is how that is verified (the failing tile's max
# is poisoned so a numeric gate fails even where a trap does not).
comptime _MLA_SMX_SENTINEL_TRAP = get_defined_int[
    "MLA_SPARSE_SENTINEL_TRAP", 0
]()
# Planted fault for the bound-only path: admit this many columns past the
# bound, so a short row's masked tail takes the mask-free body and its numeric
# gate must fail. Timing-only ceiling when set to a large value.
comptime _MLA_SMX_BOUND_SLACK = get_defined_int["MLA_SMX_BOUND_SLACK", 0]()
# The bound-only masked arm needs the row's live count from the bound alone, and
# only a causal mask supplies it: the row's valid entries are packed, so the
# count is `min(top_k, causal count)` and the causal term is in the bound. A
# null mask's bound is the caller's stated length, which the kernel cannot
# check against the row, so those forms keep the gathered-row index check.
# 1 is a planted fault
# (`test_mla_prefill_sparse_nvfp4_convert_once_nullmask_fast` must fail): it
# admits the null-mask forms without the check.
comptime _MLA_SMX_NULL_MASK_FAST = get_defined_int[
    "MLA_SMX_NULL_MASK_FAST", 0
]()
# FA4's split loop: the walk becomes a prefix whose tiles are wholly inside the
# bound (mask-free body, no per-tile test, the split point computed once before
# the walk and uniform across the softmax's threads) followed by a masked tail.
# `MLA_SMX_SPLIT_LOOP` 1 forces it on every fast form (the A/B arm), 0 keeps the
# per-tile predicate everywhere, and the default (-1) reads
# `_MLA_SMX_SPLIT_HEADS`: on where a split walks many wholly visible tiles and
# off where it walks few, since the second copy of the body is only repaid over
# a long walk. The BF16/FP8 pair wins 0.4-1.3% on prompt-shaped walks and loses
# 2-5% on its multi-token decode splits; the decode cells run the pair's plain
# arm and the prompts its masked (prefill-on-decode) arm, so the default turns
# it on for the masked pair only.
comptime _MLA_SMX_SPLIT_LOOP = get_defined_int["MLA_SMX_SPLIT_LOOP", -1]()
# Bit per head count, as `_MLA_SMX_UNPAIRED_HEADS`: the NVFP4 unpaired forms at
# h=64 and h=128 win 0.3-1.2% on every measured cell, h=8/16/32 are mixed.
comptime _MLA_SMX_SPLIT_HEADS = 24


def _mla_smx_head_bit[heads: Int]() -> Int:
    """Returns the bit `_MLA_SMX_*_HEADS` uses for this head count."""
    comptime if heads == 8:
        return 0
    elif heads == 16:
        return 1
    elif heads == 32:
        return 2
    elif heads == 64:
        return 3
    elif heads == 128:
        return 4
    return -1


@inline(.always)
def decode_pdl_wait[pdl: Bool]():
    """Orders a PDL-launched decode after its predecessor grid.

    Every thread calls it before the kernel's first global access, early
    exits included.

    Parameters:
        pdl: Whether the kernel is launched with PDL.
    """
    comptime if pdl and _MLA_DECODE_PDL_WAIT != 0:
        wait_on_dependent_grids()


# Direct gather test-only negative controls (`MLA_P6_BREAK`): 5 skips the
# rope column group in the gather; 8 stores tile t + 1's prefetched indices
# into tile t's slot.
comptime _MLA_P6_BREAK = get_defined_int["MLA_P6_BREAK", 0]()
# Direct gather ring depths: the S, P and index rings take N = 3. The K/V
# ring, whose WAR back-edge sets the steady-state rate, takes the rest of
# SMEM (M = 5 at 64-key tiles, with or without split-K; M = 5 beats 4 by
# 2-7% at B = 16 to 64).
comptime _DIRECT_RING_N = 3
# Width of one SW64 FP8 column block (64 B). The SMEM layouts of Q, K and V
# are built from these blocks whatever the keys per tile.
comptime SW64_COLS = 64

# Thread map of `tcgen05.ld/st.16x32bx2` (measured on B200): thread `t` of a
# warp holds TMEM lane `t & 15` of its quadrant, and register `r` is column
# `r + imm * (t >> 4)`, so the two threads of a row are `HR_PAIR_XOR` apart
# and each holds a contiguous half.
comptime HR_PAIR_XOR = 16


@inline(.always)
def hr_row(t: Int) -> Int:
    """Returns the row (TMEM lane in the quadrant) warp lane `t` holds."""
    return t & 15


@inline(.always)
def hr_half(t: Int) -> Int:
    """Returns which column half warp lane `t` holds (0 or 1)."""
    return t >> 4


@inline(.always)
def tcgen05_ld_16x32bx2[
    repeat: Int, imm: Int
](tmem_addr: UInt32) -> Array[Float32, repeat]:
    """Loads `tcgen05.ld.16x32bx2.x<repeat>` with column split `imm`.

    The shared wrapper rejects this shape. `tmem_addr` must carry the warp's
    quadrant base in its lane field.
    """
    comptime assert repeat in [8, 16, 32], "repeat must be 8, 16 or 32"
    comptime asm = (
        "tcgen05.ld.sync.aligned.16x32bx2.x"
        + String(repeat)
        + ".b32 {"
        + _str_iota[repeat, prefix="$", sep=","]()
        + "}, [$"
        + String(repeat)
        + "], "
        + String(imm)
        + ";"
    )
    comptime constraints = "=f," * repeat + "r"
    comptime S = Float32

    @inline(.nodebug)
    def ld[pack_type: TrivialRegisterPassable]() {imm} -> Array[S, repeat]:
        var r = inlined_assembly[
            asm, pack_type, constraints=constraints, has_side_effect=True
        ](tmem_addr)
        var ptr = Pointer(to=r).unsafe_bitcast[S]()
        return Array[S, repeat](
            fill_with_unrolled=lambda [i: Int]() -> S: ptr[unsafe_offset=i]
        )

    # fmt: off
    comptime if repeat == 8:
        return ld[_RegisterPackType[S, S, S, S, S, S, S, S]]()
    elif repeat == 16:
        return ld[_RegisterPackType[S, S, S, S, S, S, S, S,
                                    S, S, S, S, S, S, S, S]]()
    else:
        return ld[_RegisterPackType[S, S, S, S, S, S, S, S,
                                    S, S, S, S, S, S, S, S,
                                    S, S, S, S, S, S, S, S,
                                    S, S, S, S, S, S, S, S]]()
    # fmt: on


@inline(.always)
def write_fp8_half_row_to_smem[
    local_tile_size: Int,
    *,
    out_dtype: DType,
    in_dtype: DType,
    config: MLA_SM100_Decode_Config,
](
    shared_mem: Pointer[Scalar[out_dtype], MutAnyOrigin, address_space=.SHARED],
    local_mem: LocalTensor[in_dtype, row_major[local_tile_size](), _],
    col_start: Int,
    row_start: Int,
):
    """Writes 8 float32 values of a P row as FP8 with one 8 B store.

    `col_start` is a multiple of 8. SWIZZLE_64B permutes 16 B chunks, so the
    8 B half of a chunk keeps its offset inside the chunk.
    """
    comptime assert local_tile_size == 8, "half-row P is 8 columns"
    comptime swz = make_ldmatrix_swizzle[
        dtype=out_dtype,
        row_size=config.BN_QK,
        log2_vector_width=4,
    ]()
    var off = swz(row_start * config.BN_QK + (col_start & ~15)) + (
        col_start & 15
    )
    var lmv = local_mem.vectorize[4]()
    var w0 = bitcast[.uint32, 1](lmv[0].cast[out_dtype]())
    var w1 = bitcast[.uint32, 1](lmv[1].cast[out_dtype]())
    _ = inlined_assembly[
        "st.shared.v2.b32 [$0], {$1, $2};",
        NoneType,
        constraints="l,r,r",
        has_side_effect=True,
    ](shared_mem.unsafe_offset(off), w0, w1)


def fp8_p_rescale_threshold[fp8_p: Bool](threshold: Float32) -> Float32:
    """Returns the lazy-rescale threshold the softmax uses.

    Parameters:
        fp8_p: Whether P is converted to fp8 before the P@V MMA.

    Args:
        threshold: The configured threshold, in the log2 domain.

    Returns:
        The threshold, capped at -2 when `fp8_p` is True.
    """
    return max(threshold, _FP8_P_MAX_RESCALE_THRESHOLD) if fp8_p else threshold


def fp8_p_bias[fp8_p: Bool](threshold: Float32) -> Float32:
    """Returns the log2 of the scale applied to P before its fp8 conversion.

    Parameters:
        fp8_p: Whether P is converted to fp8 before the P@V MMA.

    Args:
        threshold: The configured lazy-rescale threshold, in the log2 domain.

    Returns:
        `8 + fp8_p_rescale_threshold(threshold)` when `fp8_p` is True, else 0.
    """
    return (
        8 + fp8_p_rescale_threshold[fp8_p](threshold)
    ) if fp8_p else Float32(0)


# ------------------------------------------------------------------------------
# MLA decoding configuration for SM100
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_Config:
    """Holds the tile sizes, swizzle modes, and SMEM/TMEM layout for an SM100 MLA decode kernel.
    """

    var MMA_M: Int
    var MMA_PV_N: Int
    var MMA_QK_N: Int
    var BM: Int
    var BN_PV: Int  # N of PV MMA = output (V) head_dim per CTA. Anchors output writeback path.
    var BN_QK: Int  # N of QK MMA = KV cache tile width (keys per k-tile)
    # K of the QK MMA. The SMEM tile width, not the gmem row width (see
    # `input_q_depth`), so the NoPE tail stays contracted as zero.
    var BK_QK: Int
    var q_depth: Int
    var depth: Int  # this is V depth
    var padded_depth: Int
    var padded_q_depth: Int
    # Gmem row width the TMA reads. May be narrower than `padded_q_depth` for a
    # NoPE model, which zero-fills the tail in SMEM. Drives the TMA descriptors
    # and byte count only, never the SMEM layout or MMA width.
    var input_q_depth: Int
    var rope_depth: Int  # this is Q depth - V depth
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_O: Int = 0
    comptime TMEM_S0: Int = Self.TMEM_O + 256
    comptime TMEM_S1: Int = Self.TMEM_S0 + 32
    # Upper bound on pipeline stages — kernels pick `num_kv_stages` ≤ this
    # at config time, and TMEM is laid out to fit the worst case so
    # downstream offsets (CORR_SCALE etc.) stay stable across configs.
    # 6 stages × 32 cols/stage = 192 cols, sitting in TMEM cols 256..447
    # between TMEM_O (256 cols) and CORR_SCALE (col 448), within the 512
    # total TMEM columns on SM100. Bump only if a kernel needs deeper
    # KV pipelining and the resulting layout still fits.
    comptime MAX_TMEM_S_SLOTS: Int = 6
    comptime DEFAULT_TMEM_CORR_SCALE: Int = (
        Self.TMEM_S0 + Self.MAX_TMEM_S_SLOTS * 32
    )
    # Under `q_in_tmem`, Q's 576 FP8 columns take 144 TMEM columns from here,
    # past three 32-column S slots; the correction columns follow them.
    comptime TMEM_Q_FULL: Int = Self.TMEM_S0 + 3 * 32
    comptime TMEM_Q_FULL_COLS: Int = 576 // 4
    # The h64 Q192 split stages Q's upper blocks in TMEM from here: 192
    # columns past two 32-column S slots.
    comptime TMEM_Q_LO: Int = Self.TMEM_S0 + 2 * 32
    var tmem_used: Int
    # The correction scale and row-sum columns.
    var tmem_corr_scale: Int
    var tmem_corr_li: Int
    # Q's SW64 blocks live in TMEM at `TMEM_Q_FULL` (`MLA_Q_TMEM`).
    var q_in_tmem: Bool
    var num_kv_stages: Int
    # Depth of the MMA-operand KV ring. Equal to `num_kv_stages` everywhere
    # except the unified-gather path, whose two KV buffers need not share a
    # depth: only the one the MMA consumes carries the loop-carried WAR edge
    # that bounds the steady-state rate.
    var num_kv_mma_stages: Int
    var smem_used: Int
    var dtype_size: Int
    # Bytes per element of the MMA operands in SMEM (Q, K, V, P).
    var operand_size: Int
    var num_threads: Int  # bf16: 3 WGs (MMA, softmax, correction); fp8: 4 WGs (+convert)
    var swizzle_mode: TensorMapSwizzle
    var kv_mma_swizzle_mode: TensorMapSwizzle
    var kv_tma_swizzle_mode: TensorMapSwizzle
    var content_swizzle_mode: TensorMapSwizzle  # FP8 content: SWIZZLE_64B
    var rope_swizzle_mode: TensorMapSwizzle  # BF16 rope: SWIZZLE_128B
    comptime MMA_K = 16
    comptime sm100_smem_carveout = (
        B200.shared_memory_per_multiprocessor - SM100_RESERVED_SMEM_BYTES
    )
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()  # 8
    # 1: this CTA's MMAs read only its own SMEM and TMEM (`.ws`, local
    # mbarrier arrives). 2: a 2-CTA cluster whose leader issues
    # `cta_group::2` MMAs over both CTAs' SMEM, with multicast commits and
    # cross-CTA barrier arrives (the FP8 h=128 pair). A var field, not a
    # comptime one, so the comptime TMEM constants below stay
    # cta-group-independent; the derived geometry lives in `s_slot_stride`,
    # `max_s_slots` and `o_round_cols`.
    var cta_group: Int
    var decoding_warp_split_k: Bool
    var out_rows: Int
    var page_size: Int  # KV cache physical page size (e.g., 128)
    var split_page_size: Int  # Page size for split-K work partitioning (must be <= page_size)
    var scale_block_size: Int  # 0 = tensorwise, 32/64/128 = blockwise FP8 scaling
    var scales_per_token: Int  # ceildiv(q_depth, scale_block_size) when blockwise, else 0
    var fp8_scale_bf16: Bool
    var scale_smem_per_stage: Int  # Bytes per tile of staged scales
    var per_token_scale_rope_aware: Bool  # Split content(FP8)/rope(BF16) with per-token FP8 scaling
    var per_token_scales_per_stage: Int  # BN_QK(64) tokens * 1 scale * sizeof(float32)(4) = 256 bytes per stage
    var decode_layout_g: Bool  # Layout G fold path (BM=32, MMA_M=32)
    # Under Layout G with at most 16 softmax rows (`half_row_softmax`), two
    # threads share each row and load 8 key columns each through the 16-lane
    # `tcgen05.ld.16x32bx2` shape. Touches no SMEM, TMEM or stage math.
    var half_rows: Bool
    # Unified gather: TMA gathers K/V straight into the SW128 ring the MMAs
    # read, so there is no linear ring and the re-swizzle warpgroup issues
    # the gathers. Layout G by default; Layout E FP8 behind
    # `MLA_SPARSE_DIRECT_GATHER_E`.
    var direct_gather: Bool
    # Unified sparse kernel with BF16 Q over an FP8 cache: TMA gathers each
    # FP8 row into the upper half of its BF16 K/V stage, and a warpgroup
    # converts it in place (applying blockwise scales), so the gather and
    # MMA rings are one ring.
    var kv_convert: Bool
    # Bytes per slot of the unified gather's linear staging ring when that
    # slot is smaller than a full K/V stage: the NVFP4 form's packed rows
    # and scales (`MLA_NVFP4_PACKED_LINEAR`). 0 sizes the slot as a full
    # stage.
    var gather_stage_bytes: Int
    var BK_PV: Int  # K of PV MMA. Sentinel default = BN_QK.
    # Threshold (in log2 domain) below which the softmax rescale of the
    # running output `O` and `mi` is SKIPPED. When `diff = mi - new_max`
    # is greater than or equal to this threshold (i.e. the running max
    # grew by less than `-threshold` in log2 domain), the rescale is
    # treated as a no-op (scale_for_old_max = 1.0, new_max = mi). This
    # matches FlashMLA's `-6.0` (TokenSpeed's `6.0`) and trades a tiny
    # numerical bias for a saved `exp2` + TMEM publish on every iteration
    # where the running max grows by less than ~6 in log2 domain.
    var skip_correction_threshold: Float32

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        depth: Int,
        q_depth: Int,
        dtype_size: Int,
        kv_type_size: Int,
        swizzle_mode: TensorMapSwizzle,
        kv_mma_swizzle_mode: TensorMapSwizzle,
        page_size: Int,
        decoding_warp_split_k: Bool,
        split_page_size: Int = 128,
        scale_block_size: Int = 0,
        fp8_scale_bf16: Bool = False,
        native_fp8: Bool = False,
        per_token_scale_rope_aware: Bool = False,
        # Selects the Layout G fork of the qkv_fp8 native-FP8 kernel
        # (1x4 datapath, BM=32, MMA_M=32). False is the default for all
        # other backends.
        decode_layout_g: Bool = False,
        # Selects the unified sparse decode kernel
        # (MLA_SM100_Decode_Sparse_QKV_FP8). With a BF16 operand
        # (`native_fp8` False) it always takes the direct gather. Its FP8
        # Layout E form is the re-swizzle-staged gather:
        # KV is gathered contiguously (SWIZZLE_NONE/int64, matching the old
        # converter kernel's efficient gather) into a LINEAR staging buffer,
        # then a re-swizzle warpgroup permutes it FP8->FP8 into the SW64
        # layout the native tcgen05.mma.kind::f8f6f4 operand expects. Adds a
        # second, same-sized KV buffer (linear-in) + a second KV pipeline's
        # worth of barriers per stage, and goes back to 4 warpgroups (the
        # re-swizzle WG). False (default) preserves the plain native-FP8
        # single-buffer SW64-gather layout for every other call site
        # (dense native FP8, Layout G). Behind
        # `MLA_SPARSE_DIRECT_GATHER_E` the FP8 Layout E form instead takes
        # the direct SW128 gather, keeping the separate P ring.
        native_fp8_unified_gather: Bool = False,
        # Selects the half-row Layout G softmax (see `half_rows`).
        half_rows: Bool = False,
        # Threshold (in log2 domain) below which the softmax rescale of
        # `O` and `mi` is skipped. Default `-6.0` matches FlashMLA's
        # value.
        skip_correction_threshold: Float32 = -6.0,
        # 0 means "use `padded_q_depth`" (every rotary-carrying call site).
        input_q_depth: Int = 0,
        # Selects the direct gather (see `direct_gather`).
        direct_gather: Bool = False,
        # Selects the in-place FP8 conversion (see `kv_convert`).
        kv_convert: Bool = False,
        # 2 runs the kernel as one CTA pair: a (2, 1, 1) cluster whose
        # leader issues `cta_group::2` MMAs over both CTAs' SMEM (see
        # `cta_group`). Layout E direct gather only (FP8, or BF16 at a 64-key
        # pair tile).
        cta_group: Int = 1,
        # Overrides the unified gather's per-slot linear staging size (see
        # the `gather_stage_bytes` field).
        gather_stage_bytes: Int = 0,
        # Independent of CTA grouping; 0 retains the existing tile geometry.
        key_tile: Int = 0,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.depth = depth
        self.q_depth = q_depth
        self.rope_depth = q_depth - depth
        self.skip_correction_threshold = skip_correction_threshold
        self.cta_group = cta_group

        self.decode_layout_g = decode_layout_g
        self.half_rows = half_rows
        self.direct_gather = direct_gather
        self.kv_convert = kv_convert
        self.gather_stage_bytes = gather_stage_bytes
        # Layout G uses the 1x4 datapath (BM=MMA_M=32); Layout E uses 2x2
        # (=64), or 128 split across a CTA pair under `cta_group == 2`.
        if decode_layout_g:
            self.BM = 32
            self.MMA_M = 32
        else:
            self.BM = 64
            self.MMA_M = 64 * cta_group
        self.MMA_PV_N = 256
        self.BN_PV = self.MMA_PV_N

        self.dtype_size = dtype_size
        self.operand_size = kv_type_size if native_fp8 else dtype_size
        # A pair tile holds twice the keys: each CTA's K slot (Z) carries
        # half of them, gathered from this CTA's share of the index ring. A
        # 2-byte pair keeps the 64-key tile (32 keys per CTA): a 128-key
        # tile's Z, V and P rings fit only one stage beside a 2-byte Q.
        self.BN_QK = key_tile if key_tile > 0 else 64 * (
            cta_group if self.operand_size == 1 else 1
        )
        self.BK_PV = 64
        # QK MMA writes all N cols that softmax tcgen05_ld[repeat=BN_QK/2] reads.
        self.MMA_QK_N = self.BN_QK
        self.swizzle_mode = swizzle_mode
        self.kv_mma_swizzle_mode = kv_mma_swizzle_mode
        var swizzle_elems = swizzle_mode.bytes() // dtype_size
        self.padded_depth = align_up(depth, swizzle_elems)
        self.padded_q_depth = align_up(q_depth, swizzle_elems)
        self.input_q_depth = (
            input_q_depth if input_q_depth > 0 else self.padded_q_depth
        )

        self.kv_tma_swizzle_mode = (
            TensorMapSwizzle.SWIZZLE_64B if kv_type_size
            == 1 else TensorMapSwizzle.SWIZZLE_128B
        )
        # Split content/rope swizzle modes for per_token_scale_rope_aware:
        # Content is FP8 (1 byte) -> SWIZZLE_64B, Rope is BF16 (2 bytes) -> SWIZZLE_128B.
        # When not per_token_scale_rope_aware, both use the same swizzle as kv_tma_swizzle_mode.
        if per_token_scale_rope_aware:
            self.content_swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
            self.rope_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
        else:
            self.content_swizzle_mode = self.kv_tma_swizzle_mode
            self.rope_swizzle_mode = self.kv_tma_swizzle_mode
        self.per_token_scale_rope_aware = per_token_scale_rope_aware
        # Per-token scales for SnapMLA: 1 float32 scale per KV token (sigma_KV).
        # In MLA's absorbed mode K and V both derive from the same latent c_KV,
        # so they share a single per-token quantization scale.
        # Per stage: BN_QK(64) tokens * 1 * sizeof(float32)(4) = 256 bytes.
        if per_token_scale_rope_aware:
            self.per_token_scales_per_stage = self.BN_QK * 1 * 4
        else:
            self.per_token_scales_per_stage = 0
        # All paths use 3 WGs (384 threads) except: the old FP8 converter path
        # (Q=BF16, KV=FP8, both blockwise and tensorwise), which uses 4 WGs
        # (512 threads) for the extra FP8-to-BF16 conversion warpgroup; and
        # the unified native-FP8 kernel, which uses 4 WGs for the re-swizzle
        # staging warpgroup (no dtype conversion, pure FP8->FP8 layout permute).
        var _old_fp8_converter = (
            kv_type_size == 1
            and not native_fp8
            and not per_token_scale_rope_aware
        )
        if _old_fp8_converter or native_fp8_unified_gather:
            self.num_threads = WARPGROUP_SIZE * 4
        else:
            self.num_threads = WARPGROUP_SIZE * 3

        # 4 bytes for the TMEM base pointer
        var smem_use = 4
        self.q_in_tmem = (
            _MLA_Q_TMEM != 0
            and native_fp8
            # The 64-key linear front end, or the FP8 direct SW128 ring.
            and (native_fp8_unified_gather or direct_gather)
            and not kv_convert
            and not decode_layout_g
            and not per_token_scale_rope_aware
            and cta_group == 1
            and self.operand_size == 1
            and self.BN_QK == 64
            and self.BM == 64
            and self.padded_q_depth == 576
        )
        self.tmem_corr_scale = (
            Self.TMEM_Q_FULL
            + Self.TMEM_Q_FULL_COLS if self.q_in_tmem else Self.DEFAULT_TMEM_CORR_SCALE
        )
        self.tmem_corr_li = self.tmem_corr_scale + 1
        self.tmem_used = self.tmem_corr_li + 1
        self.decoding_warp_split_k = decoding_warp_split_k
        self.page_size = page_size
        self.split_page_size = split_page_size
        self.scale_block_size = scale_block_size
        self.fp8_scale_bf16 = fp8_scale_bf16
        self.scales_per_token = (
            ceildiv(q_depth, scale_block_size) if scale_block_size > 0 else 0
        )
        # BF16 staging preserves arbitrary Float32 scales' converter rounding.
        self.scale_smem_per_stage = (
            self.BN_QK
            * self.scales_per_token
            * (2 if fp8_scale_bf16 else 1) if scale_block_size
            > 0 else 0
        )
        self.BK_QK = self.padded_q_depth
        self.out_rows = min(self.BM, self.num_q_heads)
        # Q SMEM sizing:
        # - BF16 / old FP8 converter: BF16-sized Q (64x576x2 = 73728 bytes)
        # - Native FP8: FP8-sized Q (64x576x1 = 36864 bytes), Q arrives as FP8
        #   from TMA with SWIZZLE_64B. No conversion needed.
        # - Per-token-scale: split Q = FP8 content (BM*512*1=32768) + BF16 rope (BM*64*2=8192) = 40960 bytes
        if per_token_scale_rope_aware:
            smem_use += (
                self.BM * self.padded_depth * 1 + self.BM * self.rope_depth * 2
            )
        elif native_fp8:
            # Under `q_in_tmem`, Q lands in the last MMA stage instead; its
            # copy to TMEM completes on one more barrier.
            if self.q_in_tmem:
                smem_use += Self.mbar_size
            else:
                smem_use += self.BM * self.padded_q_depth * kv_type_size
        else:
            smem_use += self.BM * self.padded_q_depth * dtype_size
        # Three scratch buffers (float32) for softmax running max and li:
        # max uses double-buffering (2 x 128 elements) to avoid a race
        # condition between consecutive loop iterations (write in iteration
        # N+1 can overlap read in iteration N without an extra barrier).
        # li uses a single buffer (only accessed post-loop).
        # Total: 128 threads x 1 element x 4 bytes x 3 buffers = 1536 bytes
        comptime smem_for_max_and_li = 128 * 1 * 4 * 3
        smem_use += smem_for_max_and_li
        # Scale SMEM: for FP8 blockwise, store e8m0 scales (1 byte each).
        # Double-buffered to match the KV pipeline.
        # For tensorwise: no scale SMEM needed.
        # For blockwise: scale_smem_per_stage * 2 stages
        var smem_for_scale: Int
        if scale_block_size > 0 and not kv_convert:
            # Per stage: BN_QK * scales_per_token * 1 byte (e8m0)
            # Double-buffered: * 2 stages (for blockwise FP8 converter path)
            smem_for_scale = self.scale_smem_per_stage * 2
        else:
            smem_for_scale = 0
        smem_use += smem_for_scale
        # KV SMEM per stage sizing:
        # - BF16 / old FP8 converter: BF16-sized stages (BN_QK * padded_q_depth * dtype_size)
        #   The Softmax writes BF16 P into the KV stage at NumVOBlocks * BlockElems offset.
        # - Native FP8: FP8-sized stages (BN_QK * padded_q_depth * kv_type_size)
        #   P lives in a separate SMEM region (not inside KV stages), so KV stages
        #   can be FP8-sized. This gives 3 stages instead of 2.
        # - Per-token-scale: split KV = FP8 content (BN_QK*512*1=32768) + BF16 rope (BN_QK*64*2=8192) = 40960 bytes/stage
        #   P lives in a separate SMEM region (same as native FP8). P stage = BM*BN_QK*1 = 4096 bytes.
        var smem_per_kv: Int
        var smem_for_p: Int
        # Extra depth given to the MMA-operand KV ring alone; see the
        # unified-gather branch below. 0 on every other path.
        var extra_mma_stages: Int = 0
        var direct_ring = native_fp8_unified_gather and (
            direct_gather or kv_convert
        )
        # The direct ring's K/V stages and Q leave room for a P ring at 1
        # byte per element; a 2-byte operand's do not, which is what P in the
        # K/V stage (`p_in_kv`) exists for. The softmax's FP8 P write only
        # knows the separate ring, so `p_in_kv` must stay 2-byte only.
        var p_in_kv = (
            direct_ring
            and not decode_layout_g
            and self.operand_size == 2
            and cta_group == 1
        )
        if per_token_scale_rope_aware:
            # Per-token-scale: KV stage = content FP8 + rope BF16
            smem_per_kv = (
                self.BN_QK * self.padded_depth * 1
                + self.BN_QK * self.rope_depth * 2
            )
            # P reuses the KV rope SMEM region (no separate P allocation).
            # This is safe because rope is consumed by QK MMA (warp 9)
            # BEFORE softmax produces P, and KV stage barriers prevent
            # the Load warp from overwriting until PV MMA (warp 10) finishes.
            # P (4096 bytes FP8) fits inside rope (8192 bytes BF16).
            # P_i maps to rope stage i: same stage indexing, same barriers.
            # Per-token scales: 256 bytes per stage (64 tokens * 1 * 4 bytes).
            var smem_per_stage_total = (
                smem_per_kv
                + self.per_token_scales_per_stage
                + 6 * Self.mbar_size
            )
            var fixed_barrier_reserve = 11 * Self.mbar_size
            var available = (
                Self.sm100_smem_carveout - smem_use - fixed_barrier_reserve
            )
            var out_bar_count = (self.depth // self.BN_QK) * 2
            var extra_bar_count = ((self.depth // self.BN_QK) - 1) * 2
            available -= (out_bar_count + extra_bar_count) * Self.mbar_size
            self.num_kv_stages = min(
                Self.MAX_TMEM_S_SLOTS, available // smem_per_stage_total
            )
            smem_for_p = 0  # P reuses rope SMEM, no separate allocation
        elif direct_ring:
            # Direct gather into SW128 blocks (the row padded
            # to whole blocks). The S, P and index rings take
            # `_DIRECT_RING_N` stages and the K/V ring the rest; with fewer
            # K/V stages than that (2-byte operands), the S, P and index
            # rings shrink to match.
            # `kv_row_elems()`, which `self` cannot call before it is built.
            var kv_row_elems = align_up(
                self.padded_q_depth, 128 // self.operand_size
            )
            if cta_group == 2:
                # CTA pair: one ring depth for the Z (K gather), V, P, S
                # and index rings, the deepest that fits. Per CTA and
                # stage: Z holds this CTA's key half as full rows; the V
                # ring all pair keys' V columns this CTA owns for the
                # MMA's N split; P this CTA's BM rows over all pair keys;
                # the index ring the pair's indices (both CTAs gather from
                # the same list). Barriers per stage: 12 (z_tx, z_empty,
                # z_full, v_tx, v_full, v_free, s[2], p[2], idx[2]).
                var z_stage = self.BN_QK // 2 * kv_row_elems * self.operand_size
                var e_stage = self.BN_QK // 2 * self.depth * self.operand_size
                var p_stage = self.BM * self.BN_QK * self.operand_size
                var idx_stage = (
                    self.BN_QK * 4 * (2 if gather_stage_bytes > 0 else 1)
                )
                var stage_bars = 12 * Self.mbar_size
                var stage_bytes = (
                    z_stage + e_stage + p_stage + idx_stage + stage_bars
                )
                # Fixed: the TMEM pointer, idx alignment pad, mbar_q, the
                # O/C/corr_done pairs and the output pipeline (the generic
                # barrier accounting below counts 15 fixed + 30 output
                # barriers, the same 45 the carve holds at depth 2).
                var fixed_bytes = (
                    4 + 12 + (1 + 4 + 2 + 4 + 2 * 4) * Self.mbar_size
                )
                # S slots are MMA_QK_N / 2 physical TMEM columns each; three
                # fit between TMEM_O and the correction scale column.
                var max_slots = 192 // (self.BN_QK // 2)
                var avail_stages = (
                    Self.sm100_smem_carveout - smem_use - fixed_bytes
                ) // stage_bytes
                self.num_kv_stages = max(2, min(avail_stages, max_slots))
                smem_per_kv = z_stage
                smem_for_p = self.num_kv_stages * p_stage
                smem_use += self.num_kv_stages * (e_stage + idx_stage) + 12
            else:
                # A converted stage also carries its tile's scales.
                smem_per_kv = self.BN_QK * kv_row_elems * self.operand_size + (
                    self.scale_smem_per_stage if kv_convert else 0
                )
                var p_per_stage = (
                    0 if p_in_kv else self.BM * self.BN_QK * self.operand_size
                )
                # A converted stage publishes each SW128 block to the QK MMA on
                # its own barrier.
                var cvt_bar_bytes = (
                    kv_row_elems // (128 // self.operand_size) * Self.mbar_size
                ) if kv_convert else 0
                var fixed_barrier_reserve = 11 * Self.mbar_size
                var available = (
                    Self.sm100_smem_carveout - smem_use - fixed_barrier_reserve
                )
                var out_bar_count = (self.depth // SW64_COLS) * 2
                var extra_bar_count = ((self.depth // SW64_COLS) - 1) * 2
                available -= (out_bar_count + extra_bar_count) * Self.mbar_size
                var n = _DIRECT_RING_N
                var m = Self._direct_mma_depth(
                    available, n, p_per_stage, smem_per_kv, self.BN_QK
                )
                if p_in_kv or kv_convert:
                    # One depth for both rings: the deepest that fits.
                    n = 1
                    while (
                        Self._direct_mma_depth(
                            available,
                            n + 1,
                            p_per_stage + cvt_bar_bytes,
                            smem_per_kv,
                            self.BN_QK,
                        )
                        >= n + 1
                    ):
                        n += 1
                    m = n
                elif m < n:
                    # Shrinking N frees SMEM, so M can only grow.
                    n = m
                    m = Self._direct_mma_depth(
                        available, n, p_per_stage, smem_per_kv, self.BN_QK
                    )
                self.num_kv_stages = n
                extra_mma_stages = m - n
                smem_for_p = self.num_kv_stages * p_per_stage
                smem_use += self.num_kv_stages * cvt_bar_bytes
        elif native_fp8:
            smem_per_kv = self.BN_QK * self.padded_q_depth * kv_type_size
            # Native FP8: P lives in a separate SMEM region.
            # Per-stage cost = KV tile + P tile + 6 barriers (kv:2 + s:2 + p:2).
            # Unified-gather adds a second, same-sized linear-gather KV
            # buffer (contiguous SWIZZLE_NONE stage the re-swizzle WG reads
            # from) plus its own 2-barrier producer/consumer pair per stage.
            var p_per_stage = self.BM * self.BN_QK * kv_type_size
            # A packed NVFP4 linear slot (`gather_stage_bytes`) is smaller
            # than the full FP8 stage the re-swizzle front end otherwise
            # stages; the kernel's `LinStageElems` reads the same value.
            # The freed 14,336 B per slot deepens the gather ring at Layout
            # G and is what fits the Layout E carveout at all.
            var gather_per_stage = (
                gather_stage_bytes if gather_stage_bytes > 0 else smem_per_kv
            ) if native_fp8_unified_gather else 0
            var gather_bar_per_stage = 2 if native_fp8_unified_gather else 0
            var smem_per_stage_total = (
                smem_per_kv
                + p_per_stage
                + gather_per_stage
                + (6 + gather_bar_per_stage) * Self.mbar_size
            )
            # Reserve stage-independent barriers (11) plus output barriers.
            var fixed_barrier_reserve = 11 * Self.mbar_size
            var available = (
                Self.sm100_smem_carveout - smem_use - fixed_barrier_reserve
            )
            var out_bar_count = (self.depth // self.BN_QK) * 2
            var extra_bar_count = ((self.depth // self.BN_QK) - 1) * 2
            available -= (out_bar_count + extra_bar_count) * Self.mbar_size
            self.num_kv_stages = min(
                Self.MAX_TMEM_S_SLOTS, available // smem_per_stage_total
            )
            # Layout G halves Q/P SMEM, so pin to 4 stages. The unified gather
            # doubles each stage, so 4 would overflow; it keeps the derived
            # count.
            if decode_layout_g and not native_fp8_unified_gather:
                self.num_kv_stages = 4
            if self.q_in_tmem:
                # Three S slots end where Q's TMEM columns begin.
                self.num_kv_stages = min(self.num_kv_stages, 3)
            smem_for_p = self.num_kv_stages * (p_per_stage + gather_per_stage)
            # Unified gather stages KV twice: the linear gather destination
            # and the SW64 buffer the MMA reads. Only the second carries the
            # WAR back-edge that closes the steady-state loop, so the loop's
            # rate is bounded by (cycle weight) / (that ring's depth) and the
            # gather ring's depth does not enter it. A symmetric extra stage
            # costs `smem_per_stage_total` and does not fit; one extra SW64
            # stage and its barrier pair does. Spend the leftover there only.
            if native_fp8_unified_gather:
                var extra_kv_stage = smem_per_kv + 2 * Self.mbar_size
                # The sparse dispatch adds idx_bars, idx_smem and the TMEM
                # pointer on top of `smem_used`; reserve them or the deeper
                # ring overflows the carveout at launch rather than here.
                var sparse_reserve = (
                    2 * self.num_kv_stages * Self.mbar_size
                    + 4
                    + self.num_kv_stages * self.BN_QK * 4
                )
                var leftover = (
                    available - self.num_kv_stages * smem_per_stage_total
                )
                if leftover >= extra_kv_stage + sparse_reserve:
                    extra_mma_stages = 1
        else:
            smem_per_kv = self.BN_QK * self.padded_q_depth * dtype_size
            smem_for_p = 0  # P lives inside KV stages for BF16/old FP8
            # now we need to calculate how many slots per K/V we can fit in the remaining memory
            # the carveout reserves 1K for L1 cache so
            # for b200 we have sm100_smem_carveout 233472 - 1024 =  232448 bytes
            self.num_kv_stages = (
                Self.sm100_smem_carveout - smem_use
            ) // smem_per_kv
        self.num_kv_mma_stages = self.num_kv_stages + extra_mma_stages
        smem_use += smem_for_p
        smem_use += self.num_kv_stages * (smem_per_kv)
        # The deeper MMA ring's own tile plus the producer/consumer mbar pair
        # the `fixed_barriers` count below does not cover.
        smem_use += extra_mma_stages * (smem_per_kv + 2 * Self.mbar_size)
        # Per-token scale SMEM: N stages * 256 bytes each (only for per_token_scale_rope_aware)
        smem_use += self.num_kv_stages * self.per_token_scales_per_stage
        # We have the following resources that need smem barriers:

        # bar_write_prod[depth/BN_QK] → 8  producer pipeline - softmax epilogue
        # bar_write_cons[depth/BN_QK] → 8  consumer pipeline - TMA store
        var num_out_barrier = (self.depth // SW64_COLS) * 2
        # total number of barriers is fixed_transaction_barriers + num_out_barrier
        # bar_q → 1           producer pipeline - load consumer - mma
        # bar_kv_ready[2] → 2  consumer pipeline - mma
        # bar_kv_free[2] → 2   producer pipeline - load
        # bar_s_done[2] → 2  producer pipeline - mma
        # bar_s_ready[2] → 2  consumer pipeline - softmax
        # bar_p_done[2] → 2  producer pipeline- softmax
        # bar_p_ready[2] → 2  consumer pipeline - mma
        # bar_correction_done[1] → 1  producer pipeline- softmax
        # bar_correction_ready[1] → 1  consumer pipeline - correction
        # bar_o_done[2] → 2  producer pipeline- MMA PV
        # bar_o_ready[2] → 2  consumer pipeline - Correction
        # corr_done_prod[2] → 2  producer pipeline - correction
        # corr_done_cons[2] → 2  consumer pipeline - softmax

        # Fixed barrier count depends on the path:
        # BF16: 23 barriers (2-stage KV/S/P pipelines)
        # Old FP8 converter: 27 barriers (23 + 4 for convert pipeline)
        # Native FP8 / per_token_scale_rope_aware: 6*N + 11 barriers where N = num_kv_stages
        #   bar_q(1) + kv(2N) + s(2N) + p(2N) + o(4) + c(2) + corr_done(4)
        # Unified-gather adds 2*N more (the linear-gather->re-swizzle pipeline's
        # own producer/consumer barrier pair per stage): 8*N + 11.
        var fixed_barriers: Int
        if cta_group == 2:
            # The pair's barrier carve is exact (see the direct-ring sizing
            # above): 12 barriers per stage plus 19 fixed and 8 output
            # barriers. The generic accounting below adds 30 output
            # barriers on top, so the fixed share is 12*D - 11. The 30
            # output barriers on the accounting side hardwire
            # depth/SW64_COLS == 8 and out_blocks/blocks_per_stage == 4,
            # both pinned by `supported()`'s depth == 512.
            fixed_barriers = 12 * self.num_kv_stages - 11
        elif (
            per_token_scale_rope_aware
            or native_fp8
            or native_fp8_unified_gather
        ):
            fixed_barriers = (
                8 * self.num_kv_stages + 11
            ) if native_fp8_unified_gather else (6 * self.num_kv_stages + 11)
        elif kv_type_size == 1:
            fixed_barriers = (
                27  # Old FP8 converter: 4 extra for convert pipeline
            )
        else:
            fixed_barriers = 23  # BF16: 2-stage pipelines
        smem_use += (fixed_barriers + num_out_barrier) * Self.mbar_size + (
            ((self.depth // SW64_COLS) - 1) * 2 * Self.mbar_size
        )

        # Summary of smem layout:
        # BF16: Q(73728) + KV_stages(2*73728) + max/li(1536) + barriers(23)
        # Old FP8: Q(73728) + KV_stages(2*73728) + max/li(1536) + scale + barriers(27)
        # Native FP8: Q(36864) + KV_stages(N*36864) + P_stages(N*4096) + max/li(1536) + barriers(6N+11)
        #   where N = num_kv_stages (dynamically computed, typically 4)
        # Per-token-scale: Q(40960) + KV_stages(N*40960) + scales(N*256) + max/li(1536) + barriers(6N+11)
        #   Q = FP8 content(32768) + BF16 rope(8192), KV = FP8 content(32768) + BF16 rope(8192)
        #   Per-token scales: N * 256 bytes (64 tokens * 1 scale * 4 bytes float32)
        #   P reuses KV rope SMEM (P_i maps to rope stage i; 4096B FP8 fits in 8192B BF16 rope)
        # max uses double-buffered SMEM (2x128x4=1024B) to avoid race; li uses 1x128x4=512B
        # Plus num_out_barrier = (depth/BN_QK)*2 output barriers,
        # plus ((depth/BN_QK)-1)*2 additional barriers.
        self.smem_used = smem_use

    @staticmethod
    def _direct_mma_depth(
        available: Int, n: Int, p_per_stage: Int, smem_per_kv: Int, BN_QK: Int
    ) -> Int:
        """Returns the K/V stages the direct ring fits beside N S/P stages.

        Each of the N stages costs a P tile and 6 barriers (S, P and the
        unused linear pair, which the kernel's barrier carve keeps); each
        K/V stage a tile and its full/empty pair. The reserve is the
        launcher's index ring, TMEM pointer, idx_smem alignment pad and
        stamp buffer (see `supported`).
        """
        var sparse_reserve = (
            2 * n * Self.mbar_size + 4 + 12 + n * BN_QK * 4 + 16
        )
        return (
            available - n * (p_per_stage + 6 * Self.mbar_size) - sparse_reserve
        ) // (smem_per_kv + 2 * Self.mbar_size)

    def supported(self) -> Bool:
        # BM is 32 (Layout G) or 64 (everyone else).
        # `input_q_depth` narrows only the TMA, so it must cover every V block,
        # fit within the SMEM row, and land on a gather column-group boundary.
        var input_row_ok = (
            self.padded_depth <= self.input_q_depth
            and self.input_q_depth <= self.padded_q_depth
            and self.input_q_depth % SW64_COLS == 0
        )
        var tile_ok = self.BN_QK == 64 * (
            self.cta_group if self.operand_size == 1 else 1
        ) or (
            self.BN_QK == 128
            and self.operand_size == 1
            and not self.decode_layout_g
        )
        var base = (
            self.q_depth == 576
            and input_row_ok
            and tile_ok
            and (self.BM == 32 or self.BM == 64)
            and self.depth == 512
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )
        if not self.decode_layout_g and (
            self.half_rows
            or (
                self.direct_gather
                and self.operand_size == 1
                and self.cta_group == 1
                and _MLA_SPARSE_DIRECT_GATHER_E == 0
                and self.BN_QK != 128
            )
        ):
            return False
        if self.cta_group == 2:
            # The pair: Layout E's direct gather over both CTAs, one ring
            # depth for the Z, E, P, S and index rings, no in-place
            # conversion and no fused split-K combine (that cluster is
            # (1, 1, np) and Layout G-gated). Its SMEM accounting folds the
            # index ring, TMEM pointer and alignment pad into `smem_used`,
            # so there is no extra to add.
            return (
                base
                and not self.decode_layout_g
                and not self.kv_convert
                and not self.p_in_kv()
                and ceildiv(self.num_q_heads, self.BM) == 2
                and self.direct_gather
                and (self.operand_size == 1 or self.operand_size == 2)
                and self.num_threads == WARPGROUP_SIZE * 4
                and self.num_kv_stages == self.num_kv_mma_stages
                and self.num_kv_stages <= self.max_s_slots()
            )
        if self.direct_gather or self.kv_convert:
            # Include the launcher's index ring, TMEM pointer, alignment pad
            # and stamp buffer in the SMEM check.
            var sparse_extra = (
                2 * self.num_kv_stages * Self.mbar_size
                + 4
                + 12
                + self.num_kv_stages * self.BN_QK * 4
                + 16
            )
            # Its S ring is sized by `_DIRECT_RING_N`, not clamped like the
            # other rings, so bound it here: an S slot past the last one would
            # overlap the correction scale and li columns.
            base = (
                base
                and self.num_threads == WARPGROUP_SIZE * 4
                and self.num_kv_stages <= self.max_s_slots()
                and self.smem_used + sparse_extra <= Self.sm100_smem_carveout
            )
        if not self.decode_layout_g:
            return base
        # Layout G: BM=MMA_M=32 (1x4 datapath). The dense kernel pins 4 KV
        # stages; the unified-gather sparse kernel derives its ring depths,
        # which an FP8 ring needs at least 3 of to keep the gather ahead.
        # The fold_q / num_heads * q_len_fold gates live in the dispatcher.
        return (
            base
            and self.BM == 32
            and self.MMA_M == 32
            and self.num_kv_mma_stages >= (3 if self.operand_size == 1 else 2)
        )

    def kv_sw128(self) -> Bool:
        """Returns whether K/V are stored as SW128 blocks: the direct
        gather, or the in-place conversion."""
        return self.direct_gather or self.kv_convert

    def q_swizzle(self) -> TensorMapSwizzle:
        """Returns the swizzle of Q's 64-column SMEM blocks: SW64 for an FP8
        operand, SW128 for a 2-byte one."""
        return (
            TensorMapSwizzle.SWIZZLE_64B if self.operand_size
            == 1 else TensorMapSwizzle.SWIZZLE_128B
        )

    def kv_block_cols(self) -> Int:
        """Returns the columns per K/V SMEM column block."""
        return 128 // self.operand_size if self.kv_sw128() else SW64_COLS

    def p_in_kv(self) -> Bool:
        """Returns whether P lives in the rope block of its K/V stage.

        Only unpaired 2-byte operands under Layout E (the direct gather or
        the in-place conversion), whose Q and two K/V stages leave no room
        for a P ring. A 1-byte operand fits one beside its K/V stages, and
        so does the 2-byte pair's 64-key tile.
        """
        return (
            self.kv_sw128()
            and not self.decode_layout_g
            and self.operand_size == 2
            and self.cta_group == 1
        )

    def kv_row_elems(self) -> Int:
        """Returns the SMEM width of one K/V row, padded to whole blocks."""
        return align_up(self.padded_q_depth, self.kv_block_cols())

    def half_row_softmax(self, rows: Int) -> Bool:
        """Returns whether the softmax runs two threads per row.

        Only Layout G with at most 16 live rows (`num_q_heads` times the
        folded query tokens).

        Args:
            rows: The softmax rows the CTA holds.

        Returns:
            True when `half_rows` applies to a CTA with `rows` rows.
        """
        return self.half_rows and self.decode_layout_g and rows <= 16

    def o_round_cols(self) -> Int:
        """Returns the TMEM column stride between the two O accumulators.

        One PV MMA of N = MMA_PV_N packs its accumulator into MMA_PV_N / 2
        physical columns per CTA: 128 under Layout E, 64 under Layout G
        (FA4's `sk_o_stride`). A `cta_group::2` M = 128 accumulator holds
        each CTA's half of the rows in the same N / 2 columns, so the
        stride is unchanged by the pair.
        """
        if self.cta_group == 2:
            return self.MMA_PV_N // 2
        return self.MMA_PV_N * self.MMA_M // 128

    def s_slot_stride(self) -> Int:
        """Returns the TMEM columns per S slot.

        An S accumulator of N = MMA_QK_N keys occupies MMA_QK_N / 2
        physical columns per CTA: the `.ws` M = 64 form and the
        `cta_group::2` M = 128 form both split each row across the two
        lane halves. 32 per slot unpaired, 64 under the pair.
        """
        return self.MMA_QK_N // 2

    def max_s_slots(self) -> Int:
        """Returns the S slots that fit between TMEM_O and the correction
        scale column (or Q's columns under `q_in_tmem`)."""
        var end = Self.TMEM_Q_FULL if self.q_in_tmem else self.tmem_corr_scale
        return (end - Self.TMEM_S0) // self.s_slot_stride()


@inline(.always)
def rows_owned[config: MLA_SM100_Decode_Config](block_x: Int) -> Int:
    """Returns the number of output rows the CTA at grid x-index block_x
    owns.

    Every head group is out_rows tall except the last, which holds the
    remainder of num_q_heads.

    Parameters:
        config: Decode config supplying the head-group geometry.

    Args:
        block_x: The CTA's grid x-index.

    Returns:
        The number of output rows the CTA owns, at most out_rows.
    """
    return min(config.out_rows, config.num_q_heads - block_x * config.BM)


@inline(.always)
def sparse_kv_row[
    physical_indices: Bool
](kv_lut: Some[MHAOperand], batch_idx: UInt32, index: Int32) -> Int32:
    """Returns the KV TMA row a sparse index names, or `-1` for an unused slot.

    Parameters:
        physical_indices: Whether `index` is an already-paged
            `physical_block * page_size + offset` row (the indexer's physical
            rows, or latent sparse attention's flat rows) rather than a token
            position within `batch_idx`'s entry.

    Args:
        kv_lut: Cache operand the index addresses.
        batch_idx: Batch entry the index belongs to; unread when
            `physical_indices`.
        index: The sparse index, `-1` for an unused slot.

    Returns:
        The TMA row, or `-1`.
    """
    comptime if physical_indices:
        if index == -1:
            return -1
        return kv_lut.get_tma_row(index)
    else:
        return kv_lut.sparse_tma_row(batch_idx, index)


@inline(.always)
def decode_head_tile[config: MLA_SM100_Decode_Config]() -> Int:
    """Returns this CTA's head-tile index.

    The decode grid flattens (query token, head tile) into x, head tile
    fastest, so the head tiles of one token are adjacent CTAs and share that
    token's gathered K/V in L2. Keeping the token index off `gridDim.y`
    lifts CUDA's 65535 cap on it.

    Parameters:
        config: Decode config supplying `num_q_heads` and `BM`.

    Returns:
        The head-tile index in `[0, ceildiv(num_q_heads, BM))`.
    """
    return block_idx.x % ceildiv(config.num_q_heads, config.BM)


@inline(.always)
def decode_seq_idx[config: MLA_SM100_Decode_Config]() -> Int:
    """Returns this CTA's query-token index within its batch.

    See `decode_head_tile` for the grid layout.

    Parameters:
        config: Decode config supplying `num_q_heads` and `BM`.

    Returns:
        The query-token index; always 0 under `fold_q`.
    """
    return block_idx.x // ceildiv(config.num_q_heads, config.BM)


# ------------------------------------------------------------------------------
# Offset position struct
# ------------------------------------------------------------------------------
struct OffsetPosition[
    config: MLA_SM100_Decode_Config,
    KVLUTType: MHAOperand,
    ragged: Bool,
    is_cache_length_accurate: Bool,
    ValidLengthType: OptionalPointer,
    decoding_warp_split_k: Bool = False,
    sparse: Bool = False,
    has_extra_kv: Bool = False,
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
](TrivialRegisterPassable):
    """Computes and stores per-CTA row offsets and KV key ranges for the decode kernel.

    Parameters:
        config: Decode config supplying tile sizes and head counts used to
            compute Q and output row offsets.
        KVLUTType: `MHAOperand` providing the KV cache tensor and its
            `cache_length` accessor.
        ragged: When `True`, the valid-lengths tensor is interpreted as
            input row offsets enabling ragged batching.
        is_cache_length_accurate: When `False`, the kernel adds the local
            sequence length to the cache length to compute the total key
            count.
        ValidLengthType: `OptionalPointer` type wrapping the per-batch
            valid-sequence-length tensor.
        decoding_warp_split_k: When `True`, the CTA processes a split-K
            partition of the KV cache (defaults to `False`).
        sparse: When `True`, the kernel iterates over a sparse subset of
            tokens selected by `d_indices` instead of the full KV cache
            (defaults to `False`).
        has_extra_kv: When `True`, sparse attention additionally attends
            to a separate extra-KV cache (defaults to `False`).
        TopkLengthsPtrType: `OptionalPointer` type of sparse-slot lengths
            (defaults to `NullPointer`). The constructor reads per batch
            unless `topk_lengths_per_token` is set; tile skipping reads per
            packed query token after rejecting padded CTAs.
    """

    comptime has_variable_topk = not Self.TopkLengthsPtrType.is_null
    # Narrowing the BF16-KV sparse prologue caused spills in its tile loops.
    # Layout G and sparse FP8-KV retain the measured 32-bit divides.
    comptime narrow_split = (
        Self.sparse
        and (
            Self.config.decode_layout_g
            or Self.KVLUTType.dtype == DType.float8_e4m3fn
        )
    )

    var seq_len: Int
    var max_seq_len: Int  # q_max_seq_len (padded seq dimension for all batches)
    var num_keys: Int  # Total keys for this batch (full KV cache length)
    var q_row_offset: Int  # Row offset for Q tensor (no split dimension)
    var out_row_offset: Int  # Row offset for output tensor (includes split dimension)
    var split_idx: Int  # Which split partition this CTA handles
    var batch_idx: Int  # Which batch this CTA handles
    var kv_start_row: Int  # Starting KV row for this split
    var num_keys_this_split: Int  # Number of keys this split processes
    var q_token_idx: Int  # Global Q token index for per-token Q scale lookup
    var seq_idx: Int  # This CTA's query token within its batch
    # Logical (pre-sparse-override) total key count for this batch. `num_keys`
    # is overridden below (sparse=True) to the slot count (topk + extra_topk),
    # which only equals the logical count when topk >= actual_tokens; sparse
    # causal masking uses `cache_len_logical()` off this field instead.
    var actual_num_keys: Int
    # Valid main-cache sparse slots (clamped topk). With `has_extra_kv` the
    # main segment is padded to a whole `BN_QK` tile so every tile belongs to
    # one cache; the padding slots `[sparse_topk, align_up(sparse_topk,
    # BN_QK))` are masked and never gathered.
    var sparse_topk: Int

    @inline(.always)
    def __init__(
        out self,
        k: Self.KVLUTType,
        valid_length: ImmPointer[
            Scalar[Self.ValidLengthType.dtype], origin=ImmutAnyOrigin
        ],
        max_seq_len: Int,
        num_partitions: Int,
        batch_size: Int,
        # Sparse attention parameters — only used when sparse=True (comptime).
        sparse_indices_stride: Int = 0,
        sparse_topk_lengths: Self.TopkLengthsPtrType = null_pointer[
            Self.TopkLengthsPtrType
        ](),
        sparse_extra_indices_stride: Int = 0,
        sparse_extra_topk_lengths: Self.TopkLengthsPtrType = unread_pointer[
            Self.TopkLengthsPtrType
        ](),
        topk_lengths_per_token: Bool = False,
        role_block_x: Int = -1,
        role_block_z: Int = -1,
    ):
        self.seq_len = 0
        self.max_seq_len = max_seq_len
        self.num_keys = 0
        self.q_row_offset = 0
        self.out_row_offset = 0
        self.split_idx = 0
        self.batch_idx = 0
        self.kv_start_row = 0
        self.num_keys_this_split = 0
        self.q_token_idx = 0
        self.seq_idx = decode_seq_idx[Self.config]()
        self.actual_num_keys = 0
        self.sparse_topk = 0
        var head_tile = decode_head_tile[Self.config]()
        comptime if (
            Self.sparse
            and Self.decoding_warp_split_k
            and Self.config.cta_group == 1
            and Self.KVLUTType.dtype == DType.uint8
            and Self.config.num_q_heads == 128
        ):
            if role_block_x >= 0:
                self.seq_idx = role_block_x // ceildiv(
                    Self.config.num_q_heads, Self.config.BM
                )
                head_tile = role_block_x % ceildiv(
                    Self.config.num_q_heads, Self.config.BM
                )

        comptime if (
            Self.sparse
            and Self.decoding_warp_split_k
            and Self.config.cta_group == 1
            and Self.KVLUTType.dtype == DType.uint8
            and Self.config.num_q_heads == 128
        ):
            # Decode block_idx.z into split_idx and batch_idx
            # Grid layout: block_z = batch_size * num_partitions
            # block_idx.z = batch_idx * num_partitions + split_idx
            comptime if Self.decoding_warp_split_k:
                comptime if Self.narrow_split:
                    var bz = UInt32(
                        block_idx.z if role_block_z < 0 else role_block_z
                    )
                    var np32 = UInt32(num_partitions)
                    var b32 = bz // np32
                    self.batch_idx = Int(b32)
                    self.split_idx = Int(bz - b32 * np32)
                    comptime if _MLA_HALF_ROWS_BREAK == 6:
                        self.split_idx = 0
                else:
                    self.batch_idx, self.split_idx = divmod(
                        block_idx.z if role_block_z < 0 else role_block_z,
                        num_partitions,
                    )
            else:
                self.batch_idx = (
                    block_idx.z if role_block_z < 0 else role_block_z
                )
                self.split_idx = 0
        else:
            # Decode block_idx.z into split_idx and batch_idx
            # Grid layout: block_z = batch_size * num_partitions
            # block_idx.z = batch_idx * num_partitions + split_idx
            comptime if Self.decoding_warp_split_k:
                comptime if Self.narrow_split:
                    var bz = UInt32(block_idx.z)
                    var np32 = UInt32(num_partitions)
                    var b32 = bz // np32
                    self.batch_idx = Int(b32)
                    self.split_idx = Int(bz - b32 * np32)
                    comptime if _MLA_HALF_ROWS_BREAK == 6:
                        self.split_idx = 0
                else:
                    self.batch_idx, self.split_idx = divmod(
                        block_idx.z, num_partitions
                    )
            else:
                self.batch_idx = block_idx.z
                self.split_idx = 0

        comptime if Self.ragged:
            # treat valid_lengths as input_row_offsets
            # Use batch_idx (not block_idx.z) to index into valid_length
            var start_of_seq = Int(valid_length[self.batch_idx])
            var end_of_seq = Int(valid_length[self.batch_idx + 1])
            self.seq_len = end_of_seq - start_of_seq

            # Global Q token index for per-token scale lookup.
            # In ragged mode, Q tokens are packed: token = start_of_seq + seq_idx
            self.q_token_idx = start_of_seq + self.seq_idx

            # Q row offset: no split dimension
            # Q shape: (total_tokens * num_heads, depth)
            self.q_row_offset = (
                start_of_seq * Self.config.num_q_heads
                + head_tile * Self.config.BM
                + self.seq_idx * Self.config.num_q_heads
            )

            # Output row offset: includes split dimension for split-K
            comptime if Self.decoding_warp_split_k:
                # For ragged with split-K, o_accum_split uses PADDED layout:
                # Shape: (num_partitions, batch_size, max_seq_len, num_heads, depth)
                # This must match the combine kernel's read pattern which uses
                # batch_idx * max_seq_len * num_heads as the stride per batch.
                var rows_per_split = (
                    batch_size * max_seq_len * Self.config.num_q_heads
                )
                self.out_row_offset = (
                    self.split_idx * rows_per_split
                    + self.batch_idx * max_seq_len * Self.config.num_q_heads
                    + self.seq_idx * Self.config.num_q_heads
                    + head_tile * Self.config.BM
                )
            else:
                self.out_row_offset = self.q_row_offset

        # This is when the sequence length is Fixed
        else:
            self.seq_len = max_seq_len

            # Global Q token index for per-token scale lookup.
            # In fixed mode: token = batch_idx * seq_len + seq_idx
            self.q_token_idx = self.batch_idx * self.seq_len + self.seq_idx

            # Q row offset: (batch * seq_len * num_heads, depth)
            # Row = batch_idx * (seq_len * num_heads) + seq_idx * num_heads + head_block * BM
            self.q_row_offset = (
                Self.config.num_q_heads * self.seq_len * self.batch_idx
                + head_tile * Self.config.BM
                + self.seq_idx * Self.config.num_q_heads
            )

            # Output row offset for split-K:
            # Out shape: (split_k * batch * seq_len * num_heads, depth)
            # Row = split_idx * (batch * seq_len * num_heads) + q_row_offset
            comptime if Self.decoding_warp_split_k:
                var rows_per_split = (
                    batch_size * self.seq_len * Self.config.num_q_heads
                )
                self.out_row_offset = (
                    self.split_idx * rows_per_split + self.q_row_offset
                )
            else:
                self.out_row_offset = self.q_row_offset

        # Get num_keys from KV cache for this batch
        # Use batch_idx (not block_idx.z) to get the correct cache length
        self.num_keys = k.cache_length(self.batch_idx)

        comptime if not Self.is_cache_length_accurate:
            self.num_keys += self.seq_len

        # Logical total key count, captured before the sparse override
        # below can replace num_keys with the (smaller) sparse slot count.
        self.actual_num_keys = self.num_keys

        # Compute KV range for this split
        # Each split handles a portion of the KV cache: [kv_start_row, kv_start_row + num_keys_this_split)
        comptime if Self.decoding_warp_split_k:
            # Split-page-aligned strategy: only last CTA handles ragged remainder.
            # All other CTAs process complete split_page_size-element chunks.
            comptime page_size = Self.config.split_page_size
            var total_pages = ceildiv(self.num_keys, page_size)
            var pages_per_split = ceildiv(total_pages, num_partitions)

            # Split boundaries are page-aligned
            var start_page = self.split_idx * pages_per_split
            var end_page = min(
                (self.split_idx + 1) * pages_per_split, total_pages
            )

            self.kv_start_row = start_page * page_size
            var kv_end_row = min(end_page * page_size, self.num_keys)
            self.num_keys_this_split = max(kv_end_row - self.kv_start_row, 0)
        else:
            # No split: process all keys starting from row 0
            self.kv_start_row = 0
            self.num_keys_this_split = self.num_keys

        # -------------------------------------------------------------------
        # Sparse attention: override num_keys with topk (clamped to
        # actual_tokens). When sparse=True (comptime) the kernel
        # iterates over a sparse subset of tokens selected by d_indices
        # instead of the full KV cache.
        # -------------------------------------------------------------------
        comptime if Self.sparse:
            # self.num_keys already holds the correct total token count
            # (cache_length + seq_len when _is_cache_length_accurate=False,
            # or just cache_length otherwise). Use it as the upper bound.
            var actual_tokens = self.num_keys

            var topk: Int
            comptime if Self.has_variable_topk:
                if topk_lengths_per_token:
                    topk = sparse_indices_stride
                else:
                    topk = Int(sparse_topk_lengths.value()[Int(self.batch_idx)])
            else:
                topk = sparse_indices_stride

            # Per-query physical rows may repeat keys; their stored lengths
            # bound the row, while the causal specialization clamps visibility.
            if not topk_lengths_per_token:
                topk = min(topk, actual_tokens)
            self.sparse_topk = topk

            # Extra KV: always-attend tokens from a separate cache.
            var extra_topk: Int = 0
            var main_slots = topk
            comptime if Self.has_extra_kv:
                comptime if Self.has_variable_topk:
                    extra_topk = Int(
                        sparse_extra_topk_lengths.value()[Int(self.batch_idx)]
                    )
                else:
                    extra_topk = sparse_extra_indices_stride
                main_slots = align_up(topk, Self.config.BN_QK)

            # Override num_keys with the sparse slot count.
            self.num_keys = main_slots + extra_topk

            self._split_sparse_slots(main_slots + extra_topk, num_partitions)

    @inline(.always)
    def _split_sparse_slots(mut self, total_slots: Int, num_partitions: Int):
        comptime if Self.decoding_warp_split_k:
            comptime page_size = Self.config.split_page_size
            var total_pages_s = ceildiv(total_slots, page_size)
            var pages_per_split_s: Int
            comptime if Self.narrow_split:
                pages_per_split_s = Int(
                    ceildiv(UInt32(total_pages_s), UInt32(num_partitions))
                )
                comptime if _MLA_HALF_ROWS_BREAK == 7:
                    pages_per_split_s -= 1
            else:
                pages_per_split_s = ceildiv(total_pages_s, num_partitions)
            var start_page_s = self.split_idx * pages_per_split_s
            var end_page_s = min(
                (self.split_idx + 1) * pages_per_split_s, total_pages_s
            )
            self.kv_start_row = start_page_s * page_size
            var kv_end_row_s = min(end_page_s * page_size, total_slots)
            self.num_keys_this_split = max(kv_end_row_s - self.kv_start_row, 0)
        else:
            self.kv_start_row = 0
            self.num_keys_this_split = total_slots

    @inline(.always)
    def skip_masked_tiles[
        causal: Bool = True
    ](
        mut self,
        num_partitions: Int,
        sparse_indices_stride: Int,
        sparse_topk_lengths: Self.TopkLengthsPtrType,
    ):
        """Shrinks this token's slot range to the keys it can see.

        A sparse row packs its valid entries first and never holds more than
        the token's causal key count, so slots past that count are `-1`
        padding the mask would discard. Every token otherwise walks the
        sequence-wide slot count, which on a prompt shorter than `top_k`
        loads and scores up to twice the tiles prefill needs. Split-K
        partitions are recomputed over the shrunk range; a partition left
        empty takes the existing zero-key early exit.

        Lengths here cover packed query tokens. Legacy decode reads one
        length per batch; the constructor must suppress that read in this mode.

        Args:
            num_partitions: Number of split-K partitions in the grid.
            sparse_indices_stride: Row stride of the sparse indices, the
                slot count when no lengths are given.
            sparse_topk_lengths: Per-token valid top-k counts.
        """
        comptime assert (
            Self.sparse and not Self.has_extra_kv
        ), "skip_masked_tiles needs a sparse row with no extra-KV segment"
        # A padded CTA can lie beyond the packed token/length allocation.
        comptime if Self.ragged:
            if self.seq_idx >= self.seq_len:
                return
        var visible = self.causal_key_count() if causal else self.sparse_topk
        comptime if Self.has_variable_topk:
            # Clamp a corrupt stored length so it cannot read a neighboring
            # token's index row or skip the zero-selection store.
            visible = max(
                min(
                    visible,
                    Int(sparse_topk_lengths.value()[self.q_token_idx]),
                    self.sparse_topk,
                ),
                0,
            )
        else:
            visible = min(visible, sparse_indices_stride)
            if visible >= self.sparse_topk:
                return
        self.sparse_topk = visible
        self.num_keys = visible
        self._split_sparse_slots(visible, num_partitions)

    @inline(.always)
    def cache_len(self) -> Int:
        # num_keys is total keys, seq_len is chunk length
        return max(self.num_keys - self.seq_len, 0)

    @inline(.always)
    def cache_len_logical(self) -> Int:
        # Logical cached-prefix length, unaffected by the sparse num_keys
        # override. Equal to cache_len() in dense mode, and whenever sparse
        # topk >= actual_tokens (topk clamps to the full causal set). Must
        # be used (not cache_len()) for sparse causal masking by LOGICAL
        # key position once topk < actual context length.
        return max(self.actual_num_keys - self.seq_len, 0)

    @inline(.always)
    def causal_key_count(self) -> Int:
        # Logical keys this CTA's query token sees under a causal mask, which
        # also bounds how many valid entries its packed sparse row holds.
        return self.cache_len_logical() + self.seq_idx + 1

    @inline(.always)
    def assert_row_packed[
        sparse_row: SparseRowLayout
    ](
        self,
        d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
        indices_stride: Int,
        selection_lag: Int,
    ):
        """Asserts from one grid thread that this token's row is packed.

        See `assert_sparse_row_packed`.

        Parameters:
            sparse_row: The producer's row layout.

        Args:
            d_indices: The `[total_q_tokens, indices_stride]` sparse indices.
            indices_stride: Row stride of `d_indices`.
            selection_lag: Positions between the token the row was computed
                for and this one.
        """
        if (
            thread_idx.x == 0
            and decode_head_tile[Self.config]() == 0
            and self.split_idx == 0
        ):
            assert_sparse_row_packed[sparse_row](
                d_indices,
                indices_stride,
                self.q_token_idx,
                self.causal_key_count(),
                selection_lag,
                self.sparse_topk,
            )

    @inline(.always)
    def start_pos(self, cache_start_pos: UInt32) -> UInt32:
        # start_pos is the base absolute Q index for this chunk (plus any external base)
        return UInt32(self.cache_len()) + cache_start_pos

    @inline(.always)
    def q_row_offset_at(self, q_local: Int) -> Int:
        # Per-q_token Q-row offset for TMA load coord (Option A q_len fold,
        # stored q_row_offset bakes in seq_idx as the q-local term in
        # both ragged and fixed modes, so we swap seq_idx for q_local.
        return (
            self.q_row_offset
            + (q_local - self.seq_idx) * Self.config.num_q_heads
        )

    @inline(.always)
    def out_row_offset_at(self, q_local: Int) -> Int:
        # Per-q_token output-row offset for TMA store coord (q_len
        # fold. the stored out_row_offset has
        # exactly one (seq_idx * num_q_heads) term in every
        # ragged/fixed x split/no-split mode.
        return (
            self.out_row_offset
            + (q_local - self.seq_idx) * Self.config.num_q_heads
        )

    @inline(.always)
    def q_token_idx_at(self, q_local: Int) -> Int:
        # Global Q-token index for the q_local-th q_token in this CTA's
        # batch. Stored q_token_idx bakes in seq_idx; swaps
        # seq_idx for q_local in both ragged and fixed modes.
        return self.q_token_idx + (q_local - self.seq_idx)


# ------------------------------------------------------------------------------
# MLA decoding Load fp8 to bf16 ProducerKVPipeline
# ------------------------------------------------------------------------------
struct KVLoad2CvtProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Producer side of the FP8-to-BF16 load-and-convert KV pipeline.

    Parameters:
        dtype: Element type of the FP8 KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
    """

    # For blockwise FP8 scaling, warp 8's 32 threads also arrive on the
    # producer mbar after writing scale data to SMEM (release semantics).
    # This eliminates separate named barriers for scale synchronization.
    comptime _load2cvt_num_prod = 1 + (
        32 if Self.config.scale_block_size > 0 else 0
    )
    comptime KVPipeType = KVPipelineGeneric[
        Self.config.num_kv_stages,
        1,
        Self._load2cvt_num_prod,
        WARPGROUP_SIZE + 2,
    ]

    # BF16-stage element count (64*576 = 36864)
    comptime bf16_stage_elems = Self.config.BN_QK * Self.config.q_depth

    # FP8 overlay stride in FP8 elements:
    # lower-half(fp8) + upper-half(fp8) = 2 * bf16_stage_elems
    comptime fp8_stage_stride_elems = 2 * Self.bf16_stage_elems

    var pipe: Self.KVPipeType

    # IMPORTANT: this pointer must already point to the UPPER HALF (1:fp8) of stage0
    @__allow_legacy_any_origin_fields
    var smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]]

    @inline(.always)
    def __init__(
        out self,
        pipe: Self.KVPipeType,
        smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem_upper_fp8 = smem_upper_fp8
        self.pipe.state._phase = 1

    @inline(.always)
    def init(self):
        self.pipe.init()

    @inline(.always)
    def stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        return self.smem_upper_fp8.unsafe_offset(
            stage_idx * UInt32(Self.fp8_stage_stride_elems)
        )

    @inline(.always)
    def producer_mbar[*, qk_stage: Int = 0](self) -> MBarType:
        return self.pipe.producer_mbar[qk_stage]()

    @inline(.nodebug)
    def acquire[*, qk_stage: Int = 0](self):
        self.pipe.producer_acquire[qk_stage]()

    @inline(.nodebug)
    def commit_step(mut self):
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding Convert fp8 to bf16 ConsumerKVPipeline
# ------------------------------------------------------------------------------


struct KVLoad2CvtConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Consumer side of the FP8-to-BF16 load-and-convert KV pipeline.

    Parameters:
        dtype: Element type of the FP8 KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
    """

    # Must match KVLoad2CvtProducer's num_producer for type compatibility.
    comptime _load2cvt_num_prod = 1 + (
        32 if Self.config.scale_block_size > 0 else 0
    )
    comptime PipeT = KVPipelineGeneric[
        Self.config.num_kv_stages,
        1,
        Self._load2cvt_num_prod,
        WARPGROUP_SIZE + 2,
    ]

    comptime bf16_stage_elems = Self.config.BN_QK * Self.config.q_depth
    comptime fp8_stage_stride_elems = 2 * Self.bf16_stage_elems

    var pipe: Self.PipeT

    # points to UPPER HALF (1:fp8) of stage0
    @__allow_legacy_any_origin_fields
    var smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]]

    @inline(.always)
    def __init__(
        out self,
        pipe: Self.PipeT,
        smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem_upper_fp8 = smem_upper_fp8

    @inline(.always)
    def stage_base_ptr(self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var idx: UInt32 = self.pipe.state.index()
        return self.smem_upper_fp8.unsafe_offset(
            idx * UInt32(Self.fp8_stage_stride_elems)
        )

    @inline(.nodebug)
    def wait(self):
        self.pipe.consumer_wait[0]()

    @inline(.nodebug)
    def release_all(mut self):
        _ = self.pipe.consumer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding produce bf16 for MMAConsumerKVPipeline
# ------------------------------------------------------------------------------


struct KVCvt2MmaProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Produces converted BF16 KV tiles for the MMA consumer pipeline.

    Parameters:
        dtype: Element type of the converted KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
    """

    comptime PipeT = KVPipelineGeneric[
        Self.config.num_kv_stages, 1, WARPGROUP_SIZE, 2
    ]
    comptime kv_stage_elems = Self.config.BN_QK * Self.config.q_depth

    var pipe: Self.PipeT

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @inline(.always)
    def __init__(
        out self, pipe: Self.PipeT, smem: SharedMemPointer[Scalar[Self.dtype]]
    ):
        self.pipe = pipe
        self.smem = smem
        self.pipe.state._phase = 1

    @inline(.nodebug)
    def acquire(self):
        # waits until MMA (2 consumers) released this stage
        self.pipe.producer_acquire[0]()

    @inline(.always)
    def stage_index(self) -> UInt32:
        return self.pipe.state.index()

    @inline(.always)
    def stage_base_ptr(self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var idx = self.pipe.state.index()
        return self.smem.unsafe_offset(idx * UInt32(Self.kv_stage_elems))

    @inline(.nodebug)
    def commit_all(mut self):
        # 128 threads arrive on producer mbar
        _ = self.pipe.producer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding consume bf16  for MMAConsumerKVPipeline
# ------------------------------------------------------------------------------
struct KVCvt2MmaConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Consumes BF16 KV tiles from the convert producer for the MMA pipeline.

    Parameters:
        dtype: Element type of the converted KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
    """

    comptime KVPipeType = KVPipelineGeneric[
        Self.config.num_kv_stages, 1, WARPGROUP_SIZE, 2
    ]
    comptime kv_stage_elems = Self.config.BN_QK * Self.config.q_depth

    var pipe: Self.KVPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @inline(.always)
    def __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

    @inline(.always)
    def stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem.unsafe_offset(stage_offset)

    @inline(.always)
    def stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @inline(.nodebug)
    def wait[*, qk_stage: Int = 0](self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait[qk_stage]()

    @inline(.nodebug)
    def release[*, qk_stage: Int = 0](mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release[qk_stage](e)


# ------------------------------------------------------------------------------
# MLA decoding ProducerKVPipeline
# ------------------------------------------------------------------------------


struct DecodeKVProducer[
    dtype: DType,
    config: MLA_SM100_Decode_Config,
    num_producer: Int = 1,
    num_consumer: Int = 2,
    num_stages: Int = config.num_kv_stages,
    stage_elems: Int = 0,
](TrivialRegisterPassable):
    """Producer side of the decode KV pipeline that loads KV tiles via TMA.

    Parameters:
        dtype: Element type of the KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
        num_producer: Number of producer threads arriving on each producer
            mbarrier (defaults to 1).
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier (defaults to 2, matching the standard mmaQK+mmaPV
            dual-consumer KV pipeline). Set to e.g. `WARPGROUP_SIZE` when
            the consumer side is a full warpgroup independently arriving
            (not TMA/MMA-elected), as with a manual SMEM-to-SMEM
            re-staging producer/consumer pair.
        num_stages: Ring depth (defaults to `config.num_kv_stages`). Set
            explicitly when a kernel stages KV through two rings of
            different depths.
        stage_elems: Elements per ring slot (0, the default, means one
            BN_QK x q_depth tile). Set for a ring of partial tiles.
    """

    comptime KVPipeType = KVPipelineGeneric[
        Self.num_stages, 1, Self.num_producer, Self.num_consumer
    ]

    # One KV stage = a BN_QK x 576 logical K tile (loaded as
    # NumQKBlocks x BN_QK x BK_QKT).
    comptime kv_stage_elems = (
        Self.stage_elems if Self.stage_elems
        > 0 else Self.config.BN_QK * Self.config.q_depth
    )
    comptime kv_stage_bytes = Self.kv_stage_elems * size_of[Self.dtype]()

    var pipe: Self.KVPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @inline(.always)
    def __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

        # IMPORTANT: producer starts at phase 1, like FA4
        self.pipe.state._phase = 1

    @inline(.always)
    def init(self):
        self.pipe.init()

    @inline(.always)
    def stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem.unsafe_offset(stage_offset)

    @inline(.always)
    def stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @inline(.always)
    def producer_mbar[*, qk_stage: Int = 0](self) -> MBarType:
        return self.pipe.producer_mbar[qk_stage]()

    @inline(.nodebug)
    def acquire[*, qk_stage: Int = 0](self):
        # Block until consumer has released this stage
        self.pipe.producer_acquire[qk_stage]()

    @inline(.nodebug)
    def commit_step(mut self):
        # After we have launched TMA copies for this stage
        # we advance producer's logical stage index.
        self.pipe.state.step()

    @inline(.nodebug)
    def commit_all(mut self):
        """Explicit-arrive commit for a non-TMA (manual SMEM-write) producer.

        Every thread of the `num_consumer`-wide... actually `num_producer`
        -wide producer role (e.g. a full warpgroup doing a manual SMEM
        re-swizzle store) calls this once; each thread's own local
        `PipelineState` copy steps identically, and the `num_producer`
        independent `arrive()` calls satisfy the mbar's expected count.
        Use this instead of `commit_step()` when there is no TMA hardware
        `expect_bytes` auto-arrival (i.e. the producer wrote SMEM directly,
        not via `cp.async.bulk.tensor`).
        """
        _ = self.pipe.producer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding ConsumerKVPipeline
# ------------------------------------------------------------------------------
struct DecodeKVConsumer[
    dtype: DType,
    config: MLA_SM100_Decode_Config,
    num_producer: Int = 1,
    num_consumer: Int = 2,
    num_stages: Int = config.num_kv_stages,
    stage_elems: Int = 0,
](TrivialRegisterPassable):
    """Consumer side of the decode KV pipeline that waits for and releases KV stages.

    Parameters:
        dtype: Element type of the KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
        num_producer: Number of producer threads arriving on each producer
            mbarrier (defaults to 1).
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier (defaults to 2, matching the standard mmaQK+mmaPV
            dual-consumer KV pipeline).
        num_stages: Ring depth (defaults to `config.num_kv_stages`). Set
            explicitly when a kernel stages KV through two rings of
            different depths.
        stage_elems: Elements per ring slot (0, the default, means one
            BN_QK x q_depth tile). Set for a ring of partial tiles.
    """

    comptime KVPipeType = KVPipelineGeneric[
        Self.num_stages, 1, Self.num_producer, Self.num_consumer
    ]
    # Stage element count tracks the producer (BN_QK x q_depth).
    comptime kv_stage_elems = (
        Self.stage_elems if Self.stage_elems
        > 0 else Self.config.BN_QK * Self.config.q_depth
    )

    var pipe: Self.KVPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @inline(.always)
    def __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        # NOTE: we copy the KVPipeline value – that's how FA4 does it.
        # Both sides keep their own PipelineState; the *barriers* do the real sync.
        self.pipe = pipe
        self.smem = smem

    @inline(.always)
    def stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem.unsafe_offset(stage_offset)

    @inline(.always)
    def stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @inline(.nodebug)
    def wait[*, qk_stage: Int = 0](self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait[qk_stage]()

    @inline(.nodebug)
    def release[*, qk_stage: Int = 0](mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release[qk_stage](e)

    @inline(.nodebug)
    def release_all(mut self):
        """Explicit-arrive release for a non-MMA (independent-thread) consumer.

        Every thread of the `num_consumer`-wide consumer role (e.g. a full
        warpgroup doing a manual SMEM re-swizzle read) calls this once; the
        `num_consumer` independent `arrive()` calls satisfy the mbar's
        expected count. Use this instead of `release[qk_stage](e)` (which
        uses `elect_mma_arrive` for a single elected thread per warp) when
        every thread independently participates, not just one MMA-eligible
        lane per warp.
        """
        _ = self.pipe.consumer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding KVPipelineGeneric
# ------------------------------------------------------------------------------
struct KVPipelineGeneric[
    num_kv_stages: Int,
    num_qk_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    """
    KVPipeline has `num_kv_stages * num_qk_stages` stages.
    `num_kv_stages` refers to how many `K` and `V` tiles we pipeline
    for performing the `S = Q@K'` and `O += P@V` MMAs.
    Each of these MMAs is broken up into `num_qk_stages` pipelined
    MMAs. We set `step=False` for all but the last MMA that completes
    the operation.
    An alternative implementation would separate the two, and potentially
    allow for more overall stages at the cost of slightly more bookkeeping.

    Parameters:
        num_kv_stages: Number of KV tiles pipelined for the `S = Q@K'`
            and `O += P@V` MMAs.
        num_qk_stages: Number of pipelined sub-MMAs each QK or PV MMA is
            broken into.
        num_producer: Number of producer threads arriving on each producer
            mbarrier.
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier.
    """

    comptime num_stages: Int = Self.num_kv_stages * Self.num_qk_stages

    # mbars are ordered in {producer, consumer} pairs
    @__allow_legacy_any_origin_fields
    var mbar: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @inline(.always)
    def __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @inline(.always)
    def init(self):
        # Consumer & Producer mbars: arrived by 1 thread performing TMA/mma
        comptime for i in range(Self.num_stages):
            self.mbar[i].init(Int32(Self.num_producer))

        comptime for i in range(Self.num_stages, Self.num_stages * 2):
            self.mbar[i].init(Int32(Self.num_consumer))

    @inline(.always)
    def producer_mbar[qk_stage: Int](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar.unsafe_offset(
            UInt32(Self.num_qk_stages) * idx
        ).unsafe_offset(qk_stage)

    @inline(.always)
    def consumer_mbar[qk_stage: Int](self, idx: UInt32) -> MBarType:
        comptime const_offset = qk_stage + Self.num_stages
        return self.mbar.unsafe_offset(
            UInt32(Self.num_qk_stages) * idx
        ).unsafe_offset(const_offset)

    @inline(.always)
    def consumer_mbar[qk_stage: Int](self) -> MBarType:
        return self.consumer_mbar[qk_stage](self.state.index())

    @inline(.nodebug)
    def producer_acquire[qk_stage: Int = Self.num_qk_stages - 1](self):
        """
        Returns the dynamic pipe idx.

        Parameters:
            qk_stage: QK sub-stage index whose consumer mbarrier to wait on
                (defaults to the last QK stage).
        """
        self.consumer_mbar[qk_stage]()[].wait(self.state.phase())

    @inline(.nodebug)
    def consumer_wait[qk_stage: Int = Self.num_qk_stages - 1](self):
        self.producer_mbar[qk_stage]()[].wait(self.state.phase())

    @inline(.nodebug)
    def consumer_release[
        qk_stage: Int = Self.num_qk_stages - 1
    ](mut self, e: Int32):
        elect_mma_arrive(self.consumer_mbar[qk_stage](), e)

        comptime if qk_stage == Self.num_qk_stages - 1:
            self.state.step()

    @staticmethod
    @inline(.always)
    def num_mbars() -> UInt32:
        return UInt32(2 * Self.num_qk_stages * Self.num_kv_stages)


# ------------------------------------------------------------------------------
# MLA decoding MiscMBars for producer and consumer
# ------------------------------------------------------------------------------
struct DecodeSM100MiscMBars[
    num_stages: Int, num_producer: Int, num_consumer: Int
](TrivialRegisterPassable):
    """Manages a generic producer/consumer mbarrier pair for the S, P, C, and O pipelines.

    Parameters:
        num_stages: Number of pipeline slots managed by the barrier pair.
        num_producer: Number of producer threads arriving on each producer
            mbarrier.
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier.
    """

    @__allow_legacy_any_origin_fields
    var mbar_base: MBarType

    # Generic barrier pair (producer + consumer) with num_stages slots.

    @inline(.always)
    def __init__(out self, mbar_base: MBarType):
        self.mbar_base = mbar_base

    @inline(.always)
    def init(self):
        # Layout: [prod[0..num_stages-1], cons[0..num_stages-1]]
        var s_pipe = MBarPipeline[Self.num_stages](self.mbar_base)
        # e.g. for S: 1 producer thread (elect in MMA warpgroup), 128 consumer threads (softmax warpgroup)
        # e.g. for P: 128 producer threads (softmax warpgroup), 1 consumer thread (elect in MMA warpgroup)
        s_pipe.init[
            num_producer=UInt32(Self.num_producer),
            num_consumer=UInt32(Self.num_consumer),
        ]()

    @inline(.always)
    def producer(self) -> ProducerPipeline[Self.num_stages]:
        return {self.mbar_base, self.mbar_base.unsafe_offset(Self.num_stages)}

    @inline(.always)
    def consumer(self) -> ConsumerPipeline[Self.num_stages]:
        return {self.mbar_base, self.mbar_base.unsafe_offset(Self.num_stages)}

    @inline(.always)
    def end(self) -> MBarType:
        # We consumed 2 * s_num_stages mbars: prod[2] + cons[2]
        return self.mbar_base.unsafe_offset(2 * Self.num_stages)


# ------------------------------------------------------------------------------
# MLA decoding S pipeline between MMA and Softmax
# ------------------------------------------------------------------------------
########## Producer of the S slot ##########
struct DecodeSProducer(TrivialRegisterPassable):
    """Producer side of the two-stage S pipeline between MMA and softmax."""

    comptime SNumStages = 2
    var pipe: ProducerPipeline[Self.SNumStages]

    @inline(.always)
    def __init__(out self, pipe: ProducerPipeline[Self.SNumStages]):
        # Copy initialized pipeline (state: index=0, phase=1)
        self.pipe = pipe

    @inline(.always)
    def acquire(self):
        # Wait for softmax to mark this S slot "free"
        self.pipe.acquire()

    @inline(.always)
    def slot_index(self) -> UInt32:
        return self.pipe.state.index()

    @inline(.always)
    def commit_mma(mut self, elect: Int32):
        # Signal "S slot is filled" to softmax
        self.pipe.commit_mma(elect)
        # Advance producer's stage/phase bookkeeping
        self.pipe.step()


########## Consumer of the S slot ##########
struct DecodeSConsumer(TrivialRegisterPassable):
    """Consumer side of the two-stage S pipeline between MMA and softmax."""

    comptime SNumStages = 2
    var pipe: ConsumerPipeline[Self.SNumStages]

    @inline(.always)
    def __init__(out self, pipe: ConsumerPipeline[Self.SNumStages]):
        self.pipe = pipe

    @inline(.always)
    def wait(self) -> UInt32:
        # Block until MMA has filled the current S slot
        self.pipe.wait()
        return self.pipe.state.index()

    @inline(.always)
    def release(mut self):
        # Mark this S slot as "consumed" so MMA can reuse it
        self.pipe.release()


# Parameterized versions for N-stage S pipeline (used by native FP8 with 3 stages)
struct DecodeSProducerN[num_stages: Int](TrivialRegisterPassable):
    """N-stage parameterized producer side of the S pipeline between MMA and softmax.

    Parameters:
        num_stages: Number of pipeline slots in the S buffer between MMA and
            softmax.
    """

    var pipe: ProducerPipeline[Self.num_stages]

    @inline(.always)
    def __init__(out self, pipe: ProducerPipeline[Self.num_stages]):
        self.pipe = pipe

    @inline(.always)
    def acquire(self):
        self.pipe.acquire()

    @inline(.always)
    def slot_index(self) -> UInt32:
        return self.pipe.state.index()

    @inline(.always)
    def commit_mma(mut self, elect: Int32):
        self.pipe.commit_mma(elect)
        self.pipe.step()


struct DecodeSConsumerN[num_stages: Int](TrivialRegisterPassable):
    """N-stage parameterized consumer side of the S pipeline between MMA and softmax.

    Parameters:
        num_stages: Number of pipeline slots in the S buffer between MMA and
            softmax.
    """

    var pipe: ConsumerPipeline[Self.num_stages]

    @inline(.always)
    def __init__(out self, pipe: ConsumerPipeline[Self.num_stages]):
        self.pipe = pipe

    @inline(.always)
    def wait(self) -> UInt32:
        self.pipe.wait()
        return self.pipe.state.index()

    @inline(.always)
    def release(mut self):
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding P Pipeline between Softmax and MMA
# ------------------------------------------------------------------------------
########## Producer of the P slot ##########
struct DecodePProducer(TrivialRegisterPassable):
    """Producer side of the two-stage P pipeline between softmax and MMA."""

    comptime PNumStages = 2
    var pipe: ProducerPipeline[Self.PNumStages]

    @inline(.always)
    def __init__(out self, pipe: ProducerPipeline[Self.PNumStages]):
        self.pipe = pipe

    # Softmax threads collectively wait until MMA has released P
    @inline(.always)
    def acquire(self):
        self.pipe.acquire()
        # -> consumer_mbar.wait(phase), all 128 threads see the same phase

    # After writing P, all 128 threads call commit()
    @inline(.nodebug)
    def commit(mut self):
        self.pipe.commit()
        # -> producer_mbar.arrive() (128 arrivals total)
        # -> state.step() (phase toggles for next iteration)

    # optional helper
    @inline(.always)
    def stage_index(self) -> UInt32:
        return self.pipe.state.index()


########## Consumer of the P slot ##########
struct DecodePConsumer(TrivialRegisterPassable):
    """Consumer side of the two-stage P pipeline between softmax and MMA."""

    comptime PNumStages = 2
    var pipe: ConsumerPipeline[Self.PNumStages]

    @inline(.always)
    def __init__(out self, pipe: ConsumerPipeline[Self.PNumStages]):
        self.pipe = pipe

    # Should be called by MMA elect thread only
    @inline(.nodebug)
    def wait(self) -> UInt32:
        self.pipe.wait()
        return self.pipe.state.index()
        # -> producer_mbar.wait(phase)
        # blocks until 128 Softmax commits complete

    # Also called by MMA elect thread only

    @inline(.nodebug)
    def release_mma(mut self, elect: Int32):
        # Like KVPipeline.consumer_release but for generic pipeline
        var mbar = self.pipe.consumer_mbar()
        elect_mma_arrive(mbar, elect)
        self.pipe.step()


# Parameterized versions for N-stage P pipeline (used by native FP8 with 3 stages)
struct DecodePProducerN[num_stages: Int](TrivialRegisterPassable):
    """N-stage parameterized producer side of the P pipeline between softmax and MMA.

    Parameters:
        num_stages: Number of pipeline slots in the P buffer between softmax
            and MMA.
    """

    var pipe: ProducerPipeline[Self.num_stages]

    @inline(.always)
    def __init__(out self, pipe: ProducerPipeline[Self.num_stages]):
        self.pipe = pipe

    @inline(.always)
    def acquire(self):
        self.pipe.acquire()

    @inline(.nodebug)
    def commit(mut self):
        self.pipe.commit()

    @inline(.always)
    def stage_index(self) -> UInt32:
        return self.pipe.state.index()


struct DecodePConsumerN[num_stages: Int](TrivialRegisterPassable):
    """N-stage parameterized consumer side of the P pipeline between softmax and MMA.

    Parameters:
        num_stages: Number of pipeline slots in the P buffer between softmax
            and MMA.
    """

    var pipe: ConsumerPipeline[Self.num_stages]

    @inline(.always)
    def __init__(out self, pipe: ConsumerPipeline[Self.num_stages]):
        self.pipe = pipe

    @inline(.nodebug)
    def wait(self) -> UInt32:
        self.pipe.wait()
        return self.pipe.state.index()

    @inline(.nodebug)
    def release_mma(mut self, elect: Int32):
        var mbar = self.pipe.consumer_mbar()
        elect_mma_arrive(mbar, elect)
        self.pipe.step()


# ------------------------------------------------------------------------------
# MLA decoding O pipeline between MMA and Correction
# ------------------------------------------------------------------------------
########## Producer of the O slot ##########
struct DecodeOProducer(TrivialRegisterPassable):
    """Producer side of the two-stage O pipeline between MMA and correction."""

    comptime ONumStages = 2
    var pipe: ProducerPipeline[Self.ONumStages]

    @inline(.always)
    def __init__(out self, pipe: ProducerPipeline[Self.ONumStages]):
        # Copy initialized pipeline (state: index=0, phase=1)
        self.pipe = pipe

    @inline(.always)
    def acquire(self):
        # Wait for correction to mark this O slot "free"
        self.pipe.acquire()

    @inline(.always)
    def slot_index(self) -> UInt32:
        return self.pipe.state.index()

    @inline(.always)
    def commit_mma(mut self, elect: Int32):
        # Signal "O slot is filled" to correction
        self.pipe.commit_mma(elect)
        # Advance producer's stage/phase bookkeeping
        self.pipe.step()


########## Consumer of the O slot ##########
struct DecodeOConsumer(TrivialRegisterPassable):
    """Consumer side of the two-stage O pipeline between MMA and correction."""

    comptime ONumStages = 2
    var pipe: ConsumerPipeline[Self.ONumStages]

    @inline(.always)
    def __init__(out self, pipe: ConsumerPipeline[Self.ONumStages]):
        self.pipe = pipe

    @inline(.always)
    def wait(self):
        # Block until MMA has filled the current O slot
        self.pipe.wait()
        _ = self.pipe.state.index()

    @inline(.always)
    def release(mut self):
        # Mark this O slot as "consumed" so MMA can reuse it
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding C Pipeline between Softmax and Correction
# ------------------------------------------------------------------------------
struct DecodeCProducer(TrivialRegisterPassable):
    """Producer side of the single-stage C pipeline between softmax and correction.
    """

    comptime CNumStages = 1
    var pipe: ProducerPipeline[Self.CNumStages]

    @inline(.always)
    def __init__(out self, pipe: ProducerPipeline[Self.CNumStages]):
        self.pipe = pipe

    # Softmax warpgroup: all 128 threads call acquire() before writing corr scalars
    @inline(.nodebug)
    def acquire(self):
        self.pipe.acquire()
        # -> consumer_mbar.wait(phase) on correction side (prev iteration)

    # After writing correction scalars for this O:
    @inline(.nodebug)
    def commit(mut self):
        self.pipe.commit()
        # producer_mbar.arrive() from 128 threads + state.step()


struct DecodeCConsumer(TrivialRegisterPassable):
    """Consumer side of the single-stage C pipeline between softmax and correction.
    """

    comptime CNumStages = 1
    var pipe: ConsumerPipeline[Self.CNumStages]

    # Correction warpgroup: all 128 threads wait until correction scalars are ready
    @inline(.always)
    def __init__(out self, pipe: ConsumerPipeline[Self.CNumStages]):
        self.pipe = pipe

    @inline(.nodebug)
    def wait(self):
        # perform producer_mbar.wait(phase)
        self.pipe.wait()

    @inline(.nodebug)
    def release(mut self):
        # perform consumer_mbar.arrive() from 128 threads + state.step()
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding  pipeline correction is the producer and write is the consumer
# ------------------------------------------------------------------------------


struct OutPipeline[
    num_out_stages: Int,
    num_producer: Int,
    num_consumer: Int,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """
    OutPipeline has `num_out_stages` stages.
    `num_out_stages` refers to how many output stages we pipeline
    for performing the output store.

    Parameters:
        num_out_stages: Number of output tiles pipelined for the output
            store.
        num_producer: Number of producer threads arriving on each producer
            mbarrier.
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier.
        cta_group: The kernel's tcgen05 CTA granularity; the store's commit
            must match it -- a pair may not mix cta_group::1 and ::2 (ptxas
            rejects the mix).
    """

    comptime num_stages: Int = Self.num_out_stages

    # mbars are ordered in {producer, consumer} pairs
    @__allow_legacy_any_origin_fields
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @inline(.always)
    def __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @inline(.always)
    def init(self):
        # Consumer & Producer mbars: arrived by num_producer and num_consumer threads
        comptime for i in range(Self.num_stages):
            self.mbar[i].init(Int32(Self.num_producer))

        comptime for i in range(Self.num_stages):
            self.mbar.unsafe_offset(Self.num_stages)[i].init(
                Int32(Self.num_consumer)
            )

    @inline(.always)
    def producer_mbar(self) -> MBarType:
        return self.mbar

    @inline(.always)
    def consumer_mbar(self) -> MBarType:
        return self.mbar.unsafe_offset(Self.num_stages)

    @inline(.nodebug)
    def producer_acquire(self):
        """
        Returns the dynamic pipe idx.
        """
        var idx = self.state.index()
        self.consumer_mbar()[idx].wait(self.state.phase())

    @inline(.nodebug)
    def consumer_wait(self):
        var idx = self.state.index()
        self.producer_mbar()[idx].wait(self.state.phase())

    @inline(.nodebug)
    def consumer_release[](mut self, e: Int32):
        var idx = self.state.index()
        comptime if Self.cta_group == 2:
            # The pair's store runs on both CTAs, but a cta_group::2
            # tcgen05 commit may be issued only by the leader. This
            # barrier is local to each CTA, so the elected lane arrives it
            # directly.
            if e != 0:
                _ = (self.consumer_mbar().unsafe_offset(idx))[].arrive()
        else:
            elect_mma_arrive[cta_group=Self.cta_group](
                self.consumer_mbar().unsafe_offset(idx), e
            )
        self.state.step()

    @inline(.nodebug)
    def producer_commit(mut self):
        # All 128 producer threads should call this.
        # mbar was initialized with num_producer = WARPGROUP_SIZE,
        # so producer_mbar()[].arrive() must be called by each producer thread.
        var idx = self.state.index()
        _ = self.producer_mbar()[idx].arrive()
        self.state.step()

    @staticmethod
    @inline(.always)
    def num_mbars() -> UInt32:
        return UInt32(2 * Self.num_stages)


struct DecodeOutProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Producer side of the output writeback pipeline that stages output tiles in SMEM for TMA store.

    Parameters:
        dtype: Element type of the output tiles stored in SMEM.
        config: Decode config supplying output tile dimensions and stage
            count.
    """

    # Output writeback uses BN_PV/4 (the per-warp stripe width), not BN_QK,
    # so Layout-G-128 (BN_QK=128) still emits 64-col stripes.
    comptime col_per_warp = Self.config.MMA_PV_N // 2
    comptime num_out_blocks: Int = Self.config.depth // (Self.config.BN_PV // 4)
    comptime block_per_warp = Self.col_per_warp // (Self.config.BN_PV // 4)
    comptime blocks_per_stage = 2 if Self.block_per_warp != 0 else 1
    comptime num_out_stages: Int = Self.num_out_blocks // Self.blocks_per_stage
    comptime OutPipeType = OutPipeline[
        Self.num_out_stages, WARPGROUP_SIZE, 1, Self.config.cta_group
    ]

    # Per-stage SMEM = BM rows x BN_PV/4 cols.
    comptime out_stage_elems = Self.config.BM * (Self.config.BN_PV // 4)
    comptime out_stage_bytes = Self.out_stage_elems * size_of[Self.dtype]()

    var pipe: Self.OutPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @inline(.always)
    def __init__(
        out self,
        pipe: Self.OutPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

        # IMPORTANT: producer starts at phase 1, like FA4
        self.pipe.state._phase = 1

    @inline(.always)
    def init(self):
        # Only producer OR consumer should call init(), not both.
        self.pipe.init()

    @inline(.always)
    def stage_base_ptr(
        self, half_idx: Int
    ) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(
            Self.out_stage_elems
        ) * UInt32(Self.blocks_per_stage) + UInt32(
            half_idx * Self.out_stage_elems
        )
        return self.smem.unsafe_offset(stage_offset)

    @inline(.always)
    def producer_mbar(self) -> MBarType:
        return self.pipe.producer_mbar()

    @inline(.nodebug)
    def acquire(self):
        # Block until consumer has released this stage
        self.pipe.producer_acquire()

    @inline(.nodebug)
    def commit_step(mut self):
        # After we have launched TMA copies for this stage
        # we advance producer's logical stage index.

        self.pipe.producer_commit()


struct DecodeOutConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Consumer side of the output writeback pipeline that waits for and releases output stages.

    Parameters:
        dtype: Element type of the output tiles stored in SMEM.
        config: Decode config supplying output tile dimensions and stage
            count.
    """

    # Mirrors `DecodeOutProducer` — see there for the BN_PV/4 anchoring.
    comptime col_per_warp = Self.config.MMA_PV_N // 2
    comptime num_out_blocks: Int = Self.config.depth // (Self.config.BN_PV // 4)
    comptime block_per_warp = Self.col_per_warp // (Self.config.BN_PV // 4)
    comptime blocks_per_stage = 2 if Self.block_per_warp != 0 else 1
    comptime num_out_stages: Int = Self.num_out_blocks // Self.blocks_per_stage
    comptime OutPipeType = OutPipeline[
        Self.num_out_stages, WARPGROUP_SIZE, 1, Self.config.cta_group
    ]
    comptime out_stage_elems = Self.config.BM * (Self.config.BN_PV // 4)

    var pipe: Self.OutPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @inline(.always)
    def __init__(
        out self,
        pipe: Self.OutPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

    @inline(.always)
    def stage_base_ptr(
        self, half_idx: Int
    ) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(
            Self.out_stage_elems
        ) * UInt32(Self.blocks_per_stage) + UInt32(
            half_idx * Self.out_stage_elems
        )
        return self.smem.unsafe_offset(stage_offset)

    @inline(.nodebug)
    def wait(self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait()

    @inline(.nodebug)
    def release(mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release(e)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the QK^T MMA with both Q and K operands in SMEM.

    Parameters:
        operand_type: Element type of the Q and K operands in SMEM.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = Self.config.MMA_K  # 16
    comptime BK = Self.config.BK_QK  # 576
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        (UInt32(Self.MMA_M), UInt32(Self.MMA_N)),
        transpose_b=True,  # QKᵀ
    ]()

    @staticmethod
    @inline(.always)
    def descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        # Q: 64 x 64, k-major, same swizzle as TMA
        var base = q_smem
        return smem_descriptor[
            BMN=Self.config.BM,  # 64 rows
            BK=Self.BK,  # 576 (padded_q_depth)
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is 64 x 64, k-major, same swizzle as k_tma
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 576 columns
            swizzle_mode=Self.config.kv_mma_swizzle_mode,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F16,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=Self.config.swizzle_mode,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=Self.config.kv_mma_swizzle_mode,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)

    @staticmethod
    @inline(.always)
    def mma_block[
        *, block_idx: Int, num_blocks: Int
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        # One K-block slice of `mma` (same descriptors, same layouts):
        # absolute k-mmas so the emitted sequence over all blocks matches
        # the bulk form. Block 0 applies c_scale; later blocks accumulate.
        comptime assert Self.num_k_mmas % num_blocks == 0, "uneven K blocks"
        comptime block_k_mmas = Self.num_k_mmas // num_blocks
        bulk_mma_ws[
            UMMAKind.KIND_F16,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=Self.config.swizzle_mode,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=Self.config.kv_mma_swizzle_mode,
            b_is_k_major=True,
            num_k_mmas=block_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            k_start=block_idx * block_k_mmas,
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


struct DecodeSM100PVSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the PV MMA with both P and V operands in SMEM.

    Parameters:
        operand_type: Element type of the P and V operands in SMEM.
        accum_type: Accumulator dtype used for the PV MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_PV_N
    comptime MMA_K = Self.config.MMA_K  # 16
    comptime BM = Self.config.BM  # 64
    comptime BN_PV = Self.MMA_N  # 256
    comptime BK = Self.config.BK_PV  # 64
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor -----
    comptime UMMAPVSS = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        (UInt32(Self.MMA_M), UInt32(Self.MMA_N)),
        transpose_b=False,  # P (k-major) * V (mn-major) = no transpose
    ]()

    @staticmethod
    @inline(.always)
    def descriptor_v_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is BDepth_max x 64, mn-major, same swizzle as k_tma
        return smem_descriptor[
            BMN=Self.BN_PV,
            BK=Self.BK,  # 64 rows
            swizzle_mode=Self.config.kv_mma_swizzle_mode,
            is_k_major=False,
        ](base)

    @staticmethod
    @inline(.always)
    def descriptor_p_block(
        p_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = p_smem
        # P: 64 x 64, k-major, same swizzle as Q/K
        return smem_descriptor[
            BMN=Self.BM,  # 64 rows
            BK=Self.BK,  # 64 columns
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=True,  # P is k-major
        ](base)

    @staticmethod
    @inline(.always)
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F16,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.BM,
            a_BK=Self.BK,
            a_swizzle=Self.config.swizzle_mode,
            a_is_k_major=True,
            b_BMN=Self.BN_PV,
            b_BK=Self.BK,
            b_swizzle=Self.config.kv_mma_swizzle_mode,
            b_is_k_major=False,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAPVSS, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorTS for QKT (TMEM Q, SMEM K)
# ------------------------------------------------------------------------------
struct DecodeSM100QKTTS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the QK^T MMA with Q in TMEM and K in SMEM.

    Parameters:
        operand_type: Element type of the Q (in TMEM) and K (in SMEM)
            operands.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = Self.config.MMA_K  # 16
    comptime BK = Self.config.BK_QK  # 576
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        (UInt32(Self.MMA_M), UInt32(Self.MMA_N)),
        transpose_b=True,  # QKᵀ
    ]()

    @staticmethod
    @inline(.always)
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is 64 x 576, k-major, same swizzle as k_tma
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 576 columns
            swizzle_mode=Self.config.kv_mma_swizzle_mode,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def mma[
        *, stage_idx: Int = 0
    ](
        a: UInt32,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws_ts[
            UMMAKind.KIND_F16,
            Self.operand_type,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=Self.config.kv_mma_swizzle_mode,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT — FP8 (KIND_F8F6F4, MMA_K=32), or
# KIND_F16 (MMA_K=16) for a 2-byte operand.
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS_FP8[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
    k_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_64B,
    k_row_elems: Int = 0,
    q_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_64B,
](TrivialRegisterPassable):
    """Tensor accumulator for the QK^T MMA with both Q and K in SMEM.

    Parameters:
        operand_type: Element type of the Q and K operands in SMEM: FP8, or
            a 2-byte type for the `KIND_F16` form.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
        k_swizzle: SMEM swizzle mode of the K operand (defaults to
            `SWIZZLE_64B`).
        k_row_elems: SMEM width of a K row in elements (0, the default,
            means `BK`). Wider when K is padded to whole swizzle blocks.
        q_swizzle: SMEM swizzle mode of the Q operand (defaults to
            `SWIZZLE_64B`).
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows, or 128 across the pair
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols, or the pair's 128 keys
    comptime operand_size = size_of[Self.operand_type]()
    comptime mma_kind = (
        UMMAKind.KIND_F8F6F4 if Self.operand_size == 1 else UMMAKind.KIND_F16
    )
    comptime MMA_K = 32 // Self.operand_size
    comptime BK = Self.config.BK_QK
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime K_BK = Self.k_row_elems if Self.k_row_elems > 0 else Self.BK
    # The non-ws `cta_group::2` form (see `mma`): the leader issues one MMA
    # over both CTAs' SMEM, each CTA supplying its MMA_M / 2 Q rows and its
    # MMA_N / 2 K keys, so the SMEM layouts step K with the per-CTA extents
    # (mirrors `SM100TensorAccumulator`'s cta_group handling).
    comptime is_cg2 = Self.config.cta_group == 2
    comptime swizzle_granularity = max(
        Self.q_swizzle.bytes(),
        Self.k_swizzle.bytes(),
    ) // Self.operand_size
    comptime padded_BK = align_up(Self.BK, Self.swizzle_granularity)
    comptime cg2_a_layout = tile_layout_k_major_typed[
        Self.operand_type,
        align_up(Self.MMA_M // 2, 8),
        Self.padded_BK,
        Self.q_swizzle,
    ]
    comptime cg2_b_layout = tile_layout_k_major_typed[
        Self.operand_type,
        Self.MMA_N // 2,
        max(Self.K_BK, Self.padded_BK),
        Self.k_swizzle,
    ]

    comptime UMMAInstDesc = UMMAInsDescriptor[Self.mma_kind].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        (UInt32(Self.MMA_M), UInt32(Self.MMA_N)),
        transpose_b=True,  # QKT
    ]()

    @staticmethod
    @inline(.always)
    def descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = q_smem
        return smem_descriptor[
            BMN=Self.config.BM,  # 64 rows
            BK=Self.BK,
            swizzle_mode=Self.q_swizzle,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        return smem_descriptor[
            # Each CTA of the pair supplies half of the MMA's N keys from
            # its own K slot; the descriptor describes that per-CTA half.
            BMN=Self.config.BN_QK // Self.config.cta_group,
            BK=Self.K_BK,  # the K row's SMEM width, block-padded
            swizzle_mode=Self.k_swizzle,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        comptime if Self.is_cg2:
            # The pair's non-ws form: one `tcgen05.mma.cta_group::2` over
            # both CTAs' SMEM, issued by the leader's elected lane only.
            bulk_mma[
                type_of(Self.cg2_a_layout),
                Self.cg2_a_layout,
                type_of(Self.cg2_b_layout),
                Self.cg2_b_layout,
                num_k_mmas=Self.num_k_mmas,
                mma_k=Self.MMA_K,
                operand_size=Self.operand_size,
                cta_group=2,
            ](Self.UMMAInstDesc, a, b, c, c_scale, elect)
        else:
            bulk_mma_ws[
                Self.mma_kind,
                Self.operand_type,
                Self.operand_type,
                a_BMN=Self.config.BM,
                a_BK=Self.BK,
                a_swizzle=Self.q_swizzle,
                a_is_k_major=True,
                b_BMN=Self.config.BN_QK,
                b_BK=Self.K_BK,
                b_swizzle=Self.k_swizzle,
                b_is_k_major=True,
                num_k_mmas=Self.num_k_mmas,
                operand_size=Self.operand_size,
                tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
                mma_k=Self.MMA_K,
            ](Self.UMMAInstDesc, a, b, c, c_scale, elect)

    @staticmethod
    @inline(.always)
    def mma_block[
        *, block_idx: Int, num_blocks: Int
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        """Issues K-block `block_idx` of `num_blocks` equal slices of `mma`.

        The slices use the full-tile descriptors and absolute k-mmas, so
        issuing every block in order emits `mma`'s sequence. Block 0 applies
        `c_scale`; later blocks accumulate. The pair's block form slices the
        non-ws cg2 contraction the same way, at the same absolute k-mmas.
        """
        comptime assert Self.num_k_mmas % num_blocks == 0, "uneven K blocks"
        comptime block_k_mmas = Self.num_k_mmas // num_blocks
        comptime if Self.is_cg2:
            bulk_mma[
                type_of(Self.cg2_a_layout),
                Self.cg2_a_layout,
                type_of(Self.cg2_b_layout),
                Self.cg2_b_layout,
                num_k_mmas=block_k_mmas,
                mma_k=Self.MMA_K,
                operand_size=Self.operand_size,
                cta_group=2,
                k_start=block_idx * block_k_mmas,
            ](Self.UMMAInstDesc, a, b, c, c_scale, elect)
        else:
            bulk_mma_ws[
                Self.mma_kind,
                Self.operand_type,
                Self.operand_type,
                a_BMN=Self.config.BM,
                a_BK=Self.BK,
                a_swizzle=Self.q_swizzle,
                a_is_k_major=True,
                b_BMN=Self.config.BN_QK,
                b_BK=Self.K_BK,
                b_swizzle=Self.k_swizzle,
                b_is_k_major=True,
                num_k_mmas=block_k_mmas,
                operand_size=Self.operand_size,
                tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
                mma_k=Self.MMA_K,
                k_start=block_idx * block_k_mmas,
            ](Self.UMMAInstDesc, a, b, c, c_scale, elect)

    @staticmethod
    @inline(.always)
    @staticmethod
    @inline(.always)
    def mma_cols[
        *, col_start: Int, cols: Int
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        """Issues the k-mmas of K columns `[col_start, col_start + cols)`.

        For K blocks wider than the contraction's tail: the last block's
        slice stops at `BK`, so issuing every block in order still emits
        `mma`'s sequence. The first slice applies `c_scale`.
        """
        comptime k0 = col_start // Self.MMA_K
        comptime n = min(cols // Self.MMA_K, Self.num_k_mmas - k0)
        comptime assert (
            col_start % Self.MMA_K == 0 and cols % Self.MMA_K == 0 and n > 0
        ), "column slice off the k-mma grid"
        comptime assert not Self.is_cg2, "unpaired only"
        bulk_mma_ws[
            Self.mma_kind,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=Self.q_swizzle,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.K_BK,
            b_swizzle=Self.k_swizzle,
            b_is_k_major=True,
            num_k_mmas=n,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            mma_k=Self.MMA_K,
            k_start=k0,
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT — split content FP8 (KIND_F8F6F4)
# Content-only: Q_nope and K_nope are FP8 e4m3 in SMEM.
# BK = padded_depth (512), MMA_K=32, SWIZZLE_64B.
# Used by the per-token-scale rope-aware split content/rope kernel.
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS_Content_FP8[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the content-only FP8 QK^T MMA used by the per-token-scale rope-aware kernel.

    Parameters:
        operand_type: FP8 element type of the content Q and K operands in
            SMEM.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = 32  # FP8 MMA_K
    comptime BK = Self.config.padded_depth  # 512 (content only)
    comptime num_k_mmas = Self.BK // Self.MMA_K  # 16
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor for FP8 -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        (UInt32(Self.MMA_M), UInt32(Self.MMA_N)),
        transpose_b=True,  # QK^T
    ]()

    @staticmethod
    @inline(.always)
    def descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = q_smem
        return smem_descriptor[
            BMN=Self.config.BM,  # 64 rows
            BK=Self.BK,  # 512 (padded_depth)
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 512 columns
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F8F6F4,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            mma_k=Self.MMA_K,
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT — split rope BF16 (KIND_F16)
# Rope-only: Q_rope and K_rope are BF16 in SMEM.
# BK = rope_depth (64), MMA_K=16, SWIZZLE_128B.
# Used by the per-token-scale rope-aware split content/rope kernel.
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS_Rope_BF16[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the rope-only BF16 QK^T MMA used by the per-token-scale rope-aware kernel.

    Parameters:
        operand_type: BF16 element type of the rope Q and K operands in
            SMEM.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = 16  # BF16 MMA_K
    comptime BK = Self.config.rope_depth  # 64 (rope only)
    comptime num_k_mmas = Self.BK // Self.MMA_K  # 4
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor for BF16 -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        (UInt32(Self.MMA_M), UInt32(Self.MMA_N)),
        transpose_b=True,  # QK^T
    ]()

    @staticmethod
    @inline(.always)
    def descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = q_smem
        return smem_descriptor[
            BMN=Self.config.BM,  # 64 rows
            BK=Self.BK,  # 64 (rope_depth)
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 64 columns
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=True,
        ](base)

    @staticmethod
    @inline(.always)
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F16,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for PV — FP8 (KIND_F8F6F4, MMA_K=32), or
# KIND_F16 (MMA_K=16) for a 2-byte operand.
# ------------------------------------------------------------------------------
struct DecodeSM100PVSS_FP8[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
    p_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_64B,
    v_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_64B,
](TrivialRegisterPassable):
    """Tensor accumulator for the PV MMA with both P and V in SMEM.

    Parameters:
        operand_type: Element type of the P and V operands in SMEM: FP8, or
            a 2-byte type for the `KIND_F16` form.
        accum_type: Accumulator dtype used for the PV MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
        p_swizzle: SMEM swizzle mode applied to the P operand descriptor
            (defaults to `SWIZZLE_64B`).
        v_swizzle: SMEM swizzle mode of the V operand (defaults to
            `SWIZZLE_64B`).
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows, or 128 across the pair
    comptime MMA_N = Self.config.MMA_PV_N
    comptime operand_size = size_of[Self.operand_type]()
    comptime mma_kind = (
        UMMAKind.KIND_F8F6F4 if Self.operand_size == 1 else UMMAKind.KIND_F16
    )
    comptime MMA_K = 32 // Self.operand_size
    comptime BM = Self.config.BM  # 64
    comptime BN_PV = Self.MMA_N  # 256
    comptime BK = Self.config.BK_PV  # 64
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime VPhysicalKeys = (
        Self.config.BN_QK if Self.config.cta_group == 1 else Self.BK
    )
    # The non-ws `cta_group::2` form (see `mma`): the leader issues one MMA
    # over both CTAs' SMEM; each CTA supplies its MMA_M / 2 P rows and the
    # MMA_N / 2 V depth columns it owns of its V ring, so the layouts step K
    # with the per-CTA extents (mirrors `SM100TensorAccumulator`'s cta_group
    # handling).
    comptime is_cg2 = Self.config.cta_group == 2
    comptime swizzle_granularity = max(
        Self.p_swizzle.bytes(),
        Self.v_swizzle.bytes(),
    ) // Self.operand_size
    comptime padded_BK = align_up(Self.BK, Self.swizzle_granularity)
    comptime cg2_a_layout = tile_layout_k_major_typed[
        Self.operand_type,
        align_up(Self.MMA_M // 2, 8),
        Self.padded_BK,
        Self.p_swizzle,
    ]
    comptime cg2_b_layout = tile_layout_mn_major_typed[
        Self.operand_type,
        Self.MMA_N // 2,
        Self.padded_BK,
        Self.v_swizzle,
    ]

    comptime UMMAPVSS = UMMAInsDescriptor[Self.mma_kind].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        (UInt32(Self.MMA_M), UInt32(Self.MMA_N)),
        transpose_b=False,  # P (k-major) * V (mn-major) = no transpose
    ]()

    @staticmethod
    @inline(.always)
    def descriptor_v_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        return smem_descriptor[
            # Each CTA of the pair supplies half of the MMA's N depth
            # columns from its own V-pair buffer; the descriptor describes
            # that per-CTA half.
            BMN=Self.BN_PV // Self.config.cta_group,
            BK=Self.VPhysicalKeys,
            swizzle_mode=Self.v_swizzle,
            is_k_major=False,
        ](base)

    @staticmethod
    @inline(.always)
    def descriptor_p_block(
        p_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = p_smem
        return smem_descriptor[
            BMN=Self.BM,  # 64 rows
            BK=Self.BK,  # 64 columns
            swizzle_mode=Self.p_swizzle,
            is_k_major=True,  # P is k-major
        ](base)

    @staticmethod
    @inline(.always)
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        comptime if Self.is_cg2:
            # The pair's non-ws form: one `tcgen05.mma.cta_group::2` over
            # both CTAs' SMEM, issued by the leader's elected lane only.
            bulk_mma[
                type_of(Self.cg2_a_layout),
                Self.cg2_a_layout,
                type_of(Self.cg2_b_layout),
                Self.cg2_b_layout,
                num_k_mmas=Self.num_k_mmas,
                mma_k=Self.MMA_K,
                operand_size=Self.operand_size,
                cta_group=2,
            ](Self.UMMAPVSS, a, b, c, c_scale, elect)
        else:
            bulk_mma_ws[
                Self.mma_kind,
                Self.operand_type,
                Self.operand_type,
                a_BMN=Self.BM,
                a_BK=Self.BK,
                a_swizzle=Self.p_swizzle,
                a_is_k_major=True,
                b_BMN=Self.BN_PV,
                b_BK=Self.VPhysicalKeys,
                b_swizzle=Self.v_swizzle,
                b_is_k_major=False,
                num_k_mmas=Self.num_k_mmas,
                operand_size=Self.operand_size,
                tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
                mma_k=Self.MMA_K,
            ](Self.UMMAPVSS, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# Helper functions for writing from local memory to shared memory using swizzle
# ------------------------------------------------------------------------------


@inline(.always)
def write_bf16x2_row_to_smem_chunked[
    local_tile_size: Int,
    *,
    out_dtype: DType,
    in_dtype: DType,
    config: MLA_SM100_Decode_Config,
    chunk_size: Int = 16,
    scale_needed: Bool = False,
](
    shared_mem: MutPointer[
        Scalar[out_dtype], MutAnyOrigin, address_space=.SHARED
    ],
    local_mem: LocalTensor[in_dtype, row_major[local_tile_size](), _],
    col_start: Int,
    row_start: Int,
    scale: Scalar[in_dtype] = 1.0,
):
    """Chunked write with optional scaling. Reduces register pressure.

    Parameters:
        local_tile_size: Number of source elements held by each lane in
            the local tile; must be divisible by `chunk_size`.
        out_dtype: Output dtype written to SMEM, packed as `bf16x2` stores.
        in_dtype: Input dtype of the per-lane local register tile.
        config: Decode config supplying `BN_PV` used to size the SMEM
            row stripe and the ldmatrix swizzle.
        chunk_size: Number of source elements written per chunk (must be
            a multiple of 8; defaults to 16).
        scale_needed: When `True`, multiply each register by `scale`
            before casting to `out_dtype` (defaults to `False`).

    Args:
        shared_mem: Pointer to the destination shared memory region in
            swizzled `bf16x2` layout.
        local_mem: Per-lane local tensor holding the source tile in
            `row_major[local_tile_size]` layout.
        col_start: Starting column offset within the SMEM row stripe.
        row_start: Starting row offset within the SMEM row stripe.
        scale: Scalar multiplier applied before cast when `scale_needed`
            is `True` (defaults to `1.0`).
    """
    comptime num_chunks = local_tile_size // chunk_size
    comptime groups_per_chunk = chunk_size // 8
    comptime total_groups = num_chunks * groups_per_chunk

    # SMEM row width = BN_PV/4 (per-warp stripe).
    comptime swz = make_ldmatrix_swizzle[
        dtype=out_dtype,
        row_size=config.BN_PV // 4,
        log2_vector_width=3,
    ]()

    # Precompute all swizzle offsets before the loop
    var phys_offsets = StaticTuple[Int, total_groups]()

    comptime for i in range(total_groups):
        comptime chunk_idx, group_idx = divmod(i, groups_per_chunk)
        comptime col_offset = chunk_idx * chunk_size + group_idx * 8
        var logical_elem = (
            row_start * (config.BN_PV // 4) + col_start + col_offset
        )
        phys_offsets[i] = swz(logical_elem)

    var lmv = local_mem.vectorize[8]()

    comptime for chunk in range(0, num_chunks):
        comptime for g in range(0, groups_per_chunk):
            # Compute the correct index into the vectorized view
            # vec_idx accounts for both chunk offset and position within chunk
            comptime vec_idx = chunk * groups_per_chunk + g

            var vec_val = lmv[vec_idx]

            comptime if scale_needed:
                vec_val *= scale

            var bf16_vec = vec_val.cast[out_dtype]()
            var packed = bitcast[.uint32, 4](bf16_vec)
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=out_dtype](
                shared_mem,
                phys_offsets[vec_idx],
                packed,
            )


@inline(.always)
def write_fp8_row_to_smem_chunked[
    local_tile_size: Int,
    *,
    out_dtype: DType,
    in_dtype: DType,
    config: MLA_SM100_Decode_Config,
    chunk_size: Int = 16,
    scale_needed: Bool = False,
    row_size: Int = config.BN_QK,
    swizzle_kind: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_64B,
](
    shared_mem: MutPointer[
        Scalar[out_dtype], MutAnyOrigin, address_space=.SHARED
    ],
    local_mem: LocalTensor[in_dtype, row_major[local_tile_size](), _],
    col_start: Int,
    row_start: Int,
    scale: Scalar[in_dtype] = 1.0,
):
    """Writes float32 data to SMEM as FP8 with FP8-byte swizzle.

    Each group writes 16 FP8 elements = 16 bytes = 4 x uint32.

    Parameters:
        local_tile_size: Number of fp32 source elements held by this lane.
        out_dtype: Output FP8 dtype written to SMEM.
        in_dtype: Input fp32 register dtype.
        config: MLA decode config (used for default `row_size`).
        chunk_size: Number of fp32 elements per chunk (must be a multiple of 16).
        scale_needed: When True, multiply each register by `scale` before cast.
        row_size: Logical SMEM row width in FP8 elements. Defaults to
            `config.BN_QK` for backward compatibility with Layout-G-64 / Layout-E.
            Layout-G-128 passes `row_size = BK_PV = 128`.
        swizzle_kind: Swizzle scheme used by the consuming MMA descriptor.
            Defaults to `SWIZZLE_64B` (existing behaviour). Layout-G-128 must
            pass `SWIZZLE_128B` so the writer's address pattern matches the
            new K=128 P-tile descriptor.

    Args:
        shared_mem: Pointer to the destination shared memory region in
            FP8-byte-swizzled layout.
        local_mem: Per-lane local tensor holding the float32 source tile
            in `row_major[local_tile_size]` layout.
        col_start: Starting column offset within the SMEM row.
        row_start: Starting row offset within the SMEM row.
        scale: Scalar multiplier applied to each register before cast
            when `scale_needed` is `True` (defaults to `1.0`).
    """
    comptime num_chunks = local_tile_size // chunk_size
    comptime groups_per_chunk = chunk_size // 16  # 16 FP8 elements per store
    comptime total_groups = num_chunks * groups_per_chunk

    # `make_ldmatrix_swizzle` produces the right SWIZZLE_64B / SWIZZLE_128B
    # address pattern when given the matching row_size and
    # log2_vector_width=4 (16 FP8 elements = 16 B per store).
    comptime swz = make_ldmatrix_swizzle[
        dtype=out_dtype,
        row_size=row_size,
        log2_vector_width=4,  # log2(16) for 16 FP8 elements
    ]()

    # Precompute all swizzle offsets before the loop
    var phys_offsets = StaticTuple[Int, total_groups]()

    comptime for i in range(total_groups):
        comptime chunk_idx, group_idx = divmod(i, groups_per_chunk)
        comptime col_offset = chunk_idx * chunk_size + group_idx * 16
        var logical_elem = row_start * row_size + col_start + col_offset
        phys_offsets[i] = swz(logical_elem)

    var lmv = local_mem.vectorize[4]()

    comptime for chunk in range(0, num_chunks):
        comptime for g in range(0, groups_per_chunk):
            comptime vec_base = (chunk * groups_per_chunk + g) * 4
            # Process 16 FP8 elements: load 4x float32, cast to 4x FP8
            # But we need to pack 16 FP8 values into 4 uint32 registers.
            # Load 4 groups of 4 float32, cast each group to 4 FP8, pack into uint32.
            var packed = SIMD[.uint32, 4](0)
            comptime for sub in range(4):
                var vec_val = lmv[vec_base + sub]
                comptime if scale_needed:
                    vec_val *= scale
                var fp8_vec = vec_val.cast[out_dtype]()
                packed[sub] = bitcast[.uint32, 1](fp8_vec)

            st_shared_v4_b32_at_fp8_elem_off[out_dtype=out_dtype](
                shared_mem,
                phys_offsets[chunk * groups_per_chunk + g],
                packed,
            )


@inline(.always)
def st_shared_v4_b32_at_fp8_elem_off[
    out_dtype: DType
](
    dst_fp8: MutPointer[Scalar[out_dtype], MutAnyOrigin, address_space=.SHARED],
    elem_off: Int,  # FP8 element offset
    packed: SIMD[.uint32, 4],
):
    """Stores four uint32 values to shared memory at an FP8 element offset.

    Parameters:
        out_dtype: FP8 element type of the shared memory destination.

    Args:
        dst_fp8: Shared memory pointer typed to `out_dtype` elements.
        elem_off: Offset from `dst_fp8` in `out_dtype` elements to the
            first stored value.
        packed: Four uint32 registers holding the packed FP8 values to
            store.
    """
    # Delegates to the shared `st_shared_v4_b32` (moved to attention_utils);
    # `elem_off` is in fp8 elements (the pointer is fp8-typed).
    st_shared_v4_b32(dst_fp8, elem_off, packed)


@inline(.always)
def ld_shared_v4_u32[
    ordered: Bool = False
](
    src_u8: MutPointer[UInt8, _, address_space=.SHARED],
    byte_off: Int,
) -> SIMD[
    .uint32, 4
]:
    """Loads four contiguous uint32 values from shared memory at a byte offset.

    Parameters:
        ordered: Keeps the load in program order. The plain form is a pure asm
            statement, so LLVM may merge two loads of one address across the
            mbarrier wait between them, or hoist one above its wait; a ring
            slot reread at the same address once per lap is exactly that
            pattern.

    Args:
        src_u8: Byte-typed shared memory pointer to the base of the load.
        byte_off: Byte offset from `src_u8` to the first uint32 to load.
    """
    var addr = src_u8.unsafe_offset(byte_off)
    var result = inlined_assembly[
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
        # 4 outputs (return struct has 4 elems) + 1 input (addr)
        constraints=("=r,=r,=r,=r,l,~{memory}" if ordered else "=r,=r,=r,=r,l"),
        has_side_effect=ordered,
    ](addr)
    return SIMD[.uint32, 4](result[0], result[1], result[2], result[3])


@inline(.always)
def cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
    *,
    fp8_dtype: DType,
    out_dtype: DType,
](w0: UInt32, w1: UInt32,) -> SIMD[.uint32, 4]:
    """Converts eight FP8 values packed in two uint32 registers to eight BF16 values packed in four uint32 registers.

    Parameters:
        fp8_dtype: FP8 element type of the source values packed in `w0`
            and `w1`.
        out_dtype: Target element type of the converted output values
            (`bfloat16`).

    Args:
        w0: Lower uint32 register holding the first four packed FP8
            values.
        w1: Upper uint32 register holding the next four packed FP8
            values.
    """
    var u32x2: SIMD[.uint32, 2] = SIMD[.uint32, 2](w0, w1)
    var fp8x8: SIMD[fp8_dtype, 8] = bitcast[fp8_dtype, 8](u32x2)
    var bf16x8: SIMD[out_dtype, 8] = fp8x8.cast[out_dtype]()
    return bitcast[.uint32, 4](bf16x8)


@inline(.always)
def cvt_fp8x16_from_u32x4_to_bf16x16_packed_2xu32x4[
    *,
    fp8_dtype: DType,
    out_dtype: DType,
](w: SIMD[.uint32, 4]) -> StaticTuple[SIMD[.uint32, 4], 2]:
    """Converts 16 FP8 bytes (one v4.b32 load) to 16 packed BF16 values."""
    return StaticTuple[SIMD[.uint32, 4], 2](
        cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
            fp8_dtype=fp8_dtype, out_dtype=out_dtype
        ](w[0], w[1]),
        cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
            fp8_dtype=fp8_dtype, out_dtype=out_dtype
        ](w[2], w[3]),
    )


@inline(.always)
def st_shared_v4_b32_at_bf16_elem_off[
    out_dtype: DType
](
    dst_bf16: MutPointer[Scalar[out_dtype], _, address_space=.SHARED],
    elem_off: Int,  # bf16 element offset
    packed: SIMD[.uint32, 4],
):
    """Stores four uint32 values to shared memory at a BF16 element offset.

    Parameters:
        out_dtype: Element type of the shared-memory destination (`bfloat16`).

    Args:
        dst_bf16: Shared memory pointer typed to `out_dtype` elements.
        elem_off: Offset from `dst_bf16` in `out_dtype` elements to the
            first stored value.
        packed: Four uint32 registers holding the packed values to store.
    """
    # Delegates to the shared `st_shared_v4_b32` (moved to attention_utils);
    # `elem_off` is in bf16 elements (the pointer is bf16-typed).
    st_shared_v4_b32(dst_bf16, elem_off, packed)


@inline(.always)
def e8m0_to_bf16_broadcast(scale_byte: UInt8) -> UInt32:
    """Convert an e8m0 scale byte to a bf16 value broadcast into both halves of a uint32.

    e8m0 format: value = 2^(byte - 127). The bf16 representation is
    obtained by left-shifting the 8-bit exponent by 7 to place it in the
    bf16 exponent field (bits 7-14), with sign=0 and mantissa=0.
    Broadcasting into both halves of a uint32 prepares the value for
    use with the packed bf16x2 multiply instruction.

    Args:
        scale_byte: E8M0 exponent byte encoding the scale as
            `2^(byte - 127)`.
    """
    var bf16_bits = UInt16(scale_byte) << 7
    return UInt32(bf16_bits) | (UInt32(bf16_bits) << 16)


@inline(.always)
def hmul2_bf16x8_by_scalar[
    out_dtype: DType,
](packed: SIMD[.uint32, 4], scale_bf16: UInt32) -> SIMD[.uint32, 4]:
    """Multiply 8 packed bf16 values (in 4 uint32 registers) by a bf16x2 scalar broadcast.

    Parameters:
        out_dtype: Element type of the packed bf16 values (`bfloat16`).

    Args:
        packed: Four uint32 registers holding the eight packed bf16 values
            to scale.
        scale_bf16: Bf16x2 scalar broadcast in a uint32, as produced by
            `e8m0_to_bf16_broadcast`.
    """
    var res = type_of(packed)()

    comptime for i in range(packed.length):
        res[i] = inlined_assembly[
            "mul.rn.bf16x2 $0, $1, $2;",
            UInt32,
            constraints="=r,r,r",
            has_side_effect=False,
        ](packed[i], scale_bf16)
    return res


# --------------------------------------------------------------------------
# MLA decoding softmax Pipeline
# --------------------------------------------------------------------------
@inline(.always)
def clamped_index_coordinate(
    var prompt_idx: UInt32,
    var q_head_idx: UInt32,
    var q_idx_abs: UInt32,
    var col: UInt32,
    var tile_key_base: UInt32,
    var num_keys: Int,
    var cache_start_pos: UInt32,
) -> Coord[Int, Int, Int, Int]:
    """Builds a four-component index coordinate with the key index clamped to the last valid key.

    Args:
        prompt_idx: Batch index of the prompt this coordinate belongs to.
        q_head_idx: Query head index of this coordinate.
        q_idx_abs: Absolute query token index used by the mask.
        col: Column offset within the current KV tile.
        tile_key_base: Starting key index of the current KV tile in
            global KV coordinates, before `cache_start_pos` is added.
        num_keys: Total number of valid keys in the KV cache for this
            batch.
        cache_start_pos: External base offset added to key indices to
            compute absolute positions.
    """
    # Global key index (column) for this element
    var score_col: UInt32 = tile_key_base + col
    var k_idx_abs: UInt32 = score_col + cache_start_pos
    # Clamp k to last valid key so MaterializedMask never reads OOB.
    var last_k_abs: UInt32 = cache_start_pos + UInt32(max(num_keys - 1, 0))
    var k_idx_abs_safe: UInt32 = min(k_idx_abs, last_k_abs)
    return (
        Int(prompt_idx),
        Int(q_head_idx),
        Int(q_idx_abs),
        Int(k_idx_abs_safe),
    )


struct MLA_SM100_Decode_Common[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_dtype: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](TrivialRegisterPassable):
    """Provides the shared softmax, correction, and store logic for SM100 MLA decode kernels.

    Parameters:
        q_type: Element type of the Q and P operands stored in SMEM.
        KVLUTType: `MHAOperand` providing the KV cache tensor and its
            element type.
        output_dtype: Element type of the output tile written back via TMA
            (must be `bfloat16`).
        SplitAccumType: `OptionalPointer` type wrapping the split-K LSE
            accumulator buffer.
        MaskType: `MHAMask` type applied to the attention scores.
        config: Decode config supplying tile sizes, swizzle modes, and
            TMEM/SMEM layout.
        ValidLengthType: `OptionalPointer` type wrapping the per-batch
            valid-sequence-length tensor.
        _is_cache_length_accurate: When `False`, the kernel adds the local
            sequence length to the cache length to compute the total key
            count (defaults to `False`).
        ragged: When `True`, the valid-lengths tensor is interpreted as
            input row offsets enabling ragged batching (defaults to
            `False`).
    """

    comptime kv_type = Self.KVLUTType.dtype
    comptime role_local_lane = (
        Self.config.cta_group == 2
        and Self.q_type == DType.float8_e4m3fn
        and Self.kv_type == DType.float8_e4m3fn
    )
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    # 2 bytes for float16
    comptime bytes_per_element = size_of[Self.q_type]()
    # the stage element is the same for both K and V
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    # Output tile width uses BN_PV/4 (the per-warp stripe), not BN_QK.
    comptime output_tile_width = ((Self.config.BN_PV // 4) // 2) * (
        4 // size_of[Self.output_dtype]()
    )
    # O: 128 x 256
    comptime O_M = Self.config.BM * 2  # 128
    comptime O_N = Self.config.padded_depth // 2  # 256

    # S: 128 x 32
    comptime S_M = Self.config.BM * 2  # 128
    comptime S_N = Self.config.BN_QK // 2  # 32
    comptime UMMAQKTSS = DecodeSM100QKTSS[
        operand_type=Self.q_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS[
        operand_type=Self.q_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    # --------------------------------------------------------------------------
    # PDL early exit cleanup for split-K CTAs with no work.
    # Writes -inf to LSE so the combine kernel gives this split zero weight,
    # then calls barrier() + launch_dependent_grids().
    #
    # o_accum_split is deliberately left untouched: the combine kernel's
    # `select` guard (scale != 0) keeps uninitialised memory out of the
    # result when scale == 0 (i.e. LSE == -inf).
    # --------------------------------------------------------------------------
    @staticmethod
    @inline(.always)
    def pdl_early_exit[
        # When True, the fold caller owns all q_max_seq_len LSE
        # slots and must emit -inf to each.
        fold_q: Bool = False,
    ](
        split_idx: Int,
        batch_idx: Int,
        max_seq_len: Int,
        batch_size: Int,
        lse_accum_split_ptr: Self.SplitAccumType,
        # Explicit seq_idx for fold callers iterating q_local 0..q_len_fold-1.
        # Under fold the grid holds one token per batch, so the CTA's own
        # token index can't address all seq slots.
        # Only consumed when fold_q=True.
        seq_idx_fold: UInt32 = 0,
    ):
        var tid = thread_idx.x

        # -- 1. Write -inf to LSE so combine gives this split zero weight --
        # LSE layout: (num_splits, batch_size, max_seq_len, num_heads)
        # Use max_seq_len (not per-batch seq_len) for strides to match
        # the PADDED buffer layout and the combine kernel's read pattern.
        var head_start = decode_head_tile[Self.config]() * Self.config.BM
        var seq_idx: Int
        comptime if fold_q:
            seq_idx = Int(seq_idx_fold)
        else:
            seq_idx = decode_seq_idx[Self.config]()
        var stride_seq = Self.config.num_q_heads
        var stride_batch = max_seq_len * stride_seq
        var stride_split = batch_size * stride_batch
        var neg_inf_val = min_or_neg_inf[Self.AccumType]()

        # First BM threads each write one head's LSE
        if tid < Self.config.BM:
            var head_idx = head_start + tid
            if head_idx < Self.config.num_q_heads:
                var lse_offset = (
                    split_idx * stride_split
                    + batch_idx * stride_batch
                    + seq_idx * stride_seq
                    + head_idx
                )
                var lse_ptr = rebind[
                    MutPointer[Scalar[Self.AccumType], origin=MutAnyOrigin]
                ](lse_accum_split_ptr.value())
                lse_ptr[lse_offset] = neg_inf_val

        # -- 2. Barrier + PDL signal, then return --
        barrier()
        launch_dependent_grids()

    # --------------------------------------------------------------------------
    # Unsplit LSE for a query token with no keys, when the caller asked for
    # the LSE. `lse_out_ptr` is then the ragged `[total_q_tokens, num_heads]`
    # LSE output; the token's O rows are zeroed by `store_empty`.
    # --------------------------------------------------------------------------
    @staticmethod
    @inline(.always)
    def store_empty_lse(q_token_idx: Int, lse_out_ptr: Self.SplitAccumType):
        """Writes `-inf` LSE for this CTA's heads of an empty query token.

        Args:
            q_token_idx: Global index of the query token.
            lse_out_ptr: The ragged `[total_q_tokens, num_heads]` LSE output.
        """
        var tid = Int(thread_idx.x)
        var head_idx = decode_head_tile[Self.config]() * Self.config.BM + tid
        if tid < Self.config.BM and head_idx < Self.config.num_q_heads:
            var lse_ptr = rebind[
                Pointer[Scalar[Self.AccumType], origin=MutAnyOrigin]
            ](lse_out_ptr.value())
            lse_ptr[
                q_token_idx * Self.config.num_q_heads + head_idx
            ] = min_or_neg_inf[Self.AccumType]()

    # --------------------------------------------------------------------------
    # MLA decoding load_q and load_kv function
    # --------------------------------------------------------------------------
    @staticmethod
    @inline(.always)
    def load_kv(
        tma: KVTMATile[
            dtype=Self.kv_type,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
            BN=Self.config.BN_QK,  # tile_m: 64 (Layout-E / Layout-G-64) or 128 (Layout-G-128)
            BK=Self.config.input_q_depth,  # the row as stored
        ],
        smem: SharedMemPointer[Scalar[Self.kv_type]],
        mbar: MBarType,
        col_start: Int,
        row_start: Int,
    ):
        # TMA only uses .ptr from the destination — layout is irrelevant
        # (swizzle is in the TMA descriptor). Use flat row_major TileTensor.
        comptime kv_elements = Self.config.BN_QK * Self.config.input_q_depth
        comptime kv_tt_layout = tt_row_major[kv_elements]()
        var smem_tensor = TileTensor[
            Self.kv_type, type_of(kv_tt_layout), address_space=.SHARED
        ](smem, kv_tt_layout)
        tma.async_copy_3d(smem_tensor, mbar[], (col_start, 0, row_start))

    @staticmethod
    @inline(.always)
    def load_q(
        tma: QOTMATile[
            dtype=Self.q_type,
            BM=Self.config.BM,  # tile_m =64
            BK=Self.config.input_q_depth,  # the row as stored
            swizzle_mode=Self.config.swizzle_mode,
        ],
        smem: SharedMemPointer[Scalar[Self.q_type]],
        mbar: MBarType,
        col_start: Int,
        row_start: Int,
    ):
        comptime q_elements = Self.config.BM * Self.config.input_q_depth
        comptime q_tt_layout = tt_row_major[q_elements]()
        var smem_tensor = TileTensor[
            Self.q_type, type_of(q_tt_layout), address_space=.SHARED
        ](smem, q_tt_layout)

        tma.async_copy(smem_tensor, mbar[], (col_start, row_start))

    @staticmethod
    @inline(.always)
    def apply_mask[
        half_load: Int,
        NonCausalMask: Bool,
        CausalMask: Bool,
        # when > 0, causal_limit is per-row
        # and derived from (score_row // fold_q_num_heads) + cache_len + 1,
        # treating score_row as the fold tile's per-thread row index.
        fold_q_num_heads: Int = 0,
        # When > 0, additionally mask out keys that are MORE than
        # SlidingWindowSize positions before the current query (i.e. clear
        # the low bits of mask_bits below `causal_limit - SlidingWindowSize`).
        # Implies causal upper bound (CausalMask must be True).
        SlidingWindowSize: Int = 0,
        # When True, `col_base + i` (the tile-relative gather SLOT) is not a
        # logical key position -- it indexes a score-sorted sparse top-k
        # selection. Causality must instead be decided by looking up each
        # slot's LOGICAL key position via `logical_indices` and comparing it
        # to `causal_limit`, rather than by counting slots. Requires
        # CausalMask=True and SlidingWindowSize=0 (checked by the caller).
        # See Kernels/claude_kb -- sparse-mla-decode-causal-slot-bug.
        SparseCausalLogical: Bool = False,
        SparsePhysicalSentinels: Bool = False,
        # Half-row softmax only: decides the logical mask from one position
        # per key slot (`ballot_pos`, which lane j of the warp loaded a tile
        # ahead for slot j of its 2 * half_load) and a warp ballot. Needs a
        # causal horizon shared by the warp (no fold).
        logical_ballot: Bool = False,
        # The mask runs in windows of at most 32 columns (see `Softmax`):
        # `s_row[chunk_off, chunk_off + chunk)`, with every column offset
        # -- `col_base`, the gap, the window bounds -- relative to
        # `col0 + chunk_off`. A 32-bit mask word must never cover more than
        # 32 columns: a shift past the word width is LLVM poison, which
        # ptxas folds to "valid", silently unmasking the overflow columns.
        chunk_off: Int = 0,
        chunk: Int = half_load,
    ](
        tiles_done: Int,
        col0: Int,
        num_keys: Int,
        s_row: LocalTensor[Self.AccumType, row_major[half_load](), _],
        mask: Self.MaskType,
        prompt_idx: UInt32,
        q_head_idx: UInt32,
        score_row: UInt32,
        cache_len: Int,
        start_pos: UInt32,
        cache_start_pos: UInt32,
        kv_start_row: Int = 0,  # Starting KV row for split-K (0 for non-split)
        # Raw (pre-physical-remap) logical sparse indices, same
        # [total_q_tokens, indices_stride] layout as the physical `d_indices`
        # gather buffer, with -1 for invalid/padding slots. Only read when
        # SparseCausalLogical=True.
        logical_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
        physical_sentinel_indices: OptionalReg[
            MutPointer[Int32, MutAnyOrigin]
        ] = None,
        logical_indices_stride: Int = 0,
        q_token_idx: Int = 0,
        # Valid per-q_token length of `logical_indices` (the main sparse topk,
        # excluding extra_d_indices). The physical gather clamps overflow slots
        # to a duplicate in-range row, but `logical_indices` has no such row,
        # so slots >= this bound must be masked without a read or they run off
        # the end of the q_token's row.
        logical_indices_len: Int = 0,
        # Slots `[gap_lo, gap_hi)` are masked: the padding between the main
        # sparse segment and the extra-KV segment (see
        # `OffsetPosition.sparse_topk`).
        gap_lo: Int = 0,
        gap_hi: Int = 0,
        # Slots `[row_slot_limit, main_slots_hi)` are masked: the `-1`
        # padding past the row's valid length (`SparseRowLayout`), below the
        # causal horizon. Slots from `main_slots_hi` on are the extra-KV
        # segment, which the row's length does not describe.
        row_slot_limit: Int = Int.MAX,
        main_slots_hi: Int = Int.MAX,
        # This lane's prefetched position; read only under `logical_ballot`.
        ballot_pos: Int32 = -1,
    ) -> Scalar[Self.AccumType]:
        comptime assert chunk >= 1 and chunk <= 32, (
            "mask windows are chunked to 32 columns: a wider one shifts a"
            " 32-bit mask word past its width (poison), which ptxas folds to"
            " valid"
        )
        comptime assert (
            chunk_off + chunk <= half_load
        ), "mask window runs past the thread's columns"
        # Tile / column base this thread covers in num_keys in global KV cache
        # For split-K: kv_start_row + tiles_done * BN_QK gives global position
        # For non-split: kv_start_row=0, so this is just tiles_done * BN_QK
        var tile_key_base: Int = kv_start_row + tiles_done * Self.config.BN_QK
        # First column of this window; every column offset below is relative
        # to it, so a chunked call masks exactly its window.
        var col_base: Int = tile_key_base + col0 + chunk_off

        # Per-row causal masking for chunked decode
        # Allowed keys for this query row in a chunked causal decode:
        # cache_len + score_row + 1
        var causal_limit: Int

        comptime if CausalMask:
            comptime if fold_q_num_heads > 0:
                # Fold: score_row is the fold tile row index; q_local =
                # score_row // num_q_heads selects the causal horizon.
                causal_limit = (
                    cache_len + (Int(score_row) // fold_q_num_heads) + 1
                )
            else:
                causal_limit = cache_len + Int(score_row) + 1
        else:
            causal_limit = num_keys
        # The causal horizon counts keys, not slots: past the padding gap,
        # the slot sits `gap_hi - gap_lo` slots after the key it holds, so
        # extra-KV keys keep the position they had before the main segment
        # was padded. A half-tile never straddles the 64-aligned `gap_hi`.
        # A sparse horizon can also pass the last slot, whose index is a
        # clamped repeat, so the slot count bounds it as well.
        var key_base = col_base
        var keys_remaining: Int
        comptime if CausalMask:
            if gap_hi > gap_lo and col_base >= gap_hi:
                key_base = col_base - (gap_hi - gap_lo)
            keys_remaining = min(causal_limit - key_base, num_keys - col_base)
            if col_base < main_slots_hi:
                keys_remaining = min(keys_remaining, row_slot_limit - col_base)
        else:
            keys_remaining = causal_limit - col_base
        var n_valid = max(min(keys_remaining, chunk), 0)
        # Build mask_bits with lowest n_valid bits = 1. The word is 32-bit
        # and n_valid <= chunk <= 32, so the 64-bit construction never
        # shifts past a word width and every consumer shift stays below 32
        # (both are LLVM poison otherwise, which ptxas folds to "valid" and
        # the mask silently stops masking).
        var mask_bits: UInt32 = UInt32(
            (UInt64(1) << UInt64(n_valid)) - UInt64(1)
        )

        # Sliding window: also clear bits BELOW the per-row lower limit.
        # Per-row lower limit (in global KV index) = causal_limit -
        # SlidingWindowSize. Bits in mask_bits correspond to columns
        # [col_base, col_base + chunk), so bit i maps to global key
        # index `col_base + i`. Clear bits where `col_base + i <
        # per_row_lo`, i.e. `i < per_row_lo - col_base`. Clamp to
        # [0, chunk].
        comptime if SlidingWindowSize > 0:
            var per_row_lo: Int = causal_limit - SlidingWindowSize
            var n_invalid_low: Int = max(per_row_lo - key_base, 0)
            n_invalid_low = min(n_invalid_low, chunk)
            var low_mask: UInt32 = UInt32(
                (UInt64(1) << UInt64(n_invalid_low)) - UInt64(1)
            )
            mask_bits &= ~low_mask

        # Initialize the per-row running max to the finite mask sentinel so
        # an all-masked tile produces a finite `current_max` (= MASK_VALUE)
        # instead of true -inf — keeps later softmax math NaN-free.
        var current_max: Scalar[Self.AccumType] = MASK_VALUE

        comptime assert not (
            SparseCausalLogical and SlidingWindowSize > 0
        ), "SparseCausalLogical does not support sliding window."

        # Runtime null guard: callers that request the logical-position path
        # (comptime) but don't actually thread a logical index buffer (e.g.
        # direct-kernel tests, or sparse kernels not yet wired for it) fall
        # back to the prior slot-count behavior instead of a null deref.
        comptime if SparseCausalLogical:
            var use_sparse_causal_logical = Bool(logical_indices)
            comptime if logical_ballot:
                comptime assert (
                    half_load == Self.config.BN_QK >> 3
                    and fold_q_num_heads == 0
                ), "ballot logical mask needs the half-row softmax, no fold"
                if use_sparse_causal_logical:
                    var visible = ballot_pos >= 0 and ballot_pos <= Int32(
                        min(causal_limit - 1, Int(Int32.MAX))
                    )
                    # Lane j voted for slot j of the warp's 2 * half_load;
                    # this thread's half_load start at `col0 & half_load`.
                    var half_shift = col0 & half_load
                    comptime if _MLA_HALF_ROWS_BREAK == 5:
                        half_shift ^= half_load
                    mask_bits = (
                        vote[.uint32](visible) >> UInt32(half_shift)
                    ) & UInt32((1 << half_load) - 1)
            else:
                if use_sparse_causal_logical:
                    # Decide visibility for the whole group up front and hand the
                    # element loop the same `mask_bits` word every other mask
                    # produces. Deciding it per element instead put the read
                    # INSIDE the loop's branch, so each of the half_load reads
                    # paid its own memory latency. The slots are contiguous in
                    # `i`; reading them in batches lets a batch share one
                    # latency, and a short batch keeps the live set from growing
                    # by all half_load positions.
                    #
                    # Slots at or past `logical_indices_len` are the gather's
                    # clamped overflow and must be masked WITHOUT a read (see
                    # that argument's docstring). Carrying that bound on the LOOP
                    # rather than per element keeps the set of addresses read
                    # here exactly the set the per-element form read.
                    comptime logical_batch = 8
                    var n_readable = max(
                        min(logical_indices_len - col_base, chunk), 0
                    )
                    var logical_row = (
                        logical_indices.unsafe_value().unsafe_offset(
                            (q_token_idx * logical_indices_stride + col_base)
                        )
                    )
                    # `pos <= causal_limit - 1` in 32 bits is `pos <
                    # causal_limit` in Int, for every position `logical_indices`
                    # can represent: causal_limit >= 1 here (CausalMask is
                    # required), and saturating the bound keeps a horizon wider
                    # than Int32 admitting every slot, as the Int compare did.
                    var causal_last = SIMD[.int32, logical_batch](
                        Int32(min(causal_limit - 1, Int(Int32.MAX)))
                    )
                    # -1 marks a padding slot, so the low bound rejects it.
                    comptime lowest_pos = SIMD[.int32, logical_batch](0)
                    var logical_bits: UInt32 = 0
                    var slot = 0
                    while slot + logical_batch <= n_readable:
                        var pos = logical_row.unsafe_load[
                            width=logical_batch, alignment=4
                        ](slot)
                        var visible = pos.ge(lowest_pos) & pos.le(causal_last)
                        var batch_bits: UInt32 = 0
                        comptime for j in range(logical_batch):
                            if visible[j]:
                                batch_bits |= UInt32(1) << UInt32(j)
                        logical_bits |= batch_bits << UInt32(slot)
                        slot += logical_batch
                    while slot < n_readable:
                        var pos = logical_row[slot]
                        if pos >= 0 and pos <= causal_last[0]:
                            logical_bits |= UInt32(1) << UInt32(slot)
                        slot += 1
                    mask_bits = logical_bits

        # A zero-filled gathered row is still a softmax slot. Exclude its
        # padding index without interpreting physical row IDs as positions.
        comptime if SparsePhysicalSentinels:
            if physical_sentinel_indices:
                var readable = max(
                    min(logical_indices_len - col_base, chunk), 0
                )
                var raw_row = (
                    physical_sentinel_indices.unsafe_value().unsafe_offset(
                        (q_token_idx * logical_indices_stride + col_base)
                    )
                )
                comptime use_ballot = (
                    _MLA_SPARSE_SENTINEL_BALLOT != 0
                    and not Self.config.decode_layout_g
                )
                comptime if use_ballot:
                    comptime assert fold_q_num_heads == 0
                var valid_bits = _sparse_physical_sentinel_bits[
                    use_ballot, role_local_lane=Self.role_local_lane
                ](raw_row, readable)
                mask_bits &= valid_bits

        if gap_hi > gap_lo:
            var g0 = min(max(gap_lo - col_base, 0), chunk)
            var g1 = min(max(gap_hi - col_base, 0), chunk)
            var gap_bits: UInt32 = UInt32(
                ((UInt64(1) << UInt64(g1)) - UInt64(1))
                ^ ((UInt64(1) << UInt64(g0)) - UInt64(1))
            )
            mask_bits &= ~gap_bits

        comptime for i in range(0, chunk):
            # rank1-style mask_r2p: turn bit into predicate and use it to select
            var bit: UInt32 = (mask_bits >> UInt32(i)) & UInt32(1)
            var in_bound: Bool = bit != UInt32(0)
            # masked_val = s_row[i]      if in_bound
            #            = MASK_VALUE    otherwise (finite sentinel; see
            #            module-level comment on MASK_VALUE for why)
            var val: Scalar[Self.AccumType] = s_row[chunk_off + i][0]
            var masked_val: Scalar[
                Self.AccumType
            ] = val if in_bound else MASK_VALUE

            comptime if NonCausalMask:
                var v: SIMD[Self.AccumType, 1] = masked_val
                var coord = clamped_index_coordinate(
                    prompt_idx,
                    q_head_idx,
                    score_row + start_pos + cache_start_pos,
                    UInt32(col0 + chunk_off + i),
                    UInt32(tile_key_base),
                    num_keys,
                    cache_start_pos,
                )
                v = mask.mask(coord, v)
                masked_val = v[0]

            s_row[chunk_off + i][0] = masked_val
            current_max = max(current_max, masked_val)

        return current_max

    @staticmethod
    @inline(.always)
    def Softmax[
        native_fp8: Bool = False,
        mask_physical_sentinels: Bool = False,
        # P in its own ring, as under `native_fp8`, but as `q_type` (BF16)
        # and unscaled.
        bf16_p_ring: Bool = False,
        num_sp_stages: Int = 2,
        # Arrivals per S-free / P-full barrier phase: one per softmax
        # thread, or -- under the CTA pair -- one elected lane per warp
        # per CTA (2 * 4).
        num_barrier_warps: Int = WARPGROUP_SIZE,
        fp8_p_stage_stride: Int = 0,
        has_per_token_scales: Bool = False,
        has_attn_sink: Bool = False,
        _op_sparse: Bool = False,
        _op_has_extra_kv: Bool = False,
        _OpTopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
        # When True, per-row score_row decomposes via integer
        # division by num_q_heads to select the causal horizon per q_token.
        fold_q: Bool = False,
        # comptime number of q_tokens packed
        # into BM=64 under fold_q=True. Needed for the ragged LSE -inf fill
        # loop. Only consumed inside `comptime if fold_q` branches.
        q_len_fold: Int = 1,
        # How the sparse producer packs each row; bounds every row by its
        # valid length (see `SparseRowLayout`). Not applied under fold_q,
        # whose rows share one list.
        sparse_row: SparseRowLayout = SparseRowLayout(),
        # Shared O: barriers from pipeline 0's P ring to pipeline 1's. The
        # other pipeline's P-free barriers are its PV-done.
        corr_smem: Bool = False,
    ](
        tmem_addr: UInt32,
        s_bars: DecodeSM100MiscMBars[
            num_stages=num_sp_stages,
            num_producer=1,
            num_consumer=num_barrier_warps,
        ],
        p_bars: DecodeSM100MiscMBars[
            num_stages=num_sp_stages,
            num_producer=num_barrier_warps,
            num_consumer=1,
        ],
        p_smem_ptr: SharedMemPointer[Scalar[Self.q_type]],
        max_smem: SharedMemPointer[
            Scalar[Self.AccumType]
        ],  # 256x1 double-buffered
        li_smem: SharedMemPointer[Scalar[Self.AccumType]],  # 128x1 buffer
        out_smem: SharedMemPointer[Scalar[Self.output_dtype]],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        out_pipeline: OutPipeline[
            num_out_stages=DecodeOutProducer[
                Self.output_dtype, Self.config
            ].num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
            cta_group=Self.config.cta_group,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            _op_sparse,
            _op_has_extra_kv,
            _OpTopkLengthsPtrType,
        ],
        scale: Float32,
        mask: Self.MaskType,
        prompt_idx: UInt32,  # batch index
        lse_accum_split_ptr: Self.SplitAccumType,
        batch_size: Int,
        scale_k_smem: OptionalReg[SharedMemPointer[Float32]] = None,
        q_scale_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]] = None,
        attn_sink_log2: Float32 = Float32(min_or_neg_inf[.float32]()),
        # Logical sparse indices and their valid per-q_token length, forwarded
        # to the inner apply_mask (documented there); required non-null when
        # SparseCausalLogical is derived below.
        logical_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]] = None,
        physical_sentinel_indices: OptionalReg[
            MutPointer[Int32, MutAnyOrigin]
        ] = None,
        logical_indices_stride: Int = 0,
        logical_indices_len: Int = 0,
        # Positions between the token the sparse row was computed for and
        # this one, for a selection reused at a later position.
        selection_lag: Int = 0,
        corr_scale_smem: OptionalReg[
            SharedMemPointer[Scalar[Self.AccumType]]
        ] = None,
    ):
        comptime MaskName: String = Self.MaskType.name()
        comptime MaskTypeName: String = Self.MaskType.get_type_name()
        comptime assert Self.AccumType.is_floating_point()
        # With per-token KV scales, P is multiplied by the KV scale before the
        # fp8 conversion, so its range is not bounded by 1 and it is not scaled.
        comptime scale_fp8_p = native_fp8 and not has_per_token_scales
        comptime p_bias = fp8_p_bias[scale_fp8_p](
            Self.config.skip_correction_threshold
        ).cast[Self.AccumType]()

        comptime NoMask: Bool = (MaskName == "NullMask")
        comptime CausalMask: Bool = (MaskName == "CausalMask")
        # Layout G (MMA_M=32): each warp owns BN_QK/4 key columns of all 32
        # rows, so the row max and row sum are 4-way exchanges and each warp
        # holds one 64-column depth band of O.
        comptime layout_g = Self.config.decode_layout_g
        comptime assert not layout_g or (
            (native_fp8 or bf16_p_ring)
            and not has_per_token_scales
            and fp8_p_stage_stride == 0
        ), "Layout G softmax supports a P ring without per-token scales."
        # Sparse gather slots are ordered by indexer SCORE, not by logical
        # key position, so the fast slot-count causal mask (correct for
        # dense decode, where col_base IS the logical key index) is wrong
        # once topk selects a genuine subset of the causal prefix. Gated to
        # CausalMask only: sliding window is not supported on the sparse
        # kernels (enforced by their own comptime asserts).
        comptime SparseCausalLogical: Bool = _op_sparse and CausalMask
        # Sliding window: SlidingWindowCausalMask is causal + lower bound at
        # `causal_limit - window_size`. Detected via get_type_name (since
        # name() embeds the window value, e.g. "SlidingWindowCausalMask[64]").
        comptime SlidingWindowMask: Bool = (
            MaskTypeName == "SlidingWindowCausalMask"
        )
        # Window size: 0 if not sliding. The `MHAMask.sliding_window_size()`
        # trait method is the canonical channel for this — `Self.MaskType.
        # window_size` would fail type-checking inside the comptime if guard
        # because it's a struct parameter, not exposed on the trait.
        comptime _sliding_window_size: Int = Self.MaskType.sliding_window_size()
        # At most 16 rows: two threads per row, each loading 8 of its warp's
        # 16 key columns through the 16-lane TMEM shape (see `hr_row`).
        comptime half_rows = layout_g and Self.config.half_row_softmax(
            Self.config.num_q_heads * (q_len_fold if fold_q else 1)
        )
        comptime if half_rows:
            comptime assert (
                Self.config.BN_QK == 64
                and Self.config.BN_PV // 4 == 64
                and not SlidingWindowMask
            ), "half-row softmax needs BN_QK of 64, no sliding window"
        # The logical causal mask decided by a warp ballot: the rows share
        # one causal horizon only without the fold.
        comptime ballot_logical = (
            half_rows and SparseCausalLogical and not fold_q
        )

        # Same S base / stride as in mma()
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var s_stride = UInt32(comptime (Self.config.s_slot_stride()))
        # The pair's S slot spans both halves' 64 columns per lane, so each
        # softmax thread covers one whole key half and the S release
        # reaches the leader from both CTAs.
        comptime is_cg2 = Self.config.cta_group == 2
        # The pair's softmax sets its period, and on a fully visible tile
        # most of its issue is the mask. A tile whose columns are all
        # visible (a per-thread 32-bit bound, set before the walk) takes a
        # path with no mask, its max taken over the raw scores in eight
        # independent chains. Exact: the scale is positive, rounding is
        # monotone and max is associative, so P is bit for bit the same.
        # The bound is also sound under `mask_physical_sentinels`, but only
        # where the mask supplies the row's live count: a causal row packs its
        # live entries first, so its count is `min(top_k, causal count)` and
        # that count is a term of the bound -- a bound-admitted tile cannot
        # hold a -1 (see the module-level `SparseRowLayout` note). A null
        # mask's bound is the caller's stated length, which the kernel cannot
        # compare with the row, so a row that states more entries than it
        # fills would score its `-1` padding as keys. Those forms keep the
        # gathered-row check; `MLA_SMX_NULL_MASK_FAST=1` is the planted fault
        # that shows it.
        # This gate only keeps the measured per-form set of masked-arm
        # launches: `MLA_SMX_SENTINEL_FAST=0` restores their generic mask.
        comptime masked_fast_form = (
            mask_physical_sentinels
            and _MLA_SMX_SENTINEL_FAST != 0
            and (
                # A 128-key pair tile (64 columns per softmax thread) takes
                # the same body: the bound covers all 64 of them, and the only
                # reader of a 32-bit ballot word is the opt-in trap check,
                # which is two words wide at this width.
                (Self.config.BN_QK >> 1) <= 32
                or (Self.config.BN_QK == 128 and Self.config.cta_group == 2)
            )
        )
        # A form whose softmax threads own a narrower window than the warp's
        # 32-slot ballot word (Layout G's 16, the half-row softmax's 8) tests
        # only its own bits when `MLA_SMX_WINDOW_SENTINEL` is on; that matters
        # to the trap's check alone, which runs a tile ahead of its use.
        comptime windowed = half_rows or layout_g
        comptime sentinel_check = _MLA_SMX_SENTINEL_TRAP != 0 and (
            not windowed or _MLA_SMX_WINDOW_SENTINEL != 0
        )
        # Which unpaired forms take the mask-free max, plain arm and masked
        # arm separately, because the two arms set the softmax's period:
        # `_MLA_SMX_UNPAIRED_HEADS`/`_MASK_HEADS` (and the `_FP8_` pair) carry
        # the measured sets. Both cache formats that reach an unpaired form
        # take it: the same tile and the same bound make the mask pure
        # overhead on either.
        # `MLA_SMX_UNPAIRED=1` opens the path on every form for a sweep.
        comptime unpaired_form = (
            Self.kv_type == DType.uint8
            or Self.kv_type == DType.float8_e4m3fn
            or _MLA_SMX_UNPAIRED == 1
        ) and (
            (NoMask or CausalMask)
            and not SlidingWindowMask
            and not fold_q
            and not has_per_token_scales
        )
        comptime smx_head_bit = _mla_smx_head_bit[Self.config.num_q_heads]()
        comptime fp8_cache = Self.kv_type == DType.float8_e4m3fn
        comptime plain_heads = (
            _MLA_SMX_UNPAIRED_FP8_HEADS if fp8_cache else _MLA_SMX_UNPAIRED_HEADS
        )
        comptime mask_heads = (
            _MLA_SMX_UNPAIRED_FP8_MASK_HEADS if fp8_cache else _MLA_SMX_UNPAIRED_MASK_HEADS
        )
        comptime unpaired_plain = unpaired_form and (
            _MLA_SMX_UNPAIRED == 1
            or (
                _MLA_SMX_UNPAIRED < 0
                and smx_head_bit >= 0
                and (plain_heads & (1 << smx_head_bit)) != 0
            )
        )
        comptime unpaired_mask = unpaired_form and (
            _MLA_SMX_UNPAIRED == 1
            or (
                _MLA_SMX_UNPAIRED < 0
                and smx_head_bit >= 0
                and (mask_heads & (1 << smx_head_bit)) != 0
            )
        )
        comptime unpaired_fast = (
            unpaired_plain if not mask_physical_sentinels else unpaired_mask
        )
        # Every masked-arm gate above only picks the measured per-form set; the
        # mask itself has to supply the row's live count, which is what the
        # bound-only body needs. See `_MLA_SMX_NULL_MASK_FAST`.
        comptime smx_fast = (
            (is_cg2 and (not mask_physical_sentinels or masked_fast_form))
            or (not is_cg2 and unpaired_fast)
        ) and (
            not mask_physical_sentinels
            or CausalMask
            or _MLA_SMX_NULL_MASK_FAST != 0
        )
        comptime if smx_fast:
            comptime assert (
                (NoMask or CausalMask)
                and not SlidingWindowMask
                and not fold_q
                and not has_per_token_scales
            ), "the softmax fast path needs a plain causal or null mask"
        # Double-buffered max SMEM: two 128-element buffers to eliminate the
        # race between the read at `lane_id ^ 64` and the next iteration's
        # write. Consecutive iterations alternate buffers so no extra barrier
        # is needed between the read and the following write.
        # Buffer selection uses branchless pointer arithmetic:
        #   buf_offset = (tiles_done & 1) * WARPGROUP_SIZE
        # yielding 0 for even iterations and WARPGROUP_SIZE for odd ones.
        comptime smem_1d_layout = tt_row_major[WARPGROUP_SIZE]()
        var li_Smem_Tensor = TileTensor[
            Self.AccumType, type_of(smem_1d_layout), address_space=.SHARED
        ](li_smem, smem_1d_layout)

        var corr_scale_tmem = tmem_addr + UInt32(Self.config.tmem_corr_scale)
        # For split-K: use num_keys_this_split for loop bounds
        # but keep num_keys (total) for masking with global KV positions
        var num_keys = offset_position.num_keys  # Total keys for masking
        var num_keys_this_split = (
            offset_position.num_keys_this_split
        )  # Keys for this split
        var kv_start_row = (
            offset_position.kv_start_row
        )  # Starting KV position for this split
        var cache_start_pos: UInt32 = 0
        # Logical cache length: equals cache_len() in dense mode and
        # whenever sparse topk >= actual_tokens; diverges only when sparse
        # topk genuinely subsets the causal prefix, which is exactly when
        # the causal horizon must be computed against logical (not slot)
        # positions. See cache_len_logical() docstring.
        var cache_len: Int = offset_position.cache_len_logical()
        var start_pos: UInt32 = offset_position.start_pos(cache_start_pos)
        var gap_lo: Int = 0
        var gap_hi: Int = 0
        comptime if _op_sparse and _op_has_extra_kv:
            gap_lo = offset_position.sparse_topk
            gap_hi = align_up(gap_lo, Self.config.BN_QK)

        # S consumer / P producer (N-stage wrappers, works for any num_sp_stages)
        var s_cons = DecodeSConsumerN[num_sp_stages](s_bars.consumer())
        var p_prod = DecodePProducerN[num_sp_stages](p_bars.producer())
        var c_prod = DecodeCProducer(c_bars.producer())
        var warp_idx = warp_id[broadcast=True]()
        # 0..127 inside the softmax WG. Re-derived so it does not share the
        # entry-hoisted `tid & 31` that other roles keep live in a stack slot.
        var lane_id: Int
        comptime if Self.role_local_lane:
            lane_id = Int(llvm_opaque_tid())
        else:
            lane_id = Int(opaque_nonneg_i32(Int32(thread_idx.x)))
        # Lane mapping inside the softmax warpgroup
        var row: Int
        # Column range this thread owns in P
        var col0: Int
        # This warp's TMEM lane quadrant, which the 16-lane shapes address.
        var quad_base: UInt32 = 0
        comptime if half_rows:
            var t = lane_id & (WARP_SIZE - 1)
            row = hr_row(t)
            col0 = (lane_id >> 5) * (Self.config.BN_QK >> 2) + hr_half(t) * (
                Self.config.BN_QK >> 3
            )
            # From the broadcast warp id, so the TMEM addresses stay uniform.
            quad_base = UInt32((WARP_SIZE * (warp_idx & 3)) << 16)
        elif layout_g:
            row = lane_id & 0x1F  # 0..31
            col0 = (lane_id >> 5) * (Self.config.BN_QK >> 2)  # 0, 16, 32, 48
        else:
            row = lane_id & 0x3F  # 0..63
            var half: Int = lane_id >> 6  # 0 or 1
            col0 = half * Self.config.BN_QK >> 1  # 0 or 32

        var q_head_idx: UInt32 = UInt32(
            decode_head_tile[Self.config]()
        ) * UInt32(Self.config.BM) + UInt32(row)
        # Per-row score_row under fold BM=64 packs q_len_fold *
        # num_q_heads rows; `row` identifies (q_local, head_local) and
        # apply_mask derives the causal horizon via row // num_q_heads.
        # Non-fold: one q-token per CTA.
        var score_row: UInt32
        comptime if fold_q:
            score_row = UInt32(row)
        else:
            score_row = UInt32(decode_seq_idx[Self.config]())

        var row_slot_limit: Int = Int.MAX
        var main_slots_hi: Int = Int.MAX
        comptime if _op_sparse and CausalMask and not fold_q:
            row_slot_limit = sparse_row.row_limit(
                cache_len + Int(score_row) + 1, selection_lag
            )
            comptime if _op_has_extra_kv:
                main_slots_hi = gap_hi

        var mi: Scalar[Self.AccumType] = min_or_neg_inf[Self.AccumType]()
        var li: Scalar[Self.AccumType] = 0.0
        comptime log2e_f32 = Scalar[Self.AccumType](log2e)
        comptime half_load = (
            Self.config.BN_QK
            >> 3 if half_rows else (
                Self.config.BN_QK >> 2 if layout_g else Self.config.BN_QK >> 1
            )
        )
        # ------------------------------------------------------------------
        # Fold sigma_Q (per-query-token scale) into scale_log2e.
        #
        # sigma_Q is per-token (varies by Q sequence position), but all
        # BM=64 rows in this CTA are different heads of the SAME Q token,
        # so sigma_Q is constant for the entire CTA. We fold it into
        # scale_log2e once (not per-KV-tile) so softmax scaling becomes:
        #   score * (scale * sigma_Q) * log2e
        #
        # The per-KV-token sigma_KV[t] is applied separately per column
        # (Place 1 for QK dequant, Place 2 for PV pre-fuse) and is
        # unchanged by this folding.
        # ------------------------------------------------------------------
        var effective_scale = scale
        comptime if has_per_token_scales:
            var _q_token_idx = offset_position.q_token_idx
            effective_scale = scale * q_scale_ptr.unsafe_value()[_q_token_idx]
        var scale_log2e = effective_scale.cast[Self.AccumType]()

        var tiles_done: Int = 0
        # Use num_keys_this_split for loop bounds (each split processes its portion)
        var num_k_tiles = ceildiv(num_keys_this_split, Self.config.BN_QK)
        # Sliding-window leading-tile skip + empty guard (comptime-gated;
        # entire block compiles away for non-sliding masks). MUST match the
        # producer (load/mmaQK/mmaPV) skip exactly so barrier counts agree.
        comptime if SlidingWindowMask:
            var _W_sw: Int = _sliding_window_size
            var _global_lo_sw = max(cache_len + 1 - _W_sw, 0)
            var _local_lo_sw = max(_global_lo_sw - kv_start_row, 0)
            var _tile_skip_sw = _local_lo_sw // Self.config.BN_QK
            # tiles_done starts at the skip count so apply_mask sees the
            # correct global key position via `kv_start_row + tiles_done * BN_QK`.
            tiles_done = _tile_skip_sw
            # Empty-split guard: split lies entirely below the window.
            if _tile_skip_sw >= num_k_tiles:
                num_k_tiles = _tile_skip_sw  # loop condition false
        # Index of the FIRST tile processed by this Softmax invocation.
        # Used to skip the c_prod.commit() on the very first tile (no prior
        # O accumulator to correct). For non-sliding masks this is 0,
        # recovering the original `tiles_done > 0` semantics. For sliding
        # window it equals `tiles_done`'s initial value above.
        var first_processed_tile_sw: Int = tiles_done
        comptime exch_bar_id: Int32 = 2
        comptime if half_rows:
            # Rows 16-31 of P are never written per tile; zero them once so
            # O rows 16-31 stay finite. Ordered for the first PV by the
            # first tile's proxy fence and P commit.
            comptime p_stage_bytes = Self.BlockElems * (
                size_of[Self.q_type]() if bf16_p_ring else 1
            )
            comptime chunks = num_sp_stages * (p_stage_bytes // 32)
            var p8 = p_smem_ptr.unsafe_bitcast[Scalar[DType.float8_e4m3fn]]()
            comptime for i in range(ceildiv(chunks, WARPGROUP_SIZE)):
                var idx = i * WARPGROUP_SIZE + lane_id
                var stage, chunk = divmod(idx, p_stage_bytes // 32)
                var off = (
                    stage * p_stage_bytes + p_stage_bytes // 2 + chunk * 16
                )
                comptime if (i + 1) * WARPGROUP_SIZE <= chunks:
                    st_shared_v4_b32_at_fp8_elem_off[
                        out_dtype=DType.float8_e4m3fn
                    ](p8, off, SIMD[.uint32, 4](0))
                else:
                    if idx < chunks:
                        st_shared_v4_b32_at_fp8_elem_off[
                            out_dtype=DType.float8_e4m3fn
                        ](p8, off, SIMD[.uint32, 4](0))
        # Ballot logical mask, one pipeline tile ahead: this lane's logical
        # position for the next tile, carried across the iteration.
        var hr_pos_next: Int32 = -1
        comptime if ballot_logical:
            hr_pos_next = Self._hr_logical_pos(
                tiles_done,
                num_k_tiles,
                kv_start_row,
                lane_id,
                logical_indices,
                logical_indices_stride,
                offset_position.q_token_idx,
                logical_indices_len,
            )
        # A tile takes the fast path when `tiles_done * BN_QK` is at most
        # this bound: every column of this thread's half then lies below
        # each limit `apply_mask` applies, including the row's packed live
        # count (`row_slot_limit`), so the mask is the identity. Under the
        # logical sparse mask a slot's visibility is a lookup, so no tile
        # qualifies.
        var smx_fast_lim: Int32 = -1
        # Slots of this split that every limit admits. Thread-independent, so
        # it also sets the split loop's prefix length.
        var smx_fast_end: Int = 0
        comptime if smx_fast:
            var logical_ok = True
            comptime if SparseCausalLogical:
                logical_ok = not Bool(logical_indices)
            if logical_ok:
                var end = num_keys
                comptime if CausalMask:
                    end = min(end, cache_len + Int(score_row) + 1)
                    end = min(end, row_slot_limit)
                if gap_hi > gap_lo:
                    end = min(end, gap_lo)
                smx_fast_end = end
                smx_fast_lim = Int32(
                    min(
                        max(
                            end
                            - kv_start_row
                            - col0
                            - half_load
                            + _MLA_SMX_BOUND_SLACK,
                            -1,
                        ),
                        Int(Int32.MAX),
                    )
                )
        # Trap state: this lane's slot of the gathered row, kept one tile ahead
        # so the load's latency lands in the previous tile's work rather than on
        # the tile that tests it. Compiled only under `MLA_SPARSE_SENTINEL_TRAP`.
        # A windowed form's ballot word starts at the thread's `col0` rounded
        # down to its 32 slots; everywhere else the thread's window starts the
        # word, so the base is `col0`.
        comptime sent_slots = 32 if windowed else half_load
        var fast_idx_lane: Int32 = -1
        var fast_idx_lane_hi: Int32 = -1
        comptime if sentinel_check:
            if Bool(physical_sentinel_indices) and num_k_tiles > tiles_done:
                var first_col_base = kv_start_row + col0
                comptime if windowed:
                    first_col_base = kv_start_row + (col0 & ~31)
                var first_readable = max(
                    min(logical_indices_len - first_col_base, sent_slots), 0
                )
                if first_readable > 0:
                    var first_row = (
                        physical_sentinel_indices.unsafe_value().unsafe_offset(
                            (
                                offset_position.q_token_idx
                                * logical_indices_stride
                                + first_col_base
                            )
                        )
                    )
                    fast_idx_lane = _sparse_physical_sentinel_lane[
                        Self.role_local_lane
                    ](first_row, first_readable)
                    comptime if half_load > 32:
                        fast_idx_lane_hi = _sparse_physical_sentinel_lane_hi[
                            Self.role_local_lane
                        ](first_row, first_readable)
        var _sb = stamp_begin[0]()
        # FA4's split loop: the prefix lies wholly inside the bound, so it runs
        # the mask-free body with no per-tile test and only the tail is tested.
        # `first_fast_tiles` is uniform across the softmax's threads (it comes
        # from `smx_fast_end`, which has no `col0` in it), so both phases keep
        # every thread on the same barrier schedule.
        var first_fast_tiles: Int = 0
        comptime split_loop = smx_fast and (
            _MLA_SMX_SPLIT_LOOP > 0
            or (
                _MLA_SMX_SPLIT_LOOP < 0
                and (
                    (is_cg2 and mask_physical_sentinels)
                    or (
                        not is_cg2
                        and Self.kv_type == DType.uint8
                        and smx_head_bit >= 0
                        and (_MLA_SMX_SPLIT_HEADS & (1 << smx_head_bit)) != 0
                    )
                )
            )
        )
        comptime if split_loop:
            # Clamped to the split's own walk: an empty or short split has
            # fewer tiles than the row's bound describes, and the prefix must
            # never step past `num_k_tiles` (its S slot would never arrive).
            first_fast_tiles = min(
                (max(smx_fast_end - kv_start_row, 0) + _MLA_SMX_BOUND_SLACK)
                // Self.config.BN_QK,
                num_k_tiles,
            )
        comptime for phase in range(2 if split_loop else 1):
            comptime fast_phase = (split_loop and phase == 0)
            var phase_end = first_fast_tiles if fast_phase else num_k_tiles
            while tiles_done < phase_end:
                var hr_pos: Int32 = -1
                comptime if ballot_logical:
                    hr_pos = hr_pos_next
                    hr_pos_next = Self._hr_logical_pos(
                        tiles_done + 1,
                        num_k_tiles,
                        kv_start_row,
                        lane_id,
                        logical_indices,
                        logical_indices_stride,
                        offset_position.q_token_idx,
                        logical_indices_len,
                    )
                # Wait for an S slot to become ready
                var slot_idx: UInt32 = s_cons.wait()
                stamp_tile[0](_sb, tiles_done, 0)
                var s_tmem_slot = s0_tmem + slot_idx * s_stride

                tcgen05_fence_after()

                # Each thread reads one full 32-element row (128 rows x 32 columns)
                var s_row = stack_allocation[
                    dtype=Self.AccumType, address_space=.LOCAL
                ](row_major[half_load]())
                # The S read and the mask run in windows of at most 32 columns
                # (`softmax_warp`'s `load_mask_max_impl` batching): a 64-column
                # pair row is two 32-column loads, each masked with its own
                # 32-bit word, so no mask shift ever crosses a word width (a
                # shift past the width is LLVM poison, which ptxas folds to
                # "valid"). Both loads are issued before the single
                # `tcgen05_load_wait`, so the second overlaps the first's
                # latency, and a trailing window may extend past the row; the
                # mask clamps the excess. It also halves the load's live
                # registers: one 32-value chunk at a time beside the row.
                comptime s_chunk = min(half_load, 32)
                comptime if half_rows:
                    var s_hr = tcgen05_ld_16x32bx2[half_load, imm=half_load](
                        s_tmem_slot + quad_base
                    )
                    comptime for _i in range(half_load):
                        s_row.raw_store(
                            _i, rebind[Scalar[Self.AccumType]](s_hr[_i])
                        )
                else:
                    comptime for c in range(ceildiv(half_load, s_chunk)):
                        comptime c_off = c * s_chunk
                        comptime c_len = min(32, half_load - c_off)
                        var s_row_val = tcgen05_ld[
                            datapaths=32,
                            bits=32,
                            repeat=c_len,
                            dtype=Self.AccumType,
                            pack=False,
                        ](s_tmem_slot + UInt32(c_off))

                        comptime for _i in range(type_of(s_row_val).length):
                            s_row[c_off + _i] = s_row_val[_i]
                tcgen05_load_wait()

                comptime if is_cg2:
                    # The read is warp-collective, so one elected lane per warp
                    # can release. The release must reach the leader's S-free
                    # barrier from both CTAs (count: one per warp per Cta).
                    var _s_elect = elect()
                    if _s_elect != 0:
                        s_cons.pipe.consumer_mbar()[].arrive_cluster(
                            UInt32(0), UInt32(1)
                        )
                    s_cons.pipe.step()
                else:
                    s_cons.release()

                # ------------------------------------------------------------------
                # Per-token KV scale: load into registers ONCE per tile.
                #
                # When has_per_token_scales is True, each KV token t has a float32
                # scale sigma_KV[t] stored in scale SMEM. We need these scales in
                # TWO places: (1) QK dequant and (2) PV pre-fuse. To avoid
                # reading SMEM twice (2 x 32 x 4 = 256 bytes per place), we cache
                # all 32 sigma_KV values for this thread's columns into registers
                # ONCE here, then reuse them in both places.
                # ------------------------------------------------------------------
                # Register-cached per-token scales for this tile.
                # Declared outside the comptime if so it's in scope for Place 2.
                var _sigma_kv_regs = stack_allocation[
                    dtype=Self.AccumType, address_space=.LOCAL
                ](row_major[half_load]())
                comptime if has_per_token_scales:
                    # Compute the scale SMEM pointer for this pipeline stage.
                    # per_token_scales_per_stage bytes = BN_QK * 1 * sizeof(f32) = 256
                    # In float32 elements per stage: BN_QK = 64.
                    comptime _scale_elems_per_stage = Self.config.BN_QK
                    var _scale_stage_ptr = (
                        scale_k_smem.unsafe_value().unsafe_offset(
                            slot_idx * UInt32(_scale_elems_per_stage)
                        )
                    )
                    # Load all 32 sigma_KV values for this thread's columns into
                    # registers ONCE. This is the ONLY SMEM read for scales in the
                    # entire tile processing.
                    # The last k-tile TMA may load OOB scale slots that
                    # contain uninitialized NaN. After softmax, P=0 for those
                    # columns, but 0*NaN=NaN poisons the PV MMA output. Clamping
                    # via max(sigma, 0) maps NaN→0 per PTX semantics (max(NaN,0)=0)
                    # and is a no-op for valid positive scales.
                    comptime for _j in range(half_load):
                        _sigma_kv_regs[_j] = max(
                            rebind[Scalar[Self.AccumType]](
                                _scale_stage_ptr[col0 + _j]
                            ),
                            Scalar[Self.AccumType](0),
                        )

                    # Place 1: QK dequant — multiply each score column by its
                    # token's sigma_KV[t] BEFORE the scale_log2e multiplication.
                    # Uses register-cached scales (no SMEM read).
                    # Fused with scale_log2e below when per-token scales active.

                var s_row_val_vectorized = s_row.vectorize[2]()
                comptime vs_count = ceildiv(half_load, 2)
                var current_max: Scalar[Self.AccumType] = MASK_VALUE
                var smx_fast_tile = False
                comptime if smx_fast:
                    comptime if fast_phase:
                        smx_fast_tile = True
                    else:
                        smx_fast_tile = (
                            Int32(tiles_done * Self.config.BN_QK)
                            <= smx_fast_lim
                        )
                    comptime if sentinel_check:
                        if Bool(physical_sentinel_indices):
                            # Same column and read bound as `apply_mask`'s
                            # `SparsePhysicalSentinels` block.
                            var fast_col_base = (
                                kv_start_row
                                + tiles_done * Self.config.BN_QK
                                + col0
                            )
                            comptime if windowed:
                                fast_col_base = (
                                    kv_start_row
                                    + tiles_done * Self.config.BN_QK
                                    + (col0 & ~31)
                                )
                            var fast_live: Bool
                            comptime if half_load > 32:
                                fast_live = _sparse_physical_sentinel_live_pair(
                                    fast_idx_lane, fast_idx_lane_hi
                                )
                            elif windowed:
                                fast_live = (
                                    _sparse_physical_sentinel_live_window[
                                        half_load
                                    ](
                                        fast_idx_lane,
                                        max(
                                            min(
                                                logical_indices_len
                                                - fast_col_base,
                                                sent_slots,
                                            ),
                                            0,
                                        ),
                                        col0 & 31,
                                    )
                                )
                            else:
                                fast_live = _sparse_physical_sentinel_live_word[
                                    half_load
                                ](
                                    fast_idx_lane,
                                    max(
                                        min(
                                            logical_indices_len - fast_col_base,
                                            sent_slots,
                                        ),
                                        0,
                                    ),
                                )
                            # The check reads the main segment's row only, so a
                            # window in the extra-KV segment is past its readable
                            # length: it says nothing about a -1 and is skipped.
                            # Within the row, the bound admitted a window the
                            # check rejects, i.e. the row holds a -1 the mask
                            # would have cut, so the mask-free body would read
                            # padding as a key. Poison the tile's max so a numeric
                            # gate fails as well.
                            var window_readable = max(
                                min(
                                    logical_indices_len - fast_col_base,
                                    sent_slots,
                                ),
                                0,
                            )
                            var rejected = (
                                window_readable >= sent_slots
                                and smx_fast_tile
                                and not fast_live
                            )
                            if rejected:
                                s_row[0][0] = -min_or_neg_inf[Self.AccumType]()
                            debug_assert(
                                not rejected,
                                "sentinel trap: -1 inside the bound",
                            )
                            fast_idx_lane = Int32(-1)
                            fast_idx_lane_hi = Int32(-1)
                            var next_col_base = (
                                fast_col_base + Self.config.BN_QK
                            )
                            if tiles_done + 1 < num_k_tiles:
                                var next_readable = max(
                                    min(
                                        logical_indices_len - next_col_base,
                                        sent_slots,
                                    ),
                                    0,
                                )
                                if next_readable > 0:
                                    var next_row = physical_sentinel_indices.unsafe_value().unsafe_offset(
                                        (
                                            offset_position.q_token_idx
                                            * logical_indices_stride
                                            + next_col_base
                                        )
                                    )
                                    fast_idx_lane = (
                                        _sparse_physical_sentinel_lane[
                                            Self.role_local_lane
                                        ](next_row, next_readable)
                                    )
                                    comptime if half_load > 32:
                                        fast_idx_lane_hi = (
                                            _sparse_physical_sentinel_lane_hi[
                                                Self.role_local_lane
                                            ](next_row, next_readable)
                                        )
                    if smx_fast_tile:
                        var m8 = SIMD[Self.AccumType, 8]()
                        comptime for j in range(8):
                            m8[j] = s_row[j][0]
                        comptime for i in range(8, half_load):
                            m8[i % 8] = max(m8[i % 8], s_row[i][0])
                        var raw_max = m8.reduce_max()
                        comptime for _vi in range(vs_count):
                            s_row_val_vectorized[_vi] = (
                                s_row_val_vectorized[_vi] * scale_log2e
                            )
                        # round(max(s) * scale) is max(round(s * scale)).
                        current_max = max(current_max, raw_max * scale_log2e)
                if not smx_fast_tile:
                    comptime if has_per_token_scales:
                        # Fused Place 1 + scale_log2e: vectorized multiply of
                        # sigma_KV and scale_log2e in a single pass to halve
                        # instruction count vs two separate scalar loops.
                        var _sigma_kv_vec_p1 = _sigma_kv_regs.vectorize[2]()
                        comptime for _vi in range(vs_count):
                            s_row_val_vectorized[_vi] = (
                                s_row_val_vectorized[_vi]
                                * _sigma_kv_vec_p1[_vi]
                                * scale_log2e
                            )
                    else:
                        comptime for _vi in range(vs_count):
                            s_row_val_vectorized[_vi] = (
                                s_row_val_vectorized[_vi] * scale_log2e
                            )

                    # under fold, pass num_q_heads so apply_mask's causal
                    # branch can derive per-row horizon via `row // num_q_heads`.
                    # fold_q_num_heads=0 (default)
                    comptime _fold_q_num_heads: Int = (
                        Self.config.num_q_heads if fold_q else 0
                    )
                    # Sliding window is causal + per-row lower bound; so the fast
                    # path treats it as CausalMask=True with SlidingWindowSize set.
                    comptime _causal_for_apply: Bool = (
                        CausalMask or SlidingWindowMask
                    )
                    # Mask the row in the same <= 32-column windows the S loads use,
                    # combining each window's max. One window is the whole row
                    # everywhere except the CTA pair's 64-column rows.
                    comptime for c in range(ceildiv(half_load, s_chunk)):
                        comptime c_off = c * s_chunk
                        comptime c_len = min(32, half_load - c_off)
                        var window_max: Scalar[Self.AccumType]
                        comptime if NoMask or CausalMask or SlidingWindowMask:
                            window_max = Self.apply_mask[
                                half_load,
                                NonCausalMask=False,
                                CausalMask=_causal_for_apply,
                                fold_q_num_heads=_fold_q_num_heads,
                                SlidingWindowSize=_sliding_window_size,
                                SparseCausalLogical=SparseCausalLogical,
                                SparsePhysicalSentinels=mask_physical_sentinels,
                                logical_ballot=ballot_logical,
                                chunk_off=c_off,
                                chunk=c_len,
                            ](
                                tiles_done,
                                col0,
                                num_keys,
                                s_row,
                                mask,
                                prompt_idx,
                                q_head_idx,
                                score_row,
                                cache_len,
                                start_pos,
                                cache_start_pos,
                                # Pass kv_start_row for split-K global position
                                kv_start_row,
                                logical_indices=logical_indices,
                                physical_sentinel_indices=physical_sentinel_indices,
                                logical_indices_stride=logical_indices_stride,
                                q_token_idx=offset_position.q_token_idx,
                                logical_indices_len=logical_indices_len,
                                ballot_pos=hr_pos,
                                gap_lo=gap_lo,
                                gap_hi=gap_hi,
                                row_slot_limit=row_slot_limit,
                                main_slots_hi=main_slots_hi,
                            )
                        else:
                            window_max = Self.apply_mask[
                                half_load,
                                NonCausalMask=True,
                                CausalMask=False,
                                chunk_off=c_off,
                                chunk=c_len,
                            ](
                                tiles_done,
                                col0,
                                num_keys,
                                s_row,
                                mask,
                                prompt_idx,
                                q_head_idx,
                                score_row,
                                cache_len,
                                start_pos,
                                cache_start_pos,
                                # Pass kv_start_row for split-K global position
                                kv_start_row,
                            )
                        # apply_mask seeds each window's max with MASK_VALUE, so
                        # the first window is the row max as is. Folding it into
                        # the MASK_VALUE seed above adds a dependent max to every
                        # tile's softmax.
                        comptime if c == 0:
                            current_max = window_max
                        else:
                            current_max = max(current_max, window_max)
                current_max *= log2e_f32

                comptime rescale_threshold: Float32 = fp8_p_rescale_threshold[
                    scale_fp8_p
                ](Self.config.skip_correction_threshold)
                # Double-buffered write/read: even iterations use buffer 0,
                # odd iterations use buffer 1. Branchless selection via
                # (tiles_done & 1) * WARPGROUP_SIZE — one AND + one MUL + one ADD,
                # no divergent branch on the critical path.
                var buf_parity = tiles_done
                var buf_offset = (buf_parity & 1) * WARPGROUP_SIZE
                var max_buf = TileTensor[
                    Self.AccumType,
                    type_of(smem_1d_layout),
                    address_space=.SHARED,
                ](max_smem.unsafe_offset(buf_offset), smem_1d_layout)
                comptime if half_rows:
                    # Both threads of a row then hold its warp's max bit for bit.
                    comptime if _MLA_HALF_ROWS_BREAK != 1:
                        current_max = max(
                            current_max,
                            shuffle_xor(current_max, UInt32(HR_PAIR_XOR)),
                        )
                    Self._hr_exchange_store(
                        max_smem.unsafe_offset(buf_offset), current_max, lane_id
                    )
                else:
                    max_buf[lane_id] = current_max
                var m_base = mi
                named_barrier[Int32(WARPGROUP_SIZE)](exch_bar_id)
                comptime if half_rows:
                    current_max = Self._hr_reduce[is_max=True](
                        max_smem.unsafe_offset(buf_offset), lane_id
                    )
                elif layout_g:
                    current_max = Self._reduce4[is_max=True](
                        max_smem.unsafe_offset(buf_offset), row
                    )
                else:
                    # 0 ^ 64 = 64, 1 ^ 64 = 65, ... 63 ^ 64 = 127
                    # 64 ^ 64 = 0, 65 ^ 64 = 1, ... 127 ^ 64 = 63
                    var other_half_max = max_buf[lane_id ^ 64][0]
                    current_max = max(current_max, other_half_max)
                var new_max: Scalar[Self.AccumType] = max(m_base, current_max)
                var diff = sub_ftz(
                    rebind[Float32](m_base), rebind[Float32](new_max)
                )
                # `current_max` is initialized to
                # the finite MASK_VALUE in apply_mask, so `new_max >= MASK_VALUE`
                # (finite) on every iteration. First-iter `mi=-inf` gives
                # `diff = -inf - finite = -inf`, exp2(-inf)=0 (finite), no NaN.
                var scale_for_old_max: Scalar[Self.AccumType]
                # Per-lane predicate, not a warp vote: under `fold_q`, `row =
                # lane_id & 0x3F` packs `q_len_fold` distinct (never-committed
                # draft included) query tokens into one warp, so an OR here
                # would leak a sibling token's rescale trajectory into this
                # one's.
                if diff < rescale_threshold:
                    scale_for_old_max = rebind[Scalar[Self.AccumType]](
                        exp2(diff)
                    )
                else:
                    scale_for_old_max = 1.0
                    new_max = m_base
                var float2_register = s_row.vectorize[2]()
                var float2_current_sum: SIMD[Self.AccumType, 2] = 0.0

                # With the finite MASK_VALUE in apply_mask, both `score` and
                # `new_max` are >= MASK_VALUE, so `score - new_max` is finite
                # (worst case `MASK_VALUE - MASK_VALUE = 0` for fully-masked rows,
                # giving `exp2(0) = 1` and `li = N`; the resulting partial_lse is
                # so negative that the combine kernel weights this split as 0).
                var p_offset = -new_max
                comptime if p_bias != 0:
                    p_offset = p_bias - new_max
                comptime for i in range(0, half_load // 2):
                    var element = float2_register[i]
                    float2_register[i] = exp2(element.fma(log2e_f32, p_offset))
                    float2_current_sum += float2_register[i]

                # compute softmax using S_tmem_slot -> produce probabilities in regs
                # Expose correction scalars in SMEM for Correction warpgroup.
                # Skip the FIRST processed tile since there's no prior O
                # accumulator to correct. For non-sliding masks
                # `first_processed_tile_sw` is 0 (original `tiles_done > 0`).
                stamp_tile[0](_sb, tiles_done, 1)
                if tiles_done > first_processed_tile_sw:
                    c_prod.acquire()
                    # write back the exp2f(mi - new_max); to the correction_max_smem
                    # corr_max_Smem_Tensor[lane_id] = scale_for_old
                    # Issue the TMEM store: 32 datapaths × 32 bits × repeat=1
                    # Half rows: lane t + 16 already holds row t's scale
                    # bit for bit, which lands in TMEM lane t + 16.
                    comptime if corr_smem:
                        # C's release/acquire handshake publishes ordinary SMEM
                        # stores and prevents overwrite until Correction reads them.
                        corr_scale_smem.value()[
                            Int(thread_idx.x) % WARPGROUP_SIZE
                        ] = scale_for_old_max
                    else:
                        var _scale_tuple = Array[Scalar[Self.AccumType], 1](
                            fill=scale_for_old_max
                        )
                        tcgen05_st[
                            datapaths=32,
                            bits=32,
                            repeat=1,
                            pack=False,
                        ](corr_scale_tmem, _scale_tuple)
                        tcgen05_store_wait()
                    #  signal to the correction warpgroup:
                    c_prod.commit()

                # wait until MMA has released P (consumer_mbar.phase matches)
                p_prod.acquire()
                stamp_tile[0](_sb, tiles_done, 2)
                var p_stage = p_prod.stage_index()

                # ------------------------------------------------------------------
                # Place 2: Per-token KV scale: pre-fuse sigma_KV[t] into P for
                # PV dequant. Uses register-cached scales loaded at the top of
                # the tile loop (no SMEM read).
                #
                # In MLA absorbed mode V derives from the same FP8 latent as K,
                # so it shares the same per-token scale sigma_KV[t]. The correct
                # PV output is: O[d] = sum_t P[t] * sigma_KV[t] * V_fp8[t][d].
                # We fuse sigma_KV[t] into P before it is written to SMEM and
                # consumed by the PV MMA: P'[t] = P[t] * sigma_KV[t].
                # ------------------------------------------------------------------
                comptime if has_per_token_scales:
                    # Reuse register-cached sigma_KV values from Place 1.
                    # Vectorized: use SIMD[Float32, 2] to halve instruction count.
                    var _sigma_kv_vec_p2 = _sigma_kv_regs.vectorize[2]()
                    var _s_row_vec_p2 = s_row.vectorize[2]()
                    comptime _vs_count_p2 = ceildiv(half_load, 2)
                    comptime for _vi in range(_vs_count_p2):
                        _s_row_vec_p2[_vi] = (
                            _s_row_vec_p2[_vi] * _sigma_kv_vec_p2[_vi]
                        )

                comptime if native_fp8:
                    # FP8 path: P lives in SMEM (separate region or reusing rope)
                    comptime fp8_p_type = DType.float8_e4m3fn
                    # When fp8_p_stage_stride > 0, P reuses KV rope SMEM:
                    # P_i is at rope_base + i * fp8_p_stage_stride (in FP8 elems).
                    # When 0 (default), P stages are contiguous at BlockElems
                    # apart. Under the CTA pair BlockElems is already the whole
                    # BN_QK-wide stage (two 64-key blocks), so the stride is
                    # BlockElems itself -- doubling it walked slot 1 past the
                    # P ring at tile 1, and the out-of-range STS.128 faulted.
                    comptime _p_stride = (
                        Self.BlockElems if Self.config.cta_group
                        == 2 else (
                            fp8_p_stage_stride if fp8_p_stage_stride
                            > 0 else Self.BlockElems
                        )
                    )
                    var p_smem_stage = p_smem_ptr.unsafe_bitcast[
                        Scalar[fp8_p_type]
                    ]().unsafe_offset(p_stage * UInt32(_p_stride))
                    comptime if half_rows:
                        var p_col = col0
                        comptime if _MLA_HALF_ROWS_BREAK == 3:
                            p_col = col0 & ~(half_load)
                        comptime if half_load == 16:
                            # A 128-key P row is one SW128 row; 16 keys are one
                            # 16 B chunk.
                            write_fp8_row_to_smem_chunked[
                                half_load,
                                out_dtype=fp8_p_type,
                                in_dtype=Self.AccumType,
                                config=Self.config,
                            ](p_smem_stage, s_row, p_col, row)
                        else:
                            write_fp8_half_row_to_smem[
                                half_load,
                                out_dtype=fp8_p_type,
                                in_dtype=Self.AccumType,
                                config=Self.config,
                            ](p_smem_stage, s_row, p_col, row)
                    elif Self.config.BN_QK == 128:
                        # The 128-key P stage is two self-contained 64-key SW64
                        # blocks; each softmax thread's columns (one key half)
                        # lie inside exactly one of them.
                        write_fp8_row_to_smem_chunked[
                            half_load,
                            out_dtype=fp8_p_type,
                            in_dtype=Self.AccumType,
                            config=Self.config,
                            row_size=SW64_COLS,
                        ](
                            p_smem_stage.unsafe_offset(
                                UInt32(
                                    (col0 >> 6) * (Self.config.BM * SW64_COLS)
                                )
                            ),
                            s_row,
                            col0 & 63,
                            row,
                        )
                    else:
                        write_fp8_row_to_smem_chunked[
                            half_load,
                            out_dtype=fp8_p_type,
                            in_dtype=Self.AccumType,
                            config=Self.config,
                        ](p_smem_stage, s_row, col0, row)
                elif bf16_p_ring:
                    comptime assert (
                        Self.config.BN_QK == SW64_COLS
                    ), "the BF16 P ring holds 64-key SW128 rows"
                    var p_col = col0
                    comptime if half_rows and _MLA_HALF_ROWS_BREAK == 3:
                        p_col = col0 & ~(half_load)
                    write_bf16x2_row_to_smem_chunked[
                        half_load,
                        out_dtype=Self.q_type,
                        in_dtype=Self.AccumType,
                        config=Self.config,
                        chunk_size=min(half_load, 16),
                    ](
                        p_smem_ptr.unsafe_offset(
                            p_stage * UInt32(Self.BlockElems)
                        ),
                        s_row,
                        p_col,
                        row,
                    )
                else:
                    # `NumVOBlocks` and `KVStageElems` count BN_QK-wide blocks,
                    # which is the SW64 block layout only at 64-key tiles.
                    comptime assert (
                        Self.config.BN_QK == SW64_COLS
                    ), "the BF16 P offset assumes 64-key tiles"
                    # BF16 path: P is embedded inside KV stage SMEM
                    var p_smem = p_smem_ptr.unsafe_offset(
                        p_stage * UInt32(Self.KVStageElems)
                        + UInt32(Self.NumVOBlocks * Self.BlockElems)
                    )
                    write_bf16x2_row_to_smem_chunked[
                        half_load,
                        out_dtype=Self.q_type,
                        in_dtype=Self.AccumType,
                        config=Self.config,
                    ](p_smem, s_row, col0, row)

                fence_async_view_proxy()
                comptime if is_cg2:
                    # One elected lane per warp arrives the leader's P-full
                    # barrier from both CTAs, after the warp's generic-proxy
                    # stores are complete (count: one per warp per CTA).
                    syncwarp()
                    var _p_elect = elect()
                    if _p_elect != 0:
                        p_prod.pipe.producer_mbar()[].arrive_cluster(
                            UInt32(0), UInt32(1)
                        )
                    p_prod.pipe.step()
                else:
                    # 128 threads call -> producer_mbar.arrive() (128 arrivals) + state.step()
                    p_prod.commit()
                stamp_tile[0](_sb, tiles_done, 3)
                mi = new_max
                li = li.fma(
                    scale_for_old_max,
                    float2_current_sum[0] + float2_current_sum[1],
                )
                # now update the li scale for the next tile
                tiles_done += 1
        stamp_finish[0](_sb, num_k_tiles)

        comptime if half_rows:
            li += shuffle_xor(li, UInt32(HR_PAIR_XOR))
            Self._hr_exchange_store(li_smem, li, lane_id)
        else:
            li_Smem_Tensor[lane_id] = li
        named_barrier[Int32(WARPGROUP_SIZE)](exch_bar_id)
        comptime if half_rows:
            li = Self._hr_reduce[is_max=False](li_smem, lane_id)
        elif layout_g:
            li = Self._reduce4[is_max=False](li_smem, row)
        else:
            li += li_Smem_Tensor[lane_id ^ 64][0]

        # li is a sum of exp2(s - p_ref).
        var p_ref = mi - p_bias

        # --------------------------------------------------------------------------
        # Split-K: Store partial LSE to lse_accum_split for combine kernel
        # --------------------------------------------------------------------------
        # LSE (Log-Sum-Exp) in log2 format: lse = log2(li) + p_ref
        # This allows the combine kernel to merge partial results:
        #   global_lse = log2(sum(exp2(lse_i - max_lse))) + max_lse
        #   scale_i = exp2(lse_i - global_lse)
        #   final_output = sum(scale_i * partial_output_i)
        #
        # LSE accumulator shape: (num_splits, batch_size, seq_len, num_heads)
        # Strides: stride_split = batch_size * seq_len * num_heads
        #          stride_batch = seq_len * num_heads
        #          stride_seq = num_heads
        comptime if Self.config.decoding_warp_split_k:
            # Only threads with valid heads should write LSE
            # head_idx = head_tile * BM + row (where row is 0-63 for each half)
            # Each thread in the warpgroup handles one row (one head)
            # row = lane_id & 0x3F gives 0-63 for both halves
            # half = lane_id >> 6 gives 0 or 1
            # We only need one write per head, so half=0 threads write
            # 0 for the threads that own the LSE write (the first half under
            # Layout E, warp 0 under Layout G).
            var half_idx = lane_id >> (5 if layout_g else 6)

            # Under fold, BM=64 packs q_len_fold * num_q_heads rows
            # and THIS CTA owns all q_max_seq_len LSE slots for the batch. So
            # rows where q_local >= seq_len (ragged) must explicitly write
            # -inf into their LSE slots (the accum buffer is uninitialized
            # for those slots, unlike the non-fold path where ragged CTAs
            # take pdl_early_exit and never reach Softmax).
            comptime if fold_q:
                var q_local, head_local = divmod(row, Self.config.num_q_heads)
                if half_idx == 0 and row < (
                    q_len_fold * Self.config.num_q_heads
                ):
                    var partial_lse: Scalar[Self.AccumType]
                    if q_local >= offset_position.seq_len:
                        partial_lse = min_or_neg_inf[Self.AccumType]()
                    else:
                        partial_lse = (
                            log2(max(li[0], Scalar[Self.AccumType](0))) + p_ref
                        )
                    var stride_batch = (
                        offset_position.max_seq_len * Self.config.num_q_heads
                    )
                    var stride_split = batch_size * stride_batch
                    var stride_seq = Self.config.num_q_heads
                    var lse_offset = (
                        offset_position.split_idx * stride_split
                        + offset_position.batch_idx * stride_batch
                        + q_local * stride_seq
                        + head_local
                    )
                    var lse_ptr = rebind[
                        Pointer[Scalar[Self.AccumType], origin=MutAnyOrigin]
                    ](lse_accum_split_ptr.value())
                    lse_ptr[unsafe_offset=lse_offset] = partial_lse
            else:
                var head_idx = (
                    decode_head_tile[Self.config]() * Self.config.BM + row
                )
                if half_idx == 0 and head_idx < Self.config.num_q_heads:
                    # Compute LSE in log2 format: log2(li) + p_ref
                    # li is the running sum of exp2 values; mi is the running max
                    # in log2 scale. When all scores in this split are causally
                    # masked, the online softmax produces NaN via exp2(-inf+inf),
                    # poisoning li. Clamping li to 0 makes log2(0)=-inf, and
                    # -inf + mi(-inf) = -inf, giving this split zero weight in
                    # the combine kernel (same as pdl_early_exit for empty splits).
                    # On NVIDIA GPUs, max(NaN, 0) = 0 per PTX semantics.
                    var partial_lse = (
                        log2(max(li[0], Scalar[Self.AccumType](0))) + p_ref
                    )

                    # LSE offset calculation:
                    # lse_accum_split shape: (num_splits, batch_size, max_seq_len, num_heads)
                    # Use max_seq_len (not per-batch seq_len) for strides to match
                    # the PADDED buffer layout and the combine kernel's read pattern.
                    var seq_idx = decode_seq_idx[Self.config]()
                    var stride_batch = (
                        offset_position.max_seq_len * Self.config.num_q_heads
                    )
                    var stride_split = batch_size * stride_batch
                    var stride_seq = Self.config.num_q_heads

                    var lse_offset = (
                        offset_position.split_idx * stride_split
                        + offset_position.batch_idx * stride_batch
                        + seq_idx * stride_seq
                        + head_idx
                    )
                    # need to rebind the pointer to mutable pointer for write access
                    var lse_ptr = rebind[
                        Pointer[Scalar[Self.AccumType], origin=MutAnyOrigin]
                    ](lse_accum_split_ptr.value())
                    lse_ptr[lse_offset] = partial_lse
        elif not Self.SplitAccumType.is_null:
            # No split, but the caller asked for the LSE (decode context
            # parallelism merges ranks with it). The pointer is then the op's
            # ragged `[total_q_tokens, num_heads]` output, not a split-K
            # workspace, and the LSE is final, so no sink may be folded into
            # O behind it.
            comptime assert not fold_q, "the LSE output does not support fold_q"
            comptime assert (
                not has_attn_sink
            ), "the LSE output does not support an attention sink"
            var head_idx = (
                decode_head_tile[Self.config]() * Self.config.BM + row
            )
            if (lane_id >> (5 if layout_g else 6)) == 0 and (
                head_idx < Self.config.num_q_heads
            ):
                var lse_ptr = rebind[
                    Pointer[Scalar[Self.AccumType], origin=MutAnyOrigin]
                ](lse_accum_split_ptr.value())
                lse_ptr[
                    offset_position.q_token_idx * Self.config.num_q_heads
                    + head_idx
                ] = (log2(max(li[0], Scalar[Self.AccumType](0))) + p_ref)

        # --------------------------------------------------------------------------
        # Epilogue: scale output by recip(li) and write to shared memory as bf16
        # --------------------------------------------------------------------------
        comptime assert (
            Self.AccumType == .float32
        ), "accumulator type should be float32"
        comptime assert (
            Self.output_dtype == .bfloat16 or Self.output_dtype == .float16
        ), "output type must be bfloat16 or float16"

        comptime DecodeOutProducerType = DecodeOutProducer[
            Self.output_dtype, Self.config
        ]
        comptime blocks_per_stage = DecodeOutProducerType.blocks_per_stage
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)

        # By the time we reach to epilogue the KV is free.
        # So we can safely use the KV buffer for writing the output and have more async write.
        # however the write function massively changes based on the mma_n size
        # so when mma_n is 256 we have the first 128 columns with warp0/1 and the
        # next 128 column with warp2/3 tiles and so on for the next 256 columns
        # it is 256/32 which is equivalent of 512/64

        # Half-stripe width in fp32 per warp_pair = BN_PV/8.
        comptime epi_half_load: UInt32 = UInt32((Self.config.BN_PV // 4) >> 1)
        comptime chunk_size: Int = 16
        comptime total_elems: Int = Int(epi_half_load) * blocks_per_stage
        var out_prod = DecodeOutProducer[Self.output_dtype, Self.config](
            out_pipeline, out_smem
        )

        # Pre-compute scale factor.
        # Guard against NaN in li (possible when all scores in a split are
        # masked, producing exp2(-inf+inf)=NaN that poisons li). Using
        # `li[0] > 0` instead of `li[0] != 0` ensures NaN maps to 0,
        # zeroing the output for this split — consistent with the LSE path's
        # max(li, 0) guard and the combine kernel's weighting.
        #
        # Attn sink (no-split only): account for non-selected tokens by
        # adding exp2(attn_sink_log2 - p_ref) to the denominator. For the
        # split path, attn_sink is deferred to the combine kernel to
        # avoid double-counting across splits.
        var o_scale_li: Scalar[Self.AccumType]
        comptime if has_attn_sink and not Self.config.decoding_warp_split_k:
            # No-split path with attn_sink: o_scale = 1 / (li + exp2(attn_sink_log2 - p_ref))
            # FlashMLA reference: kernel.cuh:346
            var denominator = li[0] + exp2(
                attn_sink_log2.cast[Self.AccumType]() - p_ref
            )
            o_scale_li = (
                recip(SIMD[Self.AccumType, 1](denominator))[0] if li[0]
                > 0 else 0
            )
        else:
            # Split path or no attn_sink: standard recip(li).
            # FlashMLA reference: kernel.cuh:387
            o_scale_li = recip(li)[0] if li[0] > 0 else 0

        var warp_pair = UInt32(warp_idx >> 1)
        var epi_col0: Int = Int(
            warp_pair * epi_half_load * UInt32((blocks_per_stage >> 1) ^ 1)
        )

        # Number of MMA PV rounds (outer loop) and iterations within each round (inner loop)
        # MMA_PV_N=256 processes 4 blocks (256/64=4) at a time
        # depth=512 has 8 blocks total, so 2 MMA PV rounds (512/256=2)
        # Each round has (MMA_PV_N/BN_QK)/blocks_per_stage = (256/64)/2 = 2 iterations
        # corr_done_bars has 2 slots matching the 2 MMA PV rounds
        #   0       64     128     192      256      320      384     448     512
        #   |-------|-------|-------|--------|--------|--------|-------|-------|
        #     w0/1    w0/1     w2/3    w2/3     w0/1     w0/1     w2/3    w2/3
        # The pattern repeats every MMA_PV_N (256) columns
        comptime num_mma_pv_rounds = Self.config.depth // Self.config.MMA_PV_N
        # iters_per_mma_round = (MMA_PV_N / (BN_PV/4)) / blocks_per_stage = 4 / 2.
        comptime iters_per_mma_round = 4 // blocks_per_stage

        comptime if layout_g:
            comptime if half_rows:
                Self._epilogue_layout_g_half[
                    num_mma_pv_rounds, iters_per_mma_round
                ](
                    o_tmem,
                    corr_done_bars,
                    out_prod,
                    row,
                    hr_half(lane_id & (WARP_SIZE - 1)),
                    o_scale_li,
                )
            else:
                Self._epilogue_layout_g[num_mma_pv_rounds, iters_per_mma_round](
                    o_tmem, corr_done_bars, out_prod, row, o_scale_li
                )
            return

        comptime for mma_round in range(num_mma_pv_rounds):
            # Wait for Correction to finish corrections for this MMA PV round
            corr_done_bars.mbar_base[unsafe_offset=mma_round].wait(0)

            # Fence to ensure all MMA writes to O TMEM are visible before we read
            tcgen05_fence_after()

            comptime for slot in range(iters_per_mma_round):
                # Global iteration index combining mma_round and slot
                comptime i = mma_round * iters_per_mma_round + slot

                var o_tmem_base: UInt32 = o_tmem + UInt32(
                    i
                ) * epi_half_load * UInt32(blocks_per_stage)

                # Load all data for this tile into a LocalTensor
                var o_row_subtile = stack_allocation[
                    dtype=Self.AccumType, address_space=.LOCAL
                ](row_major[total_elems]())
                var _o_ld_result = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=total_elems,
                    dtype=Self.AccumType,
                    pack=False,
                ](o_tmem_base)

                comptime for _i in range(total_elems):
                    o_row_subtile[_i] = _o_ld_result[_i]
                tcgen05_load_wait()

                out_prod.acquire()
                var stage_ptr = out_prod.stage_base_ptr(
                    Int(warp_pair * UInt32(blocks_per_stage >> 1))
                )

                # Write O to shared memory with scaling
                write_bf16x2_row_to_smem_chunked[
                    total_elems,
                    out_dtype=Self.output_dtype,
                    in_dtype=Self.AccumType,
                    config=Self.config,
                    chunk_size=chunk_size,
                    scale_needed=True,
                ](stage_ptr, o_row_subtile, epi_col0, row, o_scale_li)

                out_prod.commit_step()

    @staticmethod
    @inline(.always)
    def _reduce4[
        is_max: Bool
    ](buf: SharedMemPointer[Scalar[Self.AccumType]], row: Int) -> Scalar[
        Self.AccumType
    ]:
        """Reduces one row's four per-warp partials under Layout G.

        Every warp combines the four values in the same fixed order, so all
        four hold a bit-identical result and their per-lane lazy-rescale
        decisions agree.
        """
        var w0 = buf[unsafe_offset=row]
        var w1 = buf[unsafe_offset=row + 32]
        var w2 = buf[unsafe_offset=row + 64]
        var w3 = buf[unsafe_offset=row + 96]
        comptime if is_max:
            return max(max(w0, w1), max(w2, w3))
        else:
            return (w0 + w1) + (w2 + w3)

    @staticmethod
    @inline(.always)
    def _hr_exchange_store(
        buf: SharedMemPointer[Scalar[Self.AccumType]],
        val: Scalar[Self.AccumType],
        lane_id: Int,
    ):
        """Posts this thread's half-row partial for `_hr_reduce`.

        The layout is `[row][warp]`; both threads of a row store the same
        value. The slot inside a row is permuted by the row so a warp's store
        is one wavefront; every reader of a row reads it whole in the same
        order, so the results stay bit-identical.
        """
        var t = lane_id & (WARP_SIZE - 1)
        var w = lane_id >> 5
        var r = hr_row(t)
        buf[unsafe_offset=r * 4 + ((w + 2 * (r >> 3)) & 3)] = val

    @staticmethod
    @inline(.always)
    def _hr_reduce[
        is_max: Bool
    ](buf: SharedMemPointer[Scalar[Self.AccumType]], lane_id: Int) -> Scalar[
        Self.AccumType
    ]:
        """Reduces one row's half-row partials posted by `_hr_exchange_store`.

        Every thread of the row combines them in the same fixed order, so all
        hold a bit-identical result.
        """
        var t = lane_id & (WARP_SIZE - 1)
        var v = (buf.unsafe_offset(hr_row(t) * 4)).unsafe_load[
            width=4, alignment=16
        ]()
        comptime if is_max:
            return max(max(v[0], v[1]), max(v[2], v[3]))
        else:
            return (v[0] + v[1]) + (v[2] + v[3])

    @staticmethod
    @inline(.always)
    def _hr_logical_pos(
        tile: Int,
        num_k_tiles: Int,
        kv_start_row: Int,
        lane_id: Int,
        logical_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        logical_indices_stride: Int,
        q_token_idx: Int,
        logical_indices_len: Int,
    ) -> Int32:
        """Returns the logical position of key slot `lane` of this warp's
        `BN_QK / 4` columns of `tile`, or -1 (masked) for lanes past them,
        slots past `logical_indices_len`, tiles past `num_k_tiles` and a null
        buffer.
        """
        var j = lane_id & (WARP_SIZE - 1)
        var col_base16 = (
            kv_start_row
            + tile * Self.config.BN_QK
            + (lane_id >> 5) * (Self.config.BN_QK >> 2)
        )
        var n_readable = max(
            min(logical_indices_len - col_base16, Self.config.BN_QK >> 2), 0
        )
        var pos: Int32 = -1
        if logical_indices and tile < num_k_tiles and j < n_readable:
            pos = logical_indices.unsafe_value()[
                q_token_idx * logical_indices_stride + col_base16 + j
            ]
        return pos

    @staticmethod
    @inline(.always)
    def _epilogue_layout_g[
        num_mma_pv_rounds: Int,
        out_stages_per_round: Int,
    ](
        o_tmem: UInt32,
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        mut out_prod: DecodeOutProducer[Self.output_dtype, Self.config],
        row: Int,
        o_scale_li: Scalar[Self.AccumType],
    ):
        """Writes O to the output stages under Layout G.

        Warp `w` holds depth `[64w, 64w + 64)` of each round's 256 columns,
        for all 32 rows. Output stage `m` of a round holds depth
        `m*64 + k*128` in its half `k`, so warp `w` writes half `w >> 1` of
        stage `w & 1`. Every thread acquires and commits every stage, since
        the stage's producer count is the whole warpgroup.

        """
        comptime band = Self.config.BN_PV // 4
        comptime o_round_cols = Self.config.o_round_cols()
        comptime assert (
            out_stages_per_round == 2
            and DecodeOutProducer[
                Self.output_dtype, Self.config
            ].blocks_per_stage
            == 2
        ), "Layout G epilogue assumes 2 stages of 2 halves per round."
        var warp_in_wg = Int(warp_id[broadcast=True]() & 3)
        comptime half_band = band // 2
        comptime for mma_round in range(num_mma_pv_rounds):
            corr_done_bars.mbar_base[unsafe_offset=mma_round].wait(0)
            tcgen05_fence_after()
            var o_band = stack_allocation[
                dtype=Self.AccumType, address_space=.LOCAL
            ](row_major[band]())
            var _o_ld_result = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=band,
                dtype=Self.AccumType,
                pack=False,
            ](o_tmem + UInt32(mma_round * o_round_cols))
            comptime for _i in range(band):
                o_band.raw_store(_i, _o_ld_result[_i])
            tcgen05_load_wait()
            comptime for stage in range(out_stages_per_round):
                out_prod.acquire()
                if (warp_in_wg & 1) == stage:
                    write_bf16x2_row_to_smem_chunked[
                        band,
                        out_dtype=Self.output_dtype,
                        in_dtype=Self.AccumType,
                        config=Self.config,
                        chunk_size=16,
                        scale_needed=True,
                    ](
                        out_prod.stage_base_ptr(warp_in_wg >> 1),
                        o_band,
                        0,
                        row,
                        o_scale_li,
                    )
                out_prod.commit_step()

    @staticmethod
    @inline(.always)
    def _epilogue_layout_g_half[
        num_mma_pv_rounds: Int,
        out_stages_per_round: Int,
    ](
        o_tmem: UInt32,
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        mut out_prod: DecodeOutProducer[Self.output_dtype, Self.config],
        row: Int,
        half: Int,
        o_scale_li: Scalar[Self.AccumType],
    ):
        """Writes O to the output stages from 16 lanes under Layout G.

        `_epilogue_layout_g` with two threads per row: each reads 32 of its
        warp's 64 band columns (`half`) through the 16-lane shape and writes
        them; output rows 16-31 are left unwritten (the store covers at most
        16 rows here). Single pipeline or shared O only.
        """
        comptime band = Self.config.BN_PV // 4
        comptime half_band = band // 2
        comptime o_round_cols = Self.config.o_round_cols()
        comptime assert (
            out_stages_per_round == 2
            and DecodeOutProducer[
                Self.output_dtype, Self.config
            ].blocks_per_stage
            == 2
        ), "Layout G epilogue assumes 2 stages of 2 halves per round."
        var warp_in_wg = Int(warp_id[broadcast=True]() & 3)
        var quad_base = UInt32((WARP_SIZE * warp_in_wg) << 16)
        comptime for mma_round in range(num_mma_pv_rounds):
            corr_done_bars.mbar_base[unsafe_offset=mma_round].wait(0)
            tcgen05_fence_after()
            var o_half = stack_allocation[
                dtype=Self.AccumType, address_space=.LOCAL
            ](row_major[half_band]())
            var _o_ld = tcgen05_ld_16x32bx2[half_band, imm=half_band](
                o_tmem + quad_base + UInt32(mma_round * o_round_cols)
            )
            comptime for _i in range(half_band):
                o_half.raw_store(_i, rebind[Scalar[Self.AccumType]](_o_ld[_i]))
            tcgen05_load_wait()
            comptime for stage in range(out_stages_per_round):
                out_prod.acquire()
                if (warp_in_wg & 1) == stage:
                    write_bf16x2_row_to_smem_chunked[
                        half_band,
                        out_dtype=Self.output_dtype,
                        in_dtype=Self.AccumType,
                        config=Self.config,
                        chunk_size=16,
                        scale_needed=True,
                    ](
                        out_prod.stage_base_ptr(warp_in_wg >> 1),
                        o_half,
                        half * half_band,
                        row,
                        o_scale_li,
                    )
                out_prod.commit_step()

    # --------------------------------------------------------------------------
    # MLA decoding Correction kernel
    # --------------------------------------------------------------------------
    @staticmethod
    @inline(.always)
    def Correction[
        _op_sparse: Bool = False,
        _op_has_extra_kv: Bool = False,
        _OpTopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
        # Arrivals per O-free barrier phase: one per correction thread, or
        # -- under the CTA pair -- one elected lane per warp per CTA
        # (2 * 4).
        num_o_warps: Int = WARPGROUP_SIZE,
        corr_smem: Bool = False,
        phase_from_o: Bool = False,
        countdown: Bool = False,
    ](
        tmem_addr: UInt32,
        o_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=1,
            num_consumer=num_o_warps,
        ],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            _op_sparse,
            _op_has_extra_kv,
            _OpTopkLengthsPtrType,
        ],
        corr_scale_smem: OptionalReg[
            SharedMemPointer[Scalar[Self.AccumType]]
        ] = None,
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var corr_scale_tmem = tmem_addr + UInt32(Self.config.tmem_corr_scale)
        var o_cons = DecodeOConsumer(o_bars.consumer())
        var c_cons = DecodeCConsumer(c_bars.consumer())
        # The clamped sparse bound keeps the trip count inside Int32, so one
        # 32-bit countdown carries the iteration state instead of a 64-bit
        # bound and a 64-bit up-counter, which spill across the FP body.
        comptime if countdown:
            comptime assert (
                Self.config.cta_group == 1
                and Self.config.num_q_heads == 128
                and Self.config.BN_QK == 64
                and Self.MaskType.get_type_name() == "CausalMask"
            ), "the countdown form requires the clamped NV geometry"
        comptime iteration_dtype = DType.int32 if countdown else DType.int
        comptime IterationType = Scalar[iteration_dtype]
        var tiles_done = IterationType(1)
        # The pair's O release reaches the leader from both CTAs: one
        # elected lane per warp per CTA (the TMEM walk is warp-collective,
        # so the elected lane arrives for the whole warp).
        comptime is_cg2 = Self.config.cta_group == 2

        var num_k_tiles = IterationType(
            ceildiv(offset_position.num_keys_this_split, Self.config.BN_QK)
        )

        # Sliding-window leading-tile skip — comptime-gated; entire block
        # compiles away for non-sliding masks. Correction starts AFTER
        # Softmax's first processed tile, i.e. at `tile_skip + 1`. Must
        # match the load skip exactly so producer/consumer iterations align.
        # Empty-split (tile_skip >= num_k_tiles) cannot reach here in
        # split-K mode because the kernel-level pdl_early_exit fires first.
        comptime _sliding_window_mask_corr: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        comptime if _sliding_window_mask_corr:
            comptime _W_corr: Int = Self.MaskType.sliding_window_size()
            var _global_lo_corr = max(
                offset_position.cache_len() + 1 - _W_corr, 0
            )
            var _local_lo_corr = max(
                _global_lo_corr - offset_position.kv_start_row, 0
            )
            var _tile_skip_corr = _local_lo_corr // Self.config.BN_QK
            tiles_done = IterationType(_tile_skip_corr + 1)

        comptime if countdown:
            tiles_done = num_k_tiles - IterationType(1)
        var _sb = stamp_begin[1]()
        while (
            tiles_done
            > IterationType(0) if countdown else tiles_done
            < num_k_tiles
        ):
            # after computing per-row c_scalar from max/li:
            comptime if phase_from_o:
                comptime assert DecodeOConsumer.ONumStages == 2
                comptime assert DecodeCConsumer.CNumStages == 1
                comptime assert Self.config.depth // Self.config.MMA_PV_N == 2
                # One C release and two O releases advance both phases once.
                c_cons.pipe.state = PipelineState[1](
                    0, Int(o_cons.pipe.state.phase()), 0
                )
            c_cons.wait()
            stamp_tile[1](
                _sb,
                Int(num_k_tiles - tiles_done) if countdown else Int(tiles_done),
                0,
            )
            # 2) Issue TMEM load: 32 datapaths × 32 bits × repeat=1
            var scale_value: Scalar[Self.AccumType]
            comptime if corr_smem:
                scale_value = corr_scale_smem.value()[
                    Int(thread_idx.x) % WARPGROUP_SIZE
                ]
            else:
                var scale_value_tuple = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    dtype=Self.AccumType,
                    pack=False,
                ](corr_scale_tmem)
                tcgen05_load_wait()
                scale_value = scale_value_tuple[0]
            c_cons.release()
            var change = vote[.uint32](scale_value < 1.0) != 0
            comptime if _MLA_CORR_BREAK != 0:
                change = False
            # Layout G: each warp's 64-column band is one tile.
            comptime num_o_tiles = 1 if Self.config.decode_layout_g else (
                Self.config.MMA_PV_N // (Self.output_tile_width * 2)
            )
            comptime o_range = Self.config.depth // Self.config.MMA_PV_N
            comptime o_stride = Self.config.o_round_cols()
            # Columns of one O tile: under Layout G a warp's depth band of a
            # PV round, whatever the keys per tile. Else a PV round's
            # o_round_cols() split across num_o_tiles; the pair keeps the
            # per-CTA O layout, so this stays 64 however many keys a tile
            # holds.
            comptime o_band = (
                Self.config.BN_PV // 4
            ) if Self.config.decode_layout_g else o_stride // num_o_tiles

            comptime for slot_idx in range(o_range):
                o_cons.wait()
                stamp_tile[1](
                    _sb,
                    Int(num_k_tiles - tiles_done) if countdown else Int(
                        tiles_done
                    ),
                    1 + slot_idx,
                )
                if change:
                    comptime for i in range(0, num_o_tiles):
                        # Here we load from o_tmem. it is 32 bit float and we load 64 fp32 element per tile
                        var o_tmem_subtile: UInt32 = (
                            o_tmem
                            + UInt32(i) * UInt32(o_band)
                            + UInt32(slot_idx) * UInt32(o_stride)
                        )
                        var o_row_subtile = stack_allocation[
                            dtype=Self.AccumType,
                            address_space=.LOCAL,
                        ](row_major[o_band]())
                        var _o_ld_corr = tcgen05_ld[
                            datapaths=32,
                            bits=32,
                            repeat=o_band,
                            dtype=Self.AccumType,
                            pack=False,
                        ](o_tmem_subtile)

                        comptime for _i in range(o_band):
                            o_row_subtile[_i] = _o_ld_corr[_i]
                        tcgen05_load_wait()

                        var float2_register = o_row_subtile.vectorize[2]()

                        comptime for j in range(0, o_band // 2):
                            var element = rebind[SIMD[Self.AccumType, 2]](
                                float2_register[j]
                            )
                            float2_register[j] = rebind[
                                type_of(float2_register[j])
                            ](element * SIMD[Self.AccumType, 2](scale_value))
                        var _o_st_corr = Array[_, o_band](
                            fill_with_unrolled=lambda [i: Int]() -> Scalar[
                                Self.AccumType
                            ]: o_row_subtile[i]
                        )
                        tcgen05_st[
                            datapaths=32,
                            bits=32,
                            repeat=o_band,
                            pack=False,
                        ](
                            o_tmem_subtile,
                            _o_st_corr,
                        )
                        tcgen05_store_wait()
                comptime if is_cg2:
                    var _o_elect = elect()
                    if _o_elect != 0:
                        o_cons.pipe.consumer_mbar()[].arrive_cluster(
                            UInt32(0), UInt32(1)
                        )
                    o_cons.pipe.step()
                else:
                    o_cons.release()
            stamp_tile[1](
                _sb,
                Int(num_k_tiles - tiles_done) if countdown else Int(tiles_done),
                3,
            )
            comptime if countdown:
                tiles_done -= IterationType(1)
            else:
                tiles_done += IterationType(1)

        # Wait on the final O from MMA before signaling Softmax
        o_cons.wait()
        stamp_finish[1](_sb, Int(num_k_tiles))
        # Signal to Softmax that first 4 blocks are ready (slot 0)
        _ = corr_done_bars.mbar_base[].arrive()
        comptime if is_cg2:
            var _o_elect = elect()
            if _o_elect != 0:
                o_cons.pipe.consumer_mbar()[].arrive_cluster(
                    UInt32(0), UInt32(1)
                )
            o_cons.pipe.step()
        else:
            o_cons.release()
        # second stage of the correction pipeline
        o_cons.wait()
        # Signal to Softmax that all corrections are done and O is ready (slot 1)
        _ = corr_done_bars.mbar_base[unsafe_offset=1].arrive()
        # Release the final O barrier
        comptime if is_cg2:
            var _o_elect = elect()
            if _o_elect != 0:
                o_cons.pipe.consumer_mbar()[].arrive_cluster(
                    UInt32(0), UInt32(1)
                )
            o_cons.pipe.step()
        else:
            o_cons.release()

    # --------------------------------------------------------------------------
    # MLA decoding store kernel
    # --------------------------------------------------------------------------
    # If it goes to the batch loop remember correction is out of sync with MMA on
    # O_tmem wait and release as it starts one stage before MMA
    @staticmethod
    @inline(.always)
    def store_empty(
        o_tma: ORaggedTMATile[
            dtype=Self.output_dtype,
            BM=Self.config.out_rows,
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        row: Int,
    ):
        # No MMA publishes an output tile for a valid zero-length selection.
        # All-zero storage is invariant under the descriptor's swizzle.
        comptime columns = Self.config.BN_PV // 4
        comptime elements = Self.config.out_rows * columns
        var ptr = external_memory[
            Scalar[Self.output_dtype],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        for i in range(Int(thread_idx.x), elements, Int(block_dim.x)):
            ptr[i] = 0
        fence_async_view_proxy()
        barrier()
        if thread_idx.x == 0:
            comptime layout = tt_row_major[elements]()
            var tile = TileTensor[
                Self.output_dtype,
                type_of(layout),
                address_space=.SHARED,
            ](ptr, layout)
            var rows = rows_owned[Self.config](decode_head_tile[Self.config]())
            comptime for col in range(0, Self.config.padded_depth, columns):
                o_tma.async_store_3d(
                    tile, store_row_coords[Self.config.out_rows](col, row, rows)
                )
            o_tma.commit_group()
            o_tma.wait_group[0]()
        barrier()

    @staticmethod
    @inline(.always)
    def store[
        _op_sparse: Bool = False,
        _op_has_extra_kv: Bool = False,
        _OpTopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
        # When True, the fold caller strides the output-store
        # TMA per-q_token via a dedicated out_row_offset_at(q_local) accessor.
        fold_q: Bool = False,
        # comptime number of q_tokens packed
        # into BM=64 under fold_q=True. Only consumed inside the
        # `comptime if fold_q` branch; when fold_q=False, ignored.
        q_len_fold: Int = 1,
    ](
        out_pipeline: OutPipeline[
            num_out_stages=DecodeOutProducer[
                Self.output_dtype, Self.config
            ].num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
            cta_group=Self.config.cta_group,
        ],
        out_smem: SharedMemPointer[Scalar[Self.output_dtype]],
        o_tma: ORaggedTMATile[
            dtype=Self.output_dtype,
            BM=Self.config.out_rows,
            # BF16/SWIZZLE_128B clamps innermost to 64 (= BN_PV/4).
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            _op_sparse,
            _op_has_extra_kv,
            _OpTopkLengthsPtrType,
        ],
    ):
        comptime DecodeOutConsumerType = DecodeOutConsumer[
            Self.output_dtype, Self.config
        ]
        comptime col_per_warp = DecodeOutConsumerType.col_per_warp
        comptime blocks_per_stage = DecodeOutConsumerType.blocks_per_stage
        comptime num_out_stages = DecodeOutConsumerType.num_out_stages
        comptime num_out_stages_per_mma = num_out_stages // num_mma_pv
        comptime num_mma_pv = Self.config.padded_depth // Self.config.MMA_PV_N
        var out_cons = DecodeOutConsumer[Self.output_dtype, Self.config](
            out_pipeline, out_smem
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.out_row_offset
        comptime if Self.config.cta_group == 2:
            # Keep the second TMA row coordinate out of the entry lifetime;
            # otherwise the driver spills row + 64 across the compute roles.
            row = Int(opaque_u32(UInt32(row)))
        var rows_to_store = rows_owned[Self.config](
            decode_head_tile[Self.config]()
        )

        #   0       64     128     192      256      320      384     448     512
        #   |-------|-------|-------|--------|--------|--------|-------|-------|
        #     w0/1    w0/1     w2/3    w2/3     w0/1     w0/1     w2/3    w2/3

        comptime for n in range(0, num_mma_pv):
            comptime for m in range(0, num_out_stages_per_mma):
                out_cons.wait()

                comptime for k in range(0, blocks_per_stage):
                    var stage_ptr = out_cons.stage_base_ptr(k)
                    var col: Int = (
                        n * Self.config.MMA_PV_N
                        + m * (Self.config.BN_PV // 4)
                        + k * col_per_warp
                    )
                    comptime o_elements = (
                        Self.config.out_rows * (Self.config.BN_PV // 4)
                    )
                    comptime o_tt_layout = tt_row_major[o_elements]()
                    comptime if fold_q:
                        # Fold: BM=64 TMEM packs q_len_fold * num_q_heads;
                        # emit one TMA store per q_token.
                        comptime for q_local in range(q_len_fold):
                            var q_stage_ptr = stage_ptr.unsafe_offset(
                                q_local
                                * Self.config.num_q_heads
                                * (Self.config.BN_PV // 4)
                            )
                            var smem_tensor = TileTensor[
                                Self.output_dtype,
                                type_of(o_tt_layout),
                                address_space=.SHARED,
                            ](q_stage_ptr, o_tt_layout)
                            if is_leader:
                                fence_async_view_proxy()
                                o_tma.async_store_3d(
                                    smem_tensor,
                                    store_row_coords[Self.config.out_rows](
                                        col,
                                        offset_position.out_row_offset_at(
                                            q_local
                                        ),
                                        rows_to_store,
                                    ),
                                )
                    else:
                        var smem_tensor = TileTensor[
                            Self.output_dtype,
                            type_of(o_tt_layout),
                            address_space=.SHARED,
                        ](stage_ptr, o_tt_layout)
                        if is_leader:
                            fence_async_view_proxy()
                            o_tma.async_store_3d(
                                smem_tensor,
                                store_row_coords[Self.config.out_rows](
                                    col, row, rows_to_store
                                ),
                            )
                out_cons.release(elect_mask)
        if is_leader:
            o_tma.commit_group()
        o_tma.wait_group[0]()
