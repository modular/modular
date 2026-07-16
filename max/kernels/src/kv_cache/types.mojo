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
This module contains the types for the key-value cache APIs.

The module includes structs implementing several different types of
[KV caches](/glossary/ai/kv-cache).

This module defines two traits that define the roles of the different structs

- `KVCacheT`: Defines the interface for a single (key or value) cache.
- `KVCollectionT`: Defines the interface for a pair of caches (keys and values).
"""

from std.math import align_up
from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapL2Promotion, TensorMapSwizzle
from std.gpu.memory import (
    CacheEviction,
    cp_async_bulk_tensor_shared_cluster_global_elect,
)
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    IntTuple,
    Layout,
    LayoutTensor,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord,
    lt_to_tt,
)
from layout.tma_async import (
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    TMATensorTile,
    _gather4_box_width,
    create_split_tma,
    create_tma_tile_gather4,
)
from layout.tile_layout import RowMajorLayout, Layout as InternalLayout
from layout.coord import DynamicCoord

from std.collections import OptionalReg
from std.utils import Index, IndexList
from std.utils.coord import dyn_coord
from std.sys import size_of
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import ceildiv

from std.gpu import thread_idx


@always_inline
def swizzle_granularity[dtype: DType, swizzle_mode: TensorMapSwizzle]() -> Int:
    comptime sg = swizzle_mode.bytes() // size_of[dtype]()
    return sg


@always_inline
def padded_depth[
    dtype: DType, swizzle_mode: TensorMapSwizzle, depth: Int
]() -> Int:
    comptime padded_depth = align_up(
        depth, swizzle_mode.bytes() // size_of[dtype]()
    )
    return padded_depth


@always_inline
def _compute_kv_cache_dynamic_shape_strides[
    dtype: DType, //, kv_cache_rank: Int, drop_list: Tuple
](blocks: TileTensor[dtype, ...]) -> Tuple[
    IndexList[kv_cache_rank],
    IndexList[kv_cache_rank],
]:
    var kv_cache_shape = IndexList[kv_cache_rank]()
    var kv_cache_strides = IndexList[kv_cache_rank]()
    var out_index = kv_cache_rank - 1
    var stride = 1

    comptime for i in reversed(range(blocks.flat_rank)):
        var dim = Int(blocks.dim[i]())

        # Skip dimensions in the drop list (kv_idx and layer_idx).
        comptime if i not in drop_list:
            kv_cache_shape[out_index] = dim
            kv_cache_strides[out_index] = stride
            out_index = out_index - 1

        stride *= dim

    return (kv_cache_shape, kv_cache_strides)


@always_inline
def _make_cache_tt[
    dtype: DType,
    ResultLayout: TensorLayout,
    rank: Int,
](
    ptr: UnsafePointer[mut=_, Scalar[dtype], _],
    shape: IndexList[rank],
    strides: IndexList[rank],
) -> TileTensor[
    dtype,
    InternalLayout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ],
    ptr.origin,
]:
    """Construct a TileTensor from a pointer and IndexList shape/strides.

    Static dims in ResultLayout are left at their compile-time values;
    dynamic dims are filled from the IndexList arguments.
    """
    comptime ConcLayout = InternalLayout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ]
    var shape_c = Coord[*ConcLayout.shape_types]()
    var stride_c = Coord[*ConcLayout.stride_types]()
    comptime for i in range(rank):
        comptime if not shape_c.element_types[i].is_static_value:
            shape_c[i] = rebind[shape_c.element_types[i]](
                Scalar[DType.int64](shape[i])
            )
        comptime if not stride_c.element_types[i].is_static_value:
            stride_c[i] = rebind[stride_c.element_types[i]](
                Scalar[DType.int64](strides[i])
            )
    return TileTensor[dtype, ConcLayout](
        ptr=ptr, layout=ConcLayout(shape_c, stride_c)
    )


struct KVCacheStaticParams(Equatable, TrivialRegisterPassable):
    var num_heads: Int
    var head_size: Int
    var is_mla: Bool

    def __init__(
        out self, num_heads: Int, head_size: Int, is_mla: Bool = False
    ):
        """
        Initialize KVCacheStaticParams.
        Args:
            num_heads (Int): Number of attention heads.
            head_size (Int): Size of each attention head.
            is_mla (Bool, optional): Whether to use Multi-Linear Attention (MLA) mode.
                If true, we only store k cache. If False, we store k and v cache.
                Defaults to False.
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.is_mla = is_mla


# Explicit 1D TileTensor layout that lets the compiler prove flat_rank == 1,
# bypassing the LTToTTLayout comptime alias chain where the compiler can't
# simplify TypeList[_Flattened[...]].size to 1.
comptime _1d_tt_layout = InternalLayout[
    shape_types=Coord[Int64].element_types,
    stride_types=Coord[ComptimeInt[1]].element_types,
]

comptime _2d_row_major_tt_layout = InternalLayout[
    shape_types=Coord[Int64, Int64].element_types,
    stride_types=Coord[Int64, ComptimeInt[1]].element_types,
]


# ---- Paged KV cache sub-tile helpers ----------------------------------------


def kv_sub_tile_rows(tile_BN: Int, page_size: Int) -> Int:
    """Sub-tile row count for a TMA load of `tile_BN` rows.

    When `page_size` is zero (non-paged) or at least `tile_BN`, returns
    `tile_BN` (no splitting). Otherwise returns `page_size`, so that each
    sub-tile TMA load stays within one page.
    """
    if page_size <= 0 or page_size >= tile_BN:
        return tile_BN
    return page_size


def kv_num_sub_tiles(tile_BN: Int, page_size: Int) -> Int:
    """Number of sub-tile TMA copies needed for `tile_BN` rows."""
    return tile_BN // kv_sub_tile_rows(tile_BN, page_size)


# Swizzle-atom / core-matrix row count. Mirrors `_SWIZZLE_ATOM_ROWS` in
# `layout/tma_async.mojo` and `_CM_NUM_ROWS` in `layout/tensor_core_async.mojo`
# (module-private there): the canonical MMA core matrix is 8 rows tall and the
# SWIZZLE_128B 8-row swizzle tile is exactly one atom. The chunk-inner
# (row-major-atoms) rank-5 fold splits a page's `box_rows` into
# `box_rows / _SWIZZLE_ATOM_ROWS` atom-rows.
comptime _SWIZZLE_ATOM_ROWS = 8


@always_inline
def _kv_fold_base_ok(bk: Int, gran: Int, head_size: Int) -> Bool:
    """Shared SM100 depth-chunk-fold geometry gate.

    Single source of truth for the `base_ok` condition used by BOTH
    `kv_tma_fold_chunks` (comptime) and `FA4Config.{k,v}_row_major()` (the
    runtime config accessors). The fold is well-defined only when the contiguous
    depth `bk` is a whole number of swizzle atoms (`gran`), spans at least two of
    them (something to fold), and tiles the full `head_size` exactly (so the
    folded descriptor's `[head_size // gran, gran]` chunk axis is well-formed).

    Takes plain runtime `Int`s — not comptime params — so the runtime accessors
    can call it (a `def` method cannot feed `self.field` into a comptime param);
    when all args are comptime it folds to a comptime `Bool`."""
    return bk % gran == 0 and bk // gran >= 2 and head_size % bk == 0


@always_inline
def kv_tma_fold_chunks[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    BK: Int,
    head_size: Int,
    box_rows: Int,
    smem_BN: Int,
    page_size: Int,
    row_major: Bool = False,
]() -> Int:
    """Single source of truth for the SM100 depth-chunk TMA-fold predicate.

    When a K/V tile's contiguous depth `BK` spans
    `num_chunks = BK // swizzle_granularity >= 2` (e.g. bf16 `BK=128`,
    `SWIZZLE_128B`, `gran=64` -> 2 chunks), the per-chunk TMA loop in
    `PagedRowIndices._tma_copy_kv_impl` can be replaced by ONE rank-4
    `cp.async.bulk.tensor` that folds the depth-chunk dimension into an extra,
    non-innermost box dim. This returns `num_chunks` (the fold factor) when that
    rewrite is byte-equivalent, and `1` (no fold = current per-chunk behavior)
    otherwise.

    The fold is byte-equivalent to the per-chunk loop ONLY when the folded box's
    per-chunk SMEM stride (`box_rows * gran`) equals the producer chunk stride
    (`smem_BN * gran`) — i.e. `box_rows == smem_BN` — AND the tile occupies a
    single page (`pages_per_iter == 1`, encoded as `page_size == 0` or
    `page_size >= box_rows`). Both conditions are checked here so a caller cannot
    request an illegal fold. The fold is a pure producer-side instruction-count
    rewrite: it writes byte-identical SMEM to the loop, so it is correct for both
    the K-major (K) and mn-major (V) consumers — the caller just supplies the
    side-correct `smem_BN` (K: `k_rows_per_cta`; V: `tile_rows = BN //
    num_v_sub_tiles`).

    The folded rank-4 descriptor reshapes the full gmem `head_size` into a
    `[head_size // gran, gran]` chunk axis, so the fold requires
    `head_size % BK == 0` (which implies `head_size % gran == 0`, the builder's
    requirement, and that each per-stage `BK`-wide window tiles `head_size`
    exactly). This rejects unaligned head dims (e.g. `head_size=127` padded to
    `BK=128`) where the descriptor's chunk axis would be ill-defined.

    Returning the same comptime value to both the descriptor builder and the issue
    site is what keeps the baked descriptor rank and the issue-time coord rank from
    drifting.

    Parameters:
        dtype: The KV element dtype (drives swizzle granularity).
        swizzle_mode: The TMA swizzle mode (drives swizzle granularity).
        BK: The tile's contiguous depth per stage (K: `BK0`; V: `v_cols_per_cta`).
        head_size: The descriptor's full gmem depth (the cache `head_size`); the
            fold's chunk axis spans this, so it must satisfy `head_size % BK == 0`.
        box_rows: The TMA box's row count (`kv_sub_tile_rows(tile_rows, page_size)`).
        smem_BN: The SMEM depth-chunk stride in rows (K: `smem_BN` arg to
            `tma_copy_k`; V: `tile_rows`).
        page_size: KV cache page size (`0` = non-paged).
        row_major: When `True`, predicate the chunk-inner (row-major-atoms) rank-5
            fold, which lets a tile span MULTIPLE pages (one TMA per page). The
            rank-5 descriptor box sets the chunk/atom-row SMEM strides, so the
            chunk-outer fold's `box_rows == smem_BN` and single-page requirements
            do not apply; instead `box_rows` must split into swizzle-atom rows.
            `False` (default) predicates today's chunk-outer rank-4 fold.

    Returns:
        The fold factor: `num_chunks` when foldable, else `1`.
    """
    comptime gran = swizzle_mode.bytes() // size_of[dtype]()
    comptime num_chunks = BK // gran
    comptime pages_per_iter_is_one = page_size == 0 or page_size >= box_rows
    # Shared geometry gate (single source of truth, also used by
    # `FA4Config.{k,v}_row_major()`): BK % gran == 0, >= 2 chunks, head_size % BK.
    comptime base_ok = _kv_fold_base_ok(BK, gran, head_size)
    # The chunk-inner (row_major) rank-5 fold drops the single-page /
    # `box_rows == smem_BN` requirements (its descriptor box sets the SMEM
    # strides) but needs `box_rows` to split into swizzle-atom rows; the default
    # chunk-outer rank-4 fold needs `box_rows == smem_BN` and a single page.
    comptime geometry_ok = (
        box_rows % _SWIZZLE_ATOM_ROWS
        == 0 if row_major else (box_rows == smem_BN and pages_per_iter_is_one)
    )
    comptime if base_ok and geometry_ok:
        return num_chunks
    else:
        return 1


struct PagedRowIndices[
    BN: Int,
    page_size: Int,
    pair_cta: Bool = False,
    is_leader: Bool = True,
](ImplicitlyCopyable):
    """Pre-computed physical row indices for a BN-row range of paged KV cache.

    `BN` is V's tile row count. `MHAOperand.populate` (or its
    `PagedKVCache` override) fills indices for the full `BN` range (so
    V can reuse them); K's TMA (`tma_copy_k`) covers only a subset
    when `pair_cta=True` (the `BN/2` rows owned by this CTA). The K
    half is selected at comptime from `Self.is_leader`: when
    `num_pages >= 2` the peer shifts its index into `rows[]` by
    `num_pages/2`; when `num_pages == 1` (e.g. `page_size >= BN`) the
    peer reuses `rows[0]` but adds `BN/2` to the issued row.

    When `page_size >= BN` (or `page_size == 0` for non-paged), stores a
    single entry — zero overhead compared to a single `row_idx` call.

    Under `pair_cta=True`, K's TMA covers `num_pages // 2` entries
    (the CTA-rank-specific half) when `num_pages >= 2`, or the full
    single entry when `num_pages == 1`; V's TMA covers all `num_pages`.
    Storage is sized to V (`num_pages = BN / eff_page`) regardless of
    `pair_cta` — K populates the full range so V can reuse the rows
    without any lazy LUT lookup.
    """

    comptime eff_page: Int = kv_sub_tile_rows(Self.BN, Self.page_size)
    # One entry per sub-tile page, sized to V's full range so both
    # K and V share the same buffer.
    comptime num_pages: Int = Self.BN // Self.eff_page
    comptime cta_group = 2 if Self.pair_cta else 1

    var rows: InlineArray[UInt32, Self.num_pages]

    @always_inline
    def __init__(out self):
        self.rows = InlineArray[UInt32, Self.num_pages](uninitialized=True)

    @always_inline
    def get_row(self, offset: UInt32) -> UInt32:
        """Physical row for an arbitrary offset within the BN range.

        For sub-tile loads: `get_row(sub_tile_idx * eff_page)`.
        For depth-512 V: `get_row(pv_stage * BK1)` avoids re-reading the LUT.
        Requires the base `kv_row` that was passed to `populate` to be
        page-aligned (guaranteed by mask alignment).
        """
        comptime if Self.num_pages == 1:
            return self.rows[0] + offset
        else:
            return self.rows[Int(offset) // Self.eff_page] + UInt32(
                Int(offset) % Self.eff_page
            )

    @always_inline
    def _tma_copy_kv_impl[
        dtype: DType,
        tile_shape: IndexList[3],
        desc_shape: IndexList[3],
        //,
        *,
        is_k: Bool,
        needs_partial: Bool,
        num_v_sub_tiles: Int = 1,
        v_sub_tile_idx: Int = 0,
        smem_BN: Int = Self.BN,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
        num_iters: Int = -1,
        oob_fill_pages: Bool = False,
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](
        self,
        tma_op: TMATensorTile[dtype, 3, tile_shape, desc_shape, True],
        stage_base: UnsafePointer[
            mut=True, Scalar[dtype], _, address_space=AddressSpace.SHARED
        ],
        ref[AddressSpace.SHARED] mbar: SharedMemBarrier,
        *,
        kv_head_idx: UInt32,
        elect: Int32,
        valid_pages: UInt32,
        depth_offset: UInt32 = 0,
    ):
        """Shared TMA-issue body for `tma_copy_k` and `tma_copy_v`.

        `is_k=True` emits the K-side subset (pair-CTA-aware index/intra-page
        offsets, `smem_BN` depth-chunk stride); `is_k=False` emits the V-side
        sub-tile defined by `num_v_sub_tiles` / `v_sub_tile_idx`.

        `num_v_sub_tiles` and `v_sub_tile_idx` apply only when `is_k=False`;
        `smem_BN` applies only when `is_k=True`. The wrappers always pass
        `valid_pages` (named `num_valid_pages` for V and `k_num_valid_pages`
        for K in their public signatures); it is only consulted when
        `needs_partial=True`.

        `oob_fill_pages` (only consulted when `needs_partial=True`): when
        True, after dispatching the `valid_pages` valid-block TMAs, also
        dispatch deliberately out-of-bounds TMAs for the remaining
        `[valid_pages, pages_per_iter)` page slots. With `OOBFill.NONE`
        (the default for our descriptors — see
        `mojo/stdlib/std/gpu/host/nvidia/tma.mojo:431`), OOB coordinates
        return 0, so the corresponding SMEM rows are zero-initialized.
        This is required by callers whose downstream MMA reads the full
        `pages_per_iter` row range regardless of mask — e.g. depth-512
        FA4's `O += P * V` reads the full BN V-tile so masked rows must
        contain 0 (not stale `+inf`/`NaN` from prior compute) to avoid
        `0 * non-finite = NaN` propagation. Callers opting in MUST set
        `expect_bytes` to the full (non-partial) byte count, since every
        `pages_per_iter * num_depth_chunks` TMA arrives at the mbar.
        """
        comptime swizzle_gran = desc_shape[2]
        comptime num_depth_chunks = ceildiv(tile_shape[2], swizzle_gran)

        comptime tile_rows = (
            (Self.BN // 2 if Self.pair_cta else Self.BN) if is_k else (
                Self.BN // num_v_sub_tiles
            )
        )
        comptime tma_per_issue_rows = kv_sub_tile_rows(
            tile_rows, Self.page_size
        )
        comptime pages_per_iter = tile_rows // tma_per_issue_rows
        comptime effective_iters = (
            pages_per_iter if num_iters == -1 else num_iters
        )
        comptime idx_offset_ct: Int = (
            (
                Self.num_pages // 2 if Self.pair_cta
                and not Self.is_leader
                and Self.num_pages >= 2 else 0
            ) if is_k else (
                v_sub_tile_idx * pages_per_iter if Self.num_pages
                >= num_v_sub_tiles else 0
            )
        )
        comptime intra_page_row_ct: Int = (
            (
                Self.BN // 2 if Self.pair_cta
                and not Self.is_leader
                and Self.num_pages == 1 else 0
            ) if is_k else (
                v_sub_tile_idx * tile_rows if Self.num_pages
                < num_v_sub_tiles else 0
            )
        )
        comptime smem_j_stride_rows = smem_BN if is_k else tile_rows
        comptime dispatch_start = 1 if (is_k and Self.is_leader) else 0

        # Depth-chunk TMA fold (SM100 / B200, K-only). When `fold>=2`, one rank-4
        # `cp.async.bulk.tensor` (built with a rank-4 descriptor by the matching
        # `create_split_tma[..., fold_chunks=fold]` call) replaces the per-chunk
        # `for j` loop. The descriptor's box already spans all `fold` chunks with a
        # per-chunk SMEM stride of `tma_per_issue_rows * gran`, so byte-equivalence
        # requires `tma_per_issue_rows == smem_j_stride_rows` and `pages_per_iter==1`.
        # `fold` must also equal `num_depth_chunks` (the box covers every chunk).
        comptime fold = fold_chunks
        comptime assert fold == 1 or (
            fold == num_depth_chunks
            and (
                # Chunk-inner rank-5 fold: the descriptor box sets the
                # chunk/atom-row SMEM strides, so it lifts the chunk-outer
                # fold's `box_rows == smem_j_stride_rows` + single-page
                # requirements; it needs `box_rows` to split into atom-rows.
                (row_major and tma_per_issue_rows % _SWIZZLE_ATOM_ROWS == 0)
                or (
                    not row_major
                    and tma_per_issue_rows == smem_j_stride_rows
                    and pages_per_iter == 1
                )
            )
        ), (
            "kv TMA fold requires fold == num_depth_chunks; the chunk-outer"
            " (rank-4) fold additionally requires box_rows =="
            " smem_j_stride_rows and pages_per_iter == 1 (the chunk-inner"
            " row_major rank-5 fold lifts both but needs box_rows divisible by"
            " the swizzle-atom row count); a folded descriptor was paired with"
            " an unfoldable issue-site geometry"
        )

        var desc_ptr = UnsafePointer(to=tma_op.descriptor).bitcast[NoneType]()

        comptime if needs_partial:
            # valid_pages is always in [1, pages_per_iter]; the dispatch loop
            # skips _p == 0 (impossible since valid_pages >= 1) and relies on
            # fall-through for _p == pages_per_iter (full count) so each leaf
            # call is a non-partial, straight-line unroll. K-leader skips _p
            # == 0 explicitly (dispatch_start == 1); V and K-peer rely on the
            # `if` check.
            comptime for _p in range(dispatch_start, pages_per_iter):
                if UInt32(_p) == valid_pages:
                    comptime if _p > 0:
                        self._tma_copy_kv_impl[
                            is_k=is_k,
                            needs_partial=False,
                            num_v_sub_tiles=num_v_sub_tiles,
                            v_sub_tile_idx=v_sub_tile_idx,
                            smem_BN=smem_BN,
                            eviction_policy=eviction_policy,
                            num_iters=_p,
                            fold_chunks=fold_chunks,
                            row_major=row_major,
                        ](
                            tma_op,
                            stage_base,
                            mbar,
                            kv_head_idx=kv_head_idx,
                            elect=elect,
                            valid_pages=valid_pages,
                            depth_offset=depth_offset,
                        )
                    comptime if oob_fill_pages:
                        # Issue OOB TMAs for the remaining `[_p,
                        # pages_per_iter)` page slots. The TMA descriptor
                        # is built with `OOBFill.NONE`, which writes 0
                        # for any OOB coordinate; we use a row coord
                        # (`Int32.MAX >> 1`) that is unconditionally
                        # past `globalDim[0]` (block-row count is
                        # bounded by `total_blocks * stride`, well
                        # below 2^30 for any realistic workload). Each
                        # OOB TMA still arrives at `mbar` with its
                        # byte count, so the caller's `expect_bytes`
                        # MUST cover the full
                        # `pages_per_iter * num_depth_chunks` issues.
                        comptime _OOB_ROW: Int = 1 << 30
                        comptime for _q in range(_p, pages_per_iter):
                            comptime if fold >= 2 and row_major:
                                # One rank-5 chunk-inner TMA per OOB page slot.
                                # Page-outer SMEM base spans the full per-page
                                # chunk-inner block (num_depth_chunks *
                                # tma_per_issue_rows * gran). Coord is fast-first
                                # (gran, in-atom-row, chunk-base, atom_row, head);
                                # atom_row = _OOB_ROW (>> globalDim atom-row extent)
                                # so OOBFill.NONE zero-fills this page slot.
                                comptime smem_off_oob_rm = (
                                    _q
                                    * num_depth_chunks
                                    * tma_per_issue_rows
                                    * swizzle_gran
                                )
                                cp_async_bulk_tensor_shared_cluster_global_elect[
                                    cta_group=Self.cta_group,
                                    eviction_policy=eviction_policy,
                                ](
                                    stage_base + smem_off_oob_rm,
                                    desc_ptr,
                                    mbar.unsafe_ptr(),
                                    Index(
                                        0,
                                        0,
                                        Int(depth_offset) // swizzle_gran,
                                        _OOB_ROW,
                                        Int(kv_head_idx),
                                    ),
                                    elect,
                                )
                            elif fold >= 2:
                                # One rank-4 TMA folds all `fold` depth chunks.
                                # SMEM base for this page slot (chunk dim is the
                                # box's slowest dim, stride tma_per_issue_rows*gran
                                # == smem_j_stride_rows*gran). Coord is fast-first
                                # (gran, head, row, chunk); chunk-base =
                                # depth_offset // gran selects this stage's window
                                # over the full-head_size chunk axis, gran coord 0.
                                comptime smem_off_oob_f = (
                                    _q * tma_per_issue_rows * swizzle_gran
                                )
                                cp_async_bulk_tensor_shared_cluster_global_elect[
                                    cta_group=Self.cta_group,
                                    eviction_policy=eviction_policy,
                                ](
                                    stage_base + smem_off_oob_f,
                                    desc_ptr,
                                    mbar.unsafe_ptr(),
                                    Index(
                                        0,
                                        _OOB_ROW,
                                        Int(depth_offset) // swizzle_gran,
                                        Int(kv_head_idx),
                                    ),
                                    elect,
                                )
                            else:
                                comptime for j in range(num_depth_chunks):
                                    comptime smem_off_oob = (
                                        j * smem_j_stride_rows * swizzle_gran
                                        + _q * tma_per_issue_rows * swizzle_gran
                                    )
                                    cp_async_bulk_tensor_shared_cluster_global_elect[
                                        cta_group=Self.cta_group,
                                        eviction_policy=eviction_policy,
                                    ](
                                        stage_base + smem_off_oob,
                                        desc_ptr,
                                        mbar.unsafe_ptr(),
                                        Index(
                                            Int(depth_offset)
                                            + j * swizzle_gran,
                                            Int(kv_head_idx),
                                            _OOB_ROW,
                                        ),
                                        elect,
                                    )
                    return
        comptime for _p in range(effective_iters):
            comptime src_idx = idx_offset_ct + _p
            comptime if fold >= 2 and row_major:
                # One rank-5 chunk-inner TMA writes this whole multi-atom-row
                # page in chunk-inner SMEM order (off(ar,c) =
                # ar*num_chunks*CM*gran + c*CM*gran). Page-outer SMEM base spans
                # the full per-page chunk-inner block (num_depth_chunks *
                # tma_per_issue_rows * gran). Coord is fast-first
                # (gran, in-atom-row, chunk-base, atom_row, head): atom_row =
                # row // CM (row is CM-aligned by page alignment), the box covers
                # all CM rows of each atom-row and `fold` chunks, and chunk-base =
                # depth_offset // gran selects this stage's window over the
                # full-head_size chunk axis. Validated by
                # test_kv_rowmajor_fold_spike.mojo.
                comptime smem_off_rm = (
                    _p * num_depth_chunks * tma_per_issue_rows * swizzle_gran
                )
                var row_rm = Int(self.rows[src_idx]) + intra_page_row_ct
                debug_assert(
                    row_rm % _SWIZZLE_ATOM_ROWS == 0,
                    (
                        "row_major fold: page row must be swizzle-atom-aligned"
                        " for the rank-5 atom-row coordinate"
                    ),
                )
                cp_async_bulk_tensor_shared_cluster_global_elect[
                    cta_group=Self.cta_group,
                    eviction_policy=eviction_policy,
                ](
                    stage_base + smem_off_rm,
                    desc_ptr,
                    mbar.unsafe_ptr(),
                    Index(
                        0,
                        0,
                        Int(depth_offset) // swizzle_gran,
                        row_rm // _SWIZZLE_ATOM_ROWS,
                        Int(kv_head_idx),
                    ),
                    elect,
                )
            elif fold >= 2:
                # One rank-4 TMA folds all `fold` depth chunks for this page.
                # SMEM base = _p * tma_per_issue_rows * gran (chunk dim is the
                # box's slowest dim with stride tma_per_issue_rows*gran ==
                # smem_j_stride_rows*gran). Coord is fast-first
                # (gran, head, row, chunk): the descriptor's chunk axis (stride
                # gran) spans the full head_size, so this stage's window is
                # selected by chunk-base = depth_offset // gran while the box
                # covers `fold` chunks; the gran coord is 0.
                comptime smem_off_f = (_p * tma_per_issue_rows * swizzle_gran)
                cp_async_bulk_tensor_shared_cluster_global_elect[
                    cta_group=Self.cta_group,
                    eviction_policy=eviction_policy,
                ](
                    stage_base + smem_off_f,
                    desc_ptr,
                    mbar.unsafe_ptr(),
                    Index(
                        0,
                        Int(self.rows[src_idx]) + intra_page_row_ct,
                        Int(depth_offset) // swizzle_gran,
                        Int(kv_head_idx),
                    ),
                    elect,
                )
            else:
                comptime for j in range(num_depth_chunks):
                    comptime smem_off = (
                        j * smem_j_stride_rows * swizzle_gran
                        + _p * tma_per_issue_rows * swizzle_gran
                    )
                    cp_async_bulk_tensor_shared_cluster_global_elect[
                        cta_group=Self.cta_group,
                        eviction_policy=eviction_policy,
                    ](
                        stage_base + smem_off,
                        desc_ptr,
                        mbar.unsafe_ptr(),
                        Index(
                            Int(depth_offset) + j * swizzle_gran,
                            Int(kv_head_idx),
                            Int(self.rows[src_idx]) + intra_page_row_ct,
                        ),
                        elect,
                    )

    @always_inline
    def tma_copy_v[
        dtype: DType,
        tile_shape: IndexList[3],
        desc_shape: IndexList[3],
        //,
        *,
        needs_partial: Bool,
        num_v_sub_tiles: Int = 1,
        v_sub_tile_idx: Int = 0,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
        num_iters: Int = -1,
        oob_fill_pages: Bool = False,
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](
        self,
        tma_op: TMATensorTile[dtype, 3, tile_shape, desc_shape, True],
        stage_base: UnsafePointer[
            mut=True, Scalar[dtype], _, address_space=AddressSpace.SHARED
        ],
        ref[AddressSpace.SHARED] mbar: SharedMemBarrier,
        *,
        kv_head_idx: UInt32,
        elect: Int32,
        num_valid_pages: UInt32 = UInt32(Self.num_pages // num_v_sub_tiles),
        depth_offset: UInt32 = 0,
    ):
        """TMA-copy a V sub-tile, with comptime partial switch.

        Consumes pre-populated rows from an earlier `MHAOperand.populate`
        call. In pair_cta mode, that call populates the full `num_pages`
        range (both CTAs' halves) so V can reuse them directly without
        any lazy LUT lookup.

        `num_v_sub_tiles` / `v_sub_tile_idx` select a row sub-range of
        the BN tile when V is split across multiple SMEM slots (e.g.
        depth512's `num_pv_stages=2` split: `BK1 = BN/2` rows per
        slot). Default `(1, 0)` loads the full `Self.BN` rows into a
        single SMEM slot of row stride `Self.BN` — byte-identical to
        fa4's previous behavior.

        With `num_v_sub_tiles > 1`:
        - `v_rows_per_sub_tile = Self.BN // num_v_sub_tiles` is the
          SMEM depth-chunk stride (rows per slot).
        - `v_tma_tile_rows = kv_sub_tile_rows(v_rows_per_sub_tile,
          Self.page_size)` is the TMA's tile-row count per issue.
        - When `Self.num_pages >= num_v_sub_tiles`: sub-tile `s`
          loads `rows[s * v_pages_per_sub_tile .. )`.
        - When `Self.num_pages == 1 < num_v_sub_tiles` (page covers
          the full BN): all sub-tiles share `rows[0]` and add
          `v_sub_tile_idx * v_rows_per_sub_tile` as intra-page row
          offset.

        `needs_partial=False` — comptime-unrolled over `num_iters`
        sub-tile entries (default `v_pages_per_sub_tile`).

        `needs_partial=True` — comptime-unrolls a runtime dispatch that
        tests `num_valid_pages` against each `_p in [1,
        v_pages_per_sub_tile)` and tail-calls the `needs_partial=False`
        form with `num_iters=_p` so the actual TMA issues always emit
        as a straight-line, fully static unroll of exactly
        `num_valid_pages` issues. Callers must guarantee
        `1 <= num_valid_pages <= v_pages_per_sub_tile`.

        `num_iters` is an internal dispatch knob: `-1` (default) means
        "unroll `v_pages_per_sub_tile` iterations"; any other value
        fully unrolls exactly that many. Only the `needs_partial=True`
        wrapper sets it, when it recurses.

        `oob_fill_pages` (consulted only when `needs_partial=True`):
        when True, after dispatching the `num_valid_pages` valid TMAs,
        also issue OOB TMAs for the remaining
        `[num_valid_pages, v_pages_per_sub_tile)` page slots. The TMA
        descriptor's `OOBFill.NONE` policy zero-fills SMEM for OOB
        coordinates, ensuring the full V-tile region holds finite (0)
        data — required by depth-512 FA4 whose `O += P * V` reads the
        full BN V-tile and would otherwise propagate
        `0 * non-finite = NaN` from uninitialized SMEM (the bug only
        materializes when this is the very first write to the SMEM
        slot — typically `seq_len <= BN` so the only iter is partial).
        Callers opting in MUST predicate `expect_bytes` on the full
        (non-partial) byte count; every
        `v_pages_per_sub_tile * num_depth_chunks` TMA arrives at the
        mbar.

        `fold_chunks` (default `1` = no fold = per-chunk loop) folds the
        `num_depth_chunks` depth columns into ONE rank-4 `cp.async.bulk.tensor`
        when `>= 2`. The caller MUST pass the value returned by
        `kv_tma_fold_chunks` (with V's geometry: `BK=v_cols_per_cta`,
        `box_rows=kv_sub_tile_rows(tile_rows, page_size)`, `smem_BN=tile_rows`
        where `tile_rows = BN // num_v_sub_tiles`) AND build `v_tma_op` with the
        matching `create_split_tma[..., fold_chunks=...]` so the baked descriptor
        rank and the issue-time coord rank agree; a comptime backstop assert in
        `_tma_copy_kv_impl` rejects a fold paired with an unfoldable geometry.
        The fold is a producer-side rewrite that writes byte-identical SMEM, so
        it is correct for V's mn-major consumer.

        `elect` is the raw `Int32` returned by `elect()`. Each
        `cp_async_bulk_tensor_shared_cluster_global_elect` call predicates
        its TMA issue in-PTX on `elect`, so no Mojo-level `if elect != 0:`
        branch is needed here — all lanes follow the same PTX control
        flow and only the elected lane actually issues the TMA.
        """
        self._tma_copy_kv_impl[
            is_k=False,
            needs_partial=needs_partial,
            num_v_sub_tiles=num_v_sub_tiles,
            v_sub_tile_idx=v_sub_tile_idx,
            eviction_policy=eviction_policy,
            num_iters=num_iters,
            oob_fill_pages=oob_fill_pages,
            fold_chunks=fold_chunks,
            row_major=row_major,
        ](
            tma_op,
            stage_base,
            mbar,
            kv_head_idx=kv_head_idx,
            elect=elect,
            valid_pages=num_valid_pages,
            depth_offset=depth_offset,
        )

    @always_inline
    def tma_copy_k[
        dtype: DType,
        tile_shape: IndexList[3],
        desc_shape: IndexList[3],
        //,
        *,
        needs_partial: Bool,
        smem_BN: Int = Self.BN,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
        num_iters: Int = -1,
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](
        self,
        tma_op: TMATensorTile[dtype, 3, tile_shape, desc_shape, True],
        stage_base: UnsafePointer[
            mut=True, Scalar[dtype], _, address_space=AddressSpace.SHARED
        ],
        ref[AddressSpace.SHARED] mbar: SharedMemBarrier,
        *,
        kv_head_idx: UInt32,
        elect: Int32,
        k_num_valid_pages: UInt32 = UInt32(
            Self.num_pages // 2 if Self.pair_cta else Self.num_pages
        ),
        depth_offset: UInt32 = 0,
    ):
        """TMA-copy K-side rows into scattered smem positions.

        K counterpart to `tma_copy_v`. Loops over
        `k_pages_per_cta = num_pages // 2 if pair_cta else num_pages`
        entries, using `self.rows[k_idx_offset_ct + _p_k]` as the source
        row (the index offset is comptime-derived from `Self.is_leader`
        and `Self.pair_cta`). Smem destination packs the K subset into
        the first `k_pages_per_cta` page slots.

        Non-pair-CTA / pair-CTA leader load from entry 0 with no
        intra-page offset; pair-CTA peer with `num_pages >= 2` shifts
        the entry index by `num_pages/2`; pair-CTA peer with
        `num_pages == 1` reuses `rows[0]` but adds `BN/2` to the issued
        row so it covers the second half of the single page.

        `smem_BN` controls the depth-chunk stride: depth-chunk stride
        is `smem_BN * swizzle_gran`. Defaults to `Self.BN` (fa4 layout);
        depth512 passes `Self.BN // 2 = BK1`.

        `fold_chunks` (default `1` = no fold = per-chunk loop) folds the
        `num_depth_chunks` depth chunks into ONE rank-4 `cp.async.bulk.tensor`
        when `>= 2`. The caller MUST pass the value returned by
        `kv_tma_fold_chunks` AND build `k_tma_op` with the matching
        `create_split_tma[..., fold_chunks=...]` so the baked descriptor rank
        and the issue-time coord rank agree; a comptime backstop assert in
        `_tma_copy_kv_impl` rejects a fold paired with an unfoldable geometry.

        `needs_partial=False` — comptime-unrolled over `num_iters`
        entries (default `k_pages_per_cta`); `k_num_valid_pages` is
        unused.

        `needs_partial=True` — comptime-unrolls a runtime dispatch that
        tests `k_num_valid_pages` against each `_p_k in [1,
        k_pages_per_cta)` and tail-calls the `needs_partial=False`
        form with `num_iters=_p_k` so the actual TMA issues always
        emit as a straight-line, fully static unroll of exactly
        `k_num_valid_pages` issues. Callers must guarantee
        `1 <= k_num_valid_pages <= k_pages_per_cta`.

        `num_iters` is an internal dispatch knob: `-1` (default) means
        "unroll `k_pages_per_cta` iterations"; any other value fully
        unrolls exactly that many. Only the `needs_partial=True`
        wrapper sets it, when it recurses.

        In non-pair_cta mode, `k_pages_per_cta == num_pages` and the
        comptime offsets are zero — full-range behavior.

        `elect` is the raw `Int32` returned by `elect()`. Each
        `cp_async_bulk_tensor_shared_cluster_global_elect` call predicates
        its TMA issue in-PTX on `elect`, so no Mojo-level `if elect != 0:`
        branch is needed — all lanes follow the same PTX control flow and
        only the elected lane actually issues the TMA.
        """
        self._tma_copy_kv_impl[
            is_k=True,
            needs_partial=needs_partial,
            smem_BN=smem_BN,
            eviction_policy=eviction_policy,
            num_iters=num_iters,
            fold_chunks=fold_chunks,
            row_major=row_major,
        ](
            tma_op,
            stage_base,
            mbar,
            kv_head_idx=kv_head_idx,
            elect=elect,
            valid_pages=k_num_valid_pages,
            depth_offset=depth_offset,
        )


@always_inline
def _populate_via_row_idx[
    BN: Int,
    page_size: Int,
    pair_cta: Bool,
    is_leader: Bool,
    row_idx_fn: def(UInt32, UInt32) capturing -> UInt32,
](batch_idx: UInt32, base_kv_row: UInt32) -> PagedRowIndices[
    BN, page_size, pair_cta, is_leader
]:
    """Scalar-loop fallback shared by `MHAOperand.populate` and
    `KVCacheT.populate`. Calls `row_idx_fn` once per sub-tile page,
    populating the full `num_pages` range so V (and pair-CTA peers) can
    consume it without any lazy LUT lookup. The `PagedKVCache` override
    replaces `populate` with a SIMD LUT load and does not call this
    helper.
    """
    comptime Result = PagedRowIndices[BN, page_size, pair_cta, is_leader]
    var result = Result()
    comptime for i in range(Result.num_pages):
        result.rows[i] = row_idx_fn(
            batch_idx, base_kv_row + UInt32(i * Result.eff_page)
        )
    return result


trait KVCacheT(DevicePassable, TrivialRegisterPassable):
    """Trait for different KVCache types and implementations.

    Represents a single (key or value) cache.
    """

    comptime dtype: DType
    comptime kv_params: KVCacheStaticParams
    comptime page_size_: Int
    comptime scale_dtype: DType = DType.invalid
    comptime quantization_enabled: Bool = False
    comptime quantization_granularity: Int = 1

    def cache_lengths_nd(
        self,
    ) -> TileTensor[DType.uint32, _1d_tt_layout, ImmutAnyOrigin,]:
        """Returns the cache lengths as a TileTensor."""
        ...

    def cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    def load[
        width: Int,
        output_dtype: DType = Self.dtype,
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        output_dtype, width
    ]:
        """Loads an element from the given index."""
        ...

    def store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        """Stores an element at the given index."""
        ...

    def store_scale(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        scales: SIMD[Self.scale_dtype, ...],
    ):
        """Stores the quantization scales at the given index."""
        ...

    def load_scale[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.scale_dtype, width
    ]:
        """Loads the quantization scales from the given index."""
        ...

    def load_quantized[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.dtype, width
    ]:
        """Loads a quantized element from the given index."""
        ...

    def empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        ...

    def max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        ...

    def max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        ...

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns a pointer to the KVCache block at the given index.

        Paged KVCache implementations must have a block_size which is a multiple of the
        and greater than the layout's first dimension.
        """
        ...

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a pointer to the scales block at the requested indices."""
        ...

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns the base pointer to the scales tensor.

        For PagedKVCache with quantization enabled, this returns the raw
        base pointer of the scales TileTensor. For caches without
        quantization, returns a null pointer.
        """
        ...

    @staticmethod
    def max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        ...

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in this KV cache view.

        For paged caches this accounts for the paging stride:
        ``(total_blocks - 1) * stride + page_size``.
        """
        ...

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        ...

    @always_inline
    def populate[
        BN: Int,
        base_alignment: Int,
        pair_cta: Bool = False,
        is_leader: Bool = True,
    ](self, batch_idx: UInt32, base_kv_row: UInt32) -> PagedRowIndices[
        BN, Self.page_size_, pair_cta, is_leader
    ]:
        """Populate a full `PagedRowIndices[BN, ...]` for a BN-row tile.

        `base_alignment` is a comptime promise that
        `base_kv_row % base_alignment == 0` at runtime — typically
        `mask.start_column_alignment[...]()`. The `PagedKVCache`
        override uses it to pick the largest legal SIMD chunk for its
        LUT vector load and to skip the intra-page divmod when
        `base_alignment % page_size == 0`.

        Default: scalar loop over `num_pages` calls to `row_idx`. The
        `PagedKVCache` override replaces this with a single aligned
        SIMD load against the lookup table.
        """

        @parameter
        def _row(batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
            return self.row_idx(batch_idx, start_tok_idx)

        return _populate_via_row_idx[
            BN, Self.page_size_, pair_cta, is_leader, _row
        ](batch_idx, base_kv_row)

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        For paged caches the encoded index is
        ``physical_block * page_size + offset`` and this method returns
        ``physical_block * stride + offset``.  Non-paged caches return
        the encoded index unchanged.
        """
        ...

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Self.kv_params.head_size
        ](),
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        Self.dtype,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a TMA tile for this KV cache.
        This is useful for `k-major` MMA operations where we don't
        need to mask any extra rows.

        `fold_chunks >= 2` builds a depth-chunk-folded descriptor (SM100); `1`
        (default) keeps the original 3D descriptor. `row_major=True` (with
        `fold_chunks >= 2`) builds the rank-5 chunk-inner box (one TMA per
        multi-atom-row page); `False` builds the rank-4 chunk-outer box."""
        ...

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        DType.bfloat16,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a BF16 TMA tile for the rope portion of the KV cache.

        For the per-tensor rope-aware layout, each token row in the KV cache is
        stored as `padded_depth` FP8 bytes (content) followed by `BK` BF16
        elements (rope). This method returns a TMA descriptor that points at
        the rope data starting at byte offset `padded_depth` within each row,
        reinterpreted as BF16.
        """
        ...

    @always_inline
    def create_gather4_tma_tile[
        *,
        tile_height: Int = 4,
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        tma_dtype,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a 2D TMA gather4 descriptor for this KV cache.

        The descriptor views the KV cache as a flat 2D matrix of
        ``[num_kv_rows, tile_width]`` and is configured for gather4 operations
        that load 4 non-contiguous rows per TMA instruction. The box width
        is derived from the swizzle mode; for SWIZZLE_NONE it equals
        ``tile_width``.

        The ``tile_height`` parameter records the full tile height (e.g. 64
        rows) in the returned ``TMATensorTile.tile_shape``. The hardware
        descriptor shape stays ``(1, box_width)`` as required by TMA gather4.

        When ``tma_dtype`` differs from ``Self.dtype``, the underlying data
        pointer is bitcast to ``tma_dtype`` at descriptor creation time.
        This allows, for example, creating an INT64/SWIZZLE_NONE descriptor
        over FP8 data for linear SMEM layout.

        Parameters:
            tile_height: Number of rows in the tile. Must be a multiple of 4.
                Defaults to 4 for backward compatibility.
            tile_width: Number of elements per row to load (box width) in
                ``tma_dtype`` elements.
            tile_stride: Row stride in elements in global memory. Defaults to
                ``tile_width``. Use a larger value when the global row is
                wider than the portion to load.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to SWIZZLE_NONE.
            tma_dtype: The data type used for the TMA descriptor. Defaults to
                ``Self.dtype``. When different, the pointer is bitcast.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A TMATensorTile with box width derived from the swizzle mode.
        """
        ...

    @always_inline
    def create_rope_gather4_tma_tile[
        *,
        tile_height: Int = 4,
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a BF16 gather4 TMA descriptor for the rope portion of the
        KV cache.

        For the per-tensor rope-aware layout each token row is stored as
        ``padded_depth`` FP8 bytes (content) followed by BF16 rope elements.
        This method offsets the base pointer by ``padded_depth`` bytes,
        reinterprets as BF16, and creates a gather4 TMA descriptor with
        ``tile_width`` BF16 elements per row.

        Parameters:
            tile_height: Number of rows in the tile. Must be a multiple of 4.
            tile_width: Number of BF16 elements per row in global memory.
            padded_depth: Byte offset from row start to the rope data.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A BF16 TMATensorTile configured for gather4.
        """
        ...


struct ContinuousBatchingKVCache[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    blocks_origin: MutOrigin,
    cache_lengths_origin: ImmutOrigin,
    lookup_table_origin: ImmutOrigin,
](KVCacheT, TrivialRegisterPassable):
    """Wrapper for the ContinuousKVCache of a given layer in the transformer
    model.

    Parameters:
        dtype_: The dtype of the kv-cache.
        kv_params_: The kv-cache static parameters.
        blocks_origin: Origin of the KV cache blocks buffer.
        cache_lengths_origin: Origin of the cache lengths buffer.
        lookup_table_origin: Origin of the lookup table buffer.

    This abstracts the Pointer indirection for accessing the ContinuousKVCache
    for a given batch entry.

    THIS IS THE TYPE THAT IS PASSED TO KV PROJECTION AND FLASH ATTENTION
    KERNELS.
    """

    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime page_size_ = 0
    # Note: quantization not supported for `ContinuousBatchingKVCache`.
    comptime scale_dtype = DType.float32
    comptime quantization_granularity = 1
    # Shape is [num_blocks, max_seq_len, num_heads, head_size].
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        Self.kv_params.num_heads,
        Self.kv_params.head_size,
    )
    comptime blocks_layout = Layout.row_major(Self.blocks_shape)

    # Direct TileTensor layout for `blocks_shape` (row-major): leading two dims
    # are runtime (Int64), inner dims are static. stride[0] is runtime because
    # it folds in the runtime second dim.
    comptime blocks_tt_layout = InternalLayout[
        shape_types=Coord[
            Int64,
            Int64,
            ComptimeInt[Self.kv_params.num_heads],
            ComptimeInt[Self.kv_params.head_size],
        ].element_types,
        stride_types=Coord[
            Int64,
            ComptimeInt[Self.kv_params.num_heads * Self.kv_params.head_size],
            ComptimeInt[Self.kv_params.head_size],
            ComptimeInt[1],
        ].element_types,
    ]
    comptime blocks_tt_type = TileTensor[
        Self.dtype, Self.blocks_tt_layout, Self.blocks_origin
    ]

    comptime cache_lengths_tt_layout = _1d_tt_layout
    comptime cache_lengths_tt_type = TileTensor[
        DType.uint32, Self.cache_lengths_tt_layout, Self.cache_lengths_origin
    ]

    comptime lookup_table_tt_layout = _1d_tt_layout
    comptime lookup_table_tt_type = TileTensor[
        DType.uint32, Self.lookup_table_tt_layout, Self.lookup_table_origin
    ]

    var blocks: Self.blocks_tt_type
    var cache_lengths: Self.cache_lengths_tt_type
    var lookup_table: Self.lookup_table_tt_type

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ContinuousBatchingKVCache"

    @always_inline
    def _get_idx_tuple(
        self, block_idx: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> DynamicCoord[DType.int64, 4]:
        assert (
            head_idx < Self.kv_params.num_heads
        ), "KVCache head_idx out of range"
        assert (
            head_dim_idx < Self.kv_params.head_size
        ), "KVCache head_dim_idx is out of range"
        assert tok_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"
        return dyn_coord[DType.int64](
            (
                block_idx,
                tok_idx,
                head_idx,
                head_dim_idx,
            )
        )

    @staticmethod
    def max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return -1

    def __init__(
        out self,
        blocks: Self.blocks_tt_type,
        cache_lengths: Self.cache_lengths_tt_type,
        lookup_table: Self.lookup_table_tt_type,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        comptime assert (
            not self.quantization_enabled
        ), "ContinuousBatchingKVCache does not support quantization"
        assert (
            Int(blocks.dim[2]()) == Self.kv_params.num_heads
        ), "blocks.dim[2]() must be equal to kv_params.num_heads"
        assert (
            Int(blocks.dim[3]()) == Self.kv_params.head_size
        ), "blocks.dim[3]() must be equal to kv_params.head_size"

        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length

    @always_inline
    def _batch_size(self) -> Int:
        return Int(self.cache_lengths.dim[0]())

    @always_inline
    def cache_lengths_nd(self) -> Self.cache_lengths_tt_type:
        return self.cache_lengths

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        assert (
            batch_idx < self._batch_size()
        ), "KVCache batch_idx is out of bounds"
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    def load[
        width: Int,
        output_dtype: DType = Self.dtype,
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        output_dtype, width
    ]:
        assert bs < self._batch_size(), "KVCache::load batch_size out of range"

        var block_idx = self.lookup_table[bs]
        var idx = self._get_idx_tuple(
            Int(block_idx), head_idx, tok_idx, head_dim_idx
        )
        # Bypass TileTensor.load's `where` constraint by using ptr directly.
        return self.blocks.load[width=width](idx).cast[output_dtype]()

    @always_inline
    def store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        assert bs < self._batch_size(), "KVCache::store batch_size out of range"
        var block_idx = self.lookup_table[bs]
        var idx = self._get_idx_tuple(
            Int(block_idx), head_idx, tok_idx, head_dim_idx
        )
        # Bypass TileTensor.store's `where` constraint by using ptr directly.
        self.blocks.store(idx, val)

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.scale_dtype, width
    ]:
        """Loads a quantization scale from the given index.

        Note: ContinuousBatchingKVCache does not support KVCache quantization.
        """
        return SIMD[Self.scale_dtype, width](0)

    @always_inline
    def store_scale(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        scales: SIMD[Self.scale_dtype, ...],
    ):
        """Stores the quantization scales at the given index.

        Note: ContinuousBatchingKVCache does not support KVCache quantization.
        """
        ...

    @always_inline
    def load_quantized[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.dtype, width
    ]:
        """Loads a quantized element from the given index.

        Note: ContinuousBatchingKVCache does not support KVCache quantization.
        """
        return SIMD[Self.dtype, width](0)

    def empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        return self.max_cache_length == 0

    def max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        return self.max_seq_length

    def max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        return self.max_cache_length

    @always_inline
    def _stride(self) -> UInt32:
        return UInt32(self.blocks.layout.stride[0]().value()) // UInt32(
            self.kv_params.num_heads * self.kv_params.head_size
        )

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        For non-paged caches the encoded index is already the row, so
        this is an identity operation.
        """
        return encoded_index

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in this KV cache view."""
        var total_blocks = self.blocks.dim[0]()
        return Int(
            UInt32(total_blocks - 1) * self._stride()
            + UInt32(self.blocks.dim[1]())
        )

    @always_inline
    def row_idx(self, batch_idx: UInt32, tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        block_idx = self.lookup_table[Int(batch_idx)]
        return block_idx * self._stride() + tok_idx

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Self.kv_params.head_size
        ](),
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        Self.dtype,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a TMA tile for this KV cache."""
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0, "BK must be a multiple of swizzle granularity"
        # The continuous cache is laid out as [num_blocks, num_layers, seq_len, num_heads, head_size]
        # We create a view of the data as a flattened 2D tensor
        var total_blocks = Int(self.blocks.dim[0]())
        # An axis's size is 1 + maximum valid idx
        # Idx calc is:
        # block_idx * self._stride() + tok_idx
        # max values
        # (total_blocks - 1) * self._stride() + self.blocks.dim[1]() - 1
        # yields number of rows:
        # (total_blocks - 1) * self._stride() + self.blocks.dim[1]()
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            self.blocks.dim[1]()
        )

        comptime smem_dim = IndexList[3](BN, 1, BK)
        comptime gmem_dim = IndexList[3](
            UNKNOWN_VALUE,
            Self.kv_params.num_heads,
            Self.kv_params.head_size,
        )
        return create_split_tma[
            smem_dim,
            gmem_dim,
            swizzle_mode,
            fold_chunks=fold_chunks,
            row_major=row_major,
        ](ctx, self.blocks.ptr, Int(rows))

    @always_inline
    def create_gather4_tma_tile[
        *,
        tile_height: Int = 4,
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        tma_dtype,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a 2D TMA gather4 descriptor for this KV cache.

        The descriptor views the KV cache as a flat 2D matrix of
        ``[num_kv_rows, tile_width]`` and is configured for gather4 operations
        that load 4 non-contiguous rows per TMA instruction. The box width
        is derived from the swizzle mode; for SWIZZLE_NONE it equals
        ``tile_width``.

        When ``tma_dtype`` differs from ``Self.dtype``, the underlying data
        pointer is bitcast to ``tma_dtype`` at descriptor creation time.

        Parameters:
            tile_height: Number of rows in the tile. Must be a multiple of 4.
                Defaults to 4 for backward compatibility.
            tile_width: Number of elements per row to load (box width) in
                ``tma_dtype`` elements.
            tile_stride: Row stride in elements in global memory. Defaults to
                ``tile_width``. Use a larger value when the global row is
                wider than the portion to load.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to SWIZZLE_NONE.
            tma_dtype: The data type used for the TMA descriptor. Defaults to
                ``Self.dtype``. When different, the pointer is bitcast.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A TMATensorTile with box width derived from the swizzle mode.
        """
        return create_tma_tile_gather4[
            tma_dtype,
            tile_height=tile_height,
            tile_width=tile_width,
            tile_stride=tile_stride,
            swizzle_mode=swizzle_mode,
            l2_promotion=l2_promotion,
        ](
            ctx,
            self.blocks.ptr.bitcast[Scalar[tma_dtype]](),
            self.num_kv_rows(),
        )

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Not supported for ContinuousBatchingKVCache."""
        comptime assert (
            False
        ), "create_rope_tma_tile is not supported for ContinuousBatchingKVCache"

    @always_inline
    def create_rope_gather4_tma_tile[
        *,
        tile_height: Int = 4,
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
        ),
    ]:
        """Not supported for ContinuousBatchingKVCache."""
        comptime assert False, (
            "create_rope_gather4_tma_tile is not supported for"
            " ContinuousBatchingKVCache"
        )

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        var block_idx = Int(self.lookup_table[batch_idx])
        var full_block_idx = self._get_idx_tuple(
            block_idx, head_idx, start_tok_idx, head_dim_idx
        )
        var offset_ptr = self.blocks.ptr + Int(
            self.blocks.layout(full_block_idx)
        )
        return offset_ptr.as_unsafe_any_origin()

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a pointer to the scales block at the requested indices.

        Note: ContinuousBatchingKVCache does not support KVCache quantization.
        This function returns a dangling pointer.
        """
        # SAFETY: Callers only dereference scales pointers behind comptime
        # `quantization_enabled` guards, which are False for this cache type.
        return UnsafePointer[
            Scalar[Self.scale_dtype], MutAnyOrigin
        ].unsafe_dangling()

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a dangling pointer. ContinuousBatchingKVCache does not support
        quantization."""
        # SAFETY: Callers only dereference scales pointers behind comptime
        # `quantization_enabled` guards, which are False for this cache type.
        return UnsafePointer[
            Scalar[Self.scale_dtype], MutAnyOrigin
        ].unsafe_dangling()


struct PagedKVCache[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
    blocks_origin: MutOrigin,
    cache_lengths_origin: ImmutOrigin,
    lookup_table_origin: ImmutOrigin,
    scales_origin: MutOrigin,
    *,
    scale_dtype_: DType = DType.invalid,
    quantization_granularity_: Int = 1,
](KVCacheT, TrivialRegisterPassable):
    """The PagedKVCache is a wrapper around the KVCache blocks for a given layer.
    It is used to access the KVCache blocks for PagedAttention.

    Note: This struct represents a 4D view of a 6D `PagedKVCacheCollection`
    tensor. The compile-time layout has `UNKNOWN_VALUE` for stride[0] because
    the actual stride depends on `num_layers` from the parent tensor, which is
    only known at runtime. This ensures offset calculations use the correct
    runtime strides rather than incorrect compile-time values.

    Parameters:
        dtype_: The dtype of the kv-cache.
        kv_params_: The kv-cache static parameters.
        page_size: The size of the page.
        blocks_origin: Origin of the KV cache blocks buffer.
        cache_lengths_origin: Origin of the cache lengths buffer.
        lookup_table_origin: Origin of the lookup table buffer.
        scales_origin: Origin of the quantization scales buffer.
        scale_dtype_: Dtype of the quantization scales (if quantization enabled).
        quantization_granularity_:  Block size used for quantization (e.g. 128).
    """

    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime page_size_ = Self.page_size
    comptime scale_dtype = Self.scale_dtype_
    comptime quantization_enabled = Self.scale_dtype_ != DType.invalid
    comptime quantization_granularity = Self.quantization_granularity_

    # Shape is [total_num_blocks, page_size, num_heads, head_size].
    # This tensor is a view of a 6D parent tensor with shape
    # [num_blocks, 2, num_layers, page_size, num_heads, head_size].
    # The outer stride depends on num_layers (unknown), so stride[0] must be
    # UNKNOWN_VALUE to ensure we use runtime strides for offset calculations.
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        Self.page_size,
        Self.kv_params.num_heads,
        Self.kv_params.head_size,
    )
    comptime blocks_strides = IntTuple(
        # Runtime value: 2 * num_layers * page_size * num_heads * head_size
        UNKNOWN_VALUE,
        Self.kv_params.num_heads * Self.kv_params.head_size,
        Self.kv_params.head_size,
        1,
    )
    comptime blocks_layout = Layout(Self.blocks_shape, Self.blocks_strides)

    # TileTensor layout for blocks, built directly from `blocks_shape` /
    # `blocks_strides`: leading dim is a runtime view stride (Int64), inner
    # dims are static.
    comptime blocks_tt_layout = InternalLayout[
        shape_types=Coord[
            Int64,
            ComptimeInt[Self.page_size],
            ComptimeInt[Self.kv_params.num_heads],
            ComptimeInt[Self.kv_params.head_size],
        ].element_types,
        stride_types=Coord[
            Int64,
            ComptimeInt[Self.kv_params.num_heads * Self.kv_params.head_size],
            ComptimeInt[Self.kv_params.head_size],
            ComptimeInt[1],
        ].element_types,
    ]
    comptime blocks_tt_type = TileTensor[
        Self.dtype, Self.blocks_tt_layout, Self.blocks_origin
    ]

    comptime cache_lengths_tt_layout = _1d_tt_layout
    comptime cache_lengths_tt_type = TileTensor[
        DType.uint32, Self.cache_lengths_tt_layout, Self.cache_lengths_origin
    ]

    comptime lookup_table_tt_layout = _2d_row_major_tt_layout
    comptime lookup_table_tt_type = TileTensor[
        DType.uint32, Self.lookup_table_tt_layout, Self.lookup_table_origin
    ]

    var blocks: Self.blocks_tt_type
    var cache_lengths: Self.cache_lengths_tt_type
    var lookup_table: Self.lookup_table_tt_type

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32

    # Number of quantization scale values per token.
    comptime head_dim_granularity = ceildiv(
        Self.kv_params.head_size,
        Self.quantization_granularity,
    )
    # Scales layout for a single K-or-V cache view.
    # Shape: [total_num_blocks, page_size, num_heads, head_dim_granularity].
    # stride[0] is Int64 because the parent 6D scales tensor has
    # outer stride = 2 * num_layers * page_size * num_heads * head_dim_gran,
    # which is only known at runtime (num_layers is runtime). Using a
    # comptime-derived stride[0] = page_size * num_heads * head_dim_gran
    # (as RowMajorLayout would produce) silently ignores the kv_idx and
    # num_layers multipliers, causing K-scale writes at block B to alias
    # V-scale writes at block B-1. Making stride[0] explicit Int64 lets
    # _make_cache_tt fill in the correct value from
    # kv_cache_scales_dynamic_strides[0].
    comptime scales_tt_layout = InternalLayout[
        shape_types=Coord[
            Int64,
            ComptimeInt[Self.page_size],
            ComptimeInt[Self.kv_params.num_heads],
            ComptimeInt[Self.head_dim_granularity],
        ].element_types,
        stride_types=Coord[
            Int64,
            ComptimeInt[Self.kv_params.num_heads * Self.head_dim_granularity],
            ComptimeInt[Self.head_dim_granularity],
            ComptimeInt[1],
        ].element_types,
    ]
    comptime scales_tt_type = TileTensor[
        Self.scale_dtype, Self.scales_tt_layout, Self.scales_origin
    ]

    # KV Cache quantization scales
    var scales: OptionalReg[Self.scales_tt_type]

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "PagedKVCache"

    def __init__(
        out self,
        blocks: Self.blocks_tt_type,
        cache_lengths: Self.cache_lengths_tt_type,
        lookup_table: Self.lookup_table_tt_type,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        scales: OptionalReg[Self.scales_tt_type] = None,
    ):
        assert (
            Int(blocks.dim[1]()) == Self.page_size
        ), "blocks.dim[1]() must be equal to page_size"
        assert (
            Int(blocks.dim[2]()) == Self.kv_params.num_heads
        ), "blocks.dim[2]() must be equal to kv_params.num_heads"
        assert (
            Int(blocks.dim[3]()) == Self.kv_params.head_size
        ), "blocks.dim[3]() must be equal to kv_params.head_size"

        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.scales = scales

    @staticmethod
    def max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return Self.page_size

    @always_inline
    def cache_lengths_nd(self) -> Self.cache_lengths_tt_type:
        return self.cache_lengths

    def cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    def _stride(self) -> UInt32:
        return UInt32(self.blocks.layout.stride[0]().value()) // UInt32(
            self.kv_params.num_heads * self.kv_params.head_size
        )

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        The encoded index is ``physical_block * page_size + offset``.  This
        method decomposes it and returns
        ``physical_block * stride + offset`` where *stride* is the distance
        (in rows) between consecutive physical blocks in the flattened
        memory view.
        """
        var phys_block = encoded_index // Int32(Self.page_size)
        var offset = encoded_index % Int32(Self.page_size)
        var stride = Int32(self._stride())
        return phys_block * stride + offset

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in this KV cache view."""
        var total_blocks = self.blocks.dim[0]()
        return Int(
            UInt32(total_blocks - 1) * self._stride() + UInt32(Self.page_size)
        )

    @always_inline
    def row_idx(self, batch_idx: UInt32, tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        var lut_block_index, tok_in_block_idx = divmod(
            Int(tok_idx), Self.page_size
        )
        assert tok_in_block_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"

        assert batch_idx < UInt32(
            self.cache_lengths.num_elements()
        ), "batch_idx is oob"
        debug_assert(
            lut_block_index < Int(self.lookup_table.dim[1]()),
            "lut_block_index is OOB. Attempted to access LUT column ",
            lut_block_index,
            " with lookup_table inner dim ",
            Int(self.lookup_table.dim[1]()),
        )
        block_idx = self.lookup_table[Int(batch_idx), lut_block_index]
        # alias row_stride = Int(num_heads * head_size * Self.collection_size)
        return block_idx * self._stride() + UInt32(tok_in_block_idx)

    @always_inline
    def populate[
        BN: Int,
        base_alignment: Int,
        pair_cta: Bool = False,
        is_leader: Bool = True,
    ](self, batch_idx: UInt32, base_kv_row: UInt32) -> PagedRowIndices[
        BN, Self.page_size_, pair_cta, is_leader
    ]:
        """SIMD LUT-load the `num_pages` block indices in one shot.

        Computes `result.rows[i] = lookup_table[batch, first_lut_idx+i]
        * stride + tok_in_block` for all `num_pages` entries using one
        (or a small fixed number of) aligned `ld.global.v{N}.u32` loads
        from the lookup table row.

        Invariants:
          - `self.lookup_table.dim[1]` is large enough that a SIMD read
            of `num_pages` uint32s starting at any valid
            `first_lut_idx` stays in bounds (see `PagedKVCacheManager`
            for the allocation-side padding).
          - `base_kv_row % base_alignment == 0` holds at runtime
            (typically `mask.start_column_alignment[...]()`).
            For `num_pages > 1`, `base_alignment` must be at least
            `page_size` — required so `tok_in_block_idx == 0` and the
            SIMD `multiply-add` collapses to a `multiply`. Larger
            `base_alignment` values let us pick a wider SIMD chunk
            (`chunk * page_size` must divide `base_alignment`).

        The per-load width `chunk` is the largest power of two that
        divides both `num_pages` and `base_alignment / page_size`,
        capped at 8. With `base_alignment == BN` (the historical
        contract), this matches the previous behaviour: `chunk =
        min(num_pages & -num_pages, 8)`. With looser alignments
        (e.g. `ChunkedMask` providing only `page_size` alignment when
        `BN > page_size`), the chunk degrades to 1 (scalar loads).
        """
        comptime Result = PagedRowIndices[
            BN, Self.page_size_, pair_cta, is_leader
        ]
        comptime num_pages = Result.num_pages
        var result = Result()
        comptime if num_pages == 1:
            comptime if base_alignment % Self.page_size == 0:
                # `base_kv_row` is page_size-aligned, so
                # `tok_in_block_idx == 0`: skip the divmod and the
                # `+ tok_in_block` add baked into `row_idx`.
                debug_assert(
                    base_kv_row % UInt32(Self.page_size) == 0,
                    (
                        "PagedKVCache.populate fast path requires"
                        " base_kv_row to be page_size-aligned"
                    ),
                )
                var lut_idx = base_kv_row // UInt32(Self.page_size)
                var block_idx = self.lookup_table[Int(batch_idx), Int(lut_idx)]
                result.rows[0] = block_idx * self._stride()
            else:
                result.rows[0] = self.row_idx(batch_idx, base_kv_row)
        else:
            # `chunk` is the largest power of two that
            #   1. divides `num_pages` (so the `comptime for` covers
            #      every LUT entry exactly once), and
            #   2. satisfies `chunk * page_size <= base_alignment` (so
            #      `first_lut_idx = base_kv_row / page_size` is a
            #      multiple of `chunk`, giving the natural
            #      `chunk * 4`-byte alignment the
            #      `ld.global.v{chunk}.u32` emitter needs).
            # Capped at 8 by hardware. With the historical contract of
            # `base_alignment == BN`, `alignment_chunks == num_pages`,
            # and this collapses to `min(num_pages & -num_pages, 8)`.
            comptime num_pages_pow2 = num_pages & -num_pages
            comptime alignment_chunks = base_alignment // Self.page_size
            comptime alignment_chunks_pow2 = (
                alignment_chunks & -alignment_chunks
            )
            comptime chunk = min(min(num_pages_pow2, alignment_chunks_pow2), 8)
            comptime assert (
                chunk >= 1
            ), "base_alignment must be >= page_size when num_pages > 1"
            comptime num_chunks = num_pages // chunk

            var stride = self._stride()
            # `tok_in_block` is zero because `base_alignment` is
            # required to be at least `page_size` whenever
            # `num_pages > 1` (every shipped mask satisfies this; see
            # the chunk derivation above). With
            # `tok_in_block_idx == 0`, `row_idx` collapses to
            # `block_idx * stride`, so the SIMD path emits a plain
            # multiply with no add.
            debug_assert(
                base_kv_row % UInt32(Self.page_size) == 0,
                (
                    "PagedKVCache.populate SIMD path requires"
                    " base_kv_row to be page_size-aligned when"
                    " num_pages > 1"
                ),
            )
            var first_lut_idx = base_kv_row // UInt32(Self.page_size)
            var row_stride = UInt32(
                self.lookup_table.layout.stride[0]().value()
            )
            # The address passed to the `ld.global.v{chunk}.u32`
            # emitter must be naturally aligned to `chunk * 4` bytes
            # AND each `ceildiv(num_pages, chunk)`-width vector load
            # must stay in-bounds of the LUT row. The three runtime
            # invariants below name each independent contract:
            #   1. `row_stride` chunk-aligned — LUT layout contract
            #      (see `_padded_lut_cols` in `cache_manager.py` /
            #      `padded_lut_cols` in `kv_cache_test_utils`).
            #   2. `first_lut_idx` chunk-aligned — mask contract (the
            #      mask's `start_column_alignment` must guarantee
            #      `base_kv_row` is `chunk * page_size`-aligned).
            #   3. `first_lut_idx + num_pages <= row_stride` — LUT
            #      allocation contract (the row has enough columns
            #      for a full SIMD sweep at the rightmost
            #      `first_lut_idx`).
            # Catch any violation under ``MOJO_ASSERT_LEVEL=safe`` so
            # misaligned or OOB vector loads don't silently produce
            # garbage.
            debug_assert(
                row_stride % UInt32(chunk) == 0,
                (
                    "PagedKVCache.populate SIMD path requires the LUT"
                    " row stride (lookup_table.dim[1]) to be"
                    " chunk-aligned. Production allocates via"
                    " `_padded_lut_cols` in cache_manager.py; tests"
                    " should use `padded_lut_cols` from"
                    " kv_cache_test_utils."
                ),
            )
            debug_assert(
                first_lut_idx % UInt32(chunk) == 0,
                (
                    "PagedKVCache.populate SIMD path requires"
                    " first_lut_idx (= base_kv_row / page_size) to be"
                    " chunk-aligned. The mask's"
                    " `start_column_alignment[BM, BN, page_size]()`"
                    " must return a value such that every"
                    " `base_kv_row` is `chunk * page_size`-aligned."
                ),
            )
            debug_assert(
                first_lut_idx + UInt32(num_pages) <= row_stride,
                (
                    "PagedKVCache.populate SIMD path requires the LUT"
                    " row to have at least `first_lut_idx + num_pages`"
                    " columns. Production adds a 16-element tail pad"
                    " in `_padded_lut_cols`."
                ),
            )
            var lut_row_ptr = (
                self.lookup_table.ptr + batch_idx * row_stride + first_lut_idx
            )
            comptime for c in range(num_chunks):
                var simd = lut_row_ptr.load[width=chunk, alignment=4 * chunk](
                    c * chunk
                )
                var rows_simd = simd * SIMD[DType.uint32, chunk](stride)
                comptime for i in range(chunk):
                    result.rows[c * chunk + i] = rows_simd[i]
        return result

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Self.kv_params.head_size
        ](),
        fold_chunks: Int = 1,
        row_major: Bool = False,
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        Self.dtype,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a TMA tile for this KV cache."""
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0, "BK must be a multiple of swizzle granularity"
        # Paged cache collection is (where `$idx` means subsetting that idx):
        # [total_num_blocks, $kv_idx, $layer_idx, page_size, num_heads, head_size]
        #
        # An axis's size is 1 + maximum valid idx
        # Idx calc is:
        # block_idx * self._stride() + tok_in_block_idx
        # max values
        # (total_blocks - 1) * self._stride() + Self.page_size - 1
        # yields number of rows:
        # (total_blocks - 1) * self._stride() + Self.page_size
        #
        # Create a view that accounts for the paged layout
        var total_blocks = Int(self.blocks.dim[0]())
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            Self.page_size
        )
        comptime smem_dim = IndexList[3](BN, 1, BK)
        comptime gmem_dim = IndexList[3](
            UNKNOWN_VALUE,
            Self.kv_params.num_heads,
            Self.kv_params.head_size,
        )
        return create_split_tma[
            smem_dim,
            gmem_dim,
            swizzle_mode,
            fold_chunks=fold_chunks,
            row_major=row_major,
        ](ctx, self.blocks.ptr, Int(rows))

    @always_inline
    def create_gather4_tma_tile[
        *,
        tile_height: Int = 4,
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        tma_dtype,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a 2D TMA gather4 descriptor for this KV cache.

        The descriptor views the KV cache as a flat 2D matrix of
        ``[num_kv_rows, tile_width]`` and is configured for gather4 operations
        that load 4 non-contiguous rows per TMA instruction. The box width
        is derived from the swizzle mode; for SWIZZLE_NONE it equals
        ``tile_width``.

        When ``tma_dtype`` differs from ``Self.dtype``, the underlying data
        pointer is bitcast to ``tma_dtype`` at descriptor creation time.

        Parameters:
            tile_height: Number of rows in the tile. Must be a multiple of 4.
                Defaults to 4 for backward compatibility.
            tile_width: Number of elements per row to load (box width) in
                ``tma_dtype`` elements.
            tile_stride: Row stride in elements in global memory. Defaults to
                ``tile_width``. Use a larger value when the global row is
                wider than the portion to load.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to SWIZZLE_NONE.
            tma_dtype: The data type used for the TMA descriptor. Defaults to
                ``Self.dtype``. When different, the pointer is bitcast.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A TMATensorTile with box width derived from the swizzle mode.
        """
        return create_tma_tile_gather4[
            tma_dtype,
            tile_height=tile_height,
            tile_width=tile_width,
            tile_stride=tile_stride,
            swizzle_mode=swizzle_mode,
            l2_promotion=l2_promotion,
        ](
            ctx,
            self.blocks.ptr.bitcast[Scalar[tma_dtype]](),
            self.num_kv_rows(),
        )

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Creates a BF16 TMA tile for the rope portion of the per-tensor rope-aware KV cache.

        In the per-tensor rope-aware layout each token row is:
          `padded_depth` FP8 bytes (content) | `BK` BF16 elements (rope)
        Total row bytes = padded_depth + BK * 2.

        The TMA descriptor points at the rope data by offsetting `blocks.ptr`
        by `padded_depth` bytes, then reinterpreting as BF16.  The global
        memory stride dimension (last dim of gmem_shape) is the total row size
        expressed in BF16 units: (padded_depth + BK * 2) // 2.
        """
        comptime assert (
            BK % swizzle_granularity[DType.bfloat16, swizzle_mode]()
        ) == 0, "BK must be a multiple of swizzle granularity for BF16"
        # Compute the total row width in BF16 elements:
        #   padded_depth FP8 bytes + BK BF16 elements
        #   = (padded_depth + BK * 2) bytes total
        #   = (padded_depth + BK * 2) // 2 BF16 elements per row
        comptime bf16_row_stride = (padded_depth + BK * 2) // 2

        var total_blocks = self.blocks.dim[0]()
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            Self.page_size
        )
        # Offset past the FP8 content to reach the BF16 rope data,
        # then reinterpret the pointer as BF16.
        var rope_ptr = (self.blocks.ptr + padded_depth).bitcast[
            Scalar[DType.bfloat16]
        ]()
        comptime smem_dim = IndexList[3](BN, 1, BK)
        comptime gmem_dim = IndexList[3](
            UNKNOWN_VALUE,
            Self.kv_params.num_heads,
            bf16_row_stride,
        )
        tma = create_split_tma[smem_dim, gmem_dim, swizzle_mode](
            ctx, rope_ptr, Int(rows)
        )

    @always_inline
    def create_rope_gather4_tma_tile[
        *,
        tile_height: Int = 4,
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a BF16 gather4 TMA descriptor for the rope portion of the
        KV cache.

        For the per-tensor rope-aware layout each token row is stored as
        ``padded_depth`` FP8 bytes (content) followed by BF16 rope elements.
        The total row width in BF16 units is
        ``(padded_depth + tile_width * 2) // 2``.

        This method offsets ``blocks.ptr`` by ``padded_depth`` bytes,
        reinterprets as BF16, and creates a gather4 TMA descriptor whose row
        stride is the full row width in BF16 elements.
        """
        var rope_ptr = (self.blocks.ptr + padded_depth).bitcast[
            Scalar[DType.bfloat16]
        ]()
        return create_tma_tile_gather4[
            DType.bfloat16,
            tile_height=tile_height,
            tile_width=tile_width,
            swizzle_mode=swizzle_mode,
            l2_promotion=l2_promotion,
        ](ctx, rope_ptr, self.num_kv_rows())

    @always_inline
    def _get_idx(
        self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> DynamicCoord[DType.int64, 4]:
        debug_assert(
            head_idx < Self.kv_params.num_heads,
            "KVCache head_idx out of range (",
            head_idx,
            ")",
        )
        assert (
            head_dim_idx < Self.kv_params.head_size
        ), "KVCache head_dim_idx is out of range"

        var lut_block_idx, tok_in_block_idx = divmod(tok_idx, self.page_size)

        assert tok_in_block_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"

        assert bs < self.cache_lengths.num_elements(), "batch_idx is oob"
        debug_assert(
            lut_block_idx < Int(self.lookup_table.dim[1]()),
            "lut_block_idx is OOB. Attempted to access LUT column ",
            lut_block_idx,
            " with lookup_table inner dim ",
            Int(self.lookup_table.dim[1]()),
        )
        block_idx = Int(self.lookup_table[bs, lut_block_idx])
        return dyn_coord[DType.int64](
            (
                block_idx,
                tok_in_block_idx,
                head_idx,
                head_dim_idx,
            ),
        )

    @always_inline
    def _get_scale_idx(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> DynamicCoord[DType.int64, 4]:
        debug_assert(
            head_idx < Self.kv_params.num_heads,
            "KVCache head_idx out of range (",
            head_idx,
            ")",
        )
        var lut_block_idx, tok_in_block_idx = divmod(tok_idx, self.page_size)

        assert tok_in_block_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"

        assert bs < self.cache_lengths.num_elements(), "batch_idx is oob"
        debug_assert(
            lut_block_idx < Int(self.lookup_table.dim[1]()),
            "lut_block_idx is OOB. Attempted to access LUT column ",
            lut_block_idx,
            " with lookup_table inner dim ",
            Int(self.lookup_table.dim[1]()),
        )
        block_idx = Int(self.lookup_table[bs, lut_block_idx])
        # floordiv: head_dim_idx is the *start* of the quantization block
        # (e.g. 0, 64, 128, …), so we want which block slot this maps to.
        # ceildiv would be wrong here: ceildiv(64, 64) == 1 (correct for the
        # second block) but ceildiv(0, 64) == 0 (OK), ceildiv(63, 64) == 1
        # (wrong — element 63 is still in block 0). floordiv correctly maps
        # any element at position d to block d // granularity.
        var scale_block_idx = head_dim_idx // Self.quantization_granularity
        return dyn_coord[DType.int64](
            (
                block_idx,
                tok_in_block_idx,
                head_idx,
                scale_block_idx,
            ),
        )

    @always_inline
    def load[
        width: Int,
        output_dtype: DType = Self.dtype,
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        output_dtype, width
    ]:
        """Loads an element from the given index."""

        comptime if Self.quantization_enabled:
            comptime assert output_dtype != Self.dtype, (
                "Output type should not be FP8 when KVCache quantization is"
                " disabled"
            )

        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)

        # Bypass TileTensor.load's `where` constraint by using ptr directly.
        comptime if Self.quantization_enabled:
            var quantized_val = self.blocks.load[width=width](idx)
            var scale = self.load_scale[width=1](
                bs, head_idx, tok_idx, head_dim_idx
            )
            var dequantized = quantized_val.cast[Self.scale_dtype]() * scale
            return dequantized.cast[output_dtype]()
        else:
            return self.blocks.load[width=width](idx).cast[output_dtype]()

    @always_inline
    def store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        """Stores an element at the given index.

        Skips the write when the LUT entry for ``(bs, tok_idx // page_size)``
        is the unassigned-slot sentinel — i.e. when the resolved
        ``block_idx`` is outside ``[0, total_num_blocks)``. The cache
        manager fills LUT columns past a request's allocated block count
        with the sentinel value ``total_num_pages`` (see
        ``cache_manager.py``'s ``lut_table_np.fill(self._total_num_pages)``)
        so that SIMD over-reads of the LUT row are safe, but the *value*
        of the sentinel times the page stride lands one page past the
        end of the cache buffer. Without this guard a sentinel-resolved
        store corrupts whatever device allocation happens to sit
        immediately after the KV cache.
        """
        var lut_block_idx, tok_in_block_idx = divmod(tok_idx, self.page_size)
        var block_idx = Int(self.lookup_table[bs, lut_block_idx])
        debug_assert(
            block_idx < Int(self.blocks.dim[0]()),
            "KVCache block_idx resolved to sentinel/unassigned LUT entry (",
            block_idx,
            ")",
        )
        debug_assert(
            head_idx < Self.kv_params.num_heads,
            "KVCache head_idx out of range (",
            head_idx,
            ")",
        )
        assert (
            head_dim_idx < Self.kv_params.head_size
        ), "KVCache head_dim_idx is out of range"
        assert tok_in_block_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"
        var idx = Coord((block_idx, tok_in_block_idx, head_idx, head_dim_idx))
        # Bypass TileTensor.store's `where` constraint by using ptr directly.
        self.blocks.store(idx, val)

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.scale_dtype, width
    ]:
        """Loads a quantization scale from the given index."""
        comptime assert (
            Self.quantization_enabled
        ), "Scales only exist for quantized KVCache"
        comptime assert (
            Self.scale_dtype != DType.invalid
        ), "Invalid scale data type"
        assert (
            self.scales is not None
        ), "Scales missing, yet KVCache quantization enabled"
        var idx = self._get_scale_idx(bs, head_idx, tok_idx, head_dim_idx)
        # Bypass TileTensor.load's `where` constraint by using ptr directly.
        return self.scales.value().load[width=width](idx)

    @always_inline
    def store_scale(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        scales: SIMD[Self.scale_dtype, ...],
    ):
        """Stores the quantization scales at the given index."""

        comptime if Self.quantization_enabled:
            comptime assert (
                Self.scale_dtype != DType.invalid
            ), "Valid quantization scale data type needed"

        var lut_block_idx, tok_in_block_idx = divmod(tok_idx, self.page_size)
        var block_idx = Int(self.lookup_table[bs, lut_block_idx])
        debug_assert(
            block_idx < Int(self.blocks.dim[0]()),
            "KVCache block_idx resolved to sentinel/unassigned LUT entry (",
            block_idx,
            ")",
        )
        var scale_block_idx = head_dim_idx // Self.quantization_granularity
        var scale_idx = Coord(
            (
                block_idx,
                tok_in_block_idx,
                head_idx,
                scale_block_idx,
            )
        )
        # Bypass TileTensor.store's `where` constraint by using ptr directly.
        self.scales.value().store(scale_idx, scales)

    @always_inline
    def load_quantized[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.dtype, width
    ]:
        """Loads a quantized element from the given index."""
        comptime assert Self.quantization_enabled, (
            "Output type should not be quantized when KVCache quantization is"
            " disabled"
        )
        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)
        # Bypass TileTensor.load's `where` constraint by using ptr directly.
        return self.blocks.load[width=width](idx)

    def empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        return self.max_cache_length == 0

    def max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        return self.max_seq_length

    def max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        return self.max_cache_length

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        comptime assert (
            tile_size <= Self.page_size and Self.page_size % tile_size == 0
        ), (
            "Invalid tile size for PagedKVCache. tile_size must be less"
            " than or equal to the page size and divisible by the page size"
        )

        var full_block_idx = self._get_idx(
            batch_idx, head_idx, start_tok_idx, head_dim_idx
        )

        var ptr = self.blocks.ptr + Int(self.blocks.layout(full_block_idx))
        return ptr.as_unsafe_any_origin()

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a pointer to the scales block at the requested indices."""
        comptime assert (
            self.quantization_enabled
        ), "Quantization must be enabled to request scales block"
        var full_scale_block_idx = self._get_scale_idx(
            batch_idx, head_idx, start_tok_idx, head_dim_idx
        )
        assert self.scales is not None, "Quantization scale factors not set."
        var scales_block = self.scales.value()

        var scales_ptr = scales_block.ptr + Int(
            scales_block.layout(full_scale_block_idx)
        )
        return scales_ptr.as_unsafe_any_origin()

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns the base pointer to the scales tensor, or a
        dangling pointer if scales are not set."""

        comptime if Self.quantization_enabled:
            return self.scales.value().ptr.as_unsafe_any_origin()
        # SAFETY: Only reached when quantization is disabled; callers guard
        # scales access behind comptime `quantization_enabled` checks.
        return UnsafePointer[
            Scalar[Self.scale_dtype], MutAnyOrigin
        ].unsafe_dangling()


trait KVCollectionT(ImplicitlyCopyable):
    """Trait for a pair of caches (keys and values)."""

    comptime CacheType: KVCacheT
    comptime name_str: StaticString
    comptime dtype: DType
    comptime kv_params: KVCacheStaticParams

    def get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        ...

    def get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        ...

    def cache_length(self, bs_idx: Int) -> Int:
        ...


struct ContinuousBatchingKVCacheCollection[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    blocks_origin: MutOrigin,
    cache_lengths_origin: ImmutOrigin,
    lookup_table_origin: ImmutOrigin,
](KVCollectionT):
    """This is a "view" of the cache for the given sequences
    in the batch.

    Parameters:
        dtype_: The dtype of the kv-cache.
        kv_params_: The kv-cache static parameters.
        blocks_origin: Origin of the KV cache blocks buffer.
        cache_lengths_origin: Origin of the cache lengths buffer.
        lookup_table_origin: Origin of the lookup table buffer.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our KVCacheManager.
    """

    comptime name_str = "continuous_batching"
    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime CacheType = ContinuousBatchingKVCache[
        Self.dtype,
        Self.kv_params,
        Self.blocks_origin,
        Self.cache_lengths_origin,
        Self.lookup_table_origin,
    ]
    comptime scale_dtype: DType = DType.invalid

    # Shape is [num_blocks, 2, num_layers, max_seq_len, num_heads, head_size].
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        Self.kv_params.num_heads,
        Self.kv_params.head_size,
    )
    comptime blocks_layout = Layout.row_major(Self.blocks_shape)
    # Direct row-major TileTensor layout: the four leading dims are runtime
    # (Int64), so every stride that folds one in is also runtime.
    comptime blocks_tt_layout = InternalLayout[
        shape_types=Coord[
            Int64,
            Int64,
            Int64,
            Int64,
            ComptimeInt[Self.kv_params.num_heads],
            ComptimeInt[Self.kv_params.head_size],
        ].element_types,
        stride_types=Coord[
            Int64,
            Int64,
            Int64,
            ComptimeInt[Self.kv_params.num_heads * Self.kv_params.head_size],
            ComptimeInt[Self.kv_params.head_size],
            ComptimeInt[1],
        ].element_types,
    ]
    comptime blocks_tt_type = TileTensor[
        Self.dtype, Self.blocks_tt_layout, Self.blocks_origin
    ]

    var blocks: Self.blocks_tt_type
    var cache_lengths: Self.CacheType.cache_lengths_tt_type
    var lookup_table: Self.CacheType.lookup_table_tt_type
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var kv_cache_dynamic_shape: IndexList[4]
    var kv_cache_dynamic_strides: IndexList[4]

    def __init__(
        out self,
        blocks: LayoutTensor[
            Self.dtype, Layout.row_major[6](), Self.blocks_origin
        ],
        cache_lengths: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), Self.cache_lengths_origin
        ],
        lookup_table: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), Self.lookup_table_origin
        ],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        """Construct from LayoutTensor params (MOGG boundary)."""
        comptime assert blocks.rank == 6
        self.blocks = lt_to_tt[ResultLayout=Self.blocks_tt_layout](blocks)
        self.cache_lengths = lt_to_tt[
            ResultLayout=Self.CacheType.cache_lengths_tt_layout
        ](cache_lengths)
        self.lookup_table = lt_to_tt[
            ResultLayout=Self.CacheType.lookup_table_tt_layout
        ](lookup_table)
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )

    def __init__(
        out self,
        blocks: Self.blocks_tt_type,
        cache_lengths: Self.CacheType.cache_lengths_tt_type,
        lookup_table: Self.CacheType.lookup_table_tt_type,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        """Construct from TileTensor fields directly."""
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )

    @always_inline
    def get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[0](layer_idx)

    @always_inline
    def get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[1](layer_idx)

    @always_inline
    def _get_cache[kv_idx: Int](self, layer_idx: Int) -> Self.CacheType:
        assert (
            kv_idx == 0 or self.blocks.dim[1]() > 1
        ), "invalid kv_idx for MLA cache"
        var offset = Int(
            self.blocks.layout(
                dyn_coord[DType.int64](
                    (
                        0,
                        kv_idx,
                        layer_idx,
                        0,
                        0,
                        0,
                    )
                )
            )
        )
        return self.CacheType(
            _make_cache_tt[
                Self.CacheType.dtype,
                Self.CacheType.blocks_tt_layout,
                4,
            ](
                self.blocks.ptr + offset,
                self.kv_cache_dynamic_shape,
                self.kv_cache_dynamic_strides,
            ),
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
        )

    def cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])


struct PagedKVCacheCollection[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
    blocks_origin: MutOrigin,
    cache_lengths_origin: ImmutOrigin,
    lookup_table_origin: ImmutOrigin,
    scales_origin: MutOrigin,
    *,
    scale_dtype_: DType = DType.invalid,
    quantization_granularity_: Int = 1,
](KVCollectionT):
    comptime name_str = "paged"
    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime scale_dtype = Self.scale_dtype_
    comptime CacheType = PagedKVCache[
        Self.dtype,
        Self.kv_params,
        Self.page_size,
        Self.blocks_origin,
        Self.cache_lengths_origin,
        Self.lookup_table_origin,
        Self.scales_origin,
        scale_dtype_=Self.scale_dtype,
        quantization_granularity_=Self.quantization_granularity_,
    ]

    # Shape is [total_num_blocks, 2, num_layers, page_size, num_heads, head_size].
    # Matrix view is
    # (total_num_blocks, 2, num_layers, page_size) x (num_heads, head_size)
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        2 if not Self.kv_params.is_mla else 1,
        UNKNOWN_VALUE,
        Self.page_size,
        Self.kv_params.num_heads,
        Self.kv_params.head_size,
    )
    comptime blocks_layout = Layout.row_major(Self.blocks_shape)
    # Direct row-major TileTensor layout. dims 0 and 2 (total_num_blocks and
    # num_layers) are runtime, so strides[0..1] that fold them in are runtime.
    comptime blocks_tt_layout = InternalLayout[
        shape_types=Coord[
            Int64,
            ComptimeInt[2 if not Self.kv_params.is_mla else 1],
            Int64,
            ComptimeInt[Self.page_size],
            ComptimeInt[Self.kv_params.num_heads],
            ComptimeInt[Self.kv_params.head_size],
        ].element_types,
        stride_types=Coord[
            Int64,
            Int64,
            ComptimeInt[
                Self.page_size
                * Self.kv_params.num_heads
                * Self.kv_params.head_size
            ],
            ComptimeInt[Self.kv_params.num_heads * Self.kv_params.head_size],
            ComptimeInt[Self.kv_params.head_size],
            ComptimeInt[1],
        ].element_types,
    ]
    comptime blocks_tt_type = TileTensor[
        Self.dtype, Self.blocks_tt_layout, Self.blocks_origin
    ]

    # Match PagedKVCache.head_dim_granularity.
    comptime head_dim_granularity = ceildiv(
        Self.kv_params.head_size,
        Self.CacheType.quantization_granularity,
    )
    # Define scales tensor with shape [total_num_blocks, 2, num_layers, page_size, num_heads, granularity]
    comptime scales_shape = IntTuple(
        UNKNOWN_VALUE,  # total_num_blocks
        2 if not Self.kv_params.is_mla else 1,
        UNKNOWN_VALUE,  # num_layers
        Self.page_size,  # page_size
        Self.kv_params.num_heads,  # num_heads
        Self.head_dim_granularity,  # scales per token
    )
    comptime scales_layout = Layout.row_major(Self.scales_shape)
    # Direct row-major TileTensor layout, mirroring `blocks_tt_layout` but with
    # `head_dim_granularity` as the inner dim. dims 0 and 2 are runtime.
    comptime scales_tt_layout = InternalLayout[
        shape_types=Coord[
            Int64,
            ComptimeInt[2 if not Self.kv_params.is_mla else 1],
            Int64,
            ComptimeInt[Self.page_size],
            ComptimeInt[Self.kv_params.num_heads],
            ComptimeInt[Self.head_dim_granularity],
        ].element_types,
        stride_types=Coord[
            Int64,
            Int64,
            ComptimeInt[
                Self.page_size
                * Self.kv_params.num_heads
                * Self.head_dim_granularity
            ],
            ComptimeInt[Self.kv_params.num_heads * Self.head_dim_granularity],
            ComptimeInt[Self.head_dim_granularity],
            ComptimeInt[1],
        ].element_types,
    ]
    comptime scales_tt_type = TileTensor[
        Self.scale_dtype, Self.scales_tt_layout, Self.scales_origin
    ]

    var scales: OptionalReg[Self.scales_tt_type]
    var kv_cache_scales_dynamic_shape: IndexList[4]
    var kv_cache_scales_dynamic_strides: IndexList[4]
    var blocks: Self.blocks_tt_type
    var cache_lengths: Self.CacheType.cache_lengths_tt_type
    var lookup_table: Self.CacheType.lookup_table_tt_type
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var kv_cache_dynamic_shape: IndexList[4]
    var kv_cache_dynamic_strides: IndexList[4]

    def __init__(
        out self,
        blocks: LayoutTensor[
            Self.dtype, Layout.row_major[6](), Self.blocks_origin
        ],
        cache_lengths: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), Self.cache_lengths_origin
        ],
        lookup_table: LayoutTensor[
            DType.uint32, Layout.row_major[2](), Self.lookup_table_origin
        ],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        scales: OptionalReg[
            LayoutTensor[
                Self.scale_dtype, Layout.row_major[6](), Self.scales_origin
            ]
        ] = OptionalReg[
            LayoutTensor[
                Self.scale_dtype, Layout.row_major[6](), MutUntrackedOrigin
            ]
        ](),
    ):
        """Construct from LayoutTensor params (MOGG boundary)."""
        comptime assert blocks.rank == 6
        self.blocks = lt_to_tt[ResultLayout=Self.blocks_tt_layout](blocks)
        self.cache_lengths = lt_to_tt[
            ResultLayout=Self.CacheType.cache_lengths_tt_layout
        ](cache_lengths)
        self.lookup_table = lt_to_tt[
            ResultLayout=Self.CacheType.lookup_table_tt_layout
        ](lookup_table)
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )
        if scales is not None:
            self.scales = lt_to_tt[ResultLayout=Self.scales_tt_layout](
                scales.value()
            )
            self.kv_cache_scales_dynamic_shape, self.kv_cache_scales_dynamic_strides = _compute_kv_cache_dynamic_shape_strides[
                4, (1, 2)
            ](
                self.scales.value()
            )
        else:
            self.scales = None
            self.kv_cache_scales_dynamic_shape = IndexList[4](0, 0, 0, 0)
            self.kv_cache_scales_dynamic_strides = IndexList[4](0, 0, 0, 0)

    def __init__(
        out self,
        blocks: Self.blocks_tt_type,
        cache_lengths: Self.CacheType.cache_lengths_tt_type,
        lookup_table: Self.CacheType.lookup_table_tt_type,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        scales: OptionalReg[Self.scales_tt_type] = None,
    ):
        """Construct from TileTensor fields directly."""
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )
        if scales is not None:
            self.scales = scales.value()
            self.kv_cache_scales_dynamic_shape, self.kv_cache_scales_dynamic_strides = _compute_kv_cache_dynamic_shape_strides[
                4, (1, 2)
            ](
                self.scales.value()
            )
        else:
            self.scales = None
            self.kv_cache_scales_dynamic_shape = IndexList[4](0, 0, 0, 0)
            self.kv_cache_scales_dynamic_strides = IndexList[4](0, 0, 0, 0)

    @always_inline
    def get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[0](layer_idx)

    @always_inline
    def get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        comptime assert (
            not Self.kv_params.is_mla
        ), "Cannot call get_value_cache for MLA cache"
        return self._get_cache[1](layer_idx)

    @always_inline
    def _get_cache[kv_idx: Int](self, layer_idx: Int) -> Self.CacheType:
        comptime assert (
            kv_idx >= 0 and kv_idx < 2
        ), "Invalid kv_idx for KV cache"

        var kv_layer_coord = dyn_coord[DType.int64](
            (
                0,
                kv_idx,
                layer_idx,
                0,
                0,
                0,
            )
        )

        var scales_tt: OptionalReg[Self.CacheType.scales_tt_type] = None
        comptime if Self.CacheType.quantization_enabled:
            if self.scales is not None:
                var scale_offset = Int(
                    self.scales.value().layout(kv_layer_coord)
                )
                scales_tt = _make_cache_tt[
                    Self.CacheType.scale_dtype,
                    Self.CacheType.scales_tt_layout,
                    4,
                ](
                    self.scales.value().ptr + scale_offset,
                    self.kv_cache_scales_dynamic_shape,
                    self.kv_cache_scales_dynamic_strides,
                )

        var blocks_offset = Int(self.blocks.layout(kv_layer_coord))
        return self.CacheType(
            _make_cache_tt[
                Self.CacheType.dtype,
                Self.CacheType.blocks_tt_layout,
                4,
            ](
                self.blocks.ptr + blocks_offset,
                self.kv_cache_dynamic_shape,
                self.kv_cache_dynamic_strides,
            ),
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            scales_tt,
        )

    def cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])
