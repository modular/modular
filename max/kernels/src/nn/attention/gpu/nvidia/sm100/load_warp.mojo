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
"""TMA load warp logic for FA4 (SM100 Flash Attention)."""

from std.math import ceildiv
from std.sys import size_of
from std.gpu.memory import CacheEviction
from std.gpu.primitives.cluster import block_rank_in_cluster
from layout.tma_async import SharedMemBarrier
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.sm100.attention import FA4Config
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    elect,
    expect_bytes_pred,
    KProducerPipeline,
    VProducerPipeline,
    StagedPipeline,
    PagedRowIndices,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
)
from nn.attention.gpu.nvidia.sm90.attention import (
    KVTMATile,
    MHAPosition,
    OptionalPointer,
    QTMATile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo
from nn.attention.mha_utils import OptionallyStaticInt, _is_decoding
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from .smem import SM100AttentionSMem


@always_inline
def fa4_load[
    KVLUTType: MHAOperand,
    MaxSeqLenType: OptionallyStaticInt,
    MaskType: MHAMask,
    //,
    config: FA4Config,
    *,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    is_leader: Bool,
](
    smem: SM100AttentionSMem[config],
    score_row: UInt32,
    num_keys: UInt32,
    seq_info: SeqInfo,
    max_seq_len: MaxSeqLenType,
    mask: MaskType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BM=config.BM // 2,
        depth=config.qk_depth,
        group=config.group,
        decoding=False,
        fuse_gqa=config.fuse_gqa,
        num_qk_stages=config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=kv_sub_tile_rows(config.k_rows_per_cta(), KVLUTType.page_size),
        BK=config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=kv_sub_tile_rows(config.BN, KVLUTType.page_size),
        BK=config.v_cols_per_cta(),
    ],
    kv_lut: KVLUTType,
):
    comptime assert KVLUTType.dtype == config.qkv_dtype
    comptime qkv_type = KVLUTType.dtype
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime group = config.group
    comptime fuse_gqa = config.fuse_gqa
    # For pair-CTA, use PairBM so both CTAs make identical mask decisions.
    comptime BM_mask: Int = config.PairBM_eff()
    comptime page_size = KVLUTType.page_size
    # Alignment of `kv_row` values produced by mask-driven iteration.
    # Used by `kv_lut.populate` to pick the largest legal SIMD chunk.
    comptime base_alignment: Int = MaskType.start_column_alignment[
        BM_mask, BN, page_size
    ]()
    comptime ragged = not ValidLengthType.is_null
    comptime cta_group: Int = config.cta_group()
    comptime pair_cta: Bool = config.pair_cta
    comptime assert pair_cta or is_leader

    # Unified paged-rows type shared by fused-KV and split-KV: populate
    # covers V's full tile so V can consume K's pre-populated indices
    # with no lazy LUT lookup. In non-pair-CTA mode `num_pages` is
    # simply `BN / eff_page`. In pair-CTA mode the struct's
    # `is_leader`/`pair_cta` params select K's half at comptime (index
    # shift for `num_pages >= 2`, intra-page row shift for
    # `num_pages == 1`); storage is always V-sized so V reuses the same
    # array without re-LUT.
    comptime KVPagedRows = PagedRowIndices[
        BN=BN,
        page_size=page_size,
        pair_cta=pair_cta,
        is_leader=is_leader,
    ]
    # K's per-CTA TMA page count. Must derive from K's tile size
    # (k_rows_per_cta) rather than `num_pages // 2`: when
    # `page_size >= BN` (e.g. ps256 hs128), `num_pages = 1` but K's
    # TMA still issues once per CTA (it covers BN/2 rows from a single
    # page), so `k_pages_per_cta = 1`, not 0.
    comptime k_pages_per_cta = kv_num_sub_tiles(
        config.k_rows_per_cta(), page_size
    )

    var mbars = smem.misc_mbars()

    comptime PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.qk_depth,
        config.padded_qk_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    comptime KPipeType = KProducerPipeline[KVLUTType.dtype, config]
    comptime VPipeType = VProducerPipeline[KVLUTType.dtype, config]

    # If two-qo, we produce qkv in a pattern of
    # q0 & k0, q1, v0, k1, v1, k2, v2...
    # TMA only uses .ptr — flat row_major TileTensor is sufficient.
    # q_elements = HalfBM * BK0 regardless of fuse_gqa
    # (fused: BM//(2*group) * group * BK0 = HalfBM * BK0)
    comptime q_elements = HalfBM * config.BK0
    comptime QType = TileTensor[
        KVLUTType.dtype,
        type_of(tt_row_major[q_elements]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    var kv_head_idx: UInt32
    comptime if fuse_gqa:
        kv_head_idx = seq_info.head_idx
    else:
        kv_head_idx = seq_info.head_idx // UInt32(group)

    var q_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
        smem.q_smem()
    )
    comptime q_bytes = size_of[qkv_type]() * q_elements

    var q_gmem_row: UInt32 = PositionType.get_q_gmem_row[ragged=ragged](
        seq_info, max_seq_len
    )

    # Pair-CTA: each CTA loads different Q positions and half of K/V.
    var k_row_offset = UInt32(0)
    var v_col_offset = 0
    comptime if not is_leader:
        q_gmem_row += UInt32(config.BM_eff())
        k_row_offset = UInt32(BN // 2)
        v_col_offset = config.v_cols_per_cta()

    e = elect()

    # Default eviction_policy: pair-CTA disallows cache_hint with
    # cta_group=2 in the stdlib TMA intrinsic, so we fall back to
    # EVICT_NORMAL there. Non-pair-CTA keeps EVICT_FIRST (Q is read
    # once per stage then discarded).
    comptime q_default_eviction: CacheEviction = (
        CacheEviction.EVICT_NORMAL if pair_cta else CacheEviction.EVICT_FIRST
    )

    @parameter
    @always_inline
    def q_async_copy[
        eviction_policy: CacheEviction = q_default_eviction,
    ](
        smem_dst: QType,
        ref[AddressSpace.SHARED] mbar: SharedMemBarrier,
        depth_idx: UInt32 = 0,
    ):
        """Issue Q TMA elect-predicated on `e`. Caller no longer needs
        `if e != 0:` around the call — the TMA fires only on the elected
        lane via the PTX predicate inside `_elect`."""
        comptime if fuse_gqa:
            q_tma_op.async_copy_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                smem_dst,
                mbar,
                StaticTuple[UInt32, 4](depth_idx, 0, kv_head_idx, q_gmem_row),
                e,
            )
        else:
            q_tma_op.async_copy_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                smem_dst,
                mbar,
                StaticTuple[UInt32, 3](
                    depth_idx, seq_info.head_idx, q_gmem_row
                ),
                e,
            )

    # Partial-page handling: when page_size < BN, a BN-sized tile may span
    # more pages than the sequence has allocated. We detect this and use
    # `tma_copy_{k,v}[needs_partial=True]` with a runtime-bounded page
    # count to avoid OOB page lookups.
    comptime needs_partial = page_size > 0 and page_size < BN
    comptime k_bytes_pp = config.BK0 * KVPagedRows.eff_page * size_of[
        qkv_type
    ]()
    comptime v_bytes_pp = config.v_cols_per_cta() * KVPagedRows.eff_page * size_of[
        qkv_type
    ]()

    @parameter
    @always_inline
    def _k_num_valid_pages(current_kv_row: UInt32) -> UInt32:
        """Valid K sub-tile pages at `current_kv_row` (per-CTA range)."""
        if current_kv_row >= num_keys:
            return UInt32(0)
        return min(
            UInt32(k_pages_per_cta),
            UInt32(
                ceildiv(Int(num_keys - current_kv_row), KVPagedRows.eff_page)
            ),
        )

    @parameter
    @always_inline
    def _v_num_valid_pages(current_kv_row: UInt32) -> UInt32:
        """Valid V sub-tile pages at `current_kv_row` (full BN range)."""
        return min(
            UInt32(KVPagedRows.num_pages),
            UInt32(
                ceildiv(Int(num_keys - current_kv_row), KVPagedRows.eff_page)
            ),
        )

    var kv_row: UInt32 = mask.start_column[BM_mask, BN, page_size](score_row)
    var iter_count: UInt32 = (
        mask.last_masked_set_end[BM_mask, BN, page_size](score_row, num_keys)
        - 1
    )

    # Valid page counts for the first tile (shared between fused-KV and
    # split-KV). When `needs_partial`, these reflect how many sub-tile
    # pages are actually in-bounds for the sequence; otherwise every
    # sub-tile is assumed fully populated.
    #
    # `k_nvp` is this CTA's count (the half it will TMA). `k_nvp_peer`
    # holds the *other* CTA's count and is read only by the leader when
    # it accumulates `expect_bytes` across the cluster's shared barrier.
    # Non-leader keeps it at 0; the peer's `expect_bytes` branch is
    # comptime-pruned so the value is dead there.
    var k_nvp: UInt32 = UInt32(k_pages_per_cta)
    var k_nvp_peer: UInt32 = UInt32(0)
    var v_nvp: UInt32 = UInt32(KVPagedRows.num_pages)
    comptime if needs_partial:
        k_nvp = _k_num_valid_pages(kv_row + k_row_offset)
        comptime if is_leader and pair_cta:
            k_nvp_peer = _k_num_valid_pages(
                kv_row + UInt32(config.k_rows_per_cta())
            )
        v_nvp = _v_num_valid_pages(kv_row)

    # Full-tile expect_bytes (accounts for both CTAs in pair mode).
    # Shared between fused-KV and split-KV: `KPipeType.bytes` and
    # `VPipeType.bytes` resolve to the same per-CTA values, so we
    # define them once here.
    comptime k_per_cta_bytes = (
        config.BK0 * config.k_rows_per_cta() * size_of[qkv_type]()
    )
    comptime v_per_cta_bytes = (
        config.v_cols_per_cta() * BN * size_of[qkv_type]()
    )
    comptime k_expect_bytes = cta_group * k_per_cta_bytes
    comptime v_expect_bytes = cta_group * v_per_cta_bytes
    comptime qk_expect_bytes = cta_group * (q_bytes + k_per_cta_bytes)

    # Mode-shared K/V producer closures. These cover both fused-KV and
    # split-KV call sites (and the inlined main-loop / peeled-last V in
    # split mode). Captures: `is_leader`, `e`, `cta_group`, `q_bytes`,
    # `q_smem`, `q_elements`, `tt_row_major`, `QType`, `q_async_copy`,
    # `k_bytes_pp`, `v_bytes_pp`, `k_per_cta_bytes`, `v_expect_bytes`,
    # `k_nvp_peer`, `kv_head_idx`, `v_col_offset`, `k_tma_op`,
    # `v_tma_op`, `config`. Caller owns `populate`, `smem_ptr`, and the
    # producer-pipeline acquire/step lifecycle.
    @parameter
    @always_inline
    def _produce_k[
        partial: Bool,
        qk_stage: Int = 0,
        with_q: Bool = False,
    ](
        kv_paged_rows: KVPagedRows,
        smem_ptr: SharedMemPointer[Scalar[qkv_type]],
        mbar: SharedMemPointer[SharedMemBarrier],
        k_num_valid_pages: UInt32,
    ):
        comptime d_idx = qk_stage * config.BK0
        comptime if is_leader:
            comptime q_term = (cta_group * q_bytes if with_q else 0)
            var bytes: Int32 = Int32(q_term)
            comptime if partial:
                bytes += Int32(k_bytes_pp) * Int32(
                    k_num_valid_pages + k_nvp_peer
                )
            else:
                bytes += Int32(cta_group * k_per_cta_bytes)
            expect_bytes_pred(mbar, bytes, e)
        comptime if with_q:
            # Elect-predicated in-PTX by q_async_copy; no if-guard here.
            q_async_copy(
                QType(
                    q_smem + q_elements * qk_stage,
                    tt_row_major[q_elements](),
                ),
                mbar[],
                depth_idx=UInt32(d_idx),
            )
        kv_paged_rows.tma_copy_k[needs_partial=partial](
            k_tma_op,
            smem_ptr,
            mbar[],
            kv_head_idx=kv_head_idx,
            elect=e,
            k_num_valid_pages=k_num_valid_pages,
            depth_offset=UInt32(d_idx),
        )

    @parameter
    @always_inline
    def _produce_v[
        partial: Bool
    ](
        kv_paged_rows: KVPagedRows,
        smem_ptr: SharedMemPointer[Scalar[qkv_type]],
        mbar: SharedMemPointer[SharedMemBarrier],
        v_num_valid_pages: UInt32,
    ):
        comptime if is_leader:
            var v_bytes: Int32
            comptime if partial:
                v_bytes = Int32(cta_group * v_bytes_pp * Int(v_num_valid_pages))
            else:
                v_bytes = Int32(v_expect_bytes)
            expect_bytes_pred(mbar, v_bytes, e)
        kv_paged_rows.tma_copy_v[needs_partial=partial](
            v_tma_op,
            smem_ptr,
            mbar[],
            kv_head_idx=kv_head_idx,
            elect=e,
            num_valid_pages=v_num_valid_pages,
            depth_offset=UInt32(v_col_offset),
        )

    comptime if config.use_fused_kv:
        # ---- Fused KV mode ----
        # Single StagedPipeline with alternating K and V stages.
        # Stages: K0, V0, K1, V1, ...
        # For MHA: padded_qk_depth == padded_ov_depth, rope_depth == 0.
        # num_qk_stages=1 in fused mode.

        var kv_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.k_smem_base()
        )
        # Per-CTA SMEM: halved for pair-CTA.
        comptime kv_stage_elems = (config.padded_ov_depth * BN // cta_group)

        comptime KVPipeType = StagedPipeline[config.num_kv_stages, 1]
        var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}
        kv_pipeline.state._phase = 1  # producer starts at phase 1

        # ---- Peeled: K0 + Q0 on same barrier ----
        var k0_mbar = kv_pipeline.producer_mbar()
        var k0_smem = kv_smem + kv_pipeline.state.index() * UInt32(
            kv_stage_elems
        )
        var kv_paged_rows = kv_lut.populate[
            BN, base_alignment, pair_cta, is_leader
        ](seq_info.prompt_idx, kv_row)
        _produce_k[partial=needs_partial, qk_stage=0, with_q=True](
            kv_paged_rows, k0_smem, k0_mbar, k_nvp
        )
        kv_pipeline.state.step()  # step -> stage 1

        # ---- Q1 (separate barrier) ----
        comptime if fuse_gqa:
            q_gmem_row += UInt32(HalfBM // group)
        else:
            q_gmem_row += UInt32(HalfBM)
        var q1_mbar = mbars.q1_wait_mbar()
        comptime if is_leader:
            expect_bytes_pred(q1_mbar, Int32(cta_group * q_bytes), e)
        # Elect-predicated in-PTX by q_async_copy; no if-guard here.
        comptime q1_smem_offset = q_elements * 1  # num_qk_stages=1
        q_async_copy(
            QType(q_smem + q1_smem_offset, tt_row_major[q_elements]()),
            q1_mbar[0],
        )

        # ---- V0 (reuses kv_paged_rows from K0's populate) ----
        kv_pipeline.producer_acquire()
        var v0_mbar = kv_pipeline.producer_mbar()
        var v0_smem = kv_smem + kv_pipeline.state.index() * UInt32(
            kv_stage_elems
        )
        _produce_v[partial=needs_partial](
            kv_paged_rows, v0_smem, v0_mbar, v_nvp
        )
        kv_pipeline.state.step()

        comptime check_mask = mask.nonfull_sets[BM_mask, BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

        # ---- KV producer loop ----
        # Main body: always full-page (partial=False). When needs_partial,
        # peel off the last iteration for runtime-bounded populate/TMA.
        var main_iters = iter_count
        comptime if needs_partial:
            if main_iters > 0:
                main_iters -= 1
        while main_iters != 0:
            main_iters -= 1
            kv_row += UInt32(config.BN)

            comptime if check_mask:
                if (
                    mask.status(
                        Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                        Index[dtype=DType.int32](BM_mask, BN),
                    )
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            # Produce Kn (full, populate + K TMA)
            kv_pipeline.producer_acquire()
            var kn_mbar = kv_pipeline.producer_mbar()
            var kn_smem = kv_smem + kv_pipeline.state.index() * UInt32(
                kv_stage_elems
            )
            kv_paged_rows = kv_lut.populate[
                BN, base_alignment, pair_cta, is_leader
            ](seq_info.prompt_idx, kv_row)
            _produce_k[partial=False, qk_stage=0](
                kv_paged_rows,
                kn_smem,
                kn_mbar,
                UInt32(KVPagedRows.num_pages),
            )
            kv_pipeline.state.step()
            # Produce Vn (full, reuses kv_paged_rows)
            kv_pipeline.producer_acquire()
            var vn_mbar = kv_pipeline.producer_mbar()
            var vn_smem = kv_smem + kv_pipeline.state.index() * UInt32(
                kv_stage_elems
            )
            _produce_v[partial=False](
                kv_paged_rows,
                vn_smem,
                vn_mbar,
                UInt32(KVPagedRows.num_pages),
            )
            kv_pipeline.state.step()

        # ---- Peeled last iteration (partial pages) ----
        comptime if needs_partial:
            if iter_count > 0:
                kv_row += UInt32(config.BN)
                var _skip_last = False
                comptime if check_mask:
                    if (
                        mask.status(
                            Index[dtype=DType.int32](
                                Int(score_row), Int(kv_row)
                            ),
                            Index[dtype=DType.int32](BM_mask, BN),
                        )
                        == TileMaskStatus.FULL_MASK
                    ):
                        _skip_last = True
                if not _skip_last:
                    k_nvp = _k_num_valid_pages(kv_row + k_row_offset)
                    comptime if is_leader and pair_cta:
                        k_nvp_peer = _k_num_valid_pages(
                            kv_row + UInt32(config.k_rows_per_cta())
                        )
                    v_nvp = _v_num_valid_pages(kv_row)
                    # Produce Kn (partial, populate + K TMA)
                    kv_pipeline.producer_acquire()
                    var kn_mbar = kv_pipeline.producer_mbar()
                    var kn_smem = kv_smem + kv_pipeline.state.index() * UInt32(
                        kv_stage_elems
                    )
                    kv_paged_rows = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, kv_row)
                    _produce_k[partial=True, qk_stage=0](
                        kv_paged_rows, kn_smem, kn_mbar, k_nvp
                    )
                    kv_pipeline.state.step()
                    # Produce Vn (partial, reuses kv_paged_rows)
                    kv_pipeline.producer_acquire()
                    var vn_mbar = kv_pipeline.producer_mbar()
                    var vn_smem = kv_smem + kv_pipeline.state.index() * UInt32(
                        kv_stage_elems
                    )
                    _produce_v[partial=True](
                        kv_paged_rows, vn_smem, vn_mbar, v_nvp
                    )
                    kv_pipeline.state.step()

    else:
        # ---- Split KV mode ----
        # One `populate` per outer iteration yields a shared
        # `kv_paged_rows` whose row-indices feed both the K and V TMA
        # copies (`tma_copy_k` / `tma_copy_v`). K's per-CTA half and
        # pair-CTA peer offsets are derived at comptime from
        # `is_leader` inside KVPagedRows.

        var k_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.k_smem_base()
        )
        var v_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.v_smem_base()
        )
        var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
        var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

        # ---- First tile: K stage 0 (Q0 + populate + K TMA) ----
        var mbark0 = pipeline_k.get_k[qk_stage=0]()  # no wait
        var kv_paged_rows = kv_lut.populate[
            BN, base_alignment, pair_cta, is_leader
        ](seq_info.prompt_idx, kv_row)
        _produce_k[partial=needs_partial, qk_stage=0, with_q=True](
            kv_paged_rows, mbark0.smem.ptr, mbark0.mbar, k_nvp
        )

        # ---- First tile: K stages 1..num_qk_stages-1 (Q + K TMA) ----
        comptime for qk_stage in range(1, config.num_qk_stages):
            mbark = pipeline_k.get_k[qk_stage=qk_stage]()  # no wait
            _produce_k[partial=needs_partial, qk_stage=qk_stage, with_q=True](
                kv_paged_rows, mbark.smem.ptr, mbark.mbar, k_nvp
            )

        pipeline_k.commit_step()

        # Q1 (separate barriers, one per qk_stage).
        comptime if fuse_gqa:
            q_gmem_row += UInt32(HalfBM // group)
        else:
            q_gmem_row += UInt32(HalfBM)
        var q1_mbar = mbars.q1_wait_mbar()

        comptime for qk_stage in range(config.num_qk_stages):
            comptime q_smem_offset = q_elements * (
                config.num_qk_stages + qk_stage
            )
            comptime d_idx = qk_stage * config.BK0
            comptime if is_leader:
                expect_bytes_pred(
                    q1_mbar + qk_stage, Int32(cta_group * q_bytes), e
                )
            # Elect-predicated in-PTX by q_async_copy; no if-guard here.
            q_async_copy(
                QType(q_smem + q_smem_offset, tt_row_major[q_elements]()),
                q1_mbar[qk_stage],
                depth_idx=UInt32(d_idx),
            )

        # ---- V0 (reuses kv_paged_rows from stage-0 populate) ----
        mbarv0 = pipeline_v.get_tile[qk_stage=0]()
        _produce_v[partial=needs_partial](
            kv_paged_rows, mbarv0.smem.ptr, mbarv0.mbar, v_nvp
        )
        pipeline_v.commit_step()

        comptime check_mask = mask.nonfull_sets[BM_mask, BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

        # ---- Main body + peeled last iteration ----
        # Main body: always full tiles (partial=False). When
        # needs_partial, peel off the last iteration for
        # runtime-bounded populate/TMA.
        var main_iters = iter_count
        comptime if needs_partial:
            if main_iters > 0:
                main_iters -= 1
        while main_iters != 0:
            main_iters -= 1
            kv_row += UInt32(config.BN)

            comptime if check_mask:
                if (
                    mask.status(
                        Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                        Index[dtype=DType.int32](BM_mask, BN),
                    )
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            # K stage 0 (full, populate + K TMA, no Q).
            pipeline_k.acquire_k[qk_stage=0]()
            mbark0 = pipeline_k.get_k[qk_stage=0]()
            kv_paged_rows = kv_lut.populate[
                BN, base_alignment, pair_cta, is_leader
            ](seq_info.prompt_idx, kv_row)
            _produce_k[partial=False, qk_stage=0](
                kv_paged_rows,
                mbark0.smem.ptr,
                mbark0.mbar,
                UInt32(k_pages_per_cta),
            )
            # K stages 1..num_qk_stages-1 (full, no Q, reuse rows).
            comptime for k_stage in range(1, config.num_qk_stages):
                pipeline_k.acquire_k[qk_stage=k_stage]()
                mbarkn = pipeline_k.get_k[qk_stage=k_stage]()
                _produce_k[partial=False, qk_stage=k_stage](
                    kv_paged_rows,
                    mbarkn.smem.ptr,
                    mbarkn.mbar,
                    UInt32(k_pages_per_cta),
                )
            pipeline_k.commit_step()

            # V (full, reuses rows from K stage 0's populate).
            pipeline_v.acquire_v()
            mbarvn = pipeline_v.get_tile[qk_stage=0]()
            _produce_v[partial=False](
                kv_paged_rows,
                mbarvn.smem.ptr,
                mbarvn.mbar,
                UInt32(KVPagedRows.num_pages),
            )
            pipeline_v.commit_step()

        # ---- Peeled last iteration (partial pages) ----
        comptime if needs_partial:
            if iter_count > 0:
                kv_row += UInt32(config.BN)
                var _skip_last = False
                comptime if check_mask:
                    if (
                        mask.status(
                            Index[dtype=DType.int32](
                                Int(score_row), Int(kv_row)
                            ),
                            Index[dtype=DType.int32](BM_mask, BN),
                        )
                        == TileMaskStatus.FULL_MASK
                    ):
                        _skip_last = True
                if not _skip_last:
                    k_nvp = _k_num_valid_pages(kv_row + k_row_offset)
                    comptime if is_leader and pair_cta:
                        k_nvp_peer = _k_num_valid_pages(
                            kv_row + UInt32(config.k_rows_per_cta())
                        )
                    v_nvp = _v_num_valid_pages(kv_row)
                    # K stage 0 (partial, populate + K TMA, no Q).
                    pipeline_k.acquire_k[qk_stage=0]()
                    mbark0 = pipeline_k.get_k[qk_stage=0]()
                    kv_paged_rows = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, kv_row)
                    _produce_k[partial=True, qk_stage=0](
                        kv_paged_rows, mbark0.smem.ptr, mbark0.mbar, k_nvp
                    )
                    # K stages 1+ (partial, no Q).
                    comptime for k_stage in range(1, config.num_qk_stages):
                        pipeline_k.acquire_k[qk_stage=k_stage]()
                        mbarkn = pipeline_k.get_k[qk_stage=k_stage]()
                        _produce_k[partial=True, qk_stage=k_stage](
                            kv_paged_rows,
                            mbarkn.smem.ptr,
                            mbarkn.mbar,
                            k_nvp,
                        )
                    pipeline_k.commit_step()

                    # V (partial, reuses rows).
                    pipeline_v.acquire_v()
                    mbarvn = pipeline_v.get_tile[qk_stage=0]()
                    _produce_v[partial=True](
                        kv_paged_rows,
                        mbarvn.smem.ptr,
                        mbarvn.mbar,
                        v_nvp,
                    )
                    pipeline_v.commit_step()
