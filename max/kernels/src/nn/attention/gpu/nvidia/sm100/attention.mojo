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
"""FA4 (Flash Attention 4) configuration for SM100 (Blackwell) kernels."""

from std.math import ceildiv, align_up, align_down, gcd
from std.sys import size_of
from std.sys import get_defined_bool
from std.bit import prev_power_of_two
from std.gpu.globals import WARP_SIZE
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.host.info import B200
from std.gpu.primitives.grid_controls import PDLLevel
from kv_cache.types import _kv_fold_base_ok


comptime EnableForcedOrdering = get_defined_bool[
    "FA4ForcedSoftmaxOrdering", False
]()
comptime EnableEarlyAdd = get_defined_bool["FA4AddEarly", False]()

# Programmatic Dependent Launch level for the SM100 MHA prefill kernel.  On by
# default so back-to-back attention grids in a stream overlap launch/prologue
# latency; disable with `-D MHA_PDL=false`.  When > OFF the kernel emits
# `wait_on_dependent_grids()` / `launch_dependent_grids()` and the dispatch
# attaches the PROGRAMMATIC_STREAM_SERIALIZATION launch attribute.
comptime MHA_PDL_LEVEL = PDLLevel.OVERLAP_AT_END if get_defined_bool[
    "MHA_PDL", True
]() else PDLLevel.OFF

# Bytes per CTA in shared memory that the CUDA runtime reserves for
# its own use; subtracted from `B200.shared_memory_per_multiprocessor`
# to get the usable smem budget for SM100 attention kernels.
comptime SM100_RESERVED_SMEM_BYTES = 1024


struct FA4Config[
    qkv_dtype: DType,
    *,
    rope_dtype: DType = DType.invalid,
    scale_dtype: DType = DType.invalid,
](TrivialRegisterPassable):
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var qk_depth: Int
    var padded_qk_depth: Int  # align_up(qk_depth, swizzle_elems)
    var ov_depth: Int
    var padded_ov_depth: Int
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_S0: Int = 0
    var TMEM_S1: Int
    var TMEM_O0: Int
    var TMEM_O1: Int
    var TMEM_P0: Int
    var TMEM_P1: Int
    var tmem_used: Int
    var num_kv_stages: Int
    var num_qk_stages: Int  # Stages for Q@K' (K loading pipelining)
    var num_pv_stages: Int  # Stages for P@V (P writing pipelining)
    var smem_used: Int
    comptime num_threads: Int = 512  # 2x softmax, 1x correction, 1x other
    var fuse_gqa: Bool
    var swizzle_mode: TensorMapSwizzle
    var use_fused_kv: Bool
    var pair_cta: Bool
    var num_qo: Int
    var page_size: Int
    var is_mla: Bool
    var row_major_v_atoms: Bool
    var row_major_k_atoms: Bool

    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime rope_dtype_size: Int = size_of[Self.rope_dtype]()
    comptime scale_dtype_size: Int = size_of[Self.scale_dtype]()

    comptime MMA_K: Int = 16 if Self.qkv_dtype.is_half_float() else 32
    comptime sm100_smem_carveout = (
        B200.shared_memory_per_multiprocessor - SM100_RESERVED_SMEM_BYTES
    )
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()
    comptime num_correction_cols = 1

    @always_inline
    def BM_eff(self) -> Int:
        """Number of distinct sequence positions per full tile.
        When fuse_gqa, each tile covers BM // group seq positions x group heads.
        """
        if self.fuse_gqa:
            return self.BM // self.group
        return self.BM

    @always_inline
    def cta_group(self) -> Int:
        return 2 if self.pair_cta else 1

    @always_inline
    def PairBM_eff(self) -> Int:
        """Sequence positions covered by both CTAs in a pair."""
        return self.BM_eff() * self.cta_group()

    @always_inline
    def v_cols_per_cta(self) -> Int:
        """V columns stored in this CTA's SMEM."""
        if self.pair_cta:
            return self.padded_ov_depth // 2
        return self.padded_ov_depth

    @always_inline
    def k_rows_per_cta(self) -> Int:
        """K rows stored in this CTA's SMEM."""
        if self.pair_cta:
            return self.BN // 2
        return self.BN

    @always_inline
    def v_row_major(self) -> Bool:
        """Effective row-major (page-dense, chunk-inner) V layout selector.

        Drives BOTH the V TMA producer fold (`tma_copy_v[row_major=...]`) and
        the P@V MMA consumer descriptor (`smem_descriptor[page_dense=...]` /
        `SM100TensorAccumulator[b_page_dense=...]`); a single accessor keeps the
        producer's page-dense SMEM and the consumer's descriptor in agreement.

        Returns True only when the page-dense layout is actually applicable —
        i.e. when the V-side `kv_tma_fold_chunks[row_major=True]` would fold
        (`>= 2`). The `base_ok` geometry comes from the SHARED `_kv_fold_base_ok`
        gate (the single source of truth the predicate also uses), so the two
        cannot drift; a `comptime assert` at the dispatch site still cross-checks
        this accessor against the real fold result (it spans the
        `box_rows == page_size` bridge the shared gate does not).

        Restricted to genuine multi-page paging (`0 < page_size < BN`):
          - This is the only regime where the fold helps: with `page_size >= BN`
            or `page_size == 0` the tile is a single page (`pages_per_iter == 1`)
            that the chunk-outer rank-4 fold already loads in one TMA.
          - It is also the only regime where the rank-5 atom-row coordinate
            (`gmem_row // _SWIZZLE_ATOM_ROWS`) is safe: a block-indirected
            `PagedKVCache` gives `gmem_row = block_idx * stride` with `stride` a
            multiple of `page_size`, so `page_size % 8 == 0` (checked below)
            makes every page row 8-aligned. Continuous / ragged operands
            (`page_size >= BN`) place tiles at arbitrary token offsets that are
            NOT 8-aligned, which would corrupt the atom-row coordinate.

        Gated to single-CTA: under `pair_cta` the descriptor (`BMN =
        v_cols_per_cta() = ov/2`) and the accumulator advance (`b_BMN = MMA_N =
        ov`) disagree on `mn_dim` for the native layout — a fast-follow change.
        SWIZZLE_128B only (the native layout is defined there).
        """
        if not self.row_major_v_atoms:
            return False
        if self.swizzle_mode != TensorMapSwizzle.SWIZZLE_128B:
            return False
        if self.pair_cta:
            return False
        # Multi-page paging only (see docstring): single-page / continuous /
        # ragged stay chunk-outer.
        if not (self.page_size > 0 and self.page_size < self.BN):
            return False
        var gran = self.swizzle_mode.bytes() // Self.qkv_dtype_size
        # base_ok: shared with kv_tma_fold_chunks (single source of truth) —
        # BK % gran == 0, >= 2 chunks, head_size (= ov_depth) divisible by BK.
        if not _kv_fold_base_ok(self.v_cols_per_cta(), gran, self.ov_depth):
            return False
        # geometry_ok (row_major): box_rows == page_size here (page_size < BN),
        # so the TMA sub-tile must split into _SWIZZLE_ATOM_ROWS (= 8) atom-rows.
        # This also guarantees the gmem page rows are 8-aligned (see docstring).
        return self.page_size % 8 == 0

    @always_inline
    def k_row_major(self) -> Bool:
        """Effective row-major (page-dense, chunk-inner) K layout selector.

        K-side analog of `v_row_major()`: drives BOTH the K TMA producer fold
        (`tma_copy_k[row_major=...]`) and the Q@K' MMA consumer descriptor
        (`smem_descriptor[is_k_major=True, page_dense=...]` /
        `SM100TensorAccumulator[b_page_dense=...]` for the k-major B operand).
        A single accessor keeps the producer's page-dense SMEM and the
        consumer's descriptor in agreement (disagreement is silent wrong
        output, not a crash).

        Gated identically to V: `SWIZZLE_128B` only, single-CTA only
        (`not pair_cta`), and genuine multi-page paging
        (`0 < page_size < k_rows_per_cta()` with `page_size % 8 == 0`, so a
        block-indirected `PagedKVCache` gives 8-aligned page rows for the
        rank-5 atom-row coordinate — see `v_row_major()` for the full
        rationale). The `base_ok` geometry comes from the SHARED
        `_kv_fold_base_ok` gate (the K-side `kv_tma_fold_chunks` uses the same
        gate: `BK0 % gran == 0`, `num_chunks >= 2`, `qk_depth % BK0 == 0`), so
        the two cannot drift; a `comptime assert` at the dispatch site still
        cross-checks this against the real fold result.

        K and V share the same constructor-computed default
        (`row_major_{v,k}_atoms = not is_mla and 0 < page_size < BN`); the two
        accessors then apply their own feasibility gates independently.
        """
        if not self.row_major_k_atoms:
            return False
        if self.swizzle_mode != TensorMapSwizzle.SWIZZLE_128B:
            return False
        if self.pair_cta:
            return False
        # Multi-page paging only (see v_row_major docstring): single-page /
        # continuous / ragged stay chunk-outer. `k_rows_per_cta()` is the K
        # tile's seq_k extent (== BN single-CTA, which is enforced above).
        if not (self.page_size > 0 and self.page_size < self.k_rows_per_cta()):
            return False
        var gran = self.swizzle_mode.bytes() // Self.qkv_dtype_size
        # base_ok: shared with the K-side kv_tma_fold_chunks (single source of
        # truth) — BK0 % gran == 0, >= 2 chunks, qk_depth divisible by BK0.
        #
        # The `>= 2 chunks` term naturally restricts the fold to the regime where
        # it helps: in FUSED-KV mode `num_qk_stages == 1` so `BK0 == padded_qk_depth`
        # (multiple gran-chunks per K tile to fold). In split-KV mode `BK0 == gran`
        # (one gran-chunk per stage, separate buffers), so there is one chunk and
        # this returns False — correct, since K already loads one TMA per page
        # per stage there (nothing to fold).
        if not _kv_fold_base_ok(self.BK0, gran, self.qk_depth):
            return False
        # geometry_ok (row_major): box_rows == page_size here (< k_rows_per_cta),
        # so the TMA sub-tile splits into _SWIZZLE_ATOM_ROWS (= 8) atom-rows and
        # the gmem page rows are 8-aligned.
        return self.page_size % 8 == 0

    @always_inline
    def q_nope_bytes(self) -> Int:
        """Q nope region bytes: BM * padded_ov_depth * dtype_size."""
        return self.BM * self.padded_ov_depth * Self.qkv_dtype_size

    @always_inline
    def q_rope_bytes(self) -> Int:
        """Q rope region bytes. Uses rope_dtype_size when set, else dtype_size.
        """
        return self.BM * self.rope_depth() * Self.rope_dtype_size

    @always_inline
    def rope_depth(self) -> Int:
        """Depth of the rope part. Calculated as:
        padded_qk_depth - padded_ov_depth (0 for MHA where qk_depth == ov_depth).
        """
        return self.padded_qk_depth - self.padded_ov_depth

    @always_inline
    def num_rope_buffers(self) -> Int:
        """Number of separate rope smem buffers (fused mode only).

        In fused mode K tiles alternate with V tiles in the pipeline.
        At most ceildiv(num_kv_stages, 2) K tiles can be in-flight
        simultaneously, so we only need that many rope buffers.
        For MHA (rope_depth=0), no rope buffers are needed.
        """
        if self.use_fused_kv and self.rope_depth() > 0:
            return ceildiv(self.num_kv_stages, 2)
        return 0

    @always_inline
    def num_k_scale_bufs(self) -> Int:
        """Number of staged k_scale smem buffers.

        In fused mode, K tiles alternate with V tiles so at most
        ceildiv(num_kv_stages, 2) K tiles are in-flight simultaneously.
        In split mode, each KV stage has its own K buffer.
        Returns 0 when scale_dtype_size == 0 (no per-token scaling).
        """
        if self.scale_dtype_size == 0:
            return 0
        if self.use_fused_kv:
            return ceildiv(self.num_kv_stages, 2)
        return self.num_kv_stages

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        qk_depth: Int,
        ov_depth: Int,
        swizzle_mode: TensorMapSwizzle,
        page_size: Int,
        is_mla: Bool,
        pair_cta: Bool = False,
        num_qo: Int = 2,
        num_qk_stages: Int = 0,
    ):
        # num_qk_stages == 0 (default) derives the optimal Q@K' staging.
        # A nonzero value pins it (used by the in-kernel 1Q/2Q switch, which
        # requires the 1Q variant's staging to match the 2Q config's — see
        # `switch_1q_config`). The caller must pass a value that is valid for
        # this shape, i.e. one the constructor could itself derive for the
        # same `padded_qk_depth`/`swizzle_mode`. If the pinned staging's extra
        # barriers do not fit in smem, the constructor falls back to 1 stage.
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.qk_depth = qk_depth
        self.pair_cta = pair_cta
        self.num_qo = num_qo
        self.page_size = page_size
        self.is_mla = is_mla
        self.MMA_M = 256 if pair_cta else 128
        # num_qo=1 halves BM to MMA_M (=128) — each CTA now covers half as
        # many Q rows. supported() forbids num_qo=1 with pair_cta, so MMA_M
        # is always 128 here when num_qo == 1.
        if num_qo == 1:
            self.BM = self.MMA_M
        else:
            self.BM = 256
        self.fuse_gqa = group > 1 and (self.MMA_M % group == 0) and not is_mla
        comptime if Self.qkv_dtype.is_float8():
            self.swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.swizzle_mode = swizzle_mode
        swizzle_elems = self.swizzle_mode.bytes() // Self.qkv_dtype_size
        self.ov_depth = ov_depth
        self.padded_qk_depth = align_up(qk_depth, swizzle_elems)
        self.padded_ov_depth = align_up(ov_depth, swizzle_elems)

        # we use two q and o
        # determine BN via tmem:
        # 2*BN + 2*ov_depth <= 512 -> BN + ov_depth <= 256
        self.BN = min(
            256,
            align_down(
                (Self.sm100_tmem_cols // 2 - self.padded_ov_depth),
                Self.MMA_K,
            ),
        )
        # page_size == 0 means non-paged (no constraint).
        # page_size >= BN: page contains full tile (page_size % BN == 0).
        # page_size < BN: tile spans multiple pages (BN % page_size == 0).
        if (
            page_size != 0
            and page_size % self.BN != 0
            and self.BN % page_size != 0
        ):
            self.BN = prev_power_of_two(self.BN)
        # Row-major (page-dense) K/V is the default in the multi-page paging
        # regime (0 < page_size < BN); single-page / continuous / ragged
        # (page_size == 0 or >= BN) stay chunk-outer. MLA is structurally
        # chunk-outer (its own MMA warps never fold), so it is excluded here.
        # v_row_major()/k_row_major() still apply the full
        # geometry/swizzle/pair-CTA/_kv_fold_base_ok feasibility gating.
        var page_dense_default = (
            not is_mla and page_size > 0 and page_size < self.BN
        )
        self.row_major_v_atoms = page_dense_default
        self.row_major_k_atoms = page_dense_default
        self.TMEM_S1 = Self.TMEM_S0 + self.BN
        self.TMEM_P0 = Self.TMEM_S0
        self.TMEM_P1 = self.TMEM_S1
        self.TMEM_O0 = self.TMEM_S1 + self.BN
        self.TMEM_O1 = self.TMEM_O0 + self.padded_ov_depth
        self.tmem_used = self.TMEM_O1 + self.padded_ov_depth

        # We have the following resources that need smem barriers:
        # KV: num_kv_stages
        # S: 2
        # C: 2
        # O: 2
        # softmax order: 2
        # q: 1, for Q1 synchronization
        # 4 for `o_pipeline` (2 consumer + 2 producer)
        # we need two per stage
        # Compute staging for Q@K' and P@V operations
        # num_qk_stages: Controls how K loading is pipelined for Q@K' MMA
        # num_pv_stages: Controls how P writing is pipelined for P@V MMA
        #
        # For Q@K': K can be loaded in stages, MMA starts after first stage arrives
        # For P@V: V must be complete, but P writing can be staged to unblock MMA sooner
        #
        # Divisibility constraints:
        # - num_qk_stages must divide padded_depth (for K column splitting)
        # - num_pv_stages must divide BN (for P column splitting)
        # - Both must respect MMA_K alignment (16 elements)
        #
        # Staging infrastructure:
        # - SM100TensorAccumulator.mma (both a_tmem=False/True quadrants)
        #   supports a stage_idx parameter for processing in chunks when
        #   num_stages > 1
        # - KPipeline and VPipeline structs support separate K/V barrier management
        # - FA4MiscMBars is parameterized by num_pv_stages for S barriers
        # - load() loads K in num_qk_stages chunks with separate barriers per stage
        # - store_exp() writes P in num_pv_stages chunks with barriers per stage
        # - mma() loops over qk_stages for Q@K' and pv_stages for P@V
        #
        # Computed staging values:
        # - num_qk_stages: How many chunks to split K processing into for Q@K' MMA
        # - num_pv_stages: How many chunks to split P writing into for P@V MMA
        #
        if is_mla:
            self.num_qk_stages = 1
        elif num_qk_stages != 0:
            self.num_qk_stages = num_qk_stages
        else:
            # Q@K' staging is enabled: MMA processes K in num_qk_stages chunks,
            # allowing register pressure reduction and potential overlap.
            self.num_qk_stages = gcd(
                self.padded_qk_depth // swizzle_elems,
                self.padded_qk_depth // Self.MMA_K,
            )

        # P@V staging requires coordinated changes to store_exp and mma functions:
        # - store_exp must write P in stages and signal barriers per stage
        # - mma must wait for each P stage barrier before processing
        self.num_pv_stages = 2

        var smem_use = 4
        # Compute misc_mbars fixed size (barriers that don't scale with num_kv_stages):
        # - S consumers: 2 * num_pv_stages (num_pv_stages per warp group)
        # - S producers: 2 (1 per warp group)
        # - C barriers: 4 (C0/C1 producer/consumer)
        # - Order barriers: 2 (only when EnableForcedOrdering)
        # - Q1Sync barriers: num_qk_stages (only when num_qo == 2; num_qo=1
        #   shares Q across both pipelines so no Q1Sync slot is needed —
        #   FA4MiscMBars collapses Q1SyncIdx in that mode)
        # - O producers: 2 (O consumers reuse S_consumer[0], not separate)
        # Total fixed = 8 + order_barrier_count + 2*num_pv_stages
        #             + (num_qk_stages if num_qo == 2 else 0)
        comptime order_barrier_count: Int = 2 if EnableForcedOrdering else 0
        misc_mbars_fixed_size = (
            8
            + order_barrier_count
            + 2 * self.num_pv_stages
            + (self.num_qk_stages if num_qo == 2 else 0)
        )
        smem_use += misc_mbars_fixed_size * Self.mbar_size

        rope_depth = self.padded_qk_depth - self.padded_ov_depth

        # smem use is (NOTE: smem uses padded depth):
        # BM*depth*dtype_size + num_kv_stages*(2*mbar_size + BN*depth*dtype_size) <= smem_remaining
        # num_kv_stages <= (smem_remaining - 2*BM*depth*dtype_size) // (2*mbar_size + BN*depth*dtype_size)
        # Q region: when rope_dtype_size > 0, Q nope and Q rope have different
        # dtype sizes (e.g. FP8 nope + BF16 rope for per-token-scale MLA).
        var qk_depth_bytes: Int
        comptime if Self.rope_dtype_size > 0:
            qk_depth_bytes = (
                self.padded_ov_depth * Self.qkv_dtype_size
                + rope_depth * Self.rope_dtype_size
            )
        else:
            qk_depth_bytes = self.padded_qk_depth * Self.qkv_dtype_size
        smem_use += self.BM * qk_depth_bytes
        # q_scale: always 1 buffer (per-token scale only; 0 when no scaling).
        smem_use += self.BM * Self.scale_dtype_size
        # Add space for correction smem when not using tmem for correction.
        # Must match `SM100AttentionSMem.correction_bytes` in smem.mojo: the
        # layout reserves one Float32 slot per softmax thread, i.e.
        # `2 * WARPGROUP_SIZE = 256` Float32 entries (1 KiB) regardless of
        # `num_qo`. In 2Q this equals `BM * num_correction_cols`, but 1Q
        # halves `BM` to 128 and needs the doubling factor here too.
        # Without it, `smem_use` (passed as `shared_mem_bytes` at launch) is
        # 512 bytes short of the smem.mojo layout, and the trailing mbar /
        # tmem_addr regions overflow into unmapped __shared__ on init.
        smem_use += (
            (2 if num_qo == 1 else 1)
            * self.BM
            * Self.num_correction_cols
            * size_of[DType.float32]()
        )

        # We use one of two strategies:
        #  - split kv: more efficient/neater to track smem separately.
        #              nope and rope smem can be tracked together
        #  - fused kv: if the maximum number of `nope`s we can store is odd
        #              then splitting would require us to round down to
        #              an even number of stages. Fusing avoids this.
        # We divide bytes needed by `k` and `v` into shared and k-specific:
        # In pair-CTA mode each CTA stores half of K/V:
        # K: BN/2 rows × full depth, V: full BN rows × ov_depth/2 cols.
        kv_data_elems = self.BN * self.padded_ov_depth
        if pair_cta:
            kv_data_elems //= 2
        bytes_per_kv = (
            kv_data_elems * Self.qkv_dtype_size + 2 * Self.mbar_size
        )  # KV barriers
        kv_rows = self.BN // 2 if pair_cta else self.BN
        bytes_per_k = (
            kv_rows * rope_depth * Self.rope_dtype_size
            + kv_rows * Self.scale_dtype_size
        )  # k scale buffers

        # total k + v bytes is thus
        # fused_pipeline_stages * bytes_per_kv
        #   + ceildiv(fused_pipeline_stages,2) * bytes_per_k
        # If `fused_pipeline_stages` is even, we split the pipelines.

        remaining = Self.sm100_smem_carveout - smem_use
        # remaining >= fused_pipeline_stages * bytes_per_kv
        #   + ceildiv(fused_pipeline_stages,2) * bytes_per_k
        #   >= fused_pipeline_stages * bytes_per_kv
        #   +  (fused_pipeline_stages/2) * bytes_per_k
        #   = fused_pipeline_stages * (bytes_per_kv + bytes_per_k/2)
        fused_stages = remaining // (bytes_per_kv + bytes_per_k // 2)
        # A pinned num_qk_stages > 1 requires the split-KV pipeline (fused
        # mode never stages K), so round an odd stage count down to even to
        # force the split path below.
        if num_qk_stages > 1 and fused_stages % 2 == 1:
            fused_stages -= 1
        bytes_used = (
            fused_stages * bytes_per_kv + ceildiv(fused_stages, 2) * bytes_per_k
        )
        if bytes_used > remaining:
            fused_stages -= 1
            bytes_used = (
                fused_stages * bytes_per_kv
                + ceildiv(fused_stages, 2) * bytes_per_k
            )
        smem_use += bytes_used

        if fused_stages % 2 == 1:  # odd, fused
            self.use_fused_kv = True
            self.num_kv_stages = fused_stages
            self.num_qk_stages = 1
        else:
            self.use_fused_kv = False
            self.num_kv_stages = fused_stages // 2
            if is_mla:
                self.num_qk_stages = 1
            else:
                # we try to split num_qk_stages
                if num_qk_stages != 0:
                    self.num_qk_stages = num_qk_stages
                else:
                    self.num_qk_stages = gcd(
                        self.padded_qk_depth // swizzle_elems,
                        self.padded_qk_depth // Self.MMA_K,
                    )
                # we need an extra bytes
                barrier_bytes_per_stage = (
                    self.num_kv_stages * 2 * Self.mbar_size
                )
                total_smem_use = (
                    smem_use
                    + (self.num_qk_stages - 1) * barrier_bytes_per_stage
                )
                if total_smem_use < Self.sm100_smem_carveout:
                    smem_use = total_smem_use
                else:
                    self.num_qk_stages = 1

        # BK0: K-dimension chunk size for Q@K' per stage
        self.BK0 = self.padded_qk_depth // self.num_qk_stages
        # BK1: Full BN since V loading is not staged (V must be complete
        # for P@V)
        self.BK1 = self.BN
        self.smem_used = smem_use

    def supported(self) -> Bool:
        # Runtime-k partial-page contraction (mma_maybe_partial_k, used only
        # by the non-MLA fa4_mma path) cuts the P@V contraction at the loaded
        # V boundary to avoid reading uninitialized SMEM. That cut is only
        # safe when the loaded region is MMA_K-aligned, i.e. page_size is a
        # multiple of MMA_K. A sub-tile page (page_size < BN) that is not
        # MMA_K-aligned is therefore unsupported here. MLA prefill has its own
        # MMA warps (does not use fa4_mma) and is exempt.
        if (
            not self.is_mla
            and self.page_size != 0
            and self.page_size < self.BN
            and self.page_size % Self.MMA_K != 0
        ):
            return False
        base = (
            self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )
        if self.num_qo == 1:
            # num_qo=1 is single-CTA only (pair-CTA only requires double
            # the seq-len of single-CTA, num_qo=1 is for small seq-len).
            # pair-CTA decreases perf in every benchmark I've tried
            # anyway, so it especially doesn't make sense for small
            # seq-len.
            return (
                base
                and self.qk_depth >= 64
                and self.qk_depth <= 256
                and not self.pair_cta
            )
        if self.pair_cta:
            # Pair-CTA: depth > 64 (depth=64 needs 32B swizzles) and <= 128.
            return base and self.qk_depth > 64 and self.qk_depth <= 128
        return base and self.qk_depth >= 64

    @always_inline
    def with_num_qo(self, num_qo: Int, *, num_qk_stages: Int = 0) -> Self:
        """Reconstruct this config with a different `num_qo` (single-CTA).

        `num_qk_stages == 0` (default) lets the constructor derive the
        optimal staging for the new shape — appropriate for the dispatch-time
        1Q/2Q selection, where each launch config is free-standing. A nonzero
        value pins the staging (see `switch_1q_config`).

        `pair_cta` is forced False because `num_qo == 1` is single-CTA only
        (see `supported()`). Re-passing the stored `swizzle_mode` is faithful:
        the constructor re-derives it (FP8 re-forces 64B), and it is already
        the post-override value here. The `row_major_{v,k}_atoms` fields are
        not re-passed: the constructor recomputes them from
        `page_size`/`BN`/`is_mla`, and `BN` is `num_qo`-independent, so the
        value is identical to `self`'s.
        """
        return Self(
            num_q_heads=self.num_q_heads,
            group=self.group,
            qk_depth=self.qk_depth,
            ov_depth=self.ov_depth,
            swizzle_mode=self.swizzle_mode,
            page_size=self.page_size,
            is_mla=self.is_mla,
            pair_cta=False,
            num_qo=num_qo,
            num_qk_stages=num_qk_stages,
        )

    @always_inline
    def switch_1q_config(self) -> Self:
        """The 1Q variant used by the in-kernel per-sequence 1Q/2Q switch.

        Unlike the dispatch-time conversion (`with_num_qo(1)`), which is free
        to pick the optimal staging, this pins `num_qk_stages` to this (2Q)
        config's value: the switch feeds the 2Q-built TMA ops to the 1Q body,
        so the per-stage K split (`QTMATile`'s smem-tile last dim and
        `k_tma`'s `BK = padded_qk_depth // num_qk_stages`) must match. The
        pinned value is always arithmetically valid here because
        `padded_qk_depth` and `swizzle_mode` are identical across the two
        configs; if its extra barriers do not fit in 1Q smem, the constructor
        falls back to 1 stage and `can_switch_to_1q()` rejects the switch.
        """
        return self.with_num_qo(1, num_qk_stages=self.num_qk_stages)

    @always_inline
    def can_switch_to_1q(self) -> Bool:
        """Whether a 2Q-launched kernel may dispatch to the 1Q body at runtime.

        True only when this is a 2Q single-CTA config AND a valid 1Q variant
        exists whose TMA-op types match the 2Q ones. `switch_1q_config()`
        pins `num_qk_stages` (the one TMA-op parameter that could otherwise
        diverge — `BN`, the per-half Q `BM` (128), `v_tma_op`, and
        `ragged_tma_store` already match), so the equality check below only
        fails when the pinned staging could not be honored (smem fallback to
        1 stage). When this returns False the kernel runs pure 2Q.
        """
        if self.num_qo != 2 or self.pair_cta:
            return False
        var cfg1 = self.switch_1q_config()
        return cfg1.supported() and cfg1.num_qk_stages == self.num_qk_stages

    @always_inline
    def launch_smem_used(self) -> Int:
        """Dynamic smem to reserve when launching this config's kernel.

        When the launched kernel may dispatch to the 1Q body at runtime
        (`can_switch_to_1q()`), it constructs the 1Q `SM100AttentionSMem` over
        the same dynamic smem region, so the launch must reserve the max of
        both footprints. Otherwise this is just `smem_used`.
        """
        if self.can_switch_to_1q():
            return max(self.smem_used, self.switch_1q_config().smem_used)
        return self.smem_used

    def description(self) -> String:
        return String(
            "pair_cta = ",
            self.pair_cta,
            "\nnum_qo = ",
            self.num_qo,
            "\nMMA_M = ",
            self.MMA_M,
            "\nqk_depth = ",
            self.qk_depth,
            "\nBN = ",
            self.BN,
            "\nnum_kv_stages = ",
            self.num_kv_stages,
            "\ntmem_used = ",
            self.tmem_used,
            "\nsmem_used = ",
            self.smem_used,
            "\nsm100_smem_carveout = ",
            Self.sm100_smem_carveout,
            "\nnope_dtype_size = ",
            Self.qkv_dtype_size,
            "\nrope_dtype_size = ",
            Self.rope_dtype_size,
            "\nscale_dtype_size = ",
            Self.scale_dtype_size,
            "\nuse_fused_kv = ",
            self.use_fused_kv,
        )

    def correction_smem_elements(self) -> Int:
        return self.BM * Self.num_correction_cols

    def num_active_warps_per_group(self) -> Int:
        return 4

    def num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()
