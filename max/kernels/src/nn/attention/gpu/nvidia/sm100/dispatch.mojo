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
Provides the SM100 (Blackwell) flash-attention host-side dispatch entry point
that selects a 1Q or 2Q FA4 kernel configuration, builds the Q/K/V/O TMA
descriptors and tile scheduler, and enqueues the kernel onto the device.
"""

from std.collections import OptionalReg
from std.math import ceildiv, clamp
from std.sys import get_defined_int
from max.gpu.primitives.grid_controls import pdl_launch_attributes
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    Dim,
    FuncAttribute,
)
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D
from layout.tma_async import RaggedTMA3DTile
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.logger import Logger
from nn.attention.gpu.nvidia.sm100.attention import FA4Config, MHA_PDL_LEVEL
from nn.attention.gpu.nvidia.common import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    Pack,
    q_tma,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import TransientScheduler
from nn.attention.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    NoPartition,
    OptionallyStaticInt,
    SplitKPartition,
    StaticInt,
)
from .attention_utils import (
    clusters_per_wave,
    kv_sub_tile_rows,
    kv_tma_fold_chunks,
    o_store_tma_blocks_per_op,
    splitk_p_ladder,
)
from .kernel import SM100MHA2Q
from .fa4_splitk_combine import fa4_splitk_combine

comptime logger = Logger()


@always_inline
def _bucket_ws[sm_count: Int](n: Int, p_max: Int) -> Int:
    """Snap a desired workspace-split-K partition count UP to the nearest
    `splitk_p_ladder` rung, then CAP at `p_max` -- a CAPTURE-INVARIANT ceiling
    (see `ws_p_ceiling`: one GPC wave's SM-fill `sm_count // raw_grid`, or a
    shape-driven `target` above it once one wave alone is small).

    The ladder is shared with `fa4_splitk_combine`, which compiles one unrolled
    combine per rung -- so a rung added here automatically gains its
    specialization there. NOTE the cap is NOT snapped: when `p_max` bites, the
    returned `P` is off-rung and the combine takes its generic runtime-`P`
    kernel. That is the common case when the ceiling binds off-rung
    (B200: 37 through `raw_grid <= 24`, then decaying), so the
    static-`P` win is not available there.

    `raw_grid` (`prompt_tiles * kv_heads * batch`) is INVARIANT within a CUDA-graph
    capture: batch is fixed per capture, kv_heads per compilation, and seq len per
    decode/EAGLE config (prefill is uncaptured). So `p_max` is a single fixed value
    per capture -- launching it EXACTLY costs no extra captures. The ladder's job is
    only to bound the distinct `grid.x` values the `num_keys`-driven ramp
    (`by_cache`) produces BELOW `p_max`; the top is reached exactly via the cap, not
    by a rung. Bucket-UP then cap; over-bucketing past `by_cache` (but never past
    `p_max`) is correctness-safe: surplus partitions get empty KV windows, write
    `lse = -inf`, and the combine folds them (M2). Mirrors the
    `_bucket_num_partitions` idiom in `mla_decode_dispatch.mojo`.

    The ladder's rungs below 12 (`2, 4, 6, 8, 10`) exist so a SHORT cache at a
    mid/high-batch
    `raw_grid` (small one-wave, boosted `p_max` -- see `ws_p_ceiling`) doesn't
    over-split: callers used to guarantee `n > 10` before ever reaching this
    ladder (the old crossover gate required the desired count to beat the whole
    cluster ladder top), so starting at 12 never wasted more than one rung of
    slack. M4 removed that gate (desired count and mechanism are chosen
    independently now), so a `by_cache`-bounded `n` as small as 1 can reach here;
    without low rungs it would jump straight to 12 partitions for a cache that
    barely supports 1-2. `2`, `4`, and `10` also double as the fill-all-SM cluster
    candidates (`_cluster_splitk_candidates`), so an `n` that lands exactly on one
    of them buckets to a cluster-compatible value instead of overshooting it.
    """
    comptime _LADDER = splitk_p_ladder[sm_count]()
    comptime for v in _LADDER:
        if n <= v:
            return min(v, p_max)
    return min(sm_count, p_max)


# The `raw_grid` value where `ws_p_ceiling` stops following one GPC wave's
# SM-fill and starts its measured plateau.
comptime WS_RAW_GRID_CLAMP: UInt32 = 4

# Last `raw_grid` covered by the B200 ragged partition sweep.
comptime WS_SWEEP_MAX_RAW_GRID: UInt32 = 24


@always_inline
def ws_p_ceiling[sm_count: Int](raw_grid: UInt32) -> UInt32:
    """Capture-invariant partition-count ceiling for the split-K crossover
    (M4): follows one GPC wave's SM-fill, holds the measured plateau through
    `WS_SWEEP_MAX_RAW_GRID`, then decays toward one partition.

    `raw_grid` is capture-invariant (see `_bucket_ws`), so this is a single
    fixed value per CUDA-graph capture.

    Why plateau at `WS_RAW_GRID_CLAMP == 4`, checked against a measured
    B200 ragged partition sweep (batch sweep at cache=131072): a flat
    MULTIPLE of `one_wave` reproduces the moderate-batch optimum
    (`3*one_wave` at batch=22, `one_wave==6` -> 20, close to the measured
    workspace optimum) but ALSO scales up the already-good small-`raw_grid`
    case into a measured regression (that same `3x` turns batch=4's
    `one_wave==37` into 111, ~45% slower than 37). Clamping `raw_grid` instead
    leaves batch=4's 37 untouched -- it is exactly `sm_count // 4` -- and
    lifts a small one-wave (batch=22's 6) to 37. The forced workspace P-sweep
    at batch=22 peaked at P=47, so 37 undershoots that peak by ~10-25% rather
    than matching it; `4` is the largest clamp that does not push batch=4
    past its own measured optimum (P=49 costs batch=4 ~7-10% versus its P=37
    peak). All four route arms in that sweep peaked in roughly the same
    absolute-P band, so this is shared across 1Q/WS-G/WS-E rather than
    per-route.

    Beyond the sweep, the inverse target keeps the split grid near a fixed
    wave budget instead of growing with `raw_grid`.
    """
    comptime assert (
        WS_SWEEP_MAX_RAW_GRID % WS_RAW_GRID_CLAMP == 0
    ), "WS_RAW_GRID_CLAMP must divide WS_SWEEP_MAX_RAW_GRID exactly"
    comptime oversub = WS_SWEEP_MAX_RAW_GRID // WS_RAW_GRID_CLAMP
    var g = max(raw_grid, 1)
    return clamp(
        UInt32(sm_count) * oversub // g,
        1,
        UInt32(sm_count) // min(g, WS_RAW_GRID_CLAMP),
    )


@always_inline
def _visible_keys[
    MaskType: MHAMask, //, BM_mask: Int, BN: Int, page_size: Int
](mask: MaskType, num_keys: UInt32) -> UInt32:
    """The key count a split-K partition ladder should actually be sized
    against: how many keys the mask leaves visible, not how long the cache is.

    Split-K `P` divides the key range that a SINGLE query tile iterates, so the
    right measure is that per-tile band width, `num_keys - start_column`. This is
    the same quantity `mha.mojo`'s FA2 decode correction computes
    (`partition_num_keys`), and the same frame the FA4 warps work in
    (`load_warp.mojo` iterates KV from `start_column`; `mma_warp.mojo` calls its
    `num_keys - start_column` `v_eff_keys`).

    Evaluated at the LAST query row, which needs no prefill/decode distinction:

    * `Null` / `Causal` / `CausalPadding`: `start_column` is identically 0, so
      the row is irrelevant and the result is exactly `num_keys` -- every such
      shape keeps a bit-identical `P`, and this whole correction is inert.
    * `SlidingWindow*` / `Chunked`: the band is ~`window` wide at EVERY row, and
      the last row is the one whose band width this expression reports
      faithfully (`num_keys - (num_keys - 1 - window)`). Taking the first row
      instead would report `window + max_prompt_len` and under-correct long
      windowed prefill.

    So there is deliberately no `max_prompt_len` test and no prefill/decode
    split here.

    WHY `start_column` AND NOT `total_iters`: `total_iters` is the tighter
    quantity (it also bounds the RIGHT edge) but it is NOT host-callable for
    every mask -- `CausalPaddingMask.total_iters` dereferences its
    `valid_lengths` tensor, which lives in DEVICE memory, so calling it from this
    host-side dispatch segfaults. `start_column` is host-safe for every mask in
    the tree (`CausalPaddingMask` delegates it to `CausalMask`, and
    `MaterializedMask`'s naive scan exits on the first tile because its
    `status()` is a constant `PARTIAL_MASK`), and `mha.mojo` already depends on
    exactly that property. Using `num_keys` as the right edge instead is
    conservative: it can only over-estimate the band, i.e. never under-partition.

    COST: O(1) for every mask that can reach here. The only two masks whose
    `start_column` still resolves through the linear
    `naively_get_first_nonempty_mask_col` scan are `MaterializedMask` and
    `AndMask`, and both report `UNKNOWN_MASK`, which every caller gates on
    (`ws_mask_ok` / `ws_e_mask_ok` / `ws1q_mask_ok`) before calling this --
    split-K is comptime-pruned for them. `OrMask` -- hence `ChunkedCausalMask`,
    the Llama4 production mask -- is closed-form (`mha_mask.mojo`'s
    `max(lhs, rhs)`), so it costs two child `start_column` calls, not a scan.

    Parameters:
        MaskType: Concrete `MHAMask` implementation to query (inferred).
        BM_mask: Query tile height the kernel masks against, i.e. the route's
            `config.PairBM_eff()` -- matching every FA4 warp's `BM_mask`.
        BN: Key tile width in columns.
        page_size: The KV cache page size in key columns (0 or 1 if unpaged).

    Args:
        mask: The mask functor.
        num_keys: Raw batch-max KV length (`max_cache_valid_length`).

    Returns:
        Visible key count in `[0, num_keys]`, for `P` sizing ONLY. The kernel
        must still receive the RAW `num_keys` as its key range.
    """
    if num_keys == 0:
        return 0
    # `start_column <= num_keys - 1` for every mask, so this cannot underflow
    # and is always >= 1.
    return num_keys - mask.start_column[BM_mask, BN, page_size](
        # Pre-launch dispatch is batch-aggregate; no per-sequence id is
        # available. Masks whose `start_column` depends on `seq_id` must not size
        # `P` through here (the same caveat the FA2 decode correction in
        # `mha.mojo` carries).
        UInt32(0),
        num_keys - 1,
    )


@always_inline
def _cluster_splitk_candidates[sm_count: Int]() -> List[Int]:
    """Cluster/DSMEM split-K candidate set: ONLY partition counts whose
    per-GPC tiling wastes zero SMs (`clusters_per_wave[C] * C == sm_count`).

    A cluster/DSMEM launch is co-resident within a GPC (`clusters_per_wave`),
    so a `C` that does not evenly divide every GPC size leaves idle SMs once
    oversubscribed past one wave -- the GPC-fragmentation cliffs the old
    per-route ladders (`SPLITK_CANDIDATES` / `WS_E_SPLITK_CANDIDATES`) had to
    scan around via `fits_wave`. Restricting to the zero-waste sizes means a
    cluster/DSMEM launch can be oversubscribed across multiple waves exactly
    as cheaply as the workspace route (no per-wave idle-SM cliff), so
    partition-count and mechanism become independent choices in
    `mha_sm100_dispatch`'s crossover: pick one `P` via the shared
    `ws_p_ceiling` / `_bucket_ws` formula, then use cluster/DSMEM iff `P` is
    one of these, else workspace.

    B200 (148 SMs, GPCs `{20,18,10,2}` x `{3,4,1,3}`): only `C=2` divides
    every GPC size. B300 (160 SMs, uniform 8x20 GPCs): every divisor of 20
    tiles with zero waste; keep the already-validated even candidates
    `{2,4,10}` (skip the odd `5` and the extreme `20`). Every candidate
    returned is already a member of some existing per-route ladder, so this
    only prunes the auto-selected set -- it adds no new kernel
    instantiations. B300 is untested in this repo today (no local/remote
    B300 hardware), matching `clusters_per_wave`'s own unverified B300
    branch; the `comptime assert`s below at least keep it self-consistent
    with `clusters_per_wave` if that GPC model ever changes.
    """
    comptime if sm_count == 148:
        comptime assert (
            clusters_per_wave[2, sm_count]() * 2 == sm_count
        ), "B200 cluster candidate 2 does not fill all SMs"
        return [2]
    elif sm_count == 160:
        comptime for c in [2, 4, 10]:
            comptime assert (
                clusters_per_wave[c, sm_count]() * c == sm_count
            ), "B300 cluster candidate does not fill all SMs"
        return [2, 4, 10]
    else:
        comptime assert (
            False
        ), "_cluster_splitk_candidates: only B200 (148) / B300 (160) modeled"


@always_inline
def _ws_splitk_force_pin(fsk: Int) -> Int:
    """Pins a `FA4_WS_SPLITK_FORCE` value to a validated WS split-K P.

    Returns `fsk` when it is one of the validated even partition counts
    {2, 4, 6, 8, 10, 16} (a superset of `CLUSTER_SPLITK_CANDIDATES`, the
    auto-selected set, which is trimmed for compile time -- see
    `_cluster_splitk_candidates`), else 1 (single-CTA WS). Shared by the
    Layout-G and Layout-E force routes so the pin set lives in one place.
    """
    return fsk if (
        fsk == 2 or fsk == 4 or fsk == 6 or fsk == 8 or fsk == 10 or fsk == 16
    ) else 1


@always_inline
def mha_sm100_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: UnsafePointer[Scalar[q_type], _],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask: MaskType,
    valid_length: UnsafePointer[UInt32, _],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[DType.uint32]],
    batch_size_arg: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
    # Caller-supplied EXACT split-K partition count (`mha.mojo`'s
    # `num_partitions`). `0` => auto, i.e. the `ws_p_ceiling` / `_bucket_ws`
    # ladder picks `P`. Non-zero is honored verbatim rather than bucketed --
    # pinning `P` is the whole point (determinism tests, partition sweeps), so
    # snapping it to a ladder rung would defeat the caller. See the
    # `num_partitions_override` handling in the num_q dispatch below for what
    # "honored" costs: it also forces BM < 256, because the 2Q route has no
    # split-K at all.
    num_partitions_override: Int = 0,
) raises:
    """Dispatches the SM100 FA4 flash-attention kernel for a prefill or decode workload.

    Selects between the 1Q split-K and 2Q FA4 configurations based on a
    occupancy and prompt-length heuristic, constructs the Q/K/V/O TMA tile
    descriptors and transient tile scheduler, threads optional ragged
    valid-length, KV-row-offset, and sink-attention arguments through to the
    compiled kernel, and enqueues the launch onto the supplied device context.

    Decode is not a separate route: a single-token prompt is just the shortest
    prefill, so it takes the same ladder and lands on the smallest tile that
    single-tiles it. `max_prompt_len_arg` must be dynamic even when it is 1 --
    see the note on the missing `_is_decoding` guard in the body.

    Parameters:
        q_type: Element type of the query tensor (inferred).
        KVType: Key/value operand descriptor with dtype and page size
            (inferred).
        MaskType: Attention mask scheme applied to the Q@K' scores (inferred).
        output_type: Element type of the attention output buffer (inferred).
        MaxPromptLenType: Optionally-static type encoding the maximum prompt
            length (inferred).
        config: MHA configuration supplying dtype, head count, depth, and
            swizzle mode.
        group: Number of query heads per KV head (GQA group size).
        ragged: Whether to dispatch the variable-length valid-length path.
        sink: Whether to thread sink-attention weights into the kernel.
        _is_cache_length_accurate: Whether the supplied cache length is
            accurate, threaded to the compiled kernel.

    Args:
        output: Device buffer that receives the attention output rows.
        q_arg: Pointer to the query tensor data.
        k: Key operand descriptor.
        v: Value operand descriptor.
        num_rows_q: Number of query rows to attend over.
        mask: Attention mask applied to the Q@K' scores.
        valid_length: Per-row valid KV length pointer, used when `ragged` is set.
        max_prompt_len_arg: Maximum prompt length, optionally static.
        max_cache_valid_length_arg: Maximum valid KV cache length across the batch.
        scale: Scalar applied to the Q@K' product before softmax.
        kv_input_row_offsets: Optional per-row KV input offsets for ragged layouts.
        batch_size_arg: Number of sequences in the batch.
        ctx: Device context used to build TMA descriptors and enqueue the kernel.
        sink_weights: Optional sink-attention weights used when `sink` is set.
        num_partitions_override: Exact split-K partition count to use, or `0`
            for the automatic ladder. A non-zero value is honored verbatim
            (cluster/DSMEM when it is a `CLUSTER_SPLITK_CANDIDATES` member,
            else the workspace route, which admits any `P`) and additionally
            forces a BM < 256 route, since 2Q cannot split at all. Overriding
            bypasses the capture-invariant `ws_p_ceiling` ceiling and the
            measured Layout-E workspace guards.
    """
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match for FA3."
    # NOTE: there is deliberately NO `_is_decoding[MaxPromptLenType]()` guard
    # here. Decode traffic is served: `mha.mojo` routes every SM100
    # `depth <= 128` half-float/fp8 shape here regardless of
    # `is_token_generation`, and a `seq_len == 1` prompt arrives as a
    # `DynamicInt(1)` that simply lands on the smallest single-tiling tile.
    #
    # What is still unsupported is a *statically* known `1`: the Q TMA builder
    # below hard-codes `decoding=False` (see `decoding=False` in the `QTMATile`
    # construction), and the decoding Q view is `rows x depth` where prefill's
    # is `rows x (depth * num_heads)` -- a `StaticInt[1]` caller would silently
    # get the wrong descriptor. `kernel.mojo`'s
    # `comptime assert _is_decoding[Self.MaxSeqLenType]() == False` catches that
    # one level down and is the fail-loud backstop. Do not reinstate a guard
    # here: it would re-block the runtime-1 decode route above.
    comptime fa4_config_2q = FA4Config[KVType.dtype](
        num_q_heads=config.num_heads,
        group=group,
        qk_depth=config.depth,
        ov_depth=config.depth,
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
        is_mla=False,
    )
    comptime assert fa4_config_2q.supported(), fa4_config_2q.description()

    var q = q_arg.bitcast[Scalar[KVType.dtype]]().unsafe_origin_cast[
        q_arg.origin
    ]()

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)

    @__parameter
    @always_inline
    def with_fa4_config[
        NumPartitionsType: OptionallyStaticInt,
        PartitionType: MHAPartitionScheme,
        OWorkspaceType: OptionalPointer,
        //,
        fa4_config: FA4Config[KVType.dtype],
    ](
        num_partitions: NumPartitionsType,
        partition: PartitionType,
        # Workspace (traditional/unfused) split-K O target. Supplied only when
        # `PartitionType.do_partition`: the O store is redirected here (a
        # `[P, num_rows_q, num_q_heads, ov_depth]` buffer) instead of `output`,
        # and a separate combine kernel merges it into `output`. Every other
        # launch omits it and infers the null conformer from this default.
        o_workspace_ptr: OWorkspaceType = NullPointer[output_type](),
    ) raises:
        # `num_partitions` is ALWAYS static: the 2Q config (and its in-kernel 1Q
        # switch) carry a static 1, and each num_q==1 split-K config is compiled
        # once per static partition count `P` (cluster size `P`) — the dispatch
        # picks which `P` kernel to launch at runtime. So the cluster size is a
        # comptime constant that matches the kernel's static `nvvm.cluster_dim`.
        comptime assert Bool(
            NumPartitionsType.static_value
        ), "split-K num_partitions must be static (compiled once per P)"
        # The O-store redirect and the partition scheme are one decision:
        # workspace split-K writes partials to `o_partial`, every other route
        # writes straight to `output`. Biconditional, so BOTH halves of a
        # mismatch are a compile error rather than a TMA descriptor built over a
        # dangling address.
        comptime assert PartitionType.do_partition == (
            not OWorkspaceType.is_null
        ), (
            "workspace split-K must supply the O-partial target, and only it"
            " may: `do_partition` and the O pointer must agree"
        )
        comptime assert (
            OWorkspaceType.is_null or OWorkspaceType.dtype == output_type
        ), "the O-partial workspace must match the kernel's output dtype"
        comptime swizzle_mode = fa4_config.swizzle_mode
        # O output store is row-major SWIZZLE_NONE (decoupled from the swizzled
        # Q/K/V/S/P buffers governed by `swizzle_mode`). The softmax warp loads
        # O one-row-per-thread and writes it row-major, avoiding cross-thread
        # shuffles and swizzling while staying bank-conflict-free.
        comptime output_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
        comptime BM = fa4_config.BM
        comptime fuse_gqa = fa4_config.fuse_gqa
        comptime num_threads = fa4_config.num_threads
        # `MMA_M // cta_group` drives q_tma_op and ragged_tma_store BM under a
        # unified expression: 128 for 2Q single-CTA (128 // 1), 2Q pair-CTA
        # (256 // 2), and 1Q single-CTA (128 // 1); 32 for the WS BM=32 config
        # (MMA_M=32 // 1).
        comptime BM_per_mma = fa4_config.MMA_M // fa4_config.cta_group()
        comptime assert BM == 32 or BM == 64 or BM == 128 or BM == 256

        # Batch the O store into one TMA per issuer: the box covers
        # `ceil(n_blocks/2)` swizzle-granularity blocks, so the single-issuer
        # writeback emits 2 pipelined copies and the 1Q combine emits 1 per WG
        # (vs `n_blocks` per-block copies). Fused GQA (group > 1) batches too —
        # the RaggedTMA3DTile (middle_dim, rows) selector merge keeps it within
        # the 5D TMA limit (rank-5; rank-4 for group==1). Only swizzled-output
        # callers fall back to per-block (0). Shared formula keeps this in sync
        # with the kernel param type.
        # 1Q split-K (reduce-scatter): each partition CTA TMA-stores only its OWN
        # depth-column BAND via per-block `async_copy_from_col`, and the band
        # offset `p*ceil(blocks/P)` is not a {0, half} batched-box boundary for
        # P>=4. So the split-K config needs the PER-BLOCK (rank-3) O-store
        # descriptor, NOT the batched one (which only `async_copy_batched` over
        # the two {0, ceil(blocks/2)} halves can drive). `fa4_splitk_combine_write`
        # infers `tma_bpo==0` from this store and takes its per-block path. Every
        # non-split config keeps the batched store (single-issuer/intra-CTA WG
        # combine, where the {0, half} boxes hold).
        # The WS (MMA_M=32) combine likewise TMA-stores from WG0 via the
        # PER-BLOCK `fa4_tma_store_o_smem` (the B200-verified egress), so it too
        # needs the rank-3 store rather than the batched {0, half} box.
        comptime store_blocks_per_op = 0 if (
            fa4_config.splitk_partitions > 1 or fa4_config.use_ws
        ) else o_store_tma_blocks_per_op[
            output_type,
            output_swizzle_mode,
            fa4_config.ov_depth,
            fa4_config.group if fuse_gqa else 1,
            depth_splits=2,
        ]()

        comptime RaggedStoreType = RaggedTMA3DTile[
            output_type,
            output_swizzle_mode,
            BM=BM_per_mma,
            BN=fa4_config.ov_depth,
            middle_dim=fa4_config.num_kv_heads if fuse_gqa else fa4_config.num_q_heads,
            group=fa4_config.group if fuse_gqa else 1,
            tma_blocks_per_op=store_blocks_per_op,
        ]

        # Workspace (traditional/unfused) split-K redirects the O store to the
        # per-partition workspace `o_partial` with a P-extended row axis; the
        # kernel writes partition `p`'s tile at ragged row `p*num_rows_q + token`
        # and a separate combine kernel merges into `output`. Every other config
        # stores straight to `output`. `do_partition` is comptime, so exactly one
        # branch is codegen'd.
        var ragged_tma_store: RaggedStoreType
        comptime if PartitionType.do_partition:
            ragged_tma_store = RaggedStoreType.create(
                ctx,
                # `OptionalPointer.value()` is immutable — its other users are
                # all read-only inputs, whereas this is a store target. `rebind`
                # reconciles `OWorkspaceType.dtype` with `output_type`, which the
                # assert above pins to the same dtype (the tie is not expressible
                # through the trait).
                rebind[UnsafePointer[Scalar[output_type], MutAnyOrigin]](
                    o_workspace_ptr.value().unsafe_mut_cast[True]()
                ),
                rows=Int(partition.max_num_partitions()) * num_rows_q,
            )
        else:
            ragged_tma_store = RaggedStoreType.create(
                ctx,
                output.unsafe_ptr(),
                rows=num_rows_q,
            )

        var q_tma_op = q_tma[
            swizzle_mode,
            BM=BM_per_mma,
            depth=fa4_config.qk_depth,
            q_num_heads=fa4_config.num_q_heads,
            group=fa4_config.group,
            decoding=False,
            fuse_gqa=fuse_gqa,
            num_qk_stages=fa4_config.num_qk_stages,
        ](ctx, q, num_rows_q)
        # Depth-chunk TMA fold (SM100): fold the BK0 (K) / v_cols_per_cta (V)
        # depth chunks into one rank-4 TMA when byte-equivalent. Each
        # `kv_tma_fold_chunks` is the single source of truth shared with the
        # `tma_copy_k` / `tma_copy_v` issue sites in `load_warp.mojo`; the builder
        # and issue site must pass identical args so the baked descriptor rank and
        # issue-coord rank agree.
        #   K: smem_BN == k_rows_per_cta (K's per-CTA tile_rows); box_rows ==
        #      kv_sub_tile_rows(k_rows_per_cta, page_size).
        #   V: smem_BN == BN (V's tile_rows, num_v_sub_tiles == 1); box_rows ==
        #      kv_sub_tile_rows(BN, page_size).
        comptime k_sub_BN = kv_sub_tile_rows(
            fa4_config.k_rows_per_cta(), KVType.page_size
        )
        comptime k_row_major = fa4_config.k_row_major()
        comptime k_fold_chunks = kv_tma_fold_chunks[
            KVType.dtype,
            fa4_config.swizzle_mode,
            BK=fa4_config.BK0,
            head_size=fa4_config.qk_depth,
            box_rows=k_sub_BN,
            smem_BN=fa4_config.k_rows_per_cta(),
            page_size=KVType.page_size,
            row_major=k_row_major,
        ]()
        # Producer/consumer agreement: if the Q@K' consumer reads the page-dense
        # (row-major) K layout, the producer MUST have actually folded it
        # (`k_fold_chunks >= 2`). `k_row_major()` mirrors this predicate, so a
        # mismatch here means the two drifted.
        comptime assert (not k_row_major) or (
            k_fold_chunks > 1
        ), "k_row_major() implies the K row-major fold; predicate drift"
        var k_tma_op = k.create_tma_tile[
            fa4_config.swizzle_mode,
            BN=k_sub_BN,
            depth=fa4_config.qk_depth,
            BK=fa4_config.BK0,
            fold_chunks=k_fold_chunks,
            row_major=k_row_major,
        ](ctx)
        # V TMA box geometry per layout comes from `v_tma_box_rows()` /
        # `v_tma_box_cols()` -- the one selector dispatch, kernel, and the
        # `fa4_load` signature all route through (see their docstrings). Byte
        # layout for Layout-E pinned by `test_ws_v_layout_e_probe.mojo`
        # (CONTIGUOUS `mn = p*ov_depth + d`).
        #
        # `v_row_major` / `v_fold_chunks` stay inline scalar ternaries (NOT a
        # `comptime if`/`else` branching the whole `create_tma_tile`): Mojo
        # unifies a `var`'s type structurally across both branches, so two
        # different `TMATensorTile` shapes assigned to `v_tma_op` in separate
        # branches fail to parse. Layout-E forces them False/1 (Phase 1:
        # correctness-only, no row-major page-dense fold).
        comptime v_sub_BN = fa4_config.v_tma_box_rows(KVType.page_size)
        comptime v_sub_cols = fa4_config.v_tma_box_cols()
        comptime v_row_major = (
            False if fa4_config.m_pack == 2 else fa4_config.v_row_major()
        )
        comptime v_fold_chunks = 1 if fa4_config.m_pack == 2 else kv_tma_fold_chunks[
            KVType.dtype,
            fa4_config.swizzle_mode,
            BK=v_sub_cols,
            head_size=fa4_config.ov_depth,
            box_rows=v_sub_BN,
            smem_BN=fa4_config.BN,
            page_size=KVType.page_size,
            row_major=v_row_major,
        ]()
        # Producer/consumer agreement: if the P@V consumer reads the page-dense
        # (row-major) V layout, the producer MUST have actually folded it
        # (`v_fold_chunks >= 2`). `v_row_major()` mirrors this predicate, so a
        # mismatch here means the two drifted. Trivially holds for Layout-E
        # (`v_row_major` forced False).
        comptime assert (not v_row_major) or (
            v_fold_chunks > 1
        ), "v_row_major() implies the V row-major fold; predicate drift"
        var v_tma_op = v.create_tma_tile[
            fa4_config.swizzle_mode,
            BN=v_sub_BN,
            depth=fa4_config.ov_depth,
            BK=v_sub_cols,
            fold_chunks=v_fold_chunks,
            row_major=v_row_major,
        ](ctx)
        comptime PairBM_eff = fa4_config.PairBM_eff()
        comptime SchedulerType = TransientScheduler[
            UInt32(PairBM_eff),
            UInt32(
                fa4_config.num_kv_heads if fuse_gqa else fa4_config.num_q_heads
            ),
            flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
            pair_cta=fa4_config.pair_cta,
            splitk_partitions=UInt32(fa4_config.splitk_partitions),
        ]
        var scheduler: SchedulerType = SchedulerType()

        @__parameter
        @always_inline
        def with_sink[SinkType: OptionalPointer](sink_ptr: SinkType) raises:
            @__parameter
            @always_inline
            def with_kv_offsets[
                KVRowOffsetsType: OptionalPointer
            ](kv_row_offsets: KVRowOffsetsType) raises:
                @__parameter
                @always_inline
                def with_valid_length[
                    ValidLengthType: OptionalPointer
                ](valid_len: ValidLengthType) raises:
                    # the pack contains all possibly 0-sized objects
                    comptime PackType = Pack[
                        MaskType,
                        SchedulerType,
                        ValidLengthType,
                        SinkType,
                        KVRowOffsetsType,
                        MaxPromptLenType,
                        PartitionType,
                    ]
                    var pack: PackType = {
                        mask,
                        scheduler,
                        valid_len,
                        sink_ptr,
                        kv_row_offsets,
                        max_prompt_len_arg,
                        partition,
                    }

                    var max_num_prompt_tiles: UInt32 = ceildiv(
                        max_prompt_len_arg.as_uint32(), UInt32(PairBM_eff)
                    )
                    # Over-launch the prompt-tile (x) axis by the partition count
                    # for the traditional (workspace) split-K scheme. For
                    # `NoPartition` this is `*1` (byte-identical to the non-split
                    # launch); for `SplitKPartition` it widens x by `P` so the grid
                    # has one CTA per (tile, partition). This matches `get_seq_info`,
                    # which inflates the scheduler tile space by the runtime
                    # `partition.num_partitions()` (see `common.mojo`). No cluster is
                    # formed (workspace split-K keeps `splitk_partitions == 1`, so
                    # `cluster_size()==1` and `cluster_dim=None`); over-launched CTAs
                    # that map past the real prompt are marked invalid by
                    # `SeqInfo.is_valid` and early-return.
                    var block_x: UInt32 = (
                        max_num_prompt_tiles * partition.max_num_partitions()
                    )
                    logger.info(
                        "------ Dispatching to SM100 FMHA-",
                        fa4_config.num_q,
                        "Q ------",
                    )
                    logger.info(
                        "QKV Type:",
                        KVType.dtype,
                        "Depth:",
                        fa4_config.qk_depth,
                        "Number of Q // KV Heads:",
                        fa4_config.num_q_heads,
                        "//",
                        fa4_config.num_kv_heads,
                        "Batch Size:",
                        batch_size,
                        "Max Num Prompt Tiles:",
                        max_num_prompt_tiles,
                    )

                    # Covers the in-kernel 1Q/2Q switch: when
                    # `can_switch_to_1q()` the kernel constructs the 1Q smem
                    # layout over the same dynamic smem region, so this is the
                    # max of both footprints (see `FA4Config.launch_smem_used`).
                    comptime smem_use = fa4_config.launch_smem_used()

                    comptime KernelStruct = SM100MHA2Q[
                        KVType,
                        output_type,
                        MaskType,
                        SchedulerType,
                        fa4_config,
                        ValidLengthType,
                        SinkType,
                        KVRowOffsetsType,
                        _is_cache_length_accurate,
                        MaxPromptLenType,
                        PartitionType,
                    ]
                    # Every config uses the static `kernel` entry; the cluster
                    # size is baked into its `nvvm.cluster_dim` metadata.
                    comptime kernel = KernelStruct.kernel

                    var cluster_dim: OptionalReg[Dim] = None
                    # Unifies pair-CTA (cluster_size==2) and num_q==1 split-K
                    # (cluster_size==P): both are the comptime `cluster_size()`,
                    # matching the static `nvvm.cluster_dim` metadata on the
                    # `kernel` entry (split-K is compiled once per static P).
                    comptime if fa4_config.cluster_size() > 1:
                        cluster_dim = Dim(fa4_config.cluster_size(), 1, 1)
                    comptime name = String(
                        "nq",
                        fa4_config.num_q,
                        "d",
                        fa4_config.qk_depth,
                        "qh",
                        fa4_config.num_q_heads,
                        "kvh",
                        fa4_config.num_kv_heads,
                        ".",
                    )
                    ctx.enqueue_function[kernel](
                        q_tma_op,
                        k_tma_op,
                        v_tma_op,
                        ragged_tma_store,
                        k,
                        scale,
                        batch_size,
                        max_cache_valid_length,
                        pack,
                        # Total query-row count, used only by the workspace
                        # (traditional/unfused) split-K egress as the per-partition
                        # `o_partial`/`lse_partial` row stride; harmless (unused)
                        # for every other config.
                        UInt32(num_rows_q),
                        grid_dim=SchedulerType.grid_dim(
                            batch_size, block_x, num_partitions.as_uint32()
                        ),
                        block_dim=(num_threads, 1, 1),
                        cluster_dim=cluster_dim,
                        shared_mem_bytes=smem_use,
                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                            UInt32(smem_use)
                        ),
                        attributes=pdl_launch_attributes(MHA_PDL_LEVEL),
                    )

                # --- ragged dispatch ---
                comptime if ragged:
                    with_valid_length[NonNullPointer[DType.uint32]](
                        {valid_length.as_imm().as_unsafe_any_origin()}
                    )
                else:
                    with_valid_length[NullPointer[DType.uint32]]({})

            # --- kv_input_row_offsets dispatch ---
            if kv_input_row_offsets:
                with_kv_offsets[NonNullPointer[DType.uint32]](
                    {kv_input_row_offsets.value().ptr}
                )
            else:
                with_kv_offsets[NullPointer[DType.uint32]]({})

        # --- sink dispatch ---
        comptime if sink:
            with_sink[NonNullPointer[KVType.dtype]](
                {
                    rebind[UnsafePointer[Scalar[KVType.dtype], ImmutAnyOrigin]](
                        sink_weights.value().ptr
                    )
                }
            )
        else:
            with_sink[NullPointer[KVType.dtype]]({})

    @__parameter
    @always_inline
    def launch_workspace[fa4_config: FA4Config[KVType.dtype]](p: UInt32) raises:
        """Traditional (unfused / workspace) split-K launch: run the plain 1Q
        config over an over-launched grid that writes each partition's
        locally-normalized `O_p/l_p` + fused per-row LSE to a transient
        `o_partial`/`lse_partial` global workspace, then merge them into `output`
        with a separate `fa4_splitk_combine` kernel (same stream, so it runs after
        the attention completes). No launch cluster, no in-kernel DSMEM combine.

        `p` is both the grid over-launch factor (`block_x *= p`) and the
        workspace row extent, and every launched CTA writes a real or
        `-inf`-empty row, so it is also the count the combine reduces.
        `SplitKPartition` keeps `num`/`max` as separate fields (FA2 decode does
        distinguish them); should this route ever over-launch, split this
        parameter rather than the struct.

        The config itself stays at `splitk_partitions == 1` (so
        `cluster_size() == 1` and `cluster_dim=None`) and `P` is carried purely
        at runtime by the `SplitKPartition`. That partition SCHEME is a comptime
        TYPE flowing into `with_fa4_config`, which is why this is a separate
        generic `def` rather than a `comptime if` selecting a shared
        `var partition`: Mojo unifies a `var`'s type across both branches, so
        both ctors would have to type-check. Same idiom as
        `with_valid_length[NonNullPointer]` vs `[NullPointer]`.
        """
        # `o_partial`: `[p, num_rows_q, num_q_heads, ov_depth]` locally-
        # normalized partials. `lse_partial`: one fused-LSE f32 per (partition,
        # row, q-head), `[p, num_rows_q, num_q_heads]` -- matching the O-store
        # TMA row extent.
        var rows_x_heads = num_rows_q * fa4_config.num_q_heads
        var lse_partial = ctx.enqueue_create_buffer[DType.float32](
            Int(p) * rows_x_heads
        )
        var o_partial = ctx.enqueue_create_buffer[output_type](
            Int(p) * rows_x_heads * fa4_config.ov_depth
        )
        var partition = SplitKPartition[DType.float32](
            lse_partial.unsafe_ptr().as_unsafe_any_origin(),
            p,
            p,
        )
        # Attention: per-partition partials -> workspace (O store redirected to
        # `o_partial` via `o_workspace_ptr`; fused LSE via the `SplitKPartition`
        # pointer = `lse_partial`).
        with_fa4_config[fa4_config](
            StaticInt[1](),
            partition,
            NonNullPointer[output_type](o_partial),
        )
        # Combine: reduce the partials into `output`. Bind to
        # `partition.num_partitions()` (not `max_num_partitions()`) so an
        # over-launching caller would need no combine-side change.
        fa4_splitk_combine[output_type, fa4_config.ov_depth](
            ctx,
            output.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
            o_partial.unsafe_ptr().as_unsafe_any_origin(),
            lse_partial.unsafe_ptr().as_unsafe_any_origin(),
            partition.num_partitions(),
            UInt32(num_rows_q),
            UInt32(fa4_config.num_q_heads),
        )
        _ = o_partial^
        _ = lse_partial^

    # --- num_q dispatch ---
    # 1Q is only legal for qk_depth in [64, 256]; the comptime gate prevents
    # constructing fa4_config_1q (and its supported() assert) on shapes 1Q
    # can't run. Outside the gate we unconditionally use 2Q. This dispatch is
    # always single-CTA -- `fa4_config_2q` above takes `FA4Config`'s
    # `pair_cta=False` default -- so pair-CTA does not constrain the choice.
    comptime can_use_1q: Bool = config.depth >= 64 and config.depth <= 256
    comptime if can_use_1q:
        # Static per-C split-K: the num_q==1 split-K kernel is compiled ONCE per
        # candidate partition count `P` (== cluster size) in
        # `CLUSTER_SPLITK_CANDIDATES` (defined below, shared by all three
        # split-K routes -- see `_cluster_splitk_candidates`); the dispatch
        # picks which `P` to LAUNCH from occupancy + KV length.
        # `num_partitions` is an `OptionallyStaticInt` carrying a static `P` (or
        # a static `1` for the 2Q config and its in-kernel 1Q switch), so the
        # cluster size / grid / combine are all comptime constants -- there is no
        # runtime cluster dimension.
        comptime fa4_config_1q = fa4_config_2q.with_num_q(1)

        # The GPC fragmentation model (`clusters_per_wave`) covers B200 (148 SMs)
        # and B300 (160 SMs); its LUT folds at comptime keyed on
        # `ctx.default_device_info.sm_count` (a comptime value). The helper's own
        # `else` branch is the single source of truth for an unsupported chip, so
        # no standalone assert is needed here.

        # 1Q-vs-2Q gate (unchanged): pick 1Q when (a) max_prompt_len fits one 1Q
        # tile (2Q's BM=256 would waste >= 50% of Q rows) or (b) the unclamped 2Q
        # grid only fills <= half the SMs, so halving BM doubles the grid without
        # oversubscribing.
        var max_prompt_len_u32: UInt32 = max_prompt_len_arg.as_uint32()
        var max_num_prompt_tiles_2q: UInt32 = ceildiv(
            max_prompt_len_u32, UInt32(fa4_config_2q.PairBM_eff())
        )
        comptime num_heads_sched_2q: UInt32 = UInt32(
            fa4_config_2q.num_kv_heads if fa4_config_2q.fuse_gqa else fa4_config_2q.num_q_heads
        )
        var raw_grid_2q: UInt32 = (
            max_num_prompt_tiles_2q * num_heads_sched_2q * batch_size
        )
        # SM count is the comptime compile-target value (same source the
        # `clusters_per_wave` wave-fit uses below via `default_device_info`),
        # not a runtime `get_attribute` query -- keeps the 1Q/2Q grid gate and
        # the split-K wave-fit reasoning off one consistent occupancy model.
        comptime sm_count: UInt32 = UInt32(ctx.default_device_info.sm_count)
        comptime grid_threshold: UInt32 = sm_count // 2
        # `BM_eff`/`PairBM_eff` are P-independent (split-K forces cta_group==1),
        # so the 1Q geometry uses `fa4_config_1q` directly.
        comptime bm_eff_1q = UInt32(fa4_config_1q.BM_eff())
        # Shared cluster/DSMEM split-K candidate set (M4): only the
        # partition counts that fill every SM with zero GPC fragmentation
        # (see `_cluster_splitk_candidates`). Computed once and shared by the
        # 1Q, WS-G, and WS-E crossovers below -- with this restriction, the
        # per-route "predict what cluster would pick" scans and their
        # `clusters_per_wave`/`fits_wave` wave-fit gates are no longer
        # needed: a bucketed `P` either IS one of these (use cluster/DSMEM,
        # cheaper) or it ISN'T (use workspace, which has no such
        # restriction).
        comptime CLUSTER_SPLITK_CANDIDATES = _cluster_splitk_candidates[
            ctx.default_device_info.sm_count
        ]()
        # Config-force override (bench / test only; default 0 = auto, so
        # production folds to the shape-driven selection below, byte-identical).
        # Config selection is otherwise purely shape-driven with no way to run WS
        # vs the path it replaces at a *fixed* shape, which is exactly what the WS
        # benchmark (Phase C3) and the reroute-pinned tests (C5) need. Read inside
        # this generic `def` so it elaborates with the consuming test/bench's
        # `-D FA4_FORCE_CONFIG=...` copts (same mechanism as softmax_warp's
        # `FA4_1Q_SPLITK_WRITER`).  1 = force WS Layout-G (`BM=32`, bypass the
        # prompt-len/grid test, still comptime-gated on
        # `supported()`+`ws_mask_ok`); 2 = force the baseline 1Q/split-K/2Q
        # carve (skip the WS route entirely); 3 = force WS Layout-E (`BM=64`,
        # single-CTA unless `-D FA4_WS_SPLITK_FORCE=P` pins the cross-CTA split-K).
        # Under 0 = auto, Layout-E is now auto-selected (Phase 4) between the
        # Layout-G (BM=32) and 1Q rungs: for a prompt that single-tiles BM=64 but
        # not BM=32 (`32 < group*max_prompt_len <= 64`), the ladder is
        # BM=32 -> BM=64 -> 1Q(BM=128) -> 2Q(BM=256), picking the smallest tile
        # that single-tiles the prompt (split-K then fills the SMs).
        comptime FA4_FORCE_CONFIG = get_defined_int["FA4_FORCE_CONFIG", 0]()
        # Warp-specialized BM=32 packed-TMEM datapath for very short prompts:
        # the BM=32 tile holds `group` q-heads x BM_eff seq positions under
        # fuse_gqa (BM_eff = 32 // group; 8 at group=4), and the 8-way intra-CTA
        # split-K
        # (2 softmax WGs x 4 packed-TMEM quarters) extracts the parallelism from
        # the KV reduction. Scoped to `depth in {64, 128}` ONLY: the WS
        # per-warp band-packing combine helpers impose `ov_depth % depth_tile
        # == 0` (softmax_warp.mojo:532) where `depth_tile = 256 // m_pack`. For
        # Layout-G (m_pack=4) `depth_tile = 64`, so `depth % 64 == 0` -- only
        # 64/128 in [64,128]. depths 72/80/96 are NOT WS-supported (they fail
        # this comptime assert); they route to the 1Q carve below, whose combine
        # (`fa4_splitk_combine`) handles them via its scalar lane-strided
        # fallback. Both depths satisfy the shared sub-tile-ring invariant
        # `256 // m_pack == BK0 == 64` (depth 128 -> num_qk_stages=2, depth 64 ->
        # num_qk_stages=1; BK0 = padded_qk_depth // num_qk_stages = 64 either way,
        # P@V folds to num_d_tiles=1 at depth 64). Sinks ARE supported: the
        # intra-CTA fold is confined to WG0 quarter 0 (softmax_warp.mojo
        # `fold_sink and warp_idx == 0`), so the 8-way split adds the sink mass
        # exactly once -- the same partition-0 discipline as split-K.
        # `supported()` then prunes rope / KV-scale / non-uniform-sub-tile
        # shapes so dispatch degrades to the 1Q / split-K path below. Routed
        # BEFORE the `<= bm_eff_1q` split-K carve so WS-eligible shapes (see the
        # single-tile enablement) take WS first; everything else falls through
        # to the 1Q carve.
        comptime if config.depth == 128 or config.depth == 64:
            # Layout-E (`with_bm(64)`) force-launch intercept: a DEDICATED force
            # value (3) so Layout-G's existing FA4_FORCE_CONFIG={0,1,2} behavior
            # stays byte-identical. `-D FA4_WS_SPLITK_FORCE=P` composes here to
            # pin the Layout-E cross-CTA cluster split-K to P (Phase 3
            # correctness-gate); the AUTO candidate scan / prompt-len heuristic
            # is Phase 4. Falls through to the untouched Layout-G / auto-select
            # logic below when unset or when Layout-E's `supported()` / mask gate
            # reject the shape.
            comptime if FA4_FORCE_CONFIG == 3:
                comptime fa4_config_ws_e = fa4_config_2q.with_bm(64)
                comptime ws_e_mask_ok = MaskType.nonfull_sets[
                    fa4_config_ws_e.PairBM_eff(), fa4_config_ws_e.BN
                ]()[0] != TileMaskStatus.UNKNOWN_MASK
                comptime if fa4_config_ws_e.supported() and ws_e_mask_ok:
                    # Pin P via the shared `_ws_splitk_force_pin` (validated
                    # even counts {2,4,6,8,10,16}); 0/1/unsupported =>
                    # single-CTA. The split-K combine is m_pack-generic (bpp =
                    # ceil(m_pack/P)): for P > m_pack (== 2) only the first
                    # m_pack partitions own a depth band, while all P partition
                    # the KV and are reduced into the owning bands. Locally
                    # named (`_ws_e_fsk`) to avoid shadowing the Layout-G
                    # `FA4_WS_SPLITK_FORCE` binding read further below.
                    comptime _ws_e_fsk = get_defined_int[
                        "FA4_WS_SPLITK_FORCE", 0
                    ]()
                    comptime ws_e_P_force = _ws_splitk_force_pin(_ws_e_fsk)
                    comptime if ws_e_P_force >= 2:
                        comptime fa4_config_ws_e_splitk = (
                            fa4_config_ws_e.with_splitk(ws_e_P_force)
                        )
                        comptime assert (
                            fa4_config_ws_e_splitk.supported()
                        ), fa4_config_ws_e_splitk.description()
                        with_fa4_config[fa4_config_ws_e_splitk](
                            StaticInt[ws_e_P_force](),
                            NoPartition[DType.float32](),
                        )
                    else:
                        with_fa4_config[fa4_config_ws_e](
                            StaticInt[1](), NoPartition[DType.float32]()
                        )
                    return
            comptime fa4_config_ws = fa4_config_2q.with_bm(32)
            # Cross-CTA cluster split-K over the WS BM=32 config: each partition
            # count P groups P single-CTA WS kernels in a launch cluster that
            # partition the KV sequence and DSMEM-combine (a THIRD level, on top
            # of the 8-way intra-CTA split). Following #92167, each P compiles its
            # OWN static kernel (`StaticInt[P]`, `cluster_size() == P`) -- there is
            # no dynamic-cluster entry. Production auto-sizes P from the single-tile
            # scan below; `-D FA4_WS_SPLITK_FORCE=P` pins P for the split-K
            # correctness / bench targets. The WS route reuses the shared
            # `CLUSTER_SPLITK_CANDIDATES` set (see `_cluster_splitk_candidates`):
            # for P > m_pack (== 4) only `m_pack` partitions own a
            # depth band to write, but all P still partition the KV and are reduced
            # into the owning bands, so wider P raises SM utilization the same way
            # it does on the 1Q path (`fa4_ws_splitk_reduce_scatter_write`).
            comptime FA4_WS_SPLITK_FORCE = get_defined_int[
                "FA4_WS_SPLITK_FORCE", 0
            ]()
            # Pin the force knob to a validated WS split-K P via the shared
            # `_ws_splitk_force_pin`; 0/1 and any unsupported value =>
            # single-CTA WS.
            comptime ws_P_force = _ws_splitk_force_pin(FA4_WS_SPLITK_FORCE)
            # WS is validated only for masks whose visible range is statically
            # known and contiguous (`nonfull_sets[0] != UNKNOWN_MASK`: Null,
            # Causal, Chunked, SlidingWindow). Materialized/And/Or masks report
            # `{UNKNOWN_MASK}` and would take the WS softmax runtime-status path,
            # whose per-quarter `mask.status(...)` is issued over the 256-wide
            # tile window rather than the 64-wide packed-TMEM quarter -- correct
            # by superset but unverified on WS. Route them to the proven non-WS
            # 1Q / split-K / 2Q path instead. (Belt-and-suspenders today: these
            # masks are also blocked from this whole dispatch by the 1Q split-K
            # `UNKNOWN_MASK` comptime assert in `fa4_softmax`; this predicate
            # keeps them off the WS route if that block is ever lifted.)
            comptime ws_mask_ok = MaskType.nonfull_sets[
                fa4_config_ws.PairBM_eff(), fa4_config_ws.BN
            ]()[0] != TileMaskStatus.UNKNOWN_MASK
            # No cache-length gate. The WS 1Q shared-ring main loop
            # (`main_iters >= 1`, first exercised at T >= 4) is correct as of the
            # correction-SMEM sizing fix (2026-07-15: the region is sized by
            # softmax-thread count `2*WARPGROUP_SIZE`, not `BM`; see smem.mojo);
            # B200 matrix green for T in {1,2,3,5} across {Null,Causal,Chunked,
            # SlidingWindow} x depth{64,128}. Short prompt + long cache therefore
            # routes to single-CTA WS here. The perf guard that keeps a
            # huge-cache/short-prompt shape off single-CTA WS -- route WS only
            # while its finer BM=32 grid stays under full-SM occupancy -- is the
            # Phase C `ws_grid < sm_count` rule, landed with the WS benchmark.
            comptime if (
                fa4_config_ws.supported()
                and ws_mask_ok
                and FA4_FORCE_CONFIG != 2
            ):
                # Each WS split-K partition count P compiles its OWN static
                # single-CTA WS kernel (`StaticInt[P]`, `cluster_size() == P`),
                # mirroring the 1Q carve -- #92167 removed the dynamic-cluster
                # entry, so P is a comptime constant baked into `nvvm.cluster_dim`.
                comptime if FA4_FORCE_CONFIG == 1 or ws_P_force >= 2:
                    # Force override (bench C3 / pinned split-K tests): run WS
                    # regardless of prompt length / grid. P is the explicit pin
                    # `ws_P_force` (1 => single-CTA WS); each P is its own static
                    # kernel so forced runs stay deterministic. FA4_FORCE_CONFIG==1
                    # with no split-K pin runs single-CTA WS; a `ws_P_force >= 2`
                    # pin (with FORCE in {0, 1}) runs the static-P WS split-K.
                    comptime if ws_P_force >= 2:
                        comptime fa4_config_ws_splitk = fa4_config_ws.with_splitk(
                            ws_P_force
                        )
                        comptime assert (
                            fa4_config_ws_splitk.supported()
                        ), fa4_config_ws_splitk.description()
                        with_fa4_config[fa4_config_ws_splitk](
                            StaticInt[ws_P_force](),
                            NoPartition[DType.float32](),
                        )
                    else:
                        with_fa4_config[fa4_config_ws](
                            StaticInt[1](), NoPartition[DType.float32]()
                        )
                    return
                else:
                    # FORCE=0 production auto: route WS iff the whole prompt fits
                    # ONE WS tile -- `BM_eff >= max_prompt_len`, where fuse_gqa
                    # packs `group` q-heads into the BM=32 tile so one tile spans
                    # BM_eff = BM // group SEQ positions (8 for group=4), NOT 32.
                    # Beyond one tile the prompt shatters into ceildiv(seq, BM_eff)
                    # WS tiles (seq=32 => 4, seq=48 => 6) vs baseline's 1-2 BM=32
                    # tiles; at long cache those extra KV passes lose to baseline
                    # even when the grid is small, so an occupancy-only check
                    # over-routed the mid-seq/small-batch corner (measured
                    # seq=48/batch=1: 0.70-0.88x at cache>=8192). Single-tile WS
                    # never pays that shatter tax and still fills the SMs via the
                    # cross-CTA split-K candidate scan below.
                    var max_num_prompt_tiles_ws: UInt32 = ceildiv(
                        max_prompt_len_u32,
                        UInt32(fa4_config_ws.PairBM_eff()),
                    )
                    var raw_grid_ws: UInt32 = (
                        max_num_prompt_tiles_ws
                        * num_heads_sched_2q
                        * batch_size
                    )
                    if UInt32(fa4_config_ws.BM_eff()) >= max_prompt_len_u32:
                        # ---- M4: unified split-K partition-count + mechanism
                        # choice (mirrors the 1Q carve's `ws_p_ceiling` /
                        # `_bucket_ws` / `CLUSTER_SPLITK_CANDIDATES` design --
                        # see the comment there for the rationale). `sink` is
                        # NOT a gate on either mechanism here: cluster/DSMEM
                        # always supported it, and the workspace fallback below
                        # gained it via the `fold_sink` partition-0 gate (see
                        # the Layout-E note at the matching site). So an
                        # explicit `num_partitions` may force workspace on a
                        # sink shape.
                        # supported()/ws_mask_ok/FORCE_CONFIG are inherited from
                        # the enclosing gate.
                        var p_bucket_ws: UInt32
                        if num_partitions_override > 0:
                            p_bucket_ws = UInt32(num_partitions_override)
                        else:
                            # Size the ladder against the keys the mask leaves
                            # VISIBLE, not the raw cache length: a windowed mask
                            # over a long cache would otherwise be split into
                            # partitions that see nothing (correctness-safe --
                            # they egress `-inf` and the combine folds them --
                            # but pure waste). Inert for non-windowing masks;
                            # see `_visible_keys`.
                            var by_cache_ws: UInt32 = _visible_keys[
                                fa4_config_ws.PairBM_eff(),
                                fa4_config_ws.BN,
                                KVType.page_size,
                            ](mask, max_cache_valid_length) // UInt32(
                                512 * config.depth // 128
                            )
                            var p_ceiling_ws: UInt32 = ws_p_ceiling[
                                ctx.default_device_info.sm_count
                            ](raw_grid_ws)
                            # `_bucket_ws` buckets UP then caps at the ceiling,
                            # so handing it `by_cache` directly is identical to
                            # pre-clamping against the ceiling first.
                            p_bucket_ws = UInt32(
                                _bucket_ws[ctx.default_device_info.sm_count](
                                    Int(by_cache_ws), Int(p_ceiling_ws)
                                )
                            )
                        if p_bucket_ws > UInt32(1):
                            comptime for C in CLUSTER_SPLITK_CANDIDATES:
                                comptime ws_splitk_cfg = fa4_config_ws.with_splitk(
                                    C
                                )
                                comptime assert (
                                    ws_splitk_cfg.supported()
                                ), ws_splitk_cfg.description()
                                if p_bucket_ws == UInt32(C):
                                    with_fa4_config[ws_splitk_cfg](
                                        StaticInt[C](),
                                        NoPartition[DType.float32](),
                                    )
                                    return
                            launch_workspace[fa4_config_ws](p_bucket_ws)
                            return
                        with_fa4_config[fa4_config_ws](
                            StaticInt[1](), NoPartition[DType.float32]()
                        )
                        return
            # ---- Layout-E (BM=64, m_pack=2) production auto route (Phase 4) ----
            # One rung DOWN the tile ladder from Layout-G: reached only when the
            # prompt did NOT single-tile into BM=32 (the Layout-G auto route above
            # returns whenever `BM_eff_g >= max_prompt_len`) and FA4_FORCE_CONFIG
            # is 0 (auto). Routes BM=64 iff the prompt single-tiles into it
            # (`BM_eff_e >= max_prompt_len`, i.e. 32 < group*seq <= 64), then fills
            # the SMs with the shared `CLUSTER_SPLITK_CANDIDATES` / workspace
            # crossover. Falls through (no return) to the 1Q/2Q carve below when a bigger tile
            # is needed. FORCE in {1,2,3} skip this via the `== 0` gate, so every
            # forced bench/test cell and the Layout-G route stay byte-identical;
            # FORCE==3 (force Layout-E) is already handled by the intercept above.
            comptime fa4_config_ws_e = fa4_config_2q.with_bm(64)
            comptime ws_e_mask_ok = MaskType.nonfull_sets[
                fa4_config_ws_e.PairBM_eff(), fa4_config_ws_e.BN
            ]()[0] != TileMaskStatus.UNKNOWN_MASK
            comptime if (
                fa4_config_ws_e.supported()
                and ws_e_mask_ok
                and FA4_FORCE_CONFIG == 0
            ):
                var max_num_prompt_tiles_ws_e: UInt32 = ceildiv(
                    max_prompt_len_u32,
                    UInt32(fa4_config_ws_e.PairBM_eff()),
                )
                var raw_grid_ws_e: UInt32 = (
                    max_num_prompt_tiles_ws_e * num_heads_sched_2q * batch_size
                )
                if UInt32(fa4_config_ws_e.BM_eff()) >= max_prompt_len_u32:
                    # ---- M4: unified split-K partition-count + mechanism
                    # choice (mirrors the 1Q/WS-G design -- see the 1Q
                    # carve's comment for the general rationale). Three
                    # WS-E-specific, measured guards ride on top of the
                    # shared formula (Phase-4 sweep, 2026-07-24/25):
                    #  (1) `ws_e_min_keys`: workspace's fixed combine +
                    #      global round-trip cost isn't amortized below a
                    #      depth-keyed cache floor (8192 @ d128, 65536 @
                    #      d64 -- d64's half-FLOP/key leaves that fixed cost
                    #      a bigger fraction). Below it, skip straight to
                    #      cluster/plain.
                    #  (2) d64's workspace path is UNPROVEN beyond
                    #      raw_grid<=2: the fine-batch sweep found it goes
                    #      non-monotonic and regresses at raw_grid 4-9
                    #      (ws/cl ratio 0.91 there) using this SAME
                    #      plain-one-wave computation (no target-boost
                    #      existed yet) -- i.e. the regression is inherent to
                    #      letting d64 workspace engage there at all, not an
                    #      artifact the boost introduces. So d64 stays
                    #      restricted to raw_grid<=2 regardless of the
                    #      bucketed `P`; d128 has no such evidence and is not
                    #      restricted.
                    #  (3) `ws_e_p_cap`: past ~64 partitions the
                    #      per-partition combine overhead dominates (P148
                    #      measured strictly worse than P64 at every
                    #      long-cache point); cap the ceiling there
                    #      regardless of depth.
                    # `not sink` is no longer a gate here -- workspace now
                    # supports sink via the `fold_sink` partition-0 gate.
                    comptime ws_e_min_keys = UInt32(
                        8192 if config.depth >= 128 else 65536
                    )
                    comptime ws_e_p_cap = UInt32(64)
                    comptime ws_e_max_raw_grid_d64 = UInt32(2)
                    var p_bucket_ws_e: UInt32
                    if num_partitions_override > 0:
                        p_bucket_ws_e = UInt32(num_partitions_override)
                    else:
                        # Visible-key sizing, as in the Layout-G route above.
                        var by_cache_ws_e: UInt32 = _visible_keys[
                            fa4_config_ws_e.PairBM_eff(),
                            fa4_config_ws_e.BN,
                            KVType.page_size,
                        ](mask, max_cache_valid_length) // UInt32(
                            512 * config.depth // 128
                        )
                        var p_ceiling_ws_e: UInt32 = min(
                            ws_p_ceiling[ctx.default_device_info.sm_count](
                                raw_grid_ws_e
                            ),
                            ws_e_p_cap,
                        )
                        p_bucket_ws_e = UInt32(
                            _bucket_ws[ctx.default_device_info.sm_count](
                                Int(by_cache_ws_e), Int(p_ceiling_ws_e)
                            )
                        )
                    if p_bucket_ws_e > UInt32(1):
                        comptime for C in CLUSTER_SPLITK_CANDIDATES:
                            comptime ws_e_splitk_cfg = fa4_config_ws_e.with_splitk(
                                C
                            )
                            comptime assert (
                                ws_e_splitk_cfg.supported()
                            ), ws_e_splitk_cfg.description()
                            if p_bucket_ws_e == UInt32(C):
                                with_fa4_config[ws_e_splitk_cfg](
                                    StaticInt[C](),
                                    NoPartition[DType.float32](),
                                )
                                return
                        # An explicit override bypasses both measured perf
                        # guards: they are heuristics about when workspace
                        # split-K PAYS, and a caller naming `P` has already
                        # decided. `ws_e_min_keys` deliberately keeps reading
                        # the RAW cache length -- it is a fixed-cost
                        # amortization threshold, not a P-sizing input.
                        if num_partitions_override > 0 or (
                            max_cache_valid_length >= ws_e_min_keys
                            and (
                                config.depth >= 128
                                or raw_grid_ws_e <= ws_e_max_raw_grid_d64
                            )
                        ):
                            launch_workspace[fa4_config_ws_e](p_bucket_ws_e)
                            return
                    # Unreachable with an override >= 2: workspace split-K is
                    # unconditionally available on this route, so such an
                    # override always returns above (cluster when `P` is a
                    # candidate, workspace otherwise). Kept as a forward guard --
                    # any future comptime gate on the workspace fallback would
                    # silently serve `P == 1` and let a caller believe it pinned
                    # a partition count it never got.
                    debug_assert[assert_mode="safe"](
                        num_partitions_override <= 1,
                        (
                            "num_partitions override could not be honored on"
                            " the Layout-E route: workspace split-K is"
                            " unavailable"
                        ),
                    )
                    with_fa4_config[fa4_config_ws_e](
                        StaticInt[1](), NoPartition[DType.float32]()
                    )
                    return
        # An explicit `num_partitions` forces BM < 256. The 2Q arms below launch
        # `StaticInt[1]()` / `NoPartition` unconditionally -- 2Q has NO split-K
        # mechanism at all -- so letting an override fall through there would
        # silently pin `P == 1`. Taking the 1Q carve instead is correctness-safe
        # at any prompt length (`max_num_prompt_tiles_1q` below shatters a long
        # prompt across BM=128 tiles); the 2Q preference is purely a
        # large-tile perf heuristic. Both configs are already compiled, so this
        # adds no instantiation.
        if (
            num_partitions_override > 0
            or max_prompt_len_u32 <= bm_eff_1q
            or raw_grid_2q <= grid_threshold
        ):
            var max_num_prompt_tiles_1q: UInt32 = ceildiv(
                max_prompt_len_u32, UInt32(fa4_config_1q.PairBM_eff())
            )
            # Number of split-K CLUSTERS the launch needs (one per work item);
            # each is a size-`P` cluster occupying `P` SMs within a single GPC.
            var raw_grid_1q: UInt32 = (
                max_num_prompt_tiles_1q * num_heads_sched_2q * batch_size
            )
            # Don't split a short KV into more partitions than there is K/V to
            # go around (min keys / partition); avoids mostly-empty clusters. A
            # d64 partition does half the per-partition depth work of a d128 one,
            # so it amortizes the cross-partition combine at half the keys (256
            # vs 512) and can split 2x further -- the flat-512 floor
            # under-parallelized d64 prefill. Scoped to d64
            # (`256 if depth==64 else 512`, NOT a general `512*depth//128`) so
            # only the measured depth changes; the other 1Q-legal depths in
            # [64, 256] keep 512 pending their own sweep.
            comptime one_q_min_kpp = 256 if config.depth == 64 else 512

            # ---- M4: unified split-K partition-count + mechanism choice ----
            # Pick ONE desired partition count from the shared
            # `ws_p_ceiling` / `_bucket_ws` formula, then
            # check whether it happens to be a cluster/DSMEM-compatible size
            # (`CLUSTER_SPLITK_CANDIDATES`, the fill-every-SM values) --
            # cheaper, no combine kernel / workspace round-trip -- else fall
            # back to workspace, which admits ANY `P`. This replaces the
            # former two-scan design (an independent "would cluster reach the
            # target" prediction feeding a `p_desired > SPLITK_MAX_C` /
            # device-fill crossover, plus a separate `clusters_per_wave`
            # wave-fit cluster scan): once cluster/DSMEM is restricted to
            # zero-GPC-fragmentation sizes, oversubscribing it past one wave
            # is exactly as cheap as workspace, so there is nothing left for
            # two scans to arbitrate.
            #
            # Gated on `ws1q_mask_ok` (materialized/And/Or masks report
            # UNKNOWN_MASK and would trip the 1Q split-K `fa4_softmax`
            # comptime assert for EITHER mechanism). Sink IS supported on the
            # workspace route: the `fold_sink` partition-0 gate
            # (softmax_warp.mojo) folds the sink
            # mass into partition 0's `lse_partial` exactly once, and
            # `fa4_splitk_combine` rescales it; a sink shape whose bucketed `P`
            # is not cluster-compatible falls back to `P==1` rather than the
            # wider ladder sink had access to before this change (accepted --
            # the cluster path remains the preferred sink vehicle).
            #
            # Workspace is restricted to `depth in [64, 128]` (below, at the
            # `launch_workspace` call). The 1Q workspace split-K correctness
            # matrix historically only covered `{64, 128}`: before the M4
            # unification, the old
            # one-wave-capped crossover rarely (if ever) reached workspace for
            # the OTHER 1Q-legal depths (72/80/96/256/512: `can_use_1q` only
            # requires depth in [64,256]), so a latent gap there stayed
            # dormant; M4 broadened workspace eligibility to any shape whose
            # bucketed `P` misses the cluster candidate set, which newly
            # reaches it. It surfaced as `CUDA_ERROR_MISALIGNED_ADDRESS` on a
            # depth=96 + long-cache correctness test
            # (`test_mha_causal_mask_depth_96`) -- root-caused in
            # `fa4_splitk_combine.mojo`: depth=96 yields a non-power-of-2
            # `vec_size=3`, and the vectorized combine path's `width=3` bf16
            # loads/stores lower to a 4-byte sub-op at a 2-byte-aligned address
            # for odd lanes. Fixed by gating the vectorized path on a
            # power-of-2 `vec_size` (depths 96/160/192/224 now route to the
            # scalar lane-strided fallback). Cluster split-K (proven at these
            # other depths already, e.g. the old `SPLITK_CANDIDATES` ladder)
            # and the `P==1` fallback are unaffected and stay available.
            comptime ws1q_mask_ok = (
                MaskType.nonfull_sets[
                    fa4_config_1q.PairBM_eff(), fa4_config_1q.BN
                ]()[0]
                != TileMaskStatus.UNKNOWN_MASK
            )
            comptime if ws1q_mask_ok:
                var p_bucket: UInt32
                if num_partitions_override > 0:
                    p_bucket = UInt32(num_partitions_override)
                else:
                    # Visible-key sizing (see `_visible_keys`). Computed INSIDE
                    # the `ws1q_mask_ok` gate on purpose: for an `UNKNOWN_MASK`
                    # mask `start_column` falls back to
                    # `naively_get_first_nonempty_mask_col`, which steps
                    # `status()` -- and `status()` is exactly the method that
                    # reads DEVICE memory for a padding mask, so calling it from
                    # this host-side dispatch would segfault. Split-K is
                    # comptime-pruned for those masks right here, so the value
                    # would be unused anyway; not computing it is what keeps
                    # this host-safe. `MaterializedMask` and `AndMask` are the
                    # only two masks that still scan, and both are exactly the
                    # `UNKNOWN_MASK` set this gate excludes, so every mask that
                    # does reach `_visible_keys` resolves in O(1).
                    var by_cache: UInt32 = _visible_keys[
                        fa4_config_1q.PairBM_eff(),
                        fa4_config_1q.BN,
                        KVType.page_size,
                    ](mask, max_cache_valid_length) // UInt32(one_q_min_kpp)
                    var p_ceiling: UInt32 = ws_p_ceiling[
                        ctx.default_device_info.sm_count
                    ](raw_grid_1q)
                    p_bucket = UInt32(
                        _bucket_ws[ctx.default_device_info.sm_count](
                            Int(by_cache), Int(p_ceiling)
                        )
                    )
                if p_bucket > UInt32(1):
                    comptime for C in CLUSTER_SPLITK_CANDIDATES:
                        comptime splitk_cfg = fa4_config_1q.with_splitk(C)
                        comptime assert (
                            splitk_cfg.supported()
                        ), splitk_cfg.description()
                        if p_bucket == UInt32(C):
                            with_fa4_config[splitk_cfg](
                                StaticInt[C](), NoPartition[DType.float32]()
                            )
                            return
                    comptime if config.depth >= 64 and config.depth <= 128:
                        launch_workspace[fa4_config_1q](p_bucket)
                        return
            # P==1 (nothing to combine, or mask/depth excluded split-K):
            # launch the 1Q single-partition config -- no cluster, no DSMEM,
            # no combine.
            #
            # An override lands here when split-K is comptime-unavailable: the
            # mask reports `UNKNOWN_MASK` (`ws1q_mask_ok`), or `P` missed every
            # cluster candidate at a depth outside the workspace range
            # [64, 128]. Silently serving `P == 1` would let a test believe it
            # pinned a partition count it never got.
            debug_assert[assert_mode="safe"](
                num_partitions_override <= 1,
                (
                    "num_partitions override could not be honored on the 1Q"
                    " route: split-K unavailable (UNKNOWN_MASK, or depth"
                    " outside [64, 128])"
                ),
            )
            with_fa4_config[fa4_config_1q](
                StaticInt[1](), NoPartition[DType.float32]()
            )
        else:
            # Not reachable with an override: the gate above forces the 1Q carve
            # whenever `num_partitions_override > 0`.
            with_fa4_config[fa4_config_2q](
                StaticInt[1](), NoPartition[DType.float32]()
            )
    else:
        # `not can_use_1q` (pair-CTA, or depth outside [64, 256]): 2Q is the only
        # config for this shape and it cannot split, so an override is
        # unsatisfiable here -- there is no smaller-BM route to fall back to.
        debug_assert[assert_mode="safe"](
            num_partitions_override <= 1,
            (
                "num_partitions override could not be honored: this shape only"
                " has the 2Q (BM=256) config, which has no split-K"
            ),
        )
        with_fa4_config[fa4_config_2q](
            StaticInt[1](), NoPartition[DType.float32]()
        )
