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

from std.bit import next_power_of_two
from std.gpu.host import DeviceAttribute, DeviceContext
from std.math import ceildiv, clamp


@always_inline
def _bucket_partitions(n: Int) -> Int:
    """Bucket a partition count up to a fixed grid-shape ladder.

    Power-of-two bucketing leaves a 2x gap between 64 and 128 — meaningful
    for h=64 / h=128 shapes where the target landing point is often in the
    65..127 range and 128 is wasteful. We insert mid-points {48, 96} to
    halve the worst-case over-partitioning above 32 while keeping the
    bucket count small enough for HIP graph capture.

    Ladder: 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256.

    The SM100 path has an analogous helper, `_bucket_num_partitions` in
    `nvidia/sm100/mla_decode_dispatch.mojo`, with a different ladder
    (top = `half_sms`, parameterized on GPU SM count) — driven by SM100's
    wave-fill target instead of an AMD-reducer hard cap. They aren't
    shared because the constraints differ; the pattern is the same.
    """
    if n <= 32:
        return next_power_of_two(n)
    if n <= 48:
        return 48
    if n <= 64:
        return 64
    if n <= 96:
        return 96
    if n <= 128:
        return 128
    if n <= 192:
        return 192
    return 256


def cuda_mha_decoding_max_num_partitions(
    batch_size: Int,
    heads_per_group: Int,
    sm_count: Int,
) -> Int:
    # num_keys-independent partition target: fill one partition per idle SM,
    # clamped to [1, 32]. The 32 ceiling is the MHA split-K reducer's
    # single-warp WARP_SIZE limit; the lower bound of 1 guards the case where
    # batch_size * heads_per_group > sm_count drives the SM term to 0
    # (large-batch decode), which would otherwise return 0 partitions and
    # divide by zero downstream in get_start_and_end_for_partitions. Not rounded
    # to a power of two: the reducer handles any count in [1, 32], so an exact
    # target avoids over-partitioning (e.g. 17 -> 32). The actual count
    # (cuda_mha_decoding_num_partitions) only further mins this with
    # num_keys // 512, so this is the upper bound for every num_keys -- used to
    # launch a num_keys-independent (graph-stable) decode grid whose extra
    # partitions early-return in the kernel.
    return clamp(sm_count // (batch_size * heads_per_group), 1, 32)


def cuda_mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    sm_count: Int,
) -> Int:
    # The num_keys-independent upper bound, further limited so each partition
    # spans at least 512 keys. Deriving from the max keeps the SM-fill target
    # and the [1, 32] clamp in one place, so max >= actual holds by
    # construction. The max(1, ...) floor preserves the >= 1 guard when
    # num_keys < 512 (a 0 here divides by zero downstream).
    return min(
        cuda_mha_decoding_max_num_partitions(
            batch_size, heads_per_group, sm_count
        ),
        max(1, num_keys // 512),
    )


def hip_mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    sm_count: Int,
    is_mla: Bool = False,
) -> Int:
    """Wave-aligned split-K target for MI355X MHA + MLA decode.

    Two regimes, distinguished by whether the kernel packs queries
    by BM (MLA) or spawns one CTA per kv-head (MHA):

    - **MHA-style** (`heads_per_group < BM`): the call comes from
      `get_mha_decoding_num_partitions` passing
      `heads_per_group = num_heads // group = kv_num_heads`, which is
      typically small (≤ 8). Each (kv_head, batch) is its own CTA in
      grid_y, so `actual_ctas_per_partition = heads_per_group ×
      batch_size`. When `work_items = heads_per_group × batch_size ≥
      sm_count`, one partition already fills the GPU — use few
      partitions, just enough to amortize key reads. This matches the
      original heuristic's HIGH_OCC branch (preserved verbatim to
      avoid regressing the MHA bench grid).

    - **MLA-style** (`heads_per_group ≥ BM`): the call comes from
      `mla.mojo` passing `heads_per_group = num_heads` (≥ BM=32 for
      h ∈ {32, 64, 128}). MLA packs BM queries into one CTA, so
      `actual_ctas_per_partition = ceildiv(num_heads, BM) ×
      batch_size`. Even when `work_items` looks large (e.g. bs=8
      h=64 → 512), actual CTAs are only 16 — needs many partitions.
      Apply the 2-wave wave-aligned formula:
          one_wave    = sm_count // ctas_per_partition
          two_wave    = 2 × sm_count // ctas_per_partition
          work_floor  = ceildiv(pages, MAX_PAGES_PER_SPLIT)
          np_target   = clamp(work_floor, one_wave, two_wave)

      EXCEPTION (num_heads <= 16, e.g. Kimi-K2.5 TP=4): the one_wave floor
      underfills. With num_blocks_y=1, ctas_per_partition = batch_size, so
      one wave (np = sm/bs) gives each CU exactly one CTA — no second block
      to overlap HBM-read latency. These shapes are latency-bound, so target
      two full waves instead:
          np_target   = min(two_wave, pages)
      Measured on MI355: two-wave np is 5-10% faster than one-wave across
      bs=4 (32K-128K) and bs=8/16 short context; past two waves regresses on
      reduce cost. bs=1 (two_wave=512 -> clamps to 256 = one wave) unchanged.

    Phase 0 sweep (PARTITIONING_PLAN.md) validated MLA-style at h=64:
        bs=1  ctx=131K → np=128 (capped, fills GPU at 1-wave + cap)
        bs=2  ctx=65K  → np=64  (one_wave=64 dominates)
        bs=2  ctx=80K  → np=64  (one_wave=64 dominates; work_floor 64 capped)
        bs=2  ctx=131K → np=128 (work_floor=103 → bucket to 128)
        bs=8  ctx=80K  → np=32  (work_floor 64 capped by two_wave=32)
        bs=8  ctx=131K → np=32  (work_floor 103 capped by two_wave=32)
        bs=16 ctx=131K → np=16  (work_floor 103 capped by two_wave=16)

    AMD reducer constraint: `mla_splitk_reduce` supports MAX_PARTITIONS
    up to `parts_per_lane × WARP_SIZE`; the 256-partition specialization
    (parts_per_lane=4) lifts the MLA-style cap to 256. Only nk >= 64K
    (pages >= 256) actually reaches np=256 — smaller nk is page-limited.

    Tunables (MLA-style):
        BM                   = 32   (MLA decode block-M on MI355)
        SPLIT_PAGE_SIZE      = 256  (min keys per partition)
        MAX_PAGES_PER_SPLIT  = 5    (= 1280 keys per partition cap)
        MAX_HIP_PARTITIONS   = 256  (reducer's MAX_PARTITIONS limit; the
                                     MHA-style branch above stays pinned ≤64)
    """
    comptime BM = 32
    comptime SPLIT_PAGE_SIZE = 256
    comptime MAX_PAGES_PER_SPLIT = 5
    comptime MAX_HIP_PARTITIONS = 256
    # Empirically-tuned divisor used in the MHA HIGH_OCC branch to scale
    # partitions inversely with work_items. NOT WARP_SIZE — it's a
    # workload-shaping constant inherited from the pre-Phase-1 heuristic
    # (happens to equal 64 on gfx950 by coincidence).
    comptime MHA_OCC_SCALE_DIVISOR = 64

    # No partitioning for very short caches — split-K overhead exceeds win.
    if num_keys <= SPLIT_PAGE_SIZE:
        return 1

    var work_items = heads_per_group * batch_size

    # MHA-style: kv_num_heads spawns CTAs directly in grid_y.
    if (not is_mla) and heads_per_group < BM:
        if work_items >= sm_count:
            # High occupancy: 1 partition already fills the GPU. Scale
            # partition size up as work_items grows so we don't
            # over-partition (more concurrent CTAs → fewer per-CTA pages
            # is fine). The divisor is empirical, not WARP_SIZE.
            var occupancy_scale = max(1, work_items // MHA_OCC_SCALE_DIVISOR)
            var np_mha = min(
                ceildiv(num_keys, SPLIT_PAGE_SIZE * occupancy_scale),
                # MHA-style cheap-reducer cap; pinned at 64 so the MLA-only
                # MAX_HIP_PARTITIONS bump (128->256) does not change MHA grids.
                64,
            )
            return min(_bucket_partitions(np_mha), MAX_HIP_PARTITIONS)
        # Low occupancy MHA: rare. Fall through to MLA-style formula
        # since it handles the wave-fill case correctly with
        # ctas_per_partition = work_items (BM packing is a no-op when
        # heads_per_group < BM).

    # MLA-style (or low-occupancy MHA).
    var ctas_per_partition = max(1, ceildiv(heads_per_group, BM) * batch_size)
    var pages = ceildiv(num_keys, SPLIT_PAGE_SIZE)
    var one_wave = max(1, sm_count // ctas_per_partition)
    var two_wave = max(1, (2 * sm_count) // ctas_per_partition)
    var work_floor = ceildiv(pages, MAX_PAGES_PER_SPLIT)

    var np_target: Int
    if is_mla and heads_per_group <= 16:
        # num_heads <= 16 (Kimi-K2.5 TP=4) packs all heads into one block
        # (num_blocks_y=1), so ctas_per_partition = batch_size — tiny. Decode
        # is latency-bound: each CTA stalls on HBM K-reads, so fill TWO waves
        # — a second CTA per CU hides the first's stalls. Bounded by available
        # pages (cannot split below one page) and the 256-partition cap. The
        # one_wave floor used below underfills here: measured on MI355, the
        # two-wave np is 5-10% faster than one-wave across bs=4 (32K-128K) and
        # bs=8/16 short context, while going *past* two waves regresses (split-K
        # reduce cost). bs=1 has two_wave=512 which clamps to 256 = one wave
        # (the most CTAs it can reach), so it is unchanged.
        np_target = min(two_wave, pages)
    else:
        # num_heads >= 32 (or low-occupancy MHA fallthrough): keep the tuned
        # wave-aligned target — clamp work_floor to [one_wave, two_wave]
        # (one_wave floor, two_wave cap; validated for num_heads=64/128 in the
        # Phase-0/1 sweeps).
        np_target = clamp(work_floor, one_wave, two_wave)

    # The MHA split-K reducer runs in a single warp and only handles up to
    # WARP_SIZE partitions, so cap MHA at 64. MLA uses a partition-aware
    # reducer and keeps the full 256.
    var partition_cap = MAX_HIP_PARTITIONS if is_mla else 64
    var num_partitions = min(np_target, pages, partition_cap)

    # Bucket to a fixed ladder (1, 2, ..., 64, 96, 128, 192, 256) so
    # HIP graph capture sees a small number of decode grid shapes.
    return min(_bucket_partitions(num_partitions), partition_cap)


def mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    ctx: DeviceContext,
    is_mla: Bool = False,
) raises -> Int:
    var api = ctx.api()
    if api == "hip" or api == "cuda":
        # MULTIPROCESSOR_COUNT is only meaningful for the split-K heuristics,
        # so query it lazily here rather than for every backend.
        var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        if api == "hip":
            return hip_mha_decoding_num_partitions(
                batch_size,
                num_keys,
                heads_per_group,
                sm_count,
                is_mla=is_mla,
            )
        return cuda_mha_decoding_num_partitions(
            batch_size,
            num_keys,
            heads_per_group,
            sm_count,
        )
    # CUDA and HIP have tuned split-K decode heuristics above. Every other
    # backend (Metal, plus accelerators with no split-K decode path) runs the
    # decode unsplit; a single partition is always valid — the decode kernel
    # reads this count only to bound its split-K loop.
    return 1


def mha_decoding_max_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    ctx: DeviceContext,
) raises -> Int:
    # num_keys-independent upper bound on mha_decoding_num_partitions, used to
    # launch a graph-stable decode grid. Only the CUDA decode path over-launches
    # and early-returns the extra partitions; every other backend keeps
    # max == actual so the (max >= actual) invariant holds and no over-launch
    # path is taken.
    if ctx.api() == "cuda":
        var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        return cuda_mha_decoding_max_num_partitions(
            batch_size, heads_per_group, sm_count
        )
    return mha_decoding_num_partitions(
        batch_size, num_keys, heads_per_group, ctx
    )
