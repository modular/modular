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
from std.math import ceildiv


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


def cuda_mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    sm_count: Int,
) -> Int:
    if num_keys > 512:
        return min(
            next_power_of_two(
                min(
                    sm_count // (batch_size * heads_per_group),
                    num_keys // 512,
                )
            ),
            32,
        )
    return 1


def hip_mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    sm_count: Int,
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
          np_target   = max(one_wave, min(work_floor, two_wave))

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
    if heads_per_group < BM:
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
    var np_target = max(one_wave, min(work_floor, two_wave))
    var num_partitions = min(np_target, pages, MAX_HIP_PARTITIONS)

    # Bucket to a fixed ladder (1, 2, ..., 64, 96, 128, 192, 256) so
    # HIP graph capture sees a small number of decode grid shapes.
    return min(_bucket_partitions(num_partitions), MAX_HIP_PARTITIONS)


def mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    ctx: DeviceContext,
) raises -> Int:
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    if ctx.api() == "hip":
        return hip_mha_decoding_num_partitions(
            batch_size,
            num_keys,
            heads_per_group,
            sm_count,
        )
    if ctx.api() == "cuda":
        return cuda_mha_decoding_num_partitions(
            batch_size,
            num_keys,
            heads_per_group,
            sm_count,
        )
    if ctx.api() == "metal":
        return 1
    raise Error("Expected a CUDA, HIP, or Metal device context.")
