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

"""Host-side pins for the AMD MLA decode split-K partition heuristic.

The split count is numerically inert: every bucket reduces to the same
answer, so choosing a worse one costs wall time and nothing else. No GPU
test can see that, which leaves the policy free to drift silently.

The S>1 fold's two-tile split doubles the CTAs each (batch, partition)
spans, and the measured optimum is one wave of that true CTA count. These
pins hold the rule at the measured shapes, the b3 bucket boundary between
the b2/b4 neighbors, and — just as load-bearing — the byte-identity of
every single-tile caller (S=1, the S<=5 single-tile fold, all MHA), whose
partition counts must not move when the fold rule does.
"""

from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_mla_decode_fold_num_m_tiles,
    hip_mha_decoding_num_partitions,
)

comptime SM_COUNT = 256


def _expect(what: String, got: Int, want: Int) raises:
    if got != want:
        raise Error(what, ": got ", got, ", want ", want)


def test_fold_tile_count_matches_the_launch_ladder() raises:
    """The tile-count helper is the launch ladder's split rule as a pure
    function, so the partition heuristic and the device-graph capture key
    count the tiles the ladder actually launches. The ladder splits only
    past the 64-row occupancy cliff and only on a head-aligned tile.
    """
    # (label, heads, last single-tile width): H=16 hits the cliff one
    # width earlier (M=80 at S=5); H=8 tops out at M=64 and never splits,
    # so the drafter control keeps the single-tile partition count.
    for label, heads, last_single_width in [
        ("K3", 12, 5),
        ("H=16", 16, 4),
        ("H=8", 8, 8),
    ]:
        for s in range(1, 9):
            _expect(
                label + " fold tiles at S=" + String(s),
                amd_mla_decode_fold_num_m_tiles(heads, s),
                1 if s <= last_single_width else 2,
            )
    # Head counts whose half-height tile is not head-aligned keep the
    # single-tile (spilling, but unchanged) launch.
    for h in [13, 14, 15]:
        _expect(
            "head-unaligned fold tiles at H=" + String(h),
            amd_mla_decode_fold_num_m_tiles(h, 6),
            1,
        )
    # Plain decode never folds.
    _expect("S=1 tiles", amd_mla_decode_fold_num_m_tiles(12, 1), 1)


def test_multi_tile_fold_targets_one_wave() raises:
    """The two-tile fold fills the device once: np = sm_count // (batch *
    num_m_tiles), page-capped, ladder-bucketed. Pinned at the measured
    MI355X optima (Kimi K3 verify, H=12, fp8 KV, decode+reduce): one wave
    won at every measured point, 7-32% against the two-wave and pre-fold
    counts, and np=256 at b4/32k regressed +36%.
    """
    comptime keys_32k = 32768
    comptime keys_8k = 8192
    comptime keys_120k = 122880
    comptime tiles = 2

    # The batch ladder at 32k, including the b3 boundary: 256/6 = 42.67
    # rounds up to the 64 rung (the 48 mid-bucket rags the per-partition key
    # stream against the 128-key tiles and measured +10.7% against the 128
    # rung; 64 is b3's measured optimum).
    for b in range(1, 9):
        var want = 128
        if b == 8:
            want = 16
        elif b >= 4:
            want = 32
        elif b >= 2:
            want = 64
        _expect(
            "fold np at batch=" + String(b) + " 32k",
            hip_mha_decoding_num_partitions(
                b, keys_32k, 12, SM_COUNT, is_mla=True, num_m_tiles=tiles
            ),
            want,
        )
    # Context ladder at b4: 8k is exactly 32 split pages, meeting the wave
    # target; 120k leaves the wave target binding.
    _expect(
        "fold np b4 8k",
        hip_mha_decoding_num_partitions(
            4, keys_8k, 12, SM_COUNT, is_mla=True, num_m_tiles=tiles
        ),
        32,
    )
    _expect(
        "fold np b4 120k",
        hip_mha_decoding_num_partitions(
            4, keys_120k, 12, SM_COUNT, is_mla=True, num_m_tiles=tiles
        ),
        32,
    )
    _expect(
        "fold np b1 120k",
        hip_mha_decoding_num_partitions(
            1, keys_120k, 12, SM_COUNT, is_mla=True, num_m_tiles=tiles
        ),
        128,
    )
    _expect(
        "fold np b1 32k",
        hip_mha_decoding_num_partitions(
            1, keys_32k, 12, SM_COUNT, is_mla=True, num_m_tiles=tiles
        ),
        128,
    )
    # S=6/7 fold the same two tiles as S=8, so they take the same count.
    for s in [6, 7]:
        _expect(
            "fold tiles at S=" + String(s),
            amd_mla_decode_fold_num_m_tiles(12, s),
            tiles,
        )


def test_single_tile_callers_keep_their_partition_counts() raises:
    """Every caller that does not span a two-tile fold must keep its exact
    partition count: the fold rule is gated on num_m_tiles > 1 and the
    default argument is 1, so existing call sites are unchanged. These are
    regression pins at the pre-fold values, not restatements of the
    formula.
    """
    # S=1 and the S<=5 single-tile fold (H=12): the two-wave target,
    # page-capped at 128 by the 32k context.
    for b in range(1, 5):
        _expect(
            "S=1 np at batch=" + String(b) + " 32k",
            hip_mha_decoding_num_partitions(
                b, 32768, 12, SM_COUNT, is_mla=True
            ),
            128,
        )
        _expect(
            "single-tile fold np at batch=" + String(b) + " 32k",
            hip_mha_decoding_num_partitions(
                b, 32768, 12, SM_COUNT, is_mla=True, num_m_tiles=1
            ),
            128,
        )
    _expect(
        "S=1 np b1 120k",
        hip_mha_decoding_num_partitions(1, 122880, 12, SM_COUNT, is_mla=True),
        256,
    )
    _expect(
        "S=1 np b4 8k",
        hip_mha_decoding_num_partitions(4, 8192, 12, SM_COUNT, is_mla=True),
        32,
    )
    # The H=8 drafter control: M=56 at S=7 stays single-tile.
    _expect(
        "H=8 S=7 np b4 32k",
        hip_mha_decoding_num_partitions(4, 32768, 8, SM_COUNT, is_mla=True),
        128,
    )
    # Large-head S=1 launches span multiple grid.y tiles that the
    # ceildiv(heads, block_m) packing already counts.
    _expect(
        "H=64 S=1 np b1 32k",
        hip_mha_decoding_num_partitions(1, 32768, 64, SM_COUNT, is_mla=True),
        128,
    )
    _expect(
        "H=128 S=1 np b1 32k",
        hip_mha_decoding_num_partitions(1, 32768, 128, SM_COUNT, is_mla=True),
        64,
    )
    # MHA callers: the 16-kv-head key-stream rule, the single-kv-head
    # finer-split exception, and the plain GQA fall-through.
    _expect(
        "MHA 16kv np b16 32k",
        hip_mha_decoding_num_partitions(16, 32768, 16, SM_COUNT),
        16,
    )
    _expect(
        "MHA 1kv np b4 4096",
        hip_mha_decoding_num_partitions(4, 4096, 1, SM_COUNT),
        32,
    )
    _expect(
        "MHA 8kv np b4 32k",
        hip_mha_decoding_num_partitions(4, 32768, 8, SM_COUNT),
        64,
    )


def main() raises:
    test_fold_tile_count_matches_the_launch_ladder()
    test_multi_tile_fold_targets_one_wave()
    test_single_tile_callers_keep_their_partition_counts()
    print("AMD MLA decode partition policy pins OK")
