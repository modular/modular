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
#
# The first matching half-open M range must select the tuned DiT row.
# Check table routing on CPU: full band GEMMs exceed the GPU test memory limit.

from internal_utils import Table
from linalg.matmul.gpu.tile_scheduler import RasterOrder
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    TuningConfigSM100,
    _get_tuning_list_sm100_bf16,
)
from std.testing import assert_equal

# Eight M points of the packed-row band used by the sweep.
comptime BAND_MS = [33152, 38272, 48448, 58688, 73984, 89216, 109632, 111680]

# The landed rows cover [32768, 131072), wider than the band: above every
# small-M decode row, with headroom.
comptime BAND_M = 32768
comptime BAND_M_END = 131072


@fieldwise_init
struct Route(Copyable, ImplicitlyCopyable, Movable):
    """The launch knobs of the table row a given M would dispatch to."""

    var found: Bool
    var m_start: Int
    var m_end: Int
    var mma_m: Int
    var mma_n: Int
    var cta_group: Int
    var cluster_x: Int
    var cluster_y: Int
    var swizzle: Int
    var k_group: Int
    var accum_stages: Int
    var clc_stages: Int
    var split_k: Int
    var stages: Int
    var rst: RasterOrder
    var swap: Bool


def _bf16_rows[N: Int, K: Int]() -> List[TuningConfigSM100]:
    """The bf16 outlier rows the dispatcher would consider for this (N, K)."""
    return Table(_get_tuning_list_sm100_bf16(), "bf16_heuristic_outliers").find(
        rule=lambda (x: TuningConfigSM100) -> Bool: x.K == K and x.N == N
    )


def _row_count[N: Int, K: Int]() -> Int:
    var n = 0
    comptime for _config in _bf16_rows[N, K]():
        n += 1
    return n


def _route[N: Int, K: Int](m: Int) -> Route:
    """The row `_sm100_outlier_configs` would pick for `m`, first match wins."""
    var out = Route(
        False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, RasterOrder.AlongN, False
    )
    comptime for config in _bf16_rows[N, K]():
        if not out.found and m >= config.M and m < config.M_end:
            out = Route(
                True,
                config.M,
                config.M_end,
                config.mma_shape[0],
                config.mma_shape[1],
                config.cta_group,
                config.cluster_shape[0],
                config.cluster_shape[1],
                config.block_swizzle_size,
                config.k_group_size,
                config.num_accum_pipeline_stages,
                config.num_clc_pipeline_stages,
                config.num_split_k,
                config.num_pipeline_stages,
                config.rasterize_order,
                config.swapAB,
            )
    return out


def _assert_tuned_row[
    N: Int, K: Int, exp_mma_n: Int, exp_swap: Bool
](label: String, m: Int) raises:
    """One band M must hit the tuned row with exactly the swept knobs."""
    var route = _route[N, K](m)
    assert_equal(
        route.found,
        True,
        String(label, ": M=", m, " matches no bf16 outlier row"),
    )
    assert_equal(
        route.mma_n,
        exp_mma_n,
        String(label, ": M=", m, " does not route to the tuned row"),
    )
    assert_equal(route.m_start, BAND_M, String(label, ": row M"))
    assert_equal(route.m_end, BAND_M_END, String(label, ": row M_end"))
    # The knobs the sweep won with, so a later edit to the row (or to the
    # table `defaults`) that silently swaps the kernel fails here instead
    # of in a perf run. `defaults` supplies cta_group, k_group_size, the
    # stage counts, split-K and the rasterization order.
    assert_equal(route.mma_m, 256, String(label, ": mma_m"))
    assert_equal(route.swizzle, 8, String(label, ": swizzle"))
    assert_equal(route.cta_group, 2, String(label, ": cta_group"))
    assert_equal(route.cluster_x, 2, String(label, ": cluster_x"))
    assert_equal(route.cluster_y, 1, String(label, ": cluster_y"))
    assert_equal(route.k_group, 1, String(label, ": k_group_size"))
    assert_equal(route.accum_stages, 2, String(label, ": accum stages"))
    assert_equal(route.clc_stages, 2, String(label, ": clc stages"))
    assert_equal(route.stages, 0, String(label, ": explicit stages"))
    assert_equal(route.split_k, 1, String(label, ": split-K"))
    assert_equal(route.rst, RasterOrder.AlongM, String(label, ": rst"))
    assert_equal(route.swap, exp_swap, String(label, ": swapAB"))


def assert_band_route[
    N: Int, K: Int, exp_mma_n: Int, exp_swap: Bool
](label: String) raises:
    """Every M in the band routes to the tuned row; nothing else shares it."""
    # A second row for this (N, K) would shadow, or be shadowed by, the band
    # row depending on file order, so require the shape be exclusively ours.
    assert_equal(
        _row_count[N, K](),
        1,
        String(label, " (N=", N, ", K=", K, "): expected exactly 1 bf16 row"),
    )

    # Every band M, plus both band edges: inclusive start, exclusive end.
    comptime for m in BAND_MS:
        _assert_tuned_row[N, K, exp_mma_n, exp_swap](label, m)
    _assert_tuned_row[N, K, exp_mma_n, exp_swap](label, BAND_M)
    _assert_tuned_row[N, K, exp_mma_n, exp_swap](label, BAND_M_END - 1)

    # Out of band: nothing matches, so the dispatcher takes the heuristic.
    comptime for m in [BAND_M - 1, BAND_M_END, 4 * BAND_M_END]:
        assert_equal(
            _route[N, K](m).found,
            False,
            String(
                label,
                ": M=",
                m,
                " is outside [",
                BAND_M,
                ", ",
                BAND_M_END,
                ") but still matches a tuned row",
            ),
        )

    print(
        "[PASS] ",
        label,
        " N=",
        N,
        " K=",
        K,
        " -> mma 256x",
        exp_mma_n,
        " swapAB=",
        exp_swap,
        " over [",
        BAND_M,
        ", ",
        BAND_M_END,
        ")",
    )


def main() raises:
    # qkv and gate_up both win with swapAB; out and down with mma_n=224.
    assert_band_route[21504, 5376, 256, True]("qkv")
    assert_band_route[28672, 5376, 256, True]("gate_up")
    assert_band_route[5376, 7168, 224, False]("out")
    assert_band_route[5376, 14336, 224, False]("down")
