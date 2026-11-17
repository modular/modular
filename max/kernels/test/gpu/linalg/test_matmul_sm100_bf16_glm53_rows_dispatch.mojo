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
# SM100 (B200) bf16 outlier-table routing for GLM-5.3's decode projections.
#
# `test_matmul.mojo` checks these rows' numerics, which would still pass on
# the default config if a row stopped matching. `_sm100_outlier_configs` takes
# the first row whose half-open [M, M_end) holds m, so which row a decode M
# selects is a property of the table itself. This test asserts it at every
# band edge, and that one row past each band falls through to the heuristic.

from internal_utils import Table
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    TuningConfigSM100,
    _get_tuning_list_sm100_bf16,
)
from std.testing import assert_equal


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
    var split_k: Int
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
    var out = Route(False, 0, 0, 0, 0, 0, 0, 0, False)
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
                config.num_split_k,
                config.swapAB,
            )
    return out


def assert_band[
    N: Int,
    K: Int,
    m_start: Int,
    m_end: Int,
    mma_m: Int,
    mma_n: Int,
    cta_group: Int = 2,
    cluster_x: Int = 2,
    split_k: Int = 1,
](label: String) raises:
    """Both edges of `[m_start, m_end)` route to the row with these knobs."""
    comptime for m in [m_start, m_end - 1]:
        var route = _route[N, K](m)
        var at = String(label, " (N=", N, ", K=", K, ") M=", m)
        assert_equal(route.found, True, String(at, ": no bf16 row"))
        assert_equal(route.m_start, m_start, String(at, ": row M"))
        assert_equal(route.m_end, m_end, String(at, ": row M_end"))
        assert_equal(route.mma_m, mma_m, String(at, ": mma_m"))
        assert_equal(route.mma_n, mma_n, String(at, ": mma_n"))
        assert_equal(route.cta_group, cta_group, String(at, ": cta_group"))
        assert_equal(route.cluster_x, cluster_x, String(at, ": cluster_x"))
        assert_equal(route.split_k, split_k, String(at, ": split-K"))
        assert_equal(route.swap, True, String(at, ": swapAB"))
    print("[PASS] ", label, " M [", m_start, ", ", m_end, ") -> mma ", mma_m)


def assert_heuristic[N: Int, K: Int](label: String, m: Int) raises:
    """`m` matches no row, so the dispatcher takes the heuristic."""
    assert_equal(
        _route[N, K](m).found,
        False,
        String(label, " (N=", N, ", K=", K, ") M=", m, " matches a row"),
    )


def main() raises:
    # o_proj and the shared expert's down projection: three bands each.
    assert_equal(_row_count[6144, 16384](), 3, "o_proj rows")
    assert_band[6144, 16384, 17, 32, 128, 32]("o_proj")
    assert_band[6144, 16384, 65, 81, 128, 80]("o_proj")
    assert_band[6144, 16384, 81, 97, 128, 96]("o_proj")
    assert_equal(_row_count[6144, 2048](), 3, "shared down rows")
    assert_band[6144, 2048, 17, 32, 128, 32]("shared down")
    assert_band[6144, 2048, 65, 81, 128, 80]("shared down")
    assert_band[6144, 2048, 81, 97, 128, 96]("shared down")
    comptime for m in [16, 32, 64, 97]:
        assert_heuristic[6144, 16384]("o_proj", m)
        assert_heuristic[6144, 2048]("shared down", m)

    # q_b and the fused q_a/kv_a projection: the M 25-31 sampling gap only.
    assert_equal(_row_count[16384, 2048](), 1, "q_b rows")
    assert_band[16384, 2048, 25, 32, 256, 32]("q_b")
    assert_equal(_row_count[2624, 6144](), 1, "q_a/kv_a rows")
    assert_band[2624, 6144, 25, 32, 128, 32]("q_a/kv_a")
    comptime for m in [24, 32]:
        assert_heuristic[16384, 2048]("q_b", m)
        assert_heuristic[2624, 6144]("q_a/kv_a", m)

    # The router gate and the indexer key projection: one split-K row each,
    # over every decode M.
    assert_equal(_row_count[256, 6144](), 1, "router gate rows")
    assert_band[256, 6144, 1, 193, 128, 32, 1, 1, 8]("router gate")
    assert_heuristic[256, 6144]("router gate", 193)
    assert_equal(_row_count[128, 6144](), 1, "indexer key rows")
    assert_band[128, 6144, 1, 257, 128, 32, 1, 1, 8]("indexer key")
    assert_heuristic[128, 6144]("indexer key", 257)
