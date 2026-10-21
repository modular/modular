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
"""Grades an indexer's top-k selection against exact host scores."""

from std.collections import Set
from max.gpu.host import HostBuffer


struct SelectionStats(Copyable, Movable):
    """Accumulated top-k selection quality against an exact host reference."""

    var rows: Int
    var slots: Int
    var below: Int
    var rows_below: Int
    var count_mismatch: Int
    var deficit_sum: Float64
    var deficit_max: Float32
    var gap_max: Float32
    var spread_min: Float32
    var fp_a: Int
    var fp_b: Int

    def __init__(out self):
        self.rows = 0
        self.slots = 0
        self.below = 0
        self.rows_below = 0
        self.count_mismatch = 0
        self.deficit_sum = 0.0
        self.deficit_max = 0.0
        self.gap_max = 0.0
        self.spread_min = Float32.MAX
        self.fp_a = 0
        self.fp_b = 0

    def report(self, label: String):
        """Prints one machine-readable line for the graded rows.

        `spread_min` is the stimulus witness: the tighter the selected scores
        sit against the peak, the less any disagreement means.
        """
        print(
            "SELGATE ",
            label,
            " rows=",
            self.rows,
            " slots=",
            self.slots,
            " below=",
            self.below,
            " rows_below=",
            self.rows_below,
            " count_mismatch=",
            self.count_mismatch,
            " deficit_mean=",
            self.deficit_sum / Float64(max(self.rows, 1)),
            " deficit_max=",
            self.deficit_max,
            " gap_max=",
            self.gap_max,
            " spread_min=",
            self.spread_min,
            " fp=",
            self.fp_a,
            ":",
            self.fp_b,
            sep="",
        )

    def passed(self, max_deficit: Float32 = 0.01) -> Bool:
        """Whether no row forfeited more than `max_deficit` of its ideal mass.

        A wall, not an expectation: the default sits far above what a bf16
        score fold forfeits, so it trips only on a broken selection.
        """
        return self.deficit_max <= max_deficit and self.count_mismatch == 0


def grade_selection_row(
    host_scores: List[Float32],
    selected: HostBuffer[.int32],
    row: Int,
    top_k: Int,
    mut stats: SelectionStats,
):
    """Grades one row's selection against an exact host top-k.

    Set equality is the wrong assertion: quantized inputs make exact score
    ties routine, so two selectors disagree on membership while forfeiting
    nothing. `below` counts picks strictly under the true k-th score, so a tie
    at the threshold is not a finding; `deficit` is the fraction of ideal
    score mass the selection gave up; `gap` is how far the worst pick fell
    below the threshold as a fraction of the selected range, which catches one
    bad key a mass average would dilute.

    Args:
        host_scores: The row's exact scores, one per live key.
        selected: Selected key indices, `[rows, top_k]`, `-1` for none.
        row: The row to grade.
        top_k: Row stride of `selected`.
        stats: Accumulates the row's grade.
    """
    var nk = len(host_scores)
    var eff = min(top_k, nk)
    stats.rows += 1
    if eff == 0:
        return

    def _desc(lhs: Float32, rhs: Float32) -> Bool:
        return lhs > rhs

    var ranked = host_scores.copy()
    sort(ranked, _desc)

    var s_top = ranked[0]
    var thresh = ranked[eff - 1]
    # Mass is measured above the row's floor so a row of negative scores
    # grades the same as its shifted copy.
    var floor = ranked[nk - 1]
    var mass_ideal = Float64(0)
    for i in range(eff):
        mass_ideal += Float64(ranked[i] - floor)

    var picked = Set[Int]()
    for k in range(top_k):
        var v = Int(selected[row * top_k + k])
        if v >= 0 and v < nk:
            picked.add(v)

    var mass_got = Float64(0)
    var below = 0
    var min_taken = thresh
    for idx in picked:
        var s = host_scores[idx]
        mass_got += Float64(s - floor)
        min_taken = min(min_taken, s)
        if s < thresh:
            below += 1
        # Order-independent fingerprint: unchanged by tie order, changed by
        # any change to the selected set.
        stats.fp_a += idx + 1
        stats.fp_b += (idx + 1) * (idx + 1)

    stats.slots += eff
    stats.below += below
    if below > 0:
        stats.rows_below += 1
    if len(picked) != eff:
        stats.count_mismatch += 1

    var deficit = Float32(
        (mass_ideal - mass_got) / max(mass_ideal, Float64(1e-30))
    )
    stats.deficit_sum += Float64(deficit)
    stats.deficit_max = max(stats.deficit_max, deficit)

    var sel_range = max(s_top - thresh, Float32(1e-30))
    stats.gap_max = max(stats.gap_max, (thresh - min_taken) / sel_range)
    stats.spread_min = min(
        stats.spread_min, (s_top - thresh) / max(abs(s_top), Float32(1e-30))
    )
