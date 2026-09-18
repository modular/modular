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
"""Report assembly, text rendering and JSON export for SpecDecCheck.

A :class:`RunReport` is everything one run produced, rendered either for a
terminal or as plain Python types for :func:`json.dump`. Every table that
carries a p-value also carries a total variation distance, because a p-value
says only that a difference is larger than sampling noise and nothing about
its size: a tiny TVD beside a tiny p-value is a large sample noticing a
numerically small difference, kernel numerics for example, while a large TVD
is a real distribution bug.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .samples import Sample, SamplesHeader
from .stats import AGGREGATE_NAMES, AnalysisResult, ControlResult, NodeStats

EXIT_PASS = 0
"""The servers' token distributions are indistinguishable."""

EXIT_FAIL = 1
"""A gating aggregate rejected the null: the servers differ."""

EXIT_ABORTED = 2
"""The run could not answer the question it was asked."""

TOKEN_BUDGET_PREFIX = "max-tokens:"
"""The fixed start of the token-budget calibration line."""

RESULT_PASS = "\u2705 Pass"
"""A gating aggregate cleared its threshold: the verdict itself."""

RESULT_VERIFIED = "\u2714 Verified"
"""A precondition held, so the comparison can be trusted."""

RESULT_FAIL = "\u274c FAIL"
"""A gating aggregate rejected the null, or a precondition did not hold."""

_NULL_MEDIAN = 0.5
_TAIL_LABEL = "TAIL"


@dataclass(frozen=True)
class PromptInfo:
    """One prompt of the run as the report describes it.

    Args:
        index: The prompt's index within the run's prompt set.
        kind: How the prompt reached the run, ``"text"`` from the command
            line or ``"header"`` from a samples file.
        description: A short printable rendering of the prompt.
        num_prompt_tokens: The post-template prompt length the servers
            reported, or ``None`` when no sample carried one.
    """

    index: int
    kind: str
    description: str
    num_prompt_tokens: int | None


@dataclass(frozen=True)
class RunParameters:
    """The knobs one run was invoked with.

    Args:
        baseline_source: Where the baseline samples came from, a server URL
            or a samples file path as given; ``None`` when the switch was
            omitted.
        test_source: Where the test samples came from, as for
            ``baseline_source``.
        baseline_header: The baseline samples file's header, or ``None``
            when the baseline was a live server.
        test_header: The test file's header, as for ``baseline_header``.
        endpoint: The endpoint live text prompts were sent to.
        request_params: The extra body fields sent to live servers, empty
            when the model's defaults were used. A file source's own are in
            its header.
        baseline_model: The model behind the baseline samples.
        test_model: The model behind the test samples.
        prompts: The run's prompts, in prompt-index order.
        num_samples: The completions drawn per live server and prompt, or
            ``None`` when both sources were files.
        min_prefix_samples: The count each server needed at a prefix to test
            it.
        baseline_max_tokens: The baseline's output token budget, ``None``
            when the switch was omitted.
        test_max_tokens: The test's budget, which need not match.
        depth_limit: The exclusive depth bound the analysis kept to.
        seed: The seed base of each server, baseline then test, as a file
            source's header recorded it.
        num_permutations: The permutations behind every p-value.
        false_alarm_rate: The per-run bound on spurious failures, which
            the two gates share.
        alpha: The per-gate significance level.
        min_token_samples: The pooled count a token needed at a prefix to
            get its own bucket rather than the tail.
        effect_min_per_server: The count each server needed at a node before
            its TVD was eligible for ``max_tvd``.
        max_concurrency: The cap on requests in flight, baseline then test,
            as a file source's header recorded it.
    """

    baseline_source: str | None
    test_source: str | None
    baseline_header: SamplesHeader | None
    test_header: SamplesHeader | None
    endpoint: str
    request_params: Mapping[str, object]
    baseline_model: str | None
    test_model: str | None
    prompts: tuple[PromptInfo, ...]
    num_samples: int | None
    min_prefix_samples: int
    baseline_max_tokens: int | None
    test_max_tokens: int | None
    depth_limit: int
    seed: tuple[int, int]
    num_permutations: int
    false_alarm_rate: float
    alpha: float
    min_token_samples: int
    effect_min_per_server: int
    max_concurrency: tuple[int, int]


@dataclass(frozen=True)
class CollectionSummary:
    """What collection saw for one server and one prompt.

    Args:
        server: Which server, ``"baseline"`` or ``"test"``.
        prompt_index: Index of the prompt within the run's prompt set.
        num_samples: How many completions the server returned.
        mean_generated_length: The mean generated ids per completion.
        finish_reasons: How many completions ended for each reason.
        source: The server URL or the file path they came from.
        from_file: Whether they were read from a file rather than collected.
        created_at: When a file's samples were collected.
        recorded_url: The server a file's samples were drawn from.
        elapsed_s: The wall time this collection took, which a file row does
            not have.
        requests_per_second: The collection rate, as for ``elapsed_s``.
    """

    server: str
    prompt_index: int
    num_samples: int
    mean_generated_length: float
    finish_reasons: Mapping[str, int]
    source: str
    from_file: bool
    created_at: str | None
    recorded_url: str | None
    elapsed_s: float | None
    requests_per_second: float | None


@dataclass(frozen=True, eq=False)
class ControlEntry:
    """One prompt's depth-0 control outcome.

    Args:
        prompt_index: Index of the prompt the control covers.
        result: What :func:`.stats.control_check` decided.
    """

    prompt_index: int
    result: ControlResult


@dataclass(frozen=True)
class StabilityResult:
    """One server's first-half against second-half comparison.

    It answers a different question from the run's verdict: whether one
    server's own output drifted over the collection. Drift, or a comparison
    that shared no node, aborts the run.

    Args:
        server: Which server was split in half.
        num_nodes: How many nodes the two halves had in common.
        p_sum: The permutation p-value of the halves' summed chi-square.
        p_max: The permutation p-value of their largest chi-square per dof.
        sum_chi_square_per_dof: Their summed chi-square per summed dof, near
            ``1.0`` when the halves agree.
        max_chi_square_per_dof: Their largest per-node chi-square per dof.
        mean_tvd: The observation-weighted mean total variation distance.
        note: A printable remark, empty when the comparison was ordinary.
    """

    server: str
    num_nodes: int
    p_sum: float
    p_max: float
    sum_chi_square_per_dof: float
    max_chi_square_per_dof: float
    mean_tvd: float
    note: str

    def result(self, alpha: float) -> str:
        """``Verified`` or ``FAIL`` against the per-gate ``alpha``.

        A check that shared no node fails too: its p-values of 1.0 come from
        having compared nothing, not from the halves agreeing.
        """
        if not self.num_nodes:
            return RESULT_FAIL
        drifted = self.p_sum < alpha or self.p_max < alpha
        return RESULT_FAIL if drifted else RESULT_VERIFIED


@dataclass(frozen=True)
class TokenBudget:
    """How well ``--max-tokens`` matched the depths the run could test.

    Testing the deepest position the cap allows means the cap, not the data,
    ended the tree; leaving it untested means the cap paid for decode work no
    node could use.

    Args:
        cap: The depth limit the analysis kept to.
        deepest_tested: The deepest position the node test compared.
        tokens_generated: Generated ids summed over every sample the analysis
            saw.
        tokens_beyond: Those of them no node could test.
        wasted_fraction: Their share of ``tokens_generated``.
        frontier_children: How many prefixes just past the cap would have had
            enough samples to test.
        candidates_at_cap: Candidate prefixes at the last tested position.
        tested_at_cap: How many of those were testable.
        testable_rate: ``tested_at_cap / candidates_at_cap``, or ``0.0``.
        estimated_testable: ``frontier_children`` scaled by that rate.
        verdict: ``"cap_reached"`` or ``"cap_not_reached"``.
        line: The one-line verdict, starting with
            :data:`TOKEN_BUDGET_PREFIX`.
    """

    cap: int
    deepest_tested: int
    tokens_generated: int
    tokens_beyond: int
    wasted_fraction: float
    frontier_children: int
    candidates_at_cap: int
    tested_at_cap: int
    testable_rate: float
    estimated_testable: int
    verdict: str
    line: str


@dataclass(frozen=True, eq=False)
class RunReport:
    """Everything one SpecDecCheck run produced.

    A stage the run aborted before reaching leaves its field empty or
    ``None``.

    Args:
        parameters: The knobs the run was invoked with.
        collection: One summary per server and prompt.
        controls: One entry per prompt's depth-0 control.
        analysis: The node test and its verdict.
        pruning: How many prefixes the samples produced and how many the
            prefix minimum kept.
        nodes_dropped: How many selected nodes bucketing dropped for having
            fewer than two buckets.
        token_budget: The token-budget calibration.
        stability: One entry per server.
        exit_code: The process exit code.
        reason: A one-line account of that exit code.
    """

    parameters: RunParameters
    collection: tuple[CollectionSummary, ...]
    controls: tuple[ControlEntry, ...]
    analysis: AnalysisResult | None
    pruning: PruningSummary | None
    nodes_dropped: int
    token_budget: TokenBudget | None
    stability: tuple[StabilityResult, ...]
    exit_code: int
    reason: str

    @property
    def compares_servers(self) -> bool:
        """Whether both servers were given, so the cross-server stages ran."""
        parameters = self.parameters
        return (
            parameters.baseline_source is not None
            and parameters.test_source is not None
        )

    @property
    def verdict_label(self) -> str:
        """The exit code as the word the report leads with."""
        if self.exit_code == EXIT_PASS:
            return "PASS"
        if self.exit_code == EXIT_FAIL:
            return "FAIL"
        return "ABORTED"


def summarize_live(
    server: str,
    prompt_index: int,
    samples: Sequence[Sample],
    *,
    url: str,
    elapsed_s: float,
) -> CollectionSummary:
    """Summarizes one live server's samples for one prompt.

    Args:
        server: Which server produced them.
        prompt_index: Index of the prompt they answer.
        samples: The samples, all of one server and prompt.
        url: The server they were drawn from.
        elapsed_s: The measured wall time of the collection.

    Returns:
        The summary, with the collection rate this run measured.
    """
    count = len(samples)
    return _summary(
        server,
        prompt_index,
        samples,
        source=url,
        from_file=False,
        created_at=None,
        recorded_url=None,
        elapsed_s=elapsed_s,
        requests_per_second=count / elapsed_s if elapsed_s > 0 else 0.0,
    )


def summarize_file(
    server: str,
    prompt_index: int,
    samples: Sequence[Sample],
    *,
    path: str,
    header: SamplesHeader,
) -> CollectionSummary:
    """Summarizes one prompt's samples read back from a file.

    Args:
        server: The role the samples are playing in this run.
        prompt_index: Index of the prompt they answer.
        samples: The samples, all of one server and prompt.
        path: The file they were read from.
        header: That file's header, for when and where they came from.

    Returns:
        The summary, with no rate, since this run did not time the
        collection.
    """
    return _summary(
        server,
        prompt_index,
        samples,
        source=path,
        from_file=True,
        created_at=header.created_at,
        recorded_url=header.url,
        elapsed_s=None,
        requests_per_second=None,
    )


def _summary(
    server: str,
    prompt_index: int,
    samples: Sequence[Sample],
    *,
    source: str,
    from_file: bool,
    created_at: str | None,
    recorded_url: str | None,
    elapsed_s: float | None,
    requests_per_second: float | None,
) -> CollectionSummary:
    """Counts what one server's samples for one prompt look like."""
    count = len(samples)
    mean_length = (
        sum(len(sample.token_ids) for sample in samples) / count
        if count
        else 0.0
    )
    return CollectionSummary(
        server=server,
        prompt_index=prompt_index,
        num_samples=count,
        mean_generated_length=mean_length,
        finish_reasons=dict(
            sorted(Counter(sample.finish_reason for sample in samples).items())
        ),
        source=source,
        from_file=from_file,
        created_at=created_at,
        recorded_url=recorded_url,
        elapsed_s=elapsed_s,
        requests_per_second=requests_per_second,
    )


def stability_from_analysis(
    server: str, analysis: AnalysisResult
) -> StabilityResult:
    """Reduces one server's halves analysis to the numbers the report shows.

    Args:
        server: Which server was split into halves.
        analysis: The node test with the halves as the two groups.

    Returns:
        The summary, with a note when the halves shared no testable node.
    """
    observed = analysis.observed
    return StabilityResult(
        server=server,
        num_nodes=observed.num_nodes,
        p_sum=analysis.verdict.p_sum,
        p_max=analysis.verdict.p_max,
        sum_chi_square_per_dof=observed.sum_chi_square_per_dof,
        max_chi_square_per_dof=observed.max_chi_square_per_dof,
        mean_tvd=observed.mean_tvd,
        note=(
            ""
            if observed.num_nodes
            else "no node had enough samples in both halves; p-values are"
            " vacuous"
        ),
    )


def render_text(report: RunReport) -> str:
    """Renders a report as plain text with aligned columns.

    A single-server run stops after the stability table.

    Args:
        report: The report to render.

    Returns:
        The rendered report, newline terminated.
    """
    lines: list[str] = []
    lines.extend(_verdict_lines(report))
    lines.extend(_stability_lines(report))
    if report.compares_servers:
        lines.extend(_control_lines(report))
        lines.extend(_coverage_lines(report))
        lines.extend(_token_budget_lines(report))
        lines.extend(_aggregate_lines(report))
    return "\n".join(lines) + "\n"


def to_json_dict(report: RunReport) -> dict[str, object]:
    """Converts a report to plain Python types for :func:`json.dump`.

    The per-node statistics are included in full.

    Args:
        report: The report to convert.

    Returns:
        The report as nested dicts, lists, strings and numbers.
    """
    return {
        "verdict": report.verdict_label,
        "exit_code": report.exit_code,
        "reason": report.reason,
        "parameters": _parameters_json(report.parameters),
        "collection": [
            {
                "server": summary.server,
                "prompt_index": summary.prompt_index,
                "num_samples": summary.num_samples,
                "mean_generated_length": summary.mean_generated_length,
                "finish_reasons": dict(summary.finish_reasons),
                "source": summary.source,
                "from_file": summary.from_file,
                "created_at": summary.created_at,
                "recorded_url": summary.recorded_url,
                "elapsed_s": summary.elapsed_s,
                "requests_per_second": summary.requests_per_second,
            }
            for summary in report.collection
        ],
        "controls": [
            _control_json(entry, row)
            for entry, row in zip(
                report.controls,
                _control_rows(report.controls, alpha=report.parameters.alpha),
                strict=True,
            )
        ],
        "analysis": (
            None
            if report.analysis is None
            else _analysis_json(report.analysis, alpha=report.parameters.alpha)
        ),
        "coverage": _coverage_json(report),
        "token_budget": (
            None
            if report.token_budget is None
            else _token_budget_json(report.token_budget)
        ),
        "stability": {
            "servers": [
                {
                    "server": result.server,
                    "num_nodes": result.num_nodes,
                    "p_sum": result.p_sum,
                    "p_max": result.p_max,
                    "sum_chi_square_per_dof": result.sum_chi_square_per_dof,
                    "max_chi_square_per_dof": result.max_chi_square_per_dof,
                    "mean_tvd": result.mean_tvd,
                    "threshold": report.parameters.alpha,
                    "result": result.result(report.parameters.alpha),
                    "note": result.note,
                }
                for result in report.stability
            ],
        },
    }


@dataclass(frozen=True)
class _AggregateRow:
    """One row of the aggregates table, in its own printed units."""

    name: str
    label: str
    observed: float
    null_median: float
    null_threshold: float | None
    p_value: float
    decimals: int
    alpha: float | None

    @property
    def result(self) -> str:
        """``Pass`` or ``FAIL`` for a gate, ``n/a`` for a reported-only row."""
        if self.alpha is None:
            return "n/a"
        return RESULT_FAIL if self.p_value < self.alpha else RESULT_PASS


def _verdict_lines(report: RunReport) -> list[str]:
    """Renders the leading verdict line."""
    return [
        _wrap(f"{report.verdict_label}: {report.reason}", indent=""),
        "",
    ]


@dataclass(frozen=True)
class _ControlRow:
    """One prompt's row of the control table, in chi-square per dof units."""

    prompt_index: int
    chi_square_per_dof: float | None
    tvd: float | None
    null_median: float | None
    null_threshold: float | None
    p_value: float | None
    result: str


def _control_rows(
    controls: Sequence[ControlEntry], *, alpha: float
) -> list[_ControlRow]:
    """Builds the control table, shared by the text and JSON renderings."""
    rows = []
    for entry in controls:
        result = entry.result
        stats, null = result.stats, result.null
        if stats is None or null is None:
            # A forced first token is agreement at zero entropy; a prompt one
            # server never answered has nothing to compare.
            forced = result.passed
            rows.append(
                _ControlRow(
                    prompt_index=entry.prompt_index,
                    chi_square_per_dof=0.0 if forced else None,
                    tvd=0.0 if forced else None,
                    null_median=0.0 if forced else None,
                    null_threshold=0.0 if forced else None,
                    p_value=result.p_value,
                    result=RESULT_VERIFIED if forced else "degenerate",
                )
            )
            continue
        dof = float(stats.dof) or 1.0
        median, threshold = (
            float(value) / dof
            for value in null.quantiles("sum", (_NULL_MEDIAN, 1.0 - alpha))
        )
        rows.append(
            _ControlRow(
                prompt_index=entry.prompt_index,
                chi_square_per_dof=stats.chi_square_per_dof,
                tvd=stats.tvd,
                null_median=median,
                null_threshold=threshold,
                p_value=result.p_value,
                result=RESULT_VERIFIED if result.passed else RESULT_FAIL,
            )
        )
    return rows


def _optional(value: float | None, decimals: int) -> str:
    """Formats a float with fixed decimals, or a dash for ``None``."""
    return "-" if value is None else _fixed(value, decimals)


def _control_lines(report: RunReport) -> list[str]:
    """Renders the per-prompt depth-0 control section."""
    lines = ["Cross-Server Control: Prefill Token Check"]
    if not report.controls:
        lines.extend(["  (not reached)", ""])
        return lines
    alpha = report.parameters.alpha
    table = [
        [
            str(row.prompt_index),
            _optional(row.chi_square_per_dof, 3),
            _optional(row.tvd, 4),
            _optional(row.null_median, 3),
            _optional(row.null_threshold, 3),
            "-" if row.p_value is None else f"{row.p_value:.3g}",
            f"{alpha:.3g}",
            row.result,
        ]
        for row in _control_rows(report.controls, alpha=alpha)
    ]
    lines.extend(
        _table(
            [
                "prompt",
                "X^2",
                "TVD",
                "null p50",
                _threshold_label(alpha),
                "p-value",
                "threshold",
                "result",
            ],
            table,
            min_widths=(len("baseline"), len("nodes")),
        )
    )
    for entry in report.controls:
        if not entry.result.passed:
            lines.append(
                f"  prompt {entry.prompt_index}: {entry.result.message}"
            )
    lines.append("")
    return lines


def _aggregate_lines(report: RunReport) -> list[str]:
    """Renders the aggregates against their permutation null."""
    lines = ["Cross-Server Test: Generated Distributions"]
    if report.analysis is None:
        lines.extend(["  (not reached)", ""])
        return lines
    alpha = report.parameters.alpha
    rows = _aggregate_rows(report.analysis, alpha=alpha)
    table = [
        [
            row.label,
            _fixed(row.observed, row.decimals),
            _fixed(row.null_median, row.decimals),
            "n/a"
            if row.null_threshold is None
            else _fixed(row.null_threshold, row.decimals),
            f"{row.p_value:.3g}",
            "n/a" if row.alpha is None else f"{row.alpha:.3g}",
            row.result,
        ]
        for row in rows
    ]
    lines.extend(
        _table(
            [
                "aggregate",
                "observed",
                "null p50",
                _threshold_label(alpha),
                "p-value",
                "threshold",
                "result",
            ],
            table,
        )
    )
    lines.append("")
    return lines


@dataclass(frozen=True)
class PruningSummary:
    """What the samples produced and what the prefix minimum kept.

    Args:
        tokens_generated: Generated tokens over both servers and all prompts.
        distinct_prefixes: Prefixes at least one sequence continued past, at
            every testable depth.
        candidate_prefixes: Those that reached the prefix minimum and became
            nodes.
    """

    tokens_generated: int
    distinct_prefixes: int
    candidate_prefixes: int

    @property
    def pruned_prefixes(self) -> int:
        """Prefixes the minimum removed."""
        return self.distinct_prefixes - self.candidate_prefixes

    @property
    def pruned_fraction(self) -> float:
        """Share of distinct prefixes removed, ``0.0`` when there were none."""
        if self.distinct_prefixes == 0:
            return 0.0
        return self.pruned_prefixes / self.distinct_prefixes


@dataclass(frozen=True)
class CoverageRow:
    """One depth of the coverage table, over the tested nodes.

    ``pooled`` counts tested next-token observations, not distinct samples:
    a sample contributes one at every depth it reaches. ``mean_support`` is
    the mean bucket count, ``mean_tail_mass`` the share of those
    observations that landed in the tail bucket, and ``mean_entropy_bits``
    is taken over the buckets rather than the raw tokens. The means are over
    nodes, so the totals row, whose ``depth`` is ``None``, is the
    node-weighted mean of the depth rows.
    """

    depth: int | None
    nodes: int
    pooled: int
    mean_top_token_probability: float
    mean_entropy_bits: float
    mean_support: float
    mean_tail_mass: float

    @property
    def depth_label(self) -> str:
        """The depth as printed, ``all`` for the totals row."""
        return "all" if self.depth is None else str(self.depth)


def _entropy_bits(counts: Iterable[int]) -> float:
    """Computes the Shannon entropy in bits of a count vector."""
    collected = [int(count) for count in counts if count > 0]
    total = sum(collected)
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in collected
    )


def _tail_mass(stats: NodeStats) -> float:
    """Returns the share of a node's pooled samples that fell in the tail."""
    total = stats.node.pooled_count
    if total == 0:
        return 0.0
    pooled = stats.node.pooled_counts()
    return sum(pooled[token] for token in stats.bucketed.tail_tokens) / total


def _tested_rows(analysis: AnalysisResult) -> list[CoverageRow]:
    """Summarizes the tested nodes by depth, after selection and bucketing."""
    by_depth: dict[int, list[NodeStats]] = defaultdict(list)
    for stats in analysis.node_stats:
        by_depth[stats.depth].append(stats)
    return [
        _coverage_row(depth, group) for depth, group in sorted(by_depth.items())
    ]


def _coverage_totals(analysis: AnalysisResult) -> CoverageRow | None:
    """Summarizes every tested node at once, for the table's totals row.

    It is the same reduction as a depth row over one larger group, so its
    means are the node-weighted means of the depth rows by construction.
    """
    if not analysis.node_stats:
        return None
    return _coverage_row(None, list(analysis.node_stats))


def _coverage_row(depth: int | None, group: Sequence[NodeStats]) -> CoverageRow:
    """Reduces a group of tested nodes to one coverage row."""
    return CoverageRow(
        depth=depth,
        nodes=len(group),
        pooled=sum(stats.pooled_count for stats in group),
        mean_top_token_probability=_mean(
            stats.node.top_token_probability() for stats in group
        ),
        mean_entropy_bits=_mean(
            _entropy_bits(stats.bucketed.pooled) for stats in group
        ),
        mean_support=_mean(
            float(stats.bucketed.num_buckets) for stats in group
        ),
        mean_tail_mass=_mean(_tail_mass(stats) for stats in group),
    )


def _coverage_table(
    rows: Sequence[CoverageRow], totals: CoverageRow | None
) -> list[str]:
    """Renders the coverage table, with the totals row under its own rule."""
    return _table(
        [
            "depth",
            "nodes",
            "pooled",
            "mean p(top)",
            "mean bits",
            "mean buckets",
            "mean tail",
        ],
        [_coverage_cells(row) for row in rows],
        footer=None if totals is None else _coverage_cells(totals),
    )


def _coverage_cells(row: CoverageRow) -> list[str]:
    """Formats one coverage row for the console table."""
    return [
        row.depth_label,
        str(row.nodes),
        str(row.pooled),
        f"{row.mean_top_token_probability:.4f}",
        f"{row.mean_entropy_bits:.3f}",
        f"{row.mean_support:.1f}",
        f"{row.mean_tail_mass:.3f}",
    ]


_EXAMPLE_TOKENS = 5
_EXAMPLE_CHARS = 200
_VERBATIM_COMPLETIONS = 10


def render_control_prefill(report: RunReport, samples: Sequence[Sample]) -> str:
    """Renders what each server produced at depth 0, prompt by prompt.

    Shows the first-token distribution of both servers side by side, with an
    example completion per server for the tokens that differ most, and then
    a few completions from each server verbatim, so a server emitting
    nonsense is visible at a glance.
    """
    lines: list[str] = []
    by_prompt: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.token_ids:
            by_prompt[sample.prompt_index].append(sample)
    contributions = _control_contributions(report)
    for prompt_index in sorted(by_prompt):
        own = by_prompt[prompt_index]
        groups = {
            server: [s for s in own if s.server == server]
            for server in ("baseline", "test")
        }
        counts = {
            server: Counter(s.token_ids[0] for s in group)
            for server, group in groups.items()
        }
        totals = {server: len(group) for server, group in groups.items()}
        lines.append(
            f"prompt {prompt_index}: first-token distribution,"
            f" {totals['baseline']} baseline / {totals['test']} test samples"
        )
        tokens = sorted(
            set(counts["baseline"]) | set(counts["test"]),
            key=lambda token: (
                -abs(
                    _share(counts, totals, "test", token)
                    - _share(counts, totals, "baseline", token)
                )
            ),
        )
        per_token = contributions.get(prompt_index, {})
        rows = []
        for token in tokens:
            baseline = _share(counts, totals, "baseline", token)
            test = _share(counts, totals, "test", token)
            rows.append(
                [
                    str(token),
                    f"{baseline:.3f}",
                    f"{test:.3f}",
                    f"{test - baseline:+.3f}",
                    _contribution_cell(per_token, token),
                ]
            )
        lines.extend(
            _table(["token", "baseline", "test", "diff", "contribution"], rows)
        )
        lines.append("")
        for token in tokens[:_EXAMPLE_TOKENS]:
            lines.append(f"  examples for token {token}")
            for server in ("baseline", "test"):
                example = next(
                    (s for s in groups[server] if s.token_ids[0] == token),
                    None,
                )
                text = "(none)" if example is None else _excerpt(example.text)
                lines.append(f"    {server + ':':<10}{text}")
        lines.append("")
        for server in ("baseline", "test"):
            lines.append(
                f"  first {_VERBATIM_COMPLETIONS} completions from {server}"
            )
            for sample in sorted(groups[server], key=lambda s: s.request_index)[
                :_VERBATIM_COMPLETIONS
            ]:
                lines.append(f"    {_excerpt(sample.text)}")
            lines.append("")
    return "\n".join(lines) + "\n"


def _share(
    counts: Mapping[str, Counter[int]],
    totals: Mapping[str, int],
    server: str,
    token: int,
) -> float:
    """Returns one server's share of samples that began with ``token``."""
    total = totals[server]
    return counts[server][token] / total if total else 0.0


def _control_contributions(report: RunReport) -> dict[int, dict[int, str]]:
    """Maps each prompt's first tokens to their chi-square contribution.

    Tokens pooled into the tail share the tail's contribution and are marked.
    """
    result: dict[int, dict[int, str]] = {}
    for entry in report.controls:
        stats = entry.result.stats
        if stats is None:
            continue
        per_token: dict[int, str] = {}
        for label, contribution in zip(
            stats.bucketed.labels, stats.contributions, strict=True
        ):
            if label is None:
                for token in stats.bucketed.tail_tokens:
                    per_token[token] = f"{contribution:.3f} (tail)"
            else:
                per_token[label] = f"{contribution:.3f}"
        result[entry.prompt_index] = per_token
    return result


def _contribution_cell(per_token: Mapping[int, str], token: int) -> str:
    """Returns a token's contribution, or a dash when the control had none."""
    return per_token.get(token, "-")


def _excerpt(text: str) -> str:
    """Quotes a completion on one line, escaped and cut to a readable length."""
    escaped = (
        text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
    )
    if len(escaped) > _EXAMPLE_CHARS:
        escaped = escaped[: _EXAMPLE_CHARS - 3] + "..."
    return f'"{escaped}"' if text else "(empty)"


CSV_TABLES = (
    "per-server-control.csv",
    "cross-server-control.csv",
    "coverage.csv",
    "cross-server-test.csv",
)
"""The CSV file written for each of the report's four tables, in print order."""

SUMMARY_CSV = "summary.csv"
"""A one-row CSV of the run's main settings and its four aggregate p-values."""


def write_csv_tables(report: RunReport, outdir: Path) -> None:
    """Writes each of the report's tables as a CSV of raw values.

    A table the run never reached is written as its header alone, and a
    single-server run writes only the per-server control table. Result cells
    carry the bare word, without the symbol the console shows. The summary
    row is :func:`summary_csv`, which the caller writes.
    """
    alpha = report.parameters.alpha
    _write_csv(
        outdir / CSV_TABLES[0],
        [
            "server",
            "nodes",
            "sum_chi_square_per_dof",
            "max_chi_square_per_dof",
            "mean_tvd",
            "p_sum",
            "p_max",
            "threshold",
            "result",
        ],
        [
            [
                result.server,
                result.num_nodes,
                result.sum_chi_square_per_dof,
                result.max_chi_square_per_dof,
                result.mean_tvd,
                result.p_sum,
                result.p_max,
                alpha,
                _plain_result(result.result(alpha)),
            ]
            for result in report.stability
        ],
    )
    if not report.compares_servers:
        return
    _write_csv(
        outdir / CSV_TABLES[1],
        [
            "prompt_index",
            "chi_square_per_dof",
            "tvd",
            "null_median",
            "null_threshold",
            "p_value",
            "threshold",
            "result",
        ],
        [
            [
                row.prompt_index,
                row.chi_square_per_dof,
                row.tvd,
                row.null_median,
                row.null_threshold,
                row.p_value,
                alpha,
                _plain_result(row.result),
            ]
            for row in _control_rows(report.controls, alpha=alpha)
        ],
    )
    _write_csv(
        outdir / CSV_TABLES[2],
        list(CoverageRow.__dataclass_fields__),
        [] if report.analysis is None else _coverage_csv_rows(report.analysis),
    )
    _write_csv(
        outdir / CSV_TABLES[3],
        [
            "aggregate",
            "observed",
            "null_median",
            "null_threshold",
            "p_value",
            "threshold",
            "result",
        ],
        []
        if report.analysis is None
        else [
            [
                row.name,
                row.observed,
                row.null_median,
                row.null_threshold,
                row.p_value,
                row.alpha,
                _plain_result(row.result),
            ]
            for row in _aggregate_rows(report.analysis, alpha=alpha)
        ],
    )


def summary_csv(
    report: RunReport, *, repeat: int | None = None
) -> tuple[list[str], list[object]]:
    """Builds one run's row of the summary CSV, for tabulating runs.

    Cells a run has no value for -- an omitted server, a stage that did
    not run, a p-value never computed -- are left empty, not placeholder.

    Args:
        report: The run to summarize.
        repeat: The run's index among a job's repeats, which adds a
            ``repeat`` column after the inputs; ``None`` omits the column.

    Returns:
        The column names and the one row, in the same order.
    """
    parameters = report.parameters
    given = (
        parameters.baseline_source is not None,
        parameters.test_source is not None,
    )
    concurrency = [
        cap if present else ""
        for cap, present in zip(parameters.max_concurrency, given, strict=True)
    ]
    p_values: dict[str, object] = dict.fromkeys(AGGREGATE_NAMES, "")
    if report.analysis is not None:
        for aggregate in _aggregate_rows(
            report.analysis, alpha=parameters.alpha
        ):
            p_values[aggregate.name] = aggregate.p_value
    columns = [
        "baseline-concurrency",
        "test-concurrency",
        "num-samples",
        "max-tokens",
        "min-prefix-samples",
        "min-token-samples",
    ]
    cells: list[object] = [
        *concurrency,
        _summary_num_samples(parameters),
        parameters.depth_limit,
        parameters.min_prefix_samples,
        parameters.min_token_samples,
    ]
    if repeat is not None:
        columns.append("repeat")
        cells.append(repeat)
    columns.extend(
        [
            "control-stability",
            "control-prefill",
            *(f"p-{name.replace('_', '-')}" for name in AGGREGATE_NAMES),
        ]
    )
    cells.extend(
        [
            _summary_result(
                result.result(parameters.alpha) for result in report.stability
            ),
            _summary_result(
                control.result
                for control in _control_rows(
                    report.controls, alpha=parameters.alpha
                )
            ),
            *p_values.values(),
        ]
    )
    return columns, cells


def write_summary_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    """Writes the summary CSV, one row per run, from :func:`summary_csv`."""
    _write_csv(path, columns, rows)


def _summary_result(labels: Iterable[str]) -> str:
    """Reduces one control table's result cells to a single plain word.

    ``FAIL`` if any row failed, ``Verified`` if every row was, and blank when
    the stage did not run or a row was degenerate.
    """
    collected = [_plain_result(label) for label in labels]
    fail, verified = _plain_result(RESULT_FAIL), _plain_result(RESULT_VERIFIED)
    if fail in collected:
        return fail
    if collected and all(label == verified for label in collected):
        return verified
    return ""


def _summary_num_samples(parameters: RunParameters) -> int | str:
    """Returns the samples drawn per server and prompt, or the recorded one.

    With no live source the count falls back to what the files' headers
    recorded, and to blank when they disagree.
    """
    if parameters.num_samples is not None:
        return parameters.num_samples
    recorded = {
        header.num_samples
        for header in (parameters.baseline_header, parameters.test_header)
        if header is not None
    }
    return recorded.pop() if len(recorded) == 1 else ""


def _coverage_csv_rows(analysis: AnalysisResult) -> list[list[object]]:
    """Lists the depth rows and then the totals row, labeled ``all``."""
    rows = _tested_rows(analysis)
    totals = _coverage_totals(analysis)
    if totals is not None:
        rows.append(totals)
    return [
        list({**asdict(row), "depth": row.depth_label}.values()) for row in rows
    ]


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    """Writes one CSV with a header row."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def _plain_result(label: str) -> str:
    """Strips the console symbol from a result label."""
    for prefix in (RESULT_PASS, RESULT_VERIFIED, RESULT_FAIL):
        if label == prefix:
            return prefix.split(" ", 1)[1]
    return label


def _coverage_lines(report: RunReport) -> list[str]:
    """Renders the pruning and filter counts, then the tested-node coverage.

    The counts read chronologically: what the samples generated, what the
    prefix minimum pruned, and what bucketing then dropped.
    """
    title = "Coverage by depth, after filtering"
    lines: list[str] = []
    if (pruning := report.pruning) is not None:
        lines.extend(
            [
                f"Generated {pruning.tokens_generated:,} tokens, yielding"
                f" {pruning.distinct_prefixes:,} distinct prefixes.",
                f"Pruned {pruning.pruned_prefixes:,}"
                f" ({pruning.pruned_fraction:.1%}), leaving"
                f" {pruning.candidate_prefixes:,} candidate prefixes.",
            ]
        )
    if report.analysis is None:
        if lines:
            lines.append("")
        lines.extend([title, "  (not reached)", ""])
        return lines
    lines.extend(
        [
            f"{report.nodes_dropped} node(s) dropped for tail-only buckets.",
            "",
            title,
        ]
    )
    lines.extend(
        _coverage_table(
            _tested_rows(report.analysis), _coverage_totals(report.analysis)
        )
    )
    lines.append("")
    return lines


def _coverage_json(report: RunReport) -> dict[str, object] | None:
    """Converts the coverage table to plain types, ``None`` if not built."""
    if report.pruning is None:
        return None
    return {
        "after_selection": (
            None
            if report.analysis is None
            else [asdict(row) for row in _tested_rows(report.analysis)]
        ),
        "totals": _totals_json(report.analysis),
        "pruning": {
            "tokens_generated": report.pruning.tokens_generated,
            "distinct_prefixes": report.pruning.distinct_prefixes,
            "candidate_prefixes": report.pruning.candidate_prefixes,
            "pruned_prefixes": report.pruning.pruned_prefixes,
            "pruned_fraction": report.pruning.pruned_fraction,
        },
        "nodes_dropped_few_buckets": report.nodes_dropped,
    }


def _totals_json(analysis: AnalysisResult | None) -> dict[str, object] | None:
    """Converts the coverage totals row to plain types, minus its depth."""
    totals = None if analysis is None else _coverage_totals(analysis)
    if totals is None:
        return None
    return {
        key: value for key, value in asdict(totals).items() if key != "depth"
    }


def _token_budget_lines(report: RunReport) -> list[str]:
    """Renders the one-line token-budget calibration verdict."""
    line = (
        f"{TOKEN_BUDGET_PREFIX} (not reached)"
        if report.token_budget is None
        else report.token_budget.line
    )
    return [_wrap(line), ""]


def _stability_lines(report: RunReport) -> list[str]:
    """Renders the halves comparison, which aborts on drift or on no nodes."""
    lines = ["Per-Server Control: Hi-Lo Self-Consistency Check"]
    if not report.stability:
        lines.append(_wrap("(not reached)"))
        lines.append("")
        return lines
    alpha = report.parameters.alpha
    rows = [
        [
            result.server,
            str(result.num_nodes),
            f"{result.sum_chi_square_per_dof:.3f}",
            f"{result.max_chi_square_per_dof:.3f}",
            f"{result.mean_tvd:.4f}",
            f"{result.p_sum:.3g}",
            f"{result.p_max:.3g}",
            f"{alpha:.3g}",
            result.result(alpha),
        ]
        for result in report.stability
    ]
    lines.extend(
        _table(
            [
                "server",
                "nodes",
                "sum",
                "max",
                "mean TVD",
                "p(sum)",
                "p(max)",
                "threshold",
                "result",
            ],
            rows,
        )
    )
    for result in report.stability:
        if result.note:
            lines.append(f"  {result.server}: {result.note}")
    lines.append("")
    return lines


def _threshold_label(alpha: float) -> str:
    """Names the null quantile a gate must stay under, ``null p99.9``."""
    return f"null p{100.0 * (1.0 - alpha):g}"


def _aggregate_rows(
    analysis: AnalysisResult, *, alpha: float
) -> list[_AggregateRow]:
    """Builds the aggregates table, shared by the text and JSON renderings."""
    observed = analysis.observed
    null = analysis.null
    quantiles = (_NULL_MEDIAN, 1.0 - alpha)
    dof = float(observed.sum_dof) if observed.sum_dof > 0 else 1.0
    sum_median, sum_threshold = (
        value / dof for value in null.quantiles("sum", quantiles)
    )
    max_median, max_threshold = null.quantiles("max", quantiles)
    mean_median, mean_threshold = null.quantiles("mean_tvd", quantiles)
    tvd_median, tvd_threshold = null.quantiles("max_tvd", quantiles)
    div_median, div_threshold = null.quantiles("divergence", quantiles)
    rows = [
        _AggregateRow(
            name="sum",
            label="sum X^2/dof",
            observed=observed.sum_chi_square_per_dof,
            null_median=sum_median,
            null_threshold=sum_threshold,
            p_value=analysis.verdict.p_sum,
            decimals=3,
            alpha=alpha,
        ),
        _AggregateRow(
            name="max",
            label="max X^2/dof",
            observed=observed.max_chi_square_per_dof,
            null_median=float(max_median),
            null_threshold=float(max_threshold),
            p_value=analysis.verdict.p_max,
            decimals=3,
            alpha=alpha,
        ),
        _AggregateRow(
            name="mean_tvd",
            label="mean TVD",
            observed=observed.mean_tvd,
            null_median=float(mean_median),
            null_threshold=float(mean_threshold),
            p_value=null.p_value("mean_tvd", observed.mean_tvd),
            decimals=4,
            alpha=None,
        ),
        _AggregateRow(
            name="max_tvd",
            label="max TVD",
            observed=observed.max_tvd,
            null_median=float(tvd_median),
            null_threshold=float(tvd_threshold),
            p_value=null.p_value("max_tvd", observed.max_tvd),
            decimals=4,
            alpha=None,
        ),
        _AggregateRow(
            name="divergence",
            label="divergence",
            observed=observed.divergence,
            null_median=float(div_median),
            null_threshold=float(div_threshold),
            p_value=null.p_value("divergence", observed.divergence),
            decimals=4,
            alpha=None,
        ),
    ]
    if observed.pruned is not None:
        pruned_median, pruned_threshold = null.quantiles("pruned", quantiles)
        rows.append(
            _AggregateRow(
                name="pruned",
                label="pruned prefixes",
                observed=float(observed.pruned),
                null_median=float(pruned_median),
                null_threshold=float(pruned_threshold),
                p_value=null.p_value("pruned", float(observed.pruned)),
                decimals=0,
                alpha=None,
            )
        )
    if null.max_depth is not None:
        # Shallow is the extreme direction here, so the table's upper null
        # quantile does not apply and the p-value is lower-tailed.
        (depth_median,) = null.quantiles("max_depth", (_NULL_MEDIAN,))
        rows.append(
            _AggregateRow(
                name="max_depth",
                label="max depth",
                observed=float(observed.max_depth),
                null_median=float(depth_median),
                null_threshold=None,
                p_value=null.p_value("max_depth", float(observed.max_depth)),
                decimals=0,
                alpha=None,
            )
        )
    if null.mean_depth is not None:
        (mean_depth_median,) = null.quantiles("mean_depth", (_NULL_MEDIAN,))
        rows.append(
            _AggregateRow(
                name="mean_depth",
                label="mean depth",
                observed=observed.mean_depth,
                null_median=float(mean_depth_median),
                null_threshold=None,
                p_value=null.p_value("mean_depth", observed.mean_depth),
                decimals=3,
                alpha=None,
            )
        )
    return rows


def _parameters_json(parameters: RunParameters) -> dict[str, object]:
    """Converts the run parameters to plain types."""
    return {
        "baseline_source": parameters.baseline_source,
        "test_source": parameters.test_source,
        "baseline_header": _header_json(parameters.baseline_header),
        "test_header": _header_json(parameters.test_header),
        "endpoint": parameters.endpoint,
        "request_params": dict(parameters.request_params),
        "baseline_model": parameters.baseline_model,
        "test_model": parameters.test_model,
        "num_samples": parameters.num_samples,
        "min_prefix_samples": parameters.min_prefix_samples,
        "baseline_max_tokens": parameters.baseline_max_tokens,
        "test_max_tokens": parameters.test_max_tokens,
        "depth_limit": parameters.depth_limit,
        "seed": {
            "baseline": parameters.seed[0],
            "test": parameters.seed[1],
        },
        "num_permutations": parameters.num_permutations,
        "false_alarm_rate": parameters.false_alarm_rate,
        "alpha": parameters.alpha,
        "min_token_samples": parameters.min_token_samples,
        "effect_min_per_server": parameters.effect_min_per_server,
        "max_concurrency": {
            "baseline": parameters.max_concurrency[0],
            "test": parameters.max_concurrency[1],
        },
        "prompts": [
            {
                "index": prompt.index,
                "kind": prompt.kind,
                "description": prompt.description,
                "num_prompt_tokens": prompt.num_prompt_tokens,
            }
            for prompt in parameters.prompts
        ],
    }


def _token_budget_json(budget: TokenBudget) -> dict[str, object]:
    """Converts the token-budget calibration verdict to plain types."""
    return {
        "cap": budget.cap,
        "deepest_tested": budget.deepest_tested,
        "tokens_generated": budget.tokens_generated,
        "tokens_beyond": budget.tokens_beyond,
        "wasted_fraction": budget.wasted_fraction,
        "frontier_children": budget.frontier_children,
        "candidates_at_cap": budget.candidates_at_cap,
        "tested_at_cap": budget.tested_at_cap,
        "testable_rate": budget.testable_rate,
        "estimated_testable": budget.estimated_testable,
        "verdict": budget.verdict,
        "line": budget.line,
    }


def _control_json(entry: ControlEntry, row: _ControlRow) -> dict[str, object]:
    """Converts one control outcome to plain types."""
    stats = entry.result.stats
    return {
        "prompt_index": entry.prompt_index,
        "passed": entry.result.passed,
        "degenerate": entry.result.degenerate,
        "chi_square_per_dof": row.chi_square_per_dof,
        "tvd": row.tvd,
        "null_median": row.null_median,
        "null_threshold": row.null_threshold,
        "p_value": row.p_value,
        "result": row.result,
        "message": entry.result.message,
        "node": None if stats is None else _node_json(stats),
    }


def _analysis_json(
    analysis: AnalysisResult, *, alpha: float
) -> dict[str, object]:
    """Converts the node test, its aggregates and its verdict to plain types."""
    observed = analysis.observed
    permutations = analysis.null.num_permutations
    return {
        "num_nodes": observed.num_nodes,
        "num_permutations": permutations,
        "min_p_value": 1.0 / (permutations + 1.0),
        "observed": {
            "sum_chi_square": observed.sum_chi_square,
            "sum_dof": observed.sum_dof,
            "sum_chi_square_per_dof": observed.sum_chi_square_per_dof,
            "max_chi_square_per_dof": observed.max_chi_square_per_dof,
            "max_node_index": observed.max_node_index,
            "max_tvd": observed.max_tvd,
            "max_tvd_node_index": observed.max_tvd_node_index,
            "mean_tvd": observed.mean_tvd,
            "divergence": observed.divergence,
            "pruned_prefixes": observed.pruned,
            "max_depth": observed.max_depth,
            "mean_depth": observed.mean_depth,
            "forced_nodes": observed.forced_nodes,
        },
        "aggregates": {
            row.name: {
                "label": row.label,
                "observed": row.observed,
                "null_median": row.null_median,
                "null_threshold": row.null_threshold,
                "p_value": row.p_value,
                "threshold": row.alpha,
                "result": row.result,
            }
            for row in _aggregate_rows(analysis, alpha=alpha)
        },
        "verdict": {
            "passed": analysis.verdict.passed,
            "p_sum": analysis.verdict.p_sum,
            "p_max": analysis.verdict.p_max,
            "reasons": list(analysis.verdict.reasons),
        },
        "node_stats": [_node_json(stats) for stats in analysis.node_stats],
    }


def _node_json(stats: NodeStats) -> dict[str, object]:
    """Converts one node's statistics to plain types."""
    bucketed = stats.bucketed
    baseline_n, test_n = stats.n_per_group
    return {
        "prompt_index": stats.prompt_index,
        "depth": stats.depth,
        "prefix": list(stats.prefix),
        "pooled_count": stats.pooled_count,
        "n_baseline": baseline_n,
        "n_test": test_n,
        "chi_square": stats.chi_square,
        "dof": stats.dof,
        "chi_square_per_dof": stats.chi_square_per_dof,
        "tvd": stats.tvd,
        "top_token_probability": stats.node.top_token_probability(),
        "entropy_bits": stats.node.entropy_bits(),
        "labels": list(bucketed.labels),
        "baseline_counts": _int_list(bucketed.counts[0]),
        "test_counts": _int_list(bucketed.counts[1]),
        "contributions": _float_list(stats.contributions),
        "test_excess": _float_list(stats.test_excess),
        "tail_tokens": list(bucketed.tail_tokens),
        "tail_folded_into": bucketed.tail_folded_into,
    }


def _int_list(values: npt.NDArray[np.int64]) -> list[int]:
    """Converts an integer array to a list of Python ints."""
    return [int(value) for value in values]


def _float_list(values: npt.NDArray[np.float64]) -> list[float]:
    """Converts a float array to a list of Python floats."""
    return [float(value) for value in values]


def _mean(values: Iterable[float]) -> float:
    """Averages an iterable of floats, returning ``0.0`` when it is empty."""
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _fixed(value: float, decimals: int) -> str:
    """Formats a float with a fixed number of decimals."""
    return f"{value:.{decimals}f}"


def _header_json(header: SamplesHeader | None) -> dict[str, object] | None:
    """Converts a samples file's header to plain types."""
    if header is None:
        return None
    record: dict[str, object] = dict(asdict(header))
    record["prompts"] = [list(prompt) for prompt in header.prompts]
    return record


def _prefix_text(prefix: Sequence[int]) -> str:
    """Renders a node's prefix ids, naming the empty prefix explicitly."""
    return ", ".join(str(token) for token in prefix) if prefix else "(empty)"


def _wrap(text: str, *, indent: str = "  ") -> str:
    """Indents a message, leaving it on one line for the terminal to wrap."""
    return indent + text


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    indent: str = "  ",
    min_widths: Sequence[int] = (),
    footer: Sequence[str] | None = None,
) -> list[str]:
    """Lays out a header, a rule and one line per row, first column left.

    ``min_widths`` pins leading columns so sibling tables can line up, and
    ``footer`` is a totals row set off from the others by a second rule.
    """
    widths = [len(header) for header in headers]
    for index, width in enumerate(min_widths):
        widths[index] = max(widths[index], width)
    for row in (*rows, *([] if footer is None else [footer])):
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    rule = indent + "  ".join("-" * width for width in widths)
    lines = [indent + _table_row(headers, widths), rule]
    lines.extend(indent + _table_row(row, widths) for row in rows)
    if footer is not None:
        lines.extend([rule, indent + _table_row(footer, widths)])
    return lines


def _table_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    """Pads one row's cells, left-aligning the first column."""
    padded = [
        cell.ljust(width) if index == 0 else cell.rjust(width)
        for index, (cell, width) in enumerate(zip(cells, widths, strict=True))
    ]
    return "  ".join(padded).rstrip()
