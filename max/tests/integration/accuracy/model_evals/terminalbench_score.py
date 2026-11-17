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
"""Scores TerminalBench (Harbor + opencode) runs and gates a candidate on agentic cost.

Three subcommands:

- ``score``: reads a Harbor jobs directory and writes ``results.jsonl`` (one row
  per trial) and ``score.json`` (pass rate plus agentic-cost metrics).
- ``missing``: prints ``<task> <count>`` for every task that has fewer than
  ``k`` valid trials, so the runner can re-run only those.
- ``compare``: compares a candidate's ``results.jsonl`` with a baseline's and
  prints a PASS/FAIL verdict (exit 0/1). It gives no verdict (exit 2) when
  fewer than ``--min-tasks`` tasks pair up, or when a task has valid trials in
  one arm only and is not named with ``--exclude``: a candidate that lost the
  tasks it regressed on must not pass on the rest.

Pass rate alone does not catch the regressions this gate exists for. On the
DeepSeek-V4.1-Flash-1017 incident image, pass rate moved by less than its noise
while generated tokens per task rose ~30%, turns ~20%, and responses at the
output cap tripled. So ``compare`` fails a candidate when, paired by task and
one-sided (regressions only, Holm-corrected), any of these rises: generated
tokens, turns, total input tokens (by at least ``--min-ratio`` at the median
task), or the share of trials with a capped response, unclosed reasoning, or
tool-call markup leaked into text. Pass rate fails it only on a significant
drop of at least ``--max-pass-drop``.

A trial is valid only when the verifier produced a reward, the agent reached
the model (a finished step, or any model output in the log), and its last step
ended in ``stop`` or ``length`` (or the agent timed out). Harbor's exception
type is not used: opencode exits non-zero after a normal stop in most trials,
and the verifier still runs. Invalid trials count against neither arm, and the
runner re-runs them. Some of them are engine failures (a dropped connection, a
response that ended without ``stop``), but host-side failures inside the task
containers look the same, so they do not gate; ``score.json`` and the
``compare`` table report each arm's invalid share for a human to read.

Input tokens are counted as uncached + cache read + cache write. Engines differ
in whether they report cached tokens, so the uncached count alone is not
comparable across engines.

Stdlib-only on purpose: it runs on a bare CPU runner without bazel.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import TypedDict

DEFAULT_OUTPUT_CAP = 32768
EXCESS_OVER_K = "excess_over_k"
# Up to this many nonzero differences the Wilcoxon p-value is exact; the
# normal approximation is off by ~2x in the tail at n=4 (p 0.036 vs 1/16).
WILCOXON_EXACT_MAX_N = 50
# The ratio effect round-trips through log/exp, so 11/10 can land a few ulps
# under 1.10.
RATIO_TOLERANCE = 1e-9
SENTENCE_REPEATS = 15
TOOL_CALL_REPEATS = 8
# DeepSeek DSML and generic tool-call markup that should have been parsed into
# a tool call but reached the client as text. DeepSeek's markup opens with a
# fullwidth vertical line (U+FF5C), written as an escape to stay unambiguous.
TOOL_MARKUP = re.compile(
    r"DSML|<\s*/?\s*(tool_calls|function_calls|invoke)\b|tool▁call|<\uff5c"
)

CONTINUOUS = ("gen", "turns", "in_total")
RATES = ("capped", "unclosed", "markup")


@dataclass
class Trial:
    trial: str
    task: str
    started: str
    reward: float | None
    exception: str
    reached_model: bool
    valid: bool
    turns: int
    in_uncached: int
    in_total: int
    out: int
    reasoning: int
    gen: int
    max_step_gen: int
    capped_steps: int
    unclosed_steps: int
    markup_in_text: int
    loops: int
    tool_calls: int
    tool_errors: int

    @property
    def passed(self) -> float:
        return float(self.reward or 0.0)

    def flag(self, name: str) -> bool:
        return {
            "capped": self.capped_steps > 0,
            "unclosed": self.unclosed_steps > 0,
            "markup": self.markup_in_text > 0,
        }[name]


def _prose_sentences(text: str) -> list[str]:
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = s.strip()
        if len(s) >= 20 and len(re.findall(r"[A-Za-z]{3,}", s)) >= 4:
            out.append(s)
    return out


@dataclass
class _Step:
    input: int
    output: int
    reasoning: int
    cache: int
    reason: str | None

    @property
    def gen(self) -> int:
        return self.output + self.reasoning


def parse_trial(trial_dir: Path, output_cap: int = DEFAULT_OUTPUT_CAP) -> Trial:
    """Parses one Harbor trial directory (``result.json`` + opencode's event log)."""
    result = json.loads((trial_dir / "result.json").read_text())
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get(
        "reward"
    )
    exception = (result.get("exception_info") or {}).get("exception_type") or ""
    steps: list[_Step] = []
    tools: list[tuple[str, str, str]] = []
    loops = markup = 0
    model_output = False
    log = trial_dir / "agent" / "opencode.txt"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = event.get("part", {})
            kind = event.get("type")
            if kind in ("text", "reasoning", "tool_use"):
                model_output = True
            if kind == "tool_use":
                state = part.get("state", {})
                tools.append(
                    (
                        part.get("tool"),
                        json.dumps(state.get("input"), sort_keys=True),
                        state.get("status"),
                    )
                )
            elif kind in ("text", "reasoning"):
                text = part.get("text", "")
                common = Counter(_prose_sentences(text)).most_common(1)
                if common and common[0][1] >= SENTENCE_REPEATS:
                    loops += 1
                if TOOL_MARKUP.search(text):
                    markup += 1
            elif kind == "step_finish":
                tokens = part.get("tokens", {})
                cache = tokens.get("cache", {})
                steps.append(
                    _Step(
                        input=int(tokens.get("input", 0)),
                        output=int(tokens.get("output", 0)),
                        reasoning=int(tokens.get("reasoning", 0)),
                        cache=int(cache.get("read", 0))
                        + int(cache.get("write", 0)),
                        reason=part.get("reason"),
                    )
                )
    run = best = 1
    for a, b in pairwise(tools):
        run = run + 1 if a[:2] == b[:2] else 1
        best = max(best, run)
    if tools and best >= TOOL_CALL_REPEATS:
        loops += 1
    last_reason = steps[-1].reason if steps else None
    # A runaway first response that hits the agent timeout streams reasoning
    # but never finishes a step. It is a real (bad) outcome, so it must stay
    # valid rather than be re-run until a luckier trial replaces it. Its
    # unfinished step reports no tokens, so it counts as zero.
    reached = bool(steps) or model_output
    return Trial(
        trial=trial_dir.name,
        task=trial_dir.name.rsplit("__", 1)[0],
        started=result.get("started_at") or "",
        reward=reward,
        exception=exception,
        reached_model=reached,
        valid=reward is not None
        and reached
        and (
            last_reason in ("stop", "length")
            or exception == "AgentTimeoutError"
        ),
        turns=len(steps),
        in_uncached=sum(s.input for s in steps),
        in_total=sum(s.input + s.cache for s in steps),
        out=sum(s.output for s in steps),
        reasoning=sum(s.reasoning for s in steps),
        gen=sum(s.gen for s in steps),
        max_step_gen=max((s.gen for s in steps), default=0),
        capped_steps=sum(
            1 for s in steps if s.reason == "length" or s.gen >= output_cap - 8
        ),
        unclosed_steps=sum(
            1
            for s in steps
            if s.reason == "length" and s.output == 0 and s.reasoning > 0
        ),
        markup_in_text=markup,
        loops=loops,
        tool_calls=len(tools),
        tool_errors=sum(1 for t in tools if t[2] == "error"),
    )


def load_trials(
    jobs_dir: Path, k: int, output_cap: int = DEFAULT_OUTPUT_CAP
) -> list[Trial]:
    """Loads every trial under ``jobs_dir`` (all jobs, including re-run passes).

    Re-run passes can overshoot ``k``; only the ``k`` earliest-started valid
    trials per task stay valid, so the kept set does not depend on outcomes.
    """
    trials = [
        parse_trial(p.parent, output_cap)
        for p in sorted(jobs_dir.glob("*/*/result.json"))
    ]
    seen: Counter[str] = Counter()
    for t in sorted(trials, key=lambda t: t.started):
        if t.valid:
            seen[t.task] += 1
            if seen[t.task] > k:
                t.valid = False
                t.exception = (t.exception + "+" + EXCESS_OVER_K).lstrip("+")
    return trials


def missing(
    trials: Iterable[Trial], tasks: Iterable[str], k: int
) -> dict[str, int]:
    """Returns task -> number of valid trials still needed to reach ``k``."""
    have = Counter(t.task for t in trials if t.valid)
    return {t: k - have[t] for t in sorted(set(tasks)) if have[t] < k}


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def by_task(trials: Iterable[Trial]) -> dict[str, list[Trial]]:
    out: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        if t.valid:
            out[t.task].append(t)
    return dict(out)


@dataclass
class ArmHealth:
    """Trials the runner discarded, for a human to read; never gated on."""

    trials_seen: int
    # Trials that failed validity; trials dropped only for exceeding ``k``
    # were fine and are not counted.
    invalid_trials: int
    # The subset that reached the model: the likeliest engine-side failures.
    invalid_reached_model: int

    @property
    def invalid_share(self) -> float | None:
        return (
            self.invalid_trials / self.trials_seen if self.trials_seen else None
        )


def arm_health(
    trials: Iterable[Trial], exclude: Collection[str] = ()
) -> ArmHealth:
    seen = [t for t in trials if t.task not in exclude]
    invalid = [
        t for t in seen if not t.valid and EXCESS_OVER_K not in t.exception
    ]
    return ArmHealth(
        trials_seen=len(seen),
        invalid_trials=len(invalid),
        invalid_reached_model=sum(1 for t in invalid if t.reached_model),
    )


class ScoreSummary(TypedDict):
    score: float | None
    score_stderr: float | None
    tasks: int
    tasks_short_of_k: int
    k: int
    total: int
    trials_seen: int
    invalid_trials: int
    invalid_reached_model: int
    # Same count as ``invalid_trials``; collect_scores.py reads this name.
    errors: int
    p50_gen: float | None
    p90_gen: float | None
    mean_gen: float | None
    p50_turns: float | None
    p90_turns: float | None
    mean_turns: float | None
    p50_in_total: float | None
    p90_in_total: float | None
    mean_in_total: float | None
    p50_out: float | None
    p90_out: float | None
    mean_out: float | None
    p50_reasoning: float | None
    p90_reasoning: float | None
    mean_reasoning: float | None
    # collect_scores.py reads these two names for its token columns.
    mean_output_tokens: float | None
    p50_output_tokens: float | None
    capped_trial_rate: float | None
    unclosed_trial_rate: float | None
    markup_trial_rate: float | None
    loop_trials: int
    tool_error_rate: float | None


def summarize(trials: Sequence[Trial], k: int) -> ScoreSummary:
    """Builds ``score.json``. ``score`` is the mean over tasks of per-task pass rate."""
    valid = [t for t in trials if t.valid]
    tasks = by_task(trials)
    per_task_pass = [
        statistics.mean(t.passed for t in v) for v in tasks.values()
    ]
    health = arm_health(trials)

    def values(name: str) -> list[float]:
        return [float(getattr(t, name)) for t in valid]

    def p50(name: str) -> float | None:
        return _quantile(values(name), 0.5)

    def p90(name: str) -> float | None:
        return _quantile(values(name), 0.9)

    def mean(name: str) -> float | None:
        return statistics.mean(values(name)) if valid else None

    def rate(name: str) -> float | None:
        return sum(t.flag(name) for t in valid) / len(valid) if valid else None

    calls = sum(t.tool_calls for t in valid)
    return ScoreSummary(
        score=statistics.mean(per_task_pass) if per_task_pass else None,
        score_stderr=statistics.stdev(per_task_pass)
        / math.sqrt(len(per_task_pass))
        if len(per_task_pass) > 1
        else None,
        tasks=len(tasks),
        tasks_short_of_k=sum(1 for v in tasks.values() if len(v) < k),
        k=k,
        total=len(valid),
        trials_seen=health.trials_seen,
        invalid_trials=health.invalid_trials,
        invalid_reached_model=health.invalid_reached_model,
        errors=health.invalid_trials,
        p50_gen=p50("gen"),
        p90_gen=p90("gen"),
        mean_gen=mean("gen"),
        p50_turns=p50("turns"),
        p90_turns=p90("turns"),
        mean_turns=mean("turns"),
        p50_in_total=p50("in_total"),
        p90_in_total=p90("in_total"),
        mean_in_total=mean("in_total"),
        p50_out=p50("out"),
        p90_out=p90("out"),
        mean_out=mean("out"),
        p50_reasoning=p50("reasoning"),
        p90_reasoning=p90("reasoning"),
        mean_reasoning=mean("reasoning"),
        mean_output_tokens=mean("gen"),
        p50_output_tokens=p50("gen"),
        capped_trial_rate=rate("capped"),
        unclosed_trial_rate=rate("unclosed"),
        markup_trial_rate=rate("markup"),
        loop_trials=sum(1 for t in valid if t.loops),
        tool_error_rate=sum(t.tool_errors for t in valid) / calls
        if calls
        else None,
    )


def _doubled_signed_ranks(d: Sequence[float]) -> tuple[list[int], int]:
    """Returns twice the midranks of ``|d|`` (integers even under ties) and
    the tie-correction term ``sum(t^3 - t)``."""
    n = len(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks2 = [0] * n
    tie_term = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for m in range(i, j + 1):
            ranks2[order[m]] = i + j + 2
        size = j - i + 1
        tie_term += size**3 - size
        i = j + 1
    return ranks2, tie_term


def _wilcoxon_exact(d: Sequence[float]) -> float:
    ranks2, _ = _doubled_signed_ranks(d)
    observed = sum(r for r, x in zip(ranks2, d, strict=True) if x > 0)
    # Under H0 each rank's sign is a fair coin: count the sign assignments
    # reaching each doubled W+ (midranks keep the ties' null distribution).
    counts = [0] * (sum(ranks2) + 1)
    counts[0] = 1
    reach = 0
    for r in ranks2:
        reach += r
        for s in range(reach, r - 1, -1):
            counts[s] += counts[s - r]
    return sum(counts[observed:]) / 2 ** len(d)


def _wilcoxon_normal(d: Sequence[float]) -> float:
    n = len(d)
    ranks2, tie_term = _doubled_signed_ranks(d)
    w_plus = sum(r for r, x in zip(ranks2, d, strict=True) if x > 0) / 2
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if var <= 0:
        return 1.0
    z = (w_plus - mean - 0.5) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilcoxon_greater(diffs: Sequence[float]) -> float:
    """One-sided Wilcoxon signed-rank p-value for median(diffs) > 0.

    Zero differences are dropped (Wilcoxon's convention), so ``n`` is the
    number of tasks that moved. The p-value is exact up to
    ``WILCOXON_EXACT_MAX_N`` of them; above that it uses the normal
    approximation with tie correction and continuity correction.
    """
    d = [x for x in diffs if x != 0]
    if not d:
        return 1.0
    if len(d) <= WILCOXON_EXACT_MAX_N:
        return _wilcoxon_exact(d)
    return _wilcoxon_normal(d)


def fisher_greater(
    cand_hits: int, cand_n: int, base_hits: int, base_n: int
) -> float:
    """One-sided Fisher exact p-value that the candidate's rate exceeds the baseline's."""
    total_hits = cand_hits + base_hits
    total = cand_n + base_n
    denom = math.comb(total, total_hits)
    return (
        sum(
            math.comb(cand_n, x) * math.comb(base_n, total_hits - x)
            for x in range(cand_hits, min(cand_n, total_hits) + 1)
        )
        / denom
    )


def _log_ratio(a: float, b: float) -> float:
    """Returns log(b / a) for one task's per-trial means.

    A plain ratio keeps small counts honest (10 -> 11 turns is x1.10). A zero
    mean in either arm (every trial ended before finishing a step) has no
    ratio, so those tasks fall back to (b + 1) / (a + 1): no change when both
    are zero, and a finite, sign-correct move otherwise.
    """
    if a > 0 and b > 0:
        return math.log(b / a)
    return math.log((b + 1) / (a + 1))


def split_tasks(
    baseline: Iterable[Trial],
    candidate: Iterable[Trial],
    exclude: Collection[str] = (),
) -> tuple[list[str], list[str], list[str]]:
    """Returns (paired, baseline-only, candidate-only) tasks with valid trials."""
    base = set(by_task(baseline)) - set(exclude)
    cand = set(by_task(candidate)) - set(exclude)
    return sorted(base & cand), sorted(base - cand), sorted(cand - base)


@dataclass
class Check:
    metric: str
    baseline: float | None
    candidate: float | None
    effect: float | None
    p_value: float
    failed: bool = False


def compare(
    baseline: Sequence[Trial],
    candidate: Sequence[Trial],
    alpha: float = 0.05,
    min_ratio: float = 1.10,
    max_pass_drop: float = 0.05,
    exclude: Collection[str] = (),
) -> tuple[bool, list[Check]]:
    """Gates ``candidate`` against ``baseline``. Returns (passed, checks).

    Only tasks with a valid trial in both arms count; ``main`` refuses a
    verdict when any other task is left over. Each arm's per-task value is
    the mean over its valid trials, so the arms may have different ``k``.
    """
    base, cand = by_task(baseline), by_task(candidate)
    tasks, _, _ = split_tasks(baseline, candidate, exclude)
    checks: list[Check] = []
    for name in CONTINUOUS:
        a = [statistics.mean(getattr(t, name) for t in base[x]) for x in tasks]
        b = [statistics.mean(getattr(t, name) for t in cand[x]) for x in tasks]
        logs = [_log_ratio(x, y) for x, y in zip(a, b, strict=True)]
        checks.append(
            Check(
                metric=f"{name} (median per-task ratio)",
                baseline=statistics.median(a) if a else None,
                candidate=statistics.median(b) if b else None,
                effect=math.exp(statistics.median(logs)) if logs else None,
                p_value=wilcoxon_greater(logs),
            )
        )
    for name in RATES:
        bh = sum(t.flag(name) for x in tasks for t in base[x])
        bn = sum(len(base[x]) for x in tasks)
        ch = sum(t.flag(name) for x in tasks for t in cand[x])
        cn = sum(len(cand[x]) for x in tasks)
        checks.append(
            Check(
                metric=f"{name} (share of trials)",
                baseline=bh / bn if bn else None,
                candidate=ch / cn if cn else None,
                effect=(ch / cn - bh / bn) if bn and cn else None,
                p_value=fisher_greater(ch, cn, bh, bn) if bn and cn else 1.0,
            )
        )
    # Holm step-down over the cost checks; continuous ones also need the effect floor.
    ordered = sorted(checks, key=lambda c: c.p_value)
    for i, c in enumerate(ordered):
        if c.p_value > alpha / (len(ordered) - i):
            break
        is_ratio = c.metric.split(" ")[0] in CONTINUOUS
        c.failed = not is_ratio or (
            c.effect is not None and c.effect >= min_ratio - RATIO_TOLERANCE
        )
    a = [statistics.mean(t.passed for t in base[x]) for x in tasks]
    b = [statistics.mean(t.passed for t in cand[x]) for x in tasks]
    drop = statistics.mean(a) - statistics.mean(b) if tasks else 0.0
    pass_check = Check(
        metric="pass rate (per-task mean)",
        baseline=statistics.mean(a) if a else None,
        candidate=statistics.mean(b) if b else None,
        effect=-drop,
        p_value=wilcoxon_greater([x - y for x, y in zip(a, b, strict=True)]),
    )
    pass_check.failed = pass_check.p_value < alpha and drop >= max_pass_drop
    checks.insert(0, pass_check)
    return not any(c.failed for c in checks), checks


def render(
    status: str,
    checks: Sequence[Check],
    tasks: int,
    health: dict[str, ArmHealth],
    reasons: Sequence[str] = (),
) -> str:
    def fmt(v: float | None) -> str:
        if v is None:
            return "n/a"
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3f}"

    lines = [
        f"TerminalBench gate: {status.upper().replace('_', ' ')} ({tasks} paired tasks)",
        *reasons,
        "",
        "| check | baseline | candidate | effect | p (one-sided) | result |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c.metric} | {fmt(c.baseline)} | {fmt(c.candidate)} | "
            f"{fmt(c.effect)} | {c.p_value:.2g} | {'FAIL' if c.failed else 'ok'} |"
        )
    lines.append("")
    lines.append(
        "Invalid trials (re-run by the runner; informational, not gated):"
    )
    for arm, h in health.items():
        share = h.invalid_share
        lines.append(
            f"- {arm}: {h.invalid_trials}/{h.trials_seen}"
            f" ({'n/a' if share is None else f'{share:.1%}'}),"
            f" {h.invalid_reached_model} of them reached the model"
        )
    return "\n".join(lines)


def write_jsonl(path: Path, trials: Iterable[Trial]) -> None:
    with path.open("w") as f:
        for t in trials:
            f.write(json.dumps(asdict(t)) + "\n")


def read_jsonl(path: Path) -> list[Trial]:
    names = {f.name for f in fields(Trial)}
    rows: list[Trial] = []
    for line in path.read_text().splitlines():
        if line.strip():
            data = json.loads(line)
            rows.append(Trial(**{k: v for k, v in data.items() if k in names}))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_score = sub.add_parser("score", help="Score a Harbor jobs directory.")
    p_score.add_argument("--jobs-dir", type=Path, required=True)
    p_score.add_argument("--out-dir", type=Path, required=True)
    p_score.add_argument("-k", type=int, required=True)
    p_score.add_argument("--output-cap", type=int, default=DEFAULT_OUTPUT_CAP)
    p_score.add_argument(
        "--meta",
        type=Path,
        help="JSON file merged into score.json under 'run' (model, image, harness pins).",
    )
    p_miss = sub.add_parser(
        "missing", help="List tasks short of k valid trials."
    )
    p_miss.add_argument("--jobs-dir", type=Path, required=True)
    p_miss.add_argument("--tasks-dir", type=Path, required=True)
    p_miss.add_argument("-k", type=int, required=True)
    p_cmp = sub.add_parser(
        "compare", help="Gate a candidate against a baseline."
    )
    p_cmp.add_argument(
        "--baseline", type=Path, required=True, help="results.jsonl"
    )
    p_cmp.add_argument(
        "--candidate", type=Path, required=True, help="results.jsonl"
    )
    p_cmp.add_argument("--alpha", type=float, default=0.05)
    p_cmp.add_argument("--min-ratio", type=float, default=1.10)
    p_cmp.add_argument("--max-pass-drop", type=float, default=0.05)
    p_cmp.add_argument(
        "--min-tasks",
        type=int,
        default=80,
        help="Paired tasks needed for a verdict; fewer exits 2 (a subset is not a measurement).",
    )
    p_cmp.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TASK",
        help=(
            "Drop TASK from both arms (repeatable). Every task with valid trials"
            " in one arm only must be named here, or there is no verdict; named"
            " exclusions keep what the gate skipped visible in its output."
        ),
    )
    p_cmp.add_argument(
        "--out", type=Path, help="Write the verdict as JSON here."
    )
    args = parser.parse_args(argv)

    if args.cmd == "score":
        trials = load_trials(args.jobs_dir, args.k, args.output_cap)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out_dir / "results.jsonl", trials)
        summary = summarize(trials, args.k)
        score_json: dict[str, object] = dict(summary)
        if args.meta:
            score_json["run"] = json.loads(args.meta.read_text())
        (args.out_dir / "score.json").write_text(
            json.dumps(score_json, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "score": summary["score"],
                    "tasks": summary["tasks"],
                    "total": summary["total"],
                    "invalid_trials": summary["invalid_trials"],
                    "invalid_reached_model": summary["invalid_reached_model"],
                }
            )
        )
        return 0
    if args.cmd == "missing":
        tasks = [p.parent.name for p in args.tasks_dir.glob("*/task.toml")]
        trials = load_trials(args.jobs_dir, args.k)
        for task, count in missing(trials, tasks, args.k).items():
            print(task, count)
        return 0
    baseline, candidate = read_jsonl(args.baseline), read_jsonl(args.candidate)
    exclude = set(args.exclude)
    paired, only_base, only_cand = split_tasks(baseline, candidate, exclude)
    passed, checks = compare(
        baseline,
        candidate,
        args.alpha,
        args.min_ratio,
        args.max_pass_drop,
        exclude,
    )
    health = {
        "baseline": arm_health(baseline, exclude),
        "candidate": arm_health(candidate, exclude),
    }
    reasons = []
    if only_base:
        reasons.append(
            "NO VERDICT: tasks with valid trials only in the baseline"
            f" (rerun them or --exclude them): {' '.join(only_base)}"
        )
    if only_cand:
        reasons.append(
            "NO VERDICT: tasks with valid trials only in the candidate"
            f" (rerun them or --exclude them): {' '.join(only_cand)}"
        )
    if len(paired) < args.min_tasks:
        reasons.append(
            f"NO VERDICT: {len(paired)} paired tasks < --min-tasks {args.min_tasks}"
        )
    status = "no_verdict" if reasons else "pass" if passed else "fail"
    print(render(status, checks, len(paired), health, reasons))
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "status": status,
                    "passed": None if reasons else passed,
                    "reasons": reasons,
                    "paired_tasks": len(paired),
                    "baseline_only": only_base,
                    "candidate_only": only_cand,
                    "excluded": sorted(exclude),
                    "arms": {k: asdict(v) for k, v in health.items()},
                    "checks": [asdict(c) for c in checks],
                },
                indent=2,
            )
            + "\n"
        )
    return {"pass": 0, "fail": 1, "no_verdict": 2}[status]


if __name__ == "__main__":
    sys.exit(main())
