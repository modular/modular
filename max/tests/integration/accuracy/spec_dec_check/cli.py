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
"""Command-line orchestration for SpecDecCheck.

Each of ``--baseline`` and ``--test`` names one source of samples: a file an
earlier run wrote, or a running server to draw from now, which is what lets a
baseline that will not change be collected once and reused. Either switch may
be omitted; the run then collects that one source, saves it, and runs only the
per-server stability check. The two sources are reconciled on one thing, that
both sides answered the same prompt ids. Two live servers are collected
concurrently but their prompts sequentially, since two prompts in flight at
once would only halve each server's effective concurrency.

``--run-suite`` turns the invocation into many runs, one per row of a CSV
whose columns are option names, and ``--repeats`` runs each of them several
times; the command line alone is the one-row case. Runs that ask a server the
same questions under the same settings can share one draw, under
``--suite-reuse`` across rows and ``--repeat-reuse`` across repeats.

The exit code carries the finding: pass means the two servers are
indistinguishable, fail means a gating aggregate rejected the null, and abort
means the run could not answer the question it was asked. An abort is
deliberately not a fail: the comparison cannot be trusted to begin with, which
is a different finding from a speculative decoding bug.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .client import (
    CHAT_ENDPOINT,
    COMPLETIONS_ENDPOINT,
    ENDPOINTS,
    RESERVED_REQUEST_KEYS,
    ChatMessages,
    _is_token_ids,
    collect_samples,
    discover_model_name,
    make_client,
)
from .report import (
    EXIT_ABORTED,
    EXIT_FAIL,
    EXIT_PASS,
    SUMMARY_CSV,
    TOKEN_BUDGET_PREFIX,
    CollectionSummary,
    ControlEntry,
    PromptInfo,
    PruningSummary,
    RunParameters,
    RunReport,
    StabilityResult,
    TokenBudget,
    render_control_prefill,
    render_text,
    stability_from_analysis,
    summarize_file,
    summarize_live,
    summary_csv,
    to_json_dict,
    write_csv_tables,
    write_summary_csv,
)
from .samples import (
    Sample,
    SamplesHeader,
    load_samples,
    make_header,
    save_samples,
)
from .stats import (
    AnalysisResult,
    ControlResult,
    RelabelingNull,
    analyze,
    control_check,
)
from .tree import (
    Node,
    bucketize,
    bucketize_all,
    build_tree,
    control_nodes,
    count_prefixes,
    relabeling_null,
    select_nodes,
)

logger = logging.getLogger(__name__)

SERVERS = ("baseline", "test")
"""The two server names, baseline first, as the whole pipeline orders them."""

HALVES = ("first", "second")
"""The group names the stability check gives one server's two halves."""

DEFAULT_NUM_SAMPLES = 10000
"""``--num-samples`` when no samples file supplies one."""

DEFAULT_MAX_TOKENS = 10
"""``--max-tokens`` when no samples file supplies one."""

_PERMUTATION_SEED = 0
"""Fixed seed of the permutation generator, so the analysis of a given set of
samples is reproducible."""

_PERMUTATIONS_PER_ALPHA = 50
"""Permutations per unit of per-gate alpha, which keeps the Monte Carlo error
of a p-value small next to the threshold it is compared against."""

_DESCRIPTION_CHARS = 72

SUITE_OPTIONS = (
    "baseline",
    "test",
    "endpoint",
    "num-samples",
    "min-prefix-samples",
    "max-tokens",
    "seed",
    "max-concurrency",
    "false-alarm-rate",
    "min-token-samples",
    "request-params",
)
"""The options a suite row may set: those that describe one run and take a
value. ``--prompt`` appends rather than overrides, and the rest describe the
invocation as a whole."""


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Shows defaults, except for the options that have no fixed one.

    ``--num-samples`` and ``--max-tokens`` fall back to whatever a samples
    file recorded, so their parser default is ``None`` and printing it would
    be worse than printing nothing.
    """

    def _get_help_string(self, action: argparse.Action) -> str | None:
        """Returns the help text, with the default appended if there is one."""
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parses and validates the command line.

    With ``--run-suite``, every row is parsed and validated too, as the
    command line with that row's cells appended, and kept on ``suite_rows``.

    Args:
        argv: The arguments after the program name.

    Returns:
        The parsed arguments, with ``prompts`` always a list.
    """
    return _parse(argv, expand_suite=True)


def _parse(argv: Sequence[str], *, expand_suite: bool) -> argparse.Namespace:
    """Parses one command line, expanding a suite only at the top level."""
    parser = _build_parser()
    args = parser.parse_args(list(argv))
    if args.prompts is None:
        args.prompts = []
    _validate_suite_options(parser, args)
    if expand_suite and args.run_suite is not None:
        args.suite_rows = _read_suite(parser, args.run_suite, argv)
    else:
        args.suite_rows = []
        _validate_run(parser, args)
    return args


def _build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    parser = argparse.ArgumentParser(
        prog="spec_dec_check",
        description=(
            "Checks that speculative decoding preserves the target model's\n"
            "token distribution, by comparing many seeded completions from a\n"
            "server with drafting enabled against a server without it."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        metavar="SOURCE",
        help=(
            "Where the baseline samples come from: the root URL of a running"
            " server without speculative decoding, or a samples file an"
            " earlier run saved; an existing path is read as a file, anything"
            " else is a URL. Either --baseline or --test may be omitted; the"
            " run then collects that one source, saves it, and runs only the"
            " per-server stability check."
        ),
    )
    parser.add_argument(
        "--test",
        metavar="SOURCE",
        help=(
            "Where the test samples come from, in the same two forms as"
            " --baseline; the test server is the one with speculative"
            " decoding enabled."
        ),
    )
    parser.add_argument(
        "--endpoint",
        choices=ENDPOINTS,
        default=CHAT_ENDPOINT,
        help=(
            "Endpoint text prompts are sent to. Under the chat endpoint a"
            " text prompt becomes one user turn and the server applies the"
            " model's chat template. Ids from a samples file always use"
            " completions."
        ),
    )
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        metavar="TEXT",
        help=(
            "A prompt: text, a JSON list of chat messages, or @PATH to a"
            " file holding either. Repeatable. Message lists need the chat"
            " endpoint."
        ),
    )
    parser.add_argument(
        "--request-params",
        metavar="JSON",
        help=(
            "Extra fields for every request body, as a JSON object or @PATH"
            " to a file holding one, for example"
            ' \'{"top_k": 50, "temperature": 0.8}\'. By default only the'
            " fields the tool needs are sent, so sampling follows the"
            " model's defaults. The tool's own fields (model, prompt,"
            " messages, max_tokens, seed, return_token_ids, stream) cannot"
            " be set here."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        metavar="N",
        help=(
            "Completions to draw per live server and prompt, subject to the"
            " input rules below."
        ),
    )
    parser.add_argument(
        "--min-prefix-samples",
        type=int,
        default=100,
        metavar="M",
        help=(
            "Samples from each server that must reach a prefix for it to be"
            " tested."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=(
            "Output token budget per completion, which caps the depths the"
            " run can test; when the two sources' budgets differ, the"
            " analysis keeps to the smaller."
        ),
    )
    parser.add_argument(
        "--seed",
        type=_int_pair,
        default=(0, 1),
        metavar="N[:M]",
        help=(
            "Seed base per server, or a baseline:test pair. Each request's"
            " seed is a hash of its server's base and its index, so distinct"
            " bases give the two servers unrelated draws."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=_int_pair,
        default=(32, 32),
        metavar="N[:M]",
        help=(
            "Maximum requests in flight at once per server, or a"
            " baseline:test pair when the two should differ."
        ),
    )
    parser.add_argument(
        "--false-alarm-rate",
        type=float,
        default=0.002,
        help=(
            "Upper bound on how often a run fails when the two servers are"
            " actually identical, shared by the two gates."
        ),
    )
    parser.add_argument(
        "--min-token-samples",
        type=int,
        default=10,
        help=(
            "Pooled samples a token needs at a prefix to be tested on its"
            " own; rarer tokens pool into one tail bucket."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("auto"),
        metavar="DIR",
        help=(
            "Write report.txt, report.json, the CSV tables and, for each"
            " source that was a live server, baseline.jsonl or test.jsonl"
            " into this directory. 'auto' names it SDC-YYYY.MM.DD-HH.MM.SS"
            " from the local start time, under the working directory; 'None'"
            " writes nothing. When --repeats or --run-suite makes more than"
            " one run, each run's files go into run-NNN beneath it and"
            " summary.csv gets one row per run."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help="Run the whole job this many times in a row, as literal copies.",
    )
    parser.add_argument(
        "--repeat-reuse",
        choices=("None", "baseline", "test"),
        default="None",
        help=(
            "Which server's samples the repeats after the first reuse instead"
            " of collecting again. 'None' collects both every time, so the"
            " repeats measure fresh collection noise on both sides."
        ),
    )
    parser.add_argument(
        "--run-suite",
        type=Path,
        metavar="CSV",
        help=(
            "Run one job per row of this CSV. Its columns are option names"
            " without the leading dashes, its cells use each option's own"
            " syntax, and an empty cell falls back to the command line."
            " Options that describe the whole invocation, and --prompt,"
            " cannot be columns."
        ),
    )
    parser.add_argument(
        "--suite-reuse",
        choices=("None", "baseline", "test", "both"),
        default="both",
        help=(
            "Which of a run's servers may take an earlier suite row's"
            " samples instead of collecting, when that row asked the same"
            " server the same questions under the same settings."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG instead of INFO.",
    )

    return parser


def _read_suite(
    parser: argparse.ArgumentParser, path: Path, argv: Sequence[str]
) -> list[argparse.Namespace]:
    """Parses each suite row as the command line plus that row's cells.

    Appending the cells lets argparse's last-occurrence rule make the row
    override the command line, and puts every row through the validation a
    single run gets, before anything is collected.
    """
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        for column in columns:
            if column not in SUITE_OPTIONS:
                parser.error(
                    f"{path}: column {column!r} is not an option a suite row"
                    " may set; columns are option names without the leading"
                    f" dashes, one of: {', '.join(SUITE_OPTIONS)}"
                )
        rows: list[argparse.Namespace] = []
        for line, record in enumerate(reader, 2):
            overrides = [
                token
                for column in columns
                if (cell := (record.get(column) or "").strip())
                for token in (f"--{column}", cell)
            ]
            try:
                rows.append(_parse([*argv, *overrides], expand_suite=False))
            except SystemExit:
                print(
                    f"{parser.prog}: the error above is in line {line} of"
                    f" {path}",
                    file=sys.stderr,
                )
                raise
    if not rows:
        parser.error(f"{path}: the suite has no rows")
    return rows


def _num_permutations(alpha: float) -> int:
    """Derives the permutation count from the per-gate alpha."""
    return math.ceil(_PERMUTATIONS_PER_ALPHA / alpha)


def _parse_prompt(raw: str) -> str | ChatMessages:
    """Reads a prompt argument: text, a JSON message list, or ``@PATH``.

    Only a JSON list whose every element is an object with string ``role``
    and ``content`` counts as a conversation; anything else stays text.

    Raises:
        ValueError: If ``@PATH`` cannot be read or a JSON list is not a
            conversation.
    """
    if raw.startswith("@"):
        try:
            raw = Path(raw[1:]).read_text()
        except OSError as exc:
            raise ValueError(
                f"cannot read prompt file {raw[1:]!r}: {exc}"
            ) from exc
    stripped = raw.strip()
    if not stripped.startswith("["):
        return raw
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        return raw
    if not isinstance(loaded, list):
        return raw
    messages: list[dict[str, str]] = []
    for element in loaded:
        if not (
            isinstance(element, dict)
            and isinstance(element.get("role"), str)
            and isinstance(element.get("content"), str)
        ):
            raise ValueError(
                "a JSON prompt must be a list of objects with string role"
                " and content"
            )
        messages.append(
            {"role": element["role"], "content": element["content"]}
        )
    if not messages:
        raise ValueError("a JSON prompt must hold at least one message")
    return messages


def _parse_request_params(raw: str | None) -> dict[str, object]:
    """Reads ``--request-params``: a JSON object, or ``@PATH`` to one.

    Raises:
        ValueError: If the file cannot be read, the text is not a JSON
            object, or it sets a field the tool owns.
    """
    if raw is None:
        return {}
    text = raw
    if raw.startswith("@"):
        try:
            text = Path(raw[1:]).read_text()
        except OSError as exc:
            raise ValueError(f"cannot read {raw[1:]!r}: {exc}") from exc
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object, got {type(loaded).__name__}")
    reserved = sorted(RESERVED_REQUEST_KEYS & loaded.keys())
    if reserved:
        raise ValueError(
            f"{', '.join(reserved)}: the tool sets these itself; max_tokens"
            " and seed have their own options"
        )
    return loaded


def _int_pair(text: str) -> tuple[int, int]:
    """Parses ``N`` or ``N:M`` into a baseline and test pair of integers."""
    parts = text.split(":")
    if len(parts) > 2:
        raise argparse.ArgumentTypeError(f"expected N or N:M, got {text!r}")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected integers, got {text!r}"
        ) from exc
    return (values[0], values[-1])


def _validate_suite_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Checks the options that describe the invocation rather than a run."""
    if args.repeats < 1:
        parser.error(f"--repeats must be at least 1, got {args.repeats}")
    if args.repeat_reuse == "None":
        args.repeat_reuse = None
    if args.suite_reuse == "None":
        args.suite_reuse = None
    if args.run_suite is not None and not args.run_suite.is_file():
        parser.error(f"--run-suite {args.run_suite}: no such file")
    if args.outdir == Path("auto"):
        args.outdir = Path(f"SDC-{datetime.now():%Y.%m.%d-%H.%M.%S}")
    elif args.outdir == Path("None"):
        args.outdir = None


def _validate_run(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Rejects argument combinations and ranges one run cannot use."""
    given_sources = [
        (switch, value)
        for switch, value in (
            ("--baseline", args.baseline),
            ("--test", args.test),
        )
        if value is not None
    ]
    if not given_sources:
        parser.error("at least one of --baseline and --test is required")
    file_switches = [
        switch for switch, value in given_sources if Path(value).is_file()
    ]
    parsed: list[str | ChatMessages] = []
    for position, raw in enumerate(args.prompts):
        try:
            prompt = _parse_prompt(raw)
        except ValueError as exc:
            parser.error(f"--prompt {position + 1}: {exc}")
        if isinstance(prompt, str) and not prompt.strip():
            parser.error(
                f"All prompts must be non-empty (prompt {position + 1}:"
                f" '{prompt.strip()}')"
            )
        if not isinstance(prompt, str) and args.endpoint != CHAT_ENDPOINT:
            parser.error(
                f"--prompt {position + 1} is a chat conversation, which"
                f" needs --endpoint {CHAT_ENDPOINT}"
            )
        parsed.append(prompt)
    args.prompts = parsed
    if not file_switches and not args.prompts:
        parser.error(
            "at least one --prompt is required when no source is a samples file"
        )
    if len(file_switches) == len(given_sources):
        given = [
            switch
            for switch, passed in (
                ("--prompt", bool(args.prompts)),
                ("--num-samples", args.num_samples is not None),
                ("--max-tokens", args.max_tokens is not None),
                ("--request-params", args.request_params is not None),
            )
            if passed
        ]
        if given:
            parser.error(
                f"{', '.join(given)} cannot be combined with samples files"
                " alone: nothing is collected, so the run uses the prompts,"
                " sample counts, token budgets and request fields recorded"
                " in the file(s)"
            )
    try:
        args.request_params = _parse_request_params(args.request_params)
    except ValueError as exc:
        parser.error(f"--request-params: {exc}")

    if args.repeat_reuse is not None:
        given_roles = {switch.lstrip("-") for switch, _ in given_sources}
        if args.repeat_reuse not in given_roles:
            parser.error(
                f"--repeat-reuse {args.repeat_reuse}: --{args.repeat_reuse}"
                " was not given"
            )
        if len(given_roles) == 1:
            parser.error(
                f"--repeat-reuse {args.repeat_reuse} would make every repeat"
                " a copy of the first, since it is the only server given"
            )

    for name, value, minimum in (
        ("--num-samples", args.num_samples, 2),
        ("--min-prefix-samples", args.min_prefix_samples, 1),
        ("--max-tokens", args.max_tokens, 1),
        ("--max-concurrency", min(args.max_concurrency), 1),
        ("--min-token-samples", args.min_token_samples, 1),
    ):
        if value is not None and value < minimum:
            parser.error(f"{name} must be at least {minimum}, got {value}")
    if min(args.seed) < 0:
        parser.error(f"--seed must not be negative, got {args.seed}")
    if not 0.0 < args.false_alarm_rate < 1.0:
        parser.error(
            f"--false-alarm-rate must be in (0, 1), got {args.false_alarm_rate}"
        )
    # Two gates share the false-alarm budget, so each gets half as its alpha.
    args.alpha = args.false_alarm_rate / 2.0
    args.num_permutations = _num_permutations(args.alpha)


@dataclass(frozen=True)
class _Source:
    """One switch's source of samples, loaded when it named a file.

    Args:
        role: The role this source fills, ``"baseline"`` or ``"test"``.
        given: The option value as it was given, a server URL unless ``path``
            is set.
        path: The samples file, when the value named an existing file.
        header: That file's header, ``None`` for a live server.
        samples: That file's samples relabeled to ``role``, an earlier run's
            samples when ``reused``, or empty for a live server.
        reused: Whether a live server's samples came from an earlier run
            under ``--suite-reuse`` or ``--repeat-reuse``.
    """

    role: str
    given: str
    path: Path | None
    header: SamplesHeader | None
    samples: tuple[Sample, ...]
    reused: bool = False

    @property
    def is_file(self) -> bool:
        """Whether this source is a file rather than a live server."""
        return self.path is not None

    @property
    def is_live(self) -> bool:
        """Whether this run draws this source's samples from a server."""
        return self.path is None and not self.reused


@dataclass(frozen=True)
class _Collected:
    """One live server's draw, kept so that a later run can reuse it.

    Args:
        samples: The samples drawn, all of one role.
        model: The model name the server reported.
    """

    samples: tuple[Sample, ...]
    model: str


@dataclass(frozen=True)
class _CollectionKey:
    """Everything that decides what a live server was asked.

    Two draws with equal keys are the same experiment, so a later run may
    take the earlier draw instead of collecting again. The role is left out on
    purpose: the analysis treats which side a sample sits on as a label, so a
    baseline may take an earlier test draw.
    """

    source: str
    endpoint: str
    prompts: str
    request_params: str
    concurrency: int
    seed: int
    num_samples: int
    max_tokens: int


@dataclass(frozen=True)
class _Offer:
    """An earlier run's draw, offered to later runs with the same key.

    Args:
        run: The run that drew it, named in the log when it is reused.
        collected: The draw.
    """

    run: int
    collected: _Collected


@dataclass(frozen=True)
class _Draw:
    """What one run used for one live role.

    Args:
        key: The draw's key.
        collected: The samples and model name.
        origin: The run that drew them, or ``None`` when this run did.
    """

    key: _CollectionKey
    collected: _Collected
    origin: int | None


@dataclass(frozen=True)
class _Scheduled:
    """One run of the invocation: a suite row at one repeat index."""

    row: int
    repeat: int
    args: argparse.Namespace


@dataclass(frozen=True)
class _Resolved:
    """The given sources and everything their combination settles.

    Args:
        sources: The baseline then the test source, minus any omitted.
        prompts: The run's prompts in prompt-index order, as token ids
            whenever a samples file supplied them.
        prompts_from_header: Whether the prompts came from a file's header.
        num_samples: Completions to draw per live server and prompt.
        max_tokens: Each source's output token budget, keyed by role.
        depth_limit: The exclusive depth bound, the smallest of the
            budgets.
    """

    sources: tuple[_Source, ...]
    prompts: tuple[Sequence[int] | str | ChatMessages, ...]
    prompts_from_header: bool
    num_samples: int
    max_tokens: Mapping[str, int]
    depth_limit: int

    @property
    def single_role(self) -> str | None:
        """The only server given, or ``None`` when both were."""
        return self.sources[0].role if len(self.sources) == 1 else None

    def source(self, role: str) -> _Source | None:
        """Returns the source filling ``role``, if its switch was given."""
        return next((s for s in self.sources if s.role == role), None)


@dataclass
class _Outcome:
    """What the analysis stages decided, before the report is assembled.

    A stage the run aborted before reaching leaves its field empty or
    ``None``.

    Args:
        controls: One entry per prompt's depth-0 control.
        analysis: The node test, or ``None`` when a stage aborted first.
        pruning: What the samples produced and what the prefix minimum kept.
        nodes_dropped: How many selected nodes bucketing dropped for having
            fewer than two buckets.
        token_budget: The token-budget calibration.
        stability: One entry per server's halves comparison.
        exit_code: The process exit code.
        reason: A one-line account of that exit code.
    """

    controls: tuple[ControlEntry, ...] = ()
    analysis: AnalysisResult | None = None
    pruning: PruningSummary | None = None
    nodes_dropped: int = 0
    token_budget: TokenBudget | None = None
    stability: tuple[StabilityResult, ...] = ()
    exit_code: int = EXIT_PASS
    reason: str = ""


async def run(args: argparse.Namespace) -> int:
    """Runs every job the invocation asks for and returns the worst exit code.

    The jobs are the suite's rows, or the command line alone as a one-row
    suite, each repeated ``--repeats`` times, row by row. A lone run writes
    its files straight into ``--outdir``; several write into ``run-NNN``
    beneath it, with a ``summary.csv`` holding a row per run in schedule
    order and, for a suite, a copy of it as ``suite.csv``.

    Args:
        args: The parsed arguments from :func:`parse_args`.

    Returns:
        The largest of the runs' exit codes.

    Raises:
        RuntimeError: If collection cannot complete.
        ValueError: If a samples file is malformed.
    """
    jobs = args.suite_rows or [args]
    schedule = [
        _Scheduled(row=row, repeat=repeat, args=job)
        for row, job in enumerate(jobs, 1)
        for repeat in range(1, args.repeats + 1)
    ]
    several = len(schedule) > 1
    suite_roles = _reuse_roles(args.suite_reuse)
    repeat_roles = _reuse_roles(args.repeat_reuse)
    summary_path = (
        args.outdir / SUMMARY_CSV
        if several and args.outdir is not None
        else None
    )
    if summary_path is not None:
        args.outdir.mkdir(parents=True, exist_ok=True)
        # Copied before the first run rather than after the last, so an
        # interrupted sweep still records what it was asked to do.
        if args.run_suite is not None:
            shutil.copyfile(args.run_suite, args.outdir / "suite.csv")
    pools: dict[tuple[int, int], dict[_CollectionKey, _Offer]] = {}
    rows: list[list[object]] = []
    codes: list[int] = []
    for number, scheduled in enumerate(schedule, 1):
        if several:
            logger.info(
                "run %d of %d: suite row %d, repeat %d",
                number,
                len(schedule),
                scheduled.row,
                scheduled.repeat,
            )
        offers = {
            role: _offers_for(
                role,
                scheduled,
                pools,
                suite_roles=suite_roles,
                repeat_roles=repeat_roles,
            )
            for role in SERVERS
        }
        outdir = _run_outdir(args, number, several=several)
        report, draws = await _run_once(
            scheduled.args, outdir=outdir, offers=offers, summary=not several
        )
        origins = {
            role: number if draw.origin is None else draw.origin
            for role, draw in draws.items()
        }
        pools[(scheduled.row, scheduled.repeat)] = {
            draw.key: _Offer(run=origins[role], collected=draw.collected)
            for role, draw in draws.items()
        }
        codes.append(report.exit_code)
        if several:
            columns, row = summary_csv(
                report, repeat=scheduled.repeat if args.repeats > 1 else None
            )
            rows.append(row)
            # Rewritten whole rather than appended to, so the file on disk is
            # always a complete CSV of the runs that have finished.
            if summary_path is not None:
                write_summary_csv(summary_path, columns, rows)
    if summary_path is not None:
        logger.info("wrote %s", summary_path)
    return max(codes)


def _reuse_roles(setting: str | None) -> frozenset[str]:
    """Turns a reuse option's value into the roles it lets a run reuse."""
    if setting is None:
        return frozenset()
    if setting == "both":
        return frozenset(SERVERS)
    return frozenset((setting,))


def _offers_for(
    role: str,
    scheduled: _Scheduled,
    pools: Mapping[tuple[int, int], Mapping[_CollectionKey, _Offer]],
    *,
    suite_roles: frozenset[str],
    repeat_roles: frozenset[str],
) -> dict[_CollectionKey, _Offer]:
    """Gathers the earlier draws one role of a run may take.

    Suite reuse looks at the same repeat index of every earlier row, so
    repeats never share through the suite; repeat reuse looks at the first
    repeat of the same row. The earliest draw wins when several match.
    """
    offers: dict[_CollectionKey, _Offer] = {}
    if role in suite_roles:
        for earlier in range(1, scheduled.row):
            earlier_pool = pools.get((earlier, scheduled.repeat), {})
            for key, offer in earlier_pool.items():
                offers.setdefault(key, offer)
    if role in repeat_roles and scheduled.repeat > 1:
        for key, offer in pools.get((scheduled.row, 1), {}).items():
            offers.setdefault(key, offer)
    return offers


def _run_outdir(
    args: argparse.Namespace, number: int, *, several: bool
) -> Path | None:
    """Places one run's files: flat for a lone run, run-NNN among several."""
    if args.outdir is None:
        return None
    if not several:
        return args.outdir
    return args.outdir / f"run-{number:03d}"


async def _run_once(
    args: argparse.Namespace,
    *,
    outdir: Path | None,
    offers: Mapping[str, Mapping[_CollectionKey, _Offer]],
    summary: bool,
) -> tuple[RunReport, dict[str, _Draw]]:
    """Runs one check end to end, printing and writing its report.

    Args:
        args: This run's parsed arguments.
        outdir: Where this run's files go, or ``None`` to write nothing.
        offers: Earlier draws each role may take instead of collecting.
        summary: Whether to write this run's own ``summary.csv``.

    Returns:
        The report, and what the run used for each live role.
    """
    rng = np.random.default_rng(_PERMUTATION_SEED)
    resolved, taken = _apply_offers(args, _resolve(args), offers)
    samples: list[Sample] = [
        sample for source in resolved.sources for sample in source.samples
    ]
    drawn, discovered, elapsed = await _collect(args, resolved)
    samples.extend(drawn)
    if outdir is not None:
        _save(outdir, args, resolved, drawn, discovered)
    draws: dict[str, _Draw] = {}
    for source in resolved.sources:
        if source.is_file:
            continue
        key = _collection_key(args, resolved, source)
        if source.reused:
            offer = taken[source.role]
            draws[source.role] = _Draw(
                key=key, collected=offer.collected, origin=offer.run
            )
        else:
            own = tuple(s for s in drawn if s.server == source.role)
            draws[source.role] = _Draw(
                key=key,
                collected=_Collected(
                    samples=own, model=discovered[source.role]
                ),
                origin=None,
            )
    # A reused source was discovered by the run that drew it.
    discovered = {
        **{role: offer.collected.model for role, offer in taken.items()},
        **discovered,
    }

    models = _model_names(resolved, discovered)
    _warn_on_request_mismatch(args, resolved)
    # Depends on how many samples each source holds, so argument validation
    # could not have derived it.
    args.effect_min_per_server = _effect_min_per_server(args, samples)
    if (role := resolved.single_role) is not None:
        outcome = _evaluate_single(
            args, samples, rng, server=role, depth_limit=resolved.depth_limit
        )
    else:
        outcome = _evaluate(
            args, samples, rng, depth_limit=resolved.depth_limit
        )
    report = RunReport(
        parameters=_parameters(args, resolved, models=models, samples=samples),
        collection=_summaries(resolved, samples, elapsed),
        controls=outcome.controls,
        analysis=outcome.analysis,
        pruning=outcome.pruning,
        nodes_dropped=outcome.nodes_dropped,
        token_budget=outcome.token_budget,
        stability=outcome.stability,
        exit_code=outcome.exit_code,
        reason=outcome.reason,
    )
    _emit(report, samples, outdir=outdir, summary=summary)
    return report, draws


def _apply_offers(
    args: argparse.Namespace,
    resolved: _Resolved,
    offers: Mapping[str, Mapping[_CollectionKey, _Offer]],
) -> tuple[_Resolved, dict[str, _Offer]]:
    """Swaps earlier draws in for the live sources an offer covers.

    Returns:
        The sources with reused ones carrying the earlier samples, and the
        offer taken for each reused role.
    """
    taken: dict[str, _Offer] = {}
    sources: list[_Source] = []
    for source in resolved.sources:
        offer = None
        if source.is_live:
            key = _collection_key(args, resolved, source)
            offer = offers.get(source.role, {}).get(key)
        if offer is None:
            sources.append(source)
            continue
        logger.info(
            "reusing %d %s samples from run %d",
            len(offer.collected.samples),
            source.role,
            offer.run,
        )
        taken[source.role] = offer
        # The draw may have sat on the other side in the run that made it,
        # so its samples take the role they fill here, as file samples do.
        sources.append(
            replace(
                source,
                samples=tuple(
                    replace(sample, server=source.role)
                    for sample in offer.collected.samples
                ),
                reused=True,
            )
        )
    return replace(resolved, sources=tuple(sources)), taken


def _collection_key(
    args: argparse.Namespace, resolved: _Resolved, source: _Source
) -> _CollectionKey:
    """Describes what one live source is asked, for matching draws."""
    return _CollectionKey(
        source=source.given,
        endpoint=(
            COMPLETIONS_ENDPOINT
            if resolved.prompts_from_header
            else args.endpoint
        ),
        prompts=json.dumps(resolved.prompts, sort_keys=True),
        request_params=json.dumps(args.request_params, sort_keys=True),
        concurrency=_role_concurrency(args, source.role),
        seed=_role_seed(args, source.role),
        num_samples=resolved.num_samples,
        max_tokens=_role_max_tokens(resolved, source.role),
    )


def _resolve(args: argparse.Namespace) -> _Resolved:
    """Opens the given sources and settles what their combination implies."""
    sources = tuple(
        _open_source(role, given)
        for role, given in zip(SERVERS, (args.baseline, args.test), strict=True)
        if given is not None
    )
    files = [source for source in sources if source.is_file]
    recorded = _header_of(files[0]) if files else None
    num_samples = _override(
        args.num_samples,
        DEFAULT_NUM_SAMPLES if recorded is None else recorded.num_samples,
    )
    live_max_tokens = _override(
        args.max_tokens,
        DEFAULT_MAX_TOKENS if recorded is None else recorded.max_tokens,
    )
    max_tokens = {
        source.role: _source_max_tokens(source, live_max_tokens)
        for source in sources
    }
    return _Resolved(
        sources=sources,
        prompts=_resolve_prompts(args, files),
        prompts_from_header=not args.prompts,
        num_samples=num_samples,
        max_tokens=max_tokens,
        depth_limit=min(max_tokens.values()),
    )


def _open_source(role: str, given: str) -> _Source:
    """Classifies one switch's value and reads it when it names a file."""
    path = Path(given)
    if not path.is_file():
        return _Source(
            role=role, given=given, path=None, header=None, samples=()
        )
    header, samples = load_samples(path)
    logger.info("read %d samples from %s", len(samples), path)
    # A sample plays the role of the switch it came through, whatever the
    # header says, so the relabeled copy is the honest record of this run.
    return _Source(
        role=role,
        given=given,
        path=path,
        header=header,
        samples=tuple(replace(sample, server=role) for sample in samples),
    )


def _header_of(source: _Source) -> SamplesHeader:
    """Returns a file source's header."""
    assert source.header is not None
    return source.header


def _override(given: int | None, fallback: int) -> int:
    """Prefers an explicit command-line value over a derived default."""
    return fallback if given is None else given


def _source_max_tokens(source: _Source, live_max_tokens: int) -> int:
    """Returns a file's recorded token budget, or the live one."""
    if source.is_file:
        return _header_of(source).max_tokens
    return live_max_tokens


def _resolve_prompts(
    args: argparse.Namespace, files: Sequence[_Source]
) -> tuple[Sequence[int] | str | ChatMessages, ...]:
    """Settles the run's prompts, preferring the command line over a header.

    A header's prompts are token ids, and a live server is sent those same
    ids, so the two sources' prefixes are identical by construction rather
    than by both tokenizers agreeing.
    """
    if args.prompts:
        return tuple(args.prompts)
    # parse_args requires prompts unless a file supplies them.
    assert files
    return tuple(_header_of(files[0]).prompts)


async def _collect(
    args: argparse.Namespace, resolved: _Resolved
) -> tuple[list[Sample], dict[str, str], dict[tuple[str, int], float]]:
    """Draws every sample the live sources owe the run.

    Returns:
        The drawn samples, the model name each live server reported keyed by
        role, and the wall time each ``(role, prompt_index)`` collection took.
    """
    live = [source for source in resolved.sources if source.is_live]
    if not live:
        return [], {}, {}

    samples: list[Sample] = []
    elapsed: dict[tuple[str, int], float] = {}
    total = len(live) * len(resolved.prompts) * resolved.num_samples
    async with make_client() as client:
        names = await asyncio.gather(
            *(_discover(client, source) for source in live)
        )
        models = {
            source.role: name for source, name in zip(live, names, strict=True)
        }
        with (
            logging_redirect_tqdm(),
            tqdm(
                total=total,
                unit="req",
                desc="collecting",
                file=sys.stderr,
                dynamic_ncols=True,
                disable=not sys.stderr.isatty(),
            ) as bar,
        ):
            progress = _Progress(bar, [source.role for source in live])
            for prompt_index, prompt in enumerate(resolved.prompts):
                drawn = await asyncio.gather(
                    *(
                        _timed_collect(
                            client,
                            args=args,
                            source=source,
                            model=models[source.role],
                            prompt=prompt,
                            prompt_index=prompt_index,
                            num_samples=resolved.num_samples,
                            max_tokens=_role_max_tokens(resolved, source.role),
                            on_sample=progress.advancer(source.role),
                        )
                        for source in live
                    )
                )
                for source, (drawn_samples, took_s) in zip(
                    live, drawn, strict=True
                ):
                    samples.extend(drawn_samples)
                    elapsed[(source.role, prompt_index)] = took_s
    return samples, models, elapsed


class _Progress:
    """One bar over every live server and prompt, with per-server counts."""

    def __init__(self, bar: tqdm, roles: Sequence[str]) -> None:
        self._bar = bar
        self._counts = dict.fromkeys(roles, 0)

    def advancer(self, role: str) -> Callable[[], None]:
        """Returns a callback crediting one finished request to ``role``."""

        def advance() -> None:
            self._counts[role] += 1
            self._bar.set_postfix(self._counts, refresh=False)
            self._bar.update(1)

        return advance


async def _discover(client: httpx.AsyncClient, source: _Source) -> str:
    """Asks one live server which model it serves."""
    try:
        return await discover_model_name(client, source.given)
    except (httpx.HTTPError, RuntimeError) as exc:
        raise RuntimeError(
            f"--{source.role} {source.given!r} is not an existing file, so it"
            f" was treated as a server URL, and that server did not answer:"
            f" {exc}. Check the spelling if you meant a samples file."
        ) from exc


async def _timed_collect(
    client: httpx.AsyncClient,
    *,
    args: argparse.Namespace,
    source: _Source,
    model: str,
    prompt: Sequence[int] | str | ChatMessages,
    prompt_index: int,
    num_samples: int,
    max_tokens: int,
    on_sample: Callable[[], None],
) -> tuple[list[Sample], float]:
    """Collects one server's samples for one prompt and times the collection."""
    started_at = time.monotonic()
    samples = await collect_samples(
        client,
        base_url=source.given,
        model=model,
        prompt=prompt,
        server=source.role,
        prompt_index=prompt_index,
        num_samples=num_samples,
        max_tokens=max_tokens,
        seed_base=_role_seed(args, source.role),
        concurrency=_role_concurrency(args, source.role),
        request_params=args.request_params,
        endpoint=args.endpoint,
        on_sample=on_sample,
    )
    return samples, time.monotonic() - started_at


def _role_max_tokens(resolved: _Resolved, role: str) -> int:
    """Returns one role's output token budget."""
    return resolved.max_tokens[role]


def _role_concurrency(args: argparse.Namespace, role: str) -> int:
    """Returns one role's cap on requests in flight."""
    return args.max_concurrency[SERVERS.index(role)]


def _role_seed(args: argparse.Namespace, role: str) -> int:
    """Returns one role's seed base."""
    return args.seed[SERVERS.index(role)]


def _save(
    directory: Path,
    args: argparse.Namespace,
    resolved: _Resolved,
    samples: Sequence[Sample],
    models: Mapping[str, str],
) -> None:
    """Writes what this run drew from each live server, header first.

    A reused source is not written again; its file is in the directory of
    the run that drew it.
    """
    live = [source for source in resolved.sources if source.is_live]
    if not live:
        logger.info(
            "%s: nothing was drawn from a server, so no samples are written",
            directory,
        )
        return
    directory.mkdir(parents=True, exist_ok=True)
    for source in live:
        own = [sample for sample in samples if sample.server == source.role]
        path = directory / f"{source.role}.jsonl"
        if path.exists():
            logger.warning("overwriting %s", path)
        header = make_header(
            url=source.given,
            endpoint=(
                COMPLETIONS_ENDPOINT
                if resolved.prompts_from_header
                else args.endpoint
            ),
            model=models[source.role],
            server=source.role,
            num_samples=resolved.num_samples,
            max_tokens=_role_max_tokens(resolved, source.role),
            seed=_role_seed(args, source.role),
            max_concurrency=_role_concurrency(args, source.role),
            request_params=args.request_params,
            prompts=_collected_prompt_ids(own),
        )
        save_samples(path, header, own)
        logger.info("wrote %d samples to %s", len(own), path)


def _collected_prompt_ids(
    samples: Sequence[Sample],
) -> tuple[tuple[int, ...], ...]:
    """Recovers the prompts one server saw, as the ids it reported back."""
    ids: dict[int, tuple[int, ...]] = {}
    for sample in samples:
        ids.setdefault(sample.prompt_index, sample.prompt_token_ids)
    return tuple(ids[index] for index in sorted(ids))


def _effect_min_per_server(
    args: argparse.Namespace, samples: Sequence[Sample]
) -> int:
    """Derives the count each server needs at a node for its TVD to count.

    A TVD from a handful of samples is large by chance alone, so eligibility
    scales with the prefix minimum, capped so that a small run still reports
    a maximum.
    """
    counts = Counter((sample.server, sample.prompt_index) for sample in samples)
    smallest = min(counts.values(), default=0)
    minimum = args.min_prefix_samples
    return max(minimum, min(10 * minimum, smallest // 10))


def _evaluate(
    args: argparse.Namespace,
    samples: Sequence[Sample],
    rng: np.random.Generator,
    *,
    depth_limit: int,
) -> _Outcome:
    """Runs the stability check, the control and the node test in order.

    Each stage can abort the run, and an aborted run reports what the stages
    before it found rather than nothing at all.
    """
    servers_present = {sample.server for sample in samples}
    if servers_present != set(SERVERS):
        return _Outcome(
            exit_code=EXIT_ABORTED,
            reason=(
                f"the samples must cover both servers {list(SERVERS)},"
                f" found {sorted(servers_present)}"
            ),
        )

    conflicts = _prompt_id_conflicts(samples)
    if conflicts:
        return _Outcome(
            exit_code=EXIT_ABORTED,
            reason=(
                "the two sides did not answer the same prompt ids, so they"
                " were not answering the same question. Either the servers"
                " tokenize a prompt differently, or two sources were"
                " collected against different prompts. " + " ".join(conflicts)
            ),
        )

    min_per_server = args.min_prefix_samples
    nodes = build_tree(
        samples, min_per_group=min_per_server, max_depth=depth_limit
    )
    selected = select_nodes(nodes, min_per_group=min_per_server)
    pruning = PruningSummary(
        tokens_generated=sum(len(sample.token_ids) for sample in samples),
        distinct_prefixes=count_prefixes(samples, max_depth=depth_limit),
        candidate_prefixes=len(selected),
    )
    logger.info(
        "built %d nodes over %d samples with a per-server minimum of %d and"
        " a depth limit of %d",
        len(nodes),
        len(samples),
        min_per_server,
        depth_limit,
    )

    stability = _stability(
        args, samples, rng, depth_limit=depth_limit, servers=SERVERS
    )
    if untested := _untested(stability):
        return _Outcome(
            pruning=pruning,
            stability=stability,
            exit_code=EXIT_ABORTED,
            reason=_untested_reason(untested),
        )
    drifted = _drifted(stability, args.alpha)
    if drifted:
        return _Outcome(
            pruning=pruning,
            stability=stability,
            exit_code=EXIT_ABORTED,
            reason=_stability_reason(drifted, args.alpha),
        )

    controls = tuple(
        _control_entry(node, args, rng) for node in control_nodes(nodes)
    )
    broken = [entry for entry in controls if not entry.result.passed]
    if broken:
        return _Outcome(
            controls=controls,
            pruning=pruning,
            stability=stability,
            exit_code=EXIT_ABORTED,
            reason=_control_reason(broken),
        )

    bucketed = bucketize_all(selected, bucket_floor=args.min_token_samples)
    nodes_dropped = len(selected) - len(bucketed)
    if not bucketed:
        return _Outcome(
            controls=controls,
            pruning=pruning,
            nodes_dropped=nodes_dropped,
            stability=stability,
            exit_code=EXIT_ABORTED,
            reason=_no_nodes_reason(args),
        )

    draws = relabeling_null(
        samples,
        min_per_group=min_per_server,
        bucket_floor=args.min_token_samples,
        num_permutations=args.num_permutations,
        rng=rng,
        max_depth=depth_limit,
    )
    analysis = analyze(
        bucketed,
        num_permutations=args.num_permutations,
        alpha=args.alpha,
        effect_min_per_group=args.effect_min_per_server,
        rng=rng,
        relabeling=RelabelingNull(
            pruned=pruning.pruned_prefixes,
            pruned_null=pruning.distinct_prefixes - draws.candidates,
            max_depth_null=draws.max_depth,
            mean_depth_null=draws.mean_depth,
        ),
    )
    if analysis.observed.num_nodes == 0:
        return _Outcome(
            controls=controls,
            analysis=analysis,
            pruning=pruning,
            nodes_dropped=nodes_dropped,
            stability=stability,
            exit_code=EXIT_ABORTED,
            reason=_no_nodes_reason(args),
        )

    token_budget = _token_budget(
        args, samples, nodes, analysis, depth_limit=depth_limit
    )
    exit_code, reason = _verdict_outcome(analysis)
    return _Outcome(
        controls=controls,
        analysis=analysis,
        pruning=pruning,
        nodes_dropped=nodes_dropped,
        token_budget=token_budget,
        stability=stability,
        exit_code=exit_code,
        reason=reason,
    )


def _evaluate_single(
    args: argparse.Namespace,
    samples: Sequence[Sample],
    rng: np.random.Generator,
    *,
    server: str,
    depth_limit: int,
) -> _Outcome:
    """Runs only the stability check, for a run given a single source.

    With nothing to compare against, the run records the samples and aborts
    if the server disagreed with itself while they were drawn.
    """
    stability = _stability(
        args, samples, rng, depth_limit=depth_limit, servers=(server,)
    )
    if untested := _untested(stability):
        return _Outcome(
            stability=stability,
            exit_code=EXIT_ABORTED,
            reason=_untested_reason(untested),
        )
    drifted = _drifted(stability, args.alpha)
    if drifted:
        return _Outcome(
            stability=stability,
            exit_code=EXIT_ABORTED,
            reason=_stability_reason(drifted, args.alpha),
        )
    return _Outcome(
        stability=stability,
        exit_code=EXIT_PASS,
        reason=(
            f"only --{server} was given, so no cross-server comparison was"
            f" made; the {server} samples are self-consistent between their"
            " first and second halves."
        ),
    )


def _drifted(
    stability: Sequence[StabilityResult], alpha: float
) -> list[StabilityResult]:
    """Returns the servers whose halves disagreed at either gate."""
    return [
        result
        for result in stability
        if result.p_sum < alpha or result.p_max < alpha
    ]


def _untested(
    stability: Sequence[StabilityResult],
) -> list[StabilityResult]:
    """Returns the servers whose halves shared no testable node.

    Such a check cannot report drift, so its p-values of 1.0 say nothing
    about stationarity and are not the evidence the run needs.
    """
    return [result for result in stability if not result.num_nodes]


def _prompt_id_conflicts(samples: Sequence[Sample]) -> list[str]:
    """Finds prompt indices the two sides did not answer with the same ids.

    Every sample records the ids its server reported for the prompt, drawn
    now or read from a file alike, so this catches both ways two sides end up
    answering different questions: servers that tokenize a text prompt
    differently, and samples files collected against different prompts. The
    depth-0 control would reject such a run as well, but it would blame the
    servers' weights or numerics for what is really a mismatched input.
    """
    seen: dict[tuple[int, str], set[tuple[int, ...]]] = defaultdict(set)
    for sample in samples:
        seen[(sample.prompt_index, sample.server)].add(sample.prompt_token_ids)

    conflicts: list[str] = []
    for prompt_index in sorted({key[0] for key in seen}):
        per_server = {
            server: seen.get((prompt_index, server), set())
            for server in SERVERS
        }
        unstable = sorted(
            server for server, ids in per_server.items() if len(ids) > 1
        )
        if unstable:
            conflicts.append(
                f"Prompt {prompt_index}: {', '.join(unstable)} reported more"
                " than one prompt tokenization."
            )
            continue
        if len({next(iter(ids)) for ids in per_server.values() if ids}) > 1:
            shapes = ", ".join(
                f"{server} {len(next(iter(ids)))} ids"
                for server, ids in per_server.items()
                if ids
            )
            conflicts.append(
                f"Prompt {prompt_index}: the servers reported different"
                f" prompt ids ({shapes})."
            )
    return conflicts


def _control_entry(
    node: Node, args: argparse.Namespace, rng: np.random.Generator
) -> ControlEntry:
    """Runs one prompt's depth-0 control.

    A first token both servers always emit, as a chat template forces, is
    agreement at zero entropy and passes with a p-value of one. Only a prompt
    one server never answered is degenerate.
    """
    bucketed = bucketize(node, bucket_floor=args.min_token_samples)
    if bucketed is not None:
        result = control_check(
            bucketed,
            num_permutations=args.num_permutations,
            rng=rng,
            alpha=args.alpha,
        )
    elif all(sum(group.values()) > 0 for group in node.counts):
        result = ControlResult(
            stats=None,
            null=None,
            p_value=1.0,
            degenerate=False,
            passed=True,
            message="passed: both servers emit the same fixed first token.",
        )
    else:
        result = ControlResult(
            stats=None,
            null=None,
            p_value=None,
            degenerate=True,
            passed=False,
            message=(
                "degenerate: only one server returned anything for this"
                " prompt, so there is no comparison to make."
            ),
        )
    return ControlEntry(prompt_index=node.prompt_index, result=result)


def _control_reason(broken: Sequence[ControlEntry]) -> str:
    """Explains a control abort, a failure apart from a degeneracy."""
    failed = [entry for entry in broken if not entry.result.degenerate]
    if failed:
        prompts = ", ".join(str(entry.prompt_index) for entry in failed)
        return (
            f"the depth-0 control failed for prompt(s) {prompts}. The two"
            " servers are not equivalent, which is a different finding from a"
            " speculative decoding bug: both produce the depth-0 token"
            " straight out of prefill with no drafting involved, so the"
            " difference is in weights, sampling defaults or numerics. Fix"
            " that before reading anything deeper."
        )
    prompts = ", ".join(str(entry.prompt_index) for entry in broken)
    return (
        f"the depth-0 control is degenerate for prompt(s) {prompts}, so the"
        " run carries no information; see the control section below."
    )


def _no_nodes_reason(args: argparse.Namespace) -> str:
    """Explains a zero-testable-node abort and how to get past it."""
    return (
        "no node was testable, so there was nothing to compare. A node needs"
        f" {args.min_prefix_samples} samples from each server and two token"
        f" buckets of at least {args.min_token_samples} pooled samples each."
        " Lower"
        " --min-prefix-samples, lower --min-token-samples, or collect more"
        " samples per prompt with --num-samples."
    )


def _verdict_outcome(analysis: AnalysisResult) -> tuple[int, str]:
    """Turns the analysis verdict into an exit code and a one-line reason."""
    verdict = analysis.verdict
    if verdict.passed:
        return EXIT_PASS, (
            "the servers are indistinguishable over"
            f" {analysis.observed.num_nodes} nodes"
        )
    return EXIT_FAIL, (
        "the servers' conditional token distributions differ: "
        + "; ".join(verdict.reasons)
    )


def _token_budget(
    args: argparse.Namespace,
    samples: Sequence[Sample],
    nodes: Sequence[Node],
    analysis: AnalysisResult,
    *,
    depth_limit: int,
) -> TokenBudget:
    """Relates the run's ``--max-tokens`` to the depth the tree reached.

    ``nodes`` is the full pruned tree, the nodes bucketing dropped included,
    since a dropped prefix still has real children just past the cap.
    """
    deepest_tested = max(stats.depth for stats in analysis.node_stats)
    tokens_generated = sum(len(sample.token_ids) for sample in samples)
    tokens_beyond = sum(
        max(0, len(sample.token_ids) - (deepest_tested + 1))
        for sample in samples
    )
    wasted = tokens_beyond / tokens_generated if tokens_generated else 0.0
    frontier = sum(
        1
        for node in nodes
        if node.depth == depth_limit - 1
        for token in node.pooled_counts()
        if min(counter[token] for counter in node.counts)
        >= args.min_prefix_samples
    )
    # The conversion from candidate to testable at the last position we did
    # generate is the best guide to what the next position would have held.
    candidates_at_cap = sum(
        1 for node in nodes if node.depth == depth_limit - 1
    )
    tested_at_cap = sum(
        1 for stats in analysis.node_stats if stats.depth == depth_limit - 1
    )
    rate = tested_at_cap / candidates_at_cap if candidates_at_cap else 0.0
    estimated = round(frontier * rate)

    if deepest_tested == depth_limit - 1:
        verdict = "cap_reached"
        line = (
            f"{TOKEN_BUDGET_PREFIX} cap reached - {frontier} candidate"
            f" {_prefix_word(frontier)} at position {depth_limit}"
            f" (est. {estimated} testable ({rate:.0%}))"
        )
    else:
        verdict = "cap_not_reached"
        first, last = deepest_tested + 1, depth_limit - 1
        span = (
            f"position {first} was"
            if first == last
            else f"positions {first} through {last} were"
        )
        line = (
            f"{TOKEN_BUDGET_PREFIX} cap was not reached - {span} generated"
            f" but never testable ({wasted:.0%} of decode work)"
        )
    return TokenBudget(
        cap=depth_limit,
        deepest_tested=deepest_tested,
        tokens_generated=tokens_generated,
        tokens_beyond=tokens_beyond,
        wasted_fraction=wasted,
        frontier_children=frontier,
        candidates_at_cap=candidates_at_cap,
        tested_at_cap=tested_at_cap,
        testable_rate=rate,
        estimated_testable=estimated,
        verdict=verdict,
        line=line,
    )


def _prefix_word(count: int) -> str:
    """Names the unit of ``TokenBudget.frontier_children``."""
    return "prefix" if count == 1 else "prefixes"


def _untested_reason(untested: Sequence[StabilityResult]) -> str:
    """Explains an abort caused by a stability check with nothing to compare."""
    servers = ", ".join(result.server for result in untested)
    return (
        f"the stability check had no testable node for {servers}, so whether"
        " that server's output is stationary is unknown rather than"
        " confirmed, and no comparison built on it can be trusted. Halving"
        " the samples doubles what a node needs, so the usual cause is"
        " --num-samples below twice --min-prefix-samples: raise --num-samples"
        " or lower --min-prefix-samples."
    )


def _stability_reason(drifted: Sequence[StabilityResult], alpha: float) -> str:
    """Explains an abort caused by a server disagreeing with itself."""
    parts = ", ".join(
        f"{result.server} (p_sum = {result.p_sum:.3g}, p_max ="
        f" {result.p_max:.3g})"
        for result in drifted
    )
    return (
        f"the stability check failed for {parts} at alpha = {alpha:.3g}."
        " That server's output drifted between the first and second halves"
        " of its samples, so no comparison built on them can be trusted."
        " This is a different finding from a speculative decoding bug."
    )


def _stability(
    args: argparse.Namespace,
    samples: Sequence[Sample],
    rng: np.random.Generator,
    *,
    depth_limit: int,
    servers: Sequence[str],
) -> tuple[StabilityResult, ...]:
    """Compares each server's first half of samples against its second half."""
    return tuple(
        _server_stability(
            args, samples, rng, server=server, depth_limit=depth_limit
        )
        for server in servers
    )


def _server_stability(
    args: argparse.Namespace,
    samples: Sequence[Sample],
    rng: np.random.Generator,
    *,
    server: str,
    depth_limit: int,
) -> StabilityResult:
    """Compares one server's earlier samples against its later ones.

    The tree is grouped by completion order instead of by server, and depth 0
    is included: between two halves of one server there is no control to
    reserve it for, and drift shows there first.
    """
    own = [sample for sample in samples if sample.server == server]
    halves = _halves_by_request(own)
    min_per_half = args.min_prefix_samples
    nodes = build_tree(
        own,
        min_per_group=min_per_half,
        groups=HALVES,
        group_of=lambda sample: halves[
            (sample.prompt_index, sample.request_index)
        ],
        max_depth=depth_limit,
    )
    analysis = analyze(
        bucketize_all(
            select_nodes(nodes, min_per_group=min_per_half, min_depth=0),
            bucket_floor=args.min_token_samples,
        ),
        num_permutations=args.num_permutations,
        alpha=args.alpha,
        effect_min_per_group=args.effect_min_per_server,
        rng=rng,
    )
    return stability_from_analysis(server, analysis)


def _halves_by_request(
    samples: Sequence[Sample],
) -> dict[tuple[int, int], str]:
    """Assigns one server's samples to its earlier or later half.

    Halving is per prompt, not over the whole server at once: prompts are
    collected one after another, so a single split would put whole prompts on
    one side and leave the halves with no node in common.

    Returns:
        The half each sample belongs to, keyed by prompt and request index.
    """
    by_prompt: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_prompt[sample.prompt_index].append(sample)

    halves: dict[tuple[int, int], str] = {}
    for prompt_index in sorted(by_prompt):
        ordered = sorted(
            by_prompt[prompt_index], key=lambda sample: sample.completed_at
        )
        half = len(ordered) // 2
        for position, sample in enumerate(ordered):
            key = (sample.prompt_index, sample.request_index)
            halves[key] = HALVES[0] if position < half else HALVES[1]
    return halves


def _summaries(
    resolved: _Resolved,
    samples: Sequence[Sample],
    elapsed: Mapping[tuple[str, int], float],
) -> tuple[CollectionSummary, ...]:
    """Summarizes what each source contributed, per prompt, baseline first."""
    grouped: dict[tuple[str, int], list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.server, sample.prompt_index)].append(sample)

    summaries: list[CollectionSummary] = []
    for source in resolved.sources:
        indices = sorted(
            prompt_index
            for role, prompt_index in grouped
            if role == source.role
        )
        for prompt_index in indices:
            own = grouped[(source.role, prompt_index)]
            if source.is_file:
                summaries.append(
                    summarize_file(
                        source.role,
                        prompt_index,
                        own,
                        path=str(source.path),
                        header=_header_of(source),
                    )
                )
            else:
                summaries.append(
                    summarize_live(
                        source.role,
                        prompt_index,
                        own,
                        url=source.given,
                        elapsed_s=elapsed.get((source.role, prompt_index), 0.0),
                    )
                )
    return tuple(summaries)


def _warn_on_request_mismatch(
    args: argparse.Namespace, resolved: _Resolved
) -> None:
    """Flags sources whose requests carried different extra fields.

    A file records the fields it was drawn with, and a live server gets the
    current ``--request-params``; a comparison across different sampling
    settings answers a different question from the one the tool is for.
    """
    by_role = {
        source.role: (
            _header_of(source).request_params
            if source.is_file
            else args.request_params
        )
        for source in resolved.sources
    }
    distinct = {
        json.dumps(params, sort_keys=True) for params in by_role.values()
    }
    if len(by_role) == 2 and len(distinct) > 1:
        logger.warning(
            "the sources were requested with different extra fields"
            " (baseline %s, test %s), so they may not be answering the same"
            " question",
            json.dumps(by_role[SERVERS[0]], sort_keys=True),
            json.dumps(by_role[SERVERS[1]], sort_keys=True),
        )


def _model_names(
    resolved: _Resolved, discovered: Mapping[str, str]
) -> tuple[str | None, str | None]:
    """Names the model behind each source, logging when the two differ."""
    baseline = _model_name(resolved.source(SERVERS[0]), discovered)
    test = _model_name(resolved.source(SERVERS[1]), discovered)
    if baseline is not None and test is not None and baseline != test:
        logger.info(
            "the two sources report different models (baseline %r, test %r);"
            " each request uses its own server's name, but a genuine model"
            " difference invalidates the comparison",
            baseline,
            test,
        )
    return baseline, test


def _parameters(
    args: argparse.Namespace,
    resolved: _Resolved,
    *,
    models: tuple[str | None, str | None],
    samples: Sequence[Sample],
) -> RunParameters:
    """Records what the run was invoked with, for the report header."""
    baseline = resolved.source(SERVERS[0])
    test = resolved.source(SERVERS[1])
    any_live = any(not source.is_file for source in resolved.sources)
    return RunParameters(
        baseline_source=None if baseline is None else baseline.given,
        test_source=None if test is None else test.given,
        baseline_header=None if baseline is None else baseline.header,
        test_header=None if test is None else test.header,
        endpoint=args.endpoint,
        request_params=args.request_params,
        baseline_model=models[0],
        test_model=models[1],
        prompts=_prompt_infos(resolved, samples),
        num_samples=resolved.num_samples if any_live else None,
        min_prefix_samples=args.min_prefix_samples,
        baseline_max_tokens=resolved.max_tokens.get(SERVERS[0]),
        test_max_tokens=resolved.max_tokens.get(SERVERS[1]),
        depth_limit=resolved.depth_limit,
        seed=_recorded_pair(resolved, args, "seed", _role_seed),
        num_permutations=args.num_permutations,
        false_alarm_rate=args.false_alarm_rate,
        alpha=args.alpha,
        min_token_samples=args.min_token_samples,
        effect_min_per_server=args.effect_min_per_server,
        max_concurrency=_recorded_pair(
            resolved, args, "max_concurrency", _role_concurrency
        ),
    )


def _recorded_pair(
    resolved: _Resolved,
    args: argparse.Namespace,
    field: str,
    live: Callable[[argparse.Namespace, str], int],
) -> tuple[int, int]:
    """Returns one recorded setting for the baseline and then the test server.

    Args:
        resolved: The run's sources.
        args: The parsed arguments the live value comes from.
        field: The header field a file source records the setting under.
        live: Reads the setting one role was given on the command line.

    Returns:
        The setting for each server, baseline first.
    """
    return (
        _recorded(resolved, SERVERS[0], field, live(args, SERVERS[0])),
        _recorded(resolved, SERVERS[1], field, live(args, SERVERS[1])),
    )


def _recorded(
    resolved: _Resolved, role: str, field: str, live_value: int
) -> int:
    """Prefers what a file source's header recorded over a live setting.

    The seed and concurrency switches only steer collection, so for a role
    that came from a file the values that actually produced its samples are
    the header's.
    """
    source = resolved.source(role)
    if source is None or not source.is_file:
        return live_value
    value = getattr(_header_of(source), field)
    assert isinstance(value, int)
    return value


def _model_name(
    source: _Source | None, discovered: Mapping[str, str]
) -> str | None:
    """Returns the model behind one source's samples, if it is known yet."""
    if source is None:
        return None
    if source.is_file:
        return _header_of(source).model
    return discovered.get(source.role)


def _prompt_infos(
    resolved: _Resolved, samples: Sequence[Sample]
) -> tuple[PromptInfo, ...]:
    """Describes each prompt, in prompt-index order, for the report."""
    lengths: dict[int, int] = {}
    for sample in samples:
        lengths.setdefault(sample.prompt_index, len(sample.prompt_token_ids))
    return tuple(
        PromptInfo(
            index=index,
            kind=_prompt_kind(resolved, prompt),
            description=_describe_prompt(prompt),
            num_prompt_tokens=lengths.get(index),
        )
        for index, prompt in enumerate(resolved.prompts)
    )


def _prompt_kind(
    resolved: _Resolved, prompt: Sequence[int] | str | ChatMessages
) -> str:
    """Names how a prompt reached the run: header, chat or text."""
    if resolved.prompts_from_header:
        return "header"
    return "text" if isinstance(prompt, str) else "chat"


def _describe_prompt(prompt: Sequence[int] | str | ChatMessages) -> str:
    """Renders a prompt short enough for one report line."""
    if isinstance(prompt, str):
        text = prompt.replace("\n", "\\n")
    elif _is_token_ids(prompt):
        text = ", ".join(str(token) for token in prompt)
    else:
        turns = [dict(message) for message in prompt]
        last = turns[-1]["content"].replace("\n", "\\n")
        text = f"chat, {len(turns)} turn(s), last: {last}"
    if len(text) <= _DESCRIPTION_CHARS:
        return text
    return text[: _DESCRIPTION_CHARS - 3] + "..."


def _emit(
    report: RunReport,
    samples: Sequence[Sample],
    *,
    outdir: Path | None,
    summary: bool,
) -> None:
    """Prints the report and writes one run's output files.

    ``summary`` writes the run's own ``summary.csv``; a repeated invocation
    passes ``False`` and collects every run's row into one file instead.
    """
    text = render_text(report)
    print(text, end="")
    if outdir is None:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.txt").write_text(text)
    with (outdir / "report.json").open("w") as file:
        json.dump(to_json_dict(report), file, indent=2)
        file.write("\n")
    if report.compares_servers:
        (outdir / "control-prefill.txt").write_text(
            render_control_prefill(report, samples)
        )
    write_csv_tables(report, outdir)
    if summary:
        columns, row = summary_csv(report)
        write_summary_csv(outdir / SUMMARY_CSV, columns, [row])
    logger.info("wrote results to %s", outdir)


def main() -> None:
    """Parses the command line, runs the check, and exits with its code."""
    # Under ``bazel run`` the process starts in the runfiles tree; resolve
    # sources, @prompt files and --outdir from where the user invoked it.
    if directory := os.environ.get("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)
    args = parse_args(sys.argv[1:])
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs one line per request; the collector reports progress itself.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sys.exit(asyncio.run(run(args)))
