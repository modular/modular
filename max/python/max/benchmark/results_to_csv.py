#!/usr/bin/env python3
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

"""Post-process benchmark result JSON blobs into a CSV with selectable columns.

The serving benchmark treats its JSON output as the single source of truth: a
``save_result_json`` blob carries *every* metric the run produced. A CSV, by
contrast, is a presentation of that data — most consumers care about a small
"summary" slice, while power users opt into complementary groups (prefill /
decode batch stats, per-turn cache rates, GPU stats, ...) or name exact columns.

This module keeps that separation clean. It reads one or more result JSON files,
flattens each nested blob into a flat ``dotted.key`` column namespace, and emits
a CSV containing a curated default summary plus whatever additional columns or
groups the caller selects. It never recomputes metrics; it only projects the
columns already present in the JSON.

Run it through Bazel. Under ``bazel run`` the working directory is the
workspace root, so input/output paths are resolved relative to the repo root
(pass absolute paths to target files elsewhere)::

    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        results-1-median.json results-2-median.json -o summary.csv

    # add the prefill/decode batch stats and GPU columns:
    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        sweep-*/results-*.json -o detailed.csv --groups prefill_decode,gpu

    # emit every column found in the JSON:
    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        results.json -o full.csv --all

    # pick an exact set of columns (no summary):
    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        results.json -o custom.csv \\
        --only --columns max_concurrency,mean_ttft_ms,request_throughput
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeGuard

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# A JSON value as produced by ``json.load``. Recursive: objects nest objects
# and lists. Using this instead of a bare ``object`` lets the flattener's
# branches typecheck (``json.dumps`` only ever sees a non-str scalar or list).
JSONValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JSONValue"]
    | Mapping[str, "JSONValue"]
)
# A JSON object: the top level of a single result blob.
JSONObject = Mapping[str, JSONValue]
# One flattened output row: a string cell per (flattened) column name.
Row = dict[str, str]


class ResultEntry(TypedDict, total=False):
    """One ``results`` entry of a ``ResultSetInContext`` document."""

    iteration_config: JSONObject
    result: JSONValue


class ResultSetDocument(TypedDict, total=False):
    """The ``{run_context, results}`` envelope emitted by the JSON reporter."""

    run_context: JSONObject
    results: list[ResultEntry]


# Curated, ordered default columns. Only those actually present in the input
# are emitted, so this same list works for text-generation and pixel-generation
# runs (missing keys are silently skipped). This is the "summary" slice most
# consumers want; everything else is opt-in via ``--groups`` / ``--columns`` /
# ``--all``.
SUMMARY_COLUMNS: tuple[str, ...] = (
    # Run identity / configuration.
    "date",
    "model_id",
    "backend",
    "benchmark_task",
    "dataset_name",
    "num_prompts",
    "max_concurrency",
    "max_concurrent_conversations",
    "request_rate",
    # Top-line throughput / duration.
    "duration",
    "completed",
    "failures",
    "request_throughput",
    "total_input_tokens",
    "total_output_tokens",
    "total_generated_outputs",
    "aggregate_tokens_per_minute",
    "mean_input_throughput",
    "mean_output_throughput",
    # Latency headlines (mean + median + p99 for the metrics people quote).
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_step_tpot_ms",
    "median_step_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_latency_ms",
    "median_latency_ms",
    "p99_latency_ms",
    # Cache + speculative-decode headlines.
    "global_cached_token_rate",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
)


def _group_prefill_decode(key: str) -> bool:
    return key.startswith(("prefill_stats.", "decode_stats.")) or (
        key.startswith("server_metrics")
        and ("prefill" in key or "decode" in key)
    )


def _group_gpu(key: str) -> bool:
    return key in (
        "peak_gpu_memory_mib",
        "available_gpu_memory_mib",
        "gpu_utilization",
    )


def _group_cpu(key: str) -> bool:
    return key.startswith("cpu_metrics")


def _group_per_turn(key: str) -> bool:
    return "per_turn" in key


def _group_server_metrics(key: str) -> bool:
    return key.startswith("server_metrics")


def _group_spec_decode(key: str) -> bool:
    return key.startswith("spec_decode")


def _group_confidence(key: str) -> bool:
    return (
        key.endswith(("_confidence", "_sample_size"))
        or "_ci_lower" in key
        or "_ci_upper" in key
        or "_ci_relative_width" in key
    )


def _group_client_args(key: str) -> bool:
    return key.startswith("client_args")


def _group_steady_state(key: str) -> bool:
    return key.startswith("steady_state") or key == "num_outliers_rejected"


def _group_raw(key: str) -> bool:
    # Per-request sample arrays and other bulk lists. Kept out of the summary
    # because each is a single opaque JSON cell that bloats the CSV.
    return key in (
        "input_lens",
        "output_lens",
        "ttfts",
        "latencies",
        "num_generated_outputs",
        "errors",
        "request_submit_times",
        "request_complete_times",
        "per_turn_cached_token_rates",
        "per_turn_cache_retentions",
        "session_server_stats",
        "aggregate_server_stats",
    )


# Names of the opt-in serving column groups (the keys of COLUMN_GROUPS). A
# ``Literal`` so the group set is type-checked at definition and typos are caught.
ServingColumnGroup = Literal[
    "prefill_decode",
    "gpu",
    "cpu",
    "per_turn",
    "server_metrics",
    "spec_decode",
    "confidence",
    "client_args",
    "steady_state",
    "raw",
]

# Named opt-in column groups. Each predicate matches a *flattened* column name.
COLUMN_GROUPS: Mapping[ServingColumnGroup, Callable[[str], bool]] = {
    "prefill_decode": _group_prefill_decode,
    "gpu": _group_gpu,
    "cpu": _group_cpu,
    "per_turn": _group_per_turn,
    "server_metrics": _group_server_metrics,
    "spec_decode": _group_spec_decode,
    "confidence": _group_confidence,
    "client_args": _group_client_args,
    "steady_state": _group_steady_state,
    "raw": _group_raw,
}


def flatten(obj: JSONObject) -> Row:
    """Flattens a nested JSON object into a flat ``dotted.key -> cell`` mapping.

    Nested objects are expanded recursively with dot-separated keys. Lists and
    other non-string scalars are JSON-encoded into a single cell (matching the
    streaming CSV reporter's convention), ``None`` becomes an empty string, and
    strings pass through unchanged.

    Args:
        obj:
            The parsed JSON object (a single result blob) to flatten.

    Returns:
        A mapping from flattened column name to its string cell value.
    """
    flat: Row = {}

    def _walk(value: JSONValue, prefix: str) -> None:
        if isinstance(value, Mapping):
            if not value:
                flat[prefix] = "{}"
                return
            for k, v in value.items():
                child = f"{prefix}.{k}" if prefix else str(k)
                _walk(v, child)
        elif value is None:
            flat[prefix] = ""
        elif isinstance(value, str):
            flat[prefix] = value
        else:
            # Lists, ints, floats, bools: a single JSON-encoded cell. Lists are
            # variable-length per row and cannot become their own columns.
            flat[prefix] = json.dumps(value)

    _walk(obj, "")
    return flat


def _is_result_set_document(
    data: Mapping[str, object],
) -> TypeGuard[ResultSetDocument]:
    """Returns True when ``data`` is a ``ResultSetInContext`` envelope.

    Detected by the presence of a ``results`` list; ``run_context`` and each
    entry's fields are validated where they are read.
    """
    return isinstance(data.get("results"), list)


def _rows_from_result_set(document: ResultSetDocument) -> list[Row]:
    """Flattens a ``ResultSetInContext`` document into one row per result.

    Each row merges the shared ``run_context`` with the entry's
    ``iteration_config`` and ``result``.
    """
    run_context = document.get("run_context")
    rows: list[Row] = []
    for entry in document["results"]:
        record: dict[str, JSONValue] = {}
        if isinstance(run_context, Mapping):
            record["run_context"] = run_context
        iteration_config = entry.get("iteration_config")
        if isinstance(iteration_config, Mapping):
            record["iteration_config"] = iteration_config
        if "result" in entry:
            record["result"] = entry["result"]
        rows.append(flatten(record))
    return rows


def load_result_rows(path: Path) -> list[Row]:
    """Loads a result JSON file and returns one flattened row per result.

    Supports the shapes the serving benchmark emits:

    - a single ``save_result_json`` blob (one row);
    - a JSON array of such blobs (one row each);
    - a ``ResultSetInContext`` document with a ``results`` list, where each
      entry is merged with the shared ``run_context`` and its
      ``iteration_config`` (one row each).

    Args:
        path:
            The path to the result JSON file.

    Returns:
        A list of flattened rows (each a ``column -> cell`` mapping).

    Raises:
        ValueError: If the top-level JSON is not an object or array.
    """
    with open(path) as f:
        data: object = json.load(f)

    if isinstance(data, list):
        return [flatten(item) for item in data]
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{path}: expected a JSON object or array at the top level, "
            f"got {type(data).__name__}"
        )
    if _is_result_set_document(data):
        return _rows_from_result_set(data)
    return [flatten(data)]


def _ordered_unique(items: Iterable[str]) -> list[str]:
    """Returns ``items`` with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def select_columns(
    available: Sequence[str],
    *,
    groups: Sequence[str] = (),
    columns: Sequence[str] = (),
    all_columns: bool = False,
    only: bool = False,
) -> list[str]:
    """Chooses the ordered CSV columns from those available in the input.

    The default (no options) is :data:`SUMMARY_COLUMNS`, restricted to columns
    actually present in the input. ``groups`` and ``columns`` add to that
    summary; ``only`` drops the summary so just the requested groups/columns are
    emitted; ``all_columns`` emits every available column (summary first, then
    the rest sorted).

    For example:

    .. code-block:: python

        available = [
            "date", "mean_ttft_ms", "prefill_stats.x", "gpu_utilization",
        ]
        # Default: the summary columns present in the input, in summary order.
        select_columns(available)
        # -> ["date", "mean_ttft_ms"]
        # Opt-in groups add to the summary (group columns in definition order):
        select_columns(available, groups=["prefill_decode", "gpu"])
        # -> ["date", "mean_ttft_ms", "prefill_stats.x", "gpu_utilization"]
        # An exact, ordered set with no summary:
        select_columns(available, only=True, columns=["mean_ttft_ms", "date"])
        # -> ["mean_ttft_ms", "date"]

    Args:
        available:
            All flattened column names present across the input rows.
        groups:
            Names of opt-in :data:`COLUMN_GROUPS` to include.
        columns:
            Exact column names to include, in the given order. Included even if
            absent from ``available`` (they produce empty cells).
        all_columns:
            Emit every available column.
        only:
            Emit only the requested ``groups`` / ``columns`` (no summary).

    Returns:
        The ordered list of column names to write.

    Raises:
        ValueError: If a requested group name is unknown.
    """
    available_set = set(available)

    if all_columns:
        summary_present = [c for c in SUMMARY_COLUMNS if c in available_set]
        rest = sorted(c for c in available_set if c not in set(summary_present))
        return _ordered_unique([*summary_present, *rest])

    unknown = [g for g in groups if g not in COLUMN_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown column group(s): {', '.join(sorted(unknown))}. "
            f"Available groups: {', '.join(sorted(COLUMN_GROUPS))}."
        )

    selected: list[str] = []
    if not only:
        selected.extend(c for c in SUMMARY_COLUMNS if c in available_set)

    # Iterate the group definitions (not the caller-supplied strings) so the
    # ``Literal``-typed keys stay type-checked; requested groups are emitted in
    # definition order.
    requested = set(groups)
    for name, predicate in COLUMN_GROUPS.items():
        if name in requested:
            selected.extend(sorted(c for c in available_set if predicate(c)))

    # Explicit columns keep the caller's order and are emitted even if absent.
    selected.extend(columns)

    return _ordered_unique(selected)


def write_csv(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    output_file: Path,
) -> None:
    """Writes ``rows`` to ``output_file`` as CSV using the given ``columns``.

    Missing cells (a column absent from a given row) are written as empty
    strings.

    Args:
        rows:
            The flattened result rows.
        columns:
            The ordered column names forming the header.
        output_file:
            The CSV path to write.
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(column, "") for column in columns])


def convert(
    inputs: Sequence[Path],
    output_file: Path,
    *,
    groups: Sequence[str] = (),
    columns: Sequence[str] = (),
    all_columns: bool = False,
    only: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Reads result JSON ``inputs`` and writes a CSV with selected columns.

    Args:
        inputs:
            The result JSON files to read (each may hold one or many results).
        output_file:
            The CSV path to write.
        groups:
            Names of opt-in :data:`COLUMN_GROUPS` to include.
        columns:
            Exact column names to include, in order.
        all_columns:
            Emit every available column.
        only:
            Emit only the requested groups/columns (no summary).
        dry_run:
            Compute the selected columns and report what would be written, but
            do not create ``output_file``.

    Returns:
        The ordered list of columns that were (or would be) written.

    Raises:
        ValueError: If no inputs are given or a group name is unknown.
    """
    if not inputs:
        raise ValueError("no input files given")

    rows: list[Row] = []
    for path in inputs:
        rows.extend(load_result_rows(path))

    # Union of keys across rows, preserving first-seen order for stable output.
    available = _ordered_unique(key for row in rows for key in row)
    selected = select_columns(
        available,
        groups=groups,
        columns=columns,
        all_columns=all_columns,
        only=only,
    )
    if dry_run:
        logger.info(
            "[dry run] would write %d row(s) and %d column(s) to %s: %s",
            len(rows),
            len(selected),
            output_file,
            ", ".join(selected),
        )
        return selected
    write_csv(rows, selected, output_file)
    logger.info(
        "Wrote %d row(s) and %d column(s) to %s",
        len(rows),
        len(selected),
        output_file,
    )
    return selected


def _expand_inputs(patterns: Sequence[str]) -> list[Path]:
    """Expands CLI input arguments, globbing any that contain wildcards."""
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(glob.glob(pattern))
            if not matches:
                logger.warning("No files matched pattern: %s", pattern)
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _parse_csv_list(value: str | None) -> list[str]:
    """Splits a comma-separated CLI value into a list, trimming whitespace."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for JSON-to-CSV benchmark result post-processing."""
    # Under ``bazel run`` the process starts in the runfiles tree; hop back to
    # the workspace root so relative input/output paths (and globs) resolve
    # where the user invoked Bazel. Unset outside Bazel, so this is a no-op.
    if workspace := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace)

    parser = argparse.ArgumentParser(
        prog="results-to-csv",
        description=(
            "Flatten benchmark result JSON blobs into a CSV with a curated "
            "summary column set plus opt-in groups and columns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available column groups: " + ", ".join(sorted(COLUMN_GROUPS)) + "."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Result JSON file(s). Glob patterns are expanded.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--groups",
        default="",
        help=(
            "Comma-separated opt-in column groups to add to the summary "
            "(e.g. prefill_decode,gpu)."
        ),
    )
    parser.add_argument(
        "--columns",
        default="",
        help="Comma-separated exact column names to add, in order.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_columns",
        help="Emit every column found in the input.",
    )
    parser.add_argument(
        "--only",
        action="store_true",
        help=(
            "Emit only the requested --groups / --columns (drop the default "
            "summary)."
        ),
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print the columns available in the input and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report the rows/columns that would be written without creating "
            "the output CSV."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    inputs = _expand_inputs(args.inputs)
    if not inputs:
        parser.error("no input files found")

    if args.list_columns:
        rows: list[Row] = []
        for path in inputs:
            rows.extend(load_result_rows(path))
        for column in _ordered_unique(key for row in rows for key in row):
            print(column)
        return 0

    try:
        convert(
            inputs,
            args.output,
            groups=_parse_csv_list(args.groups),
            columns=_parse_csv_list(args.columns),
            all_columns=args.all_columns,
            only=args.only,
            dry_run=args.dry_run,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
