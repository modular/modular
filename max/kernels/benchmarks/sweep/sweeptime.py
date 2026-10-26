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
"""Sums sweep phase timings across processes and prints the phase table.

Reads the `SWEEPTIME,<tag>,<phase>,<count>,<seconds>` lines that
`bench_sweep.PhaseTimer` prints, from log files, from directories searched for
`test.log` files (such as `bazel-testlogs/...`), from `--json` outputs, or from
stdin. Every process contributes one `process_total`; the other phases tile it,
so the table shows where the time of a whole sweep went.

    ./bazelw run //max/kernels/benchmarks/sweep:sweeptime -- \\
        $PWD/bazel-testlogs/max/kernels/benchmarks/sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

PROCESS_TOTAL = "process_total"


@dataclass
class PhaseTotals:
    """Totals for one group of processes, phases in first-seen order."""

    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, phase: str, count: int, seconds: float) -> None:
        """Adds one phase's total from one process."""
        self.seconds[phase] = self.seconds.get(phase, 0.0) + seconds
        self.counts[phase] = self.counts.get(phase, 0) + count

    @property
    def processes(self) -> int:
        """Returns the number of processes summed."""
        return self.counts.get(PROCESS_TOTAL, 0)


def parse_lines(lines: Iterable[str]) -> Iterator[tuple[str, str, int, float]]:
    """Yields `(tag, phase, count, seconds)` for every `SWEEPTIME` line."""
    for line in lines:
        line = line.strip()
        if not line.startswith("SWEEPTIME,"):
            continue
        parts = line.split(",")
        if len(parts) != 5:
            continue
        _, tag, phase, count, seconds = parts
        yield tag, phase, int(count), float(seconds)


def parse_json(text: str) -> Iterator[tuple[str, str, int, float]]:
    """Yields `(tag, phase, count, seconds)` from a `--json` output."""
    data = json.loads(text)
    tag = str(data["tag"])
    for phase, totals in data["phases"].items():
        yield tag, phase, int(totals["count"]), float(totals["seconds"])


def iter_inputs(paths: Iterable[Path]) -> Iterator[Path]:
    """Expands directories to the `test.log` and `*.json` files below them."""
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("test.log"))
            yield from sorted(path.rglob("*.json"))
        else:
            yield path


def collect(paths: Iterable[Path], by_tag: bool) -> dict[str, PhaseTotals]:
    """Returns phase totals, keyed by tag when `by_tag` and by `all` if not."""
    groups: dict[str, PhaseTotals] = {}

    def add(rows: Iterable[tuple[str, str, int, float]]) -> None:
        for tag, phase, count, seconds in rows:
            key = tag if by_tag else "all"
            groups.setdefault(key, PhaseTotals()).add(phase, count, seconds)

    path_list = list(paths)
    if not path_list:
        add(parse_lines(sys.stdin))
    for path in iter_inputs(path_list):
        text = path.read_text(errors="replace")
        if path.suffix == ".json":
            add(parse_json(text))
        else:
            add(parse_lines(text.splitlines()))
    return groups


def format_table(name: str, totals: PhaseTotals) -> str:
    """Formats one group's phases with their share of process time."""
    process = totals.seconds.get(PROCESS_TOTAL, 0.0)
    rows = [("Phase", "Count", "Seconds", "Share")]
    accounted = 0.0
    for phase, seconds in totals.seconds.items():
        if phase == PROCESS_TOTAL:
            continue
        accounted += seconds
        share = f"{100 * seconds / process:.1f}%" if process > 0 else "-"
        rows.append((phase, str(totals.counts[phase]), f"{seconds:.3f}", share))
    # The pre-main estimate has 10 ms resolution, so a small remainder is
    # rounding; a large one means a phase is missing.
    if process > 0 and abs(process - accounted) > 0.01:
        rows.append(
            (
                "unaccounted",
                "",
                f"{process - accounted:.3f}",
                f"{100 * (process - accounted) / process:.1f}%",
            )
        )
    rows.append(
        (PROCESS_TOTAL, str(totals.processes), f"{process:.3f}", "100.0%")
    )
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = [f"{name}: {totals.processes} process(es)"]
    for i, row in enumerate(rows):
        lines.append(
            "  "
            + row[0].ljust(widths[0])
            + "  "
            + "  ".join(row[c].rjust(widths[c]) for c in range(1, 4))
        )
        if i == 0:
            lines.append("  " + "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parses arguments, sums the phases and prints one table per group."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="logs, --json outputs or directories; stdin when none",
    )
    parser.add_argument(
        "--by-tag",
        action="store_true",
        help="print one table per SWEEPTIME tag instead of one in total",
    )
    args = parser.parse_args(argv)
    groups = collect(args.paths, args.by_tag)
    if not groups:
        print("no SWEEPTIME lines found", file=sys.stderr)
        return 1
    print("\n\n".join(format_table(k, v) for k, v in sorted(groups.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
