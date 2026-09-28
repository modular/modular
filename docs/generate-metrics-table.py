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
"""Fills the Type and Description columns of the serving metrics tables.

Asks :func:`max.serve.telemetry._metrics_catalog.build_catalog` what the
server exports, which it answers from a Prometheus scrape of its own
exporter, so the type and description on the page are the ones a
running server reports. No server runs, and nothing is written between
the scrape and the page.

The page decides which section a metric belongs in and in what order;
that is an editorial call. Every export needs a row: a catalog entry
the page holds no row for fails regeneration, so a metric lands in the
docs with the change that adds it. Add the row with the metric's name
to the section it belongs in; the generator fills it in. ``--report``
lists catalog entries that no row documents.

Only a table inside a ``BEGIN METRICS``/``END METRICS`` marker pair is
generated. Run through ``./bazelw run //:format``; ``--check`` reports a
stale table without writing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from max.serve.telemetry._metrics_catalog import build_catalog

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = REPO_ROOT / "oss/modular/docs/max/serve/metrics.mdx"

TABLE_RE = re.compile(
    r"(?P<open>\{/\* BEGIN METRICS: (?P<section>[^*]+?) \*/\}\n\n)"
    r"(?P<head>\| *Metric[^\n]*\n\|[-| ]+\n)"
    r"(?P<rows>(?:\|[^\n]*\n)+)"
    r"(?P<close>\n\{/\* END METRICS: (?P=section) \*/\})",
    re.MULTILINE,
)
ROW_RE = re.compile(r"^\| *`(?P<name>[a-z0-9_]+)` *\|", re.MULTILINE)

# Prometheus type -> the Type column's wording.
TYPE_LABEL = {
    "counter": "Counter",
    "gauge": "Gauge",
    "histogram": "Histogram",
    "summary": "Summary",
    "unknown": "Unknown",
}

# Exported metrics that stay off the page entirely, and why. They are
# absent from the rendered tables and from --report.
UNDOCUMENTED_ON_PURPOSE = {
    "maxserve_dkv_": "The dKV connector is available only in Modular Cloud.",
}


class GeneratorError(Exception):
    """Raised when the catalog or the page cannot be rendered as-is."""


def load_catalog() -> dict[str, tuple[str, str]]:
    """Returns each exported metric's type and description, keyed by name.

    Returns:
        A mapping of series name to its Prometheus type and description.
    """
    return {
        metric.name: (metric.type, metric.description)
        for metric in build_catalog()
    }


def cell(text: str) -> str:
    """Returns text safe to place in a Markdown table cell.

    Args:
        text: The description to escape.

    Returns:
        The text with backslashes, pipes and newlines escaped, so a
        description cannot split the row it sits in.
    """
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_table(rows: str, catalog: dict[str, tuple[str, str]]) -> str:
    """Returns one table with its Type and Description columns refreshed.

    The rows keep the order the page gives them, and the padding matches
    what ``rumdl fmt`` produces, so the formatter leaves the result alone.

    Args:
        rows: The table's existing row lines, which name the metrics.
        catalog: The exported metrics, keyed by series name.

    Returns:
        The rendered table, header and separator included.

    Raises:
        GeneratorError: If a row names a metric the server does not export,
            or names one the catalog describes without a description.
    """
    built = []
    for name in ROW_RE.findall(rows):
        if name not in catalog:
            raise GeneratorError(
                f"The page documents `{name}`, which the server does not "
                "export. Remove the row, or correct the name."
            )
        kind, description = catalog[name]
        if not description:
            raise GeneratorError(
                f"{name} reaches the catalog with no description. Describe "
                "it where the metric is registered; the page renders it."
            )
        built.append(
            (
                f"`{name}`",
                TYPE_LABEL.get(kind, kind.title()),
                cell(description.rstrip(".")) + ".",
            )
        )
    header = ("Metric", "Type", "Description")
    grid = [header, *built]
    widths = [max(len(r[i]) for r in grid) for i in range(3)]
    lines = [
        "| "
        + " | ".join(c.ljust(widths[i]) for i, c in enumerate(header))
        + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in built:
        lines.append(
            "| "
            + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row))
            + " |"
        )
    return "\n".join(lines) + "\n"


def undocumented(page: str, catalog: dict[str, tuple[str, str]]) -> list[str]:
    """Returns exported metrics that no row on the page documents.

    Args:
        page: The full text of the metrics reference.
        catalog: The exported metrics, keyed by series name.

    Returns:
        The series names absent from the page, sorted, excluding the ones
        left out on purpose.
    """
    documented = set(ROW_RE.findall(page))
    return sorted(
        name
        for name in catalog
        if name not in documented
        and not any(name.startswith(p) for p in UNDOCUMENTED_ON_PURPOSE)
    )


def render_page(current: str, catalog: dict[str, tuple[str, str]]) -> str:
    """Returns the page with every table's cells refreshed.

    Args:
        current: The full text of the metrics reference.
        catalog: The exported metrics, keyed by series name.

    Returns:
        The page with generated cells written between the markers.

    Raises:
        GeneratorError: If a row names a metric the server does not
            export, or the catalog holds a metric no row documents.
    """
    updated = TABLE_RE.sub(
        lambda m: (
            m.group("open")
            + render_table(m.group("rows"), catalog)
            + m.group("close")
        ),
        current,
    )
    missing = undocumented(updated, catalog)
    if missing:
        names = ", ".join(f"`{name}`" for name in missing)
        raise GeneratorError(
            f"The page documents no row for {names}. Add a row for each "
            "in the section it belongs to, and the generator fills in "
            "its type and description."
        )
    return updated


def main() -> int:
    """Refreshes the tables, or reports on them, and returns an exit code."""
    parser = argparse.ArgumentParser(description="Refresh the metrics tables.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report a stale table and exit non-zero without writing.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="List exported metrics the page does not document.",
    )
    args = parser.parse_args()

    try:
        catalog = load_catalog()
        current = OUTPUT_FILE.read_text()
        if args.report:
            for name in undocumented(current, catalog):
                print(f"{name}: exported, not on the page")
            return 0
        updated = render_page(current, catalog)
    except GeneratorError as err:
        print(f"{OUTPUT_FILE.name}: {err}", file=sys.stderr)
        return 1

    if updated == current:
        return 0
    if args.check or os.getenv("CHECK", "").lower() in ("1", "true"):
        print(
            f"{OUTPUT_FILE.relative_to(REPO_ROOT)} is out of date. Run "
            "`./bazelw run //:format` to regenerate it.",
            file=sys.stderr,
        )
        return 1
    OUTPUT_FILE.write_text(updated)
    print(f"Updated {OUTPUT_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
