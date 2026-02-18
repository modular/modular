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
"""Terminal-friendly visualization for kbench results."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from utils import format_time

# Throughput units to detect in column names.
_THROUGHPUT_UNITS = ("GB/s", "GFLOPS/s", "GElems/s")


def _safe_eval(v: str) -> Any:
    """Parse value string as a Python literal, returning as-is if parsing fails.

    Uses ast.literal_eval to safely parse numbers, strings, tuples, lists,
    dicts, booleans, and None without executing arbitrary code.
    """
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def spec_to_dict(spec: str) -> dict:
    """Convert 'name/$p1=v1/$p2=v2/...' to {'name': name, 'p1': v1, ...}."""
    parts = spec.split("/")
    d = {"name": parts[0]}
    d.update(
        (k.strip("$"), _safe_eval(v))
        for part in parts[1:]
        if "=" in part
        for k, v in [part.split("=", 1)]
    )
    return d


def _extract_unit(col_name: str) -> str:
    """Extract unit from column name like 'DataMovement (GB/s)' -> 'GB/s'."""
    if match := re.search(r"\(([^)]+)\)", col_name):
        return match.group(1)
    return ""


def _to_hashable(val: Any) -> Any:
    """Convert unhashable types (like lists) to hashable equivalents for comparison."""
    if isinstance(val, list):
        return tuple(val)
    return val


def _count_unique(series: pd.Series) -> int:
    """Count unique values in a series, handling unhashable types like lists."""
    try:
        return int(series.nunique())
    except TypeError:
        # Handle unhashable types by converting to hashable equivalents
        return int(series.apply(_to_hashable).nunique())


def get_display_df(
    merged_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], list[str], list[str]]:
    """Convert raw kbench results to display-friendly DataFrame.

    Returns: (display_df, fixed_params, pivot_cols, throughput_cols)
    """
    # Parse spec column into parameter columns
    spec_df = merged_df["spec"].apply(spec_to_dict).apply(pd.Series)

    # Partition columns into pivots (vary) vs fixed (constant), excluding 'name'
    cols = [c for c in spec_df.columns if c != "name"]
    pivot_cols = [c for c in cols if _count_unique(spec_df[c]) > 1]
    fixed_params = {
        c: str(spec_df[c].iloc[0]) for c in cols if c not in pivot_cols
    }

    # Build display DataFrame
    display_df = spec_df[pivot_cols].copy() if pivot_cols else pd.DataFrame()

    # Time: "met (s)" is in seconds, "met (ms)" is in milliseconds (fixed as of Dec 2025)
    if "met (s)" in merged_df.columns:
        # Values are in seconds, convert to milliseconds
        time_ms = merged_df["met (s)"] * 1000
    elif "met (ms)" in merged_df.columns:
        # Values are already in milliseconds (no conversion needed)
        time_ms = merged_df["met (ms)"]
    else:
        time_ms = pd.Series([0])

    display_df["time_ms"] = time_ms
    display_df["time"] = time_ms.apply(format_time)

    # Throughput columns
    throughput_cols = [
        c for c in merged_df.columns if any(u in c for u in _THROUGHPUT_UNITS)
    ]
    for col in throughput_cols:
        display_df[col] = merged_df[col]

    return display_df, fixed_params, pivot_cols, throughput_cols


def render_bars(
    console: Console,
    display_df: pd.DataFrame,
    pivot_cols: Sequence[str],
    throughput_cols: Sequence[str],
    bar_width: int = 40,
) -> None:
    """Render Unicode bar chart grouped by all secondary pivots."""
    if display_df.empty:
        console.print("[yellow]No results to display.[/yellow]")
        return

    max_time = display_df["time_ms"].max()

    # Determine grouping: 2+ pivots -> group by all except first, compare by first
    group_cols = list(pivot_cols[1:]) if len(pivot_cols) >= 2 else []
    compare_col = pivot_cols[0] if pivot_cols else None

    def render_group(df: pd.DataFrame) -> None:
        baseline = df["time_ms"].iloc[0] if not df.empty else 1.0
        for _, row in df.iterrows():
            # Bar visualization
            ratio = row["time_ms"] / max_time if max_time > 0 else 0
            filled = int(ratio * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            label = str(row[compare_col]) if compare_col else "result"

            # Throughput values
            tput = [
                f"{row[c]:.0f} {_extract_unit(c)}"
                for c in throughput_cols
                if pd.notna(row.get(c))
            ]
            tput_str = f"  [cyan]{', '.join(tput)}[/cyan]" if tput else ""

            # Relative speedup vs baseline
            rel_str = ""
            if row["time_ms"] != baseline and baseline > 0:
                rel_str = f" [dim]({row['time_ms'] / baseline:.2f}x)[/dim]"

            console.print(
                f"  {label:12} {bar} {row['time']}{tput_str}{rel_str}"
            )

    if group_cols:
        # Convert unhashable types for groupby
        groupby_keys = [display_df[c].apply(_to_hashable) for c in group_cols]
        for group_vals, group_df in display_df.groupby(groupby_keys):
            # group_vals is a single value when 1 group col, tuple when multiple
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            group_label = ", ".join(
                f"{c}={v}" for c, v in zip(group_cols, group_vals, strict=True)
            )
            console.print(f"[bold]{group_label}:[/bold]")
            render_group(group_df)
            console.print()
    else:
        render_group(display_df)


def render_table(
    console: Console,
    display_df: pd.DataFrame,
    pivot_cols: Sequence[str],
    throughput_cols: Sequence[str],
) -> None:
    """Render Rich table view."""
    if display_df.empty:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold blue")

    # Add columns: pivots + time + throughput + baseline comparison
    for col in pivot_cols:
        table.add_column(col, justify="left")
    table.add_column("time/iter", justify="right")
    for col in throughput_cols:
        table.add_column(
            _extract_unit(col) or col, justify="right", style="cyan"
        )
    table.add_column("vs base", justify="right")

    baseline = display_df["time_ms"].iloc[0]
    for _, row in display_df.iterrows():
        values = [str(row[c]) for c in pivot_cols]
        values.append(row["time"])
        values.extend(
            f"{row[c]:.0f}" if pd.notna(row.get(c)) else "-"
            for c in throughput_cols
        )
        values.append(
            f"{row['time_ms'] / baseline:.2f}x" if baseline > 0 else "1.00x"
        )
        table.add_row(*values)

    console.print(table)


def render_summary(
    console: Console,
    display_df: pd.DataFrame,
    pivot_cols: Sequence[str],
) -> None:
    """Render summary statistics grouped by first pivot column."""
    if display_df.empty:
        return

    group_col = pivot_cols[0] if pivot_cols else None
    if not group_col:
        console.print(
            f"[bold]Summary:[/bold] avg={format_time(display_df['time_ms'].mean())}"
        )
        return

    # Convert unhashable types for groupby
    groupby_col = display_df[group_col].apply(_to_hashable)
    summary = (
        display_df.groupby(groupby_col)["time_ms"].agg(["mean"]).reset_index()
    )
    baseline = summary["mean"].iloc[0]

    console.print("[bold]Summary:[/bold]")
    for i, row in enumerate(summary.itertuples()):
        name, avg = row[1], row[2]
        avg_str = format_time(avg)
        if i == 0:
            console.print(f"  {name}: avg={avg_str} (baseline)")
        else:
            rel = avg / baseline if baseline > 0 else 1.0
            console.print(
                f"  {name}: avg={avg_str} ({rel:.2f}x {'slower' if rel > 1 else 'faster'})"
            )


def render_results(
    merged_df: pd.DataFrame,
    mode: str = "bars",
    console: Console | None = None,
) -> None:
    """Main entry point for terminal visualization."""
    if mode == "none":
        return

    console = console or Console()

    # Validate required columns
    has_time = "met (s)" in merged_df.columns or "met (ms)" in merged_df.columns
    if "spec" not in merged_df.columns or not has_time:
        console.print(
            "[yellow]Cannot render: missing 'spec' or time column.[/yellow]"
        )
        return

    display_df, fixed_params, pivot_cols, throughput_cols = get_display_df(
        merged_df
    )

    # Header
    console.print("\n" + "─" * 70)
    if fixed_params:
        console.print(
            f"[dim]Fixed: {', '.join(f'{k}={v}' for k, v in fixed_params.items())}[/dim]\n"
        )

    # Render visualization
    if mode == "bars":
        render_bars(console, display_df, pivot_cols, throughput_cols)
        console.print()
        render_summary(console, display_df, pivot_cols)
    elif mode == "table":
        render_table(console, display_df, pivot_cols, throughput_cols)
        console.print()
        render_summary(console, display_df, pivot_cols)
    elif mode == "summary":
        render_summary(console, display_df, pivot_cols)

    console.print()
