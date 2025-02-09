#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path
from typing import List, Optional, Sequence

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go

MARKER_COLORS = [
    "#c1121f",  # red
    "#0077b6",  # blue
    "#a7c957",  # green
    "#ffc300",  # yellow
    "coral",
    "lightskyblue",
    "silver",
    "magneta",
]


def draw_plot(
    x: Sequence[str],
    y_list: np.ndarray,
    y_names: Sequence[str],
    x_title: str,
    y_title: str,
    extension: str,
    prefix: str = "img",
    scale: float = 1.0,
) -> None:
    """Draw bar plots for benchmark results.

    Args:
        x: X-axis labels
        y_list: 2D array of Y values
        y_names: Names for each Y series
        x_title: X-axis title
        y_title: Y-axis title
        extension: File extension for output images
        prefix: Prefix for output filenames
        scale: Scaling factor for output image quality
    """
    layout = go.Layout(autosize=False, width=1920, height=1080)

    def plot_draw(xs: Sequence[str], ys_list: np.ndarray, name: str) -> None:
        fig = go.Figure(layout=layout)
        fig.update_layout(
            font_family="Courier New",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",
            font_size=16,
        )

        for i, ys in enumerate(ys_list):
            fig.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    name=y_names[i],
                    text=np.round(ys, 6),
                    textposition="outside",
                    marker_color=MARKER_COLORS[i % len(MARKER_COLORS)],
                    showlegend=True,
                )
            )

        fig.update_layout(
            barmode="group",
            xaxis_tickangle=0,
            bargap=0.30,
            bargroupgap=0.2,
            xaxis_title=x_title,
            yaxis_title=y_title,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="#fffcf2",
            paper_bgcolor="#fffcf2",
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black")

        fig.write_image(name, scale=scale)
        print(f"- added [{name}]")

    def wrap(s: str, d: int = 10) -> str:
        return "<br>".join(s[i : i + d] for i in range(0, len(s), d))

    x = [wrap(x[i]) for i in range(len(x))]

    delta = 10
    for i in range(0, len(x), delta):
        plot_draw(
            x[i : i + delta],
            y_list[:, i : i + delta],
            name=f"{prefix}_{i}.{extension}",
        )


def parse_and_plot(
    path_list: List[Path],
    name_list: List[str],
    target_col_idx: int = 1,
    compare: bool = False,
    extension: str = "png",
    force: bool = False,
    prefix: str = "img",
    scale: float = 1.0,
) -> None:
    """Parse CSV files and generate plots.

    Args:
        path_list: List of paths to CSV files
        name_list: List of names for each CSV
        target_col_idx: Index of column to plot
        compare: Whether to compare against baseline
        extension: File extension for output images
        force: Skip input validation if True
        prefix: Prefix for output filenames
        scale: Scaling factor for output image quality
    """
    tables = [pd.read_csv(path) for path in path_list]

    if not force:
        for t in tables[1:]:
            np.testing.assert_equal(list(tables[0].columns), list(t.columns))
            np.testing.assert_equal(tables[0].shape, t.shape)
            np.testing.assert_equal(
                list(tables[0].iloc[:, 0]),
                list(t.iloc[:, 0]),
            )
    print("ENTRIES HAVE MATCHING REFERENCE COLUMNS")

    target_col_name = tables[0].columns[target_col_idx]
    y_list, y_names = [], []

    if compare:
        ratio_df = pd.DataFrame({tables[0].columns[0]: tables[0].iloc[:, 0]})
        for i, t in enumerate(tables[1:]):
            ratio = (
                t.iloc[:, target_col_idx] / tables[0].iloc[:, target_col_idx]
            )
            name = f"{target_col_name} [{name_list[i + 1]}/{name_list[0]}]"
            ratio_df[name] = ratio
            y_list.append(ratio)
            y_names.append(name)
    else:
        for i, t in enumerate(tables):
            name = f"{target_col_name} [{name_list[i]}]"
            y_list.append(t.iloc[:, target_col_idx])
            y_names.append(name)

    x_labels = np.array(tables[0].iloc[:, 0])
    y_list = np.array(y_list)

    draw_plot(
        x_labels,
        y_list,
        y_names,
        x_title="name",
        y_title=target_col_name,
        extension=extension,
        prefix=prefix,
        scale=scale,
    )


@click.command(
    help="Drawing bar plots for kbench results", no_args_is_help=True
)
@click.option(
    "--label",
    "label_list",
    multiple=True,
    help="List of corresponding labels for CSV files",
)
@click.option(
    "--output",
    "-o",
    "output_prefix",
    default=None,
    help="Prefix for output file",
)
@click.option(
    "--plot-col", "-p", default=1, type=click.INT, help="Plot column index"
)
@click.option(
    "--compare",
    "-c",
    is_flag=True,
    default=False,
    help="Compare CSVs using first as baseline",
)
@click.option(
    "--extension",
    "-x",
    default="png",
    type=click.STRING,
    help="Output file extension",
)
@click.option(
    "--scale",
    "-s",
    default=1.0,
    type=click.FLOAT,
    help="Output image scaling factor",
)
@click.option(
    "--force", "-f", is_flag=True, default=False, help="Skip input validation"
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose output"
)
@click.argument("csv_files", nargs=-1, type=click.UNPROCESSED)
def cli(
    csv_files: click.UNPROCESSED,
    label_list: List[str],
    output_prefix: Optional[str],
    plot_col: int,
    compare: bool,
    extension: str,
    scale: float,
    force: bool,
    verbose: bool,
) -> None:
    """CLI entry point for plotting benchmark results."""
    csv_path_list = [Path(file).resolve() for file in csv_files]
    num_csv = len(csv_path_list)

    if len(label_list) < num_csv:
        label_list = list(label_list)
        label_list.extend(
            csv_path_list[i].name for i in range(len(label_list), num_csv)
        )

    assert plot_col in {1, 2}, "Plot column must be 1 or 2"

    prefix = output_prefix or "img"
    parse_and_plot(
        csv_path_list,
        label_list,
        target_col_idx=plot_col,
        compare=compare,
        extension=extension,
        force=force,
        prefix=prefix,
        scale=scale,
    )


def main() -> None:
    cli()  # type: ignore


if __name__ == "__main__":
    main()
