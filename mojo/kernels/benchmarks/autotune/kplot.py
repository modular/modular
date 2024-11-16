#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def draw_plot(
    x, y_list, y_names, x_title, y_title, extension, prefix="img", scale=1.0
):
    layout = go.Layout(
        autosize=False,
        width=1920,
        height=1080,
    )

    def plot_draw(xs, ys_list, name):
        fig = go.Figure(layout=layout)
        fig.update_layout(
            font_family="Courier New",
            # font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",
            font_size=16,
        )

        marker_color_sequence = [
            "#c1121f",  # red
            "#0077b6",  # blue
            "#a7c957",  # green
            "#ffc300",  # yellow
            "coral",
            "lightskyblue",
            "silver",
            "magneta",
        ]
        for i, ys in enumerate(ys_list):
            fig.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    name=y_names[i],
                    text=np.round(ys, 6),
                    textposition="outside",
                    marker_color=marker_color_sequence[i % 8],
                    showlegend=True,
                )
            )

        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        fig.update_layout(
            barmode="group",
            xaxis_tickangle=-0,
            bargap=0.30,
            bargroupgap=0.2,
            xaxis_title=x_title,
            yaxis_title=y_title,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="#fffcf2",
            paper_bgcolor="#fffcf2",
            # paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black")

        # fig.show()
        # plotly.offline.plot(fig, filename='lifeExp.html')
        fig.write_image(name, scale=scale)
        print(f"- added [{name}]")

    def wrap(s, d=10):
        out = []
        for i in range(0, len(s), d):
            out.append(s[i : i + d])
        return "<br>".join(out)

    x = [wrap(x[i]) for i in range(len(x))]

    delta = 10
    for i in range(0, len(x), delta):
        plot_draw(
            x[i : i + delta],
            y_list[:, i : i + delta],
            name=f"{prefix}_{i}.{extension}",
        )


help_str = "Drawing bar plots for kbench results"


## TODO: important to define which one is the baseline
## TODO: replace matplotlib with plotly
def parse_and_plot(
    path_list,
    name_list,
    target_col_idx=1,
    compare=False,
    extension="png",
    force=False,
    prefix="img",
    scale=1.0,
):
    tables = []
    for path in path_list:
        df = pd.read_csv(path)
        tables.append(df)

    # verify all the col names in two datasets are identical.
    check_entry_names = False if force else True
    ref_col_idx = 0
    for t in tables[1:]:
        # identical column names
        np.testing.assert_equal(list(tables[0].columns), list(t.columns))
        # identical shapes
        np.testing.assert_equal(tables[0].shape, t.shape)
        # identical entry names (column ref_col_idx)
        if check_entry_names:
            np.testing.assert_equal(
                list(tables[0].iloc[:, ref_col_idx]),
                list(t.iloc[:, ref_col_idx]),
            )
    print("ENTRIES HAVE MATCHING REFERENCE COLUMNS")

    target_col_name = tables[0].columns[target_col_idx]
    y_list = []
    y_names = []

    if compare:
        ratio_df = pd.DataFrame()
        ratio_df.insert(0, tables[0].columns[0], tables[0].iloc[:, 0])

        for i, t in enumerate(tables[1:]):
            ratio = (
                t.iloc[:, target_col_idx] / tables[0].iloc[:, target_col_idx]
            )
            name = f"{target_col_name} [{name_list[i+1]}/{name_list[0]}]"
            ratio_df.insert(len(ratio_df.columns), name, ratio)
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


@click.command(help=help_str, no_args_is_help=True)
@click.option(
    "--label",
    "label_list",
    help="List of corresponding labels for CSV files (can have multiple ones).",
    multiple=True,
)
@click.option(
    "--output",
    "-o",
    "output_prefix",
    default=None,
    help="Prefix for output file.",
)
@click.option(
    "--plot-col",
    "-p",
    "plot_col",
    default=1,
    type=click.INT,
    help="Plot column.",  # TODO: complete docstring
)
@click.option(
    "--compare",
    "-c",
    "compare",
    is_flag=True,
    default=False,
    help="Compare csv's, using the first one as the baseline.",  # TODO: complete docstring
)
@click.option(
    "--extension",
    "-x",
    "extension",
    default="png",
    type=click.STRING,
    help="output extension",  # TODO: complete docstring
)
@click.option(
    "--scale",
    "-s",
    "scale",
    default=1.0,
    type=click.FLOAT,
    help="scale for the output, default=1, use > 1 (~6) for better quality",  # TODO: complete docstring
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force matching input sets.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Verbose printing."
)
@click.argument("csv_files", nargs=-1, type=click.UNPROCESSED)
def cli(
    # csv_path_list,
    csv_files: click.UNPROCESSED,
    label_list,
    output_prefix,
    plot_col,
    compare,
    extension,
    scale,
    force,
    verbose,
):
    csv_path_list = [Path(file).resolve() for file in csv_files]
    num_csv = len(csv_path_list)

    if len(label_list) < num_csv:
        label_list = list(label_list)
        for i in range(len(label_list), num_csv):
            label_list.append(csv_path_list[i].name)

    # Columns should be: "name" (0), "met (ms)" (1), "iters" (2)
    assert plot_col in [1, 2]

    prefix = "img" if not output_prefix else output_prefix
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


def main():
    cli()  # type: ignore


if __name__ == "__main__":
    main()
