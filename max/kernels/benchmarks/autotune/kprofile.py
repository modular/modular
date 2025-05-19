# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


import ast
import pickle
import sys
from dataclasses import dataclass

import click
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

LINE = 80 * "-"


def spec_to_dict(spec):
    """Convert kbench spec to dictionary"""
    # TODO: move this method to `kbench`
    spec_split = spec.split("/")
    d = {}
    d["name"] = spec_split[0]
    for x in spec_split[1:]:
        k, v = x.split("=")
        k = k.strip("$")
        d[k] = v
    return d


def specs_to_df(specs):
    ds = [spec_to_dict(x) for x in specs]
    df = pd.DataFrame(ds)
    return df


def extract_pivots(x_labels):
    df = specs_to_df(x_labels)
    pivot_columns = []
    for c in df.columns:
        if len(set(df[c])) > 1:
            pivot_columns.append(c)

    # set(df.columns)-set(pivot_columns)
    non_pivot_columns = [c for c in df.columns if c not in pivot_columns]
    return pivot_columns, non_pivot_columns


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def dump_yaml(obj, out_path):
    with open(out_path, "w") as f:
        yaml.dump(obj, f, sort_keys=False)

    # TODO: add this as a separate option, probably dict->yaml-str
    yaml.dump(obj, sys.stdout, sort_keys=False)


def top_idx(x, top_percentage=0.05):
    # calculate the threshold to pick top_percentage of the results
    threshold = (top_percentage + (np.min(x) / np.max(x))) * np.max(x)
    return x.where(x < threshold).dropna().index


def replace_vals_snippet(p_spec, snippet_path, output_path):
    with open(snippet_path) as f:
        c_init = f.read()

    c = c_init[:]
    for k, v in p_spec.items():
        print(f"Replacing [{k}]:[{v}]")
        c = c.replace(f"[@{k}]", v)

    output_path = f"{output_path}"
    with open(output_path, "w") as f:
        f.write(c)
    print(f"wrote results to [{output_path}]")


def find_common_params(subset):
    spec_list = []
    for index, row in subset.iterrows():
        p = spec_to_dict(row["spec"])
        spec_list.append(pd.DataFrame([p]))
    merged_specs = pd.concat(spec_list, axis=0, ignore_index=True)

    spec = {}
    for c in merged_specs.columns:
        frequent_val = merged_specs[c].value_counts().idxmax()
        spec[c] = frequent_val

    return spec


def df_to_console_table(
    df, col_style={}, header_style="bold blue", index=False
):
    console = Console()
    table = Table(show_header=True, header_style=header_style)

    for c in df.columns:
        style = col_style.get(c, None)
        table.add_column(c, justify="left", style=style)

    def wrap(x):
        return "\n".join(x.split("/"))

    for row in df.itertuples(index=index):
        l = [wrap(str(x)) for x in list(row)]
        table.add_row(*l)
        table.add_section()
    console.print(table)


@dataclass(repr=True)
class KbenchPKL:
    merged_df: pd.DataFrame
    tune_df: pd.DataFrame
    pkl_data: dict
    # TODO: rework this
    met_col: str

    def __init__(self, pickle_path):
        self.merged_df, self.tune_df, self.pkl_data, self.met_col = (
            self.unpack_pkl(pickle_path=pickle_path)
        )

    @staticmethod
    def load(path) -> dict:
        f = load_pickle(path)
        for k in ["merged_df", "build_df"]:
            assert k in f.keys()
        return f

    @staticmethod
    def unpack_pkl(
        pickle_path,
        select_cols=["mesh_idx", "met (ms)", "Arithmetic (GFLOPS/s)", "spec"],
    ):
        pkl_data = KbenchPKL.load(pickle_path)
        merged_df = pkl_data["merged_df"]
        met_col = "met (ms)"
        assert met_col in merged_df.columns

        # Select subset of columns from merged_df that are in 'select_cols'
        if select_cols:
            common_cols = [
                c for c in select_cols if c in list(merged_df.columns)
            ]
            assert len(common_cols), "Error: could not find any common columns!"
            merged_df = merged_df[common_cols]

        tune_df = merged_df.sort_values([met_col], ascending=True)
        # Add met_ratio: current mean-exec-time/best mean-exec-time
        top_spec = tune_df.iloc[0]
        tune_df["met_ratio"] = tune_df[met_col].div(top_spec[met_col])

        return merged_df, tune_df, pkl_data, met_col


def df_round_floats(df, prec=3):
    "Round values in dataframe to specified precision"
    for c in df.columns:
        if df.dtypes[c] in (np.float64, np.float32):
            df = df.round({c: prec})
    return df


def profile_results(
    pickle_path,
    snippet_path,
    output_path="output.mojo",
    top_percentage=0.0,
    ratio=False,
    head=-1,
    tail=-1,
    pivots: list[str] = [],
    verbose=False,
):
    pkl = KbenchPKL(pickle_path=pickle_path)
    merged_df, tune_df, pkl_data, met_col = (
        pkl.merged_df,
        pkl.tune_df,
        pkl.pkl_data,
        pkl.met_col,
    )
    print(f"- num entries: {len(merged_df)}")

    if top_percentage:
        idx = top_idx(tune_df[met_col], top_percentage=top_percentage)
        subset = merged_df.iloc[idx]
        if verbose:
            print(f"common subset in [{top_percentage}]%")
            print(subset.to_string(index=False))
        print(LINE)

        # form a spec with most frequent values among top percentage of picks
        spec = find_common_params(subset)
    else:
        # get the spec from the first pick
        print(f"- best idx: {tune_df.iloc[0]['mesh_idx']}")
        print(f"- worst_met (ms): {tune_df.iloc[-1]['met (ms)']}")
        print(f"- best_met (ms) : {tune_df.iloc[0]['met (ms)']}")

        # TODO: revise this part to automatically extract pivots if not provided!
        spec = spec_to_dict(tune_df.iloc[0]["spec"])
        shape = "/".join([f"{p}={spec[p]}" for p in pivots])
        print(
            f"- met_worst/met_best ratio: {tune_df.iloc[-1]['met_ratio']:.4f} [shape: {shape}]"
        )
        print(LINE)

    if ratio:
        tune_df = df_round_floats(tune_df, prec=3)
        df_to_console_table(
            tune_df,
            col_style={"met_ratio": "bold green"},
        )
        print(LINE)

    if head > 0:
        df_to_console_table(tune_df[:head])

    if tail > 0:
        df_to_console_table(tune_df[-1 * tail :])

    if verbose:
        # TODO: select based on pivots and cleanup the view
        print(LINE)
        merged_df = df_round_floats(merged_df, prec=3)
        df_to_console_table(merged_df)

    print("[Best Spec]\n")

    out_yaml_path = "result.yaml"
    dump_yaml(
        {
            "name": str(pkl_data.get("name", None)),
            "file": str(pkl_data.get("file", None)),
            "params": [spec],
        },
        out_yaml_path,
    )
    print(LINE)
    print(f"wrote best pick to [{out_yaml_path}]")

    if snippet_path:
        replace_vals_snippet(spec, snippet_path, output_path)
        print(LINE)


class ComplexParamList(click.Option):
    """Complext parameter list

    Example:
        --pivot=[M] --pivot=[N] --pivot=[K] is equivalent to --pivot=[M,N,K] and vice versa.
    """

    def type_cast_value(self, ctx, value_in):
        """DO NOT REMOVE this function, it is called from ctx in click."""
        p = []
        assert isinstance(value_in, list)
        for v in value_in:
            p.extend(self.parse(v))
        return p

    @staticmethod
    def parse(value) -> list:
        try:
            return ast.literal_eval(value)
        except:
            # case of [x, y, z, ...]
            if value.startswith("[") and value.endswith("]"):
                return value[1:-1].split(",")
            # case of single value x
            elif (
                not value.startswith("[")
                and not value.endswith("]")
                and "," not in value
            ):
                return [value]
            else:
                raise click.BadParameter(value)


help_str = "Profile kbench output pickle"


@click.command(help=help_str, no_args_is_help=True)
@click.option(
    "--output", "-o", "output_path", default=None, help="Path to output file."
)
@click.option(
    "--top",
    "-t",
    default=0.0,
    help="Form a new spec from frequent values of each param from top percent.",
    multiple=False,
)
@click.option(
    "--snippet",
    "-s",
    "snippet_path",
    default=None,
    help="Path to snippet to replace the parameters with values.",
    multiple=False,
)
@click.option(
    "--ratio",
    "-r",
    is_flag=True,
    default=False,
    help="Print the running time ratio of each entry to the best entry.",
)
@click.option(
    "--head",
    default=-1,
    help="The number of elements at head to print (sorted by running time).",
    multiple=False,
)
@click.option(
    "--tail",
    default=-1,
    help="The number of elements at tail to print (sorted by running time).",
    multiple=False,
)
@click.option(
    "--diff",
    "-d",
    is_flag=True,
    default=False,
    help="Show the difference between best running time of multiple pkl's, "
    "first one as baseline (preferably auto-tuning results)",
)
@click.option(
    "--pivot",
    "-p",
    cls=ComplexParamList,
    default=["M", "N", "K"],
    help="Specify the pivots to select the values.",
    multiple=True,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print all the (unsorted) entries from pkl.",
)
@click.argument("files", nargs=-1, type=click.UNPROCESSED)
def cli(
    files,
    output_path,
    top,
    snippet_path,
    ratio,
    head,
    tail,
    diff,
    pivot,
    verbose,
) -> bool:
    assert files

    if diff:
        assert len(files) > 1, (
            "Should provide at least two pkl's for --diff option."
        )
    else:
        assert len(files) == 1

    # TODO: rework
    for pickle_path in files:
        print(f"pickle_path: [{pickle_path}]")
    print(f"top_percentage: [{top}]")
    print(f"snippet_path: [{snippet_path}]")
    print(LINE)

    top_percentage = float(top) if top else 0
    if diff:
        if head == -1:
            head = 1
        print(f"diff on pivots {pivot}")
        print("--diff is currently not supported")
    else:
        profile_results(
            pickle_path=files[0],
            snippet_path=snippet_path,
            output_path="output.mojo",
            top_percentage=top_percentage,
            ratio=ratio,
            head=head,
            tail=tail,
            pivots=pivot,
            verbose=verbose,
        )
    return True


def main():
    cli()


if __name__ == "__main__":
    main()
