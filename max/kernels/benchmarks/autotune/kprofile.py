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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

LINE = 80 * "-"


HEADER = """
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
"""


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


def extract_pivots(x_labels, exclude=["name", "AUTOTUNING_MODE"]):
    df = specs_to_df(x_labels)
    valid_columns = []
    for c in list(df.columns):
        if c not in exclude:
            valid_columns.append(c)

    pivot_columns = []
    for c in valid_columns:
        if len(set(df[c])) > 1:
            pivot_columns.append(c)

    # set(df.columns)-set(pivot_columns)
    non_pivot_columns = [c for c in valid_columns if c not in pivot_columns]
    return pivot_columns, non_pivot_columns


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def dump_yaml(obj, out_path) -> None:
    with open(out_path, "w") as f:
        yaml.dump(obj, f, sort_keys=False)

    # TODO: add this as a separate option, probably dict->yaml-str
    yaml.dump(obj, sys.stdout, sort_keys=False)


def top_idx(x, top_percentage=0.05):
    # calculate the threshold to pick top_percentage of the results
    threshold = (top_percentage + (np.min(x) / np.max(x))) * np.max(x)
    return x.where(x < threshold).dropna().index


def replace_vals_snippet(p_spec, snippet_path) -> str:
    # TODO: raise an error if the parameter is not successfully replaced.
    with open(snippet_path) as f:
        c_init = f.read()

    c = c_init[:]
    for k, v in p_spec.items():
        print(f"Replacing [{k}]:[{v}]")
        c = c.replace(f"[@{k}]", v)
    return c


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
) -> None:
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


# TODO: could this be refactored into a kbench class?
@dataclass(frozen=True)
class TuningSpec:
    name: str = ""
    file: Path = Path("")
    params: list[dict] = field(default_factory=list)
    pkl_path: Path = Path()
    git_sha: str = ""
    datetime: str = ""


@dataclass(repr=True)
class KbenchPKL:
    merged_df: pd.DataFrame
    tune_df: pd.DataFrame
    pkl_data: dict
    metric: str

    def __init__(self, pickle_path, metric: str) -> None:
        self.pkl_data = KbenchPKL.load(pickle_path)
        self.merged_df = self.pkl_data["merged_df"].drop(
            ["name", "iters"], axis=1
        )
        # Finding the appropriate metric
        cols = list(self.merged_df.columns)
        valid_metric = metric in cols
        if not valid_metric:
            for c in cols:
                if c.lower().startswith(metric):
                    metric = c
                    valid_metric = True
                    break
        assert valid_metric, f"ERROR: metric [{metric}] is not valid!"
        self.metric = metric
        assert pd.api.types.is_numeric_dtype(self.merged_df[metric]), (
            f"ERROR: metric [{metric}] is not numeric!"
        )
        # Setting the sort order based on the metric
        # TODO: move this dict to a global variable or singleton.
        ascending_sort = {
            "met (ms)": True,  # lower is better
            "throughput (GElems/s)": False,  # higher is better
            "DateMovement (GB/s)": False,  # higher is better
            "Arithmetic (GFLOPS/s)": False,  # higher is better
            "TheoreticalArithmetic (GFLOPS/s)": False,  # higher is better
        }[metric]

        # Set tune_df based on sort-order and merged_df
        tune_df = self.merged_df.sort_values([metric], ascending=ascending_sort)
        best_spec = tune_df.iloc[0]
        # Add ratio: current metric/best metric
        tune_df["ratio"] = tune_df[metric].div(best_spec[metric])

        self.tune_df = tune_df

    @staticmethod
    def load(path) -> dict:
        f = load_pickle(path)
        for k in ["merged_df", "build_df"]:
            assert k in f.keys()
        return f


def df_round_floats(df, prec=3):
    "Round values in dataframe to specified precision"
    for c in df.columns:
        if df.dtypes[c] in (np.float64, np.float32):
            df = df.round({c: prec})
    return df


def profile_results(
    pickle_path,
    top_percentage=0.0,
    ratio=False,
    head=-1,
    tail=-1,
    metric: str = "met (ms)",
    pivots: list[str] = [],
    verbose=False,
) -> Optional[TuningSpec]:
    try:
        pkl = KbenchPKL(pickle_path=pickle_path, metric=metric)
    except:
        print(f"Invalid pkl [{pickle_path}]")
        return None
    merged_df, tune_df, pkl_data = (
        pkl.merged_df,
        pkl.tune_df,
        pkl.pkl_data,
    )
    print(f"- num entries: {len(merged_df)}")

    if top_percentage:
        idx = top_idx(tune_df[pkl.metric], top_percentage=top_percentage)
        subset = merged_df.iloc[idx]
        if verbose:
            print(f"common subset in [{top_percentage}]%")
            print(subset.to_string(index=False))
        print(LINE)

        # form a spec with most frequent values among top percentage of picks
        spec = find_common_params(subset)
    else:
        prec = 3
        # get the spec from the first pick
        print(f"- best idx: {tune_df.iloc[0]['mesh_idx']}")
        print(f"- [{metric}] worst: {tune_df.iloc[-1][metric]:.{prec}f}")
        print(f"- [{metric}] best : {tune_df.iloc[0][metric]:.{prec}f}")

        # TODO: revise this part to automatically extract pivots if not provided!
        spec = spec_to_dict(tune_df.iloc[0]["spec"])

        if not pivots:
            _, pivots = extract_pivots(list(tune_df["spec"]))

        shape = "/".join([f"{p}={spec[p]}" for p in pivots])
        print(
            f"- [{metric}] worst/best ratio: {tune_df.iloc[-1]['ratio']:.2f}x [shape: {shape}]"
        )
        print(LINE)

    if ratio:
        tune_df = df_round_floats(tune_df, prec=3)
        df_to_console_table(
            tune_df,
            col_style={"ratio": "bold green"},
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
    spec = TuningSpec(
        name=str(pkl_data.get("name", None)),
        file=Path(pkl_data.get("file", Path())),
        params=[spec],
        pkl_path=pickle_path,
        git_sha=str(pkl_data.get("git-revision", None)),
        datetime=str(pkl_data.get("datetime", None)),
    )
    print(spec)
    print(LINE)
    return spec


def identical_pivot_values(x, y, pivots) -> bool:
    for p in pivots:
        if (
            (p not in x.keys())
            or (p not in y.keys())
            or (x.get(p, None) != y.get(p, None))
        ):
            print(f"ERROR: FAILED assert on pivot {p}: [{x[p]}] vs. [{y[p]}]")
            return False
    return True


def diff_baseline(
    files, metric: str, pivots: list = [], head: int = -1, verbose: bool = False
) -> None:
    base_pkl = KbenchPKL(files[0], metric=metric)
    metric = base_pkl.metric
    tune_df_base = base_pkl.tune_df

    base_dict = spec_to_dict(str(tune_df_base.iloc[0]["spec"]))
    if verbose:
        print("base-config", base_dict)
        print(LINE)

    if not pivots:
        _, non_pivot_columns = extract_pivots(list(tune_df_base["spec"]))
        pivots = non_pivot_columns

    shape = "/".join([f"{p}={base_dict[p]}" for p in pivots])
    for i, f in enumerate(files[1:]):
        tune_df = KbenchPKL(f, metric=metric).tune_df
        spec_dict = spec_to_dict(str(tune_df.iloc[0]["spec"]))
        assert base_dict["name"] == spec_dict["name"]
        assert identical_pivot_values(base_dict, spec_dict, pivots=pivots)

        if verbose:
            print(f"config [{i}]", spec_dict)
            print(LINE)

        base_metric = tune_df_base.iloc[0][metric]
        prec = 3

        num_rows = min(len(tune_df[:head]), len(tune_df))
        for j in range(num_rows):
            current_metric = tune_df.iloc[j][metric]
            # TODO: fix the following and check the accuracy issues.
            metric_ratio = round(current_metric / base_metric, prec)
            metric_speedup = round(base_metric / current_metric, prec)
            print(
                f"[{i}][shape:{shape}][metric:{metric}]: {metric_ratio:.{prec}f} (current/baseline = {current_metric:.{prec}f} / {base_metric:.{prec}f})"
            )
            print(
                f"[{i}][shape:{shape}][metric:{metric}]: {metric_speedup:.{prec}f} X (speedup: baseline/current = {base_metric:.{prec}f} / {current_metric:.{prec}f})"
            )
            d = {
                "shape": [shape],
                "metric": metric,
                "best_tuning_metric": [round(current_metric, prec)],
                "baseline_metric": [round(base_metric, prec)],
                "ratio": [metric_ratio],
                "speedup": [metric_speedup],
            }
            shape_path = shape.replace("/", "_")
            pd.DataFrame.from_dict(d).to_csv(f"{shape_path}.csv", index=False)
            print(LINE)


def codegen(specs: list[TuningSpec], snippet_path: Path, output_path: Path):
    details = []
    details += [HEADER]

    prefix = """alias configs = List("""
    suffix = """)"""

    details += [prefix]
    sep = ","
    for s in specs:
        config_str = replace_vals_snippet(s.params[0], snippet_path)
        print(LINE)
        details += [f"# Automatically generated from [{s.pkl_path}]"]
        details += [f"# date: [{s.datetime}]"]
        details += [f"# git-sha: [{s.git_sha}]"]
        details += [config_str + sep]
        details += [""]
    details += [suffix]

    with open(output_path, "w") as f:
        f.write("\n".join(details))
    import os

    os.system(f"mojo format {output_path}")
    print(f"wrote results to [{output_path}]")


def check_specs(specs: list[TuningSpec]):
    # TODO: check specs have the same tuning hash
    spec_list = [pd.DataFrame([s.params[0]]) for s in specs]
    merged_specs = pd.concat(spec_list, axis=0, ignore_index=True)
    print(merged_specs.to_string())

    key_cols = ["M", "N", "K"]
    key_cols_values = merged_specs[key_cols]
    key_cols_values_uniq = key_cols_values.drop_duplicates()
    if len(key_cols_values) == len(key_cols_values_uniq):
        print("PASS: UNIQUE KEY COLUMNS")
    else:
        # TODO: add check for finding the mismatched rows
        raise ValueError(
            "Found duplicates in specs! Make sure you pass each pkl once."
        )


class ComplexParamList(click.Option):
    """Complex parameter list
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
    "--output",
    "-o",
    "output_path",
    default="output.mojo",
    help="Path to output file.",
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
    "--metric",
    "-m",
    default="met (ms)",
    help="Specify the profiling metric (default='met (ms)').",
    multiple=False,
)
@click.option(
    "--pivots",
    "-p",
    cls=ComplexParamList,
    default=[],
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
    metric,
    pivots,
    verbose,
) -> bool:
    assert files

    if diff:
        assert len(files) > 1, "Provide at least two pkl's for --diff option."

    if verbose:
        print(f"top_percentage: [{top}]")
        print(f"snippet_path: [{snippet_path}]")
        print(LINE)

    top_percentage = float(top) if top else 0
    if diff:
        if head == -1:
            head = 1
        diff_baseline(
            files, metric=metric, pivots=pivots, head=head, verbose=verbose
        )
    else:
        specs = []
        invalid_pkls = []
        for pkl_path in files:
            top_spec = profile_results(
                pickle_path=pkl_path,
                top_percentage=top_percentage,
                ratio=ratio,
                head=head,
                tail=tail,
                metric=metric,
                pivots=pivots,
                verbose=verbose,
            )
            if top_spec:
                specs += [top_spec]
            else:
                invalid_pkls += [pkl_path]

        if snippet_path:
            check_specs(specs=specs)
            codegen(
                specs=specs, snippet_path=snippet_path, output_path=output_path
            )

        print(
            f"num invalid pkls: {len(invalid_pkls)} out of num-files: {len(files)}"
        )
        for i, pkl in enumerate(invalid_pkls):
            print(f"invalid pkl [{i}]: [{pkl}]")

        print(LINE)
    return True


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
