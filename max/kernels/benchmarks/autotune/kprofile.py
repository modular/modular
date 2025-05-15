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


import pickle
import sys

import click
import numpy as np
import pandas as pd
import yaml

LINE = 80 * "-"


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def dump_yaml(obj, out_path):
    with open(out_path, "w") as f:
        yaml.dump(obj, f, sort_keys=False)

    # TODO: add this as a separate option, probably dict->yaml-str
    yaml.dump(obj, sys.stdout, sort_keys=False)


def parse_params(spec):
    s = spec.split("/")
    d = {}
    d["name"] = s[0]
    for item in s[1:]:
        k, v = item.split("=")
        d[k] = v
    return d


def read_kbench_pickle(path):
    f = load_pickle(path)
    for k in ["merged_df", "build_df"]:
        assert k in f.keys()
    return f


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
        p = parse_params(row["spec"])
        spec_list.append(pd.DataFrame([p]))
    merged_specs = pd.concat(spec_list, axis=0, ignore_index=True)

    spec = {}
    for c in merged_specs.columns:
        frequent_val = merged_specs[c].value_counts().idxmax()
        spec[c] = frequent_val

    return spec


def profile_results(
    pickle_path,
    snippet_path,
    output_path="output.mojo",
    top_percentage=0.0,
    ratio=False,
    head=-1,
    tail=-1,
    verbose=False,
):
    kernel_data = read_kbench_pickle(pickle_path)
    merged_df = kernel_data["merged_df"]
    met_col = merged_df.columns[2]
    tune_df = merged_df.sort_values([met_col], ascending=True)
    top_spec = tune_df.iloc[0]
    tune_df["met_ratio"] = tune_df["met (ms)"].div(top_spec["met (ms)"])
    print(f"- num entries: {len(merged_df)}")

    if top_percentage:
        idx = top_idx(tune_df["met (ms)"], top_percentage=top_percentage)
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
        print(f"-  best_met (ms): {tune_df.iloc[0]['met (ms)']}")
        print(
            f"- met_worst/met_best ratio: {tune_df.iloc[-1]['met_ratio']:.4f}"
        )
        spec = parse_params(tune_df.iloc[0]["spec"])
        print(LINE)

    if ratio:
        print(
            tune_df[["mesh_idx", "met (ms)", "met_ratio"]].to_string(
                index=False
            )
        )
        print(LINE)

    if head > 0:
        print(tune_df[:head].to_string(index=False))
        print(LINE)

    if tail > 0:
        print(tune_df[-1 * tail :].to_string(index=False))
        print(LINE)

    if verbose:
        print(merged_df[:].to_string(index=False))
        print(LINE)
        # TODO: install in pip and install_python_deps
        # from datascroller import scroll
        #     scroll(tune_df)

    print("[Best Spec]\n")

    out_yaml_path = "result.yaml"
    dump_yaml(
        {
            "name": kernel_data.get("name", None),
            "file": kernel_data.get("file", None),
            "params": [spec],
        },
        out_yaml_path,
    )
    print(LINE)
    print(f"wrote best pick to [{out_yaml_path}]")

    if snippet_path:
        replace_vals_snippet(spec, snippet_path, output_path)
        print(LINE)


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
    verbose,
) -> bool:
    assert files
    pickle_path = files[0]

    print(f"pickle_path: [{pickle_path}]")
    print(f"top_percentage: [{top}]")
    print(f"snippet_path: [{snippet_path}]")
    print(LINE)

    top_percentage = float(top) if top else 0
    profile_results(
        pickle_path,
        snippet_path=snippet_path,
        output_path="output.mojo",
        top_percentage=top_percentage,
        ratio=ratio,
        head=head,
        tail=tail,
        verbose=verbose,
    )
    return True


def main():
    cli()


if __name__ == "__main__":
    main()
