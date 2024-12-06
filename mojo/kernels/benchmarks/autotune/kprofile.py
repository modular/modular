# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pandas as pd
import pickle
import click

LINE = 80 * "-"


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


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
    with open(snippet_path, "r") as f:
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
    pickle_path, snippet_path, output_path="output.mojo", top_percentage=0.0
):
    f = read_kbench_pickle(pickle_path)

    merged_df = f["merged_df"]
    met_col = merged_df.columns[2]
    tune_df = merged_df.sort_values([met_col], ascending=True)

    if top_percentage:
        idx = top_idx(tune_df["met (ms)"], top_percentage=top_percentage)
        subset = merged_df.iloc[idx]
        print(subset.to_string())
        print(LINE)

        # form a spec with most frequent values among top percentage of picks
        spec = find_common_params(subset)
    else:
        # get the spec from the first pick
        spec = parse_params(tune_df.iloc[0]["spec"])

    print(f"selected spec [top={top_percentage}]:\n{spec}")
    print(LINE)

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
    "--verbose", "-v", is_flag=True, default=False, help="Verbose printing."
)
@click.argument("files", nargs=-1, type=click.UNPROCESSED)
def cli(
    files: click.UNPROCESSED,
    output_path,
    top,
    snippet_path,
    verbose,
) -> bool:
    assert files
    pickle_path = files[0]

    print(f"pick_path: [{pickle_path}]")
    print(f"top_percentage: [{top}]")
    print(f"snippet_path: [{snippet_path}]")
    print(LINE)

    top_percentage = float(top) if top else 0
    profile_results(
        pickle_path,
        snippet_path=snippet_path,
        output_path="output.mojo",
        top_percentage=top_percentage,
    )
    return True


def main():
    cli()


if __name__ == "__main__":
    main()
