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
# Perform processing on Jupyter notebooks to prepare them for release

import argparse
import json
from pathlib import Path

import yaml


def cell_contains_remove_for_website(cell):
    return any(
        line_contains_remove_for_website(line) for line in cell["source"]
    )


def line_contains_remove_for_website(line):
    return "[//]: # REMOVE_FOR_WEBSITE" in line


def process_notebook(input_file: Path, website_file: Path, github_file: Path):
    with open(input_file) as infile:
        content = infile.read()

    # Strip out the CHECK lines first thing
    content = "\n".join(
        list(filter(lambda line: "#| CHECK" not in line, content.splitlines()))
    )

    # For the version of notebooks going to the docs website,
    # find cells with "REMOVE_FOR_WEBSITE" and strip the entire cell
    website_content = json.loads(content)
    website_content["cells"] = list(
        filter(
            lambda cell: not cell_contains_remove_for_website(cell),
            website_content["cells"],
        )
    )

    with open(website_file, "w") as outfile:
        json.dump(website_content, outfile, ensure_ascii=False, indent=2)

    # For the version of notebooks going to GitHub and the Playground,
    # just remove the comment for "REMOVE_FOR_WEBSITE" (leaving the cell intact)
    github_content = json.loads(content)
    for cell in github_content["cells"]:
        cell["source"] = list(
            filter(
                lambda line: not line_contains_remove_for_website(line),
                cell["source"],
            )
        )

    # Get the first 'raw' cell. That will be the front matter.
    front_matter = None
    for cell_index in range(len(github_content["cells"])):
        if github_content["cells"][cell_index]["cell_type"] == "raw":
            front_matter = github_content["cells"][cell_index]["source"]
            break

    # If we have front matter, check if it's valid YAML.
    if front_matter is not None:
        # Expecting the content to be surrounded by lines "---\n" and "---"
        if front_matter[0] == "---\n" and front_matter[-1] == "---":
            # Convert list to string, omit ends
            front_matter = "".join(front_matter[1:-1])
            if yaml.safe_load(front_matter):
                # OK great - if it is in fact YAML, we're good to remove it.
                del github_content["cells"][cell_index]

    with open(github_file, "w") as outfile:
        json.dump(github_content, outfile, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        prog="preprocess_notebooks",
        description="Preprocess a Jupyter notebook to prepare it for release",
    )
    parser.add_argument("input", type=Path, help="the input file name")
    parser.add_argument(
        "website_file", type=Path, help="the output file name for the website"
    )
    parser.add_argument(
        "github_file", type=Path, help="the output file name for Github"
    )
    args = parser.parse_args()

    input_file = args.input

    if not input_file.exists():
        print(f"the input path '{input_file}' does not exist")
        exit(1)

    process_notebook(input_file, args.website_file, args.github_file)


if __name__ == "__main__":
    main()
