#!/usr/bin/env python3
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
#
# This file will extract a Mojo program from all the cells in a notebook. It
# converts a notebook into a .mojo file. It is meant to be temporary until we
# get a true Jupyter kernel for Mojo.
#
# ===----------------------------------------------------------------------=== #

import argparse
import json
from pathlib import Path

try:
    import lit.TestRunner
    from lit.formats import FileBasedTest
except Exception:

    class FileBasedTest:
        pass


def indent(content: list[str]) -> list[str]:
    indented = []
    for line in content:
        if line.strip("\t ") == "":
            continue
        indented.append("  " + line)
    return indented


def is_cell_toplevel(cell_lines: list[str]) -> bool:
    """Determine whether the contents of the cell should be inside 'main' or
    at the top level."""
    for line in cell_lines:
        if line.startswith(("def", "fn", "alias", "struct")):
            return True
    return False


def replace_markers(cell_lines: list[str]) -> list[str]:
    """Replace '#|' filecheck markers with just '#'."""
    replaced = []
    for line in cell_lines:
        if line.startswith("#|"):
            line = "#" + line.lstrip("#|")
        replaced.append(line.rstrip())
    return replaced


def split_imports(contents: list[str]) -> tuple[list[str], list[str]]:
    """Split the code in the cell into imports and non-imports.
    We assume that there is no non-import code between import statements.
    """
    imports = contents
    non_imports = []
    while len(imports) > 0:
        line = imports.pop()
        if line.startswith(("from", "import")):
            imports.append(line)
            break
        non_imports.append(line)
    non_imports = non_imports[::-1]
    return imports, non_imports


def extract_code_cells(path: Path) -> str:
    """Extracts all the code cells and generate an equivalent standalone program."""
    top_level_lines = []
    main_body = []
    with path.open("r") as f:
        js = json.load(f)
        for cell in js["cells"]:
            if cell["cell_type"] == "code" and cell["source"]:
                contents = cell["source"]

                # Skip python cells:
                if any("%%python" in line for line in contents):
                    continue

                contents = replace_markers(contents)

                if is_cell_toplevel(contents):
                    top_level_lines += [
                        "# ===------------------------------  CELL "
                        " ----------------------=== #"
                    ] + contents
                else:
                    imports, after_imports = split_imports(contents)
                    top_level_lines += imports
                    main_body += [
                        "# ===------------------------------  CELL "
                        " ----------------------=== #"
                    ] + after_imports

    main_body = indent(main_body)
    lines = top_level_lines + ["\ndef main():"] + main_body

    return "\n".join(lines)


def import_notebook_to_mojo(path: Path) -> str:
    """Imports a notebook to a mojo file

    Args:
        path (Path): the path to the notebook

    Returns:
        str: a mojo program.
    """
    return extract_code_cells(path)


class TestNotebook(FileBasedTest):
    def __init__(
        self,
        exec_root=None,
    ):
        self.exec_root = Path(exec_root)  # type: ignore

    def execute(self, test, litConfig):
        notebook_source_path = Path(test.getSourcePath())
        mojo_output_path = (
            self.exec_root / notebook_source_path.with_suffix(".mojo").name
        )

        mojo_output_path.write_text(
            import_notebook_to_mojo(notebook_source_path)
        )

        extra_substitutions = [("%s", str(mojo_output_path))]

        return lit.TestRunner.executeShTest(
            test,
            litConfig,
            False,  # execute_external
            extra_substitutions=extra_substitutions,
            preamble_commands=["mojo %s --enable-search| FileCheck %s"],
        )


def main():
    parser = argparse.ArgumentParser(
        prog="extract_mojo_program",
        description="Extracts a mojo program from an input notebook",
    )

    parser.add_argument(
        "-o", "--output", type=Path, help="the output file name"
    )
    parser.add_argument(
        "input", metavar="input", type=Path, nargs=1, help="the input file name"
    )

    args = parser.parse_args()

    input_file = args.input[0]

    if not input_file.exists():
        print(f"the input path '{input_file}' does not exist")
        exit(1)

    txt = import_notebook_to_mojo(input_file)
    if args.output:
        args.output.write_text(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()
