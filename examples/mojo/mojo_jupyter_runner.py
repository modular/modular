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

import json
from pathlib import Path

import lit.TestRunner
from lit.formats import FileBasedTest


def import_file_check_directives(path: Path) -> list[str]:
    """Imports FileCheck directives

    Args:
        path (Path): the output path

    Returns:
        str: the directives
    """
    directives = []
    with path.open("r") as f:
        js = json.load(f)

        code_cells = filter(
            lambda cell: cell["cell_type"] == "code" and cell["source"],
            js["cells"],
        )

        for cell in code_cells:
            contents = cell["source"]
            for line_ in contents:
                line = line_.strip()
                if line.startswith("#|"):
                    directives.append("#" + line.lstrip("#|"))
    return directives


class TestNotebook(FileBasedTest):
    """
    llvm-lit test utility for testing the FileCheck directives of a mojo
    notebook.
    """

    def __init__(
        self,
        config,
    ):
        self.exec_root = Path(config.test_exec_root)  # type: ignore

    def execute(self, test, litConfig):
        if test.getSourcePath().endswith(
            ".mojo"
        ) or test.getSourcePath().endswith(".🔥"):
            return lit.TestRunner.executeShTest(
                test,
                litConfig,
                False,  # execute_external
                preamble_commands=["mojo"],
            )

        notebook_source_path = Path(test.getSourcePath())
        directives_output_path = (
            self.exec_root / notebook_source_path.with_suffix(".checks").name
        ).absolute()

        directives = import_file_check_directives(notebook_source_path)
        directives_output_path.write_text("\n".join(directives) + "\n")

        extra_substitutions = [
            ("%s", str(directives_output_path)),
        ]
        lldb_init = Path("utils/lit-lldb-init.in").absolute()
        check_cmd = f" {lldb_init} {notebook_source_path}"
        if len(directives) > 0:
            check_cmd += "| FileCheck %s"

        return lit.TestRunner.executeShTest(
            test,
            litConfig,
            False,  # execute_external
            extra_substitutions=extra_substitutions,
            preamble_commands=[
                "mojo-jupyter-executor --lldb-init-file" + check_cmd
            ],
        )
