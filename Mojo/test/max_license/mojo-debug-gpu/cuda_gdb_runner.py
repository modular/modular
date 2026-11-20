#!/usr/bin/env python3
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

import sys
import tempfile
from pathlib import Path

import lit.TestRunner
from lit.formats import FileBasedTest


class CudaGDBTest(FileBasedTest):
    """
    A custom llvm-lit executor that implements the cool features mentioned in
    the `README.md` of this directory.
    """

    def __init__(
        self,
        config,  # noqa: ANN001
    ) -> None:
        self.exec_root = Path(config.test_exec_root)  # type: ignore

    def get_breakpoint_substitutions(
        self, input_source_path: Path
    ) -> list[tuple[str, str]]:
        extra_substitutions: list[tuple[str, str]] = []
        with open(input_source_path) as input_file:
            for index, line in enumerate(input_file.readlines()):
                needle = "# breakpoint"
                if (pos := line.find(needle)) != -1:
                    name = line[pos + len(needle) :].strip()
                    line_number = index + 1
                    extra_substitutions.append(
                        (f"%breakpoint{name}:line", str(line_number))
                    )
                    extra_substitutions.append(
                        (
                            f"%breakpoint{name}:location",
                            f"{input_source_path.name}:{line_number}",
                        )
                    )
        return extra_substitutions

    def get_gdb_commands(self, input_source_path: Path) -> list[str]:
        gdb_commands = []
        with open(input_source_path) as input_file:
            for line in input_file:
                gdb_comment = "# GDB-COMMAND: "
                if line.startswith(gdb_comment):
                    gdb_commands.append(line[len(gdb_comment) :].strip())
        return gdb_commands

    def execute(self, test, litConfig):  # noqa: ANN001, ANN201
        input_source_path = Path(test.getSourcePath())
        if "%mojo-debug-cuda" not in input_source_path.read_text():
            return lit.TestRunner.executeShTest(
                test,
                litConfig,
                False,  # execute_external
            )

        # In order to run multiple cuda-gdb sessions at the same time, the
        # `TMPDIR` env var must be set to a unique place per test file.
        cuda_gdb_tmp_dir = Path(tempfile.gettempdir()) / (
            input_source_path.stem
        )
        cuda_gdb_tmp_dir.mkdir(parents=True, exist_ok=True)

        gdb_commands = self.get_gdb_commands(input_source_path)

        cuda_gdb_path = None
        for binary in ["cuda-gdb-minimal", "cuda-gdb"]:
            for version in [
                "12.5",
                "12.4",
                "12.3",
                "12.2",
                "12.1",
                "12.0",
                "12",
            ]:
                candidate = Path(f"/usr/local/cuda-{version}/bin/{binary}")
                if candidate.exists():
                    cuda_gdb_path = candidate
                    break
            if cuda_gdb_path is not None:
                break

        if cuda_gdb_path is None:
            # The REQUIRES/UNSUPPORTED tags have not been parsed yet, so if
            # we are on an unsupported platform like AMD we can't throw an
            # exception here. Set the path to 'false' so the test will
            # fail if cuda-gdb is missing, but pass if we later decide to skip.
            print("cuda-gdb not found!", file=sys.stderr)
            cuda_gdb_path = "false"

        cuda_gdb_command = (
            "env CUDBG_USE_LEGACY_DEBUGGER=1 "
            f"TMPDIR={cuda_gdb_tmp_dir} "
            "mojo "
            "debug "
            "--cuda-gdb "
            f"--cuda-gdb-path={cuda_gdb_path} "
            "-X -ex -X 'set auto-load python-scripts off' "
            "-X -ex -X 'set index-cache enabled off' "
            "-X -ex -X 'set cuda break_on_launch application' "
            "-X -ex -X r"
        )

        for gdb_command in gdb_commands:
            cuda_gdb_command += f" -X -ex -X '{gdb_command}'"

        cuda_gdb_command += " -X --batch "

        breakpoint_substitutions = (
            *self.get_breakpoint_substitutions(input_source_path),
        )

        for pattern, change in breakpoint_substitutions:
            cuda_gdb_command = cuda_gdb_command.replace(pattern, change)

        extra_substitutions: list[tuple[str, str]] = [
            *breakpoint_substitutions,
            ("%mojo-debug-cuda", cuda_gdb_command),
        ]

        return lit.TestRunner.executeShTest(
            test,
            litConfig,
            False,  # execute_external
            extra_substitutions=extra_substitutions,
        )
