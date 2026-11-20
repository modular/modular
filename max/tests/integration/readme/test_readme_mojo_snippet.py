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
"""Runs the README's Mojo snippet with the compiler from the toolchain."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from extract_readme_code_snippets import (
    COVERED_LABELS,
    bash_commands,
    bazel_run_target,
)

README = Path(os.environ["README_PATH"])
SECTION = "Get started"
LABEL = "//Mojo:mojo"


@pytest.mark.parametrize("command", bash_commands(README, SECTION))
def test_mojo_snippet(command: str) -> None:
    label, args = bazel_run_target(command)
    if label != LABEL:
        pytest.skip(f"{label} is covered by {COVERED_LABELS[label]}")
    mojo = os.path.abspath(os.environ["MODULAR_MOJO_MAX_DRIVER_PATH"])
    result = subprocess.run(
        [mojo, *shlex.split(args)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"`{command}` failed:\n{result.stderr[-4000:]}"
    )
    assert result.stdout.strip(), f"`{command}` produced no output"
