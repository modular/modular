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

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mojo._entrypoints import _forward_xlinker_args


def test_forward_xlinker_for_run() -> None:
    env: dict[str, str] = {}
    argv = [
        "mojo",
        "run",
        "-Xlinker",
        "-L",
        "-Xlinker",
        "/path/to/lib",
        "main.mojo",
        "--",
        "arg1",
    ]

    updated_env, updated_argv = _forward_xlinker_args(env, argv)

    assert updated_env["MODULAR_MOJO_MAX_SYSTEM_LIBS"] == (
        "-Xlinker,-L,-Xlinker,/path/to/lib"
    )
    assert updated_argv == ["mojo", "run", "main.mojo", "--", "arg1"]


def test_forward_xlinker_for_debug_appends_existing() -> None:
    env = {"MODULAR_MOJO_MAX_SYSTEM_LIBS": "-Xlinker,-lfoo"}
    argv = ["mojo", "debug", "-Xlinker", "-lbar", "main.mojo"]

    updated_env, updated_argv = _forward_xlinker_args(env, argv)

    assert updated_env["MODULAR_MOJO_MAX_SYSTEM_LIBS"] == (
        "-Xlinker,-lfoo,-Xlinker,-lbar"
    )
    assert updated_argv == ["mojo", "debug", "main.mojo"]


def test_forward_xlinker_handles_combined_flag() -> None:
    env: dict[str, str] = {}
    argv = [
        "mojo",
        "run",
        "-Xlinker,-rpath,-Xlinker,/opt/libs",
        "main.mojo",
    ]

    updated_env, updated_argv = _forward_xlinker_args(env, argv)

    assert updated_env["MODULAR_MOJO_MAX_SYSTEM_LIBS"] == (
        "-Xlinker,-rpath,-Xlinker,/opt/libs"
    )
    assert updated_argv == ["mojo", "run", "main.mojo"]


def test_forward_xlinker_noop_for_other_commands() -> None:
    env = {"MODULAR_MOJO_MAX_SYSTEM_LIBS": "-Xlinker,-lfoo"}
    argv = ["mojo", "build", "-Xlinker", "-lbar", "main.mojo"]

    updated_env, updated_argv = _forward_xlinker_args(env, argv)

    assert updated_env is env
    assert updated_argv == argv
