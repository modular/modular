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

"""Contains entrypoints for the Mojo wheel"""

import os
import sys

from ._package_root import get_package_root
from .run import _mojo_env

_XLINKER_COMMANDS = {"run", "debug"}


def _split_args_on_double_dash(args: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in args:
        return args, []

    divider_index = args.index("--")
    return args[:divider_index], args[divider_index:]


def _extract_xlinker_args(args: list[str]) -> tuple[list[str], list[str]]:
    remaining_args: list[str] = []
    xlinker_args: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-Xlinker":
            if index + 1 < len(args):
                xlinker_args.extend([arg, args[index + 1]])
                index += 2
                continue
            remaining_args.append(arg)
            index += 1
            continue
        if arg.startswith("-Xlinker,"):
            xlinker_args.extend([part for part in arg.split(",") if part])
            index += 1
            continue
        remaining_args.append(arg)
        index += 1
    return remaining_args, xlinker_args


def _forward_xlinker_args(
    env: dict[str, str], argv: list[str]
) -> tuple[dict[str, str], list[str]]:
    if len(argv) < 2 or argv[1] not in _XLINKER_COMMANDS:
        return env, argv

    command_args, runtime_args = _split_args_on_double_dash(argv[1:])
    filtered_args, xlinker_args = _extract_xlinker_args(command_args)
    if not xlinker_args:
        return env, argv

    existing = [
        arg
        for arg in env.get("MODULAR_MOJO_MAX_SYSTEM_LIBS", "").split(",")
        if arg
    ]
    env["MODULAR_MOJO_MAX_SYSTEM_LIBS"] = ",".join(existing + xlinker_args)
    return env, [argv[0]] + filtered_args + runtime_args


def _entrypoint(file: str) -> None:
    root = get_package_root()
    assert root
    env = _mojo_env()

    os.execve(root / "bin" / file, sys.argv, env)


def exec_mojo() -> None:
    env = _mojo_env()
    env, argv = _forward_xlinker_args(env, list(sys.argv))

    os.execve(env["MODULAR_MOJO_MAX_DRIVER_PATH"], argv, env)


def exec_lld() -> None:
    _entrypoint("lld")


def exec_modular_crashpad_handler() -> None:
    _entrypoint("modular-crashpad-handler")
