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

import os
import shutil
import sys
from pathlib import Path
from sysconfig import get_config_var as var

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst


if "MODULAR_RUNNING_TESTS" not in os.environ:
    raise SystemExit(
        "\033[31merror\033[0m: use 'bazel test' instead of 'bazel run' to run tests"
    )

# Choose between lit's internal shell pipeline runner and a real shell.  If
# LIT_USE_INTERNAL_SHELL is in the environment, we use that as an override.
use_lit_shell = os.environ.get("LIT_USE_INTERNAL_SHELL")
if use_lit_shell:
    # 0 is external, "" is default, and everything else is internal.
    execute_external = use_lit_shell == "0"
else:
    # Otherwise we default to internal on Windows and external elsewhere, as
    # bash on Windows is usually very slow.
    execute_external = sys.platform not in ["win32"]

import modular_test_format

config.test_format = modular_test_format.ModularShTest(execute_external)

if execute_external:
    config.available_features.add("shell")
# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
]

llvm_config.add_tool_substitutions([ToolSubst("python3", sys.executable)])

llvm_config.with_environment(
    "MOJO_PYTHON",
    sys.executable,
)
libpython = str(
    Path(sys.executable).resolve().parent.parent / "lib" / var("INSTSONAME")
)

llvm_config.with_environment(
    "MOJO_PYTHON_LIBRARY",
    libpython,
)

# Ensure the tests find the packages from the venv.
llvm_config.with_environment(
    "PATH",
    str(Path(sys.executable).parent),
    append_path=True,
)

extra_mojo_args = []

mojo_exe = "mojo"
if shutil.which("mojo-compiler-only"):
    mojo_exe = "mojo-compiler-only"

if config.llvm_use_sanitizer and config.llvm_use_sanitizer != "undefined":
    extra_mojo_args.extend(["--sanitize", config.llvm_use_sanitizer.lower()])

llvm_config.add_tool_substitutions(
    [
        ToolSubst(
            "%mojo-no-debug-no-assert",
            mojo_exe,
            extra_args=extra_mojo_args,
        )
    ]
)

llvm_config.add_tool_substitutions(
    [
        ToolSubst(
            "%mojo-build-no-debug-no-assert",
            mojo_exe,
            extra_args=["build"] + extra_mojo_args,
        )
    ]
)

# The rest of the mojo commands just inherit the assert options. In this case
# run with assertions enabled unless one explicitly sets
# MOJO_ENABLE_ASSERTIONS_IN_TESTS=0 environment variable.
if bool(int(os.environ.get("MOJO_ENABLE_ASSERTIONS_IN_TESTS", 1))):
    extra_mojo_args.extend(["-D", "ASSERT=all"])

llvm_config.add_tool_substitutions(
    [
        ToolSubst(
            "%mojo-no-debug",
            mojo_exe,
            extra_args=extra_mojo_args,
        )
    ]
)

llvm_config.add_tool_substitutions(
    [
        ToolSubst(
            "%mojo-build-no-debug",
            mojo_exe,
            extra_args=["build"] + extra_mojo_args,
        )
    ]
)

extra_mojo_args = ["--debug-level", "full"] + extra_mojo_args

llvm_config.add_tool_substitutions(
    [
        ToolSubst(
            "%mojo",
            mojo_exe,
            extra_args=extra_mojo_args,
        )
    ]
)

llvm_config.add_tool_substitutions(
    [
        ToolSubst(
            "%mojo-build",
            mojo_exe,
            extra_args=["build"] + extra_mojo_args,
        )
    ]
)

llvm_config.add_tool_substitutions(
    [
        ToolSubst(
            "%mpirun",
            "mpirun",
            extra_args=[
                "--allow-run-as-root",
                "--bind-to",
                "none",
            ],
        )
    ]
)

config.substitutions.append(("%bare-mojo", mojo_exe))
