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

"""Console-script entrypoints of the `mojo` development wheel.

Each entrypoint execs the corresponding binary shipped in the package's
`modular/bin` directory with the SDK environment (see `mojo.run`)."""

import os
import sys

from mojo._entrypoints import _entrypoint
from mojo._package_root import get_package_root
from mojo.run import _mojo_env


def exec_lldb_argdumper() -> None:
    _entrypoint("lldb-argdumper")


def exec_lldb_dap() -> None:
    root = get_package_root()
    assert root
    env = _mojo_env()

    lib = root / "lib"
    args = [
        "--pre-init-command",
        f"?!plugin load {lib / 'libMojoLLDB.so'}",
        "--pre-init-command",
        f"?command script import {lib / 'lldb-visualizers' / 'lldbDataFormatters.py'}",
        "--pre-init-command",
        f"?command script import {lib / 'lldb-visualizers' / 'mlirDataFormatters.py'}",
    ] + sys.argv

    os.execve(root / "bin/lldb-dap", args, env)


def exec_lldb_server() -> None:
    _entrypoint("lldb-server")


def exec_llvm_symbolizer() -> None:
    _entrypoint("llvm-symbolizer")


def exec_mojo_lldb() -> None:
    _entrypoint("mojo-lldb")


def exec_mojo_lsp_server() -> None:
    _entrypoint("mojo-lsp-server")
