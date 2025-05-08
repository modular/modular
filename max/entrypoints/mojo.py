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
import subprocess
import sys
from pathlib import Path
from typing import Any


# fmt: off
def _sdk_default_env() -> dict[str, str]:
    # max/entrypoints/mojo.py -> max
    pypi_root = Path(__file__).parent.parent
    # .magic/envs/default/lib/python3.12/site-packages/max/entrypoints/mojo.py ->
    # .magic/envs/default/
    conda_root = Path(__file__).parent.parent.parent.parent.parent.parent

    if (conda_root / "bin" / "mojo").exists():
        root = conda_root
    else:
        root = pypi_root

    bin = root / "bin"
    lib = root / "lib"

    ext = ".dylib" if sys.platform == "darwin" else ".so"

    return {
        "MODULAR_MAX_CACHE_DIR": str(root / "share" / "max" / ".max_cache"),
        "MODULAR_MAX_DRIVER_LIB": str(lib / "libDeviceDriver") + ext,
        "MODULAR_MAX_ENABLE_COMPILE_PROGRESS": "true",
        "MODULAR_MAX_ENABLE_MODEL_IR_CACHE": "true",
        "MODULAR_MAX_ENGINE_LIB": str(lib / "libmodular-framework-common") + ext,
        "MODULAR_MAX_GRAPH_LIB": str(lib / "libmax") + ext,
        "MODULAR_MAX_PATH": str(root),
        "MODULAR_MAX_NAME": "MAX Platform",
        "MODULAR_MAX_TORCH_EXT_LIB": (
            str(lib / "libmodular-framework-torch-ext") + ext
        ),
        # MODULAR_MAX_VERSION intentionally omitted

        "MODULAR_MOJO_MAX_COMPILERRT_PATH": (
            str(lib / "libKGENCompilerRTShared") + ext
        ),
        "MODULAR_MOJO_MAX_MGPRT_PATH": str(lib / "libMGPRT") + ext,
        "MODULAR_MOJO_MAX_ATENRT_PATH": str(lib / "libATenRT") + ext,
        "MODULAR_MOJO_MAX_SHARED_LIBS": (
            str(lib / "libAsyncRTMojoBindings")
            + ext
            + ","
            + str(lib / "libAsyncRTRuntimeGlobals")
            + ext
            + ","
            + str(lib / "libMSupportGlobals")
            + ext
            + ",-Xlinker,-rpath,-Xlinker,"
            + str(lib)
        ),
        "MODULAR_MOJO_MAX_DRIVER_PATH": str(bin / "mojo"),
        "MODULAR_MOJO_MAX_IMPORT_PATH": str(lib / "mojo"),
        # MODULAR_MOJO_MAX_JUPYTER_PATH
        "MODULAR_MOJO_MAX_LLDB_PATH": str(bin / "mojo-lldb"),
        "MODULAR_MOJO_MAX_LLDB_PLUGIN_PATH": str(lib / "libMojoLLDB") + ext,
        "MODULAR_MOJO_MAX_LLDB_VISUALIZERS_PATH": str(lib / "lldb-visualizers"),
        # env["MODULAR_MOJO_MAX_LLDB_VSCODE_PATH"] = str(bin / "mojo-lldb-dap")
        "MODULAR_MOJO_MAX_LSP_SERVER_PATH": str(bin / "mojo-lsp-server"),
        # MODULAR_MOJO_MAX_MBLACK_PATH
        "MODULAR_MOJO_MAX_ORCRT_PATH": str(lib / "liborc_rt.a"),
        "MODULAR_MOJO_MAX_REPL_ENTRY_POINT": str(lib / "mojo-repl-entry-point"),
        "MODULAR_MOJO_MAX_LLD_PATH": str(bin / "lld"),
        "MODULAR_MOJO_MAX_SYSTEM_LIBS": (
            "-lm,-lz,-lcurses"
            if sys.platform == "darwin"
            else "-lrt,-ldl,-lpthread,-lm,-lz,-ltinfo"
        ),
        "MODULAR_MOJO_MAX_TEST_EXECUTOR_PATH": str(lib / "mojo-test-executor"),

        "MODULAR_CRASH_REPORTING_HANDLER_PATH": str(
            bin / "modular-crashpad-handler"
        ),
    }
# fmt: on


def _mojo_env() -> dict[str, str]:
    """Returns an environment variable set that uses the Mojo SDK environment
    paths by default, but with overrides from the ambient OS environment."""

    return _sdk_default_env() | dict(os.environ)


def exec_mojo():
    env = _mojo_env()

    os.execve(env["MODULAR_MOJO_MAX_DRIVER_PATH"], sys.argv, env)


def subprocess_run_mojo(
    mojo_args: list[str],
    **kwargs: Any,
):
    """Launches the bundled `mojo` in a subprocess, configured to use the
    `mojo` assets in the `max` package.

    Arguments:
        mojo_args: Arguments supplied to the `mojo` command.
        kwargs: Additional arguments to pass to `subprocess.run()`
    """

    env = _mojo_env()

    return subprocess.run(
        # Combine the `mojo` executable path with the provided argument list.
        [env["MODULAR_MOJO_MAX_DRIVER_PATH"]] + mojo_args,
        env=env,
        **kwargs,
    )


if __name__ == "__main__":
    exec_mojo()
