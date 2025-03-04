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
import sys
from pathlib import Path

# max/entrypoints/mojo.py -> max
root = Path(__file__).parent.parent
bin = root / "bin"
lib = root / "lib"

ext = ".dylib" if sys.platform == "darwin" else ".so"

env = os.environ

env["MODULAR_MAX_CACHE_DIR"] = str(root / "share" / "max" / ".max_cache")
env["MODULAR_MAX_DRIVER_LIB"] = str(lib / "libDeviceDriver") + ext
env["MODULAR_MAX_ENABLE_COMPILE_PROGRESS"] = "true"
env["MODULAR_MAX_ENABLE_MODEL_IR_CACHE"] = "true"
env["MODULAR_MAX_ENGINE_LIB"] = str(lib / "libmodular-framework-common") + ext
env["MODULAR_MAX_GRAPH_LIB"] = str(lib / "libmof") + ext
env["MODULAR_MAX_PATH"] = str(root)
env["MODULAR_MAX_NAME"] = "MAX Platform"
env["MODULAR_MAX_TORCH_EXT_LIB"] = (
    str(lib / "libmodular-framework-torch-ext") + ext
)
# MODULAR_MAX_VERSION intentionally omitted

env["MODULAR_MOJO_MAX_COMPILERRT_PATH"] = (
    str(lib / "libKGENCompilerRTShared") + ext
)
env["MODULAR_MOJO_MAX_MGPRT_PATH"] = str(lib / "libMGPRT") + ext
env["MODULAR_MOJO_MAX_ATENRT_PATH"] = str(lib / "libATenRT") + ext
env["MODULAR_MOJO_MAX_SHARED_LIBS"] = (
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
    + ";"
)
env["MODULAR_MOJO_MAX_DRIVER_PATH"] = str(bin / "mojo")
env["MODULAR_MOJO_MAX_IMPORT_PATH"] = str(lib / "mojo")
# MODULAR_MOJO_MAX_JUPYTER_PATH
env["MODULAR_MOJO_MAX_LLDB_PATH"] = str(bin / "mojo-lldb")
env["MODULAR_MOJO_MAX_LLDB_PLUGIN_PATH"] = str(lib / "libMojoLLDB") + ext
env["MODULAR_MOJO_MAX_LLDB_VISUALIZERS_PATH"] = str(lib / "lldb-visualizers")
# env["MODULAR_MOJO_MAX_LLDB_VSCODE_PATH"] = str(bin / "mojo-lldb-dap")
env["MODULAR_MOJO_MAX_LSP_SERVER_PATH"] = str(bin / "mojo-lsp-server")
# MODULAR_MOJO_MAX_MBLACK_PATH
env["MODULAR_MOJO_MAX_ORCRT_PATH"] = str(lib / "liborc_rt.a")
env["MODULAR_MOJO_MAX_REPL_ENTRY_POINT"] = str(lib / "mojo-repl-entry-point")
env["MODULAR_MOJO_MAX_SYSTEM_LIBS"] = (
    "-lm,-lz,-lcurses"
    if sys.platform == "darwin"
    else "-lrt,-ldl,-lpthread,-lm,-lz,-ltinfo"
)
env["MODULAR_MOJO_MAX_TEST_EXECUTOR_PATH"] = str(lib / "mojo-test-executor")

env["MODULAR_CRASH_REPORTING_HANDLER_PATH"] = str(
    bin / "modular-crashpad-handler"
)


def main():
    os.execve(bin / "mojo", sys.argv, env)


if __name__ == "__main__":
    main()
