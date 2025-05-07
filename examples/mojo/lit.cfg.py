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

cwd = os.path.dirname(__file__)
sys.path.append(cwd)

import os
from pathlib import Path

import mojo_jupyter_runner
from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "Mojo Examples"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".ipynb", ".mojo", ".🔥"]

example_code = Path(".") / "open-source" / "mojo" / "examples" / "public"
llvm_config.with_environment("PYTHONPATH", str(example_code), append_path=True)

# The below notebooks interop with python and have intermittent failures in CI testing.
# They may work locally and in PR tests, but they will fail in all kinds of different
# ways intermittently. Keep excluded from tests until a different solution is found.
config.excludes.update(
    [
        ".ipynb_checkpoints",
        # The programming manual is deprecated and should not be tested anymore
        "programming-manual.ipynb",
    ]
)

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "open-source", "mojo", "examples"
)

tool_dirs = [config.modular_tools_dir]
tools = ["mojo-jupyter-executor", "mojo"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.test_format = mojo_jupyter_runner.TestNotebook(config=config)
