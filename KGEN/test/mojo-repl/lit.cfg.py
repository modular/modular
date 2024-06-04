# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

from lit.llvm import llvm_config

if config.root.host_os == "Windows":
    # TODO(#13522): LLDB currently isn't built on windows.
    config.unsupported = True

# name: The name of this test suite.
config.name = "mojo-repl"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".lldb"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "KGEN", "test", "mojo-repl"
)

python_env_path = Path(config.modular_derived_dir) / "autovenv" / "bin"
llvm_config.with_environment("PATH", str(python_env_path), append_path=True)

config.substitutions.append(
    (
        "%repl",
        (
            f"{config.lldb_env} mojo repl --source-quietly -S"
            f" {config.lit_lldb_init}"
        ),
    )
)

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["mojo"]


llvm_config.add_tool_substitutions(tools, tool_dirs)
