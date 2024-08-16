# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform

from lit.llvm import llvm_config

if config.root.host_os == "Windows":
    # TODO(#13522): LLDB currently isn't built on windows.
    config.unsupported = True

# name: The name of this test suite.
config.name = "mojo-debug"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".lldb", ".mojo"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "KGEN", "test", "mojo-debug"
)

config.substitutions.append(
    (
        "%debug",
        (
            f"{config.lldb_env} mojo debug -X --source-quietly -X -S "
            f"-X {config.lit_lldb_init}"
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
