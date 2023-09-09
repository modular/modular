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
elif config.root.host_os == "Darwin" and platform.processor() != "arm":
    # TODO(#20407): LLDB and Jupyter tests fail on macOS x86_64.
    config.unsupported = True


# name: The name of this test suite.
config.name = "mojo-lldb-repl"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".lldb"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.modular_obj_root, "KGEN", "test")

lldb_env = ""
if config.llvm_use_sanitizer:
    lldb_env = "ASAN_OPTIONS=detect_leaks=0,abort_on_error=1,disable_coredump=0"
if platform.system() == "Darwin":
    lldb_env += " " + config.asan_lib_inject_env

lit_lldb_init = config.lit_lldb_init
config.substitutions.append(
    (
        "%lldb",
        f"{lldb_env} lldb --source-quietly -S {lit_lldb_init}",
    )
)
config.substitutions.append(
    (
        "%repl",
        f"{lldb_env} mojo repl --source-quietly -S {lit_lldb_init}",
    )
)

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = []


llvm_config.add_tool_substitutions(tools, tool_dirs)
