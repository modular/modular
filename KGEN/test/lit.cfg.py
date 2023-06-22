# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.


def configure_lldb_tests(config):
    lldb_env = ""
    if config.llvm_use_sanitizer:
        lldb_env = "ASAN_OPTIONS=detect_leaks=0"
        if platform.system() == "Darwin":
            lldb_env += " " + config.asan_lib_inject_env

    # lit_lldb_init is a file with a list of commands to be executed by LLDB
    # during initialization that set it up for a correct execution during tests.
    config.substitutions.append(
        (
            "%repl",
            f"{lldb_env} mojo-repl --source-quietly -S {config.lit_lldb_init}",
        )
    )


# name: The name of this test suite.
config.name = "KGEN"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".mojo", ".test", ".🔥", ".lldb"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.modular_obj_root, "KGEN", "test")

configure_lldb_tests(config)

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "llvm-objdump",
    "kgen",
    "kgen-elaborate-opt",
    "kgen-opt",
    "kgen-translate",
    "mojo",
    "mojo-lsp-server",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_system_environment(
    ["MODULAR_PATH", "MODULAR_DERIVED_PATH", "PYTHONPATH"]
)
