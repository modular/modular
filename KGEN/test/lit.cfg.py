# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.


# name: The name of this test suite.
config.name = "mojo"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".mojo", ".test", ".🔥"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.modular_obj_root, "KGEN", "test")

# python tests shouldn't be handled by llvm-lit directly
config.excludes.add("pytests")

# These directories have their own test targets.
config.excludes.add("kgen")
config.excludes.add("mojo-debug")
config.excludes.add("mojo-integration")
config.excludes.add("mojo-jupyter")
config.excludes.add("mojo-lsp-server")
config.excludes.add("mojo-parser")
config.excludes.add("mojo-repl")

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "llvm-objdump",
    "mojo",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_system_environment(["PYTHONPATH"])
