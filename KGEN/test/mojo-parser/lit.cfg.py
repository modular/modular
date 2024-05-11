# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "mojo-parser"

config.parser_stubs_source = os.path.join(
    config.modular_src_root, "KGEN", "test", "test-packages"
)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo", ".🔥", ".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "KGEN", "test", "mojo-parser"
)

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "kgen-translate",
    "kgen-opt",
    "hash-mlir",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
