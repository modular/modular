# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform

from lit.llvm import llvm_config


# name: The name of this test suite.
config.name = "mojo-integration"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo", ".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.modular_obj_root, "KGEN", "test")

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "mojo",
    "kgen",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
