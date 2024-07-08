# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "mojo-tool"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo", ".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "KGEN", "test", "mojo-tool"
)

# Exclude directories that define Mojo packages; these are used as test inputs.
config.excludes = [
    "test_package",
    "test_package_with_main",
    "inputs",
    "test-package-moco-773",
]

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["mojo", "kgen-opt", "lldb", "llvm-objdump"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
