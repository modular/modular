# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform

from lit.llvm import llvm_config


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.processor() == "arm"


# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "Cache"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.modular_obj_root, "Cache", "test")

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["cache-mgr"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

if platform.system() == "Windows":
    config.available_features.add("windows")
