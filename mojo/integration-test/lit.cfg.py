# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import lit.formats
import lit.llvm
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

config.test_format = lit.formats.ShTest(True)

# name: The name of this test suite.
config.name = "Mojo Standard Library"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "open-source", "mojo", "integration-test"
)

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["mojo"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
