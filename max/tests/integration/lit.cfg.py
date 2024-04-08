# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "SDK Integration Tests"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".yaml"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "SDK", "integration-test"
)

config.excludes.add("EngineAPI")

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["max"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

mojo_user_package = (
    Path(config.modular_src_root)
    / "SDK"
    / "integration-test"
    / "EngineAPI"
    / "Inputs"
    / "test_user_op"
)
config.substitutions.append(("%mojo_user_pkg", mojo_user_package))
