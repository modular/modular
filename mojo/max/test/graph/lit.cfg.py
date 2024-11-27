# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "MojoGraphAPI"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "SDK", "test", "API", "graph", "mojo"
)

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]

tools = [
    "mojo",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# TODO(#23665): Fix memory leaks.
config.environment["LSAN_OPTIONS"] = "detect_leaks=0"
# TODO(#25946): Fix ODR violations.
config.environment["ASAN_OPTIONS"] = "detect_odr_violation=0,use_sigaltstack=0"
