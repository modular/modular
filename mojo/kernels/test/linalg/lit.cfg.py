# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform
from pathlib import Path
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.processor() == "arm"


# name: The name of this test suite.
config.name = "KernelsLinAlg"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "Kernels", "test", "linalg"
)

# We need to add this so that the tests can be predicated on whether we're
# running on an Apple M1 machine.
if is_apple_silicon():
    config.available_features.add("apple-m1")

if platform.system() == "Linux" and "avx2" in Path("/proc/cpuinfo").read_text():
    config.available_features.add("avx2")


tool_dirs = [config.modular_tools_dir]
tools = ["mojo"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
