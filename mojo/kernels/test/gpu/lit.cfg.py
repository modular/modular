# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import sys

from lit.llvm import llvm_config

sys.path.insert(0, os.path.dirname(__file__))
import mojo_gpu_runner


# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "GPUKernels"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "Kernels", "test", "gpu"
)

tool_dirs = [config.modular_tools_dir]
tools = ["mojo", "kgen"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.test_format = mojo_gpu_runner.TestGPUKernel(config=config)
