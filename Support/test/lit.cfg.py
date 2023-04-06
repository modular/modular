# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lit.llvm import llvm_config

config.name = "Support"

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.modular_obj_root, "Support", "test")

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["build-info", "support-dialect-opt", "system-info"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
