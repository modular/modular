# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "mojo-isolated"

config.parser_stubs_derived = os.path.join(
    config.modular_obj_root, "KGEN", "test", "test-packages"
)
config.parser_stubs_source = os.path.join(
    config.modular_src_root, "KGEN", "test", "test-packages"
)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo", ".🔥"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.modular_obj_root, "KGEN", "test")

config.excludes = [
    "test_package",
    "debuginfo_module.mojo",
    "docs_package",
]

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

translate_with_prebuilt_packages = (
    "kgen-translate -import-mojo -mojo-enable-prebuilt-packages"
    " -mojo-search-paths={0}".format(
        config.parser_stubs_source
    )
)

config.substitutions.append(
    ("%parse-mojo-isolated", translate_with_prebuilt_packages)
)

config.environment["MODULAR_HOME"] = os.path.join(
    config.modular_obj_root, "KGEN", "test", "mojo-isolated"
)
