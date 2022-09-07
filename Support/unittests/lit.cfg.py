# ===- lit.cfg.py ---------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import lit.formats
import lit.util
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "Support"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".cpp"]


# test_source_root: The root path where tests are located.
config.test_source_root = os.path.join(
    config.modular_obj_root, "Support", "unittests"
)

config.test_format = lit.formats.GoogleTest(config.test_source_root, "Tests")

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "Support", "unittests"
)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%shlibdir", config.modular_shlib_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite.
config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# See https://github.com/llvm/llvm-project/issues/56491 and
# https://github.com/llvm/llvm-project/issues/56492
os.environ["GTEST_TOTAL_SHARDS"] = "1"
os.environ["GTEST_SHARD_INDEX"] = "0"
