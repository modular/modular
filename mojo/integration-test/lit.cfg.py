# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform
import shutil

from pathlib import Path

import lit.formats
import lit.llvm

# Configuration file for the 'lit' test runner.

config.test_format = lit.formats.ShTest(True)

# name: The name of this test suite.
config.name = "Mojo Standard Library"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py"]

# This makes the OS name available for `REQUIRE` directives, e.g., `# REQUIRES: darwin`.
config.available_features.add(platform.system().lower())

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "open-source", "mojo", "integration-test"
)
