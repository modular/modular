# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "mojo-isolated"

config.parser_stubs_source = os.path.abspath(
    os.path.join("KGEN", "test", "test-packages")
)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo", ".🔥", ".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join("KGEN", "test", "mojo-isolated")

config.excludes = [
    "debuginfo/inputs",
    "test_package",
    "test_package.foo",
    "test_bad_package",
    "test_package_user",
    "debuginfo_module.mojo",
    "docs_package",
    "imported_module.mojo",
    "imported_cached_module.mojo",
]

translate_with_prebuilt_packages = (
    "kgen-translate -import-mojo -mojo-enable-prebuilt-packages"
    " -mojo-search-paths={0}".format(config.parser_stubs_source)
)

config.substitutions.append(
    ("%parse-mojo-isolated", translate_with_prebuilt_packages)
)

config.environment["MODULAR_HOME"] = os.path.join(
    "KGEN", "test", "mojo-isolated"
)
