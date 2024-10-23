# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import subprocess
import sys
from pathlib import Path

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "Max Engine Integration Tests"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".api", ".mojo"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "SDK", "integration-test", "API"
)

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["modular-api-executor", "mojo", "mt"]
llvm_config.add_tool_substitutions(tools, tool_dirs)

multi_tenancy_api_models_dir = (
    Path(config.modular_obj_root) / "multi-tenancy" / "api-models"
)
config.substitutions.append(
    (
        "%multi_tenancy_api_models_dir",
        str(multi_tenancy_api_models_dir),
    )
)

concurrent_api_models_dir = (
    Path(config.modular_obj_root) / "concurrent" / "api-models"
)
config.substitutions.append(
    (
        "%concurrent_api_models_dir",
        str(concurrent_api_models_dir),
    )
)

# Enable MEF caching in tests if not given a value via env vars.
mef_cache_config = os.environ.get("MODULAR_MAX_ENABLE_MODEL_IR_CACHE", "true")
config.environment["MODULAR_MAX_ENABLE_MODEL_IR_CACHE"] = mef_cache_config

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.excludes.update(["test_user_op"])
