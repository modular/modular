# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "API Integration Tests"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".api"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(
    config.modular_obj_root, "All", "test", "API"
)

config.substitutions.append(
    ("%modelsdir", str(Path(config.modular_src_root) / "Models"))
)

tool_dirs = [
    config.modular_tools_dir,
    config.modular_utils_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["modular-api-executor"]
llvm_config.add_tool_substitutions(tools, tool_dirs)

multi_tenancy_api_models_dir = (
    Path(config.modular_obj_root) / "multi-tenancy" / "api-models"
)
config.substitutions.append(
    ("%multi_tenancy_api_models_dir", str(multi_tenancy_api_models_dir))
)

concurrent_api_models_dir = (
    Path(config.modular_obj_root) / "concurrent" / "api-models"
)
config.substitutions.append(
    ("%concurrent_api_models_dir", str(concurrent_api_models_dir))
)

api_executor_models_dir = (
    Path(config.modular_obj_root) / "api-executor" / "api-models"
)
config.substitutions.append(
    ("%api_executor_models_dir", str(api_executor_models_dir))
)

mojo_user_kernels = (
    Path(config.modular_src_root)
    / "Kernels"
    / "mojo"
    / "Mogg"
    / "MOGGTests.mojo"
)

mojo_user_package = (
    Path(config.modular_src_root) / "All" / "test" / "API" / "test_user_op"
)
config.substitutions.append(("%mojo_user_kernels", mojo_user_kernels))
config.substitutions.append(("%mojo_user_pkg", mojo_user_package))

generated_models_path = (
    Path(config.modular_derived_dir)
    / ".derived"
    / "build"
    / "GeneratedTests"
    / "BINARIES"
    / "models"
)
if generated_models_path.exists():
    config.substitutions.append("%test_models_dir", str(generated_models_path))
    config.available_features.add("GENERATED_TESTS")
