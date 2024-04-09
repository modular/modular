# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform
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
    config.modular_obj_root, "SDK", "integration-test", "EngineAPI"
)

config.substitutions.append(
    ("%modelsdir", str(Path(config.modular_src_root) / "Models"))
)
config.substitutions.append(("%pyexe", sys.executable))

tool_dirs = [
    config.modular_tools_dir,
    config.modular_utils_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = ["modular-api-executor", "mojo", "mt", "is-cuda-available"]
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

config.substitutions.append(("%mojo_user_kernels", mojo_user_kernels))

generated_models_path = (
    Path(config.modular_derived_dir)
    / "build"
    / "GeneratedTests"
    / "BINARIES"
    / "models"
)
if generated_models_path.exists():
    config.substitutions.append(
        ("%test_models_dir", str(generated_models_path))
    )
    config.available_features.add("GENERATED_TESTS")

generated_onnx_models_path = (
    Path(config.modular_derived_dir) / "onnx-backend-tests"
)
if generated_onnx_models_path.exists():
    config.substitutions.append(
        ("%onnx_test_models_dir", str(generated_onnx_models_path))
    )
    config.available_features.add("GENERATED_ONNX_TESTS")


framework_lib_cfg = Path(config.test_exec_root) / "framework-lib-path.cfg"
modular_framework_lib_path = framework_lib_cfg.read_text().strip()
config.environment["MODULAR_AI_ENGINE_LIB_PATH"] = modular_framework_lib_path

# Enable MEF caching in tests if not given a value via env vars.
mef_cache_config = os.environ.get("MODULAR_MAX_ENABLE_MODEL_IR_CACHE", "true")
config.environment["MODULAR_MAX_ENABLE_MODEL_IR_CACHE"] = mef_cache_config

engine_pkg_dir = (
    Path(config.modular_src_root) / "GenericML" / "lib" / "API" / "mojo"
)

config.substitutions.append(("%engine_pkg_dir", engine_pkg_dir))

test_utils_dir = Path(config.modular_src_root) / "Kernels" / "test"
config.substitutions.append(("%test_utils_pkg_dir", test_utils_dir))


llvm_config.add_tool_substitutions(tools, tool_dirs)

config.available_features.add(config.llvm_use_sanitizer.lower())


if "numpy" in sys.modules:
    config.available_features.add("numpy")

config.excludes.update(["test_user_op"])

output = subprocess.run("is-cuda-available")
if output.returncode == 0:
    config.available_features.add("cuda")


pytorch_generated_tests_dir = os.path.join(
    config.modular_derived_dir, "pytorch-generated-tests"
)
config.substitutions.append(("%deriveddir", config.modular_derived_dir))

torch_vision_path = (
    Path(config.modular_derived_dir)
    / "third-party"
    / "stock-torchvision"
    / "src"
    / "build"
)

if platform.system() == "Darwin":
    torch_vision_path /= "libtorchvision.dylib"
elif platform.system() == "Linux":
    torch_vision_path /= "libtorchvision.so"
else:
    torch_vision_path /= "torchvision.dll"

config.substitutions.append(("%torchvisionlib", str(torch_vision_path)))
if Path(pytorch_generated_tests_dir).exists():
    config.substitutions.append(
        ("%pytorch_generated_tests_dir", pytorch_generated_tests_dir)
    )
    config.available_features.add("pytorch-generated-tests")
