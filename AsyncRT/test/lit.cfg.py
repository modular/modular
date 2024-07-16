# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst, FindTool
import platform

config.name = "AsyncRT"

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.modular_obj_root, "AsyncRT", "test")

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Setup substitutions.
config.substitutions.append(
    (
        "%driver-tblgen",
        (
            'driver-tblgen -I "{0}/KGEN/include" -I "{0}/AsyncRT/include" '
            '-I "{0}/third-party/llvm-project/llvm/include"'
        ).format(config.modular_src_root),
    )
)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".td"]

tool_dirs = [
    config.modular_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "build-info",
    ToolSubst("%crash-report-path-info", FindTool("crash-report-path-info")),
    "crash-test-dummy",
    "driver-tblgen",
    ToolSubst(
        "%modular-crashpad-handler", FindTool("modular-crashpad-handler")
    ),
    "support-dialect-opt",
    "system-info",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

if platform.system() == "Windows":
    config.available_features.add("windows")

if platform.system() == "Darwin" and platform.processor() == "arm":
    config.available_features.add("apple-m1")
