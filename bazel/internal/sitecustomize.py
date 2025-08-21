# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import os
import sys


def __absolutize_path(value: str) -> str:
    if value == "":
        return value

    if os.path.exists(value):
        return os.path.abspath(value)

    if workspace := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        value = value.replace("$BUILD_WORKSPACE_DIRECTORY", workspace)

    if execroot := os.getenv("BUILD_EXECROOT"):
        rebased = os.path.join(execroot, value)
        if os.path.exists(rebased):
            return os.path.abspath(rebased)

    if runfiles_dir := os.getenv("RUNFILES_DIR"):
        rebased = os.path.join(runfiles_dir, value)
        if os.path.exists(rebased):
            return os.path.abspath(rebased)

    if "BAZEL_TEST" in os.environ and value.startswith("external/"):
        rebased = "../" + value[len("external/") :]
        if os.path.exists(rebased):
            return os.path.abspath(rebased)

    return value


def __absolutize_env() -> None:
    for key in sorted(os.environ.keys()):
        value = os.environ[key]
        if key == "MODULAR_MOJO_MAX_IMPORT_PATH":
            value = ",".join(
                sorted(__absolutize_path(x) for x in value.split(","))
            )
        elif key == "MODULAR_MOJO_MAX_SHARED_LIBS":
            value = ",".join(__absolutize_path(x) for x in value.split(","))
        else:
            value = __absolutize_path(value)

        os.environ[key] = value

    if "MOJO_PYTHON" not in os.environ:
        os.environ["MOJO_PYTHON"] = sys.executable
        os.environ["MOJO_PYTHON_LIBRARY"] = sys.executable


__absolutize_env()
