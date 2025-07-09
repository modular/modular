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

if __name__ == "__main__":
    # Convert relative file paths to absolute paths based on the build workspace directory
    workspace_dir = os.getenv("BUILD_WORKSPACE_DIRECTORY", os.getcwd())

    for i, arg in enumerate(sys.argv):
        if arg.endswith(".yaml") and not os.path.isabs(arg):
            # First try relative to workspace directory
            absolute_path = os.path.join(workspace_dir, arg)

            # If not found, try relative to the autotune directory
            if not os.path.exists(absolute_path):
                autotune_path = os.path.join(
                    workspace_dir,
                    "open-source/max/max/kernels/benchmarks/autotune",
                    arg,
                )
                if os.path.exists(autotune_path):
                    absolute_path = autotune_path

            if os.path.exists(absolute_path):
                sys.argv[i] = absolute_path
            else:
                # Keep original path if the workspace-based path doesn't exist
                sys.argv[i] = arg

    # Set up KERNEL_BENCHMARKS_ROOT if BUILD_WORKSPACE_DIRECTORY is available
    if (
        "BUILD_WORKSPACE_DIRECTORY" in os.environ
        and "KERNEL_BENCHMARKS_ROOT" not in os.environ
    ):
        os.environ["KERNEL_BENCHMARKS_ROOT"] = os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            "open-source/max/max/kernels/benchmarks",
        )

    # Import and run the main kbench CLI
    from kbench import main

    main()
