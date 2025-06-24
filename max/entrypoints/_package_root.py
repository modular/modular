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

from __future__ import annotations

import logging
import os
from pathlib import Path


def get_package_root() -> Path | None:
    """Returns the package root of this installation, or None if running in a Bazel environment"""
    current_path = Path(__file__).parent

    # Walk up the directory tree until we find lib/liborc_rt.a or hit root
    # This is currently supposed to work for two cases:
    # - Set of Conda packages.
    # - Monolithic Pip wheel installation.
    while current_path != current_path.parent:  # Stop at root directory
        if (current_path / "lib" / "liborc_rt.a").exists():
            logging.debug(f"Located MAX SDK assets at {current_path}")
            return current_path
        current_path = current_path.parent

    if (
        "BUILD_WORKSPACE_DIRECTORY" in os.environ
        or "BAZEL_WORKSPACE" in os.environ
    ):
        # We're running in a Modular internal Bazel test, so let Bazel handle
        # the environment variables.
        return None
    else:
        raise RuntimeError(
            "Unable to locate MAX SDK library assets root directory."
        )
