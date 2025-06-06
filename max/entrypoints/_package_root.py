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
    # max/entrypoints/_package_root.py -> max
    pypi_root = Path(__file__).parent.parent
    # .magic/envs/default/lib/python3.12/site-packages/max/entrypoints/_package_root.py ->
    # .magic/envs/default/
    conda_root = Path(__file__).parent.parent.parent.parent.parent.parent

    # NOTE:
    #   MAX is currently available in two formats: as a set of Conda packages,
    #   and as a monolithic Pip wheel. This is a best-effort heuristic to guess
    #   which way MAX was installed, because that will change where we look to
    #   resolve the locations of the various important assets in MAX.
    if (pypi_root / "lib" / "liborc_rt.a").exists():
        logging.debug("Located MAX SDK assets assuming MAX PyPi package layout")
        return pypi_root
    elif (conda_root / "lib" / "liborc_rt.a").exists():
        logging.debug(
            "Located MAX SDK assets assuming MAX Conda package layout"
        )
        return conda_root
    elif "BAZEL_WORKSPACE" in os.environ:
        # We're running in a Modular internal Bazel test, so let Bazel handle
        # the environment variables.
        return None
    else:
        raise RuntimeError(
            "Unable to locate MAX SDK library assets root directory."
        )
