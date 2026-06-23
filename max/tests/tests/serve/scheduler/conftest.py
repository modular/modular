# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Configure NIXL plugin discovery before any test creates a nixlAgent.

The upstream nixlPluginManager is a Meyers singleton that reads NIXL_PLUGIN_DIR
once at first use.  The _core.cpp initializer sets NIXL_PLUGIN_DIR from the
detected package root, but in Bazel GPU test sandboxes the package root
detection can fail (no bin/mojo to anchor on).  This conftest sets
NIXL_PLUGIN_DIR from TEST_SRCDIR (which Bazel always sets) before any test
module is collected, ensuring the singleton picks up the correct plugins dir.

It also pre-loads libcuda.so.1 and libnvidia-ml.so.1 with RTLD_GLOBAL so
their symbols are visible when the plugin manager calls
  dlopen(libplugin_UCX.so, RTLD_NOW)
libplugin_UCX.so (cuda variant) references NVML symbols (nvmlInit_v2, etc.)
that are not pre-loaded in the Bazel test sandbox.  On CPU-only machines
(e.g. m7i) the preload silently no-ops.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path


def _configure_nixl_plugin_dir() -> None:
    if os.environ.get("NIXL_PLUGIN_DIR"):
        return  # caller already set it; respect that

    # Strategy 1: Bazel sets TEST_SRCDIR to the runfiles root.
    test_srcdir = os.environ.get("TEST_SRCDIR")
    if test_srcdir:
        candidate = Path(test_srcdir) / "+http_archive+nixl_upstream"
        if candidate.exists():
            os.environ["NIXL_PLUGIN_DIR"] = str(candidate)
            return

    # Strategy 2: walk up from __file__ looking for the nixl_upstream dir.
    path = Path(__file__).resolve()
    for parent in path.parents:
        candidate = parent / "+http_archive+nixl_upstream"
        if candidate.exists():
            os.environ["NIXL_PLUGIN_DIR"] = str(candidate)
            return


def _preload_gpu_libs() -> None:
    """Pre-load CUDA/NVML with RTLD_GLOBAL so libplugin_UCX.so can dlopen.

    libplugin_UCX.so (cuda variant) references nvmlInit_v2 and other NVML
    symbols. The upstream nixlPluginManager uses RTLD_NOW when loading
    plugins, which requires ALL symbols to be resolvable at dlopen time.
    Pre-loading with RTLD_GLOBAL makes the symbols available.  On CPU-only
    machines the libraries are absent and the OSError is silently swallowed.
    """
    for lib_name in (
        "libcuda.so.1",
        "libcuda.so",
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
    ):
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # not available on this machine; ignore silently


# Run at import time — before any test or fixture creates a nixlAgent — so
# NIXL_PLUGIN_DIR is set and GPU libs are loaded before the plugin manager
# singleton is constructed.
_configure_nixl_plugin_dir()
_preload_gpu_libs()
