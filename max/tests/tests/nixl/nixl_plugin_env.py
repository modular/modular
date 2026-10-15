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
"""Shared NIXL plugin discovery and preload for test conftests.

Per-vendor plugin flavor resolution has a single implementation
(Support/NixlPluginDir.h) that runs on ``import max._core``: the importer
locates the staged plugins (Bazel runfiles or installed layout) and sets
NIXL_PLUGIN_DIR before the upstream nixlPluginManager singleton reads it at
first use. This module only adds test policy on top: keep these suites on
the plain flavor, fail loudly when a GPU host resolves no flavor (broken
runfiles staging would otherwise surface as an obscure transport-unavailable
error mid-test), and pre-load the GPU runtime libraries the UCX plugin
references without linking (libcuda/libnvidia-ml for the CUDA flavor,
libhsa-runtime64 for the ROCm flavors) with RTLD_GLOBAL so their symbols are
visible when the plugin manager calls ``dlopen(libplugin_UCX.so, RTLD_NOW)``;
those symbols are not pre-loaded in the Bazel test sandbox. The host rdma-core
is pre-loaded too, so a verbs flavor binds it rather than the prebuilt copies
its rpath reaches in the runfiles.
"""

from __future__ import annotations

import ctypes
import os

import max._core  # noqa: F401  (sets NIXL_PLUGIN_DIR at import time)


def preload_gpu_libs() -> None:
    """Pre-loads GPU runtimes with RTLD_GLOBAL so libplugin_UCX.so can dlopen.

    The upstream nixlPluginManager uses RTLD_NOW when loading plugins, which
    requires ALL symbols to be resolvable at dlopen time. Pre-loading with
    RTLD_GLOBAL makes the symbols available. Libraries absent on the host are
    skipped silently — the flavor needing them is not selected there.
    """
    for lib_name in (
        "libcuda.so.1",
        "libcuda.so",
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "libhsa-runtime64.so.1",
        "libhsa-runtime64.so",
        # Load the host rdma-core before the plugin, so a *-verbs flavor binds
        # it rather than the prebuilt copies its rpath reaches in the runfiles.
        "libibverbs.so.1",
        "libmlx5.so.1",
    ):
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # not available on this machine; ignore silently


# TODO(MXSERV-576): Remove once test_send_recv_concurrent_gpu and test_di pass
# on the verbs flavor on an InfiniBand host.
def _force_plain_flavor() -> None:
    """Repoints NIXL_PLUGIN_DIR from a ``*-verbs`` flavor to its plain sibling.

    The suites that call :func:`configure` are validated on the plain flavor
    only. On an InfiniBand host the verbs flavor fails two of them: concurrent
    GPU send/recv hits a local protection error on the mlx5 device, and the
    disaggregated-inference suite's request cancellation cases fail, after
    which the suite times out. The rc/InfiniBand path is covered cross-process
    by the perf-smoke cross-node lane instead. This fires wherever the resolver
    picked a verbs flavor: NVIDIA hosts with an InfiniBand port, and AMD hosts
    where libibverbs and libmlx5 load, with or without one.
    """
    plugin_dir = os.environ.get("NIXL_PLUGIN_DIR")
    if not plugin_dir:
        return
    head, flavor = os.path.split(plugin_dir.rstrip("/"))
    plain = {"cuda-verbs": "cuda", "rocm-verbs": "rocm"}.get(flavor)
    if plain is not None:
        os.environ["NIXL_PLUGIN_DIR"] = os.path.join(head, plain)


def configure() -> None:
    """Configures NIXL plugin discovery and preloads GPU runtime libraries.

    Call at conftest import time, before any test or fixture creates a
    nixlAgent, so the environment is in place before the plugin manager
    singleton is constructed.
    """
    preload_gpu_libs()
    # max._core resolved NIXL_PLUGIN_DIR at import; keep these suites on the
    # plain flavor (see _force_plain_flavor) before the plugin-manager
    # singleton reads it.
    _force_plain_flavor()
    if os.environ.get("NIXL_PLUGIN_DIR"):
        return
    gpu_nodes = [
        node for node in ("/dev/nvidiactl", "/dev/kfd") if os.path.exists(node)
    ]
    if gpu_nodes:
        raise RuntimeError(
            f"GPU detected ({', '.join(gpu_nodes)}) but max._core resolved "
            "no NIXL plugin flavor: the per-vendor plugin .so is missing "
            "from the runfiles (check the target's @nixl_upstream data "
            "deps) or its load-time library dependencies do not resolve."
        )
