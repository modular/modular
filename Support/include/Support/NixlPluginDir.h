//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_NIXLPLUGINDIR_H
#define SUPPORT_NIXLPLUGINDIR_H

#include <filesystem>
#include <optional>

namespace M {

/// Picks the NIXL plugin directory for the detected host GPU vendor.
///
/// The staged layout ships the NIXL transport plugins in per-vendor
/// subdirectories of `base` (cuda/ for the CUDA flavor plus the EFA libfabric
/// plugin, rocm/ and rocm-verbs/ for the ROCm flavors, cpu/ for the GPU-free
/// flavor); NIXL discovers plugins by filename in exactly one directory, so
/// the caller must pick one before the NIXL plugin manager reads
/// NIXL_PLUGIN_DIR. Each vendor is detected explicitly via its kernel device
/// node (/dev/nvidiactl for NVIDIA, /dev/kfd for amdgpu) — never assumed from
/// the absence of the other. On AMD hosts the rocm-uccl flavor is the default;
/// an explicit MODULAR_NIXL_TRANSFER_BACKEND=ucx (or libfabric), or an
/// environment where rocm-uccl is not staged, falls back to the UCX flavors,
/// where the verbs flavor (a strict superset of the plain rocm flavor that
/// adds the uct_ib RDMA transports for internode transfers) is preferred when
/// its hard load-time dependencies (rdma-core) resolve.
///
/// Returns std::nullopt when no vendor (or no staged flavor for the detected
/// vendor) is found; callers then leave NIXL_PLUGIN_DIR unset and NIXL
/// transport construction fails downstream with its normal plugin-not-found
/// error. This runs in contexts that include CPU-only and macOS hosts where
/// NIXL is legitimately unused — so it must not hard-error.
std::optional<std::filesystem::path>
resolveNixlPluginDir(const std::filesystem::path &base);

/// Claims the `libfabric.so.1` SONAME with the copy staged alongside
/// `pluginDir`, for hosts that asked for the libfabric backend.
///
/// The libfabric plugin binds versioned symbols (FABRIC_1.8) that only the EFA
/// libfabric staged in `<prefix>/lib` provides, yet it records the generic
/// `libfabric.so.1` SONAME — which a much older distro libfabric also claims
/// (Ubuntu packages 1.14, FABRIC_1.3, for the Open MPI we install for expert
/// parallelism), and which other components pull in: torch's bundled NVSHMEM
/// ships its own libfabric transport, for one. Whichever copy loads first owns
/// the SONAME process-wide, because an already-loaded SONAME is never
/// re-resolved against the plugin's rpath, so losing that race leaves the
/// plugin permanently unloadable under a generic "backend not found".
///
/// Loading our copy up front (and keeping it loaded) makes the outcome
/// independent of load order: it is a strict superset of the distro's symbol
/// versions, so later consumers bind against it happily.
///
/// Returns false when nothing was preloaded — another backend was requested, no
/// libfabric is staged next to the plugins, or the load failed (e.g. no CUDA
/// driver, which the EFA build needs). None of those are errors here; the
/// backend that needs it reports its own failure downstream.
bool preloadStagedLibfabric(const std::filesystem::path &pluginDir);

} // namespace M

#endif // SUPPORT_NIXLPLUGINDIR_H
