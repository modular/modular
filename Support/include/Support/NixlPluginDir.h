//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
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

} // namespace M

#endif // SUPPORT_NIXLPLUGINDIR_H
