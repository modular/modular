//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/NixlPluginDir.h"

#include <dlfcn.h>

using namespace M;

// Probes whether a shared library resolves on this host, without keeping it
// loaded. Used to decide between plugin flavors whose load-time dependencies
// differ.
static bool canLoadSharedLib(const char *name) {
  if (void *handle = ::dlopen(name, RTLD_LAZY | RTLD_LOCAL)) {
    ::dlclose(handle);
    return true;
  }
  return false;
}

std::optional<std::filesystem::path>
M::resolveNixlPluginDir(const std::filesystem::path &base) {
  std::error_code ec;
  // Prefer cuda when both vendors are present.
  if (std::filesystem::exists("/dev/nvidiactl", ec) &&
      std::filesystem::exists(base / "cuda" / "libplugin_UCX.so", ec))
    return base / "cuda";
  if (std::filesystem::exists("/dev/kfd", ec)) {
    // Prefer the verbs flavor — a strict superset of the plain rocm flavor
    // that adds the uct_ib RDMA transports for internode transfers — when its
    // hard load-time dependencies (rdma-core) are present.
    if (std::filesystem::exists(base / "rocm-verbs" / "libplugin_UCX.so", ec) &&
        canLoadSharedLib("libibverbs.so.1") && canLoadSharedLib("libmlx5.so.1"))
      return base / "rocm-verbs";
    if (std::filesystem::exists(base / "rocm" / "libplugin_UCX.so", ec))
      return base / "rocm";
  }
  return std::nullopt;
}
