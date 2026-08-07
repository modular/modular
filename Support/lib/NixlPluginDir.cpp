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

#include "Support/NixlPluginDir.h"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <string>

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

// The lowercased MODULAR_NIXL_TRANSFER_BACKEND value, or "" if unset. Lets an
// explicit backend request override the AMD default flavor.
static std::string requestedBackend() {
  const char *b = std::getenv("MODULAR_NIXL_TRANSFER_BACKEND");
  if (!b)
    return "";
  std::string s(b);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

bool M::preloadStagedLibfabric(const std::filesystem::path &pluginDir) {
  if (requestedBackend() != "libfabric")
    return false;
  // The plugin dir is <prefix>/lib/nixl/<flavor>; libfabric is staged flat in
  // <prefix>/lib next to libnixl.
  const std::filesystem::path lib =
      pluginDir.parent_path().parent_path() / "libfabric.so.1";
  std::error_code ec;
  if (!std::filesystem::exists(lib, ec))
    return false;
  // Deliberately never dlclosed: the point is to hold the SONAME for the life
  // of the process.
  return ::dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL) != nullptr;
}

std::optional<std::filesystem::path>
M::resolveNixlPluginDir(const std::filesystem::path &base) {
  std::error_code ec;
  // Prefer cuda when both vendors are present.
  if (std::filesystem::exists("/dev/nvidiactl", ec) &&
      std::filesystem::exists(base / "cuda" / "libplugin_UCX.so", ec))
    return base / "cuda";
  if (std::filesystem::exists("/dev/kfd", ec)) {
    // UCCL is the AMD default (speed-of-light on RoCE fabrics UCX cannot
    // saturate). An explicit UCX/libfabric request opts out; and where the
    // UCCL flavor is not staged (e.g. the hermetic test runfiles), this
    // falls through to the UCX flavors below.
    const std::string backend = requestedBackend();
    if ((backend.empty() || backend == "uccl") &&
        std::filesystem::exists(base / "rocm-uccl" / "libplugin_UCCL.so", ec))
      return base / "rocm-uccl";
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
