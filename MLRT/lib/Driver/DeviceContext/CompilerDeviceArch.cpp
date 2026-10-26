//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Minimal, plugin-direct implementation of
// `M::Driver::Device::getAcceleratorArchOrEmpty()` for the Mojo/KGEN compiler
// family.
//
// The compiler only needs the local accelerator's arch string (to set the
// compilation target); it must NOT link a device-context runtime. The HAL
// DeviceContext implementation is Mojo-backed (it reaches `@mojo//:std`), which
// would form a build-time bootstrap cycle; the legacy C++ DeviceContext runtime
// is not linked either. This TU provides just that one static by `dlopen`ing
// the accelerator plugin named in `MODULAR_DRIVER_PLUGINS` and reading device
// 0's `arch` property over the `M_driver_*` C ABI directly. It returns the
// `vendor:arch` form `ObjectCompiler` expects, as the HAL runtime's
// `getAcceleratorArchOrEmpty` does.
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/HALDeviceContext.h"

#include "HAL/M_driver_api.h"
#include "MLRT/Driver/DeviceContext/DeviceAPINames.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Process.h"

#include <dlfcn.h>
#include <optional>
#include <string>
#include <utility>

namespace M {
namespace Driver {

namespace {

// `spec` is `MODULAR_DRIVER_PLUGINS`: ';'-separated "Name@/path.so" entries.
// A bare path names no api, so only the plugin can say whether it's the CPU's.
llvm::SmallVector<llvm::StringRef> candidatePluginPaths(llvm::StringRef spec) {
  llvm::SmallVector<llvm::StringRef> entries;
  spec.split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  llvm::SmallVector<llvm::StringRef> paths;
  for (llvm::StringRef entry : entries) {
    auto [name, path] = entry.contains('@')
                            ? entry.split('@')
                            : std::make_pair(llvm::StringRef(), entry);
    if (!name.equals_insensitive("cpu"))
      paths.push_back(path);
  }
  return paths;
}

// "<vendor>:<arch>" of device 0 when the plugin's api has a Mojo vendor, or ""
// when it has no device. `std::nullopt` when the plugin doesn't load or has no
// Mojo vendor, so the caller tries the next.
std::optional<std::string> queryPluginArch(llvm::StringRef soPath) {
  void *lib = ::dlopen(std::string(soPath).c_str(), RTLD_NOW);
  if (!lib)
    return std::nullopt;
  auto closeLib = llvm::scope_exit([&] { ::dlclose(lib); });

  auto create = reinterpret_cast<M_driver_create_func_ptr_t>(
      ::dlsym(lib, "M_driver_create"));
  auto destroy = reinterpret_cast<M_driver_destroy_func_ptr_t>(
      ::dlsym(lib, "M_driver_destroy"));
  auto property = reinterpret_cast<M_driver_property_func_ptr_t>(
      ::dlsym(lib, "M_driver_property"));
  auto deviceGet = reinterpret_cast<M_driver_device_get_func_ptr_t>(
      ::dlsym(lib, "M_driver_device_get"));
  auto deviceProperty = reinterpret_cast<M_driver_device_property_func_ptr_t>(
      ::dlsym(lib, "M_driver_device_property"));
  if (!create || !destroy || !property || !deviceGet || !deviceProperty)
    return std::nullopt;

  M_driver_version version{M_DRIVER_INTERFACE_VERSION_MAJOR,
                           M_DRIVER_INTERFACE_VERSION_MINOR,
                           M_DRIVER_INTERFACE_VERSION_PATCH};
  M_driver_driver_t driver = nullptr;
  if (create(&version, &driver) != M_DRIVER_STATUS_SUCCESS || !driver)
    return std::nullopt;
  auto destroyDriver = llvm::scope_exit([&] { destroy(driver); });

  // Plugins spell their api in any case ("CUDA", "Metal", "cpu").
  const char *apiName = nullptr;
  if (property(driver, M_DRIVER_PROPERTY_API, &apiName) !=
          M_DRIVER_STATUS_SUCCESS ||
      !apiName)
    return std::nullopt;
  std::string api = canonicalDeviceAPI(llvm::StringRef(apiName).lower()).str();
  if (!hasMojoVendor(api))
    return std::nullopt;

  M_driver_device_t device = nullptr;
  char archName[M_DRIVER_DEVICE_PROPERTY_MAX_LEN + 1] = {};
  if (deviceGet(driver, /*device_id=*/0, &device) != M_DRIVER_STATUS_SUCCESS ||
      !device ||
      deviceProperty(device, M_DRIVER_DEVICE_PROPERTY_ARCH, archName) !=
          M_DRIVER_STATUS_SUCCESS ||
      archName[0] == '\0')
    return std::string();
  return (llvm::Twine(apiToVendorPrefix(api)) + ":" + archName).str();
}

// The arch of the first plugin in `MODULAR_DRIVER_PLUGINS` whose api has a Mojo
// vendor. Returns "" when there's none; callers then fall back to an explicit
// target arch.
std::string queryAcceleratorArch() {
  std::optional<std::string> spec =
      llvm::sys::Process::GetEnv("MODULAR_DRIVER_PLUGINS");
  if (!spec)
    return {};
  for (llvm::StringRef soPath : candidatePluginPaths(*spec))
    if (std::optional<std::string> arch = queryPluginArch(soPath))
      return *arch;
  return {};
}

} // namespace

const std::string &Device::getAcceleratorArchOrEmpty() {
  static const std::string arch = [] {
    // Virtual (compile-only) mode wins over plugin detection: with the knobs
    // set, the target arch is the configured one and there is deliberately no
    // accelerator to query. Without this the compiler would fall back to "" on
    // a GPU-less host and silently compile for the wrong (or no) target.
    if (std::string virtualArch = AsyncRT::getVirtualDeviceTargetArch();
        !virtualArch.empty())
      return (llvm::Twine(apiToVendorPrefix(AsyncRT::getVirtualDeviceAPI())) +
              ":" + virtualArch)
          .str();
    return queryAcceleratorArch();
  }();
  return arch;
}

// `DeviceContext::formatError` is the error formatter behind `ErrorMacros.h`,
// reached by `Driver/CompilationDeviceImpl`. We provide an implementation here
// for targets that don't like a DeviceContext implementation at all.
Error DeviceContext::formatError(const char *file, int line, const char *func,
                                 const llvm::Twine &msg) {
  return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
               " failed: " + msg);
}

} // namespace Driver
} // namespace M
