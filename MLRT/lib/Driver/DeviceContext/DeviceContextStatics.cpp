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
// The legacy DeviceContext statics that need no device. They are in
// libMDriverCompilation so the compiler can use them without the
// device-context runtime.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

static_assert(
    !MODULAR_HAL_DEVICE_CONTEXT,
    "the legacy DeviceContext statics must compile against the legacy "
    "declarations: -UMODULAR_HAL_DEVICE_CONTEXT in this target's copts must "
    "override Driver/DeviceContextHeaders' define");

#include "AcceleratorArch.h"
#include "AsyncRT/Runtime/Globals/VirtualDeviceGlobals.h"
#include "MLRT/Driver/DeviceContext/DeviceAPINames.h"
#include "Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "Support/SymbolExport.h"
#include "Target/DriverPlugin.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <mutex>
#include <string>

namespace M::Driver {

Error DeviceContext::formatError(const char *file, int line, const char *func,
                                 const llvm::Twine &msg) {
  static bool syncModeEnabled = DeviceContext::syncModeFromConfig();

  llvm::StringLiteral kErrorMessage = "";

  if (!syncModeEnabled)
    kErrorMessage = "\nTo get more accurate error information, set "
                    "MODULAR_DEBUG=device-sync-mode.";

  static bool logVerbose = DeviceContext::configAsBool("log_verbose", false);
  if (logVerbose) {
    return Error(llvm::Twine(file) + ":" + llvm::Twine(line) + " - " + func +
                 " failed: " + msg + kErrorMessage);
  }
  return Error(msg + kErrorMessage);
}

// Mutex to protect shared, static data. E.g. config dictionary
static std::mutex &getGlobalMutex() {
  static std::mutex mu;
  return mu;
}

bool DeviceContext::configAsBool(StringRef key, bool defaultValue,
                                 bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  if (isDefault != nullptr) {
    StringRef s =
        config->getValue((llvm::Twine("device-context.") + key).str());
    *isDefault = s.empty();
  }
  return config->getValueAsBool((llvm::Twine("device-context.") + key).str(),
                                defaultValue);
}

bool DeviceContext::syncModeFromConfig() {
  // Sync mode can be enabled via the new `max-debug.device-sync-mode` debug
  // configuration (e.g. `MODULAR_DEBUG=device-sync-mode`) or via the legacy
  // `device-context.sync-mode` key (e.g. `MODULAR_DEVICE_CONTEXT_SYNC_MODE=1`).
  if (configAsBool("sync-mode", false))
    return true;
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError())
    return false;
  return config->getValueAsBool("max-debug.device-sync-mode", false);
}

int64_t DeviceContext::configAsInt(StringRef key, int64_t defaultValue,
                                   bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  StringRef s = config->getValue((llvm::Twine("device-context.") + key).str());
  if (isDefault != nullptr) {
    *isDefault = s.empty();
  }
  long long result = defaultValue;
  if (s.getAsInteger(10, result)) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }
  return result;
}

StringRef DeviceContext::configAsString(StringRef key, StringRef defaultValue,
                                        bool *isDefault) {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  if (config.isError()) {
    if (isDefault != nullptr) {
      *isDefault = true;
    }
    return defaultValue;
  }

  if (isDefault != nullptr) {
    StringRef s =
        config->getValue((llvm::Twine("device-context.") + key).str());
    *isDefault = s.empty();
  }
  return config->getValueOr((llvm::Twine("device-context.") + key).str(),
                            defaultValue);
}

bool DeviceContext::isDebugAllocatorEnabled(StringRef name) const {
  std::lock_guard<std::mutex> lock(getGlobalMutex());
  ErrorOr<Config> &config = getConfig();
  return config->isValueInList("debug.device_allocator", name);
}

ErrorOr<Config> &DeviceContext::getConfig() {
  static ErrorOr<Config> configuration = Config::open();
  return configuration;
}

MODULAR_CXX_EXPORT const std::string &Device::getAcceleratorArchOrEmpty() {
  static std::string arch = [] {
    // First check if we're in virtual device mode (cross-compilation).
    // Virtual device config is set at startup and takes precedence over
    // physical hardware detection.
    std::string virtualTargetArch = AsyncRT::getVirtualDeviceTargetArch();
    if (!virtualTargetArch.empty()) {
      std::string virtualAPI = AsyncRT::getVirtualDeviceAPI();
      return (llvm::Twine(apiToVendorPrefix(virtualAPI)) + ":" +
              virtualTargetArch)
          .str();
    }

    // Otherwise, detect from physical hardware
    std::string vendor("nvidia");
    ErrorOr<std::string> archName = getFirstCUDAArchName();

    if (archName.isError()) {
      vendor = "amdgpu";
      archName = getHIPArchName(0);
    }

#if MODULAR_MACOS
    if (archName.isError()) {
      vendor = "metal";
      archName = getMetalArchName(nullptr);
    }
#endif

    // Check all loaded plugins (generic, no hardcoded vendor names).
    if (archName.isError()) {
      Plugin::ensurePluginsLoaded();
      auto pluginResult = Plugin::getArchFromPlugins();
      if (!pluginResult.isError()) {
        vendor = pluginResult.get().first;
        archName = pluginResult.get().second;
      }
    }

    if (archName.isError())
      return std::string();

    return (llvm::Twine(vendor) + ":" + archName.get()).str();
  }();
  return arch;
}

} // namespace M::Driver
