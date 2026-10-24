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
// This file declares DriverPlugin interfaces within M::Driver.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_PLUGIN_H
#define MLRT_DRIVER_PLUGIN_H

#include "HAL/M_driver_api.h"
#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ReferenceCounted.h"

#include <memory>
#include <string>
#include <utility>

namespace M::Driver::Plugin {

class Driver;

using DriverRef = RCRef<Driver>;

class Driver : public ReferenceCounted<Driver> {
public:
  static ErrorOr<int64_t> Create(std::string const &soPath);
  static ErrorOr<DriverRef> Get(int64_t id);
  virtual ~Driver() = default;
  virtual ErrorOr<bool> isInitialized() const = 0;

  virtual ErrorOr<int64_t> deviceCount() const = 0;
};

/// Auto-load plugins from MODULAR_DRIVER_PLUGINS env var.
/// Returns the number of plugins loaded, or error.
ErrorOr<int> loadPlugins();

/// Idempotent, thread-safe wrapper around loadPlugins().
/// Call this from every public entry point that needs plugins to be available.
void ensurePluginsLoaded();

/// Get a native device from the plugin registered for `api`.
/// Returns error if no plugin handles this API.
ErrorOr<M::Driver::DeviceRef> getDevice(StringRef api, int id,
                                        int64_t deviceMemoryLimit);

/// Get device count from the plugin registered for `api`.
int deviceCount(StringRef api);

/// Check if any plugin is registered for this API.
bool hasPluginForApi(StringRef api);

/// Get arch info from any loaded plugin: returns (api_name, arch_name).
ErrorOr<std::pair<std::string, std::string>> getArchFromPlugins();

/// Get a compilation device from the plugin registered for `api`.
ErrorOr<std::unique_ptr<CompilationDevice>>
getCompilationDevice(StringRef api, StringRef arch);

} // namespace M::Driver::Plugin

#endif // MLRT_DRIVER_PLUGIN_H
