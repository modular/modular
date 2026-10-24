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

#ifndef GENERICML_API_ENGINECONFIG_H
#define GENERICML_API_ENGINECONFIG_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Context.h"
#include "Support/DeviceSpecs.h"
#include "Support/SymbolExport.h"
#include <filesystem>
#include <vector>

namespace M::Engine {

/// A device the runtime may place work on: the MLRT context backing it, plus
/// the spec carrying the label and id the API surface exposes. The legacy
/// GenericML `Driver::Device` wrapper bundled exactly this pair (DRIV-399).
struct ConfiguredDevice {
  DeviceSpecRef spec;
  M::Driver::DeviceContextRef context;
};

/// Configuration object containing options applying to an
/// entire inference session.
class MODULAR_CXX_EXPORT Config {
public:
  Config();

  void setNumThreads(size_t count);

  size_t getNumThreads() const { return numThreads; }

  /// Internal only: this is used to limit the runtime.

  void setMaxThreads(size_t count);

  size_t getMaxThreads() const { return maxThreads; }

  /// Set the type of allocator used in the runtime. The values are
  /// defined in the enum M_AllocatorType.

  void setAllocatorType(int allocType) { this->allocatorType = allocType; }

  /// Get the type of allocator used in the runtime. The values are
  /// defined in the enum M_AllocatorType.

  int getAllocatorType() const { return allocatorType; }

  void addDevice(ConfiguredDevice device) {
    this->devices.push_back(std::move(device));
  }

  const std::vector<ConfiguredDevice> &getDevices() const {
    return this->devices;
  }

  // Set up API info.
  void setAPILanguage(const std::string &apiLanguage) {
    this->apiLanguage = apiLanguage;
  }

  StringRef getAPILanguage() const { return apiLanguage; }

  /// Get global context shared with Mojo.
  ContextRef getMaxContext() const { return maxContext; }

  /// Set global context shared with Mojo.
  void setMaxContext(M::Context *ctx) { maxContext = copyRCRef(ctx); }

  std::chrono::microseconds getThreadBusyWaitTime() const {
    return threadBusyWaitTime;
  }

  std::string_view getContextName() const { return contextThreadPoolName; }

  void enableCrashLog(const std::filesystem::path &path) {
    crashDumpDir = path;
    crashLogEnabled = true;
  }

  bool isCrashLogEnabled() const { return crashLogEnabled; }

  std::optional<std::filesystem::path> getCrashDumpDir() const {
    return crashDumpDir;
  }

private:
  size_t numThreads = 0;
  size_t maxThreads = 0;

  std::vector<ConfiguredDevice> devices;

  int allocatorType = 1; /* kCaching */
  std::string contextThreadPoolName = "🔥 Thread";
  std::chrono::microseconds threadBusyWaitTime = std::chrono::microseconds(200);
  std::optional<std::filesystem::path> crashDumpDir;
  bool crashLogEnabled = false;
  std::string apiLanguage = "c";
  ContextRef maxContext;
};

} // namespace M::Engine

#endif // GENERICML_API_ENGINECONFIG_H
