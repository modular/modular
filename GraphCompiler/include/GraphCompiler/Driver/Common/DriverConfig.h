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

#ifndef GENERICML_COMMONDRIVER_DRIVERCONFIG_H
#define GENERICML_COMMONDRIVER_DRIVERCONFIG_H

#include "Support/DeviceSpecs.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <filesystem>

namespace M {
class DriverConfig {
public:
  enum OptionKind {
    kTF,
    kTorch,
    kMOF,
    kKernelLibrary,
    kDevice,
    kTileBasedFusion
  };

  DriverConfig(OptionKind kind, llvm::StringLiteral name)
      : Kind(kind), configName(name) {}

  OptionKind getKind() const { return Kind; }

  llvm::StringLiteral getConfigName() const { return configName; }

  virtual ~DriverConfig() = default;

  virtual std::unique_ptr<DriverConfig> clone() {
    return std::make_unique<DriverConfig>(*this);
  }

private:
  const OptionKind Kind;
  const llvm::StringLiteral configName;
};

class KernelLibraryDriverConfig : public DriverConfig {

public:
  KernelLibraryDriverConfig(std::filesystem::path path)
      : DriverConfig(OptionKind::kKernelLibrary, "kernel-library-path"),
        kernelLibraryPath(std::move(path)) {}

  static bool classof(const DriverConfig *config) {
    return config->getKind() == OptionKind::kKernelLibrary;
  }

  const std::filesystem::path &getPath() const { return kernelLibraryPath; }

  std::unique_ptr<DriverConfig> clone() override {
    return std::make_unique<KernelLibraryDriverConfig>(*this);
  }

private:
  std::filesystem::path kernelLibraryPath;
};

class DeviceConfig : public DriverConfig {
public:
  DeviceConfig(std::string device, size_t deviceID,
               const std::optional<DeviceSpec> &deviceSpec)
      : DriverConfig(OptionKind::kDevice, "device-configuration"),
        deviceSpec(deviceSpec), deviceLabel(std::move(device)),
        deviceID(deviceID) {}

  static bool classof(const DriverConfig *config) {
    return config->getKind() == OptionKind::kDevice;
  }

  const std::string &getDeviceLabel() const { return deviceLabel; }
  size_t getDeviceID() const { return deviceID; }

  std::unique_ptr<DriverConfig> clone() override {
    return std::make_unique<DeviceConfig>(*this);
  }

  std::optional<DeviceSpec> deviceSpec;

private:
  std::string deviceLabel;
  size_t deviceID;
};

class TileBasedFusionDriverConfig : public DriverConfig {
public:
  TileBasedFusionDriverConfig(bool tileBasedFusion)
      : DriverConfig(OptionKind::kTileBasedFusion, "tile-based-fusion"),
        tileBasedFusion(tileBasedFusion) {}

  static bool classof(const DriverConfig *config) {
    return config->getKind() == OptionKind::kTileBasedFusion;
  }

  bool getTileBasedFusion() const { return tileBasedFusion; }

  std::unique_ptr<DriverConfig> clone() override {
    return std::make_unique<TileBasedFusionDriverConfig>(*this);
  }

private:
  bool tileBasedFusion;
};

} // namespace M

#endif // GENERICML_COMMONDRIVER_DRIVERCONFIG_H
