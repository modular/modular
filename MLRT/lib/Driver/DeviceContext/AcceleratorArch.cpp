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
// This file implements the per-vendor accelerator arch queries.
//
//===----------------------------------------------------------------------===//

#include "AcceleratorArch.h"

#include "CUDA/NVML.h"
#include "HIP/HIPDriver.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "Metal/MetalDriver.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace M::Driver {

MODULAR_CXX_EXPORT std::string cudaCCToSM(int major, int minor) {
  std::string sm = "sm_";
  int cc = major * 10 + minor;
  sm += std::to_string(cc);

  // For architectures above 90 we use the "a" variants to enable hardware
  // specific features at the cost of backward and forward compatibility.
  if (cc >= 90)
    sm += "a";

  return sm;
}

ErrorOr<std::string> getFirstCUDAArchName() {
  DEFINE_OR_RETURN_ERROR(const NVML &, nvml, NVML::get());

  nvmlDevice_t device;
  if (nvml.nvmlDeviceGetHandleByIndex(0, &device) != NVML::SUCCESS)
    RETURN_ERROR("NVML: Unable to get device index");

  int major, minor;
  if (nvml.nvmlDeviceGetCudaComputeCapability(device, &major, &minor) !=
      NVML::SUCCESS)
    RETURN_ERROR("NVML: Unable to get compute capability");

  return cudaCCToSM(major, minor);
}

MODULAR_CXX_EXPORT Error formatHipError(const char *file, int line,
                                        const char *func,
                                        HIPDriver::Result result,
                                        std::string extra_msg) {
  ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
  const char *errName = nullptr;
  const char *errStr = nullptr;

  if (!errDriver) {
    // Ignore errors and use nullptr of char* instead.
    errDriver->hipDrvGetErrorName(result, &errName);
    errDriver->hipDrvGetErrorString(result, &errStr);
  }

  if (!extra_msg.empty())
    extra_msg = std::string(" - ") + extra_msg;

  if ((errName != nullptr) && (errStr != nullptr)) {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("HIP call failed: ") +
                                          errName + " (" + errStr + ")" +
                                          extra_msg);
  } else if ((errName != nullptr) || (errStr != nullptr)) {
    return DeviceContext::formatError(
        file, line, func,
        llvm::Twine("HIP call failed: ") +
            (errName != nullptr ? errName : errStr) + extra_msg);
  } else {
    return DeviceContext::formatError(file, line, func,
                                      llvm::Twine("HIP call failed: ") +
                                          llvm::Twine(result) + extra_msg);
  }
}

MODULAR_CXX_EXPORT ErrorOr<std::string> getHIPArchName(hipDevice_t device) {
  // We do need the full AMD GPU architecture name in order to compie for the
  // right LLVM target. Parsing via KFD, as used by amd-smi and rocm-smi seems
  // to be one of the two reliable ways to get the full architecture name. The
  // other way, which we use here, is through the HSA API and some PCIe device
  // matching with the HIP API. The HIP API directly does provide the
  // architecture name in the `hipDeviceProp_t::gcnArchName` field, but it is
  // not always set. Likewise, `hipDeviceAttributeGcnArchName` is now
  // deprecated. Major & minor versions are available via
  // `hipDeviceAttributeComputeCapabilityMajor` and
  // `hipDeviceAttributeComputeCapabilityMinor`, but lack the last digit.

  DEFINE_OR_RETURN_ERROR(const HIPDriver &, driver, HIPDriver::get());

  // Initialize HSA runtime if available.
  if (driver.HsaInit() != HSA_STATUS_SUCCESS)
    return Error("Failed to initialize HSA runtime");

  // Get HIP device PCIe info via device attributes
  int pciBusId = 0, pciDeviceId = 0, pciDomainId = 0;
  HIP_CHECK_ERR(driver.hipDeviceGetAttribute(
      &pciBusId, hipDeviceAttribute_t::hipDeviceAttributePciBusId, device));
  HIP_CHECK_ERR(driver.hipDeviceGetAttribute(
      &pciDeviceId, hipDeviceAttribute_t::hipDeviceAttributePciDeviceId,
      device));
  HIP_CHECK_ERR(driver.hipDeviceGetAttribute(
      &pciDomainId, hipDeviceAttribute_t::hipDeviceAttributePciDomainID,
      device));

  // Data structure to pass to HSA agent and ISA iteration callbacks.
  struct SearchData {
    const HIPDriver *driver;
    uint32_t targetPciBusID;
    uint32_t targetPciDeviceID;
    uint32_t targetPciDomainID;
    std::string archName;
  };

  SearchData search = {&driver, static_cast<uint32_t>(pciBusId),
                       static_cast<uint32_t>(pciDeviceId),
                       static_cast<uint32_t>(pciDomainId), ""};

  // Callback function for HSA agent iteration.
  auto agentCallback = [](hsaAgent_t agent, void *data) -> hsaStatus_t {
    auto *search = static_cast<SearchData *>(data);

    // Match by PCIe address of HIP and HSA device.
    uint32_t hsaDomain = 0;
    uint32_t hsaBdfId = 0;

    // Get PCIe domain. On failure, continue to next agent.
    if (search->driver->HsaAgentGetInfo(
            agent, hsaAgentInfo_t::HSA_AMD_AGENT_INFO_DOMAIN, &hsaDomain) !=
            HSA_STATUS_SUCCESS ||
        search->driver->HsaAgentGetInfo(
            agent, hsaAgentInfo_t::HSA_AMD_AGENT_INFO_BDFID, &hsaBdfId) !=
            HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;

    // BDFID is typically encoded as: (bus << 8) | (device << 3) | function
    uint32_t hsaBus = (hsaBdfId >> 8) & 0xFF;
    uint32_t hsaDevice = (hsaBdfId >> 3) & 0x1F;
    // uint32_t hsaFunction = hsaBdfId & 0x7;

    // Compare with HIP device PCIe info. On failure, continue to next agent.
    if (hsaDomain != search->targetPciDomainID ||
        hsaBus != search->targetPciBusID ||
        hsaDevice != search->targetPciDeviceID)
      return HSA_STATUS_SUCCESS;

    // Callback to iterate through ISAs for this agent.
    auto isaCallback = [](hsaIsa_t isa, void *data) -> hsaStatus_t {
      auto *search = static_cast<SearchData *>(data);

      // First, get the length of the ISA name.
      uint32_t nameLen = 0;
      if (search->driver->HsaIsaGetInfo(isa,
                                        hsaIsaInfo_t::HSA_ISA_INFO_NAME_LENGTH,
                                        &nameLen) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

      // Continue if the ISA name length is invalid.
      if (nameLen == 0 || nameLen > 256)
        return HSA_STATUS_SUCCESS;

      // Allocate memory for the ISA name
      std::vector<char> isaName(nameLen);

      // Get the actual ISA name. Continue to next ISA if it fails.
      if (search->driver->HsaIsaGetInfo(isa, hsaIsaInfo_t::HSA_ISA_INFO_NAME,
                                        isaName.data()) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;

      // Check if this ISA name contains "gfx".
      std::string isaStr(isaName.data(), nameLen > 0
                                             ? nameLen - 1
                                             : 0); // Remove null terminator.
      size_t gfxPos = isaStr.find("gfx");
      if (gfxPos != std::string::npos) {
        // Find the end of the digits after "gfx".
        size_t start = gfxPos;
        // Start after "gfx".
        size_t end = start + 3;

        // Continue until we find a non-digit character.
        while (end < isaStr.length() && std::isdigit(isaStr[end]))
          ++end;

        // Extract the gfx* part.
        std::string gfxArch = isaStr.substr(start, end - start);
        search->archName = gfxArch;
        // Stop ISA iteration.
        return HSA_STATUS_INFO_BREAK;
      }

      // Continue ISA iteration.
      return HSA_STATUS_SUCCESS;
    };

    // Iterate through ISAs for this agent. Continue agent iteration if it
    // fails.
    if (search->driver->HsaAgentIterateIsas(agent, isaCallback, search) !=
        HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;

    // If we found an architecture, stop agent iteration.
    if (search->archName.size())
      return HSA_STATUS_INFO_BREAK;

    // Continue agent iteration.
    return HSA_STATUS_SUCCESS;
  };

  // Iterate through all HSA agents to find our device.
  if (driver.HsaIterateAgents(agentCallback, &search) != HSA_STATUS_SUCCESS) {
    driver.HsaShutDown();
    return Error("Failed to iterate HSA agents");
  }

  // Clean up HSA runtime.
  driver.HsaShutDown();

  if (search.archName.empty())
    return Error("Could not determine architecture name for HIP device");

  return search.archName;
}

#if MODULAR_MACOS

MODULAR_CXX_EXPORT ErrorOr<int> getMetalComputeCapability(StringRef name) {
  if (name.empty())
    return 3;

  if (name.contains_insensitive("M6"))
    return 6;

  if (name.contains_insensitive("M5"))
    return 5;

  if (name.contains_insensitive("M4"))
    return 4;

  if (name.contains_insensitive("M3"))
    return 3;

  if (name.contains_insensitive("M2"))
    return 2;

  if (name.contains_insensitive("M1"))
    return 1;

  RETURN_ERROR("Unknown device name");
}

/// Returns true if the installed SDK supports the indicated Major.Minor
/// revision of the Metal Shading Language.
/// Parses the SDK header once on first call and caches all discovered
/// MTLLanguageVersion values for the process lifetime.
static bool sdkSupportsMSLVersion(int maj, int min) {
  static const auto supportedVersions = [] {
    std::set<std::pair<int, int>> versions;

    // Resolve the active Xcode developer directory.
    std::filesystem::path devDir;
    if (const char *env = std::getenv("DEVELOPER_DIR")) {
      devDir = env;
    } else {
      // Same symlink that `xcode-select -p` resolves.
      std::error_code ec;
      devDir = std::filesystem::read_symlink("/var/db/xcode_select_link", ec);
      if (ec)
        return versions;
    }

    // Header that defines the MTLLanguageVersion enum.
    std::filesystem::path headerPath =
        devDir / "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk" /
        "System/Library/Frameworks/Metal.framework" /
        "Versions/A/Headers/MTLLibrary.h";
    auto bufOrErr = llvm::MemoryBuffer::getFile(headerPath.c_str());
    if (!bufOrErr)
      return versions;

    // Scan for MTLLanguageVersionX_Y enum entries.
    StringRef prefix = "MTLLanguageVersion";
    for (llvm::line_iterator it(**bufOrErr, /*SkipBlanks=*/true, '\0');
         !it.is_at_end(); ++it) {
      StringRef line = *it;
      auto pos = line.find(prefix);
      if (pos == StringRef::npos)
        continue;
      StringRef suffix = line.substr(pos + prefix.size());
      auto [majStr, rest] = suffix.split('_');
      int major = 0, minor = 0;
      // majStr is clean (e.g. "1"); rest may have trailing text
      // (e.g. "1 API_AVAILABLE(...)"), so use consumeInteger.
      if (!majStr.getAsInteger(10, major) && !rest.consumeInteger(10, minor))
        versions.emplace(major, minor);
    }
    return versions;
  }();

  return supportedVersions.count({maj, min}) > 0;
}

MODULAR_CXX_EXPORT ErrorOr<std::string> getMetalArchName(MTL::Device *d) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  // CreateSystemDefaultDevice returns a +1 (Create-rule) reference that a
  // pool cannot reclaim; own it so the nullptr-device path does not leak.
  NS::SharedPtr<MTL::Device> createdDevice;
  if (d == nullptr)
    createdDevice = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  MTL::Device *device = d ? d : createdDevice.get();
  if (device == nullptr)
    RETURN_ERROR("Failed to create Metal device.");
  std::string name = device->name()->cString(NS::UTF8StringEncoding);

  DEFINE_OR_RETURN_ERROR(int, cc, getMetalComputeCapability(name));
  std::string archName = std::to_string(cc);
  if (device->supportsFamily(MTL::GPUFamilyMetal4) &&
      sdkSupportsMSLVersion(4, 0)) {
    // Prefer to use latest supported Metal version by default, but only if
    // the installed Xcode's Metal compiler can actually handle Metal 4.0 AIR.
    archName += "-metal4";
  }
  return archName;
}

#endif // MODULAR_MACOS

} // namespace M::Driver
