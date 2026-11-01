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

#include "Context.h"
#include "Driver.h"
#include "HAL/M_driver_device.h"

#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <dlpack/dlpack.h>

#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <string>

using namespace llvm;

namespace {
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

/// Returns the number of GPU cores (Apple's analogue of a CUDA multiprocessor
/// count) by reading the `gpu-core-count` property of the `AGXAccelerator`
/// IORegistry entry, or `nullopt` if it cannot be determined. Mirrors the
/// AsyncRT Metal backend (`MetalDevice::getAttribute`).
static std::optional<int32_t> gpuCoreCount() {
  static std::once_flag gpuCoreCountFlag;
  static std::optional<int32_t> cachedCount;

// Host is built for macOS 11.0, but `kIOMainPortDefault` is only available on
// macOS >= 12.0. Apple GPU support requires macOS >= 15, so the warning is
// safe to ignore.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
  std::call_once(gpuCoreCountFlag, []() {
    io_iterator_t iterator;
    CFMutableDictionaryRef matchDict = IOServiceMatching("AGXAccelerator");
    if (IOServiceGetMatchingServices(kIOMainPortDefault, matchDict,
                                     &iterator) != KERN_SUCCESS)
      return;

    io_service_t service;
    while ((service = IOIteratorNext(iterator)) != 0) {
      CFMutableDictionaryRef properties = nullptr;
      if (IORegistryEntryCreateCFProperties(
              service, &properties, kCFAllocatorDefault, 0) == KERN_SUCCESS) {
        CFNumberRef coreCountRef = (CFNumberRef)CFDictionaryGetValue(
            properties, CFSTR("gpu-core-count"));
        if (coreCountRef && CFGetTypeID(coreCountRef) == CFNumberGetTypeID()) {
          int coreCount = 0;
          CFNumberGetValue(coreCountRef, kCFNumberIntType, &coreCount);
          cachedCount = coreCount;
        }
        if (properties)
          CFRelease(properties);
      }
      IOObjectRelease(service);
      if (cachedCount)
        break;
    }
    IOObjectRelease(iterator);
  });
#pragma clang diagnostic pop
  return cachedCount;
}
} // namespace

DriverResult<int> M_driver_device::getComputeCapability(StringRef name) {
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

  return tl::unexpected(driver->status.create("Unknown device name"));
}

M_driver_status_t M_driver_device::getProperty(llvm::StringRef name,
                                               void *value) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (name == M_DRIVER_DEVICE_PROPERTY_NAME) {
    std::strncpy(static_cast<char *>(value),
                 mDevice->name()->cString(NS::UTF8StringEncoding),
                 M_DRIVER_DEVICE_PROPERTY_MAX_LEN);
    return M_DRIVER_STATUS_SUCCESS;
  }

  if (name == M_DRIVER_DEVICE_PROPERTY_ARCH) {
    if (mDevice == nullptr)
      return driver->status.create("Failed to create Metal device.");
    std::string name = mDevice->name()->cString(NS::UTF8StringEncoding);

    auto cc = getComputeCapability(name);
    if (!cc)
      return cc.error();
    std::string archName = std::to_string(*cc);
    if (mDevice->supportsFamily(MTL::GPUFamilyMetal4) &&
        sdkSupportsMSLVersion(4, 0)) {
      // Prefer to use latest supported Metal version by default, but only if
      // the installed Xcode's Metal compiler can actually handle Metal 4.0 AIR.
      archName += "-metal4";
    }
    std::strncpy(static_cast<char *>(value), archName.c_str(),
                 M_DRIVER_DEVICE_PROPERTY_MAX_LEN);

    return M_DRIVER_STATUS_SUCCESS;
  }

  if (name == M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED) {
    if (mDevice == nullptr)
      return M_DRIVER_STATUS_BAD_DEVICE;
    // Queried: an Intel Mac's discrete GPU is also a Metal device.
    *static_cast<int64_t *>(value) = mDevice->hasUnifiedMemory() ? 1 : 0;
    return M_DRIVER_STATUS_SUCCESS;
  }

  if (name == M_DRIVER_DEVICE_PROPERTY_MAX_ALLOCATION_SIZE) {
    if (mDevice == nullptr)
      return M_DRIVER_STATUS_BAD_DEVICE;
    *static_cast<uint64_t *>(value) = mDevice->maxBufferLength();
    return M_DRIVER_STATUS_SUCCESS;
  }

  if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE ||
      name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLMetal;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }

  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

extern "C" M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                                   int64_t *count) {
  if (count == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  if (driver->devices.size() == 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  *count = driver->devices.size();
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                                 int64_t id,
                                                 M_driver_device_t *device) {
  if (device == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  if (driver->devices.size() == 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  if (id < 0 || size_t(id) >= driver->devices.size())
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = &driver->devices[id];
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                                      const char *property_name,
                                                      void *value) {
  return device->getProperty(property_name, value);
}

extern "C" M_driver_status_t M_driver_device_attribute(M_driver_device_t device,
                                                       int32_t attribute,
                                                       int32_t *value) {
  if (!device)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (!value)
    return M_DRIVER_STATUS_INVALID_ARG;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::Device *dev = device->mDevice;

  // Attribute codes are CUDA-style `CUdevice_attribute` values, matching
  // `Device::Attribute` in MLRT's HALDeviceContext.h. Values (and which
  // attributes are unsupported on Apple GPUs) mirror the AsyncRT Metal backend
  // `MetalDevice::getAttribute`, keeping the HAL plugin and the built-in Metal
  // device context in agreement.
  enum : int32_t {
    MAX_THREADS_PER_BLOCK = 1,
    MAX_BLOCK_DIM_X = 2,
    MAX_BLOCK_DIM_Y = 3,
    MAX_BLOCK_DIM_Z = 4,
    MAX_GRID_DIM_X = 5,
    MAX_GRID_DIM_Y = 6,
    MAX_GRID_DIM_Z = 7,
    MAX_SHARED_MEMORY_PER_BLOCK = 8,
    MULTIPROCESSOR_COUNT = 16,
    COMPUTE_CAPABILITY_MAJOR = 75,
    COMPUTE_CAPABILITY_MINOR = 76,
    METAL_DISABLE_LOGGING = 110,
  };

  switch (attribute) {
  // 'Maximum threads per threadgroup', same for Apple7 - Apple10.
  case MAX_THREADS_PER_BLOCK:
    *value = 1024;
    return M_DRIVER_STATUS_SUCCESS;

  // Block dimensions map to max threads per threadgroup.
  case MAX_BLOCK_DIM_X:
    *value = static_cast<int32_t>(dev->maxThreadsPerThreadgroup().width);
    return M_DRIVER_STATUS_SUCCESS;
  case MAX_BLOCK_DIM_Y:
    *value = static_cast<int32_t>(dev->maxThreadsPerThreadgroup().height);
    return M_DRIVER_STATUS_SUCCESS;
  case MAX_BLOCK_DIM_Z:
    *value = static_cast<int32_t>(dev->maxThreadsPerThreadgroup().depth);
    return M_DRIVER_STATUS_SUCCESS;

  // There is no limit to the size of a grid.
  case MAX_GRID_DIM_X:
  case MAX_GRID_DIM_Y:
  case MAX_GRID_DIM_Z:
    *value = std::numeric_limits<int32_t>::max();
    return M_DRIVER_STATUS_SUCCESS;

  // 'Maximum total threadgroup memory allocation'.
  case MAX_SHARED_MEMORY_PER_BLOCK:
    *value = static_cast<int32_t>(dev->maxThreadgroupMemoryLength());
    return M_DRIVER_STATUS_SUCCESS;

  // Apple's analogue of the multiprocessor count is the GPU core count.
  case MULTIPROCESSOR_COUNT: {
    std::optional<int32_t> cores = gpuCoreCount();
    if (!cores)
      return device->driver->status.create(
          "MULTIPROCESSOR_COUNT unavailable: could not read "
          "AGXAccelerator gpu-core-count");
    *value = *cores;
    return M_DRIVER_STATUS_SUCCESS;
  }

  case COMPUTE_CAPABILITY_MAJOR: {
    std::string name = (dev && dev->name())
                           ? dev->name()->cString(NS::UTF8StringEncoding)
                           : std::string();
    DriverResult<int> cc = device->getComputeCapability(name);
    if (!cc)
      return cc.error();
    *value = *cc;
    return M_DRIVER_STATUS_SUCCESS;
  }
  case COMPUTE_CAPABILITY_MINOR:
    *value = 0;
    return M_DRIVER_STATUS_SUCCESS;

  case METAL_DISABLE_LOGGING:
    *value = 0;
    return M_DRIVER_STATUS_SUCCESS;

  default:
    return device->driver->status.create("device attribute " +
                                         std::to_string(attribute) +
                                         " not supported for Apple GPU");
  }
}
