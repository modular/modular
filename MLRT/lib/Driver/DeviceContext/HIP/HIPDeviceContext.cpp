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
// This file implements the HIP device context.
//
//===----------------------------------------------------------------------===//

#include "HIPDeviceContext.h"
#include "HIPDeviceGraphBuilder.h"

#include "HIPDriver.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "MLRT/Driver/DeviceContext/LaunchTraceRegistry.h"
#include "MLRT/lib/Driver/DeviceContext/HostMemoryInfo.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "Support/ErrorOr.h"
#include "Support/HashUtils.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "Support/max_license/Tracy/TracyRocm.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Process.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

// Helpers for logging.
#define TWINE(val) (llvm::Twine(val))

#define HIP_CHECK_ERR(hip_call)                                                \
  do {                                                                         \
    HIPDriver::Result hipResult = hip_call;                                    \
    if (hipResult != HIPDriver::SUCCESS) {                                     \
      return formatHipError(__FILE__, __LINE__, __FUNCTION__, hipResult);      \
    }                                                                          \
  } while (0)

#define HIP_CHECK(hip_call)                                                    \
  do {                                                                         \
    HIPDriver::Result hipResult = hip_call;                                    \
    if (hipResult != HIPDriver::SUCCESS) {                                     \
      M::Error err =                                                           \
          formatHipError(__FILE__, __LINE__, __FUNCTION__, hipResult);         \
      /* We are not in a context where we can return an Error here. So we   */ \
      /* print the message and abort().                                     */ \
      std::cerr << err << "\n";                                                \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_IS_HIP_ERR(dev)                                                  \
  do {                                                                         \
    if ((dev).api() != Device::hipAPI) {                                       \
      RETURN_ERROR("Not a HIP device.");                                       \
    }                                                                          \
  } while (0)

#define HIP_WITH_CTX_ERR(ctx)                                                  \
  RETURN_IF_PENDING_ERROR(ctx);                                                \
  HIPDeviceScope __hipWithContext((ctx).hipDevice(), (ctx).hipDriver());       \
  HIP_CHECK_ERR(__hipWithContext.status());

// This MUST be called during all synchronous API calls like
// hipStreamSynchronize. To reduce overhead, we omit polling during asynchronous
// API calls like hipCallKernelDirect. The call to hipEventQuery is not free!
#define POLL_PENDING_EVENTS(ctx) RETURN_IF_ERROR((ctx).pollPendingEvents());

#define HIP_WITH_CTX(ctx)                                                      \
  HIPDeviceScope __hipWithContext((ctx).hipDevice(), (ctx).hipDriver());       \
  HIP_CHECK(__hipWithContext.status());

#define HIP_WITH_DEV_ERR(dev)                                                  \
  HIPDeviceScope __hipWithContext((dev).hipDevice(), (dev).hipDriver());       \
  HIP_CHECK_ERR(__hipWithContext.status());

#define HIP_WITH_DEV(dev)                                                      \
  HIPDeviceScope __hipWithContext((dev).hipDevice(), (dev).hipDriver());       \
  HIP_CHECK(__hipWithContext.status());

#define MAYBE_SYNC(ctx)                                                        \
  do {                                                                         \
    const HIPDeviceContext &hip_ctx = (ctx);                                   \
    if (hip_ctx.syncMode() &&                                                  \
        !isStreamCapturing(hip_ctx.hipDriver(), hip_ctx.hipStream())) {        \
      HIP_CHECK_ERR(                                                           \
          hip_ctx.hipDriver().hipStreamSynchronize(hip_ctx.hipStream()));      \
    }                                                                          \
  } while (0)

namespace M::Driver {

// Avoid using nullptr for the "device context pointer". We add/subtract the
// following constant when handling the hipDevice_t as a pointer.
constexpr intptr_t devicePtrOffset =
    0x51413; // First 5 digits of pi in reverse.

static void *hipDeviceToPtr(hipDevice_t id) {
  return reinterpret_cast<void *>(static_cast<intptr_t>(id) + devicePtrOffset);
}

static hipDevice_t ptrToHipDevice(void *ptr) {
  return static_cast<hipDevice_t>(reinterpret_cast<intptr_t>(ptr) -
                                  devicePtrOffset);
}

// The helper class HIPDeviceScope is used to set and restore the hipDevice_t
// around interactions with the HIP driver.
class HIPDeviceScope {
public:
  HIPDeviceScope(hipDevice_t device, const HIPDriver &driver)
      : driver_(driver), device_(device), prevDevice_(-1),
        status_(HIPDriver::SUCCESS) {
    driver_.hipGetDevice(&prevDevice_);
    if (prevDevice_ == device_)
      return;
    status_ = driver_.hipSetDevice(device_);
  }

  ~HIPDeviceScope() {
    if ((prevDevice_ == device_) || (status_ != HIPDriver::SUCCESS))
      return;

    // Only pop the context if push was necessary and successful.
    hipDevice_t device = -1;
    driver_.hipGetDevice(&device);
    driver_.hipSetDevice(prevDevice_);
    assert(device == device_);
  }

  HIPDriver::Result status() { return status_; }

private:
  const HIPDriver &driver_;
  const hipDevice_t device_;
  hipDevice_t prevDevice_;
  HIPDriver::Result status_;
};

static Error formatHipError(const char *file, int line, const char *func,
                            HIPDriver::Result result,
                            std::string extra_msg = "") {
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

static bool isStreamCapturing(const HIPDriver &driver, hipStream_t stream) {
  hipStreamCaptureStatus status =
      hipStreamCaptureStatus::hipStreamCaptureStatusNone;
  if (driver.hipStreamIsCapturing(stream, &status) != HIPDriver::SUCCESS)
    return false;

  return status != hipStreamCaptureStatus::hipStreamCaptureStatusNone;
}

// ===== HIPDevice ===== //
// Plan:
// DeviceContext::Attribute will reuse the CUDA enum mapping
// CUDADeviceContext::getAttribute can thus use the enum values directly
// HIPDeviceContext will map the DeviceContext attributes to their HIP
// equivalents
static ErrorOr<hipDeviceAttribute_t>
gethipDeviceAttribute(Device::Attribute attr) {
#define ENTRY(X, Y)                                                            \
  case Device::Attribute::X:                                                   \
    return hipDeviceAttribute_t::hipDeviceAttribute##Y

#define RAISE_ENTRY(X)                                                         \
  case Device::Attribute::X:                                                   \
    RETURN_ERROR("Attribute \"" #X "\" not supported for AMD GPU.")

  switch (attr) {
    ENTRY(MAX_THREADS_PER_BLOCK, MaxThreadsPerBlock);
    ENTRY(MAX_BLOCK_DIM_X, MaxBlockDimX);
    ENTRY(MAX_BLOCK_DIM_Y, MaxBlockDimY);
    ENTRY(MAX_BLOCK_DIM_Z, MaxBlockDimZ);
    ENTRY(MAX_GRID_DIM_X, MaxGridDimX);
    ENTRY(MAX_GRID_DIM_Y, MaxGridDimY);
    ENTRY(MAX_GRID_DIM_Z, MaxGridDimZ);
    ENTRY(MAX_SHARED_MEMORY_PER_BLOCK, MaxSharedMemoryPerBlock);
    ENTRY(WARP_SIZE, WarpSize);
    ENTRY(MAX_REGISTERS_PER_BLOCK, MaxRegistersPerBlock);
    ENTRY(CLOCK_RATE, ClockRate);
    ENTRY(MULTIPROCESSOR_COUNT, MultiprocessorCount);
    ENTRY(MAX_THREADS_PER_MULTIPROCESSOR, MaxThreadsPerMultiProcessor);
    ENTRY(COMPUTE_CAPABILITY_MAJOR, ComputeCapabilityMajor);
    ENTRY(COMPUTE_CAPABILITY_MINOR, ComputeCapabilityMinor);
    RAISE_ENTRY(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    ENTRY(MAX_REGISTERS_PER_MULTIPROCESSOR, MaxRegistersPerMultiprocessor);
    RAISE_ENTRY(COOPERATIVE_LAUNCH);
    RAISE_ENTRY(MAX_BLOCKS_PER_MULTIPROCESSOR);
    RAISE_ENTRY(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    RAISE_ENTRY(MAX_ACCESS_POLICY_WINDOW_SIZE);
    RAISE_ENTRY(METAL_DISABLE_LOGGING);
    // No `default:` to ensure that all possible enum values are handled.
  }
#undef RAISE_ENTRY
#undef ENTRY
  // If a unsupported value is passed as a numerical value, fail with a generic
  // message here. This can happen if for example we create an unsupported value
  // from within Mojo.
  RETURN_ERROR("Attribute not defined for AMD GPU");
}

HIPDevice::HIPDevice(const HIPDriver &driver, int id, hipDevice_t device,
                     std::string name, std::string arch, MemInfo memInfo,
                     size_t allocAlignment)
    : Device(api, name, id, memInfo, allocAlignment), arch_(arch),
      driver_(driver), device_(device) {}

HIPDevice::~HIPDevice() {
  // Destroy the function map before destroying other device structures.
  clearFunctionCache();
  clearMemoryManager();
}

const HIPDriver &HIPDevice::hipDriver() const { return driver_; }

hipDevice_t HIPDevice::hipDevice() const { return device_; }

ErrorOr<int> HIPDevice::getApiVersion() const {
  int driverVersion = 0;
  HIP_CHECK_ERR(driver_.hipDriverGetVersion(&driverVersion));
  return driverVersion;
}

ErrorOr<DriverVersion &> HIPDevice::getDriverVersion() const {
  static DriverVersion version("0.0.0");
  return version;
}

ErrorOrSuccess HIPDevice::isCompatible() const {
#if !MODULAR_LINUX
  return Error("AMD GPUs are only supported on Linux based systems.");
#endif

  DEFINE_OR_RETURN_ERROR(int, computeCapability, computeCapability());
  if (computeCapability < MIN_COMPUTE_CAPABILITY) {
    RETURN_ERROR("Only HIP architectures " + TWINE(MIN_COMPUTE_CAPABILITY) +
                 " or higher are supported");
  }

  int driverVersionCoded = 0;
  HIP_CHECK_ERR(driver_.hipDriverGetVersion(&driverVersionCoded));

  // Deconstruct the driverVersionCoded into major, minor and patch.
  constexpr int driverVersionMinorFactor = 100000;
  constexpr int driverVersionMajorFactor = 10000000;

  int driverVersionPatch = driverVersionCoded % driverVersionMinorFactor;
  int driverVersionMinor =
      (driverVersionCoded / driverVersionMinorFactor) %
      (driverVersionMajorFactor / driverVersionMinorFactor);
  int driverVersionMajor = driverVersionCoded / driverVersionMajorFactor;

  if (driverVersionMajor > MIN_HIP_DRIVER_VERSION_MAJOR) {
    return success();
  } else if (driverVersionMajor == MIN_HIP_DRIVER_VERSION_MAJOR) {
    if (driverVersionMinor > MIN_HIP_DRIVER_VERSION_MINOR) {
      return success();
    } else if (driverVersionMinor == MIN_HIP_DRIVER_VERSION_MINOR) {
      if (driverVersionPatch >= MIN_HIP_DRIVER_VERSION_PATCH) {
        return success();
      }
    }
  }

  RETURN_ERROR("Your current AMD GPU driver version is not supported."
               "A minimum driver version of " +
               TWINE(MIN_HIP_DRIVER_VERSION_MAJOR) + "." +
               TWINE(MIN_HIP_DRIVER_VERSION_MINOR) + "." +
               TWINE(MIN_HIP_DRIVER_VERSION_PATCH) +
               " is required. The driver version detected is: " +
               TWINE(driverVersionMajor) + "." + TWINE(driverVersionMinor) +
               "." + TWINE(driverVersionPatch) +
               ".\nFor information about how to update your driver, see "
               "https://modul.ar/gpu-requirements.");
}

ErrorOr<int> HIPDevice::computeCapability() const {
  int computeMajor = 0, computeMinor = 0;
  HIP_CHECK_ERR(driver_.hipDeviceGetAttribute(
      &computeMajor,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMajor,
      hipDevice()));
  HIP_CHECK_ERR(driver_.hipDeviceGetAttribute(
      &computeMinor,
      hipDeviceAttribute_t::hipDeviceAttributeComputeCapabilityMinor,
      hipDevice()));
  return 10 * computeMajor + computeMinor;
}

ErrorOr<bool> HIPDevice::canAccess(const Device &peer) const {
  if (id() == peer.id()) {
    // Always disable self access (same behaviour as hipDeviceCanAccessPeer).
    return false;
  }

  const HIPDevice *peerDevice = HIPDevice::verified_cast(&peer);
  if (peerDevice == nullptr) {
    RETURN_ERROR("Peer is not a HIP device");
  }

  int canAccess = 0;
  HIP_CHECK_ERR(driver_.hipDeviceCanAccessPeer(&canAccess, hipDevice(),
                                               peerDevice->hipDevice()));
  return canAccess == 1;
}

ErrorOrSuccess HIPDevice::enablePeerAccessImpl(const Device &peer) {
  HIP_WITH_DEV_ERR(*this);

  const HIPDevice &peerHipDev = HIPDevice::cast(peer);
  DEFINE_OR_RETURN_ERROR(bool, canAccessPeer, canAccess(peerHipDev));
  if (!canAccessPeer) {
    RETURN_ERROR("Cannot enable peer access from GPU " + Twine(id()) +
                 " to GPU " + Twine(peer.id()) +
                 ": hardware does not support P2P access");
  }

  // Tolerate "already enabled" as success to keep idempotent.
  HIPDriver::Result res =
      driver_.hipDeviceEnablePeerAccess(peerHipDev.hipDevice(), /*flags=*/0);
  if (res != HIPDriver::SUCCESS) {
    const char *name = nullptr;
    driver_.hipDrvGetErrorName(res, &name);
    if (!(name && (Twine(name) == "hipErrorPeerAccessAlreadyEnabled")))
      return formatHipError(__FILE__, __LINE__, __FUNCTION__, res);
    // HIP also left the error as this thread's last error, which other HIP
    // users in the process (torch, for one) raise after their next launch.
    // It is handled here, so consume it.
    driver_.hipGetLastError();
  }

  // Also allow access to the peer's stream-ordered allocation pool.
  hipMemPool_t pool;
  HIP_CHECK_ERR(
      driver_.hipDeviceGetDefaultMemPool(&pool, peerHipDev.hipDevice()));

  hipMemAccessDesc desc{};
  desc.location.type = hipMemLocationType::hipMemLocationTypeDevice;
  desc.location.id = id();
  desc.flags = hipMemAccessFlags::hipMemAccessFlagsProtReadWrite;
  HIP_CHECK_ERR(driver_.hipMemPoolSetAccess(pool, &desc, /*count=*/1));

  return success();
}

ErrorOr<int> HIPDevice::getAttribute(Attribute attr) const {
  HIP_WITH_DEV_ERR(*this);
  DEFINE_OR_RETURN_ERROR(hipDeviceAttribute_t, attrValue,
                         gethipDeviceAttribute(attr));
  int value = 0;
  HIP_CHECK_ERR(driver_.hipDeviceGetAttribute(&value, attrValue, hipDevice()));
  return value;
}

ErrorOr<bool> HIPDevice::isHostUnified() const {
  // Not hipDeviceAttributeUnifiedAddressing: discrete GPUs set that too.
  HIP_WITH_DEV_ERR(*this);
  int integrated = 0;
  HIP_CHECK_ERR(driver_.hipDeviceGetAttribute(
      &integrated, hipDeviceAttribute_t::hipDeviceAttributeIntegrated,
      hipDevice()));
  return integrated != 0;
}

ErrorOr<std::string> HIPDevice::getPciBusId() const {
  ErrorOr<const HIPDriver &> driverOrError = HIPDriver::get();
  if (driverOrError.isError())
    return driverOrError.takeError();

  const HIPDriver &driver = *driverOrError;

  int pciDomain = 0;
  HIP_CHECK_ERR(driver.hipDeviceGetAttribute(
      &pciDomain, hipDeviceAttribute_t::hipDeviceAttributePciDomainID,
      hipDevice()));

  int pciBus = 0;
  HIP_CHECK_ERR(driver.hipDeviceGetAttribute(
      &pciBus, hipDeviceAttribute_t::hipDeviceAttributePciBusId, hipDevice()));

  int pciDevice = 0;
  HIP_CHECK_ERR(driver.hipDeviceGetAttribute(
      &pciDevice, hipDeviceAttribute_t::hipDeviceAttributePciDeviceId,
      hipDevice()));

  char pciBusId[16];
  snprintf(pciBusId, sizeof(pciBusId), "%04x:%02x:%02x.0", pciDomain, pciBus,
           pciDevice);

  return std::string(pciBusId);
}

ErrorOr<const char *> HIPDevice::getArchName() const { return arch_.c_str(); }

ErrorOr<std::string> HIPDevice::getArchName(hipDevice_t device) {
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

ErrorOr<DeviceContextRef> HIPDevice::createContext() {
  // Make sure that we associate the stream with the correct device.
  HIP_WITH_DEV_ERR(*this);

  hipStream_t stream;
  HIP_CHECK_ERR(this->hipDriver().hipStreamCreate(&stream, 0));

  return HIPDeviceContextRef::create(*this, stream);
}

ErrorOr<int> HIPDevice::numberOfGPUs() {
  static ErrorOr<int> errOrCount = []() -> ErrorOr<int> {
    DEFINE_OR_RETURN_ERROR(const HIPDriver &, driver, HIPDriver::get());
    int result;
    HIP_CHECK_ERR(driver.hipGetDeviceCount(&result));
    return result;
  }();

  if (errOrCount.isError()) {
    return errOrCount.copy();
  }
  int numGPUs = static_cast<int>(*errOrCount);

  // Users assume we use HIP under the hood
  // Multiple options for AMD, we only support one
  // https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
  // Known issue: we could be susceptible to other env vars listed
  // on this page.
  std::optional<std::string> envOr =
      llvm::sys::Process::GetEnv("HIP_VISIBLE_DEVICES");
  if (envOr) {
    // Assuming responsible users.
    // For CUDA, when such an env var is explicitly set to empty
    // string, numGPUs is to be interpreted as zero. For HIP,
    // there is no such explicit call-out. So we interpret like
    // CUDA, to keep things consistent for users who switch
    // systems without RTFM.
    int const numCommas =
        envOr->size() == 0 ? -1 : StringRef(*envOr).count(',');
    numGPUs = std::min(numGPUs, numCommas + 1);
  }
  return numGPUs;
}

int HIPDevice::deviceCount() const {
  ErrorOr<int> errOrCount = numberOfGPUs();
  if (errOrCount.isError()) {
    // Do not propagate an error.
    return 0;
  }
  return *errOrCount;
}

ErrorOrSuccess HIPDevice::chunkAlloc(void **ptr, size_t byteSize, bool isHost) {
  assert(byteSize != 0);
  HIP_WITH_DEV_ERR(*this);
  if (isHost) {
    HIP_CHECK_ERR(driver_.hipHostMalloc(
        ptr, byteSize, hipHostMallocFlags::hipHostMallocPortable));
  } else {
    HIP_CHECK_ERR(driver_.hipMalloc(ptr, byteSize));
  }
  return success();
}

ErrorOrSuccess HIPDevice::chunkFree(void *ptr, bool isHost) {
  HIP_WITH_DEV_ERR(*this);
  if (isHost) {
    HIP_CHECK_ERR(driver_.hipHostFree(ptr));
  } else {
    HIP_CHECK_ERR(driver_.hipFree(ptr));
  }
  return success();
}

// ---- HIP Virtual Memory Management (VMM) ----
//
// Builds the hipMemAllocationProp describing a pinned device granule on this
// device.
static hipMemAllocationProp hipVmmAllocProp(hipDevice_t hipDevice) {
  hipMemAllocationProp memProp = {};
  memProp.type = hipMemAllocationType::hipMemAllocationTypePinned;
  memProp.location.type = hipMemLocationType::hipMemLocationTypeDevice;
  memProp.location.id = hipDevice;
  return memProp;
}

bool HIPDevice::vmmSupported() const {
  // VMM is usable iff the device accepts the granularity query for a pinned
  // device allocation. Probe once and cache, so this reflects actual support
  // (rather than merely "the backend is HIP") and is cheap to call.
  std::call_once(vmmSupportedOnce_,
                 [&]() { vmmSupportedCached_ = !vmmGranularity().isError(); });
  return vmmSupportedCached_;
}

ErrorOr<size_t> HIPDevice::vmmGranularity() const {
  HIP_WITH_DEV_ERR(*this);
  hipMemAllocationProp memProp = hipVmmAllocProp(device_);
  size_t granularity = 0;
  HIP_CHECK_ERR(driver_.hipMemGetAllocationGranularity(
      &granularity, &memProp,
      hipMemAllocationGranularity_flags::hipMemAllocationGranularityMinimum));
  return granularity;
}

ErrorOr<void *> HIPDevice::vmmAddressReserve(size_t byteSize) {
  HIP_WITH_DEV_ERR(*this);
  void *va = nullptr;
  HIP_CHECK_ERR(driver_.hipMemAddressReserve(&va, byteSize, 0, nullptr, 0));
  return va;
}

ErrorOrSuccess HIPDevice::vmmAddressFree(void *va, size_t byteSize) {
  HIP_WITH_DEV_ERR(*this);
  HIP_CHECK_ERR(driver_.hipMemAddressFree(va, byteSize));
  return success();
}

ErrorOr<uint64_t> HIPDevice::vmmCreate(size_t byteSize) {
  HIP_WITH_DEV_ERR(*this);
  hipMemAllocationProp memProp = hipVmmAllocProp(device_);
  hipMemGenericAllocationHandle_t handle = nullptr;
  HIP_CHECK_ERR(driver_.hipMemCreate(&handle, byteSize, &memProp, 0));
  return reinterpret_cast<uint64_t>(handle);
}

ErrorOrSuccess HIPDevice::vmmRelease(uint64_t handle) {
  HIP_WITH_DEV_ERR(*this);
  HIP_CHECK_ERR(driver_.hipMemRelease(
      reinterpret_cast<hipMemGenericAllocationHandle_t>(handle)));
  return success();
}

const std::vector<hipMemAccessDesc> &HIPDevice::vmmAccessDescs() const {
  std::call_once(vmmAccessOnce_, [&]() {
    auto pushDesc = [&](hipDevice_t dev) {
      hipMemAccessDesc d = {};
      d.location.type = hipMemLocationType::hipMemLocationTypeDevice;
      d.location.id = dev;
      d.flags = hipMemAccessFlags::hipMemAccessFlagsProtReadWrite;
      vmmAccessDescs_.push_back(d);
    };
    // The owning device always has access.
    pushDesc(device_);

    // Grant access to every other GPU that can peer-access memory physically
    // resident on this device. Mirrors canAccess(): HIP exposes P2P capability
    // via hipDeviceCanAccessPeer, evaluated over raw ordinals so we do not
    // eagerly initialize peer device contexts.
    int count = 0;
    if (driver_.hipGetDeviceCount(&count) != HIPDriver::SUCCESS)
      return;
    for (int ord = 0; ord < count; ++ord) {
      if (ord == device_)
        continue;
      int canAccessPeer = 0;
      if (driver_.hipDeviceCanAccessPeer(&canAccessPeer, ord, device_) !=
              HIPDriver::SUCCESS ||
          !canAccessPeer)
        continue;
      pushDesc(ord);
    }
  });
  return vmmAccessDescs_;
}

ErrorOrSuccess HIPDevice::vmmMapAndSetAccess(void *va, size_t byteSize,
                                             uint64_t handle) {
  HIP_WITH_DEV_ERR(*this);
  HIP_CHECK_ERR(driver_.hipMemMap(
      va, byteSize, 0,
      reinterpret_cast<hipMemGenericAllocationHandle_t>(handle), 0));
  // Grant read/write access to the owning device and every P2P-capable peer so
  // multi-GPU (expert-parallel) kernels and collectives can read VMM-backed
  // buffers. VMM access does not inherit from hipDeviceEnablePeerAccess.
  //
  // NOTE: `byteSize` is always a whole VMM granule -- the MemoryManager maps
  // and sets access in uniform pageSize_ granules. Do NOT call hipMemSetAccess
  // on a sub-granule / heterogeneous size within one VA reservation: that path
  // fails with hipErrorInvalidValue on ROCm 7.x (ROCm/rocm-systems#2516).
  const std::vector<hipMemAccessDesc> &descs = vmmAccessDescs();
  HIPDriver::Result result =
      driver_.hipMemSetAccess(va, byteSize, descs.data(), descs.size());
  if (result != HIPDriver::SUCCESS) {
    // hipErrorInvalidValue here is usually not about the arguments: HIP caps
    // how much memory a process may have VMM-mapped at once and reports the
    // ceiling with this code. Say so, because the allocator's next move is to
    // fall through to the driver, and the failure then resurfaces hundreds of
    // log lines later as an unexplained hipErrorOutOfMemory. Measured at
    // roughly 1.7 TiB across 8x MI355X with 8 access descriptors per mapping;
    // granting fewer descriptors raises it. See GEX-4056.
    llvm::errs() << "DeviceContext.MemoryManager: hipMemSetAccess failed on "
                    "device "
                 << device_ << " with " << descs.size()
                 << " access descriptors. If this is hipErrorInvalidValue the "
                    "process has likely hit HIP's cap on total VMM-mapped "
                    "memory (GEX-4056); reduce memory_manager_size.\n";
    return formatHipError(__FILE__, __LINE__, __FUNCTION__, result);
  }
  return success();
}

ErrorOrSuccess HIPDevice::vmmUnmap(void *va, size_t byteSize) {
  HIP_WITH_DEV_ERR(*this);
  HIP_CHECK_ERR(driver_.hipMemUnmap(va, byteSize));
  return success();
}

ErrorOr<Device::MemInfo> HIPDevice::getMemoryInfo() const {
  Device::MemInfo memInfo;
  HIP_WITH_DEV_ERR(*this);
  HIP_CHECK_ERR(driver_.hipMemGetInfo(&memInfo.free, &memInfo.total));
#if MODULAR_LINUX
  DEFINE_OR_RETURN_ERROR(bool, unified, isHostUnified());
  if (unified)
    return getHostUnifiedMemoryInfo(memInfo);
#endif
  return memInfo;
}

ErrorOr<void *> HIPDevice::getCurrentDriverContext() const {
  hipDevice_t curDev = 0;
  HIP_CHECK_ERR(driver_.hipGetDevice(&curDev));
  return hipDeviceToPtr(curDev);
}
ErrorOrSuccess HIPDevice::setCurrentDriverContext(void *ctx) {
  HIP_CHECK_ERR(driver_.hipSetDevice(ptrToHipDevice(ctx)));
  return success();
}

const HIPDevice &HIPDevice::cast(const Device &device) {
  return static_cast<const HIPDevice &>(device);
}

HIPDevice &HIPDevice::cast(Device &device) {
  return static_cast<HIPDevice &>(device);
}

const HIPDevice *HIPDevice::verified_cast(const Device *device) {
  if (device->api() != HIPDevice::api) {
    return nullptr;
  }
  return static_cast<const HIPDevice *>(device);
}

ErrorOr<HIPDevice &> HIPDevice::get(int id) {
  // On first call initialize devices and err. devices is only set if there was
  // no error during setup.
  static std::mutex mu;
  static DeviceList *devices = nullptr;
  static const ErrorOrSuccess err = [&]() -> ErrorOrSuccess {
    DEFINE_OR_RETURN_ERROR(int, count, numberOfGPUs());
    devices = new DeviceList(count, nullptr);
    return success();
  }();

  // If we did get an error during setup, skip accessing the device and forward
  // the error.
  if (err.isError()) {
    RETURN_ERROR(err.getError());
  }

  assert(devices != nullptr);
  assert((id >= 0) && (static_cast<size_t>(id) < devices->size()));

  // Get or initialize the Device object.
  std::lock_guard<std::mutex> lock(mu);
  HIPDevice *device = (*devices)[id];
  if (device == nullptr) {
    ErrorOr<const HIPDriver &> errDriver = HIPDriver::get();
    if (errDriver) {
      RETURN_ERROR(llvm::Twine("HIP Driver initialization failed: ") +
                   errDriver.getError());
    }
    const HIPDriver &driver = *errDriver;
    char hipName[256];
    int alignment;

    hipDevice_t hipDevice = static_cast<hipDevice_t>(id);
    HIP_CHECK_ERR(driver.hipDeviceGetName(hipName, sizeof(hipName), hipDevice));
    HIP_CHECK_ERR(driver.hipDeviceGetAttribute(
        &alignment, hipDeviceAttribute_t::hipDeviceAttributeTextureAlignment,
        hipDevice));

    HIPDeviceScope withDev(hipDevice, driver);
    Device::MemInfo memInfo;
    HIP_CHECK_ERR(driver.hipMemGetInfo(&memInfo.free, &memInfo.total));

#if MODULAR_LINUX
    int integrated = 0;
    HIP_CHECK_ERR(driver.hipDeviceGetAttribute(
        &integrated, hipDeviceAttribute_t::hipDeviceAttributeIntegrated,
        hipDevice));
    if (integrated) {
      DEFINE_OR_RETURN_ERROR(Device::MemInfo, hostInfo,
                             getHostUnifiedMemoryInfo(memInfo));
      memInfo = hostInfo;
    }
#endif

    ErrorOr<std::string> errArch = getArchName(hipDevice);
    if (errArch)
      RETURN_ERROR(llvm::Twine("HIP architecture query failed: ") +
                   errArch.getError());

    // HIP often returns an empty string for the device name, in these cases
    // AMD uses the architecture as the device name.
    std::string name(hipName);
    if (name.empty())
      name = "AMD: " + *errArch;

    device = new HIPDevice(driver, id, hipDevice, name, *errArch, memInfo,
                           alignment);

    // Once the Device has been created remember it in the devices list.
    (*devices)[id] = device;

    // Run healthcheck after device init — returns error on unhealthy GPU.
    RETURN_IF_ERROR(device->runHealthcheck());
  }
  return *device;
}

/// Run a shell command and capture its first line of output.
/// Returns an empty string if the command fails or produces no output.
/// The command string must be a fixed-format string with only %d
/// substitutions (safe from injection since device IDs are always small
/// non-negative integers).
static std::string runShellCommand(const char *cmd) {
  FILE *pipe = popen(cmd, "r");
  if (!pipe)
    return {};
  char buf[512] = {};
  std::string result;
  if (fgets(buf, sizeof(buf), pipe) != nullptr)
    result = buf;
  pclose(pipe);
  return result;
}

ErrorOrSuccess HIPDevice::runHealthcheck() const {
  if (DeviceContext::configAsBool("disable-healthcheck", false))
    return success();

  GPUHealthData data;
  data.deviceId = id();
  data.deviceName = name();

  // 1. Gather hardware throttle violations via amd-smi (per-GPU).
  // Query the THROTTLE_STATUS field which outputs a comma-separated list of
  // active throttle reasons, e.g. "THROTTLE_STATUS: PROCHOT_THERMAL, PPT_PWR"
  // or "THROTTLE_STATUS: UNTHROTTLED" when clean.
  // PPT_PWR is excluded — hitting the power budget is normal GPU operation,
  // analogous to NVIDIA's SW_POWER_CAP which the CUDA healthcheck ignores.
  // WARNING: This output parsing was tested against amd-smi from ROCm 6.3.
  // The CLI output format may change across ROCm versions — if throttle
  // detection breaks after a ROCm upgrade, check `amd-smi metric` output
  // and update the parsing accordingly.
  // Safe: id() is always a small non-negative integer (%d format).
  char throttleCmd[256];
  snprintf(throttleCmd, sizeof(throttleCmd),
           "amd-smi metric --gpu %d 2>/dev/null | grep 'THROTTLE_STATUS'",
           id());
  std::string throttleLine = runShellCommand(throttleCmd);
  if (!throttleLine.empty()) {
    data.throttleQueried = true;
    // Check for each alarming throttle reason in the status line.
    // The field format is "THROTTLE_STATUS: REASON1, REASON2, ..."
    if (throttleLine.find("PROCHOT") != std::string::npos)
      data.throttleReasons |= GPUHealthData::THROTTLE_PROCHOT;
    if (throttleLine.find("SOCKET_THERMAL") != std::string::npos)
      data.throttleReasons |= GPUHealthData::THROTTLE_SOCKET_THERMAL;
    if (throttleLine.find("VR_THERMAL") != std::string::npos)
      data.throttleReasons |= GPUHealthData::THROTTLE_VR_THERMAL;
    if (throttleLine.find("HBM_THERMAL") != std::string::npos)
      data.throttleReasons |= GPUHealthData::THROTTLE_HBM_THERMAL;
  }
  // Empty output means amd-smi not found or no THROTTLE_STATUS field;
  // leave unqueried.

  return evaluateGPUHealth(data);
}

// ===== HIPDeviceEvent ===== //
HIPDeviceEvent::HIPDeviceEvent(HIPDevice &device, hipEvent_t event,
                               bool recordedDuringCapture)
    : DeviceEvent(device), event_(event),
      recordedDuringCapture_(recordedDuringCapture) {}

HIPDeviceEvent::~HIPDeviceEvent() {
  const auto &hipDevice = HIPDevice::cast(device());

  HIP_WITH_DEV(hipDevice);
  HIP_CHECK(hipDevice.hipDriver().hipEventDestroy(event_));
}

ErrorOrSuccess HIPDeviceEvent::synchronize() {
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().hipEventSynchronize(event_));
  return success();
}

ErrorOr<bool> HIPDeviceEvent::isReady() const {
  if (recordedDuringCapture_) {
    // Events recorded during stream capture become graph nodes; polling them
    // directly is not allowed, so treat them as ready.
    return true;
  }
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIPDriver::Result res = hipDevice.hipDriver().hipEventQuery(event_);
  if (res == HIPDriver::SUCCESS)
    return true;
  else if (res == HIPDriver::NOT_READY) {
    return false;
  }
  HIP_CHECK_ERR(res); // All other errors get raised here.
  // Not reachable because there is an error status in res, but compiler
  // complains if not returning a value.
  llvm_unreachable("Not Reachable");
  return Error("Not Reachable.");
}

ErrorOr<float> HIPDeviceEvent::elapsedTime(const DeviceEvent &end) const {
  const auto &hipEnd = HIPDeviceEvent::cast(end);
  const auto &hipDevice = HIPDevice::cast(device());
  HIP_WITH_DEV_ERR(hipDevice);
  float millis;
  HIP_CHECK_ERR(hipDevice.hipDriver().hipEventElapsedTime(&millis, event_,
                                                          hipEnd.hipEvent()));
  return millis;
}

hipEvent_t HIPDeviceEvent::hipEvent() const { return event_; }

const HIPDeviceEvent &HIPDeviceEvent::cast(const DeviceEvent &event) {
  return static_cast<const HIPDeviceEvent &>(event);
}

// ===== HIPDeviceGraph ===== //
HIPDeviceGraph::~HIPDeviceGraph() {
  const auto &hipDevice = HIPDevice::cast(this->device());
  HIP_WITH_DEV(hipDevice);
  if (rawGraph_)
    HIP_CHECK(hipDevice.hipDriver().hipGraphDestroy(rawGraph_));
  if (exec_)
    HIP_CHECK(hipDevice.hipDriver().hipGraphExecDestroy(exec_));
}

ErrorOrSuccess HIPDeviceGraph::instantiate() {
  const auto &hipDevice = HIPDevice::cast(this->device());
  HIP_WITH_DEV_ERR(hipDevice);

  hipGraphExec_t exec = nullptr;
  HIPDriver::Result res = hipDevice.hipDriver().hipGraphInstantiateWithFlags(
      &exec, rawGraph_,
      M::to_underlying(
          hipGraphInstantiateFlags::hipGraphInstantiateFlagAutoFreeOnLaunch));

  if (res != HIPDriver::SUCCESS)
    return formatHipError(__FILE__, __LINE__, __FUNCTION__, res,
                          "failed to instantiate HIP graph");

  // Destroy the raw graph now that instantiation succeeded; exec_ takes over.
  HIP_CHECK_ERR(hipDevice.hipDriver().hipGraphDestroy(rawGraph_));
  rawGraph_ = nullptr;
  exec_ = exec;
  return success();
}

ErrorOrSuccess HIPDeviceGraph::replay() const {
  if (!hasStream())
    RETURN_ERROR("cannot replay HIP graph: no stream stored during capture");
  const auto *hipStream = HIPDeviceStream::verified_cast(captureStream());
  if (!hipStream)
    RETURN_ERROR("cannot replay HIP graph: stored stream is not a HIP stream");
  const auto &hipDevice = HIPDevice::cast(this->device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(
      hipDevice.hipDriver().hipGraphLaunch(exec_, hipStream->hipStream()));
  return success();
}

// ===== HIPDeviceStream ===== //
HIPDeviceStream::HIPDeviceStream(HIPDeviceContext &ctx, hipStream_t stream)
    : DeviceStream(ctx), stream_(stream) {
  logGraphCapture_ = DeviceContext::configAsString("log_graph_capture", "");
}

ErrorOrSuccess HIPDeviceStream::synchronize() {
  const auto &hipDevice = HIPDevice::cast(this->device());

  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().hipStreamSynchronize(stream_));
  POLL_PENDING_EVENTS(this->context_());
  return success();
}

ErrorOr<DeviceEventRef> HIPDeviceStream::enqueueEvent() const {
  HIPDevice &hipDevice = HIPDevice::cast(this->device());
  const bool recordedDuringCapture =
      isStreamCapturing(hipDevice.hipDriver(), stream_);

  HIP_WITH_DEV_ERR(hipDevice);
  hipEvent_t event;
  HIP_CHECK_ERR(hipDevice.hipDriver().hipEventCreateWithFlags(&event, 0));
  HIP_CHECK_ERR(hipDevice.hipDriver().hipEventRecord(event, stream_));
  return HIPDeviceEventRef::create(hipDevice, event, recordedDuringCapture);
}

ErrorOrSuccess
HIPDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  const auto &hipDevice = HIPDevice::cast(this->device());

  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().hipStreamWaitEvent(
      stream_, hipEvent.hipEvent(), 0));
  return success();
}

ErrorOrSuccess HIPDeviceStream::enqueueHostFunc(void (*fn)(void *),
                                                void *userData) const {
  const auto &hipDevice = HIPDevice::cast(this->device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().hipLaunchHostFunc(stream_, fn, userData));
  return success();
}

/// Storage for the batch-mem-op parameters of a recorded value-wait node.
///
/// The HIP runtime does not deep-copy `hipBatchMemOpNodeParams`: it keeps the
/// `paramArray` pointer and dereferences it at graph launch. Freeing the
/// parameters after the node is added does not fail loudly -- the node simply
/// stops gating -- so the storage is interned per (context, address, value) and
/// kept for the life of the process. The keys are process-lifetime
/// `CompletionFlag`s, so the set stays tiny.
static const hipBatchMemOpNodeParams *
internWaitValueNodeParams(hipCtx_t ctx, hipDeviceptr_t address,
                          uint64_t value) {
  struct Params {
    hipStreamBatchMemOpParams op;
    hipBatchMemOpNodeParams node;
  };
  static std::mutex mutex;
  static std::map<std::tuple<hipCtx_t, hipDeviceptr_t, uint64_t>,
                  std::unique_ptr<Params>>
      interned;

  const std::lock_guard<std::mutex> lock(mutex);
  std::unique_ptr<Params> &params = interned[{ctx, address, value}];
  if (!params) {
    params = std::make_unique<Params>();
    params->op.waitValue.operation =
        hipStreamBatchMemOpType::hipStreamMemOpWaitValue64;
    params->op.waitValue.address = address;
    params->op.waitValue.value64 = value;
    params->op.waitValue.flags = static_cast<unsigned int>(
        hipStreamWaitValueFlags::hipStreamWaitValueEq);
    params->node.ctx = ctx;
    params->node.count = 1;
    params->node.paramArray = &params->op;
    params->node.flags = 0;
  }
  return &params->node;
}

ErrorOrSuccess
HIPDeviceStream::enqueueWaitOnHostValue(const CompletionFlag &flag,
                                        uint64_t value) const {
  const auto &hipDevice = HIPDevice::cast(this->device());

  HIP_WITH_DEV_ERR(hipDevice);
  const HIPDriver &driver = hipDevice.hipDriver();
  auto address = reinterpret_cast<hipDeviceptr_t>(flag.devicePtr());

  // Stream capture does not record hipStreamWaitValue64. Measured on ROCm
  // 7.2.4 / gfx950, under all three capture modes: the call returns success,
  // leaves the capture active, contributes no node to the graph, and is
  // executed eagerly against the capturing stream's real queue instead. A
  // graph captured that way replays with no gate at all, so every consumer of
  // the awaited host data reads whatever was already in the buffer. Splice the
  // wait in as an explicit batch-mem-op node instead.
  //
  // Fixed in ROCm 10:
  // https://github.com/ROCm/rocm-systems/commit/cc777ef7fdc72152bda568bb3b586011b9465b21
  // This workaround can be removed once that is our minimum supported ROCm
  // version.
  if (getCaptureStatus() !=
      hipStreamCaptureStatus::hipStreamCaptureStatusActive) {
    HIP_CHECK_ERR(driver.hipStreamWaitValue64(
        stream_, address, value, hipStreamWaitValueFlags::hipStreamWaitValueEq,
        /*mask*/ static_cast<uint64_t>(-1)));
    return success();
  }

  // Never silently fall back to the eager wait here: it would be dropped from
  // the graph and the caller would get a replay that does not gate.
  if (!driver.hasBatchMemOpGraphNode())
    RETURN_ERROR("cannot record a host-value wait into a graph being "
                 "captured: this HIP runtime does not provide "
                 "hipGraphAddBatchMemOpNode (needs ROCm 6.4 or newer)");

  hipCtx_t ctx = nullptr;
  HIP_CHECK_ERR(driver.hipCtxGetCurrent(&ctx));
  if (!ctx)
    RETURN_ERROR("no current HIP context to record a host-value wait node in");

  hipStreamCaptureStatus status =
      hipStreamCaptureStatus::hipStreamCaptureStatusNone;
  unsigned long long captureId = 0;
  hipGraph_t graph = nullptr;
  const hipGraphNode_t *dependencies = nullptr;
  size_t numDependencies = 0;
  HIP_CHECK_ERR(driver.hipStreamGetCaptureInfo_v2(
      stream_, &status, &captureId, &graph, &dependencies, &numDependencies));
  if (status != hipStreamCaptureStatus::hipStreamCaptureStatusActive || !graph)
    RETURN_ERROR("stream capture is no longer active while recording a "
                 "host-value wait");

  hipGraphNode_t node = nullptr;
  HIP_CHECK_ERR(driver.hipGraphAddBatchMemOpNode(
      &node, graph, dependencies, numDependencies,
      internWaitValueNodeParams(ctx, address, value)));

  // Make the new node the stream's capture frontier so subsequently captured
  // work depends on it, exactly as an eagerly enqueued wait would order it.
  HIP_CHECK_ERR(driver.hipStreamUpdateCaptureDependencies(
      stream_, &node, 1,
      hipStreamUpdateCaptureDependenciesFlags::
          hipStreamSetCaptureDependencies));
  return success();
}

ErrorOrSuccess HIPDeviceStream::eventRecord(const DeviceEvent &event) const {
  const auto &hipEvent = HIPDeviceEvent::cast(event);
  const auto &hipDevice = HIPDevice::cast(this->device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(
      hipDevice.hipDriver().hipEventRecord(hipEvent.hipEvent(), stream_));
  return success();
}

ErrorOrSuccess HIPDeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t /*argCount*/,
    const size_t * /*argSizes*/) const {
  const auto &hipFunc = HIPDeviceFunction::cast(func);
  const auto &hipDevice = HIPDevice::cast(this->device());
  HIP_WITH_DEV_ERR(hipDevice);
  if (numAttrs > 0)
    RETURN_ERROR("Function launch attributes not supported;");
  return hipFunc.hipCallKernelDirect(gridX, gridY, gridZ, blockX, blockY,
                                     blockZ, sharedMemBytes, args, stream_);
}

void *HIPDeviceStream::ptr() const { return hipStream(); }

hipStream_t HIPDeviceStream::hipStream() const { return stream_; }

const HIPDeviceStream *
HIPDeviceStream::verified_cast(const DeviceStream *stream) {
  if (stream->device().api() != HIPDevice::api) {
    return nullptr;
  }
  return static_cast<const HIPDeviceStream *>(stream);
}

hipStreamCaptureStatus HIPDeviceStream::getCaptureStatus() const {
  const auto &hipDevice = HIPDevice::cast(this->device());
  hipStreamCaptureStatus status =
      hipStreamCaptureStatus::hipStreamCaptureStatusNone;
  hipDevice.hipDriver().hipStreamIsCapturing(stream_, &status);
  return status;
}

ErrorOrSuccess HIPDeviceStream::beginCapture() const {
  hipStreamCaptureStatus status = getCaptureStatus();
  if (status == hipStreamCaptureStatus::hipStreamCaptureStatusActive)
    RETURN_ERROR("stream is already capturing");
  if (status == hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
    RETURN_ERROR("stream capture was invalidated");

  const auto &hipDevice = HIPDevice::cast(this->device());
  HIP_WITH_DEV_ERR(hipDevice);
  HIP_CHECK_ERR(hipDevice.hipDriver().hipStreamBeginCapture(
      stream_, hipStreamCaptureMode::hipStreamCaptureModeRelaxed));
  return success();
}

ErrorOr<DeviceGraphRef> HIPDeviceStream::endCapture() {
  hipStreamCaptureStatus status = getCaptureStatus();
  if (status == hipStreamCaptureStatus::hipStreamCaptureStatusNone)
    RETURN_ERROR("stream is not capturing");
  if (status == hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
    RETURN_ERROR("stream capture was invalidated");

  HIPDevice &hipDevice = HIPDevice::cast(this->device());
  HIP_WITH_DEV_ERR(hipDevice);

  hipGraph_t graph = nullptr;
  HIP_CHECK_ERR(hipDevice.hipDriver().hipStreamEndCapture(stream_, &graph));

  if (!graph)
    RETURN_ERROR("hipStreamEndCapture returned null graph");

  if (logGraphCapture_ != "") {
    static std::atomic<int32_t> graphCaptureCount = 0;

    int32_t currentId = graphCaptureCount.fetch_add(1);
    // Debug: print graph to DOT file for visualization.
    // This is only enabled at compile time for debugging purposes.
    // The DOT file can be visualized with Graphviz tools (e.g., dot -Tpng).
    // Set MODULAR_CUDA_GRAPH_DOT_PATH to customize the output path.
    std::string outputPath =
        llvm::formatv("{}/hip_graph_{}.dot", logGraphCapture_, currentId).str();
    hipDevice.hipDriver().hipGraphDebugDotPrint(
        graph, outputPath.c_str(),
        static_cast<unsigned>(
            hipGraphDebugDotFlags::hipGraphDebugDotFlagsVerbose));
  }

  // Return an uninstantiated graph. The caller must call instantiate() before
  // replay(). Keeping endCapture() cheap ensures the stream is released for
  // the next capture session before the expensive instantiation begins.
  return HIPDeviceGraphRef::create(
      DeviceContextRef::copy(&context_()), graph,
      DeviceStreamRef::copy(static_cast<DeviceStream *>(this)));
}

void HIPDeviceStream::beginLaunchTrace() const {
  auto &registry = launchTraceRegistry();
  registry.beginTraceForStream(static_cast<void *>(stream_));
}

std::vector<KernelLaunchMetadata> HIPDeviceStream::takeLaunchTrace() const {
  auto &registry = launchTraceRegistry();
  return registry.takeTraceForStream(static_cast<void *>(stream_));
}

// ===== HIPModuleFunction ===== //
HIPModuleFunction::HIPModuleFunction(const std::string name,
                                     const HIPDriver &driver,
                                     hipModule_t module, hipFunction_t function)
    : DeviceFunctionCacheEntry(std::move(name)), driver_(driver),
      module_(module), function_(function) {}

HIPModuleFunction::~HIPModuleFunction() {
  HIP_CHECK(hipDriver().hipModuleUnload(module_));
}

const HIPDriver &HIPModuleFunction::hipDriver() const { return driver_; }
hipModule_t HIPModuleFunction::hipModule() const { return module_; }
hipFunction_t HIPModuleFunction::hipFunction() const { return function_; }

const HIPModuleFunction *
HIPModuleFunction::cast(const DeviceFunctionCacheEntry *entry) {
  return static_cast<const HIPModuleFunction *>(entry);
}

// ===== HIPDeviceFunction ===== //
HIPDeviceFunction::HIPDeviceFunction(HIPDeviceContext &context,
                                     HIPModuleFunctionRef modFunc)
    : DeviceFunction(context), moduleFunction_(std::move(modFunc)) {}

HIPDeviceFunction::~HIPDeviceFunction() {}

ErrorOrSuccess HIPDeviceFunction::hipCallKernelDirect(
    uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t blockX,
    uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes, void **args,
    hipStream_t stream) const {
  // Record launch trace if enabled
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream))) {
    // HIP doesn't support parameter introspection like CUDA's
    // cuFuncGetParamInfo, so we pass empty arrays for paramSizes and args
    registry.recordLaunchTrace(
        name(), reinterpret_cast<uintptr_t>(moduleFunction_->hipFunction()),
        /*paramSizes=*/llvm::ArrayRef<size_t>(), gridX, gridY, gridZ, blockX,
        blockY, blockZ, sharedMemBytes,
        /*attrs=*/llvm::ArrayRef<uint8_t>(), /*numAttrs=*/0,
        /*args=*/llvm::ArrayRef<void *>(), static_cast<void *>(stream));
  }

  HIP_CHECK_ERR(moduleFunction_->hipDriver().hipModuleLaunchKernel(
      moduleFunction_->hipFunction(), gridX, gridY, gridZ, blockX, blockY,
      blockZ, sharedMemBytes, stream, args, nullptr));
  MAYBE_SYNC(HIPDeviceContext::cast(context()));
  return success();
}

const std::string &HIPDeviceFunction::name() const {
  return moduleFunction_->name();
}

ErrorOrSuccess HIPDeviceFunction::copyToNamedConstant(const char *name,
                                                      const void *data,
                                                      size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceFunction::copyToNamedConstant",
                       TRACY_COLOR_PURPLE);
  hipDeviceptr_t devicePtr;
  size_t deviceSize;
  auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(hipContext.hipDriver().hipModuleGetGlobal(
      &devicePtr, &deviceSize, moduleFunction_->hipModule(), name));

  if (deviceSize < byteSize) {
    RETURN_ERROR("byteSize too big");
  }

  HIP_CHECK_ERR(hipContext.hipDriver().hipMemcpyHtoDAsync(
      devicePtr, data, byteSize, hipContext.hipStream()));
  MAYBE_SYNC(HIPDeviceContext::cast(context()));
  return success();
}

ErrorOr<int32_t> HIPDeviceFunction::getAttribute(int32_t code) {
  int32_t result;
  auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(hipContext.hipDriver().hipFuncGetAttribute(
      &result, static_cast<hipFunction_attribute>(code),
      moduleFunction_->hipFunction()));
  return result;
}

ErrorOrSuccess HIPDeviceFunction::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, int blockSize, size_t dynamicSharedMemSize) {
  auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX_ERR(hipContext);
  HIP_CHECK_ERR(
      hipContext.hipDriver().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
          numBlocks, moduleFunction_->hipFunction(), blockSize,
          dynamicSharedMemSize));

  return success();
}

HIPDeviceFunction &HIPDeviceFunction::cast(DeviceFunction &func) {
  return static_cast<HIPDeviceFunction &>(func);
}

const HIPDeviceFunction &HIPDeviceFunction::cast(const DeviceFunction &func) {
  return static_cast<const HIPDeviceFunction &>(func);
}

hipFunction_t HIPDeviceFunction::hipFunction() const {
  return moduleFunction_->hipFunction();
}

hipModule_t HIPDeviceFunction::hipModule() const {
  return moduleFunction_->hipModule();
}

// ===== HIPDeviceTimer ===== //
HIPDeviceTimer::HIPDeviceTimer(HIPDeviceContext &context, hipEvent_t start,
                               hipEvent_t stop)
    : DeviceTimer(context), start_(start), stop_(stop) {}

HIPDeviceTimer::~HIPDeviceTimer() {
  auto &hipContext = HIPDeviceContext::cast(context());

  HIP_WITH_CTX(hipContext);
  HIP_CHECK(hipContext.hipDriver().hipEventDestroy(start_));
  HIP_CHECK(hipContext.hipDriver().hipEventDestroy(stop_));
}

hipEvent_t HIPDeviceTimer::hipStartEvent() const { return start_; }
hipEvent_t HIPDeviceTimer::hipStopEvent() const { return stop_; }

const HIPDeviceTimer &HIPDeviceTimer::cast(const DeviceTimer &timer) {
  return static_cast<const HIPDeviceTimer &>(timer);
}

// ===== HIPStreamPool ===== //
HIPStreamPool::HIPStreamPool(const HIPDriver &driver, hipDevice_t device,
                             hipStream_t base)
    : driver_(driver), device_(device) {
  streams_.push_back(base);
}

HIPStreamPool::~HIPStreamPool() {
  // Bind the owning device while we synchronize and destroy the streams.
  HIPDeviceScope withDev(device_, driver_);
  for (hipStream_t stream : streams_) {
    // Synchronize to ensure all enqueued work (and chunk frees) complete.
    HIP_CHECK(driver_.hipStreamSynchronize(stream));
    HIP_CHECK(driver_.hipStreamDestroy(stream));
  }
}

hipStream_t HIPStreamPool::get(unsigned i) const {
  std::lock_guard<std::mutex> lock(mu_);
  assert(i < streams_.size() && "stream index out of range");
  return streams_[i];
}

unsigned HIPStreamPool::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<unsigned>(streams_.size());
}

unsigned HIPStreamPool::emplace_back(hipStream_t stream) {
  std::lock_guard<std::mutex> lock(mu_);
  streams_.emplace_back(stream);
  return static_cast<unsigned>(streams_.size()) - 1;
}

// ===== HIPDeviceContext ===== //
HIPDeviceContext::HIPDeviceContext(HIPDevice &device, hipStream_t stream)
    : DeviceContext(device, TypeID::get<HIPDeviceContext>()),
      driver_(device.hipDriver()), device_(device.hipDevice()),
      streamPool_(HIPStreamPoolRef::create(device.hipDriver(),
                                           device.hipDevice(), stream)),
      stream_(stream) {
  // Initialize Tracy ROCm profiling context if not already initialized.
  M::initTracyRocm();
}

// Stream cleanup (synchronize + destroy) is owned by `streamPool_`, whose
// destructor runs when the last co-owner is released.
HIPDeviceContext::~HIPDeviceContext() {}

// Constructor for `selectStream` views: shares an existing stream pool (and
// therefore the base context's streams + device) and only differs in which
// pooled stream is active. `stream_` caches `streamPool_->get(activeStreamId)`.
HIPDeviceContext::HIPDeviceContext(HIPDevice &device,
                                   HIPStreamPoolRef streamPool,
                                   unsigned activeStreamId)
    : DeviceContext(device, TypeID::get<HIPDeviceContext>()),
      driver_(device.hipDriver()), device_(device.hipDevice()),
      streamPool_(std::move(streamPool)),
      stream_(streamPool_->get(activeStreamId)) {
  M::initTracyRocm();
}

ErrorOrSuccess HIPDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                          Device::AllocMode mode) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::memAlloc", TRACY_COLOR_PURPLE);

  assert(byteSize != 0);
  HIP_WITH_CTX_ERR(*this);
  switch (mode) {
  case Device::AllocMode::ASYNC: {
    // We used to support async allocations but prefer only sync ones now.
    return Error("Async allocation is not supported for HIP devices");

    // Deprecated logic. Remove later.
    // HIP_CHECK_ERR(driver_.hipMallocAsync(ptr, byteSize, stream_));
    // MAYBE_SYNC(*this);
    break;
  }
  case Device::AllocMode::HOST: {
    // We must poll here since hipHostMalloc is synchronous.
    POLL_PENDING_EVENTS(*this);
    HIP_CHECK_ERR(driver_.hipHostMalloc(
        ptr, byteSize, hipHostMallocFlags::hipHostMallocPortable));
    break;
  }
  case Device::AllocMode::SYNC: {
    POLL_PENDING_EVENTS(*this);
    HIP_CHECK_ERR(driver_.hipMalloc(ptr, byteSize));
    break;
  }
  }
  return success();
}

ErrorOrSuccess HIPDeviceContext::memWrap(void **devPtr, void *hostPtr,
                                         size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::memWrap", TRACY_COLOR_PURPLE);
  HIP_WITH_CTX_ERR(*this);
  // hipHostRegisterPortable (0x1): the registration is visible from every
  // device in the process, matching the hipHostMalloc host-alloc path.
  constexpr unsigned int kFlags = 0x1;
  HIP_CHECK_ERR(driver_.hipHostRegister(hostPtr, byteSize, kFlags));

  // A registered range is mapped at a driver-chosen address, not at `hostPtr`.
  hipDeviceptr_t mapped = nullptr;
  HIPDriver::Result result =
      driver_.hipHostGetDevicePointer(&mapped, hostPtr, 0);
  if (result != HIPDriver::SUCCESS) {
    (void)driver_.hipHostUnregister(hostPtr);
    const char *errName = nullptr;
    (void)driver_.hipDrvGetErrorName(result, &errName);
    return Error(std::string("memWrap: hipHostGetDevicePointer failed: ")
                     .append(errName ? errName : "(unknown)"));
  }
  *devPtr = reinterpret_cast<void *>(mapped);
  return success();
}

ErrorOrSuccess HIPDeviceContext::memFree(void *ptr, Device::FreeKind kind) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::memFree", TRACY_COLOR_PURPLE);

  HIP_WITH_CTX_ERR(*this);
  if (kind == Device::FreeKind::WRAPPED) {
    // `hipHostUnregister` takes the host base; `ptr` is the mapped address.
    hipPointerAttribute_t attrs{};
    HIP_CHECK_ERR(driver_.hipPointerGetAttributes(&attrs, ptr));
    if (attrs.hostPointer == nullptr)
      RETURN_ERROR("memFree: wrapped range has no host pointer to unregister");
    HIP_CHECK_ERR(driver_.hipHostUnregister(attrs.hostPointer));
    return success();
  }

  if (kind == Device::FreeKind::HOST) {
    POLL_PENDING_EVENTS(*this);
    HIP_CHECK_ERR(driver_.hipHostFree(ptr));
  } else {
    HIP_CHECK_ERR(driver_.hipFree(ptr)); // We only support sync allocations.
    MAYBE_SYNC(*this);
  }
  return success();
}

ErrorOr<CompletionFlagRef> HIPDeviceContext::allocateCompletionFlag() {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::allocateCompletionFlag",
                       TRACY_COLOR_PURPLE);
  HIP_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  void *hostPtr = nullptr;
  HIP_CHECK_ERR(
      driver_.hipHostMalloc(&hostPtr, sizeof(uint64_t),
                            hipHostMallocFlags::hipHostMallocPortable |
                                hipHostMallocFlags::hipHostMallocMapped));
  hipDeviceptr_t devPtr = nullptr;
  HIPDriver::Result r = driver_.hipHostGetDevicePointer(&devPtr, hostPtr, 0);
  if (r != HIPDriver::SUCCESS) {
    (void)driver_.hipHostFree(hostPtr);
    const char *errName = nullptr;
    (void)driver_.hipDrvGetErrorName(r, &errName);
    return Error(
        std::string("hipHostGetDevicePointer failed for CompletionFlag: ")
            .append(errName ? errName : "(unknown)"));
  }
  // Initialize the flag to zero so the first wait observes a clean state.
  *static_cast<uint64_t *>(hostPtr) = 0;
  return CompletionFlagRef::take(
      new CompletionFlag(*this, hostPtr, reinterpret_cast<uint64_t>(devPtr)));
}

ErrorOrSuccess HIPDeviceContext::freeCompletionFlag(void *hostPtr) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::freeCompletionFlag",
                       TRACY_COLOR_PURPLE);
  HIP_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  HIP_CHECK_ERR(driver_.hipHostFree(hostPtr));
  return success();
}

void *HIPDeviceContext::driverContext() const {
  return hipDeviceToPtr(hipDevice());
}

ErrorOrSuccess HIPDeviceContext::enqueueCopyToDevice(void *dst, const void *src,
                                                     size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);
  HIP_WITH_CTX_ERR(*this);

  HIP_CHECK_ERR(driver_.hipMemcpyHtoDAsync(dst, src, byteSize, stream_));
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata(
        "hipMemcpyHtoD",
        hashMemcpyTrace(LaunchTraceKind::kMemcpyHtoD, byteSize),
        KernelLaunchMetadata::MemcpyKind::HtoD, byteSize);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
  }
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess HIPDeviceContext::enqueueCopyFromDevice(void *dst,
                                                       const void *src,
                                                       size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);
  HIP_WITH_CTX_ERR(*this);

  HIP_CHECK_ERR(driver_.hipMemcpyDtoHAsync(dst, const_cast<void *>(src),
                                           byteSize, stream_));
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata(
        "hipMemcpyDtoH",
        hashMemcpyTrace(LaunchTraceKind::kMemcpyDtoH, byteSize),
        KernelLaunchMetadata::MemcpyKind::DtoH, byteSize);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
  }
  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess HIPDeviceContext::memCopyDtoD(DeviceBuffer &dst,
                                             DeviceBuffer &src,
                                             size_t srcByteOffset,
                                             size_t srcByteSize,
                                             bool crossStreamSync) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);

  CHECK_IS_HIP_ERR(src.context().device());
  CHECK_IS_HIP_ERR(dst.context().device());

  // Resolve through backendContext() so a buffer stamped with a decorating
  // context (e.g. a device-graph RecordingDeviceContext) casts to the
  // driver-backed context it decorates.
  auto &srcCtx = HIPDeviceContext::cast(src.context().backendContext());
  auto &dstCtx = HIPDeviceContext::cast(dst.context().backendContext());

  if (this->device_ != dstCtx.device_) {
    RETURN_ERROR("device context does not match context of dst");
  }

  // It is recommended to set the src device to be the current device according
  // to documentation:
  // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/
  // hip_runtime_api/modules/memory_management.html#
  // _CPPv49hipMemcpyPvPKv6size_t13hipMemcpyKind
  HIP_WITH_CTX_ERR(srcCtx);

  hipDeviceptr_t srcPtr = (uint8_t *)src.ptr() + srcByteOffset;
  hipDeviceptr_t dstPtr = dst.ptr();
  size_t dstByteSize = dst.byteSize();
  assert(srcByteOffset + srcByteSize <= src.byteSize());
  assert(srcByteSize == dstByteSize);
  // A copy whose source and destination are on different streams requires
  // cross-stream synchronization. Unless the caller opted out (because it
  // manages this synchronization itself, e.g. the graph compiler emits
  // explicit synchronization ops), bracket the copy with events so the
  // destination stream waits for the source before the copy and the source
  // stream waits for the copy to complete afterward.
  const bool needsSync =
      crossStreamSync && srcCtx.hipStream() != dstCtx.hipStream();

  if (needsSync) {
    // Ensure that we are waiting for the source data to be ready.
    DEFINE_OR_RETURN_ERROR(DeviceEventRef, srcEvent, srcCtx.enqueueEvent());
    RETURN_IF_ERROR(dstCtx.enqueueWaitForEvent(*srcEvent));
  }

  HIP_CHECK_ERR(
      driver_.hipMemcpyDtoDAsync(dstPtr, srcPtr, dstByteSize, stream_));

  if (needsSync) {
    // Wait for the copy to complete before proceeding with the source stream
    // (e.g. freeing or overwriting memory).
    DEFINE_OR_RETURN_ERROR(DeviceEventRef, dstEvent, dstCtx.enqueueEvent());
    RETURN_IF_ERROR(srcCtx.enqueueWaitForEvent(*dstEvent));
  }
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata(
        "hipMemcpyDtoD",
        hashMemcpyTrace(LaunchTraceKind::kMemcpyDtoD, dstByteSize),
        KernelLaunchMetadata::MemcpyKind::DtoD, dstByteSize);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
  }

  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess
HIPDeviceContext::memBatchCopy(llvm::ArrayRef<DeviceBuffer *> dsts,
                               llvm::ArrayRef<DeviceBuffer *> srcs) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::memBatchCopy", TRACY_COLOR_PURPLE);
  const size_t count = dsts.size();

  // No hipMemcpyBatchAsync: use the scalar loop.
  if (count == 0 || !driver_.hasMemcpyBatchAsync())
    return DeviceContext::memBatchCopy(dsts, srcs);

  HIP_WITH_CTX_ERR(*this);

  // A pointer can join the batch only if UVA resolves it from this context:
  // host and pinned memory always do, a peer's memory only under peer access.
  // Every source context running on another stream needs its own ordering
  // bracket around the submit.
  SmallVector<HIPDeviceContext *, 2> streamSrcCtxs;
  SmallVector<DeviceBuffer *, 8> hostSrcs;
  std::vector<void *> dstPtrs(count);
  std::vector<const void *> srcPtrs(count);
  std::vector<size_t> sizes(count);
  size_t totalBytes = 0;
  for (size_t i = 0; i < count; ++i) {
    DeviceBuffer &src = *srcs[i];
    if (src.isHost() || src.context().device().api() == Device::cpuAPI) {
      hostSrcs.push_back(&src);
    } else {
      auto &srcCtx = HIPDeviceContext::cast(src.context().backendContext());
      if (srcCtx.hipDevice() != this->hipDevice()) {
        DEFINE_OR_RETURN_ERROR(bool, peerEnabled,
                               Device::allPeerAccessEnabled());
        // Without a peer mapping the pointer is unreachable from here.
        if (!peerEnabled)
          return DeviceContext::memBatchCopy(dsts, srcs);
      }
      if (srcCtx.hipStream() != this->hipStream() &&
          !llvm::is_contained(streamSrcCtxs, &srcCtx))
        streamSrcCtxs.push_back(&srcCtx);
    }
    dstPtrs[i] = dsts[i]->ptr();
    srcPtrs[i] = src.ptr();
    sizes[i] = dsts[i]->byteSize();
    totalBytes += sizes[i];
  }

  // Mirror the scalar host path: a source owned by a CPU device context has to
  // be settled up front because events cannot be awaited across device types,
  // and any host source is DMA'd on our stream rather than its owning one, so
  // its deferred free must be ordered against us too.
  for (DeviceBuffer *src : hostSrcs) {
    if (src->context().device().api() == Device::cpuAPI)
      RETURN_IF_ERROR(src->context().synchronize());
    recordBufferUsedOnContext(*src, *this);
  }

  // Match scalar memCopyDtoD: our stream waits for each source stream before
  // the copy, and each source stream waits for the copy afterwards.
  for (HIPDeviceContext *srcCtx : streamSrcCtxs) {
    DEFINE_OR_RETURN_ERROR(DeviceEventRef, srcEvent, srcCtx->enqueueEvent());
    RETURN_IF_ERROR(this->enqueueWaitForEvent(*srcEvent));
  }

  hipMemcpyAttributes attrs{};
  attrs.srcAccessOrder = hipMemcpySrcAccessOrder::hipMemcpySrcAccessOrderStream;
  size_t attrsIdx = 0;
  size_t failIdx = 0;
  HIP_CHECK_ERR(driver_.hipMemcpyBatchAsync(
      dstPtrs.data(), srcPtrs.data(), sizes.data(), count, &attrs, &attrsIdx,
      /*numAttrs=*/1, &failIdx, stream_));

  if (!streamSrcCtxs.empty()) {
    DEFINE_OR_RETURN_ERROR(DeviceEventRef, dstEvent, this->enqueueEvent());
    for (HIPDeviceContext *srcCtx : streamSrcCtxs)
      RETURN_IF_ERROR(srcCtx->enqueueWaitForEvent(*dstEvent));
  }

  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata(
        "hipMemcpyBatch",
        hashMemcpyTrace(LaunchTraceKind::kMemcpyDtoD, totalBytes),
        KernelLaunchMetadata::MemcpyKind::DtoD, totalBytes);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
  }

  MAYBE_SYNC(*this);
  return success();
}

ErrorOrSuccess HIPDeviceContext::enqueueSetMemory(void *dst, size_t byteSize,
                                                  uint64_t val,
                                                  size_t valSize) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);

  if (valSize == 1) {
    HIP_CHECK_ERR(driver_.hipMemsetD8Async(dst, static_cast<uint8_t>(val),
                                           byteSize, stream_));
  } else if (valSize == 2) {
    if ((byteSize & 1) != 0) {
      return Error("byteSize of buffer is misaligned");
    }
    size_t len = byteSize >> 1;
    HIP_CHECK_ERR(driver_.hipMemsetD16Async(dst, static_cast<uint16_t>(val),
                                            len, stream_));
  } else if (valSize == 4) {
    if ((byteSize & 3) != 0) {
      return Error("byteSize of buffer is misaligned");
    }
    size_t len = byteSize >> 2;
    HIP_CHECK_ERR(driver_.hipMemsetD32Async(dst, static_cast<uint32_t>(val),
                                            len, stream_));
  } else {
    if (valSize != 8) {
      return Error("unsupported valSize");
    }
    if ((byteSize & 7) != 0) {
      return Error("byteSize of buffer is misaligned");
    }
    uint32_t hiBits = static_cast<uint32_t>(val >> 32);
    uint32_t loBits = static_cast<uint32_t>(val & 0xfffffffful);
    if (hiBits == loBits) {
      size_t len = byteSize >> 2; // Divide by 4 as we are using 32-bit memset.
      HIP_CHECK_ERR(driver_.hipMemsetD32Async(dst, loBits, len, stream_));
    } else {
      size_t len = byteSize >> 3;
      if (len > 0) {
        // Seed the first 8 bytes with the uint64 pattern by writing its
        // low and high 32-bit halves, then fill the rest via doubling
        // device-to-device copies (log2(len) calls instead of O(len)).
        //
        // HIP has no single-call strided 32-bit memset that works here.
        // `hipMemsetD2D32Async` looks like it should do the job in 2
        // calls, but ROCm forwards `width` unchanged into byte-valued
        // internal fields (sizeBytes = width * height * depth in
        // `ihipMemset3DCommand`), making the API's contract "width in
        // 32-bit elements" unusable: width=1 aborts with
        // "Pattern Size: 4 cannot be greater than fill size: 1" at
        // rocclr/device/blit.cpp:735, and width=4 (bytes) silently fails
        // to fill rows 1..N-1 on pinned host memory. This bug is
        // unchanged in ROCm 7.1.1, 7.2.0, 7.2.1, and 7.2.2.
        void *hiPtr = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(dst) + sizeof(uint32_t));
        HIP_CHECK_ERR(driver_.hipMemsetD32Async(dst, loBits, 1, stream_));
        HIP_CHECK_ERR(driver_.hipMemsetD32Async(hiPtr, hiBits, 1, stream_));
        for (size_t filled = 1; filled < len;) {
          size_t copyCount = std::min(filled, len - filled);
          void *dstCopy = reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(dst) + filled * sizeof(uint64_t));
          HIP_CHECK_ERR(driver_.hipMemcpyDtoDAsync(
              dstCopy, dst, copyCount * sizeof(uint64_t), stream_));
          filled += copyCount;
        }
      }
    }
  }
  auto &registry = launchTraceRegistry();
  if (registry.isLaunchTraceEnabled(static_cast<void *>(stream_))) {
    KernelLaunchMetadata metadata("hipMemset",
                                  hashMemsetTrace(byteSize, val, valSize),
                                  byteSize, val, valSize);
    registry.appendLaunchTraceEntry(static_cast<void *>(stream_), metadata);
  }

  MAYBE_SYNC(*this);
  return success();
}

DeviceContext::MMapFileImplResult
HIPDeviceContext::memmapFileImpl(const std::filesystem::path &filename) const {
  return Error("Unimplemented for HIP devices");
}

ErrorOr<DeviceFunctionRef>
HIPDeviceContext::loadFunction(StringRef moduleName, StringRef functionName,
                               StringRef code, int32_t maxDynamicSharedBytes,
                               StringRef debugLevel,
                               int32_t optimizationLevel) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::loadFunction", TRACY_COLOR_PURPLE);
  Device::FunctionCacheKey key{
      /*name=*/Device::FunctionCacheKey::moduleCacheName(moduleName, code),
      /*functionName=*/functionName.str(),
      /*debugLevel=*/debugLevel.str(),
      /*maxDynamicSharedBytes=*/maxDynamicSharedBytes,
      /*optimizationLevel=*/optimizationLevel};
  DeviceFunctionCacheEntryRef entry = device().findInFunctionCache(key);
  if (entry.getPointer() != nullptr) {
    HIPModuleFunctionRef moduleFunc =
        HIPModuleFunctionRef::take(HIPModuleFunction::cast(entry.release()));
    return HIPDeviceFunctionRef::create(*this, std::move(moduleFunc));
  }

  // TODO(iposva): Remove once not needed for debugging.
  if (false) {
    std::cerr << "HIPDeviceContext::loadFunction(" << functionName.str()
              << "):\n"
              << code.data() << "\n";
  }
  hipModule_t module;

  HIP_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  HIP_CHECK_ERR(driver_.hipModuleLoadData(&module, code.data()));

  hipFunction_t function;
  HIP_CHECK_ERR(
      driver_.hipModuleGetFunction(&function, module, functionName.data()));

  // Allocate func outside of the lock region to reduce the time spent under
  // lock. If this functionName has been registered concurrently, then we will
  // release this ref outside of the lock region before returning.
  auto moduleFunc = HIPModuleFunctionRef::create(
      std::string(functionName), this->hipDriver(), module, function);
  HIPDeviceFunctionRef result;
  entry = device().insertInFunctionCache(key, *moduleFunc);
  result = HIPDeviceFunctionRef::create(
      *this,
      HIPModuleFunctionRef::take(HIPModuleFunction::cast(entry.release())));
  return result;
}

ErrorOrSuccess HIPDeviceContext::occupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, DeviceFunction &func, int blockSize,
    size_t dynamicSharedMemSize) {
  auto &hipFunc = HIPDeviceFunction::cast(func);
  HIP_WITH_CTX_ERR(*this);
  return (hipFunc.occupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, blockSize, dynamicSharedMemSize));
}

ErrorOrSuccess HIPDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    const size_t *argSizes) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);

  const auto &hipFunc = HIPDeviceFunction::cast(func);
  if (numAttrs > 0)
    RETURN_ERROR("Function launch attributes not supported;");

  HIP_WITH_CTX_ERR(*this);
  return hipFunc.hipCallKernelDirect(gridX, gridY, gridZ, blockX, blockY,
                                     blockZ, sharedMemBytes, args, stream_);
}

std::optional<intptr_t> HIPDeviceContext::getDLPackStream() const {
  ErrorOr<DeviceStreamRef> errStream = stream();
  if (errStream.isError())
    return std::nullopt;
  return reinterpret_cast<intptr_t>((*errStream)->ptr());
}

ErrorOr<DeviceStreamRef> HIPDeviceContext::stream() const {
  // This function is const only for the duration of its execution;
  // `this` is non-const downstream so cast in order to provide
  // mutable reference to the new stream object.
  return HIPDeviceStreamRef::create(const_cast<HIPDeviceContext &>(*this),
                                    reinterpret_cast<hipStream_t>(stream_));
}

ErrorOrSuccess HIPDeviceContext::streamPriorityRange(int *leastPriority,
                                                     int *greatestPriority) {
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(
      driver_.hipDeviceGetStreamPriorityRange(leastPriority, greatestPriority));
  return success();
}

ErrorOr<DeviceGraphBuilderRef>
HIPDeviceContext::createGraphBuilder(DeviceGraphMemoryPoolRef pool) {
  return HIPDeviceGraphBuilder::create(*this, std::move(pool));
}

ErrorOr<DeviceStreamRef> HIPDeviceContext::createStream(int priority) {
  // Ensure stream is created for this context.
  HIP_WITH_CTX_ERR(*this);
  // Default priority is 0 (lowest); compatible with CUDA specs.
  // HIP specs don't explicitly call this out but we know they mirror CUDA.

  // HIP blocking streams are a legacy feature; interpretation of "blocking"
  // is w.r.t. the default stream and does not mean "synchronous".
  // We want our streams to be independent of any other stream.
  unsigned int constexpr kNonBlocking = 0x1;

  hipStream_t stream;
  HIP_CHECK_ERR(this->hipDriver().hipStreamCreateWithPriority(
      &stream, kNonBlocking, priority));

  // The pool takes ownership of the stream; the pool destroys the stream when
  // the context is released.
  streamPool_->emplace_back(stream);

  return HIPDeviceStreamRef::create(*this, stream);
}

unsigned HIPDeviceContext::numStreams() const { return streamPool_->size(); }

ErrorOr<DeviceContextRef> HIPDeviceContext::selectStream(unsigned streamId) {
  // Streams are created once at model setup (see ensureStreamCountForDevices in
  // MEFModel); a stream id past the end of the pool is a programming error.
  if (streamId >= streamPool_->size())
    return Error("invalid stream id: " + llvm::Twine(streamId));

  // Selecting the already-active stream is a no-op: hand back this context.
  if (streamPool_->get(streamId) == stream_)
    return DeviceContextRef::copy(this);

  // Return a retain-all view: a new context co-owning the same stream pool and
  // device (and therefore the same device memory pool), differing only in which
  // pooled stream is active.
  return DeviceContextRef::take(
      new HIPDeviceContext(HIPDevice::cast(device()), streamPool_, streamId));
}

ErrorOr<DeviceStreamRef>
HIPDeviceContext::createExternalStream(void *externalStream) {
  HIP_WITH_DEV_ERR(*this);
  return HIPDeviceStreamRef::create(
      *this, reinterpret_cast<hipStream_t>(externalStream));
}

ErrorOrSuccess HIPDeviceContext::synchronize() {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::synchronize", TRACY_COLOR_PURPLE);

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.hipStreamSynchronize(stream_));
  // Check any pending errors and events after synchronization.
  RETURN_IF_PENDING_ERROR(*this);
  POLL_PENDING_EVENTS(*this);
  return success();
}

ErrorOr<DeviceEventRef> HIPDeviceContext::enqueueEvent() {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  const bool recordedDuringCapture = isStreamCapturing(driver_, stream_);

  hipEvent_t event;
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.hipEventCreateWithFlags(&event, 0));
  HIP_CHECK_ERR(driver_.hipEventRecord(event, stream_));
  return HIPDeviceEventRef::create(HIPDevice::cast(this->device()), event,
                                   recordedDuringCapture);
}

ErrorOrSuccess HIPDeviceContext::enqueueEvent(const DeviceEvent &event) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::enqueueEvent", TRACY_COLOR_PURPLE);

  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(
      driver_.hipEventRecord(HIPDeviceEvent::cast(event).hipEvent(), stream_));
  return success();
}

ErrorOr<DeviceEventRef> HIPDeviceContext::eventCreate(bool enableTiming) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::eventCreate", TRACY_COLOR_PURPLE);

  const bool recordedDuringCapture = isStreamCapturing(driver_, stream_);

  // hipEventDisableTiming = 0x2, hipEventDefault = 0x0
  constexpr unsigned int kHipEventDefault = 0x0;
  constexpr unsigned int kHipEventDisableTiming = 0x2;
  unsigned int flags = enableTiming ? kHipEventDefault : kHipEventDisableTiming;

  hipEvent_t event;
  HIP_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  HIP_CHECK_ERR(driver_.hipEventCreateWithFlags(&event, flags));
  return HIPDeviceEventRef::create(
      HIPDevice::cast(this->device()), event,
      /*recordedDuringCapture=*/recordedDuringCapture);
}

ErrorOrSuccess HIPDeviceContext::enqueueWaitFor(DeviceEvent &event) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::enqueueWaitFor", TRACY_COLOR_PURPLE);

  const auto &hipEvent = HIPDeviceEvent::cast(event);
  HIP_WITH_CTX_ERR(*this);
  HIP_CHECK_ERR(driver_.hipStreamWaitEvent(stream_, hipEvent.hipEvent(), 0));
  return success();
}

ErrorOrSuccess HIPDeviceContext::enqueueWaitFor(DeviceContext &ctx) {
  TRACY_ZONE_SCOPED_NC("HIPDeviceContext::enqueueWaitFor", TRACY_COLOR_PURPLE);

  DEFINE_OR_RETURN_ERROR(DeviceEventRef, event, ctx.enqueueEvent());
  RETURN_IF_ERROR(enqueueWaitFor(*event));
  return success();
}

ErrorOr<DeviceTimerRef> HIPDeviceContext::startTimer() {
  hipEvent_t start;
  hipEvent_t stop;

  HIP_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  HIP_CHECK_ERR(driver_.hipEventCreateWithFlags(&start, 0));
  HIP_CHECK_ERR(driver_.hipEventCreateWithFlags(&stop, 0));

  auto timer = HIPDeviceTimerRef::create(*this, start, stop);

  HIP_CHECK_ERR(driver_.hipEventRecord(timer->hipStartEvent(), stream_));

  return timer;
}

ErrorOr<std::chrono::nanoseconds>
HIPDeviceContext::stopTimer(const DeviceTimer &timer) {
  const auto &hipTimer = HIPDeviceTimer::cast(timer);

  HIP_WITH_CTX_ERR(*this);
  POLL_PENDING_EVENTS(*this);
  HIP_CHECK_ERR(driver_.hipEventRecord(hipTimer.hipStopEvent(), stream_));
  HIP_CHECK_ERR(driver_.hipEventSynchronize(hipTimer.hipStopEvent()));

  float millis;
  HIP_CHECK_ERR(driver_.hipEventElapsedTime(&millis, hipTimer.hipStartEvent(),
                                            hipTimer.hipStopEvent()));

  constexpr double nanosPerMillisecond = 1000.0 * 1000.0;
  double nanos = static_cast<double>(millis) * nanosPerMillisecond;
  return std::chrono::nanoseconds(static_cast<int64_t>(nanos));
}

ErrorOr<Device::MemInfo> HIPDeviceContext::getDeviceMemoryInfo() const {
  HIP_WITH_CTX(*this);
  // To get more accurate memory stats, ensure that at least all pending memory
  // allocations and frees have completed for this context's stream.
  HIP_CHECK_ERR(driver_.hipStreamSynchronize(stream_));

  return device().getMemoryInfo();
}

bool HIPDeviceContext::isGraphCapturing() const {
  return isStreamCapturing(hipDriver(), hipStream());
}

const HIPDriver &HIPDeviceContext::hipDriver() const { return driver_; }

hipDevice_t HIPDeviceContext::hipDevice() const { return device_; }

hipStream_t HIPDeviceContext::hipStream() const { return stream_; }

HIPDeviceContext &HIPDeviceContext::cast(DeviceContext &ctx) {
  return DeviceContext::cast<HIPDeviceContext>(ctx);
}

const HIPDeviceContext &HIPDeviceContext::cast(const DeviceContext &ctx) {
  return DeviceContext::cast<HIPDeviceContext>(ctx);
}

const HIPDeviceContext *
HIPDeviceContext::verified_cast(const DeviceContext *ctx) {
  return DeviceContext::cast<HIPDeviceContext>(ctx);
}

} // namespace M::Driver
