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
#include "Device.h"
#include "HAL/M_driver_device.h"
#include "llvm/ADT/StringRef.h"

#include "HAL/M_driver_device.h"

#include <dlpack/dlpack.h>

#include <cstring>

using M::Target::CUDA::cudaErrorStatus;

DriverResult<M_driver_device_t> M_driver_device::create(M_driver_driver &driver,
                                                        int64_t deviceId) {
  DriverResult<int64_t> count = driver.deviceCount();
  if (!count)
    return tl::unexpected(count.error());
  if (deviceId < 0 || deviceId >= *count)
    return tl::unexpected(M_DRIVER_STATUS_BAD_DEVICE);

  CUdevice cudaDevice{};
  if (CUresult r = driver.rawCuda.cuDeviceGet(&cudaDevice, deviceId);
      r != CUDA_SUCCESS)
    return tl::unexpected(cudaErrorStatus(
        &driver, r, "cuDeviceGet device " + llvm::Twine(deviceId)));

  return new M_driver_device(driver, cudaDevice);
}

M_driver_status_t M_driver_device::destroy(M_driver_device_t device) {
  delete device;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device::canAccessPeer(const M_driver_device &peer,
                                                 bool *out_can_access) const {
  if (out_can_access == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (deviceId == peer.deviceId) {
    *out_can_access = false;
    return M_DRIVER_STATUS_SUCCESS;
  }
  int canAccess = 0;
  if (CUresult r = driver.rawCuda.cuDeviceCanAccessPeer(&canAccess, deviceId,
                                                        peer.deviceId);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&driver, r, "cuDeviceCanAccessPeer");
  int nativeAtomics = 0;
  if (CUresult r = driver.rawCuda.cuDeviceGetP2PAttribute(
          &nativeAtomics, CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
          deviceId, peer.deviceId);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&driver, r, "cuDeviceGetP2PAttribute");
  *out_can_access = canAccess != 0 && nativeAtomics != 0;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device::property(llvm::StringRef name, void *value) {
  if (name.empty() || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (name == M_DRIVER_DEVICE_PROPERTY_NAME) {
    if (CUresult r = driver.rawCuda.cuDeviceGetName(
            static_cast<char *>(value), M_DRIVER_DEVICE_PROPERTY_MAX_LEN,
            deviceId);
        r != CUDA_SUCCESS)
      return cudaErrorStatus(&driver, r, "cuDeviceGetName");
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_ARCH) {
    int32_t computeCapabilityMajor;
    if (M_driver_status_t status = attribute(75 /* COMPUTE_CAPABILITY_MAJOR */,
                                             &computeCapabilityMajor))
      return status;

    int32_t computeCapabilityMinor;
    if (M_driver_status_t status = attribute(76 /* COMPUTE_CAPABILITY_MINOR */,
                                             &computeCapabilityMinor))
      return status;

    int cc = computeCapabilityMajor * 10 + computeCapabilityMinor;
    std::string sm = "sm_" + std::to_string(cc);

    // For architectures above 90 we use the "a" variants to enable hardware
    // specific features at the cost of backward and forward compatibility.
    if (cc >= 90)
      sm += "a";

    std::strncpy(static_cast<char *>(value), sm.c_str(),
                 M_DRIVER_DEVICE_PROPERTY_MAX_LEN);
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED) {
    int32_t integrated = 0;
    // Not UNIFIED_ADDRESSING (41): discrete GPUs set that too.
    if (M_driver_status_t status = attribute(18 /* INTEGRATED */, &integrated))
      return status;
    *static_cast<int64_t *>(value) = integrated != 0 ? 1 : 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCUDA;
    dl->device_id = static_cast<int32_t>(deviceId);
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCUDAHost;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_device::attribute(int32_t attribute,
                                             int32_t *value) {
  if (value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  int cuValue = 0;
  if (CUresult r = driver.rawCuda.cuDeviceGetAttribute(
          &cuValue, static_cast<CUdevice_attribute>(attribute), deviceId);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&driver, r, "cuDeviceGetAttribute");
  *value = static_cast<int32_t>(cuValue);
  return M_DRIVER_STATUS_SUCCESS;
}
