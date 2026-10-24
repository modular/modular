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

#include "max/c/device.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "max/c/common.h"
#include "max/max_license/API/c/detail/helpers.h"
#include "max/max_license/API/c/detail/types.h"

using namespace M;

namespace {

// Mirrors GraphCompiler::Driver::createCPUDevice (Driver.cpp): builds the
// host TargetInfo, opens the "cpu" MLRT context, and gates on isCompatible().
ErrorOr<M_Device *> newCPUDevice() {
  auto targetInfoOr = M::getHostTargetInfo();
  if (targetInfoOr.isError())
    return targetInfoOr.takeError();

  auto ctxOr = M::Driver::DeviceContext::create("cpu");
  if (ctxOr.isError())
    return ctxOr.takeError();
  M::Driver::DeviceContextRef ctx = ctxOr.takeValue();

  if (auto compatibleOr = ctx->device().isCompatible(); compatibleOr.isError())
    return compatibleOr.takeError();

  DeviceSpec spec;
  spec.ref.label = kCPULabel;
  spec.ref.id = 0;
  spec.target = targetInfoOr.takeValue();
  return new M_Device{std::make_shared<const DeviceSpec>(std::move(spec)),
                      std::move(ctx)};
}

// Mirrors GraphCompiler::Driver::createAccelerator (Driver.cpp): opens (or
// reuses) the "gpu" MLRT context for `gpuID` and gates on isCompatible().
ErrorOr<M_Device *> newAcceleratorDevice(ssize_t gpuID) {
  size_t id = gpuID >= 0 ? gpuID : 0;

  auto ctxOr =
      M::Driver::DeviceContext::getOrCreate(M::Driver::Device::gpuAPI, id);
  if (ctxOr.isError())
    return ctxOr.takeError();
  M::Driver::DeviceContextRef ctx = ctxOr.takeValue();

  if (auto compatibleOr = ctx->device().isCompatible(); compatibleOr.isError())
    return compatibleOr.takeError();

  DeviceSpec spec;
  spec.ref.label = kGPULabel;
  spec.ref.id = gpuID;
  return new M_Device{std::make_shared<const DeviceSpec>(std::move(spec)),
                      std::move(ctx)};
}

} // namespace

MODULAR_API_EXPORT M_Device *M_newDevice(M_DeviceType type, int id,
                                         M_Status *status) {
  ErrorOr<M_Device *> deviceOr = [&]() -> ErrorOr<M_Device *> {
    if (type == M_HOST)
      return newCPUDevice();
    if (type == M_ACCELERATOR)
      return newAcceleratorDevice(id);
    return Error("Invalid device type.");
  }();

  if (deviceOr.isError()) {
    M::Detail::setError(status, deviceOr.takeError());
    return nullptr;
  }
  M::Detail::setOK(status);
  return deviceOr.takeValue();
}

MODULAR_API_EXPORT M_DeviceType M_getDeviceType(const M_Device *device) {
  // Any non-CPU label is treated as an accelerator. EngineContext::create
  // guarantees only known labels (cpu/gpu) reach this far.
  return device->spec->ref.label == kCPULabel ? M_HOST : M_ACCELERATOR;
}

MODULAR_API_EXPORT int M_getDeviceId(const M_Device *device) {
  return device->spec->ref.id;
}

MODULAR_API_EXPORT int M_isHostDevice(const M_Device *device) {
  return device->context->device().api() == M::Driver::Device::cpuAPI;
}

MODULAR_API_EXPORT void M_synchronizeDevice(M_Device *device,
                                            M_Status *status) {
  auto result = device->context->synchronize();
  if (result.isError()) {
    M::Detail::setError(status, result.takeError());
    return;
  }

  M::Detail::setOK(status);
}

MODULAR_API_EXPORT const char *M_getDeviceLabel(const M_Device *device) {
  return device->spec->ref.label.c_str();
}

MODULAR_API_EXPORT void M_freeDevice(M_Device *device) { delete device; }

MODULAR_API_EXPORT int M_getAcceleratorCount(void) {
  return Driver::Device::numberOfDevices(Driver::Device::gpuAPI);
}
