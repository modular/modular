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
// This file implements the Virtual device context for compile-only mode.
//
//===----------------------------------------------------------------------===//

#include "VirtualDeviceContext.h"

#include "MLRT/Driver/DeviceContext/ErrorMacros.h"
#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/max_license/Tracy/Tracy.h"

#include <cassert>
#include <string>

namespace M::Driver {

//===----------------------------------------------------------------------===//
// VirtualDevice
//===----------------------------------------------------------------------===//

VirtualDevice::VirtualDevice(
    int id, std::string name,
    std::unique_ptr<CompilationDevice> compilationDevice)
    : Device(compilationDevice->api(), std::move(name), id, Device::MemInfo(),
             M::kPreferredMemoryAlignment),
      compilationDevice_(std::move(compilationDevice)) {
  assert(compilationDevice_ &&
         "VirtualDevice requires a non-null compilation device");
}

VirtualDevice::~VirtualDevice() = default;

ErrorOr<int> VirtualDevice::getApiVersion() const {
  return compilationDevice_->getApiVersion();
}

ErrorOr<DriverVersion &> VirtualDevice::getDriverVersion() const {
  return compilationDevice_->getDriverVersion();
}

ErrorOrSuccess VirtualDevice::isCompatible() const { return success(); }

ErrorOr<int> VirtualDevice::computeCapability() const {
  RETURN_ERROR("VirtualDevice does not support compute capability");
}

ErrorOr<const char *> VirtualDevice::getArchName() const {
  return compilationDevice_->getArchName();
}

ErrorOr<bool> VirtualDevice::canAccess(const Device &peer) const {
  // Virtual devices don't support peer access
  return false;
}

ErrorOr<int> VirtualDevice::getAttribute(Attribute attr) const {
  RETURN_ERROR("VirtualDevice does not support getAttribute");
}

ErrorOr<std::string> VirtualDevice::getPciBusId() const {
  RETURN_ERROR("VirtualDevice does not support getPciBusId");
}

int VirtualDevice::deviceCount() const {
  // Virtual devices don't have real device count
  return 1;
}

ErrorOrSuccess VirtualDevice::chunkAlloc(void **ptr, size_t byteSize,
                                         bool isHost) {
  RETURN_ERROR("VirtualDevice does not support memory allocation");
}

ErrorOrSuccess VirtualDevice::chunkFree(void *ptr, bool isHost) {
  RETURN_ERROR("VirtualDevice does not support memory deallocation");
}

ErrorOr<Device::MemInfo> VirtualDevice::getMemoryInfo() const {
  // Return simulated memory info (192GB, typical for B200)
  Device::MemInfo info;
  info.free = 192ULL * 1024 * 1024 * 1024;
  info.total = 192ULL * 1024 * 1024 * 1024;
  return info;
}

ErrorOr<void *> VirtualDevice::getCurrentDriverContext() const {
  RETURN_ERROR("VirtualDevice does not have a driver context");
}

ErrorOrSuccess VirtualDevice::setCurrentDriverContext(void *ctx) {
  RETURN_ERROR("VirtualDevice does not support setting driver context");
}

ErrorOr<DeviceContextRef> VirtualDevice::createContext() {
  // Create a virtual context that uses the compilation device
  return VirtualDeviceContextRef::create(*this);
}

const VirtualDevice &VirtualDevice::cast(const Device &device) {
  // Virtual devices can have any API based on their backing device.
  // The cast is valid as long as the device is actually a VirtualDevice.
  return static_cast<const VirtualDevice &>(device);
}

//===----------------------------------------------------------------------===//
// VirtualDeviceStream
//===----------------------------------------------------------------------===//

VirtualDeviceStream::VirtualDeviceStream(VirtualDeviceContext &ctx)
    : DeviceStream(ctx) {}

ErrorOrSuccess VirtualDeviceStream::synchronize() {
  // No-op for virtual devices
  return success();
}

ErrorOr<DeviceEventRef> VirtualDeviceStream::enqueueEvent() const {
  RETURN_ERROR("VirtualDevice does not support events");
}

ErrorOrSuccess
VirtualDeviceStream::eventRecord(const DeviceEvent &event) const {
  // No-op for virtual devices
  return success();
}

ErrorOrSuccess
VirtualDeviceStream::enqueueWaitForEvent(const DeviceEvent &event) const {
  // No-op for virtual devices
  return success();
}

ErrorOrSuccess VirtualDeviceStream::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t /*argCount*/,
    const size_t * /*argSizes*/) const {
  RETURN_ERROR("VirtualDevice does not support function execution");
}

ErrorOrSuccess VirtualDeviceStream::enqueueHostFunc(void (*fn)(void *),
                                                    void *userData) const {
  RETURN_ERROR("VirtualDevice does not support host functions");
}

ErrorOrSuccess
VirtualDeviceStream::enqueueWaitOnHostValue(const CompletionFlag &flag,
                                            uint64_t value) const {
  RETURN_ERROR("VirtualDevice does not support host-value waits");
}

ErrorOrSuccess VirtualDeviceStream::beginCapture() const {
  RETURN_ERROR("VirtualDevice does not support graph capture");
}

ErrorOr<DeviceGraphRef> VirtualDeviceStream::endCapture() {
  RETURN_ERROR("VirtualDevice does not support graph capture");
}

void *VirtualDeviceStream::ptr() const {
  // Return nullptr for virtual streams
  return nullptr;
}

const VirtualDeviceStream *
VirtualDeviceStream::verified_cast(const DeviceStream *stream) {
  if (!stream)
    return nullptr;
  // Virtual device streams can have any API based on their backing device.
  return static_cast<const VirtualDeviceStream *>(stream);
}

//===----------------------------------------------------------------------===//
// VirtualDeviceContext
//===----------------------------------------------------------------------===//

VirtualDeviceContext::VirtualDeviceContext(VirtualDevice &device)
    : DeviceContext(device, TypeID::get<VirtualDeviceContext>()) {}

VirtualDeviceContext::~VirtualDeviceContext() = default;

ErrorOrSuccess VirtualDeviceContext::enqueueCopyToDevice(void *dst,
                                                         const void *src,
                                                         size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueCopyToDevice",
                       TRACY_COLOR_PURPLE);
  RETURN_ERROR("VirtualDevice does not support memory operations");
}

ErrorOrSuccess VirtualDeviceContext::enqueueCopyFromDevice(void *dst,
                                                           const void *src,
                                                           size_t byteSize) {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueCopyFromDevice",
                       TRACY_COLOR_PURPLE);
  RETURN_ERROR("VirtualDevice does not support memory operations");
}

ErrorOrSuccess VirtualDeviceContext::memCopyDtoD(DeviceBuffer &dst,
                                                 DeviceBuffer &src,
                                                 size_t srcByteOffset,
                                                 size_t srcByteSize,
                                                 bool /*crossStreamSync*/) {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::memCopyDtoD", TRACY_COLOR_PURPLE);
  RETURN_ERROR("VirtualDevice does not support memory operations");
}

ErrorOrSuccess VirtualDeviceContext::enqueueSetMemory(void *dst,
                                                      size_t byteSize,
                                                      uint64_t val,
                                                      size_t valSize) {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueSetMemory",
                       TRACY_COLOR_PURPLE);
  RETURN_ERROR("VirtualDevice does not support memory operations");
}

ErrorOr<DeviceFunctionRef> VirtualDeviceContext::loadFunction(
    StringRef moduleName, StringRef functionName, StringRef code,
    int32_t maxDynamicSharedBytes, StringRef debugLevel,
    int32_t optimizationLevel) {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::loadFunction",
                       TRACY_COLOR_PURPLE);
  RETURN_ERROR("VirtualDevice does not support function loading");
}

ErrorOrSuccess VirtualDeviceContext::enqueueFunctionExecDirect(
    const DeviceFunction &func, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ, uint32_t sharedMemBytes,
    void *attrs, uint32_t numAttrs, void **args, uint32_t argCount,
    const size_t *argSizes) {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::enqueueFunctionExecDirect",
                       TRACY_COLOR_PURPLE);
  RETURN_ERROR("VirtualDevice does not support function execution");
}

ErrorOr<DeviceStreamRef> VirtualDeviceContext::stream() const {
  // Return a virtual stream for compile-only mode
  // This function is const only for the duration of its execution;
  // `this` is non-const downstream so cast in order to provide
  // mutable reference to the new stream object.
  return VirtualDeviceStreamRef::create(
      const_cast<VirtualDeviceContext &>(*this));
}

ErrorOr<DeviceStreamRef> VirtualDeviceContext::createStream(int priority) {
  // Return a virtual stream for compile-only mode (ignore priority)
  return VirtualDeviceStreamRef::create(*this);
}

ErrorOrSuccess VirtualDeviceContext::synchronize() {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::synchronize", TRACY_COLOR_PURPLE);
  // No-op for virtual devices
  return success();
}

ErrorOr<DeviceEventRef> VirtualDeviceContext::enqueueEvent() {
  RETURN_ERROR("VirtualDevice does not support events");
}

ErrorOrSuccess VirtualDeviceContext::enqueueEvent(const DeviceEvent &event) {
  RETURN_ERROR("VirtualDevice does not support events");
}

ErrorOrSuccess VirtualDeviceContext::enqueueWaitFor(DeviceEvent &event) {
  RETURN_ERROR("VirtualDevice does not support events");
}

ErrorOrSuccess VirtualDeviceContext::enqueueWaitFor(DeviceContext &ctx) {
  RETURN_ERROR("VirtualDevice does not support events");
}

ErrorOr<DeviceTimerRef> VirtualDeviceContext::startTimer() {
  RETURN_ERROR("VirtualDevice does not support timers");
}

ErrorOr<std::chrono::nanoseconds>
VirtualDeviceContext::stopTimer(const DeviceTimer &timer) {
  RETURN_ERROR("VirtualDevice does not support timers");
}

ErrorOr<Device::MemInfo> VirtualDeviceContext::getDeviceMemoryInfo() const {
  return VirtualDevice::cast(device()).getMemoryInfo();
}

ErrorOr<DeviceStreamRef>
VirtualDeviceContext::createExternalStream(void *externalStream) {
  RETURN_ERROR("VirtualDevice does not support external streams");
}

ErrorOr<DeviceEventRef> VirtualDeviceContext::eventCreate(bool enableTiming) {
  RETURN_ERROR("VirtualDevice does not support events");
}

ErrorOr<DeviceGraphBuilderRef>
VirtualDeviceContext::createGraphBuilder(DeviceGraphMemoryPoolRef pool) {
  RETURN_ERROR("VirtualDevice does not support device graphs");
}

ErrorOr<CompletionFlagRef> VirtualDeviceContext::allocateCompletionFlag() {
  RETURN_ERROR("VirtualDevice does not support completion flags");
}

ErrorOrSuccess VirtualDeviceContext::freeCompletionFlag(void *hostPtr) {
  RETURN_ERROR("VirtualDevice does not support completion flags");
}

ErrorOr<void *>
VirtualDeviceContext::getHostAccessiblePointer(const void *devicePtr) {
  RETURN_ERROR("VirtualDevice does not support host-accessible pointers");
}

const VirtualDeviceContext &
VirtualDeviceContext::cast(const DeviceContext &ctx) {
  return DeviceContext::cast<VirtualDeviceContext>(ctx);
}

const VirtualDeviceContext *
VirtualDeviceContext::verified_cast(const DeviceContext *ctx) {
  return DeviceContext::cast<VirtualDeviceContext>(ctx);
}

ErrorOrSuccess VirtualDeviceContext::memAlloc(void **ptr, size_t byteSize,
                                              Device::AllocMode mode) {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::memAlloc", TRACY_COLOR_PURPLE);
  RETURN_ERROR("VirtualDevice does not support memory allocation");
}

ErrorOrSuccess VirtualDeviceContext::memFree(void *ptr, Device::FreeKind kind) {
  TRACY_ZONE_SCOPED_NC("VirtualDeviceContext::memFree", TRACY_COLOR_PURPLE);
  RETURN_ERROR("VirtualDevice does not support memory deallocation");
}

ErrorOrSuccess VirtualDeviceContext::memWrap(void **devPtr, void *hostPtr,
                                             size_t byteSize) {
  RETURN_ERROR("VirtualDevice does not support wrapping host memory");
}

void *VirtualDeviceContext::driverContext() const {
  // Return nullptr for virtual contexts
  return nullptr;
}

DeviceContext::MMapFileImplResult VirtualDeviceContext::memmapFileImpl(
    const std::filesystem::path &filename) const {
  RETURN_ERROR("VirtualDevice does not support memory mapping");
}

} // namespace M::Driver
