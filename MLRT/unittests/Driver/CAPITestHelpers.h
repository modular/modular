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
// Handles and helpers shared by the driver C API tests, which are a reader
// for the last error message, an owner for each handle kind that releases it
// at scope exit, and a way to leave a pending error on a driver context.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_UNITTESTS_DRIVER_CAPITESTHELPERS_H
#define MLRT_UNITTESTS_DRIVER_CAPITESTHELPERS_H

#include "MLRT/Driver/CAPI/DriverCAPI.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"

#include "gtest/gtest.h"

#include <string>

namespace capi_test {

/// Reads back the message of the last failed call.
inline std::string lastError() {
  const size_t size = M_DriverLastErrorMessage(nullptr, 0);
  if (size == 0)
    return "";
  std::string message(size - 1, '\0');
  M_DriverLastErrorMessage(message.data(), size);
  return message;
}

/// A device handle that releases itself at scope exit.
class DeviceHandle {
public:
  DeviceHandle() = default;
  DeviceHandle(const DeviceHandle &) = delete;
  DeviceHandle &operator=(const DeviceHandle &) = delete;
  ~DeviceHandle() { release(); }

  M_DriverDevice **receive() {
    release();
    return &handle_;
  }

  M_DriverDevice *get() const { return handle_; }

private:
  // The destroy calls refuse a null handle, so an empty owner skips them.
  void release() {
    if (handle_ != nullptr)
      M_DriverDeviceDestroy(handle_);
    handle_ = nullptr;
  }

  M_DriverDevice *handle_ = nullptr;
};

/// A stream handle that releases itself at scope exit.
class StreamHandle {
public:
  StreamHandle() = default;
  StreamHandle(const StreamHandle &) = delete;
  StreamHandle &operator=(const StreamHandle &) = delete;
  ~StreamHandle() { release(); }

  M_DriverStream **receive() {
    release();
    return &handle_;
  }

  M_DriverStream *get() const { return handle_; }

private:
  // The destroy calls refuse a null handle, so an empty owner skips them.
  void release() {
    if (handle_ != nullptr)
      M_DriverStreamDestroy(handle_);
    handle_ = nullptr;
  }

  M_DriverStream *handle_ = nullptr;
};

/// A buffer handle that releases itself at scope exit.
class BufferHandle {
public:
  BufferHandle() = default;
  BufferHandle(const BufferHandle &) = delete;
  BufferHandle &operator=(const BufferHandle &) = delete;
  ~BufferHandle() { release(); }

  M_DriverBuffer **receive() {
    release();
    return &handle_;
  }

  M_DriverBuffer *get() const { return handle_; }

private:
  // The destroy calls refuse a null handle, so an empty owner skips them.
  void release() {
    if (handle_ != nullptr)
      M_DriverBufferDestroy(handle_);
    handle_ = nullptr;
  }

  M_DriverBuffer *handle_ = nullptr;
};

/// A host-memory registration that unregisters at scope exit.
class MappingHandle {
public:
  MappingHandle() = default;
  MappingHandle(const MappingHandle &) = delete;
  MappingHandle &operator=(const MappingHandle &) = delete;
  ~MappingHandle() { release(); }

  M_DriverMapping **receive() {
    release();
    return &handle_;
  }

  M_DriverMapping *get() const { return handle_; }

private:
  // The destroy calls refuse a null handle, so an empty owner skips them.
  void release() {
    if (handle_ != nullptr)
      M_DriverMappingDestroy(handle_);
    handle_ = nullptr;
  }

  M_DriverMapping *handle_ = nullptr;
};

/// Builds a device handle for the CPU device.
/// Plants a pending error through public API: the deferred free of an owning
/// host buffer over foreign memory is refused and recorded, not returned, and
/// the synchronize that runs it still succeeds.
inline M::ErrorOrSuccess plantPendingError(M::Driver::DeviceContext &context) {
  static char foreign[64];
  context.createHostBufferOwning(foreign, sizeof foreign, /*owning=*/true);
  return context.synchronize();
}

inline bool makeCpuDevice(DeviceHandle &device) {
  const M_DriverStatus status = M_DriverCPUDeviceCreate(-1, device.receive());
  EXPECT_EQ(status, M_DRIVER_OK) << lastError();
  return status == M_DRIVER_OK;
}

} // namespace capi_test

#endif // MLRT_UNITTESTS_DRIVER_CAPITESTHELPERS_H
