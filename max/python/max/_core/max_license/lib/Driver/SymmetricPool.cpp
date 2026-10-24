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

#include "_core/Driver/SymmetricPool.h"

#include <nanobind/nanobind.h>

#include <stdexcept>

using namespace M;
using namespace M::Driver;

#if MODULAR_HAL_DEVICE_CONTEXT

namespace {
[[noreturn]] void throwUnsupported() {
  throw std::runtime_error(
      "symmetric multicast pools are not supported by the HAL device context");
}
} // namespace

Python::SymmetricPool
Python::SymmetricPool::allocate(const std::vector<Python::PyDeviceTy> &,
                                size_t) {
  throwUnsupported();
}

std::optional<Python::SymmetricPool>
Python::SymmetricPool::lookup(const std::vector<Python::PyDeviceTy> &) {
  return std::nullopt;
}

bool Python::SymmetricPool::isSupported(
    const std::vector<Python::PyDeviceTy> &) {
  return false;
}

size_t Python::SymmetricPool::getByteSize() const { throwUnsupported(); }

Python::Buffer
Python::SymmetricPool::unicastView(const Python::PyDeviceTy &) const {
  throwUnsupported();
}

Python::Buffer
Python::SymmetricPool::multicastView(const Python::PyDeviceTy &) const {
  throwUnsupported();
}

#else
namespace {

/// Collects the MLRT contexts behind a list of Python devices.
std::vector<DeviceContextRef>
contextsFor(const std::vector<Python::PyDeviceTy> &devices) {
  std::vector<DeviceContextRef> contexts;
  contexts.reserve(devices.size());
  for (const auto &device : devices) {
    if (!device)
      throw std::runtime_error("null device in symmetric pool device list");
    contexts.push_back(device->ctx);
  }
  return contexts;
}

/// Wraps one driver-owned view as a flat, non-owning uint8 Buffer.
Python::Buffer wrapView(DeviceBufferRef view, const Python::PyDeviceTy &device,
                        size_t byteSize) {
  int64_t numBytes = static_cast<int64_t>(byteSize);
  auto storage = std::make_shared<Python::BufferStorage>(
      std::move(view), TensorSpec(ArrayRef<int64_t>(&numBytes, 1), DType::ui8),
      Python::Usage::Default);
  // Device memory: no host-hazard record to track.
  return Python::Buffer(std::move(storage), device, nullptr);
}

} // namespace

Python::SymmetricPool
Python::SymmetricPool::allocate(const std::vector<Python::PyDeviceTy> &devices,
                                size_t byteSize) {
  std::vector<DeviceContextRef> contexts = contextsFor(devices);

  nb::gil_scoped_release release;
  ErrorOr<DeviceMulticastBufferRef> poolOr =
      M::Driver::SymmetricPool::create(contexts, byteSize);
  if (poolOr.isError())
    throw std::runtime_error(poolOr.getError());

  return Python::SymmetricPool{poolOr.takeValue(), devices};
}

std::optional<Python::SymmetricPool>
Python::SymmetricPool::lookup(const std::vector<Python::PyDeviceTy> &devices) {
  std::vector<DeviceContextRef> contexts = contextsFor(devices);
  DeviceMulticastBufferRef found = M::Driver::SymmetricPool::lookup(contexts);
  if (!found)
    return std::nullopt;
  return Python::SymmetricPool{std::move(found), devices};
}

bool Python::SymmetricPool::isSupported(
    const std::vector<Python::PyDeviceTy> &devices) {
  if (devices.empty())
    return false;
  for (const auto &device : devices) {
    if (!device)
      return false;
    ErrorOr<bool> supportedOr = device->ctx->device().supportsMulticast();
    // Cannot answer counts as unsupported: guessing "yes" reserves memory the
    // pool would never use.
    if (supportedOr.isError() || !supportedOr.get())
      return false;
  }
  return true;
}

size_t Python::SymmetricPool::getByteSize() const { return pool->byteSize(); }

Python::Buffer
Python::SymmetricPool::unicastView(const Python::PyDeviceTy &device) const {
  ErrorOr<DeviceBufferRef> viewOr = pool->unicastBufferFor(*device->ctx);
  if (viewOr.isError())
    throw std::runtime_error(viewOr.getError());
  return wrapView(viewOr.takeValue(), device, pool->byteSize());
}

Python::Buffer
Python::SymmetricPool::multicastView(const Python::PyDeviceTy &device) const {
  ErrorOr<DeviceBufferRef> viewOr = pool->multicastBuffer(*device->ctx);
  if (viewOr.isError())
    throw std::runtime_error(viewOr.getError());
  return wrapView(viewOr.takeValue(), device, pool->byteSize());
}

#endif // MODULAR_HAL_DEVICE_CONTEXT
