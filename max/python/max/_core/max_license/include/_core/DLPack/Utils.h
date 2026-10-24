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
// Private utilities for working with the DLPack python interface.
//
//===----------------------------------------------------------------------===//

#ifndef MAX_PYTHON_DLPACK_UTILS_H
#define MAX_PYTHON_DLPACK_UTILS_H

#include "nanobind/nanobind.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"

#include "Support/ML/DType.h"

namespace nb = nanobind;

namespace M::DLPack {

/// Maps a DLPack `DLDataType` to a `M::DType::Cases` (the value the Python
/// `DType` enum binds to). Rejects vector types (lanes != 1) and codes with no
/// `M::DType` equivalent.
M::DType::Cases dtypeForDLDataType(DLDataType dlDataType);

/// Maps a `M::DType` to a DLPack `DLDataType` (single lane).
DLDataType dlDataTypeForDType(M::DType dtype);

namespace Detail {
struct DLManagedTensorDeleter {
  void operator()(DLManagedTensor *tensor) {
    if (tensor->deleter)
      tensor->deleter(tensor);
  }
};
} // namespace Detail

/// RAII wrapper for DLManagedTensors
using DLManagedTensorWrapper =
    std::unique_ptr<DLManagedTensor, Detail::DLManagedTensorDeleter>;
static_assert(sizeof(DLManagedTensorWrapper) == sizeof(DLManagedTensor *));

/// Calls the __dlpack__ method on the given object, ensuring that its result
/// complies with the dlpack contract, and returns the underlying tensor. A
/// stream ID is needed (or none if on host) for correct synchronization.
DLManagedTensorWrapper callDLPack(const nb::object &arrayOrTensor,
                                  std::optional<intptr_t> streamID);

/// Calls the __dlpack_device__ method on the given object, ensures that its
/// result complies with the dlpack contract, and return the underlying device
/// type and ID.
std::pair<DLDeviceType, ssize_t>
callDLPackDevice(const nb::object &arrayOrTensor);

/// Builds a DLPack capsule sharing an existing allocation, zero-copy, from
/// plain values. Wraps the memory in a nanobind `ndarray` view that holds
/// `owner` alive (via `reference_internal`) and returns the capsule from
/// nanobind's own `__dlpack__`; nanobind's deleter releases `owner`.
/// `maxVersion` is the consumer's maximum supported DLPack version: `(1, 0)`
/// or newer yields a versioned `dltensor_versioned` capsule, otherwise the
/// classic unversioned `dltensor`.
nb::object
makeDLPackCapsule(nb::handle owner, uintptr_t data, int32_t deviceType,
                  int32_t deviceId, M::DType::Cases dtype,
                  const std::vector<int64_t> &shape,
                  std::optional<std::vector<int64_t>> strides,
                  std::optional<std::pair<int64_t, int64_t>> maxVersion);
} // namespace M::DLPack

#endif
