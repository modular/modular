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
// DLPack interop for max.sys._hal, backed by the shared _core DLPack leaf
// (`//max/python/max/_core/max_license:DLPackPython`). This is an interop skin
// on top of the Python Array: it works in DLPack terms (dlpack.h structs, via
// the leaf) plus `M::DType`, never touching the HAL/driver. The struct layout
// and the dtype<->DLPack mapping come from the leaf, so nothing is
// hand-mirrored.
//
// Exposed as the `max._core.dlpack` submodule.
//
//===----------------------------------------------------------------------===//

#include "_core/DLPack/Utils.h"
#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/vector.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Owns an imported producer's managed tensor and exposes its `dl_tensor`
// fields to Python. `array.py` reads the fields, wraps the memory via the HAL,
// and holds this object on the resulting buffer so that the producer's deleter
// runs (releasing the foreign allocation) exactly when the buffer is dropped.
class DLPackImport {
public:
  explicit DLPackImport(M::DLPack::DLManagedTensorWrapper managed)
      : managedTensor(std::move(managed)) {}

  uintptr_t data() const {
    return reinterpret_cast<uintptr_t>(managedTensor->dl_tensor.data);
  }
  int64_t byteOffset() const {
    return static_cast<int64_t>(managedTensor->dl_tensor.byte_offset);
  }
  // The producer's dtype as a `M::DType`; the shared leaf owns the DLPack
  // code/bits mapping, so array.py never handles the numeric codes.
  M::DType::Cases dtype() const {
    return M::DLPack::dtypeForDLDataType(managedTensor->dl_tensor.dtype);
  }

  std::vector<int64_t> shape() const {
    return {managedTensor->dl_tensor.shape,
            managedTensor->dl_tensor.shape + managedTensor->dl_tensor.ndim};
  }
  std::optional<std::vector<int64_t>> strides() const {
    if (!managedTensor->dl_tensor.strides)
      return std::nullopt;
    return std::vector<int64_t>{managedTensor->dl_tensor.strides,
                                managedTensor->dl_tensor.strides +
                                    managedTensor->dl_tensor.ndim};
  }

private:
  M::DLPack::DLManagedTensorWrapper managedTensor;
};

// Imports a same-device producer zero-copy. `stream` is the native queue handle
// we lend it: `callDLPack` passes it to the producer's `__dlpack__(stream=...)`
// so the producer orders its work on our stream, then renames the capsule to
// `used_dltensor` so the deleter is ours.
DLPackImport importDLPack(nb::object producer, std::optional<intptr_t> stream) {
  auto managed = M::DLPack::callDLPack(producer, stream);
  if (managed->dl_tensor.ndim < 0) {
    PyErr_SetString(PyExc_BufferError,
                    "DLPack producer returned a tensor with negative ndim");
    throw nb::python_error();
  }
  return DLPackImport{std::move(managed)};
}

} // namespace

void M::Graph::Python::bindDLPackModule(nb::module_ m) {
  // Export: build a capsule sharing an Array's device memory zero-copy.
  // `array.py` computes the raw description (address/device/dtype/shape/
  // strides) and the consumer's `max_version`; nanobind's ndarray builds the
  // capsule and holds `owner` alive until the consumer (or the capsule
  // destructor) releases it.
  m.def("make_dlpack_capsule", &M::DLPack::makeDLPackCapsule, "owner"_a,
        "data"_a, "device_type"_a, "device_id"_a, "dtype"_a, "shape"_a,
        "strides"_a, "max_version"_a.none());

  // Import: consume a producer's capsule and expose its dl_tensor fields. The
  // returned object owns the producer's deleter obligation until dropped.
  nb::class_<DLPackImport>(m, "DLPackImport")
      .def_prop_ro("data", &DLPackImport::data)
      .def_prop_ro("byte_offset", &DLPackImport::byteOffset)
      .def_prop_ro("dtype", &DLPackImport::dtype)
      .def_prop_ro("shape", &DLPackImport::shape)
      .def_prop_ro("strides", &DLPackImport::strides);
  m.def("import_dlpack", &importDLPack, "producer"_a, "stream"_a.none());
}
