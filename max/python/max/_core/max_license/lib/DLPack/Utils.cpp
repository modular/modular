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

#include "_core/DLPack/Utils.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include <string>

namespace nb = nanobind;

using namespace nanobind::literals;

/// Helper to call the `__dlpack__` hook with the correct stream. The stream ID
/// is null on cpu.
static nb::capsule callDLPackWithStream(const nb::object &arrayOrTensor,
                                        std::optional<intptr_t> streamID) {
  // We don't pass anything for stream on CPU tensors.
  if (!streamID)
    return nb::borrow<nb::capsule>(arrayOrTensor.attr("__dlpack__")());

  // TODO(AIPIPE-58): We need to handle stream synchronization correctly, but
  // the driver doesn't expose streams yet.
  return nb::borrow<nb::capsule>(
      arrayOrTensor.attr("__dlpack__")("stream"_a = nb::int_(*streamID)));
}

M::DLPack::DLManagedTensorWrapper
M::DLPack::callDLPack(const nb::object &arrayOrTensor,
                      std::optional<intptr_t> streamID) {
  nb::capsule capsule = callDLPackWithStream(arrayOrTensor, streamID);

  if (PyObject *capsulePtr = capsule.ptr();
      !PyCapsule_IsValid(capsulePtr, "dltensor")) {
    // We should expect well formed capsules, but capsule name problems are more
    // common, so we fail more gracefully in that case.
    if (!capsulePtr || !capsule.data())
      throw std::runtime_error("malformed capsule in dlpack implementation");
    throw std::runtime_error(
        "internal error: unexpected capsule name returned by __dlpack__:" +
        std::string(capsule.name()));
  }

  // By the dlpack contract, we need to disable the destructor of the capsule.
  // Note that this is not the destructor of the underlying tensor and data.

  if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0)
    throw nb::python_error();
  if (PyCapsule_SetDestructor(capsule.ptr(), nullptr) != 0)
    throw nb::python_error();

  return DLManagedTensorWrapper{static_cast<DLManagedTensor *>(capsule.data())};
}

std::pair<DLDeviceType, ssize_t>
M::DLPack::callDLPackDevice(const nb::object &arrayOrTensor) {
  // The dlpack contract ensures that __dlpack_device__ returns a tuple of two
  // elements: a device type enum as an int, and a device id as an int.
  nb::tuple deviceTuple =
      nb::borrow<nb::tuple>(arrayOrTensor.attr("__dlpack_device__")());
  assert(deviceTuple.size() == 2);
  auto deviceType = (DLDeviceType)nb::cast<int>(deviceTuple[0]);
  return {deviceType, nb::cast<ssize_t>(deviceTuple[1])};
}

// Maps a DLPack `DLDataType` to a `M::DType::Cases`.
M::DType::Cases M::DLPack::dtypeForDLDataType(DLDataType dlDataType) {
  if (dlDataType.lanes != 1)
    throw nb::type_error(
        ("unsupported lanes in DLDataType: " + std::to_string(dlDataType.lanes))
            .c_str());

  uint8_t bits = dlDataType.bits;
  switch (static_cast<DLDataTypeCode>(dlDataType.code)) {
  case DLDataTypeCode::kDLInt:
    if (bits == 8)
      return M::DType::si8;
    if (bits == 16)
      return M::DType::si16;
    if (bits == 32)
      return M::DType::si32;
    if (bits == 64)
      return M::DType::si64;
    throw nb::type_error("unsupported int DLDataType");
  case DLDataTypeCode::kDLUInt:
    if (bits == 8)
      return M::DType::ui8;
    if (bits == 16)
      return M::DType::ui16;
    if (bits == 32)
      return M::DType::ui32;
    if (bits == 64)
      return M::DType::ui64;
    throw nb::type_error("unsupported unsigned DLDataType");
  case DLDataTypeCode::kDLFloat:
    if (bits == 16)
      return M::DType::f16;
    if (bits == 32)
      return M::DType::f32;
    if (bits == 64)
      return M::DType::f64;
    throw nb::type_error("unsupported float DLDataType");
  case DLDataTypeCode::kDLBool:
    return M::DType::kBool;
  case DLDataTypeCode::kDLBfloat:
    return M::DType::bf16;
  case DLDataTypeCode::kDLFloat4_e2m1fn:
    return M::DType::f4e2m1fn;
  case DLDataTypeCode::kDLFloat6_e2m3fn:
    return M::DType::f6e2m3fn;
  case DLDataTypeCode::kDLFloat6_e3m2fn:
    return M::DType::f6e3m2fn;
  case DLDataTypeCode::kDLFloat8_e8m0fnu:
    return M::DType::f8e8m0fnu;
  case DLDataTypeCode::kDLFloat8_e3m4:
    return M::DType::f8e3m4;
  case DLDataTypeCode::kDLFloat8_e4m3fn:
    return M::DType::f8e4m3fn;
  case DLDataTypeCode::kDLFloat8_e4m3fnuz:
    return M::DType::f8e4m3fnuz;
  case DLDataTypeCode::kDLFloat8_e5m2:
    return M::DType::f8e5m2;
  case DLDataTypeCode::kDLFloat8_e5m2fnuz:
    return M::DType::f8e5m2fnuz;
  default:
    break;
  }
  throw nb::type_error(
      ("unsupported DLDataTypeCode: " + std::to_string(dlDataType.code))
          .c_str());
}

static uint8_t dlDataTypeCodeForDType(M::DType dtype) {
  switch (dtype.getValue()) {
  case M::DType::si8:
  case M::DType::si16:
  case M::DType::si32:
  case M::DType::si64:
    return DLDataTypeCode::kDLInt;
  case M::DType::ui8:
  case M::DType::ui16:
  case M::DType::ui32:
  case M::DType::ui64:
    return DLDataTypeCode::kDLUInt;
  case M::DType::f4e2m1fn:
    return DLDataTypeCode::kDLFloat4_e2m1fn;
  case M::DType::f6e2m3fn:
    return DLDataTypeCode::kDLFloat6_e2m3fn;
  case M::DType::f6e3m2fn:
    return DLDataTypeCode::kDLFloat6_e3m2fn;
  case M::DType::f8e8m0fnu:
    return DLDataTypeCode::kDLFloat8_e8m0fnu;
  case M::DType::f8e3m4:
    return DLDataTypeCode::kDLFloat8_e3m4;
  case M::DType::f8e4m3fn:
    return DLDataTypeCode::kDLFloat8_e4m3fn;
  case M::DType::f8e4m3fnuz:
    return DLDataTypeCode::kDLFloat8_e4m3fnuz;
  case M::DType::f8e5m2:
    return DLDataTypeCode::kDLFloat8_e5m2;
  case M::DType::f8e5m2fnuz:
    return DLDataTypeCode::kDLFloat8_e5m2fnuz;
  case M::DType::f16:
  case M::DType::f32:
  case M::DType::f64:
    return DLDataTypeCode::kDLFloat;
  case M::DType::bf16:
    return DLDataTypeCode::kDLBfloat;
  case M::DType::kBool:
    return DLDataTypeCode::kDLBool;
  }
  throw nb::type_error(
      ("unsupported dtype in DLPack conversion: " + dtype.getAsString())
          .c_str());
}

DLDataType M::DLPack::dlDataTypeForDType(M::DType dtype) {
  // Single lane only (no vector types).
  return DLDataType{
      /*code=*/dlDataTypeCodeForDType(dtype),
      /*bits=*/static_cast<uint8_t>(dtype.getWidthInBits()),
      /*lanes=*/1,
  };
}

nb::object M::DLPack::makeDLPackCapsule(
    nb::handle owner, uintptr_t data, int32_t deviceType, int32_t deviceId,
    M::DType::Cases dtype, const std::vector<int64_t> &shape,
    std::optional<std::vector<int64_t>> strides,
    std::optional<std::pair<int64_t, int64_t>> maxVersion) {
  // nanobind's ndarray takes the shape as `size_t`.
  std::vector<size_t> shapeVec(shape.begin(), shape.end());

  DLDataType dlDtype = dlDataTypeForDType(M::DType(dtype));

  // Build a non-owning view over the memory. A NULL strides pointer means
  // row-major, which is what array.py passes for a contiguous array. The
  // ndarray copies shape/strides internally, so the locals above need only
  // outlive this constructor.
  nb::ndarray<nb::array_api> nd{
      /*data=*/reinterpret_cast<void *>(data),
      /*ndim=*/shapeVec.size(),
      /*shape=*/shapeVec.data(),
      /*owner=*/owner,
      /*strides=*/strides ? strides->data() : nullptr,
      /*dtype=*/{dlDtype.code, dlDtype.bits, dlDtype.lanes},
      /*device_type=*/deviceType,
      /*device_id=*/deviceId,
  };

  // Cast with reference_internal and the nanobind ndarray as the owner so the
  // capsule keeps `owner` alive (the reference_internal pattern from
  // https://nanobind.readthedocs.io/en/latest/ndarray.html#array-libraries).
  // Passing `owner` as both the ndarray owner and the cast parent dedups the
  // incref to exactly one; forwarding `max_version` lets nanobind do the
  // versioned/unversioned negotiation.
  nb::object exporter = nb::cast(nd, nb::rv_policy::reference_internal, owner);
  if (maxVersion)
    return exporter.attr("__dlpack__")(
        "max_version"_a =
            nb::make_tuple(maxVersion->first, maxVersion->second));
  return exporter.attr("__dlpack__")();
}
