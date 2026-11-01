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
#include "Support/ML/DType.h"
#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

void M::Graph::Python::bindDTypeModule(nb::module_ m) {
  m.doc() = R"nbdoc(
        Defines the data types of MAX tensors.
    )nbdoc";

  nb::enum_<M::DType::Cases>(m, "DType", "The tensor data type.")
      .value("bool", DType::kBool,
             "Boolean data type. Stores ``True`` or ``False`` values.")
      .value("int8", DType::si8, "8-bit signed integer, range -128 to 127.")
      .value("int16", DType::si16,
             "16-bit signed integer, range -32,768 to 32,767.")
      .value("int32", DType::si32,
             "32-bit signed integer, range -2,147,483,648 to 2,147,483,647.")
      .value("int64", DType::si64,
             "64-bit signed integer, range -9,223,372,036,854,775,808 to "
             "9,223,372,036,854,775,807.")
      .value("uint8", DType::ui8, "8-bit unsigned integer, range 0 to 255.")
      .value("uint16", DType::ui16,
             "16-bit unsigned integer, range 0 to 65,535.")
      .value("uint32", DType::ui32,
             "32-bit unsigned integer, range 0 to 4,294,967,295.")
      .value("uint64", DType::ui64,
             "64-bit unsigned integer, range 0 to "
             "18,446,744,073,709,551,615.")
      .value("float16", DType::f16,
             "16-bit IEEE 754 half-precision floating-point. 1 sign bit, 5 "
             "exponent bits, 10 mantissa bits.")
      .value("float32", DType::f32,
             "32-bit IEEE 754 single-precision floating-point. 1 sign bit, 8 "
             "exponent bits, 23 mantissa bits.")
      .value("float64", DType::f64,
             "64-bit IEEE 754 double-precision floating-point. 1 sign bit, 11 "
             "exponent bits, 52 mantissa bits.")
      .value("bfloat16", DType::bf16,
             "16-bit bfloat16 (Brain Float) format. 1 sign bit, 8 exponent "
             "bits, 7 mantissa bits.")
      .value("float4_e2m1fn", DType::f4e2m1fn,
             "4-bit floating-point with 2 exponent bits and 1 mantissa bit, "
             "finite values only.")
      .value("float6_e2m3fn", DType::f6e2m3fn,
             "6-bit floating-point with 2 exponent bits and 3 mantissa bits, "
             "finite values only.")
      .value("float6_e3m2fn", DType::f6e3m2fn,
             "6-bit floating-point with 3 exponent bits and 2 mantissa bits, "
             "finite values only.")
      .value("float8_e8m0fnu", DType::f8e8m0fnu,
             "8-bit floating-point with 8 exponent bits and 0 mantissa bits, "
             "finite values only.")
      .value("float8_e4m3fn", DType::f8e4m3fn,
             "8-bit floating-point with 4 exponent bits and 3 mantissa bits, "
             "finite values only.")
      .value("float8_e4m3fnuz", DType::f8e4m3fnuz,
             "8-bit floating-point with 4 exponent bits and 3 mantissa bits, "
             "finite values only, no negative zero.")
      .value("float8_e5m2", DType::f8e5m2,
             "8-bit floating-point with 5 exponent bits and 2 mantissa bits.")
      .value("float8_e5m2fnuz", DType::f8e5m2fnuz,
             "8-bit floating-point with 5 exponent bits and 2 mantissa bits, "
             "finite values only, no negative zero.")
      .def_prop_ro(
          "align",
          [](M::DType::Cases dtype) { return M::DType(dtype).getAlignment(); },
          R"nbdoc(
          The alignment requirement of the data type in bytes.

          The alignment specifies the memory boundary that values of this data type
          must be aligned to for optimal performance and correctness.
          )nbdoc")
      .def_prop_ro(
          "size_in_bits",
          [](M::DType::Cases dtype) {
            return M::DType(dtype).getWidthInBits();
          },
          R"nbdoc(
          The size of the data type in bits.

          This indicates how many bits are required to store a single value
          of this data type in memory.
          )nbdoc")
      .def_prop_ro(
          "size_in_bytes",
          [](M::DType::Cases dtype) {
            return M::DType(dtype).getSizeInBytes();
          },
          R"nbdoc(
          The size of the data type in bytes.

          This indicates how many bytes are required to store a single value
          of this data type in memory.
          )nbdoc")
      .def(
          "is_integral",
          [](M::DType::Cases dtype) { return M::DType(dtype).isInt(); },
          R"nbdoc(
          Checks if the data type is an integer type.
          )nbdoc")
      .def(
          "is_unsigned_integral",
          [](M::DType::Cases dtype) { return M::DType(dtype).isUInt(); },
          R"nbdoc(
          Checks if the data type is an unsigned integer type.
          )nbdoc")
      .def(
          "is_signed_integral",
          [](M::DType::Cases dtype) { return M::DType(dtype).isSInt(); },
          R"nbdoc(
          Checks if the data type is a signed integer type.
          )nbdoc")
      .def(
          "is_float",
          [](M::DType::Cases dtype) { return M::DType(dtype).isFloat(); },
          R"nbdoc(
          Checks if the data type is a floating-point type.
          )nbdoc")
      .def(
          "is_float8",
          [](M::DType::Cases dtype) {
            M::DType mdtype(dtype);
            return mdtype.isFloat8();
          },
          R"nbdoc(
          Checks if the data type is an 8-bit floating-point type.
          )nbdoc")
      .def(
          "is_half",
          [](M::DType::Cases dtype) {
            auto mdtype = M::DType(dtype);
            return mdtype.isFloat() && mdtype.getSizeInBytes() == 2;
          },
          R"nbdoc(
          Checks if the data type is a half-precision floating-point type.
          )nbdoc")
      // The definitions of the following functions will be replaced in Python
      .def(
          "to_numpy", [](M::DType::Cases dtype) {}, R"nbdoc(
        Converts this :class:`DType` to the corresponding NumPy dtype.

        Returns:
            The corresponding NumPy dtype object.

        Raises:
            ValueError: If the dtype is not supported.
          )nbdoc",
          nb::sig("def to_numpy(self) -> numpy.dtype"))
      .def_static(
          "from_numpy", []() {}, R"nbdoc(
        Converts a NumPy dtype to the corresponding :class:`DType`.

        Args:
            dtype: The NumPy dtype to convert.

        Returns:
            The corresponding :class:`DType` enum value.

        Raises:
            ValueError: If the input dtype is not supported.
          )nbdoc")
      .def_prop_ro(
          "_mlir", [](M::DType::Cases dtype) {},
          nb::sig("def _mlir(self) -> str"));
}
