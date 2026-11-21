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

#include "GraphCompiler/Support/GGUF/GGUF.h"
#include "GraphCompiler/Support/GGUF/GGUFWriter.h"
#include "_core/Driver/Buffer.h"
#include "_core/Driver/Device.h"
#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/filesystem.h"    // IWYU pragma: keep (type caster)
#include "nanobind/stl/optional.h"      // IWYU pragma: keep (type caster)
#include "nanobind/stl/shared_ptr.h"    // IWYU pragma: keep (type caster)
#include "nanobind/stl/string.h"        // IWYU pragma: keep (type caster)
#include "nanobind/stl/string_view.h"   // IWYU pragma: keep (type caster)
#include "nanobind/stl/unordered_map.h" // IWYU pragma: keep (type caster)
#include "nanobind/stl/vector.h"        // IWYU pragma: keep (type caster)

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

/// Small wrapper around GGUF to allow easy access to the device.
struct GGUFWrapper {
  M::GGUF impl;
  M::Driver::Python::PyDeviceTy device;
};

/// The state behind `max._core.gguf.GGUFWriter`.
struct GGUFWriterWrapper {
  explicit GGUFWriterWrapper(
      size_t alignment = M::GGUFWriter::kDefaultAlignment)
      : impl(alignment) {
    if (alignment == 0 or (alignment & (alignment - 1)) != 0)
      throw nb::value_error(
          ("alignment " + std::to_string(alignment) + " is not a power of two")
              .c_str());
  }

  M::GGUFWriter impl;
};

/// Looks up a tensor, raising `KeyError` when the name is absent.
M::GGUF::TensorInfo lookupTensor(GGUFWrapper &self, std::string_view name) {
  auto infoOr = self.impl.getTensorInfo(name);
  if (infoOr.isError())
    throw nb::key_error(infoOr.getError());
  return infoOr.takeValue();
}

} // namespace

void M::Graph::Python::bindGGUFModule(nb::module_ m) {
  nb::class_<GGUFWrapper>(
      m, "GGUF", "A parser for the GGUF (GPT-Generated Unified Format) format.")
      .def_prop_ro(
          "version", [](GGUFWrapper &self) { return self.impl.getVersion(); },
          "The GGUF format version of the file.")
      .def_prop_ro(
          "alignment",
          [](GGUFWrapper &self) { return self.impl.getAlignment(); },
          "The alignment of the tensor data section, from "
          "``general.alignment``.")
      .def(
          "keys",
          [](GGUFWrapper &self) {
            auto keys = self.impl.getTensorInfo().keys();
            return std::vector<std::string>(keys.begin(), keys.end());
          },
          "Returns the list of tensor keys present.")
      .def(
          "metadata",
          [](GGUFWrapper &self) {
            std::unordered_map<std::string, std::string> out;
            for (const auto &kv : self.impl.getMetadata()) {
              if (auto str = kv.second.getAsString())
                out.emplace(kv.first.str(), str->str());
            }
            return out;
          },
          "Returns the file's string-valued metadata. Numbers, booleans and "
          "arrays are skipped, as are the tokenizer's vocabulary and merges "
          "tables, which are not needed to read tensors.")
      .def(
          "element_shape",
          [](GGUFWrapper &self, std::string_view name) {
            auto info = lookupTensor(self, name);
            return std::vector<size_t>(info.shape.begin(), info.shape.end());
          },
          "name"_a,
          "Returns the logical element shape of a tensor, before any "
          "quantization blocks are taken into account.")
      .def(
          "quantization",
          [](GGUFWrapper &self,
             std::string_view name) -> std::optional<std::string> {
            auto info = lookupTensor(self, name);
            if (!info.isQuantized())
              return std::nullopt;
            return info.getQuantizationName().str();
          },
          "name"_a,
          "Returns the name of the tensor's quantization scheme, such as "
          "``\"Q4_K\"``, or ``None`` when the tensor is not block-quantized.")
      .def(
          "get_buffer",
          [](GGUFWrapper &self, std::string_view name) {
            auto info = lookupTensor(self, name);
            auto dtypeOr = info.getDType();
            if (dtypeOr.isError())
              throw nb::value_error(dtypeOr.getError());
            auto shapeOr = info.getByteShape();
            if (shapeOr.isError())
              throw nb::value_error(shapeOr.getError());

            auto buffer = self.impl.getTensorMemory(info);
            TensorSpec spec{*shapeOr, *dtypeOr};
            auto storage = std::make_shared<M::Driver::Python::BufferStorage>(
                std::move(buffer), std::move(spec));
            return M::Driver::Python::Buffer(std::move(storage), self.device,
                                             nullptr);
          },
          "name"_a, "Returns a buffer with a given key.")
      .def(
          "__enter__", [](GGUFWrapper &self) -> GGUFWrapper & { return self; },
          nb::rv_policy::reference_internal)
      .def(
          "__exit__",
          [](GGUFWrapper &self, nb::handle, nb::handle,
             nb::handle) { /* Do we need to do anything here? */ },
          "exc_type"_a = nb::none(), "exc_value"_a = nb::none(),
          "traceback"_a = nb::none());

  nb::class_<GGUFWriterWrapper>(
      m, "GGUFWriter",
      "A writer for the GGUF (GPT-Generated Unified Format) format.")
      .def(nb::init<size_t>(), "alignment"_a = M::GGUFWriter::kDefaultAlignment,
           R"nbdoc(
        Creates a writer.

        Metadata and tensors are written in the order they are added. Tensor
        data is copied, so it may be released once added.

        Args:
            alignment: The alignment of the tensor data section, in bytes. It
                must be a power of two, and 32 is the format's default.
      )nbdoc")
      .def_prop_ro(
          "alignment",
          [](GGUFWriterWrapper &self) { return self.impl.getAlignment(); },
          "The alignment of the tensor data section.")
      .def(
          "set_string",
          [](GGUFWriterWrapper &self, std::string_view key,
             std::string_view value) {
            self.impl.setMetadataString(StringRef(key), StringRef(value));
          },
          "key"_a, "value"_a,
          "Adds or replaces a string-valued metadata entry.")
      .def(
          "set_int",
          [](GGUFWriterWrapper &self, std::string_view key, int64_t value) {
            self.impl.setMetadataInt(StringRef(key), value);
          },
          "key"_a, "value"_a,
          "Adds or replaces a signed integer metadata entry, written as an "
          "``INT64``.")
      .def(
          "set_uint",
          [](GGUFWriterWrapper &self, std::string_view key, uint64_t value) {
            self.impl.setMetadataUInt(StringRef(key), value);
          },
          "key"_a, "value"_a,
          "Adds or replaces an unsigned integer metadata entry, written as a "
          "``UINT64``.")
      .def(
          "set_float",
          [](GGUFWriterWrapper &self, std::string_view key, double value) {
            self.impl.setMetadataFloat(StringRef(key), value);
          },
          "key"_a, "value"_a,
          "Adds or replaces a floating point metadata entry, written as a "
          "``FLOAT64``.")
      .def(
          "set_bool",
          [](GGUFWriterWrapper &self, std::string_view key, bool value) {
            self.impl.setMetadataBool(StringRef(key), value);
          },
          "key"_a, "value"_a, "Adds or replaces a boolean metadata entry.")
      .def(
          "add_tensor",
          [](GGUFWriterWrapper &self, std::string_view name,
             uint32_t tensorType, std::vector<size_t> shape, nb::bytes data) {
            auto result = self.impl.addTensor(
                name, tensorType, llvm::ArrayRef<size_t>(shape),
                llvm::ArrayRef<char>(data.c_str(), data.size()));
            if (result.isError())
              throw nb::value_error(result.getError());
          },
          "name"_a, "tensor_type"_a, "shape"_a, "data"_a,
          R"nbdoc(
        Adds a tensor.

        Args:
            name: The tensor name.
            tensor_type: The GGML type id of `data`; 0 is ``F32``, 30 is
                ``BF16``, 12 is ``Q4_K``.
            shape: The logical shape, widest dimension first.
            data: The tensor data, exactly the bytes the shape and type need.
                For block-quantized types the last dimension counts elements,
                so keep it a whole number of quantization blocks.
      )nbdoc")
      .def(
          "write",
          [](GGUFWriterWrapper &self, std::filesystem::path filepath) {
            auto result = self.impl.write(filepath);
            if (result.isError())
              throw nb::value_error(result.getError());
          },
          "filepath"_a, "Writes the file, replacing it if it exists.");

  m.def(
      "gguf_open",
      [](std::filesystem::path filepath,
         std::optional<Driver::Python::PyDeviceTy> maybeDevice) {
        Driver::Python::PyDeviceTy device =
            maybeDevice ? std::move(maybeDevice.value())
                        : Driver::Python::getCpuDevice(0);
        auto result = GGUF::get(filepath, device->ctx);
        if (result.isError())
          //  TODO: more specific error
          throw nb::value_error(result.getError());
        return GGUFWrapper{std::move(result.get()), std::move(device)};
      },
      "filepath"_a, "device"_a = nb::none(), R"nbdoc(
        Loads and parses a GGUF file from the given path onto the given device.

        Defaults to loading on the CPU.

        .. note::

           Currently only implemented for CPUs.
      )nbdoc");
}
