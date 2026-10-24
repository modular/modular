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

#include "GraphCompiler/Support/SafeTensor/SafeTensor.h"
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
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

/// Small wrapper around SafeTensor to allow easy access to the device.
/// This can probably be removed.
struct SafeTensorWrapper {
  M::SafeTensor impl;
  M::Driver::Python::PyDeviceTy device;
};

void M::Graph::Python::bindSafetensorsModule(nb::module_ m) {
  nb::class_<SafeTensorWrapper>(
      m, "SafeTensor", "A parser for the Hugging Face SafeTensors format.")
      .def(
          "keys",
          [](SafeTensorWrapper &self) {
            auto keys = self.impl.getTensorInfo().keys();
            return std::vector<std::string>(keys.begin(), keys.end());
          },
          "Returns the list of tensor keys present.")
      .def(
          "metadata",
          [](SafeTensorWrapper &self) {
            std::unordered_map<std::string, std::string> out;
            for (const auto &kv : self.impl.getMetadata()) {
              if (auto str = kv.second.getAsString())
                out.emplace(kv.first.str(), str->str());
            }
            return out;
          },
          "Returns the file-level ``__metadata__`` map from the SafeTensors "
          "header. Only string-valued entries are included; non-string values "
          "(which the Hugging Face spec forbids) are skipped. Returns an empty "
          "dict when the header has no ``__metadata__`` section.")
      .def(
          "get_buffer",
          [](SafeTensorWrapper &self, std::string_view name) {
            const auto &allInfo = self.impl.getTensorInfo();
            // minor todo: avoid the intermediate std::string. shouldn't
            // c++20 handle this for us? must be something with DenseMap.
            auto info = allInfo.find(std::string(name));
            if (info == allInfo.end())
              throw nb::index_error(/*TODO*/);
            auto buffer = self.impl.getTensorMemory(info->second);
            TensorSpec spec{info->second.shape, info->second.dtype};
            auto storage = std::make_shared<M::Driver::Python::BufferStorage>(
                std::move(buffer), std::move(spec));
            return M::Driver::Python::Buffer(std::move(storage), self.device,
                                             nullptr);
          },
          "name"_a, "Returns a buffer with a given key.")
      .def(
          "__enter__",
          [](SafeTensorWrapper &self) -> SafeTensorWrapper & { return self; },
          nb::rv_policy::reference_internal)
      .def(
          "__exit__",
          [](SafeTensorWrapper &self, nb::handle, nb::handle,
             nb::handle) { /* Do we need to do anything here? */ },
          "exc_type"_a = nb::none(), "exc_value"_a = nb::none(),
          "traceback"_a = nb::none());

  m.def(
      "safe_open",
      [](std::filesystem::path filepath,
         std::optional<Driver::Python::PyDeviceTy> maybeDevice) {
        Driver::Python::PyDeviceTy device =
            maybeDevice ? std::move(maybeDevice.value())
                        : Driver::Python::getCpuDevice(0);
        auto result = SafeTensor::get(filepath, device->ctx);
        if (result.isError())
          //  TODO: more specific error
          throw nb::value_error(result.getError());
        return SafeTensorWrapper{std::move(result.get()), std::move(device)};
      },
      "filepath"_a, "device"_a = nb::none(), R"nbdoc(
        Loads and parses a SafeTensor file from the given path onto the given device.

        Defaults to loading on the CPU.

        .. note::

           Currently only implemented for CPUs.
      )nbdoc");
}
