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

#include "max/c/safetensors.h"
#include "GraphCompiler/Support/SafeTensor/SafeTensor.h"
#include "Kernels/max_license/WeightsRegistry.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "max/max_license/API/c/detail/helpers.h"
#include "max/max_license/API/c/detail/types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include <filesystem>
#include <memory>

using namespace M;

namespace {

// Looks up an entry by name, setting `status` to OK on success or to a
// not-found error on failure.
const M_Safetensors::Entry *findEntry(const M_Safetensors *safetensors,
                                      const char *name, M_Status *status) {
  if (!name) {
    M::Detail::setError(
        status, M::Error::getStaticString("tensor name must not be null"));
    return nullptr;
  }
  auto it = safetensors->entries.find(name);
  if (it == safetensors->entries.end()) {
    M::Detail::setError(status, M::Error(Twine("no tensor named '") + name +
                                         "' in the loaded safetensors"));
    return nullptr;
  }
  M::Detail::setOK(status);
  return &it->second;
}

} // namespace

MODULAR_API_EXPORT M_Safetensors *M_loadSafetensors(const char **paths,
                                                    size_t numPaths,
                                                    M_Device *device,
                                                    M_Status *status) {
  if (!status)
    return nullptr;

  if (!device) {
    M::Detail::setError(status,
                        M::Error::getStaticString("device must be provided"));
    return nullptr;
  }

  if (numPaths > 0 && !paths) {
    M::Detail::setError(
        status,
        M::Error::getStaticString("paths must be provided when numPaths > 0"));
    return nullptr;
  }

  auto safetensors = std::make_unique<M_Safetensors>();

  for (size_t i = 0; i < numPaths; ++i) {
    ErrorOr<SafeTensor> fileOr =
        SafeTensor::get(std::filesystem::path(paths[i]), device->context);
    if (fileOr.isError()) {
      M::Detail::setError(status, fileOr.takeError());
      return nullptr;
    }

    SafeTensor file = fileOr.takeValue();
    for (const auto &[name, info] : file.getTensorInfo()) {
      auto [it, inserted] = safetensors->entries.try_emplace(
          name, M_Safetensors::Entry{file.getTensorMemory(info), info});
      if (!inserted) {
        M::Detail::setError(
            status, M::Error(Twine("duplicate tensor name '") + name +
                             "' across the provided Safetensors files"));
        return nullptr;
      }
      safetensors->names.push_back(name);
    }
  }

  M::Detail::setOK(status);
  return safetensors.release();
}

MODULAR_API_EXPORT size_t
M_getSafetensorCount(const M_Safetensors *safetensors) {
  return safetensors ? safetensors->names.size() : 0;
}

MODULAR_API_EXPORT const char *
M_getSafetensorName(const M_Safetensors *safetensors, size_t index) {
  if (!safetensors || index >= safetensors->names.size())
    return nullptr;
  return safetensors->names[index].c_str();
}

MODULAR_API_EXPORT const void *
M_getSafetensorData(const M_Safetensors *safetensors, const char *name,
                    M_Status *status) {
  if (!status)
    return nullptr;
  const M_Safetensors::Entry *entry = findEntry(safetensors, name, status);
  if (!entry)
    return nullptr;
  return entry->memory->ptr();
}

MODULAR_API_EXPORT size_t M_getSafetensorNumBytes(
    const M_Safetensors *safetensors, const char *name, M_Status *status) {
  if (!status)
    return 0;
  const M_Safetensors::Entry *entry = findEntry(safetensors, name, status);
  if (!entry)
    return 0;
  return entry->info.getSizeInBytes();
}

MODULAR_API_EXPORT M_Dtype M_getSafetensorDtype(
    const M_Safetensors *safetensors, const char *name, M_Status *status) {
  if (!status)
    return M_UNKNOWN;
  const M_Safetensors::Entry *entry = findEntry(safetensors, name, status);
  if (!entry)
    return M_UNKNOWN;
  return M::Detail::mapDType(entry->info.dtype);
}

MODULAR_API_EXPORT size_t M_getSafetensorRank(const M_Safetensors *safetensors,
                                              const char *name,
                                              M_Status *status) {
  if (!status)
    return 0;
  const M_Safetensors::Entry *entry = findEntry(safetensors, name, status);
  if (!entry)
    return 0;
  return entry->info.shape.size();
}

MODULAR_API_EXPORT void M_getSafetensorShape(const M_Safetensors *safetensors,
                                             const char *name,
                                             int64_t *shapeOut,
                                             M_Status *status) {
  if (!status)
    return;
  const M_Safetensors::Entry *entry = findEntry(safetensors, name, status);
  if (!entry)
    return;
  if (!shapeOut) {
    M::Detail::setError(status,
                        M::Error::getStaticString("shapeOut must not be null"));
    return;
  }
  for (auto [i, dim] : llvm::enumerate(entry->info.shape))
    shapeOut[i] = static_cast<int64_t>(dim);
}

MODULAR_API_EXPORT M_WeightsRegistry *
M_newWeightsRegistryFromSafetensors(const M_Safetensors *safetensors,
                                    M_Status *status) {
  if (!status)
    return nullptr;

  if (!safetensors) {
    M::Detail::setError(
        status, M::Error::getStaticString("safetensors must be provided"));
    return nullptr;
  }

  llvm::SmallVector<const char *> names;
  llvm::SmallVector<const std::byte *> data;
  names.reserve(safetensors->names.size());
  data.reserve(safetensors->names.size());

  for (const std::string &name : safetensors->names) {
    const M_Safetensors::Entry &entry = safetensors->entries.find(name)->second;
    names.push_back(name.c_str());
    data.push_back(reinterpret_cast<const std::byte *>(entry.memory->ptr()));
  }

  ErrorOr<WeightsRegistryRef> weightsOr = WeightsRegistry::create(names, data);
  if (weightsOr.isError()) {
    M::Detail::setError(status, weightsOr.takeError());
    return nullptr;
  }

  M::Detail::setOK(status);
  return new M_WeightsRegistry{weightsOr.takeValue()};
}

MODULAR_API_EXPORT void M_freeSafetensors(M_Safetensors *safetensors) {
  delete safetensors;
}
