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

#include "max/c/gguf.h"
#include "GraphCompiler/Support/GGUF/GGUF.h"
#include "Kernels/max_license/WeightsRegistry.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "max/max_license/API/c/detail/helpers.h"
#include "max/max_license/API/c/detail/types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include <filesystem>
#include <memory>

using namespace M;

namespace {

// Looks up an entry by name, setting `status` to OK on success or to a
// not-found error on failure.
const M_GGUF::Entry *findEntry(const M_GGUF *gguf, const char *name,
                               M_Status *status) {
  if (!gguf) {
    M::Detail::setError(status,
                        M::Error::getStaticString("gguf must be provided"));
    return nullptr;
  }
  if (!name) {
    M::Detail::setError(
        status, M::Error::getStaticString("tensor name must not be null"));
    return nullptr;
  }
  auto it = gguf->entries.find(name);
  if (it == gguf->entries.end()) {
    M::Detail::setError(status, M::Error(Twine("no tensor named '") + name +
                                         "' in the loaded gguf"));
    return nullptr;
  }
  M::Detail::setOK(status);
  return &it->second;
}

} // namespace

MODULAR_API_EXPORT M_GGUF *M_loadGGUF(const char **paths, size_t numPaths,
                                      M_Device *device, M_Status *status) {
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

  auto gguf = std::make_unique<M_GGUF>();

  for (size_t i = 0; i < numPaths; ++i) {
    ErrorOr<GGUF> fileOr =
        GGUF::get(std::filesystem::path(paths[i]), device->context);
    if (fileOr.isError()) {
      M::Detail::setError(status, fileOr.takeError());
      return nullptr;
    }

    GGUF file = fileOr.takeValue();
    gguf->version = file.getVersion();
    gguf->alignment = file.getAlignment();
    for (const auto &[name, info] : file.getTensorInfo()) {
      auto [it, inserted] = gguf->entries.try_emplace(
          name, M_GGUF::Entry{file.getTensorMemory(info), info});
      if (!inserted) {
        M::Detail::setError(status,
                            M::Error(Twine("duplicate tensor name '") + name +
                                     "' across the provided GGUF files"));
        return nullptr;
      }
      gguf->names.push_back(name);
    }
  }

  M::Detail::setOK(status);
  return gguf.release();
}

MODULAR_API_EXPORT size_t M_getGGUFCount(const M_GGUF *gguf) {
  return gguf ? gguf->names.size() : 0;
}

MODULAR_API_EXPORT const char *M_getGGUFName(const M_GGUF *gguf, size_t index) {
  if (!gguf || index >= gguf->names.size())
    return nullptr;
  return gguf->names[index].c_str();
}

MODULAR_API_EXPORT const void *
M_getGGUFData(const M_GGUF *gguf, const char *name, M_Status *status) {
  if (!status)
    return nullptr;
  const M_GGUF::Entry *entry = findEntry(gguf, name, status);
  if (!entry)
    return nullptr;
  return entry->memory->ptr();
}

MODULAR_API_EXPORT size_t M_getGGUFNumBytes(const M_GGUF *gguf,
                                            const char *name,
                                            M_Status *status) {
  if (!status)
    return 0;
  const M_GGUF::Entry *entry = findEntry(gguf, name, status);
  if (!entry)
    return 0;
  return entry->info.getSizeInBytes();
}

MODULAR_API_EXPORT M_Dtype M_getGGUFDtype(const M_GGUF *gguf, const char *name,
                                          M_Status *status) {
  if (!status)
    return M_UNKNOWN;
  const M_GGUF::Entry *entry = findEntry(gguf, name, status);
  if (!entry)
    return M_UNKNOWN;

  ErrorOr<DType> dtypeOr = entry->info.getDType();
  if (dtypeOr.isError()) {
    M::Detail::setError(status, dtypeOr.takeError());
    return M_UNKNOWN;
  }
  return M::Detail::mapDType(*dtypeOr);
}

MODULAR_API_EXPORT size_t M_getGGUFRank(const M_GGUF *gguf, const char *name,
                                        M_Status *status) {
  if (!status)
    return 0;
  const M_GGUF::Entry *entry = findEntry(gguf, name, status);
  if (!entry)
    return 0;
  ErrorOr<SmallVector<size_t>> shapeOr = entry->info.getByteShape();
  if (shapeOr.isError()) {
    M::Detail::setError(status, shapeOr.takeError());
    return 0;
  }
  return shapeOr->size();
}

MODULAR_API_EXPORT void M_getGGUFShape(const M_GGUF *gguf, const char *name,
                                       int64_t *shapeOut, M_Status *status) {
  if (!status)
    return;
  const M_GGUF::Entry *entry = findEntry(gguf, name, status);
  if (!entry)
    return;
  if (!shapeOut) {
    M::Detail::setError(status,
                        M::Error::getStaticString("shapeOut must not be null"));
    return;
  }
  ErrorOr<SmallVector<size_t>> shapeOr = entry->info.getByteShape();
  if (shapeOr.isError()) {
    M::Detail::setError(status, shapeOr.takeError());
    return;
  }
  for (auto [i, dim] : llvm::enumerate(*shapeOr))
    shapeOut[i] = static_cast<int64_t>(dim);
}

MODULAR_API_EXPORT const char *M_getGGUFQuantization(const M_GGUF *gguf,
                                                     const char *name) {
  if (!gguf || !name)
    return nullptr;
  auto it = gguf->entries.find(name);
  if (it == gguf->entries.end())
    return nullptr;

  auto blockSize = it->second.info.getQuantizationBlockSize();
  if (!blockSize)
    return nullptr;

  // The scheme names are string literals with static lifetime, so handing one
  // out borrows nothing from the object.
  switch (it->second.info.tensorType) {
  case GGUF::Q4_0:
    return "Q4_0";
  case GGUF::Q4_K:
    return "Q4_K";
  case GGUF::Q5_K:
    return "Q5_K";
  case GGUF::Q6_K:
    return "Q6_K";
  default:
    return nullptr;
  }
}

MODULAR_API_EXPORT uint32_t M_getGGUFVersion(const M_GGUF *gguf) {
  return gguf ? gguf->version : 0;
}

MODULAR_API_EXPORT size_t M_getGGUFAlignment(const M_GGUF *gguf) {
  return gguf ? gguf->alignment : 0;
}

MODULAR_API_EXPORT M_WeightsRegistry *
M_newWeightsRegistryFromGGUF(const M_GGUF *gguf, M_Status *status) {
  if (!status)
    return nullptr;

  if (!gguf) {
    M::Detail::setError(status,
                        M::Error::getStaticString("gguf must be provided"));
    return nullptr;
  }

  llvm::SmallVector<const char *> names;
  llvm::SmallVector<const std::byte *> data;
  names.reserve(gguf->names.size());
  data.reserve(gguf->names.size());

  for (const std::string &name : gguf->names) {
    const M_GGUF::Entry &entry = gguf->entries.find(name)->second;
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

MODULAR_API_EXPORT void M_freeGGUF(M_GGUF *gguf) { delete gguf; }

MODULAR_API_EXPORT M_GGUFWriter *M_newGGUFWriter(size_t alignment,
                                                 M_Status *status) {
  if (!status)
    return nullptr;

  if (alignment == 0)
    alignment = GGUFWriter::kDefaultAlignment;
  if ((alignment & (alignment - 1)) != 0) {
    M::Detail::setError(status,
                        M::Error(Twine("alignment ") + Twine(alignment) +
                                 " is not a power of two"));
    return nullptr;
  }

  M::Detail::setOK(status);
  return new M_GGUFWriter(alignment);
}

MODULAR_API_EXPORT void M_setGGUFMetadataString(M_GGUFWriter *writer,
                                                const char *key,
                                                const char *value) {
  if (!writer or !key or !value)
    return;
  writer->writer.setMetadataString(key, value);
}

MODULAR_API_EXPORT void M_setGGUFMetadataBool(M_GGUFWriter *writer,
                                              const char *key, bool value) {
  if (!writer or !key)
    return;
  writer->writer.setMetadataBool(key, value);
}

MODULAR_API_EXPORT void M_setGGUFMetadataInt(M_GGUFWriter *writer,
                                             const char *key, int64_t value) {
  if (!writer or !key)
    return;
  writer->writer.setMetadataInt(key, value);
}

MODULAR_API_EXPORT void M_setGGUFMetadataUInt(M_GGUFWriter *writer,
                                              const char *key, uint64_t value) {
  if (!writer or !key)
    return;
  writer->writer.setMetadataUInt(key, value);
}

MODULAR_API_EXPORT void M_setGGUFMetadataFloat(M_GGUFWriter *writer,
                                               const char *key, double value) {
  if (!writer or !key)
    return;
  writer->writer.setMetadataFloat(key, value);
}

MODULAR_API_EXPORT void M_addGGUFTensor(M_GGUFWriter *writer, const char *name,
                                        uint32_t tensorType,
                                        const int64_t *shape, size_t rank,
                                        const void *data, size_t numBytes,
                                        M_Status *status) {
  if (!status)
    return;

  if (!writer) {
    M::Detail::setError(status,
                        M::Error::getStaticString("writer must be provided"));
    return;
  }
  if (!name) {
    M::Detail::setError(
        status, M::Error::getStaticString("tensor name must not be null"));
    return;
  }
  if (rank > 0 and !shape) {
    M::Detail::setError(status, M::Error::getStaticString(
                                    "shape must be provided when rank > 0"));
    return;
  }
  if (numBytes > 0 and !data) {
    M::Detail::setError(status, M::Error::getStaticString(
                                    "data must be provided when numBytes > 0"));
    return;
  }

  llvm::SmallVector<size_t> dims(rank);
  for (auto [i, dim] : llvm::enumerate(ArrayRef<int64_t>(shape, rank))) {
    if (dim < 0) {
      M::Detail::setError(status,
                          M::Error(Twine("tensor '") + name +
                                   "' has a negative dimension " + Twine(dim)));
      return;
    }
    dims[i] = static_cast<size_t>(dim);
  }

  ArrayRef<char> bytes(reinterpret_cast<const char *>(data), numBytes);
  ErrorOrSuccess result =
      writer->writer.addTensor(name, tensorType, dims, bytes);
  if (result.isError()) {
    M::Detail::setError(status, result.takeError());
    return;
  }
  M::Detail::setOK(status);
}

MODULAR_API_EXPORT void M_writeGGUF(const M_GGUFWriter *writer,
                                    const char *path, M_Status *status) {
  if (!status)
    return;

  if (!writer) {
    M::Detail::setError(status,
                        M::Error::getStaticString("writer must be provided"));
    return;
  }
  if (!path) {
    M::Detail::setError(status,
                        M::Error::getStaticString("path must not be null"));
    return;
  }

  ErrorOrSuccess result = writer->writer.write(std::filesystem::path(path));
  if (result.isError()) {
    M::Detail::setError(status, result.takeError());
    return;
  }
  M::Detail::setOK(status);
}

MODULAR_API_EXPORT void M_freeGGUFWriter(M_GGUFWriter *writer) {
  delete writer;
}
