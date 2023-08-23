//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.h"
#include "Support/AlignedAlloc.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/BLAKE3.h"

using namespace M;

//===----------------------------------------------------------------------===//
// DialectResourceManager
//===----------------------------------------------------------------------===//

MemoryHandle DialectResourceManager::declareResource(StringRef key) {
  auto modifyFn = [this, key](llvm::StringMap<ResourceEntry> &resources) {
    // Functor used to attempt insertion with a given name.
    auto tryInsertion = [&](StringRef name) -> ResourceEntry * {
      auto it = resources.try_emplace(name, ResourceEntry());
      if (it.second) {
        it.first->getValue().key = it.first->getKey();
        return &it.first->second;
      }
      return nullptr;
    };

    // Try inserting with the name provided by the user.
    if (ResourceEntry *entry = tryInsertion(key))
      return MemoryHandle(entry, static_cast<MDialect *>(getDialect()));

    // If an entry already exists for the user provided name, tweak the name
    // and re-attempt insertion until we find one that is unique.
    llvm::SmallString<32> nameStorage(key);
    nameStorage.push_back('_');
    size_t nameCounter = 1;
    do {
      Twine(nameCounter++).toVector(nameStorage);

      // Try inserting with the new name.
      if (ResourceEntry *entry = tryInsertion(nameStorage))
        return MemoryHandle(entry, static_cast<MDialect *>(getDialect()));
      nameStorage.resize(key.size() + 1);
    } while (true);
  };
  return resources.modify(std::move(modifyFn));
}

void DialectResourceManager::updateResourceWithBlob(
    StringRef key, mlir::AsmResourceBlob &&newBlob) {
  resources.modify([key, blob = std::move(newBlob)](
                       llvm::StringMap<ResourceEntry> &resources) mutable {
    auto it = resources.find(key);
    assert(it != resources.end() && "resource entry was not declared");
    it->second.updateValue(ResourceKind::Blob, std::move(blob));
  });
}

void DialectResourceManager::updateResourceWithString(StringRef key,
                                                      StringRef value) {
  mlir::AsmResourceBlob mem =
      mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
          {value.data(), value.size()}, /*align=*/16);

  // Update the resource value.
  resources.modify([key, mem = std::move(mem)](
                       llvm::StringMap<ResourceEntry> &resources) mutable {
    auto it = resources.find(key);
    assert(it != resources.end() && "resource entry was not declared");
    it->second.updateValue(ResourceKind::String, std::move(mem));
    return &it->second;
  });
}

MemoryHandle DialectResourceManager::getOrAddBlobResource(void *memory,
                                                          size_t size,
                                                          size_t align) {
  return getOrAddResource({(char *)memory, size}, align, ResourceKind::Blob);
}

MemoryHandle DialectResourceManager::getOrAddStringResource(StringRef value) {
  return getOrAddResource({value.data(), value.size()}, /*align=*/16,
                          ResourceKind::String);
}

MemoryHandle DialectResourceManager::getOrAddResource(ArrayRef<char> data,
                                                      size_t align,
                                                      ResourceKind kind) {
  // Pray to Ranald that there be no collisions!
  auto hash = llvm::BLAKE3::hash({(const uint8_t *)data.data(), data.size()});
  std::string key =
      (kind == ResourceKind::String ? "static_string_" : "memory_blob_") +
      llvm::toHex(hash, /*LowerCase=*/true);
  ResourceEntry *entry =
      resources.modify([&](llvm::StringMap<ResourceEntry> &entries) {
        auto it = entries.try_emplace(key, ResourceEntry());
        if (it.second) {
          it.first->getValue().key = it.first->getKey();
          it.first->getValue().updateValue(
              kind,
              mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(data, align));
        }
        return &it.first->second;
      });
  return MemoryHandle(entry, static_cast<MDialect *>(getDialect()));
}

//===----------------------------------------------------------------------===//
// MemoryHandle
//===----------------------------------------------------------------------===//

DialectResourceManager &MemoryHandle::getManagerInterface(MLIRContext *ctx) {
  auto *dialect = ctx->getOrLoadDialect<MDialect>();
  assert(dialect && "MDialect is not registered");
  return *dialect->getRegisteredInterface<DialectResourceManager>();
}

//===----------------------------------------------------------------------===//
// MOpAsmDialectInterface
//===----------------------------------------------------------------------===//

namespace {
class MOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
public:
  MOpAsmDialectInterface(Dialect *dialect, DialectResourceManager &blobMgr)
      : OpAsmDialectInterface(dialect), blobMgr(blobMgr) {}

  std::string
  getResourceKey(const mlir::AsmDialectResourceHandle &handle) const override {
    return cast<MemoryHandle>(handle).getKey().str();
  }
  FailureOr<mlir::AsmDialectResourceHandle>
  declareResource(StringRef key) const final {
    return blobMgr.declareResource(key);
  }

  /// Parse a dialect resource. It may be either a string or a blob. Both are
  /// passed to the dialect resource manager as blobs.
  LogicalResult parseResource(mlir::AsmParsedResourceEntry &entry) const final {
    if (entry.getKind() == mlir::AsmResourceEntryKind::String) {
      FailureOr<std::string> value = entry.parseAsString();
      if (failed(value))
        return failure();
      blobMgr.updateResourceWithString(entry.getKey(), std::move(*value));
    } else {
      FailureOr<mlir::AsmResourceBlob> value = entry.parseAsBlob();
      if (failed(value))
        return failure();
      blobMgr.updateResourceWithBlob(entry.getKey(), std::move(*value));
    }
    return success();
  }

  /// Build the dialect resources into the provider, dispatching on whether each
  /// resource blob is a string kind.
  void buildResources(Operation *op,
                      const llvm::SetVector<mlir::AsmDialectResourceHandle>
                          &referencedResources,
                      mlir::AsmResourceBuilder &provider) const final {
    for (const mlir::AsmDialectResourceHandle &handle : referencedResources) {
      if (const auto *dialectHandle = dyn_cast<MemoryHandle>(&handle)) {
        if (const mlir::AsmResourceBlob *blob = dialectHandle->getBlob()) {
          StringRef key = dialectHandle->getKey();
          if (dialectHandle->getResource()->getKind() ==
              DialectResourceManager::ResourceKind::String) {
            ArrayRef<char> data = blob->getData();
            provider.buildString(key, {data.data(), data.size()});
          } else {
            provider.buildBlob(key, *blob);
          }
        }
      }
    }
  }

private:
  /// The blob manager.
  DialectResourceManager &blobMgr;
};
} // namespace

//===----------------------------------------------------------------------===//
// MDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;

static LogicalResult readMemoryBlobs(DialectBytecodeReader &reader,
                                     SmallVectorImpl<MemoryBlob> &blobs) {
  uint64_t size;
  if (failed(reader.readVarInt(size)))
    return failure();
  blobs.reserve(size);

  auto readPointerRegion = [&](MemoryBlob::PointerRegion &region) {
    int64_t offset, blobIndex, blobOffset;
    if (failed(reader.readSignedVarInt(offset)) ||
        failed(reader.readSignedVarInt(blobIndex)) ||
        failed(reader.readSignedVarInt(blobOffset)))
      return failure();
    return LogicalResult::success();
  };

  for (unsigned i = 0; i < size; ++i) {
    FailureOr<MemoryHandle> hdl = reader.readResourceHandle<MemoryHandle>();
    uint64_t kind;
    if (failed(hdl) || failed(reader.readVarInt(kind)))
      return failure();
    SmallVector<MemoryBlob::PointerRegion> regions;
    if (failed(reader.readList(regions, readPointerRegion)))
      return failure();
    blobs.emplace_back(*hdl, static_cast<MemoryKind>(kind), std::move(regions));
  }

  return success();
}

static void writeMemoryBlobs(DialectBytecodeWriter &writer,
                             ArrayRef<MemoryBlob> blobs) {
  writer.writeVarInt(blobs.size());

  auto writePointerRegion = [&](const MemoryBlob::PointerRegion &region) {
    writer.writeSignedVarInt(region.offset);
    writer.writeSignedVarInt(region.blobIndex);
    writer.writeSignedVarInt(region.blobOffset);
  };

  for (const MemoryBlob &blob : blobs) {
    writer.writeResourceHandle(blob.getHandle());
    writer.writeVarInt(static_cast<uint64_t>(blob.getKind()));
    writer.writeList(blob.getPointerRegions(), writePointerRegion);
  }
}

static LogicalResult parseTriple(DialectBytecodeReader &reader,
                                 llvm::Triple &triple) {
  StringRef tripleStr;
  if (failed(reader.readString(tripleStr)))
    return failure();
  triple = llvm::Triple(tripleStr);
  return success();
}

static void printTriple(MLIRContext *ctx, DialectBytecodeWriter &writer,
                        const llvm::Triple &triple) {
  writer.writeOwnedString(StringAttr::get(ctx, triple.str()));
}

static LogicalResult parseDataLayout(DialectBytecodeReader &reader,
                                     DataLayout &dl) {
  StringRef dlStr;
  if (failed(reader.readString(dlStr)))
    return failure();
  ErrorOr<DataLayout> dlOr = DataLayout::parse(dlStr);
  if (dlOr.isError())
    return reader.emitError(dlOr.getError());
  dl = dlOr.takeValue();
  return success();
}

static void printDataLayout(MLIRContext *ctx, DialectBytecodeWriter &writer,
                            const DataLayout &dl) {
  writer.writeOwnedString(dl.toString());
}

namespace {
#include "Support/MDialect/MDialectBytecode.cpp.inc"

struct MDialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  MDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// MDialect
//===----------------------------------------------------------------------===//

void MDialect::initialize() {
  registerAttributes();
  registerTypes();
  injectTypeInterfaces();
  injectAttrInterfaces();

  auto &blobMgr = addInterface<DialectResourceManager>();
  addInterface<MOpAsmDialectInterface>(blobMgr);
  addInterface<MDialectBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.cpp.inc"
