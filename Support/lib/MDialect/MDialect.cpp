//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.h"
#include "Support/AlignedAlloc.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"

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
      return MemoryHandle(entry, cast<MDialect>(getDialect()));

    // If an entry already exists for the user provided name, tweak the name
    // and re-attempt insertion until we find one that is unique.
    llvm::SmallString<32> nameStorage(key);
    nameStorage.push_back('_');
    size_t nameCounter = 1;
    do {
      Twine(nameCounter++).toVector(nameStorage);

      // Try inserting with the new name.
      if (ResourceEntry *entry = tryInsertion(nameStorage))
        return MemoryHandle(entry, cast<MDialect>(getDialect()));
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
  ResourceEntry *entry =
      resources.modify([key, mem = std::move(mem)](
                           llvm::StringMap<ResourceEntry> &resources) mutable {
        auto it = resources.find(key);
        assert(it != resources.end() && "resource entry was not declared");
        it->second.updateValue(ResourceKind::String, std::move(mem));
        return &it->second;
      });

  // Add the entry to the string table.
  stringTable.get().try_emplace(
      value, MemoryHandle(entry, cast<MDialect>(getDialect())));
}

MemoryHandle DialectResourceManager::addBlobResource(StringRef baseName,
                                                     void *memory, size_t size,
                                                     size_t align) {
  MemoryHandle hdl = declareResource(baseName);
  hdl.getResource()->updateValue(
      ResourceKind::Blob, mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
                              {(char *)memory, size}, align));
  return hdl;
}

MemoryHandle DialectResourceManager::getOrAddStringResource(StringRef value) {
  return stringTable.modify(
      [this, value](llvm::StringMap<MemoryHandle> &table) {
        if (auto it = table.find(value); it != table.end())
          return it->second;
        MemoryHandle hdl = declareResource("static_string");
        updateResourceWithString(hdl.getKey(), value);
        return hdl;
      });
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
// MDialect
//===----------------------------------------------------------------------===//

void MDialect::initialize() {
  registerAttributes();
  registerTypes();
  injectTypeInterfaces();
  injectAttrInterfaces();

  auto &blobMgr = addInterface<DialectResourceManager>();
  addInterface<MOpAsmDialectInterface>(blobMgr);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.cpp.inc"
