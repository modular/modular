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

using namespace M;

//===----------------------------------------------------------------------===//
// MemoryBlob
//===----------------------------------------------------------------------===//

DialectResourceManager &MemoryHandle::getManagerInterface(MLIRContext *ctx) {
  auto *dialect = ctx->getOrLoadDialect<MDialect>();
  assert(dialect && "MDialect is not registered");
  return *dialect->getRegisteredInterface<DialectResourceManager>();
}

MemoryHandle DialectResourceManager::addBlobResource(StringRef baseName,
                                                     void *memory, size_t size,
                                                     size_t align) {
  return insert(baseName, mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
                              ArrayRef<char>((char *)memory, size), align));
}

void DialectResourceManager::provideStringResource(StringRef key,
                                                   StringRef value) {
  stringResources.modify([key](StringSet<> &strings) { strings.insert(key); });
  // There are no alignment guarantees for strings. Store them using the system
  // preferred alignment of the compiler.
  update(key, mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
                  {value.data(), value.size()}, kPreferredMemoryAlignment));
}

bool DialectResourceManager::isStringResource(StringRef key) {
  return stringResources.read(
      [key](const StringSet<> &strings) { return strings.contains(key); });
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
    return blobMgr.insert(key);
  }

  /// Parse a dialect resource. It may be either a string or a blob. Both are
  /// passed to the dialect resource manager as blobs.
  LogicalResult parseResource(mlir::AsmParsedResourceEntry &entry) const final {
    if (entry.getKind() == mlir::AsmResourceEntryKind::String) {
      FailureOr<std::string> value = entry.parseAsString();
      if (failed(value))
        return failure();
      blobMgr.provideStringResource(entry.getKey(), *value);
    } else {
      FailureOr<mlir::AsmResourceBlob> value = entry.parseAsBlob();
      if (failed(value))
        return failure();
      blobMgr.update(entry.getKey(), std::move(*value));
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
          if (blobMgr.isStringResource(key)) {
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
