//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetVector.h"

using namespace M;

//===----------------------------------------------------------------------===//
// MBlobManagerInterface
//===----------------------------------------------------------------------===//

using MBlobManagerInterface =
    mlir::ResourceBlobManagerDialectInterfaceBase<MemoryHandle>;

//===----------------------------------------------------------------------===//
// MOpAsmDialectInterface
//===----------------------------------------------------------------------===//

namespace {
class MOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
public:
  MOpAsmDialectInterface(Dialect *dialect, MBlobManagerInterface &blobMgr)
      : OpAsmDialectInterface(dialect), blobMgr(blobMgr) {}

  std::string
  getResourceKey(const mlir::AsmDialectResourceHandle &handle) const override {
    return cast<MemoryHandle>(handle).getKey().str();
  }
  FailureOr<mlir::AsmDialectResourceHandle>
  declareResource(StringRef key) const final {
    return blobMgr.insert(key);
  }
  LogicalResult parseResource(mlir::AsmParsedResourceEntry &entry) const final {
    FailureOr<mlir::AsmResourceBlob> blob = entry.parseAsBlob();
    if (failed(blob))
      return failure();
    blobMgr.update(entry.getKey(), std::move(*blob));
    return success();
  }
  void buildResources(Operation *op,
                      const llvm::SetVector<mlir::AsmDialectResourceHandle>
                          &referencedResources,
                      mlir::AsmResourceBuilder &provider) const final {
    blobMgr.buildResources(provider, referencedResources.getArrayRef());
  }

private:
  /// The blob manager.
  MBlobManagerInterface &blobMgr;
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

  auto &blobMgr = addInterface<MBlobManagerInterface>();
  addInterface<MOpAsmDialectInterface>(blobMgr);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.cpp.inc"
