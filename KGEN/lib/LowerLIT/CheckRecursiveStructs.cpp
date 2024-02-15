//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/LITDialect/LITOps.h"

using namespace M;
using namespace KGEN;

/// Find all the structs in the module.
static llvm::MapVector<SymbolRefAttr, LIT::StructDeclOp>
collectStructs(Operation *module) {
  llvm::MapVector<SymbolRefAttr, LIT::StructDeclOp> structMap;
  module->walk([&](Operation *op) {
    // Collect structs.
    if (auto structOp = dyn_cast<LIT::StructDeclOp>(op))
      structMap[getFullyResolvedSymbolRef(structOp)] = structOp;
  });
  return structMap;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_CHECKRECURSIVESTRUCTS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct CheckRecursiveStructs
    : impl::CheckRecursiveStructsBase<CheckRecursiveStructs> {
  using CheckRecursiveStructsBase::CheckRecursiveStructsBase;

  void runOnOperation() override {
    // Collect all the structs.
    llvm::MapVector<SymbolRefAttr, LIT::StructDeclOp> structMap =
        collectStructs(getOperation());

    // Check all the structs.
    if (failed(checkStructs(structMap)))
      return signalPassFailure();
  }

private:
  /// Process all the structs to see if they contain recursive nested struct
  /// fields.
  static LogicalResult
  checkStructs(llvm::MapVector<SymbolRefAttr, LIT::StructDeclOp> &structMap);
};
} // namespace

LogicalResult
scanField(Type type, DenseSet<Type> &seenTypes,
          DenseMap<SymbolRefAttr, LogicalResult> &scanned,
          llvm::MapVector<SymbolRefAttr, LIT::StructDeclOp> &structMap,
          Operation *op) {
  auto declRef = dyn_cast<LIT::DeclRefType>(type);
  if (!declRef)
    return success();

  if (seenTypes.contains(type)) {
    // Found recursive nested struct field.
    return mlir::emitError(op->getLoc(),
                           "recursive nested struct field, try "
                           "adding indirection to recursive reference");
  }

  // See if we've already looked this up, if so, this is already scanned
  SymbolRefAttr structSymbol = declRef.getSymbol();
  auto it = scanned.find(structSymbol);
  if (it != scanned.end())
    return it->second;

  // If not, we scan it recursively.  Structs cannot be infinitely deep, so
  // we can just do this recursively.
  auto smIt = structMap.find(structSymbol);
  assert(smIt != structMap.end() && smIt->second &&
         "reference to struct that wasn't declared");
  LIT::StructDeclOp decl = smIt->second;

  seenTypes.insert(type);
  bool hasRecursiveFields = false;
  for (auto field : decl.getFieldDecls()) {
    // Scan all the fields.
    hasRecursiveFields |= failed(
        scanField(field.getType(), seenTypes, scanned, structMap, field));
  }
  seenTypes.erase(type);

  scanned.insert({structSymbol, failure(hasRecursiveFields)});
  return failure(hasRecursiveFields);
}

LogicalResult CheckRecursiveStructs::checkStructs(
    llvm::MapVector<SymbolRefAttr, LIT::StructDeclOp> &structMap) {

  bool hasRecursiveStructs = false;
  DenseMap<SymbolRefAttr, LogicalResult> scanned;

  for (auto [name, structDecl] : structMap) {
    bool hasRecursiveFields = false;
    DenseSet<Type> seenTypes;
    for (auto field : structDecl.getFieldDecls()) {
      hasRecursiveFields |= failed(
          scanField(field.getType(), seenTypes, scanned, structMap, field));
    }
    if (hasRecursiveFields)
      mlir::emitError(structDecl->getLoc(),
                      "struct contains recursive reference to itself");

    hasRecursiveStructs |= hasRecursiveFields;
  }
  return failure(hasRecursiveStructs);
}
