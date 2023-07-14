//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// DIAttr
//===----------------------------------------------------------------------===//

bool DIAttr::classof(Attribute attr) {
  return llvm::isa<DebugInfoDialect>(attr.getDialect());
}

//===----------------------------------------------------------------------===//
// DIScopeAttr
//===----------------------------------------------------------------------===//

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DIFileAttr, DILocalScopeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DILocalScopeAttr
//===----------------------------------------------------------------------===//

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DISubprogramAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

DISubprogramAttr DISubprogramAttr::cloneWith(StringRef name,
                                             StringRef linkageName) const {
  return DebugInfo::DISubprogramAttr::get(
      getCompileUnit(), getScope(), name, linkageName, getFile(), getLine(),
      getScopeLine(), getSubprogramFlags(), getType());
};

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

DIScopeAttr DebugInfo::extractScope(Location loc) {
  if (auto fusedLoc =
          loc->findInstanceOf<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>())
    return fusedLoc.getMetadata();
  return {};
}

DIScopeAttr DebugInfo::extractScope(Operation *op) {
  return extractScope(op->getLoc());
}

void DIAttrTypeReplacer::replaceElementsIn(Operation *op) {
  // As an optimization, we only replace attributes within the dictionaries of
  // DebugInfo operations. For everything else, we only check the location for
  // debug info.
  bool updateAttrs =
      llvm::isa_and_present<DebugInfo::DebugInfoDialect>(op->getDialect());
  AttrTypeReplacer::replaceElementsIn(op, updateAttrs, /*replaceLocs=*/true);
}

void DIAttrTypeReplacer::recursivelyReplaceElementsIn(Operation *op) {
  op->walk([&](Operation *op) { replaceElementsIn(op); });
}

LogicalResult DebugInfo::verifyFuncLocScope(mlir::FunctionOpInterface op) {
  if (DebugInfo::DIScopeAttr scope = DebugInfo::extractScope(op.getLoc())) {
    if (!isa<DebugInfo::DISubprogramAttr>(scope)) {
      return op.emitOpError("must have subprogram scope in location, but got ")
             << scope;
    }
  }
  return success();
}
