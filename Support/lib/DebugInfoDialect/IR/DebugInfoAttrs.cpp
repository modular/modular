//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/ErrorOr.h"
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

DISubprogramAttr DebugInfo::extractScope(mlir::FunctionOpInterface funcOp) {
  if (auto fusedLoc =
          dyn_cast<mlir::FusedLocWith<DISubprogramAttr>>(funcOp->getLoc()))
    return fusedLoc.getMetadata();
  return {};
}

DIScopeAttr DebugInfo::extractScope(Operation *op) {
  if (auto scopedOp = dyn_cast<DebugInfo::ScopedLocation>(op))
    return scopedOp.getLocScope();

  // For other ops, we look for the scope recursively.
  ErrorOr<DIScopeAttr> scopeOr = getScopeWithinBody(op->getLoc());
  if (scopeOr.isError())
    return {};
  return *scopeOr;
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

void DebugInfo::updateSubprogram(mlir::FunctionOpInterface funcOp,
                                 StringAttr linkageName, StringAttr name) {
  DISubprogramAttr funcSp = extractScope(funcOp);
  if (!funcSp)
    return;

  if (!name)
    name = funcSp.getName();
  DISubprogramAttr newAttr = funcSp.cloneWith(name, linkageName);

  DIAttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](DISubprogramAttr sp) { return sp == funcSp ? newAttr : sp; });
  replacer.recursivelyReplaceElementsIn(funcOp);
}

ErrorOr<DIScopeAttr> DebugInfo::getScopeWithinBody(Location loc) {
  DIScopeAttr scope;
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    // FusedLoc _may_ contain the scope. If it doesn't, we need to ensure that
    // all the fused locations have the same scope, which we extract.
    scope = dyn_cast_or_null<DIScopeAttr>(fusedLoc.getMetadata());
    if (ArrayRef<Location> nestedLocs = fusedLoc.getLocations();
        !scope && !nestedLocs.empty()) {
      {
        auto scopeOr = getScopeWithinBody(nestedLocs.back());
        if (scopeOr.isError())
          return scopeOr.takeError();
        scope = std::move(*scopeOr);
      }
      for (Location nestedLoc : nestedLocs.drop_back()) {
        auto nestedScopeOr = getScopeWithinBody(nestedLoc);
        if (nestedScopeOr.isError())
          return nestedScopeOr.takeError();
        auto nestedScope = std::move(*nestedScopeOr);
        if (nestedScope != scope)
          return Error("contains inconsistent scopes in fused location");
      }
    }
  }

  // If not dealing with an inlined location, we return a scope (if found).
  auto callSiteLoc = dyn_cast<mlir::CallSiteLoc>(loc);
  if (!callSiteLoc)
    return scope;

  // Otherwise, we walk up the inlining chain.
  return getScopeWithinBody(callSiteLoc.getCaller());
}

void DebugInfo::updateInlinedLoc(Operation *op, Location callerLoc,
                                 bool stripDebugInfo) {
  if (auto inlined = dyn_cast<DebugInfo::InlinedSubprogramScoped>(op)) {
    if (stripDebugInfo)
      inlined.setCallLocAttr(callerLoc);
    else if (mlir::LocationAttr callLoc = inlined.getCallLocAttr())
      inlined.setCallLocAttr(mlir::CallSiteLoc::get(callLoc, callerLoc));
  } else if (!isa<DebugInfo::SubprogramScoped>(op)) {
    if (stripDebugInfo)
      op->setLoc(callerLoc);
    else
      op->setLoc(mlir::CallSiteLoc::get(op->getLoc(), callerLoc));
  }
}
