//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
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

DISubprogramAttr DebugInfo::extractScope(mlir::FunctionOpInterface funcOp) {
  if (auto fusedLoc =
          dyn_cast<mlir::FusedLocWith<DISubprogramAttr>>(funcOp->getLoc()))
    return fusedLoc.getMetadata();
  return {};
}

DIScopeAttr DebugInfo::extractScope(Operation *op) {
  if (auto scopedOp = dyn_cast<DebugInfo::ScopedLocation>(op))
    return scopedOp.getLocScope();
  if (auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op))
    if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(op->getLoc()))
      return fusedLoc.getMetadata();

  // For other ops, we look for the scope recursively.
  if (auto fusedLoc =
          op->getLoc()->findInstanceOf<mlir::FusedLocWith<DIScopeAttr>>())
    return fusedLoc.getMetadata();
  return {};
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

/// Return the scope from a location of an op within a function's body,
/// recursively walking up through a chain of inlined locations if needed,
/// always following the caller location.
static ErrorOr<DIScopeAttr> getScopeWithinBody(Location loc) {
  DIScopeAttr scope;
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    // FusedLoc _may_ contain the scope. If it doesn't, we need to ensure that
    // all the fused locations have the same scope, which we extract.
    scope = dyn_cast_or_null<DIScopeAttr>(fusedLoc.getMetadata());
    if (ArrayRef<Location> nestedLocs = fusedLoc.getLocations();
        !scope && !nestedLocs.empty()) {
      UNWRAP_ERROR_OR_SET(scope, getScopeWithinBody(nestedLocs.back()));
      for (Location nestedLoc : nestedLocs.drop_back()) {
        UNWRAP_ERROR(nestedScope, getScopeWithinBody(nestedLoc));
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

LogicalResult DebugInfo::verifyFuncLocScope(mlir::FunctionOpInterface funcOp) {
  auto fusedLoc =
      dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(funcOp->getLoc());
  if (!fusedLoc)
    return success();

  // If the function doesn't contain a location scope, we don't verify anything.
  DebugInfo::DIScopeAttr scope = fusedLoc.getMetadata();
  if (!scope)
    return success();

  auto funcScope = dyn_cast<DISubprogramAttr>(scope);
  if (!funcScope) {
    return funcOp.emitOpError(
               "must have subprogram scope in location, but got ")
           << scope;
  }

  // We walk pre-order, and skip nested functions.
  WalkResult res = funcOp.getFunctionBody().walk<mlir::WalkOrder::PreOrder>(
      [&](Operation *op) {
        if (isa<mlir::FunctionOpInterface>(op))
          return WalkResult::skip();

        ErrorOr<DIScopeAttr> scopeOr = getScopeWithinBody(op->getLoc());
        if (scopeOr.isError()) {
          res = op->emitOpError(scopeOr.getError());
          return WalkResult::interrupt();
        }

        // We might find a lexical block scope, so we look through it.
        while (auto lexBlock = dyn_cast_or_null<DILexicalBlockAttr>(*scopeOr))
          scopeOr = lexBlock.getScope();

        if (funcScope != *scopeOr) {
          res = op->emitOpError("location scope does not match scope of parent "
                                "func location: ")
                << funcScope;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  return failure(res.wasInterrupted());
}

void DebugInfo::updateSubprogram(mlir::FunctionOpInterface funcOp,
                                 StringAttr linkageName, StringAttr name) {
  auto funcSp = extractScope<DISubprogramAttr>(funcOp);
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
