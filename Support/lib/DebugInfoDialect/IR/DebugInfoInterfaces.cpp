//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/ErrorOr.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

/// Verify the location scope of ordinary op within a subprogram.
static LogicalResult verifyScope(ErrorOr<DIScopeAttr> scopeOr,
                                 DISubprogramAttr funcScope, Operation *op) {
  if (failed(scopeOr))
    return op->emitOpError(scopeOr.getError());

  while (auto lexBlock = dyn_cast_or_null<DILexicalBlockAttr>(*scopeOr))
    scopeOr = lexBlock.getScope();

  if (funcScope == *scopeOr)
    return success();
  return op->emitOpError(
             "location scope does not match scope of parent func location: ")
         << funcScope;
}

/// Verify the location scope of ordinary op within a subprogram.
static LogicalResult verifyScope(Operation *op, DISubprogramAttr funcScope) {
  return verifyScope(getScopeWithinBody(op->getLoc()), funcScope, op);
}

/// Verify the location scope of InlinedSubprogramScoped within a subprogram.
static LogicalResult verifyScope(InlinedSubprogramScoped inlined,
                                 DISubprogramAttr funcScope) {
  auto getScope = [](auto op) -> ErrorOr<DIScopeAttr> {
    if (mlir::LocationAttr callLoc = op.getCallLocAttr())
      return getScopeWithinBody(callLoc);
    return Error("must have callsite location");
  };
  return verifyScope(getScope(inlined), funcScope, inlined);
}

LogicalResult impl::verifySubprogramScoped(SubprogramScoped op) {
  Location funcLoc = op->getLoc();
  auto fusedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(funcLoc);
  if (!fusedLoc) {
    // If the function doesn't contain a debuginfo scope, we don't need to
    // verify anything. Named locations indicate that we are dealing with some
    // external location, which may not comply with our rules.
    if (isa<FileLineColLoc, mlir::NameLoc>(funcLoc))
      return success();
    return op.emitOpError(
        "without debuginfo scope must contain only file/line/col location");
  }

  ArrayRef<Location> locs = fusedLoc.getLocations();
  if (locs.size() != 1)
    return op.emitOpError("must contain exactly one location");

  if (!isa<FileLineColLoc>(locs[0]))
    return op.emitOpError("must contain only file/line/col location");

  DIScopeAttr scope = fusedLoc.getMetadata();
  auto funcScope = dyn_cast<DISubprogramAttr>(scope);
  if (!funcScope) {
    return op.emitOpError("must have subprogram scope in location, but got ")
           << scope;
  }

  // We walk pre-order, and skip nested functions.
  WalkResult res =
      op.getBodyRegion().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
        if (auto inlined = dyn_cast<InlinedSubprogramScoped>(op)) {
          if (failed(verifyScope(inlined, funcScope)))
            return WalkResult::interrupt();
          return WalkResult::skip();
        } else if (isa<SubprogramScoped>(op)) {
          return WalkResult::skip();
        }

        if (failed(verifyScope(op, funcScope)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  return failure(res.wasInterrupted());
}

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.cpp.inc"
