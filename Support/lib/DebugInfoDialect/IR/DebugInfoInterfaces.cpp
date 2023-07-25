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

LogicalResult impl::verifySubprogramScoped(SubprogramScoped op) {
  // If the function doesn't contain a location scope, we don't verify anything.
  auto fusedLoc =
      dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(op->getLoc());
  if (!fusedLoc)
    return success();

  DebugInfo::DIScopeAttr scope = fusedLoc.getMetadata();
  auto funcScope = dyn_cast<DebugInfo::DISubprogramAttr>(scope);
  if (!funcScope) {
    return op.emitOpError("must have subprogram scope in location, but got ")
           << scope;
  }

  // We walk pre-order, and skip nested functions.
  WalkResult res =
      op.getBodyRegion().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<SubprogramScoped>(op))
          return WalkResult::skip();

        ErrorOr<DebugInfo::DIScopeAttr> scopeOr =
            DebugInfo::getScopeWithinBody(op->getLoc());
        if (scopeOr.isError()) {
          res = op->emitOpError(scopeOr.getError());
          return WalkResult::interrupt();
        }

        // We might find a lexical block scope, so we look through it.
        while (auto lexBlock =
                   dyn_cast_or_null<DebugInfo::DILexicalBlockAttr>(*scopeOr))
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

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.cpp.inc"
