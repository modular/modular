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

bool DebugInfo::shouldMaterializeConstantsInto(Region &region) {
  Operation *parent = region.getParentOp();
  if (auto scopedParent = dyn_cast<DebugInfo::SubprogramScoped>(parent))
    if (scopedParent.getLocScope())
      return true;
  if (auto scopedParent = dyn_cast<DebugInfo::InlinedSubprogramScoped>(parent))
    if (scopedParent.getCallLocAttr())
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

/// Make sure that all locations within a fused location have the same scope on
/// their locations.
static ErrorOr<DIScopeAttr> getAndValidateScopeIn(Location loc) {
  DIScopeAttr scope;
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    // FusedLoc _may_ contain the scope. If it doesn't, we need to ensure that
    // all the fused locations have the same scope, which we extract.
    scope = dyn_cast_or_null<DIScopeAttr>(fusedLoc.getMetadata());
    if (ArrayRef<Location> nestedLocs = fusedLoc.getLocations();
        !scope && !nestedLocs.empty()) {
      {
        auto scopeOr = getAndValidateScopeIn(nestedLocs.back());
        if (scopeOr.isError())
          return scopeOr.takeError();
        scope = std::move(*scopeOr);
      }
      for (Location nestedLoc : nestedLocs.drop_back()) {
        auto nestedScopeOr = getAndValidateScopeIn(nestedLoc);
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
  return getAndValidateScopeIn(callSiteLoc.getCaller());
}

/// Verify the location scope of ordinary op within a subprogram.
static LogicalResult verifyScope(ErrorOr<DIScopeAttr> scopeOr,
                                 DISubprogramAttr funcScope, Operation *op) {
  if (failed(scopeOr))
    return op->emitOpError(scopeOr.getError());

  while (auto lexBlock = dyn_cast_or_null<DILexicalBlockAttr>(*scopeOr))
    scopeOr = lexBlock.getScope();

  if (funcScope == *scopeOr)
    return success();
  return (op->emitOpError(
              "location scope does not match scope of parent func location: ")
          << *scopeOr)
             .attachNote()
         << "function scope: " << funcScope;
}

/// Verify the location scope of ordinary op within a subprogram.
static LogicalResult verifyScope(Operation *op, DISubprogramAttr funcScope) {
  Location loc = op->getLoc();
  // Allow ops to not carry location (due to constant folding or inlining).
  if (isa<UnknownLoc>(loc))
    return success();
  return verifyScope(getAndValidateScopeIn(loc), funcScope, op);
}

/// Verify the location scope of InlinedSubprogramScoped within a subprogram.
static LogicalResult verifyScope(InlinedSubprogramScoped inlined,
                                 DISubprogramAttr funcScope) {
  if (mlir::LocationAttr callLoc = inlined.getCallLocAttr()) {
    // Allow ops to not carry location (due to constant folding or inlining).
    if (isa<UnknownLoc>(callLoc))
      return success();
    return verifyScope(getAndValidateScopeIn(callLoc), funcScope, inlined);
  }
  return inlined->emitOpError("must have callsite location");
}

LogicalResult impl::verifySubprogramScoped(SubprogramScoped op) {

  auto verifyNormal = [&](Location loc) -> LogicalResult {
    if (!isa<FileLineColLoc>(loc)) {
      return op.emitOpError(
                 "must contain only file/line/col location, but got ")
             << loc;
    }
    return success();
  };

  auto verifyInlinedSubprogramScoped = [&](Location loc,
                                           bool verifyInner) -> LogicalResult {
    if (auto fused = dyn_cast<mlir::FusedLocWith<DICallLocAttr>>(loc)) {
      ArrayRef<Location> locs = fused.getLocations();
      if (locs.size() != 1)
        return op.emitOpError("must contain exactly one location");
      if (verifyInner)
        return verifyNormal(locs[0]);
    } else {
      if (verifyInner)
        return verifyNormal(loc);
    }
    return success();
  };

  Location funcLoc = op->getLoc();
  bool inlinedSPS = dyn_cast<InlinedSubprogramScoped>(op.getOperation());
  auto fusedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(funcLoc);
  if (!fusedLoc) {
    // If the function doesn't contain a debuginfo scope, we don't need to
    // verify anything, except InlinedSubprogramScoped structure. Named
    // locations indicate that we are dealing with some external location, which
    // may not comply with our rules.
    if (funcLoc->findInstanceOf<mlir::FusedLocWith<DIScopeAttr>>()) {
      return op.emitOpError("without debuginfo scope must contain only "
                            "file/line/col location but it is ")
             << funcLoc;
    }
    if (inlinedSPS)
      return verifyInlinedSubprogramScoped(funcLoc, false);
    return success();
  }

  DIScopeAttr scope = fusedLoc.getMetadata();
  auto funcScope = dyn_cast<DISubprogramAttr>(scope);
  if (!funcScope) {
    return op.emitOpError("must have subprogram scope in location, but got ")
           << scope;
  }
  ArrayRef<Location> locs = fusedLoc.getLocations();
  if (locs.size() != 1)
    return op.emitOpError("must contain exactly one location");
  LogicalResult innerCheck = inlinedSPS
                                 ? verifyInlinedSubprogramScoped(locs[0], true)
                                 : verifyNormal(locs[0]);
  if (innerCheck.failed())
    return innerCheck;

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

Location impl::getLocNoInlined(InlinedSubprogramScoped iss) {
  if (auto fusedCallLoc =
          dyn_cast<mlir::FusedLocWith<DebugInfo::DICallLocAttr>>(
              iss.getLoc())) {
    assert(fusedCallLoc.getLocations().size() >= 1 &&
           "callLoc fused loc bad size");
    return fusedCallLoc.getLocations()[0];
  }
  if (auto fusedLoc =
          dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(iss.getLoc())) {
    auto locs = fusedLoc.getLocations();
    assert(locs.size() >= 1 && "callLoc fused loc bad size");
    if (auto innerFused =
            dyn_cast<mlir::FusedLocWith<DebugInfo::DICallLocAttr>>(locs[0])) {
      auto innerLocs = innerFused.getLocations();
      assert(innerLocs.size() >= 1 && "callLoc inner fused bad size");
      // Re-fuse the inner location with the scope.
      return FusedLoc::get(iss.getContext(), innerLocs[0],
                           fusedLoc.getMetadata());
    } else {
      return iss.getLoc();
    }
  }
  return iss.getLoc();
}

mlir::LocationAttr impl::getCallLocAttr(InlinedSubprogramScoped iss) {
  if (auto fusedImmediate =
          dyn_cast<mlir::FusedLocWith<DebugInfo::DICallLocAttr>>(iss.getLoc()))
    return fusedImmediate.getMetadata().getCallLoc();

  auto fusedLoc =
      dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(iss.getLoc());
  if (!fusedLoc)
    return {};

  auto locs = fusedLoc.getLocations();
  assert(locs.size() >= 1 && "callLoc fused loc bad size");

  if (auto innerFused =
          dyn_cast<mlir::FusedLocWith<DebugInfo::DICallLocAttr>>(locs[0]))
    return innerFused.getMetadata().getCallLoc();
  else
    return {};
}

void impl::setCallLocAttr(InlinedSubprogramScoped iss,
                          mlir::LocationAttr callLoc) {
  Location callLocStripped = iss.getLocNoInlined();
  if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(
          callLocStripped)) {
    auto locs = fusedLoc.getLocations();
    assert(locs.size() >= 1 && "callLoc fused loc bad size");
    Location fusedInner =
        FusedLoc::get(iss.getContext(), locs[0],
                      DebugInfo::DICallLocAttr::get(iss.getContext(), callLoc));
    iss.getOperation()->setLoc(
        FusedLoc::get(iss.getContext(), fusedInner, fusedLoc.getMetadata()));
  } else {
    iss.getOperation()->setLoc(FusedLoc::get(
        iss.getContext(), callLocStripped,
        DebugInfo::DICallLocAttr::get(iss.getContext(), callLoc)));
  }
}

#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.cpp.inc"
