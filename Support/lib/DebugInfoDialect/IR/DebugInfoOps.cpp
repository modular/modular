//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Support/DebugInfoDialect/IR/DebugInfo.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

/// Return the scope from a ValueOp's location, recursively walking up through a
/// chain of inlined locations if needed.
static ErrorOr<DIScopeAttr> getValueOpLocationScope(Location loc) {
  DIScopeAttr scope;
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    // Since ValueOp belongs to a single variable declaration, nothing should
    // ever give it a fused location.
    ArrayRef<Location> locations = fusedLoc.getLocations();
    if (size_t numLocs = locations.size(); numLocs != 1) {
      return Error(
          "with fused location must reference a single location, got " +
          Twine(numLocs));
    }

    // FusedLoc _may_ contain the scope.
    scope = dyn_cast_or_null<DIScopeAttr>(fusedLoc.getMetadata());
    loc = locations[0];
  }

  // If not dealing with an inlined location, we return a scope (if found).
  auto callSiteLoc = dyn_cast<mlir::CallSiteLoc>(loc);
  if (!callSiteLoc)
    return scope;

  // Otherwise, we walk up the inlining chain.
  return getValueOpLocationScope(callSiteLoc.getCallee());
}

LogicalResult ValueOp::verify() {
  ErrorOr<DIScopeAttr> scopeOr = getValueOpLocationScope(getLoc());
  if (scopeOr.isError())
    return emitOpError(scopeOr.getError());

  DILocalVariableAttr varAttr = getValueInfo();
  if (DIScopeAttr scope = *scopeOr) {
    if (varAttr.getScope() != scope) {
      return emitOpError("location scope must match variable scope: ")
             << scope << " vs. " << varAttr.getScope();
    }
  }

  // The surrounding subprogram op must have a subprogram scope.
  auto scope = (*this)->getParentOfType<SubprogramScoped>();
  if (!scope)
    return success();
  if (!isa_and_nonnull<DISubprogramAttr>(scope.getLocScope())) {
    return emitOpError("is contained within a subprogram scoped operation that "
                       "lacks a subprogram scope")
               .attachNote(scope.getLoc())
           << "see surrounding scope function here";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfo.cpp.inc"
