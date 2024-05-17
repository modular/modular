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

/// Returns whether the child scope is nested inside the ancestor scope.
static bool IsSubScope(DIScopeAttr child, DIScopeAttr ancestor) {
  if (child == ancestor) // short-circuit for common case.
    return true;

  return child
      .walk([&](DIScopeAttr scope) {
        if (scope == ancestor)
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

/// The local variable type must match the value type. Compare the types while
/// unwrapping debuginfo types.
static LogicalResult verifyValueOpType(ValueOp op) {
  Type inputType = op.getValue().getType();

  // All occurrences of debuginfo.expr.irvalue in the location conversion
  // expression must have types that match the ir (input) type.
  auto conversionExpr = op.getConversionExprAttr();
  auto walkResult = conversionExpr.walk([&](DIIRValueExprAttr irValue) {
    // We can only compare types if the irValue type is not yet resolved.
    if (auto unresolved = dyn_cast<DIUnresolvedMLIRType>(irValue.getDIType())) {
      if (unresolved.getType() != inputType) {
        op->emitOpError("conversion expression input expr.irvalue type ")
            << unresolved.getType() << " does not match actual IR Value type "
            << inputType;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  DIType outputType = conversionExpr.getDIType();
  DIType declaredType = op.getValueInfo().getType();
  if (declaredType != outputType) {
    return op.emitOpError("conversion expression output type ")
           << outputType << " does not match variable declared type "
           << declaredType;
  }
  return success();
}

static ParseResult parseValueOpAttrs(OpAsmParser &p,
                                     DILocalVariableAttr &varInfo,
                                     DIExprAttr &conversionExpr) {
  if (p.parseAttribute(varInfo))
    return failure();

  auto parseResult = p.parseOptionalAttribute(conversionExpr);
  if (parseResult.has_value())
    return *parseResult;

  // Does not contain a parseResult. Create an "identity" conversion.
  conversionExpr = DIIRValueExprAttr::get(varInfo.getType());
  return success();
}

static void printValueOpAttrs(OpAsmPrinter &p, ValueOp value,
                              DILocalVariableAttr varInfo,
                              DIExprAttr conversionExpr) {
  p.printAttribute(varInfo);

  if (auto irValue = llvm::dyn_cast<DIIRValueExprAttr>(conversionExpr)) {
    if (irValue.getDIType() == varInfo.getType()) {
      // Omit identity conversion.
      return;
    }
  }
  p << ' ';
  p.printAttribute(conversionExpr);
}

LogicalResult ValueOp::verify() {
  if (failed(verifyValueOpType(*this)))
    return failure();

  ErrorOr<DIScopeAttr> locationScopeOr = getValueOpLocationScope(getLoc());
  if (locationScopeOr.isError())
    return emitOpError(locationScopeOr.getError());

  DILocalVariableAttr varAttr = getValueInfo();
  if (DIScopeAttr locationScope = *locationScopeOr) {
    if (!IsSubScope(locationScope, varAttr.getScope())) {
      return emitOpError(
                 "location scope must be a child scope of the variable scope: ")
             << locationScope << " vs. " << varAttr.getScope();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfo.cpp.inc"
