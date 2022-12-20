//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
/// The ExprEmitter class is the main driver for expression emission, providing
/// helper functions used by the individual node emission hooks.
//
//===----------------------------------------------------------------------===//

#include "LitExprEmitter.h"
#include "ASTDecl.h"
#include "LitExprCalls.h"
#include "SpecialFunctions.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// Emission helpers for various value classifications.
//===----------------------------------------------------------------------===//

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
RValue ExprEmitter::emitRValue(AnyValue rep, SMLoc loc) {
  if (!rep) // Already diagnosed error.
    return {};

  // If this is already an RValue, then we are done.
  if (auto rvRep = rep.getIfRValue())
    return rvRep;

  // Finally, if this is an LValue, emit a load.
  auto pointer = rep.getIfLValue();
  assert(pointer);

  if (!builder) {
    emitError(loc, "context only permits a meta value, not a dynamic one");
    return {};
  }

  return DRValue(builder->create<POP::LoadOp>(translateLocation(loc), pointer,
                                              /*alignment=*/std::nullopt));
}

DRValue ExprEmitter::emitDRValue(RValue rep, SMLoc loc) {
  if (!rep)
    return {};
  // If this is already an DRValue, emit this.
  if (auto rvalue = rep.getIfDRValue())
    return rvalue;

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  auto attr = rep.getIfMValue().get();

  if (!builder) {
    emitError(loc, "context only permits a meta value, not a dynamic one");
    return {};
  }

  auto location = translateLocation(loc);
  // Materialize index integer constants as a special case.
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    if (intAttr.getType().isIndex()) {
      auto cst = builder->create<mlir::index::ConstantOp>(
          location, intAttr.getValue().getSExtValue());
      return DRValue(cst);
    }

  // Otherwise, emit a generalized parameter constant.
  return DRValue(builder->create<ParamConstantOp>(location, attr));
}

/// This helper emits the specified expression as a meta value, diagnosing the
/// problem if the expression is only valid as a runtime value.  This returns
/// null if emission fails.
MValue ExprEmitter::emitMValue(const ExprNode *node, const Twine &message) {
  auto rep = node->emitIR(*this, /*No Contextual Type*/ {});
  if (!rep)
    return {};

  // If this is a parameter, return it.
  if (auto value = rep.getIfMValue())
    return value;

  emitError(node->getLoc(), message);
  return {};
}

/// Emit the specified expression as an LValue which can be loaded and stored.
/// If contextualType is non-null, then an implicitly declared LValue will be
/// assigned that type.
///
/// This diagnoses the expression with the specified message if it isn't a
/// valid LValue.
LValue ExprEmitter::emitLValue(const ExprNode *node, ASTType contextualType,
                               const Twine &message) {
  AnyValue anyValue = node->emitIR(*this, contextualType);
  if (!anyValue)
    return {}; // Error already diagnosed.
  if (LValue lValue = anyValue.getIfLValue())
    return lValue;
  emitError(node->getLoc(), message);
  return {};
}

/// This helper emits the specified expression tree as a type, e.g. turning
/// "Int" into the type for it.  This never returns null - if the expression
/// is erroneous, it is diagnosed and a TypeCheckErrorType is returned.
ASTType ExprEmitter::emitType(const ExprNode *node) {
  auto value = emitMValue(node, "expected a type");
  if (!value)
    return shared.getTypeCheckErrorType();

  // If this emitted a type, we can lower it.
  if (auto type = value.getIfTypeValue())
    return type;

  // If we emitted a NoneAttr then convert it to a NoneType.  This is a
  // special case because "None" is both a value and a type, and defaults to a
  // value.
  if (isa<NoneAttr>(value.get()))
    return shared.getNoneType();

  emitError(node->getLoc(), "expected a type, not a value");
  return shared.getTypeCheckErrorType();
}

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

/// This helper emits a method call to a special function (`kind`) on `type`
/// with the provided `operands`. This emits an error if the special function
/// is not implemented by the type and returns null.
AnyValue
ExprEmitter::emitSpecialMethodCall(ASTType type, SpecialFunctionKind kind,
                                   ArrayRef<ASTExprAnd<AnyValue>> operands,
                                   SMLoc callLoc) {
  // Look up the special function based on the SpecialFunctionKind.
  auto specialFnInfo = SpecialFunctionInfo::get(kind);

  bool isErroneousDecl = false;
  CallableValue callee(type, specialFnInfo.name, callLoc,
                       /*emitErrorOnFailure=*/true, isErroneousDecl, shared);
  return callee.emitFunctionCall(operands, callLoc, *this);
}

/// Convert the specified DRValue to the expected type, invoking implicit
/// conversions if necessary.  On error, this diagnoses it and returns null.
DRValue ExprEmitter::getAsExpectedType(DRValue value, const ExprNode *expr,
                                       ASTType expectedType) {
  if (!value)
    return value;
  // If the type is already an exact match, then we are done.
  if (ASTType(value.getType()).isEqualCanon(expectedType))
    return value;

  // Check to see if we can invoke an __new__ method to convert it.
  bool isErroneousDecl = false;
  CallableValue callee(expectedType, "__new__", expr->getLoc(),
                       /*emitErrorOnFailure=*/false, isErroneousDecl, shared);
  if (callee.isNull()) {
    if (!isErroneousDecl) {
      emitError(expr->getLoc(), "value of type ")
          << ASTType(value.getType())
          << " cannot be converted to expected type " << expectedType;
    }
    return {};
  }

  ASTExprAnd<AnyValue> newArg = {DRValue(value), expr};
  auto result = callee.emitFunctionCall(newArg, expr->getLoc(), *this);
  if (!result)
    return {};

  // Make sure the result is a DRValue.
  return emitDRValue(result, expr->getLoc());
}

/// Emit the specified expression as a condition, converting it to an MLIR I1
/// value that we can test directly, and also returning the intermediate
/// result of calling `__bool__` (which is typically a Bool or object type, but
/// not guaranteed).  This reports and error and returns null on error.
DRValue ExprEmitter::emitConditionValueAsI1(ASTExprAnd<AnyValue> value,
                                            AnyValue &boolResult) {
  if (!value.ir)
    return {};

  SMLoc valueLoc = value.expr->getLoc();
  boolResult = value.ir;

  // If this is already an 'i1', then we're done.
  if (value.ir.getType().isInteger(1))
    return emitDRValue(value.ir, valueLoc);

  // Check for the presence of a __lit_bool method.  If it exists, we can avoid
  // a redundant call to __bool__ for Bool types.
  bool isErroneousDecl = false;
  if (!CallableValue(value.ir.getType(), "__lit_bool", valueLoc,
                     /*emitErrorOnFailure=*/false, isErroneousDecl, shared)) {
    // Use the __bool__ method to convert the user defined type to
    // something that is a Bool or other type that implements __lit_bool.
    boolResult =
        emitSpecialMethodCall(value.ir.getType(), SpecialFunctionKind::kBool,
                              {{value.ir, value.expr}}, valueLoc);
    if (!boolResult)
      return {};
  }

  // Then we use __lit_bool to convert to an i1 value.
  AnyValue litBoolCall =
      emitSpecialMethodCall(boolResult.getType(), SpecialFunctionKind::kLitBool,
                            {{boolResult, value.expr}}, valueLoc);
  return emitDRValue(litBoolCall, valueLoc);
}

/// Emit the specified expression as a condition, converting it to an MLIR I1
/// value that we can test directly.  This reports and error and returns null on
/// error.
DRValue ExprEmitter::emitConditionValueAsI1(ExprNode *condExpr) {
  AnyValue boolTmp; // we don't care about the intermediate Bool value.
  return emitConditionValueAsI1({emitRValue(condExpr), condExpr}, boolTmp);
}
