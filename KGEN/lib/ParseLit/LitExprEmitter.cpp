//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
/// The IREmitter class is the main driver for expression emission, providing
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
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// IREmitter implementation
//===----------------------------------------------------------------------===//

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
RValue IREmitter::emitRValue(AnyValue rep, SMLoc loc) {
  if (!rep) // Already diagnosed error.
    return {};

  // If this is already an RValue, then we are done.
  if (auto rvRep = rep.getIfRValue())
    return rvRep;

  // Finally, if this is an LValue, emit a load.
  auto pointer = rep.getIfLValue();
  assert(pointer);

  if (!builder) {
    // TODO: Add range.
    emitError(loc, "context only permits a meta value, not a dynamic one");
    return {};
  }

  return DRValue(builder->create<POP::LoadOp>(translateLocation(loc), pointer,
                                              /*alignment=*/std::nullopt));
}

DRValue IREmitter::emitDRValue(RValue rep, SMLoc loc) {
  if (!rep)
    return {};
  // If this is already an DRValue, emit this.
  if (auto rvalue = rep.getIfDRValue())
    return rvalue;

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  auto attr = rep.getIfMValue().get();
  if (!builder) {
    // TODO: Add range.
    emitError(loc, "context only permits a meta value, not a dynamic one");
    return {};
  }

  // If the value being materialized is itself parameterized, then we cannot
  // materialize it as an SSA value - there will be no way to bind parameters to
  // it.
  // TODO: We should have a general predicate from this provided by the KGEN
  // parameter utilities.
  if (auto signature = dyn_cast<SignatureType>(attr.getType())) {
    if (!signature.getInputParams().empty() ||
        !signature.getResultParamTypes().empty()) {
      // TODO: Add range.
      emitError(loc, "cannot use parameterized function of type ")
          << ASTType(attr.getType()) << " without binding all its parameters";
      return {};
    }
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

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

/// This helper emits a named method call with the provided `argValues`, where
/// the first arg is the receiver of the call. This emits an error if the
/// call is invalid and returns null.  The argValues list may not be empty.
AnyValue
IREmitter::emitNamedMethodCall(StringRef methodName,
                               ArrayRef<ASTExprAnd<AnyValue>> argValues,
                               CallSyntax syntax, SMLoc callLoc) {
  assert(!argValues.empty() && "Cannot emit a method call without a receiver!");
  ASTType type = argValues.front().ir.getRValueType();
  CallableValue callee(type, methodName, callLoc, shared);
  return callee.emitFunctionCall(argValues, syntax, callLoc, *this);
}

/// Convert the specified value to the expected type, invoking implicit
/// conversions if necessary.  On error, this diagnoses it and returns null.
AnyValue IREmitter::getAsExpectedType(AnyValue value, const ExprNode *expr,
                                      ASTType expectedType,
                                      std::function<void()> errorHandler) {
  // If the value handed to is us already erroneous, don't diagnose anything.
  if (!value)
    return value;

  // If the type is already an exact match, then we are done.
  if (ASTType(value.getType()).isEqualCanon(expectedType))
    return value;

  // Check to see if we can invoke an __new__ method to convert it.
  bool isErroneousDecl = false;
  CallableValue callee(expectedType, "__new__", expr->getLoc(), isErroneousDecl,
                       shared);
  if (callee.isNull()) {
    if (!isErroneousDecl)
      errorHandler();
    return {};
  }

  // If we have at least one candidate, we check to see if any of them can
  // work. We disable implicit conversions though, to prevent converting
  // T -> S -> U in one step.
  ASTExprAnd<AnyValue> newArg = {value, expr};
  callee.direct->disableImplicitConversions = true;
  if (failed(callee.direct->filterOverloadSet(
          {newArg}, CallSyntax::kImplicitConvert,
          /*emitDiagnosticOnFailure=*/false, shared))) {
    errorHandler();
    return {};
  }

  // Ok, cool we know it will succeed; do it.
  return callee.emitFunctionCall(newArg, CallSyntax::kImplicitConvert,
                                 expr->getLoc(), *this);
}

AnyValue IREmitter::getAsExpectedType(AnyValue value, const ExprNode *expr,
                                      ASTType expectedType,
                                      const Twine &errorSuffix) {
  auto errorHandler = [&]() {
    emitError(expr->getLoc())
        << ASTType(value.getType()) << " value cannot be converted to "
        << expectedType << errorSuffix << expr->getRange();
  };
  return getAsExpectedType(value, expr, expectedType, std::move(errorHandler));
}

/// Emit the specified expression as a condition, converting it to an MLIR I1
/// value that we can test directly, and also returning the intermediate
/// result of calling `__bool__` (which is typically a Bool or object type, but
/// not guaranteed).  This reports and error and returns null on error.
DRValue IREmitter::emitConditionValueAsI1(ASTExprAnd<AnyValue> value,
                                          AnyValue &boolResult) {
  if (!value.ir)
    return {};

  SMLoc valueLoc = value.expr->getLoc();
  boolResult = value.ir;

  // If this is already an 'i1', then we're done.
  if (value.ir.getType().isInteger(1))
    return emitDRValue(value.ir, valueLoc);

  // TODO: Python manual includes this off-hand comment:
  // Also, an object that doesn’t define a __bool__() method and whose __len__()
  // method returns zero is considered to be false in a Boolean context.

  // Check for the presence of a __lit_bool method.  If it exists, we can avoid
  // a redundant call to __bool__ for Bool types.
  bool isErroneousDecl = false;
  if (!CallableValue(value.ir.getType(), "__lit_bool", valueLoc,
                     isErroneousDecl, shared)) {
    // Use the __bool__ method to convert the user defined type to
    // something that is a Bool or other type that implements __lit_bool.
    boolResult = emitNamedMethodCall("__bool__", {{value.ir, value.expr}},
                                     CallSyntax::kImplicitConvert, valueLoc);
    if (!boolResult)
      return {};
  }

  // Then we use __lit_bool to convert to an i1 value.
  AnyValue litBoolCall =
      emitNamedMethodCall("__lit_bool", {{boolResult, value.expr}},
                          CallSyntax::kImplicitConvert, valueLoc);
  return emitDRValue(litBoolCall, valueLoc);
}

//===----------------------------------------------------------------------===//
// ExprEmitter implementation
//===----------------------------------------------------------------------===//

/// This helper emits the specified expression as a meta value, diagnosing the
/// problem if the expression is only valid as a runtime value.  This returns
/// null if emission fails.
MValue ExprEmitter::emitExprMValue(const ExprNode *node, const Twine &message) {
  // Clear the builder to indicate that an MValue must be emitted.
  llvm::SaveAndRestore<std::optional<OpBuilder>> savedBuilder(builder);
  builder.reset();

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
LValue ExprEmitter::emitExprLValue(SMLoc loc, const ExprNode *node,
                                   ASTType contextualType,
                                   const Twine &message) {
  AnyValue anyValue = node->emitIR(*this, contextualType);
  if (!anyValue)
    return {}; // Error already diagnosed.
  if (LValue lValue = anyValue.getIfLValue())
    return lValue;
  emitError(loc, message) << node->getRange();
  return {};
}

/// This helper emits the specified expression tree as a type, e.g. turning
/// "Int" into the type for it.  This never returns null - if the expression
/// is erroneous, it is diagnosed and a TypeCheckErrorType is returned.
ASTType ExprEmitter::emitExprType(const ExprNode *node) {
  auto value = emitExprMValue(node, "expected a type");
  if (!value)
    return shared.getTypeCheckErrorType();

  // If this emitted a type, we can lower it.
  if (auto type = value.getIfTypeValue()) {
    // Verify that all of the parameters for this type are bound.  We allow
    // MValues to refer to parameteric type, but anything calling `emitType` can
    // only handle fully bound types.
    if (auto *decl = type.getDecl(shared)) {
      auto structDecl = cast<StructDeclOp>(*decl);
      if (type.getParamBindings().size() !=
          structDecl.getInputParamDecls().size()) {
        size_t numMissing = structDecl.getInputParamDecls().size() -
                            type.getParamBindings().size();
        emitError(node->getLoc(), "use of type ")
            << structDecl.getNameAttr() << " with " << numMissing
            << " unbound parameter" << plural(numMissing) << node->getRange();
        return shared.getTypeCheckErrorType();
      }
    }

    return type;
  }

  // If we emitted a NoneAttr then convert it to a NoneType.  This is a
  // special case because "None" is both a value and a type, and defaults to a
  // value.
  if (isa<NoneAttr>(value.get()))
    return shared.getNoneType();

  emitError(node->getLoc(), "expected a type, not a value"), node->getRange();
  return shared.getTypeCheckErrorType();
}

/// Emit the specified expression as a condition, converting it to an MLIR I1
/// value that we can test directly.  This reports and error and returns null on
/// error.
DRValue ExprEmitter::emitExprConditionValueAsI1(ExprNode *condExpr) {
  AnyValue boolTmp; // we don't care about the intermediate Bool value.
  return emitConditionValueAsI1({emitExprRValue(condExpr), condExpr}, boolTmp);
}
