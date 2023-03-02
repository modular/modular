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
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ExprEmitter implementation
//===----------------------------------------------------------------------===//

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
RValue ExprEmitter::emitRValue(ASTExprAnd<AnyValue> value, ValueDest dest) {
  if (!value) // Already diagnosed error.
    return {};

  // If this is already an RValue, then we are done.
  if (auto rvRep = value.ir.getIfRValue())
    return emitResult(value.ir, value.expr, dest).getIfRValue();

  // Finally, if this is an LValue, emit a __clone__, a load for a primitive
  // MLIR type, or an error if neither approach works.
  auto pointer = value.ir.getIfLValue();
  assert(pointer);

  auto loc = value.expr->getLocation(*this);
  if (!builder) {
    emitError(loc, "cannot use a dynamic value in a parameter context")
        << value.expr->getRange();
    return {};
  }

  // If this is a primitive MLIR type, we can emit a direct load for it.
  ASTType rvalueType = value.ir.getRValueType();
  auto typeDecl = rvalueType.getDecl(shared);
  if (!typeDecl) {
    auto result =
        SRValue(builder->create<POP::LoadOp>(loc, pointer,
                                             /*alignment=*/std::nullopt));
    return emitResult(result, value.expr, dest).getIfRValue();
  }

  // Check for the presence of a valid __clone__ method.
  bool isErroneousDecl = false;
  OverloadSet clone(rvalueType, "__clone__", value.expr, isErroneousDecl,
                    shared);
  // If any error looking up __clone__ then the problem has been diagnosed
  // already.
  if (isErroneousDecl)
    return {};

  if (!clone.isNull()) {
    // Ok, cool we know it will succeed; do it.
    auto result = clone.emitCall(value, dest, value.expr,
                                 CallSyntax::kImplicitConvert, *this);
    if (!result)
      return {};
    assert(result.getIfRValue() &&
           "__clone__ is required to always return an RValue");
    return result.getIfRValue();
  }

  auto diag = emitError(loc, "cannot clone this value: ")
              << rvalueType << " doesn't implement '__clone__'"
              << value.expr->getRange();

  diag.attachNote(typeDecl->getLoc()) << "type declared here";
  return {};
}

CRValue ExprEmitter::emitCRValue(ASTExprAnd<AnyValue> value, ValueDest dest) {
  // If the value is an lvalue, convert it to an rvalue.
  value.ir = emitRValue(value, dest);

  if (!value)
    return {};

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (auto overloads = value.ir.getIfORValue())
    return overloads->emitAsCRValue(*this, ValueDest(), value.expr);

  assert(value.ir.getIfCRValue() && "Must be ORValue or CRValue");
  return value.ir.getIfCRValue();
}

SRValue ExprEmitter::emitSRValue(ASTExprAnd<AnyValue> value) {
  // If the value is an lvalue, convert it to an rvalue.
  value.ir = emitCRValue(value, ValueDest(/*SRValue never needs a dest*/));

  if (!value)
    return {};

  // If this is already an SRValue, emit this.
  if (auto rvalue = value.ir.getIfSRValue())
    return rvalue;

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (!builder) {
    emitError(value.expr->getLoc(),
              "cannot use a dynamic value in a parameter context")
        << value.expr->getRange();
    return {};
  }

  auto attr = value.ir.getIfPRValue().get();

  // If the value being materialized is itself parameterized, then we cannot
  // materialize it as an SSA value - there will be no way to bind parameters to
  // it.
  // TODO: We should have a general predicate from this provided by the KGEN
  // parameter utilities.
  if (auto signature = dyn_cast<SignatureType>(attr.getType())) {
    // If the value has any unbound parameters, they might be default arguments
    // or an variadic list that should be bound to an empty list.
    if (!signature.getInputParams().empty()) {
      InputParamBindings paramBindings;
      ssize_t incorrectBindingNo = 0;
      ASTType incorrectBindingExpectedType;
      auto bindingAttr = paramBindings.verifyBindings(
          signature.getInputParams(), "<<UNUSED>>", value.expr->getLoc(),
          incorrectBindingNo, incorrectBindingExpectedType, *this, nullptr,
          signature.hasParamVarargs());
      if (!bindingAttr) {
        // If it didn't work out, then it is an error because parameterized
        // values cannot be used in a dynamic context.
        emitError(value.expr->getLoc(),
                  "cannot use parameterized function of type ")
            << ASTType(attr.getType()) << " without binding all its parameters"
            << value.expr->getRange();
        return {};
      }

      // Apply whatever it produced to the attr of signature type to resolve the
      // remaining arguments.
      SmallVector<TypedAttr> bindOperands;
      bindOperands.push_back(attr);
      for (auto bind : bindingAttr)
        bindOperands.push_back(bind.getValue());
      // bindOperands.push_back(bindingAttr);
      attr = ParamOperatorAttr::get(POC::BindSignature, bindOperands);
    }

    // Reject unbound result parameters.
    if (!signature.getResultParams().empty()) {
      emitError(value.expr->getLoc(),
                "cannot use parameterized function with result parameters ")
          << ASTType(attr.getType()) << value.expr->getRange();
      return {};
    }
  }

  auto location = value.expr->getLocation(*this);
  // Materialize index integer constants as a special case.
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    if (intAttr.getType().isIndex()) {
      auto cst = builder->create<mlir::index::ConstantOp>(
          location, intAttr.getValue().getSExtValue());
      return SRValue(cst);
    }

  // Otherwise, emit a generalized parameter constant.
  return SRValue(builder->create<ParamConstantOp>(location, attr));
}

//===----------------------------------------------------------------------===//
// emitResult(ValueDest) Implementation
//===----------------------------------------------------------------------===//

/// Emit the specified value into the current destination if present.  This
/// accepts (and silently propagates) null values.
AnyValue ExprEmitter::emitResult(AnyValue value, const ExprNode *node,
                                 ValueDest dest) {
  if (!value)
    return {};

  // If we have an expression node destination, then we need to bind this value
  // to a pattern (aka "target" in Python internals nomenclature).
  if (const ExprNode *context =
          dyn_cast_or_null<const ExprNode *>(dest.representation))
    return context->emitExprResultIntoPattern({value, node}, *this);

  if (LValue lvalueDest = dyn_cast_or_null<LValue>(dest.representation))
    return emitExprResultIntoLValue({value, node}, lvalueDest);

  // Otherwise we have no prescribed context, use a default one.
  // TODO: Synthesize a vardecl if not an PRValue and the value has
  // memory-primary type.
  return value;
}

/// This method is used by node implementations of emitExprResultIntoPattern
/// to emit the result once they determine an lvalue to use.
AnyValue ExprEmitter::emitExprResultIntoLValue(ASTExprAnd<AnyValue> value,
                                               LValue dest) {
  // The final step of an assignment expression (`=`) converts the value into
  // a type that matches the destination and does a store.

  // TODO: This should be an initialization or reassignment, and needs to
  // call __clone__.
  AnyValue convertedVal = getAsExpectedType(value, dest.getRValueType(),
                                            ValueDest(), " in assignment");

  // Emit the RHS and coerce to the LHS type.
  SRValue rv = emitSRValue({convertedVal, value.expr});
  if (!rv)
    return {};

  // If everything worked out, store the resultant value into the lvalue for
  // the destination.
  auto loc = translateLocation(value.expr->getLoc());
  builder->create<POP::StoreOp>(loc, rv, dest,
                                /*alignment=*/std::nullopt);

  return MRValue(dest);
}

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

/// This helper emits a named method call with the provided `argValues`, where
/// the first arg is the receiver of the call. This emits an error if the
/// call is invalid and returns null.  The argValues list may not be empty.
///
/// `callNode` is the call like expression (e.g. a CallNode, binary operator,
/// etc) that results in the call, or potentially a random value that is being
/// fed into an implicit conversion.  This should only be used for location
/// information.
AnyValue ExprEmitter::emitNamedMethodCall(
    StringRef methodName, ArrayRef<ASTExprAnd<AnyValue>> argValues,
    ValueDest dest, CallSyntax syntax, const ExprNode *callNode) {
  assert(!argValues.empty() && "Cannot emit a method call without a receiver!");
  ASTType type = argValues.front().ir.getRValueType();
  bool isErroneousDecl = false;
  OverloadSet callee(type, methodName, callNode, isErroneousDecl, shared);

  // If the type doesn't have the specified method, emit an error.
  if (callee.isNull()) {
    if (isErroneousDecl)
      return {};
    auto diag = emitError(callNode->getLoc(), "")
                << type << " does not implement the '" << methodName
                << "' method";
    switch (syntax) {
    default:
      break;
    case CallSyntax::kMethodCall:
      diag << argValues[0].expr->getRange();
      break;
    case CallSyntax::kOperator:
      diag << argValues[0].expr->getRange();
      break;
    case CallSyntax::kReversedOperator:
      diag << argValues[1].expr->getRange();
      break;
    }
    return {};
  }

  return callee.emitCall(argValues, dest, callNode, syntax, *this);
}

/// Convert the specified value to the expected type, invoking implicit
/// conversions if necessary.  On error, this diagnoses it and returns null.
AnyValue ExprEmitter::getAsExpectedType(ASTExprAnd<AnyValue> value,
                                        ASTType expectedType, ValueDest dest,
                                        std::function<void()> errorHandler) {
  if (!value)
    return {};

  bool noConversionNeeded =
      ASTType(value.ir.getRValueType()).isEqualCanon(expectedType);

  // If this happens to be an lvalue coming in, convert to rvalue.  Emit into
  // dest if no conversion is needed.
  value.ir = emitRValue(value, noConversionNeeded ? dest : ValueDest());

  // If the value handed to is us already erroneous, don't diagnose anything.
  if (!value)
    return {};

  // If the type is already an exact match, then we are done.
  if (noConversionNeeded)
    return value.ir;

  // Check to see if we can invoke an __new__ method to convert it.
  bool isErroneousDecl = false;
  OverloadSet callee(expectedType, "__new__", value.expr, isErroneousDecl,
                     shared);
  if (callee.isNull()) {
    if (!isErroneousDecl)
      errorHandler();
    return {};
  }

  // If we have at least one candidate, we check to see if any of them can
  // work. We disable implicit conversions though, to prevent converting
  // T -> S -> U in one step.
  if (failed(callee.filterOverloadSet(
          {value}, CallSyntax::kImplicitConvert, value.expr,
          /*allowImplicitConversions=*/false,
          /*emitDiagnosticOnFailure=*/false, *this))) {
    errorHandler();
    return {};
  }

  // Ok, cool we know it will succeed; do it.
  return callee.emitCall(value, dest, value.expr, CallSyntax::kImplicitConvert,
                         *this);
}

AnyValue ExprEmitter::getAsExpectedType(ASTExprAnd<AnyValue> value,
                                        ASTType expectedType, ValueDest dest,
                                        const Twine &errorSuffix) {
  auto errorHandler = [&]() {
    if (!isa<TypeCheckErrorType>(value.ir.getType()) &&
        !isa<TypeCheckErrorType>(expectedType.mlirType))
      emitError(value.expr->getLoc())
          << ASTType(value.ir.getType()) << " value cannot be converted to "
          << expectedType << errorSuffix << value.expr->getRange();
  };
  return getAsExpectedType(value, expectedType, dest, std::move(errorHandler));
}

/// Emit the specified expression as a condition, converting it to an MLIR I1
/// value that we can test directly, and also returning the intermediate
/// result of calling `__bool__` (which is typically a Bool or object type, but
/// not guaranteed).  This reports and error and returns null on error.
RValue ExprEmitter::emitConditionValueAsI1(ASTExprAnd<AnyValue> value,
                                           AnyValue &boolResult) {
  if (!value.ir)
    return {};

  boolResult = value.ir;

  // If this is already an 'i1', then we're done.
  if (value.ir.getType().isInteger(1))
    return emitRValue(value, ValueDest());

  // TODO: Python manual includes this off-hand comment:
  // Also, an object that doesn’t define a __bool__() method and whose __len__()
  // method returns zero is considered to be false in a Boolean context.

  // Check for the presence of a __lit_bool method.  If it exists, we can avoid
  // a redundant call to __bool__ for Bool types.
  bool isErroneousDecl = false;
  if (!OverloadSet(value.ir.getType(), "__lit_bool", value.expr,
                   isErroneousDecl, shared)) {
    // Use the __bool__ method to convert the user defined type to
    // something that is a Bool or other type that implements __lit_bool.
    boolResult =
        emitNamedMethodCall("__bool__", {{value.ir, value.expr}}, ValueDest(),
                            CallSyntax::kImplicitConvert, value.expr);
    if (!boolResult)
      return {};
  }

  // Then we use __lit_bool to convert to an i1 value.
  AnyValue litBoolCall =
      emitNamedMethodCall("__lit_bool", {{boolResult, value.expr}}, ValueDest(),
                          CallSyntax::kImplicitConvert, value.expr);

  /// TODO(memory-primary):˙should emit into a ValueDest slot when known.
  return emitRValue({litBoolCall, value.expr}, ValueDest());
}

//===----------------------------------------------------------------------===//
// ExprEmitter implementation
//===----------------------------------------------------------------------===//

/// This helper emits the specified value rep as an RValue.
RValue ExprEmitter::emitExprRValue(const ExprNode *node, ValueDest dest) {
  assert(node && "cannot emit a null node");
  return emitRValue(
      {node->emitIR(*this,
                    // TODO(memory-primary): Value dest composition.
                    ValueDest()),
       node},
      dest);
}

/// This helper emits the specified value rep as an CRValue.
CRValue ExprEmitter::emitExprCRValue(const ExprNode *node, ValueDest dest) {
  assert(node && "cannot emit a null node");
  return emitCRValue({node->emitIR(*this, dest), node}, dest);
}

/// This helper emits the specified value rep as an SRValue, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
SRValue ExprEmitter::emitExprSRValue(const ExprNode *node) {
  assert(node && "cannot emit a null node");
  return emitSRValue({node->emitIR(*this, ValueDest(/*SRValue*/)), node});
}

/// This helper emits the specified expression as a parameter value, diagnosing
/// the problem if the expression is only valid as a runtime value.  This
/// returns null if emission fails.
PRValue ExprEmitter::emitExprPRValue(const ExprNode *node, ASTType resultType,
                                     const Twine &errorSuffix) {
  // Clear the builder to indicate that an PRValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder);
  builder.reset();

  // Emit the expression.
  AnyValue rep = emitExprCRValue(node, ValueDest(/*knownPRValue*/));

  // If we had an expected type, do a conversion.
  if (resultType)
    rep = getAsExpectedType({rep, node}, resultType,
                            ValueDest(/*knownPRValue*/), errorSuffix);

  if (!rep)
    return {};

  // If this is a parameter, return it.
  if (auto value = rep.getIfPRValue())
    return value;

  // Otherwise diagnose this as "not a parameter".
  emitError(node->getLoc(), "cannot use a dynamic value") << errorSuffix;
  return {};
}

/// Emit the specified expression as an LValue which can be loaded and stored.
/// If contextualType is non-null, then an implicitly declared LValue will be
/// assigned that type.
///
/// This diagnoses the expression with the specified message if it isn't a
/// valid LValue.
LValue ExprEmitter::emitExprLValue(SMLoc loc, const ExprNode *node,
                                   ValueDest dest, const Twine &message) {
  AnyValue anyValue = node->emitIR(*this, dest);
  if (!anyValue)
    return {}; // Error already diagnosed.
  if (LValue lValue = anyValue.getIfLValue())
    return lValue;
  emitError(loc, message) << node->getRange();
  return {};
}

/// This helper emits the specified expression tree as a type, e.g. turning
/// "Int" into the type for it.  This emits an error and returns null on
/// failure.
ASTType ExprEmitter::emitExprType(const ExprNode *node) {
  auto value = emitExprPRValue(node, {}, " in type specification");
  if (!value)
    return {};

  // If this emitted a type, we can lower it.
  if (auto type = value.getIfTypeValue()) {
    // Verify that all of the parameters for this type are bound.  We allow
    // PRValues to refer to parameteric type, but anything calling `emitType`
    // can only handle fully bound types.
    auto *decl = type.getDecl(shared);
    if (!decl) // MLIR types are never parameterized.
      return type;

    auto structDecl = cast<StructDeclOp>(*decl);

    // Build up a InputParamBindings set to validate and check the bindings.
    InputParamBindings paramBindings;
    for (auto binding : type.getParamBindings())
      paramBindings.add(binding);

    // Check the bindings.
    ssize_t incorrectBindingNo = 0;
    ASTType incorrectBindingExpectedType;
    auto bindingAttr = paramBindings.verifyBindings(
        structDecl.getInputParamDeclsAttr(), structDecl.getName(),
        node->getLoc(), incorrectBindingNo, incorrectBindingExpectedType, *this,
        structDecl, structDecl.getParamVarargs());
    if (!bindingAttr)
      return {};

    // If verifyBindings changed the bindings set, then we may have had an
    // empty varargs list or something.  Rebind the DeclRefType.
    if (bindingAttr != type.getParamBindings()) {
      auto symbol = cast<DeclRefType>(type.mlirType).getSymbol();
      type = DeclRefType::get(symbol, bindingAttr);
    }

    return type;
  }

  // If we emitted a NoneAttr then convert it to a NoneType.  This is a
  // special case because "None" is both a value and a type, and defaults to a
  // value.
  if (isa<NoneAttr>(value.get()))
    return shared.getNoneType();

  emitError(node->getLoc(), "expected a type, not a value"), node->getRange();
  return {};
}

/// Emit the specified expression as a condition, converting it to an MLIR I1
/// value that we can test directly.  This reports and error and returns null on
/// error.
RValue ExprEmitter::emitExprConditionValueAsI1(const ExprNode *condExpr) {
  AnyValue boolTmp; // we don't care about the intermediate Bool value.
  return emitConditionValueAsI1({emitExprRValue(condExpr), condExpr}, boolTmp);
}

SRValue ExprEmitter::emitBoxedIntAsPopScalar(Value numberValue,
                                             const ExprNode *source) {
  if (numberValue.getType().isIndex()) {
    return SRValue(builder->create<POP::CastFromBuiltinOp>(
        translateLocation(source->getLoc()),
        POP::SIMDType::get(builder->getContext(), 1,
                           KGENDType(KGENDType::index)),
        numberValue));
  }
  assert(numberValue.getType().isa<KGEN::DeclRefType>() &&
         "number value must be a struct");
  AnyValue index =
      emitNamedMethodCall("__as_mlir_index", {{SRValue(numberValue), source}},
                          ValueDest(), CallSyntax::kImplicitConvert, source);
  if (!index) {
    return {};
  }
  auto popscalar = builder->create<POP::CastFromBuiltinOp>(
      translateLocation(source->getLoc()),
      POP::SIMDType::get(builder->getContext(), 1, KGENDType(KGENDType::index)),
      index.getIfSRValue());
  return SRValue(popscalar);
}
