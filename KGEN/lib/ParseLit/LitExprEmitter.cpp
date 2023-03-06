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
#include "LitParameterEvaluator.h"
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

const char *LIT::getContextMessage(ExprContext context) {
  switch (context) {
  case EC_Silent:
    return "";

  case EC_VarInit:
    return " in 'var' initializer";
  case EC_LetInit:
    return " in 'let' initializer";
  case EC_Assignment:
    return " in assignment";
  case EC_Type:
    return " in type specification";
  case EC_AttributeRefBase:
    return " in attribute base value";
  case EC_AliasValue:
    return " in alias value";
  case EC_CallArgValue:
    return " in call argument";
  case EC_CallCalleeValue:
    return " in callee";
  case EC_TypeParamValue:
    return " in type parameter";
  case EC_CallParamValue:
    return " in call parameter";
  case EC_OperatorOperandValue:
    return " in operator argument";
  case EC_FieldInitValue:
    return " in field initializer";
  case EC_DefaultArgument:
    return " in default argument";
  case EC_BoolCondition:
    return " in boolean condition";
  case EC_ForIterator:
    return " in for iterator expression";
  case EC_RaiseValue:
    return " in raised value";
  case EC_ReturnResultParamList:
    return " in return parameter";
  case EC_ReturnValue:
    return " in return value";
  case EC_MLIRMagic:
    return " in MLIR magic";
  }
}

//===----------------------------------------------------------------------===//
// ValueDest implementation
//===----------------------------------------------------------------------===//

ValueDest::ValueDest(VarLetDeclOp dest, ExprContext context)
    : representation(dest.getOperation()), context(context) {}

/// If this value destination has a known type, e.g. "var x : Int = 42" or
/// "x = 42", return it.  If not (e.g. _ = 42) then return null.
ASTType ValueDest::getTypeIfKnown() const {
  if (representation.isNull())
    return {};

  // If we have an lvalue already specified, return it.
  if (LValue lvalue = dyn_cast<LValue>(representation))
    return lvalue.getRValueType();

  // If we just have a contextual type, return it.
  if (ASTType type = dyn_cast<ASTType>(representation))
    return type;

  // TODO: Infer from expression target like:
  //   var x : FunctionType; x = overloadedFn

  // Can't infer from an Operation*, since it is inferring from the initializer.
  return {};
}

/// Project a ValueDest into an lvalue with the specified underlying (RValue)
/// type.  This uses 'resultType' for inference when the ValueDest is untyped
/// (e.g. `var x = expr`), but may return an LValue of another type when the
/// dest is typed (e.g. `var x : F32 = 1`).
///
/// This consumes the ValueDest.
LValue ValueDest::takeLValueForResult(SMLoc loc, ASTType resultType,
                                      ExprEmitter &emitter) {
  // If we have an lvalue already specified, return it.
  if (LValue lvalue = dyn_cast_or_null<LValue>(representation)) {
    representation = nullptr; // Consumed!
    return lvalue;
  }

  // If we have an expression node destination, then we need to bind this value
  // to a pattern (aka "target" in Python internals nomenclature).
  if (const ExprNode *target =
          dyn_cast_or_null<const ExprNode *>(representation)) {
    representation = nullptr; // Consumed!
    return target->getLValueForResult(resultType, emitter);
  }

  // If we are inferring the type for a var or let declaration, do that.
  if (auto *opDest = dyn_cast_or_null<Operation *>(representation)) {
    representation = nullptr; // Consumed!

    auto varOp = cast<VarLetDeclOp>(opDest);
    assert(isa<UnresolvedType>(varOp.getType().getResolvedElementType()) &&
           "Cannot resolve an already-resolved vardecl");
    varOp.getResult().setType(POP::PointerType::get(resultType));
    return LValue(varOp);
  }

  // Finally, if no destination specifies otherwise, we synthesize a new LValue
  // on demand.
  if (!emitter.builder) {
    representation = nullptr; // Consumed!
    emitter.emitError(
        loc, "cannot synthesize lvalue in parameter expression context");
    return {};
  }

  // If we're generating a memory location, use a required type if present or
  // the value type if not. TODO(autopromotion).
  ASTType slotType = resultType;
  if (auto requiredType = dyn_cast_or_null<ASTType>(representation)) {
    slotType = requiredType;
    representation = nullptr; // Consumed!
  }

  Type declIRType = POP::PointerType::get(slotType);
  auto nameAttr = StringAttr::get(emitter.getContext(), "<anonymous>");
  // We model this as an immutable let value with a separately stored
  // initializer.  We return an LValue for it because this method is used for
  // the initialization.
  return LValue(emitter.builder->create<VarLetDeclOp>(
      emitter.translateLocation(loc), declIRType, nameAttr, /*isVar*/ 0));
}

//===----------------------------------------------------------------------===//
// ExprEmitter implementation
//===----------------------------------------------------------------------===//

RValue ExprEmitter::emitRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  ValueDest dest(resultType, context);
  if (auto result = emitRValue(value, dest))
    return result;
  dest.resetForError();
  return {};
}

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
RValue ExprEmitter::emitRValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
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

  auto emitNoCloneError = [&]() {
    auto diag = emitError(loc, "cannot clone this value: ")
                << rvalueType << " doesn't implement '__clone__'"
                << value.expr->getRange();
    diag.attachNote(typeDecl->getLoc()) << "type declared here";
  };

  // Check for the presence of a valid __clone__ method.
  OverloadSet clone(rvalueType, "__clone__", value.expr,
                    CallSyntax::kImplicitConvert, shared, emitNoCloneError);
  if (clone.isNull())
    return {};

  // Ok, cool we know it will succeed; do it.
  auto result = clone.emitCall(value, dest, *this);
  if (!result)
    return {};
  assert(result.getIfRValue() &&
         "__clone__ is required to always return an RValue");
  return result.getIfRValue();
}

CRValue ExprEmitter::emitCRValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  // If the value is an lvalue, convert it to an rvalue.
  value.ir = emitRValue(value, dest);
  if (!value)
    return {};

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (auto overloads = value.ir.getIfORValue())
    return overloads->emitAsCRValue(*this, dest);

  assert(value.ir.getIfCRValue() && "Must be ORValue or CRValue");
  return value.ir.getIfCRValue();
}

/// This helper emits the specified value as a SRValue which has an SSA
/// value representation, materializing PRValues and loading LValues as
/// needed.  This returns null if emission fails, and should never be used with
/// values that are memory-primary.
SRValue ExprEmitter::emitSRValue(ASTExprAnd<AnyValue> value,
                                 ExprContext context, ASTType resultType) {
  // Emit using resultType if present, and eliminate LValue/ORValue's.
  ValueDest dest(resultType, context);
  value.ir = emitCRValue(value, dest);
  if (!value) {
    dest.resetForError();
    return {};
  }

  // If this is already an SRValue, return it.
  if (auto rvalue = value.ir.getIfSRValue())
    return rvalue;

  // Make sure this method isn't getting called inappropriately.
  assert(value.ir.getRValueType().isRegisterPrimary(value.expr->getLoc(),
                                                    shared) &&
         "cannot emit a memory-primary type as an SRValue");

  // If this is an MRValue containing a loadable value, use emitSRValue from an
  // LValue to load it and emit the proper clone call.
  if (auto mrValue = value.ir.getIfMRValue())
    return emitSRValue({LValue(mrValue), value.expr}, context);

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (!builder) {
    emitError(value.expr->getLoc(), "cannot use a dynamic value")
        << getContextMessage(context) << value.expr->getRange();
    return {};
  }

  auto attr = value.ir.getIfPRValue().get();
  assert(attr && "must be PRValue if register primary and not SRValue");

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

/// This helper emits the specified expression as a parameter value, diagnosing
/// the problem if the expression is only valid as a runtime value.  This
/// returns null if emission fails.
PRValue ExprEmitter::emitPRValue(ASTExprAnd<AnyValue> value,
                                 ExprContext context, ASTType resultType) {
  if (!value)
    return {};

  // Clear the builder to indicate that an PRValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder);
  builder.reset();

  // If there is a result type, coerce before checking for PRValue.
  if (resultType) {
    value.ir = emitRValue(value, context, resultType);
    if (!value)
      return {};
  }

  // If this is a parameter, return it.
  if (auto result = value.ir.getIfPRValue())
    return result;

  // Otherwise diagnose this as "not a parameter".
  emitError(value.expr->getLoc(), "cannot use a dynamic value")
      << getContextMessage(context);
  return {};
}

//===----------------------------------------------------------------------===//
// emitResult(ValueDest) Implementation
//===----------------------------------------------------------------------===//

/// When emitting a result value, attempt to "refine" the value type by
/// evaluating 'apply' expressions in its type. Rebind the value if the type can
/// be further specialized.
static AnyValue refineResultValue(AnyValue value, SMLoc loc,
                                  ExprEmitter &emitter) {
  // Only LValues and CRValues can be specialized. ORValues don't have a type.
  if (value.getIfORValue())
    return value;

  LitParameterEvaluator evaluator(emitter.getDeclResolver());
  Type refinedType = evaluator.refineType(value.getType());
  if (refinedType == value.getType())
    return value;

  // Materialize a parameter rebind.
  if (auto prvalue = value.getIfPRValue())
    return PRValue(
        ParamOperatorAttr::get(POC::Rebind, prvalue.get(), refinedType));

  // Materialize a rebind operation.
  auto rebind = [&](Value value) -> Value {
    return emitter.builder->create<RebindOp>(emitter.translateLocation(loc),
                                             refinedType, value);
  };
  if (auto lvalue = value.getIfLValue())
    return LValue(rebind(lvalue));
  if (auto mrvalue = value.getIfMRValue())
    return MRValue(rebind(mrvalue));
  return SRValue(rebind(value.getIfSRValue()));
}

/// Emit a conversion from the specified value to the specified destination
/// type, plopping the value into the designated value destination.  We know the
/// types mismatch so the conversion must be emitted.
static AnyValue emitConversionTo(CRValue value, const ExprNode *expr,
                                 ASTType expectedType, ValueDest &dest,
                                 ExprEmitter &emitter) {

  auto errorHandler = [&]() {
    if (dest.getContext() == EC_Silent ||
        isa<TypeCheckErrorType>(value.getType().mlirType) ||
        isa<TypeCheckErrorType>(expectedType.mlirType))
      return;

    auto diag = emitter.emitError(expr->getLoc())
                << value.getRValueType() << " value cannot be converted to "
                << expectedType << getContextMessage(dest.getContext())
                << expr->getRange();
  };

  // Check to see if we can invoke an __new__ method to convert it.
  OverloadSet callee(expectedType, "__new__", expr,
                     CallSyntax::kImplicitConvert, emitter.shared,
                     errorHandler);
  if (callee.isNull())
    return {};

  // If we have at least one candidate, we check to see if any of them can
  // work. We disable implicit conversions though, to prevent converting
  // T -> S -> U in one step.
  if (failed(callee.filterOverloadSet({{value, expr}},
                                      /*allowImplicitConversions=*/false,
                                      /*emitDiagnosticOnFailure=*/false,
                                      emitter))) {
    errorHandler();
    return {};
  }

  // Ok, cool we know it will succeed; do it.
  return callee.emitCall({{value, expr}}, dest, emitter);
}

/// Emit the specified value into the current destination if present.  This
/// accepts (and silently propagates) null values.
///
/// Note that the `value` provided here may require an implicit conversion into
/// the destination slot, so the input may be memory-primary and result be
/// register-primary (and visa-versa).
AnyValue ExprEmitter::emitResult(AnyValue value, const ExprNode *node,
                                 ValueDest &dest) {
  if (!value)
    return {};

  // Attempt to further specialize the result value.
  value = refineResultValue(value, node->getLoc(), *this);

  // If no destination is specified, then we can propagate the value directly.
  if (!dest.isSpecified())
    return value;

  // OK, if there is a destination specified, handle them by converging the set
  // of value types we have.

  // If the value is an LValue, then emit a load into the destination using
  // emitRValue to do the heavy lifting.
  if (value.getIfLValue())
    return emitRValue({value, node}, dest);

  // We cannot infer from an unresolved overload set, collapse into a concrete
  // value with a concrete type if we can.
  if (auto overloads = value.getIfORValue()) {
    // ORValues always resolve to PRValues, which are never memory resident.
    // Concretize it so we get something with a type.
    return overloads->emitAsCRValue(*this, dest);
  }

  // We know we have a CRValue now.

  // If there is a known type for the destination but the value disagrees, emit
  // an implicit conversion directly into the destination.  This keeps values in
  // registers and avoids a "convert + clone" pair for memory->memory
  // conversions.
  ASTType requiredType = dest.getTypeIfKnown();
  if (requiredType && !requiredType.isEqualCanon(value.getRValueType()))
    return emitConversionTo(value.getIfCRValue(), node, requiredType, dest,
                            *this);

  // If the destination is just a required type, then we now know it must agree
  // and therefore don't need to do anything more.
  if (isa<ASTType>(dest.representation)) {
    dest = ValueDest(); // Resolved the ValueDest;
    return value;
  }

  // Eliminate the MRValue case by emitting a __clone__ call into the
  // destination using LValue -> RValue conversion.
  if (auto mrValue = value.getIfMRValue())
    return emitRValue({LValue(mrValue), node}, dest);

  // We know we have a SRValue or PRValue, and the destination is some kind of
  // LValue.  Emit the value and figure out where to store it.
  auto destLV =
      dest.takeLValueForResult(node->getLoc(), value.getRValueType(), *this);
  if (!destLV)
    return {};

  SRValue rv =
      emitSRValue({value, node}, dest.getContext(), destLV.getRValueType());
  if (!rv)
    return {};

  auto loc = translateLocation(node->getLoc());
  builder->create<POP::StoreOp>(loc, rv, destLV, /*alignment=*/std::nullopt);
  return MRValue(destLV);
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
    ValueDest &dest, CallSyntax syntax, const ExprNode *callNode) {
  assert(!argValues.empty() && "Cannot emit a method call without a receiver!");
  ASTType type = argValues.front().ir.getRValueType();

  auto emitNoMethodError = [&]() {
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
  };

  OverloadSet callee(type, methodName, callNode, syntax, shared,
                     emitNoMethodError);

  // If the type doesn't have the specified method, emit an error.
  if (callee.isNull())
    return {};

  return callee.emitCall(argValues, dest, *this);
}

/// Return true if 'value' may be implicitly converted to 'requiredType'
/// by invoking (one level of) conversion operations.  This does not generate
/// any IR.
bool ExprEmitter::canImplicitlyConvertToType(ASTExprAnd<AnyValue> value,
                                             ASTType requiredType) {
  // If it already matches, then we're done.
  if (value.ir.getRValueType().isEqualCanon(requiredType))
    return true;

  // Otherwise, check to see if we can do an implicit conversion by invoking a
  // `__new__` method on the expected type.
  OverloadSet callee(requiredType, "__new__", value.expr,
                     CallSyntax::kImplicitConvert, shared,
                     /*no error emission on failure */ {});

  // If there are no viable candidates for the implicit conversion, we fail.
  if (!callee)
    return false;

  // If we have at least one candidate, we check to see if any of them can
  // work. We disable implicit conversions though, to prevent converting
  // T -> S -> U in one step.

  // This needs to call filterOverloadSet manually because we cannot allow
  // implicit conversions here.
  return succeeded(callee.filterOverloadSet({value},
                                            /*allowImplicitConversions=*/false,
                                            /*emitDiagnosticOnFailure=*/false,
                                            *this));
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
  if (value.ir.getType().mlirType.isInteger(1))
    return emitRValue(value, ValueDest::none());

  // TODO: Python manual includes this off-hand comment:
  // Also, an object that doesn’t define a __bool__() method and whose __len__()
  // method returns zero is considered to be false in a Boolean context.

  // Check for the presence of a __lit_bool method.  If it exists, we can avoid
  // a redundant call to __bool__ for Bool types.
  if (!OverloadSet(value.ir.getType(), "__lit_bool", value.expr,
                   CallSyntax::kImplicitConvert, shared,
                   [&]() { /*no error*/ })) {
    // Use the __bool__ method to convert the user defined type to
    // something that is a Bool or other type that implements __lit_bool.
    boolResult = emitNamedMethodCall("__bool__", {{value.ir, value.expr}},
                                     ValueDest::none(),
                                     CallSyntax::kImplicitConvert, value.expr);
    if (!boolResult)
      return {};
  }

  // Then we use __lit_bool to convert to an i1 value.
  AnyValue litBoolCall = emitNamedMethodCall(
      "__lit_bool", {{boolResult, value.expr}}, ValueDest::none(),
      CallSyntax::kImplicitConvert, value.expr);

  return emitRValue({litBoolCall, value.expr}, ValueDest::none());
}

//===----------------------------------------------------------------------===//
// ExprEmitter implementation
//===----------------------------------------------------------------------===//

/// This helper emits the specified value rep as an RValue.
RValue ExprEmitter::emitExprRValue(const ExprNode *node, ValueDest &dest) {
  assert(node && "cannot emit a null node");
  if (dest.isSpecified()) {
    auto result = node->emitIR(dest, *this);
    assert((!result || result.getIfRValue()) &&
           "destination provided should force an RValue result");
    return result.getIfRValue();
  }

  // If we have no destination specified, emit it and load the result if it is
  // an RValue.
  return emitRValue({node->emitIR(dest, *this), node}, dest);
}

/// This helper emits the specified value rep as an CRValue.
CRValue ExprEmitter::emitExprCRValue(const ExprNode *node, ValueDest &dest) {
  assert(node && "cannot emit a null node");
  return emitCRValue({node->emitIR(dest, *this), node}, dest);
}

/// This helper emits the specified value rep as an SRValue, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
SRValue ExprEmitter::emitExprSRValue(const ExprNode *node, ExprContext context,
                                     ASTType resultType) {
  assert(node && "cannot emit a null node");
  ValueDest dest(resultType, context);
  if (SRValue result = emitSRValue({node->emitIR(dest, *this), node}, context))
    return result;
  dest.resetForError();
  return {};
}

/// This helper emits the specified expression as a parameter value, diagnosing
/// the problem if the expression is only valid as a runtime value.  This
/// returns null if emission fails.
PRValue ExprEmitter::emitExprPRValue(const ExprNode *node, ExprContext context,
                                     ASTType resultType) {
  // Clear the builder to indicate that an PRValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder);
  builder.reset();

  // Emit the expression using the contextual type if present.
  ValueDest dest(resultType, context);
  auto rep = emitExprCRValue(node, dest);
  if (!rep) {
    dest.resetForError();
    return {};
  }

  return emitPRValue({rep, node}, context);
}

/// Emit the specified expression as an LValue which can be loaded and stored.
/// If contextualType is non-null, then an implicitly declared LValue will be
/// assigned that type.
///
/// This diagnoses the expression with the specified message if it isn't a
/// valid LValue.
LValue ExprEmitter::emitExprLValue(SMLoc loc, const ExprNode *node,
                                   const Twine &message) {
  AnyValue anyValue = node->emitIR(ValueDest::none(), *this);
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
  auto value = emitExprPRValue(node, EC_Type);
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
  return emitConditionValueAsI1(
      {emitExprRValue(condExpr, ValueDest::none()), condExpr}, boolTmp);
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
  AnyValue index = emitNamedMethodCall(
      "__as_mlir_index", {{SRValue(numberValue), source}}, ValueDest::none(),
      CallSyntax::kImplicitConvert, source);
  if (!index) {
    return {};
  }
  auto popscalar = builder->create<POP::CastFromBuiltinOp>(
      translateLocation(source->getLoc()),
      POP::SIMDType::get(builder->getContext(), 1, KGENDType(KGENDType::index)),
      index.getIfSRValue());
  return SRValue(popscalar);
}
