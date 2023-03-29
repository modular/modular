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
#include "LitExprNodes.h"
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
  case EC_Unknown:
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
    return " in alias initializer";
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
  case EC_InplaceBinOpDest:
    return " for in-place operator destination";
  case EC_FieldInitValue:
    return " in field initializer";
  case EC_DefaultArgument:
    return " in default argument";
  case EC_DefArgumentShadow:
    return " in def argument shadow";
  case EC_BoolCondition:
    return " in boolean condition";
  case EC_BoolParamCondition:
    return " in '@parameter if' condition";
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
  case EC_TopLevelStmt:
    return " in expression statement";
  case EC_ListField: // [x, y]
    return " in list field initializer";
  case EC_TupleElement: // (x, y)
    return " in tuple element";
  case EC_SubscriptBase: // x[y]
    return " in subscript base";
  case EC_ParameterList: // something[paramValue]
    return " in parameter list";
  }
}

//===----------------------------------------------------------------------===//
// ValueDest implementation
//===----------------------------------------------------------------------===//

ValueDest::ValueDest(VarLetDeclOp dest, ExprContext context)
    : representation(dest.getOperation()), context(context) {}

/// Inspect the ValueDest to see if it implies a specific type for the value
/// being computed, emiting ExprNode targets if present to get their implied
/// type if present.  This returns null if there is no implied type.
///
/// This may be used in concrete value context with a known type (in which
/// case 'existingValueType' will hold the known value type) or in ambiguous
/// cases where this is being used to resolve a type (in which case it will be
/// null).
///
/// Note that this will mutate the ValueDest if it is an ExprNode, turning it
/// into an LValue to store to.
ASTType ValueDest::resolveImpliedType(SMLoc loc, Type existingValueType,
                                      ExprEmitter &emitter) {
  // These have no implied type.
  if (isa<NullRepresentation, LValueBufferTaken, Operation *>(representation))
    return {};

  // If we just have a contextual type, return it.
  if (ASTType type = dyn_cast<ASTType>(representation))
    return type;

  assert(!isa<LValueInitializerType>(representation) &&
         "LValueInitializerType should be resolved before this");

  // If we have an un-emitted expression, emit it using our existintValueType to
  // get an LValue.
  if (auto *expr = dyn_cast<const ExprNode *>(representation)) {
    // If we have a contextual type available, pass that down to the emitter so
    // implicitly declared variables and discard patterns can know their type.
    ValueDest dest;
    if (existingValueType)
      dest = ValueDest(LValueInitializerType{existingValueType}, context);

    /// Emit the target as an LValue to understand what we're assigning into. If
    /// this fails, it will produce an error.
    LValue exprLValue = emitter.emitExprLValue(expr, dest);
    if (!exprLValue) {
      dest.resetForError();
      representation = NullRepresentation();
      return {};
    }
    representation = exprLValue;
  }

  // If we have an lvalue already specified, return it.
  return cast<LValue>(representation).getRValueType();
}

/// Project a ValueDest into an lvalue with the specified underlying (RValue)
/// type.
///
/// When `allowIncompatibleTypes` is true, the method is allowed to return an
/// LValue of a different type when the underlying storage requires this. This
/// is a guarantee from the caller that it is prepared to handle a type
/// conversion on its side, eliminating a temporary buffer in register-passable
/// cases like `var x : F32 = 1`.
///
/// When `allowIncompatibleTypes` is false, this always returns an LValue of
/// the requested type, which may return a temporary buffer.  In this case it
/// will not consume the ValueDest, so any user should reemit the ultimate
/// value through it with emitResult.
LValue ValueDest::getLValueForResult(SMLoc loc, ASTType resultType,
                                     bool allowIncompatibleTypes,
                                     bool requireSLValue,
                                     ExprEmitter &emitter) {
  // If we are inferring the type for a var or let declaration, then we can
  // always succeed and consume this ValueDest.
  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    representation = LValueBufferTaken(); // Buffer used!

    auto varOp = cast<VarLetDeclOp>(opDest);
    assert(isa<UnresolvedType>(varOp.getType().getResolvedElementType()) &&
           "Cannot resolve an already-resolved vardecl");
    varOp.getResult().setType(POP::PointerType::get(resultType));
    return SLValue(varOp);
  }

  // Otherwise, we have one of a few cases where we can produce an LValue but
  // it may have the wrong type.  The client may be cool with this (when
  // allowIncompatibleTypes is true), but if not we generate a new temporary
  // buffer.

  // If we have an expression node destination, then we need to bind this
  // value to a pattern (aka "target" in Python internals nomenclature).
  if (const ExprNode *target = dyn_cast<const ExprNode *>(representation)) {
    ValueDest dest(LValueInitializerType{resultType}, getContext());
    if (LValue lValue = emitter.emitExprLValue(target, dest)) {
      representation = lValue;
    } else {
      dest.resetForError();
      representation = NullRepresentation(); // Consumed!
    }
  }

  // If we have an lvalue already specified, return it.
  if (LValue lValue = dyn_cast<LValue>(representation)) {
    // If asking for a buffer of the type we happen to have, or if the client
    // doesn't care if it matches, then we can directly return it.
    if (allowIncompatibleTypes ||
        lValue.getRValueType().isEqualCanon(resultType)) {
      // If the client requires a stored LValue and we don't have one, don't
      // consume it.
      if (!requireSLValue || lValue.getIfSLValue()) {
        representation = LValueBufferTaken(); // Buffer taken!
        return lValue;
      }
    }

    // Otherwise, create a temporary buffer.
  }

  // Finally, if no destination specifies otherwise, we synthesize a new
  // LValue on demand.
  if (!emitter.builder) {
    representation = NullRepresentation();
    emitter.emitError(loc, "cannot synthesize lvalue")
        << getContextMessage(emitter.paramContext);
    return {};
  }

  // If we're generating a memory location, use a required type if present or
  // the value type if not.
  // TODO(autopromotion).
  ASTType slotType = resultType;
  if (auto requiredType = dyn_cast_or_null<ASTType>(representation)) {
    if (allowIncompatibleTypes || requiredType.isEqualCanon(slotType))
      slotType = requiredType;
  }

  Type declIRType = POP::PointerType::get(slotType);
  auto nameAttr = StringAttr::get(emitter.getContext(), "anonymous*");
  // We model this as an immutable let value with a separately stored
  // initializer.  We return an LValue for it because this method is used for
  // the initialization.
  return SLValue(emitter.builder->create<VarLetDeclOp>(
      emitter.translateLocation(loc), declIRType, nameAttr, /*isVar*/ 0));
}

/// Return an SLValue for this destination of the specified type that we can
/// initialize.  This uses and consumes the destination if it matches the type
/// of the value dest.
SLValue ValueDest::getSLValueForResult(SMLoc loc, ASTType resultType,
                                       ExprEmitter &emitter) {
  LValue lv =
      getLValueForResult(loc, resultType, /*allowIncompatibleTypes=*/false,
                         /*requireSLValue=*/true, emitter);
  assert(!lv || lv.getIfSLValue());
  return lv.getIfSLValue();
}

//===----------------------------------------------------------------------===//
// ExprEmitter implementation
//===----------------------------------------------------------------------===//

/// Emit an error about use of a dynamic value (the expression) in a context
/// that only allows parameter expressions.  This always returns a null
/// PValue.
PValue ExprEmitter::emitErrorForDynamicValueInParameter(const ExprNode *expr,
                                                        const char *message) {
  assert(paramContext != EC_Unknown && "parameter context not set correctly");
  if (!message)
    message = "cannot use a dynamic value";
  emitError(expr->getLoc(), message)
      << getContextMessage(paramContext) << expr->getRange();
  return {};
}

RValue ExprEmitter::emitRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  ValueDest dest(resultType, context);
  if (auto result = emitRValue(value, dest)) {
    assert(result.getIfRValue() && "dest won't consume so can't be BValue");
    return result.getIfRValue();
  }
  dest.resetForError();
  return {};
}

/// This helper emits the specified value rep as an SSA value, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
AnyValue ExprEmitter::emitRValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value) // Already diagnosed error.
    return {};

  // If this is already an RValue, then we are done.
  if (auto rvRep = value.ir.getIfRValue()) {
    if (!dest.isSpecified())
      return rvRep;
    return emitResult(value.ir, value.expr, dest);
  }

  // If this is an LValue, invoke the getter or load it.
  if (auto lValue = value.ir.getIfLValue()) {
    auto rbValue = emitLoadOfLValue({lValue, value.expr}, dest);
    return emitRValue({rbValue, value.expr}, dest);
  }

  // Finally, if this is a BValue emit a __copy__, a load for a
  // primitive MLIR type, or an error if neither approach works.
  auto bValue = value.ir.getIfBValue();
  assert(bValue && "No other value types");
  return emitBValueToRValue({bValue, value.expr}, dest);
}

CValue ExprEmitter::emitCValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  ValueDest dest(resultType, context);
  if (auto cr = emitCValue(value, dest))
    return cr;
  dest.resetForError();
  return {};
}

CValue ExprEmitter::emitCValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value) // Already diagnosed error.
    return {};
  // If this is already an CValue, then we are done.
  if (auto cRep = value.ir.getIfCValue()) {
    if (!dest.isSpecified())
      return cRep;
    auto result = emitResult(value.ir, value.expr, dest);
    assert(!result || result.getIfCValue());
    return result.getIfCValue();
  }

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  ORValue overloads = value.ir.getIfORValue();
  assert(overloads && "unknown overloaded value");
  return overloads->emitAsCValue(*this, dest);
}

/// Emit an expression providing a immutable borrowed reference to a value.
BValue ExprEmitter::emitBValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value)
    return {};

  // Handle dynamic LValues by loading from them.
  if (auto lv = value.ir.getIfDLValue()) {
    value.ir = emitLoadOfLValue({lv, value.expr}, dest);
    if (!value.ir)
      return {};
  }
  // Handle SLValue's by decaying to MBValue.
  if (auto lv = value.ir.getIfSLValue())
    value.ir = MBValue(lv);

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (auto overloads = value.ir.getIfORValue()) {
    value.ir = overloads->emitAsCValue(*this, dest);
    if (!value.ir)
      return {};
  }

  // If there is a value destination, resolve it into an RValue or BValue.
  if (dest.isSpecified()) {
    value.ir = emitResult(value.ir, value.expr, dest);
    // Emitting the result to the dest could promote back to RValue, so re-emit
    // it with a now-empty (assigned from context) destination.
    return emitBValue(value, dest);
  }

  // Decay RValue's into BValue's.
  if (auto srVal = value.ir.getIfSRValue()) // Decay SRValue -> SBValue
    value.ir = SBValue(srVal);
  else if (auto mrVal = value.ir.getIfMRValue()) // Decay MRValue -> MBValue
    value.ir = MBValue(mrVal);

  // Finally, we know we have a BValue.
  auto resultBV = value.ir.getIfBValue();
  assert(resultBV && "unknown value kind");
  return resultBV;
}

BValue ExprEmitter::emitBValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  ValueDest dest(resultType, context);
  if (auto result = emitBValue(value, dest))
    return result;
  dest.resetForError();
  return {};
}

LValue ExprEmitter::emitLValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value)
    return {};

  if (LValue lValue = value.ir.getIfLValue()) {
    if (!dest.isSpecified())
      return lValue;
    auto result = emitResult(value.ir, value.expr, dest);
    assert(!result || result.getIfLValue());
    return result.getIfLValue();
  }

  emitError(value.expr->getLoc())
      << "expression must be mutable" << getContextMessage(dest.context)
      << value.expr->getRange();
  return {};
}

/// Given an BValue, produce a standalone rvalue in the specified destination
/// by emitting a clone call.  This returns a BValue if dest takes ownership,
/// otherwise it returns an RValue.
CValue ExprEmitter::emitBValueToRValue(ASTExprAnd<BValue> value,
                                       ValueDest &dest) {
  if (!value.ir)
    return {};

  // PValues are both BValues and RValues until materialized.
  if (auto pValue = value.ir.getIfPValue())
    return emitCResult(pValue, value.expr, dest);

  // If this is a non-trivial type,
  if (value.ir.getRValueType().getRegisterPassability(value.expr->getLoc(),
                                                      shared) !=
      StructDeclOp::RP_RegisterPassableTrivial)
    return emitNamedMethodCall("__copy__", {value}, dest,
                               CallSyntax::kImplicitConvert, value.expr);

  // Otherwise we can emit a direct use/load for trivial types.
  SRValue result;
  if (auto sbVal = value.ir.getIfSBValue()) {
    result = SRValue(sbVal);
  } else {
    if (!builder) {
      emitError(value.expr->getLoc(),
                "cannot use a dynamic value in a parameter context")
          << value.expr->getRange();
      return {};
    }
    auto mbVal = value.ir.getIfMBValue();
    assert(mbVal && "Unknown BValue");
    result = builder->create<POP::LoadOp>(value.expr->getLocation(*this), mbVal,
                                          /*alignment=*/std::nullopt);
  }
  return emitCResult(result, value.expr, dest);
}

/// Emit a register primary PValue to an SRValue.
SRValue ExprEmitter::emitPValueToSRValue(ASTExprAnd<PValue> value,
                                         ExprContext context) {
  TypedAttr attr = value.ir.get();
  const ExprNode *expr = value.expr;

  // Make sure this method isn't getting called inappropriately.
  assert(value.ir.getType().isRegisterPassable(expr->getLoc(), shared) &&
         "emitPValueToSRValue called on non-register-passable value");

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (!builder) {
    emitError(expr->getLoc(), "cannot use a dynamic value")
        << getContextMessage(context) << expr->getRange();
    return {};
  }

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
          signature.getInputParams(), "<<UNUSED>>", expr->getLoc(),
          incorrectBindingNo, incorrectBindingExpectedType, *this, nullptr,
          signature.hasParamVarargs());
      if (!bindingAttr) {
        // If it didn't work out, then it is an error because parameterized
        // values cannot be used in a dynamic context.
        emitError(expr->getLoc(), "cannot use parameterized function of type ")
            << ASTType(attr.getType()) << " without binding all its parameters"
            << expr->getRange();
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
      emitError(expr->getLoc(),
                "cannot use parameterized function with result parameters ")
          << ASTType(attr.getType()) << expr->getRange();
      return {};
    }
  }

  auto location = expr->getLocation(*this);
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

/// Emit any kind of PValue to an SLValue.
MBValue ExprEmitter::emitPValueToSLValue(ASTExprAnd<PValue> value, SLValue dest,
                                         ExprContext context) {
  llvm_unreachable("TODO: memory-only parameter expressions not supported yet");
}

/// This helper emits the specified value as a SRValue which has an SSA
/// value representation, materializing PValues and loading LValues as
/// needed.  This returns null if emission fails, and should never be used with
/// values that are memory-only.
SRValue ExprEmitter::emitSRValue(ASTExprAnd<AnyValue> anyValue,
                                 ExprContext context, ASTType resultType) {
  const ExprNode *expr = anyValue.expr;

  // Emit using resultType if present, and eliminate LValue/ORValue's.
  anyValue.ir = emitRValue(anyValue, context, resultType);
  CValue value = emitCValue(anyValue, context);
  if (!value)
    return {};

  assert(value.getRValueType().isRegisterPassable(expr->getLoc(), shared) &&
         "cannot load non-register passable type into SSA register");

  // If we have a value in memory, load it.
  if (auto mrValue = value.getIfMRValue()) {
    // TODO(moves): we should move from the value instead of clone+destroy it.
    // FIXME: can't this just emit a load from the memory directly if the RValue
    // type is register primary because we own it?  Will need to check to see
    // how the borrow checker handles this.
    auto rv = emitBValueToRValue({MBValue(mrValue), expr}, ValueDest::none());
    if (auto sr = rv.getIfSRValue())
      return sr;
    return emitSRValue({rv, expr}, context);
  }

  // If this is already an SRValue, return it.
  if (auto rvalue = value.getIfSRValue())
    return rvalue;

  auto pValue = value.getIfPValue();
  assert(pValue && "must be PValue if register-passable and not SRValue");
  return emitPValueToSRValue({pValue, expr}, context);
}

/// This helper emits the specified expression as a parameter value, diagnosing
/// the problem if the expression is only valid as a runtime value.  This
/// returns null if emission fails.
PValue ExprEmitter::emitPValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  if (!value)
    return {};

  // Clear the builder to indicate that an PValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder, {});
  llvm::SaveAndRestore savedContext(paramContext, context);

  // If there is a result type, coerce before checking for PValue.
  if (resultType) {
    value.ir = emitRValue(value, context, resultType);
    if (!value)
      return {};
  }

  // If this is an ORValue, it must resolve to a single entry.
  if (auto overloads = value.ir.getIfORValue()) {
    ValueDest dest(context);
    value.ir = overloads->emitAsCValue(*this, dest);
    if (!value.ir)
      return {};
  }

  // If this is a parameter, return it.
  if (auto result = value.ir.getIfPValue())
    return result;

  // Otherwise diagnose this as "not a parameter".
  emitError(value.expr->getLoc(), "cannot use a dynamic value")
      << getContextMessage(context) << value.expr->getRange();
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
  Type valueType;
  // Only CValues can be specialized. ORValues don't have a type.
  if (auto cValue = value.getIfCValue())
    valueType = cValue.getType();
  else
    return value;

  LitParameterEvaluator evaluator(emitter.getDeclResolver());
  Type refinedType = evaluator.refineType(valueType);
  if (refinedType == valueType)
    return value;

  // Materialize a parameter rebind.
  if (auto pvalue = value.getIfPValue())
    return PValue(
        ParamOperatorAttr::get(POC::Rebind, pvalue.get(), refinedType));

  // Materialize a rebind operation.
  auto rebind = [&](Value value) -> Value {
    return emitter.builder->create<RebindOp>(emitter.translateLocation(loc),
                                             refinedType, value);
  };
  if (auto lvalue = value.getIfSLValue())
    return SLValue(rebind(lvalue));
  if (auto mrValue = value.getIfMRValue())
    return MRValue(rebind(mrValue));
  if (auto mbValue = value.getIfMBValue())
    return MBValue(rebind(mbValue));
  if (auto sbValue = value.getIfSBValue())
    return SBValue(rebind(sbValue));
  if (auto dlValue = value.getIfDLValue()) {
    dlValue->elementType = refinedType;
    return dlValue;
  }

  auto srValue = value.getIfSRValue();
  assert(srValue && "Unknown value kind");
  return SRValue(rebind(srValue));
}

/// Emit a conversion from the specified value to the specified destination
/// type, plopping the value into the designated value destination.  We know the
/// types mismatch so the conversion must be emitted.
static AnyValue emitConversionTo(CValue value, const ExprNode *expr,
                                 ASTType expectedType, ValueDest &dest,
                                 ExprEmitter &emitter) {

  auto errorHandler = [&]() {
    emitter.emitError(expr->getLoc())
        << value.getRValueType() << " value cannot be converted to "
        << expectedType << getContextMessage(dest.getContext())
        << expr->getRange();
  };

  // We disable implicit conversions though, to prevent converting T -> S -> U
  // in one step, and to avoid infinite conversion cycles.
  return emitter.emitConstructorCall(
      expectedType, {{value, expr}}, expr, CallSyntax::kImplicitConvert, dest,
      errorHandler, /*allowImplicitConversion=*/false);
}

/// Emit the specified value into the current destination if present.  This
/// accepts (and silently propagates) null values.
///
/// Note that the `value` provided here may require an implicit conversion into
/// the destination slot, so the input may be memory-only and result be
/// register-passable (and visa-versa).
AnyValue ExprEmitter::emitResult(AnyValue value, const ExprNode *expr,
                                 ValueDest &dest) {
  if (!value)
    return {};

  // Attempt to further specialize the result value.
  value = refineResultValue(value, expr->getLoc(), *this);

  // If no destination is specified or it is just a contextual type hint, then
  // we can propagate the value directly.
  if (!dest.isSpecified() || isa<LValueInitializerType>(dest.representation)) {
    dest.representation = NullRepresentation();
    return value;
  }

  // OK, if there is a destination specified, handle them by converging the set
  // of value types we have.

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (auto overloads = value.getIfORValue())
    return overloads->emitAsCValue(*this, dest);

  auto cValue = value.getIfCValue();
  assert(cValue && "Must be a CValue if not an ORValue");
  auto rvalueType = cValue.getRValueType();

  // If there is a known type for the destination but the value disagrees, emit
  // an implicit conversion directly into the destination.  This keeps values in
  // registers and avoids a "convert + clone" pair for memory->memory
  // conversions.
  if (ASTType requiredType =
          dest.resolveImpliedType(expr->getLoc(), rvalueType, *this)) {
    if (!requiredType.isEqualCanon(rvalueType))
      return emitConversionTo(cValue, expr, requiredType, dest, *this);
  }

  // If the destination is just a required type, then we now know it must agree
  // and therefore don't need to do anything more.
  if (isa<ASTType>(dest.representation)) {
    dest = ValueDest(); // Resolved the ValueDest;
    return cValue;
  }

  // If this destination was an LValue whose buffer was already taken to be
  // filled in by a client, then this is just completing the transaction.
  if (isa<LValueBufferTaken>(dest.representation)) {
    dest = ValueDest(); // Resolved the ValueDest;
    // The client directly filled in an LValue we provided which is great, but
    // that LValue we provided took ownership of the value, so we need to return
    // the result as a borrow, not an owned reference.
    auto memValue = value.getIfMRValue();
    assert(memValue && "Must be an MRValue providing result");
    return MBValue(memValue);
  }

  // We know we have an CRValue/BValue and the destination is some kind of
  // LValue.  Emit the dest to figure out where to store it.
  LValue destLV = dest.getLValueForResult(expr->getLoc(), rvalueType,
                                          /*allowIncompatibleTypes=*/true,
                                          /*requireSLValue=*/false, *this);
  if (!destLV)
    return {};

  // This will have completely resolved all the ValueDest possibilities.
  assert(!dest.isSpecified() || isa<LValueBufferTaken>(dest.representation));
  dest = ValueDest();

  // Finally, store the value into the lvalue.
  return emitStoreToLValue({cValue, expr}, destLV, dest.getContext());
}

/// Given a value with a known type, emit a store to the specified LValue.  This
/// returns an borrowed reference to the value after it is done.  The types must
/// match for this call.
BValue ExprEmitter::emitStoreToLValue(ASTExprAnd<CValue> value, LValue destLV,
                                      ExprContext context) {
  assert(value.ir.getRValueType().isEqualCanon(destLV.getRValueType()) &&
         "Types should match");

  // If this is a computed LValue, then perform a writeback.
  if (auto dlValue = destLV.getIfDLValue()) {
    // If the value itself is an LValue, emit a load so we can call the setter.
    if (auto valueLV = value.ir.getIfLValue())
      value.ir = emitLoadOfLValue({valueLV, value.expr}, ValueDest::none());

    // Then store into the dest DLValue.
    dlValue->emitStore(value, *this);

    // Decay the input value to a BValue since ownership was taken by the store.
    return emitBValue(value, context, {});
  }

  // Otherwise we have an SLValue destination.
  SLValue destPtr = destLV.getIfSLValue();
  assert(destPtr && "No other known LValue");

  // If the input is an LValue, load it into the destination.
  if (auto lvalue = value.ir.getIfLValue()) {
    ValueDest dest(destPtr, context);
    auto result = emitLoadOfLValue({lvalue, value.expr}, dest);
    assert((!result || result.getIfBValue()) &&
           "dest specified, so this should return BValue");
    dest.resetForError();
    return result.getIfBValue();
  }

  // We will be doing a store, so if we have a PValue emit it to something with
  // a dynamic representation.
  if (auto pValue = value.ir.getIfPValue()) {
    // If the parameter expression is memory only, emit it directly into the
    // destination slot.
    if (!destPtr.getRValueType().isRegisterPassable(value.expr->getLoc(),
                                                    shared))
      return emitPValueToSLValue({pValue, value.expr}, destPtr, context);

    // Otherwise, materialize into an SSA register.
    value.ir = emitPValueToSRValue({pValue, value.expr}, context);
    if (!value.ir)
      return {};
  }

  // If this is an SRValue then store the value to memory.
  if (auto rv = value.ir.getIfSRValue()) {
    // Does this kill las use of emitSRValue?
    auto loc = translateLocation(value.expr->getLoc());
    builder->create<POP::StoreOp>(loc, rv, destPtr, /*alignment=*/std::nullopt);
    return SBValue(rv);
  }

  // If we have an MRValue in the wrong spot, emit a __copy__ call into the
  // destination using MBValue -> RValue conversion.
  // TODO(move): This won't take ownership of the value.  When we have move
  // operations we should use them.
  if (auto mrValue = value.ir.getIfMRValue())
    value.ir = MBValue(mrValue);

  // Finally this must be a BValue, emit a clone into the specified destination.
  auto bValue = value.ir.getIfBValue();
  assert(bValue && "Should only have bvalues here");
  ValueDest dest(destPtr, context);
  auto result = emitBValueToRValue({bValue, value.expr}, dest);
  if (!result)
    dest.resetForError();
  assert((!result || result.getIfBValue()) &&
         "Dest specified should return a BValue");
  return result.getIfBValue();
}

/// Emit a call to the getter of the specified LValue, loading the value into
/// dest (if specified) or returning it if not.  This returns an RValue if
/// there is no consuming dest, otherwise a BValue.
CValue ExprEmitter::emitLoadOfLValue(ASTExprAnd<LValue> value,
                                     ValueDest &dest) {
  // If this is a computed LValue emit call to the "getter".
  if (auto dlValue = value.ir.getIfDLValue())
    return dlValue->emitLoad(dest, value.expr, *this);

  // Decay a stored LValue to an MBValue.
  auto slValue = value.ir.getIfSLValue();
  assert(slValue && "unknown lvalue kind");

  // Emit a __copy__ or load of the value.
  return emitBValueToRValue({MBValue(slValue), value.expr}, dest);
}

// Emitting a CValue always produces a CValue.
CValue ExprEmitter::emitCResult(CValue value, const ExprNode *expr,
                                ValueDest &dest) {
  auto result = emitResult(value, expr, dest);
  assert((!result || result.getIfCValue()) &&
         "emitting a CValue as a result should always produce a CValue");
  return result.getIfCValue();
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
CValue ExprEmitter::emitNamedMethodCall(
    StringRef methodName, ArrayRef<ASTExprAnd<AnyValue>> argValues,
    ValueDest &dest, CallSyntax syntax, const ExprNode *callNode) {
  assert(!argValues.empty() && "Cannot emit a method call without a receiver!");

  // Emit the first/self operand to a CValue so we can figure out which type to
  // lookup on.
  CValue selfVal = argValues[0].ir.getIfCValue();
  SmallVector<ASTExprAnd<AnyValue>> updatedArgValues;
  if (!selfVal) {
    selfVal = emitCValue(argValues[0], ValueDest::none());
    if (!selfVal)
      return {};
    // We can't mutate argValues because it's an ArrayRef.  If something
    // changed, recurse with a temporary buffer.
    updatedArgValues.append(argValues.begin(), argValues.end());
    updatedArgValues[0].ir = selfVal;
    argValues = updatedArgValues;
  }

  ASTType type = selfVal.getRValueType();

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
bool ExprEmitter::canImplicitlyConvertToType(ASTExprAnd<CValue> value,
                                             ASTType requiredType) {
  // If it already matches, then we're done.
  if (value.ir.getRValueType().isEqualCanon(requiredType))
    return true;

  // Check to see if we can do an implicit conversion by invoking a `__init__`
  // method on the expected type.
  OverloadSet callee(requiredType, "__init__", value.expr,
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
RValue ExprEmitter::emitI1(ASTExprAnd<CValue> value, CValue &boolResult) {
  if (!value.ir)
    return {};

  boolResult = value.ir;
  ASTType valueRValueType = value.ir.getRValueType();

  // If this is already an 'i1', then we're done.
  if (valueRValueType.mlirType.isInteger(1))
    return emitRValue(value, EC_BoolCondition);

  // TODO: Python manual includes this off-hand comment:
  // Also, an object that doesn’t define a __bool__() method and whose __len__()
  // method returns zero is considered to be false in a Boolean context.

  // Check for the presence of a __mlir_i1__ method.  If it exists, we can avoid
  // a redundant call to __bool__ for Bool types.
  if (!OverloadSet(valueRValueType, "__mlir_i1__", value.expr,
                   CallSyntax::kImplicitConvert, shared,
                   [&]() { /*no error*/ })) {
    // Use the __bool__ method to convert the user defined type to
    // something that is a Bool or other type that implements __mlir_i1__.
    boolResult = emitNamedMethodCall("__bool__", {{value.ir, value.expr}},
                                     ValueDest::none(),
                                     CallSyntax::kImplicitConvert, value.expr);
    if (!boolResult)
      return {};
  }

  // Then we use __mlir_i1__ to convert to an i1 value.
  CValue litBoolCall = emitNamedMethodCall(
      "__mlir_i1__", {{boolResult, value.expr}}, ValueDest::none(),
      CallSyntax::kImplicitConvert, value.expr);

  return emitRValue({litBoolCall, value.expr}, EC_BoolCondition);
}

//===----------------------------------------------------------------------===//
// ExprEmitter implementation
//===----------------------------------------------------------------------===//

/// Emit the specified node with the indicated expression context and an
/// optional contextual type.
AnyValue ExprEmitter::emitExpr(const ExprNode *expr, ExprContext context,
                               ASTType resultType) {
  ValueDest dest(resultType, context);
  if (auto result = expr->emitIR(dest, *this))
    return result;
  dest.resetForError();
  return {};
}

/// This helper emits the specified value rep as an RValue.
RValue ExprEmitter::emitExprRValue(const ExprNode *expr, ExprContext context,
                                   ASTType resultType) {
  return emitRValue({emitExpr(expr, context, resultType), expr}, context);
}

/// This helper emits the specified value rep as an CRValue.
CValue ExprEmitter::emitExprCValue(const ExprNode *expr, ExprContext context) {
  assert(expr && "cannot emit a null node");
  return emitCValue({emitExpr(expr, context), expr}, context);
}

/// This helper emits the specified value rep as an SRValue, materializing
/// it as a parameter constant if it is a parameter.  This returns null if
/// emission fails.
SRValue ExprEmitter::emitExprSRValue(const ExprNode *expr, ExprContext context,
                                     ASTType resultType) {
  assert(expr && "cannot emit a null node");
  return emitSRValue({emitExpr(expr, context), expr}, context);
}

/// This helper emits the specified expression as a parameter value, diagnosing
/// the problem if the expression is only valid as a runtime value.  This
/// returns null if emission fails.
PValue ExprEmitter::emitExprPValue(const ExprNode *expr, ExprContext context,
                                   ASTType resultType) {
  // Clear the builder to indicate that an PValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder, {});
  llvm::SaveAndRestore savedContext(paramContext, context);

  // Emit the expression using the contextual type if present.
  AnyValue rep = emitExpr(expr, context, resultType);
  return emitPValue({rep, expr}, context);
}

/// Emit the specified expression as an LValue which can be loaded and stored.
/// If contextualType is non-null, then an implicitly declared LValue will be
/// assigned that type.
///
/// This diagnoses the expression with the specified message if it isn't a
/// valid LValue.
LValue ExprEmitter::emitExprLValue(const ExprNode *expr, ValueDest &dest) {
  AnyValue anyValue = expr->emitIR(dest, *this);
  if (!anyValue)
    return {}; // Error already diagnosed.
  return emitLValue({anyValue, expr}, dest);
}

/// This helper emits the specified expression tree as a type, e.g. turning
/// "Int" into the type for it.  This emits an error and returns null on
/// failure.
ASTType ExprEmitter::emitExprType(const ExprNode *expr, bool isPack) {
  auto value = emitExprPValue(expr, EC_Type);
  if (!value)
    return {};

  // If we emitted a NoneAttr then convert it to a NoneType.  This is a
  // special case because "None" is both a value and a type, and defaults to a
  // value.
  if (isa<NoneAttr>(value.get()))
    return shared.getNoneType();

  ASTType type;
  if (isPack) {
    // Pack types expect a backing variadic type.
    if (!isa<VariadicType>(value.get().getType())) {
      emitError(expr->getLoc(), "only variadic types may be unpacked")
          << expr->getRange();
      return {};
    }
    type = POP::PackType::get(value.get().getContext(), value.get());
  } else {
    // If this emitted a type, we can lower it.
    type = value.getIfTypeValue();
  }

  if (!type) {
    emitError(expr->getLoc(), "expected a type, not a value")
        << expr->getRange();
    return {};
  }

  // Verify that all of the parameters for this type are bound.  We allow
  // PValues to refer to parameteric type, but anything calling `emitType`
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
      structDecl.getInputParamsAttr(), structDecl.getName(), expr->getLoc(),
      incorrectBindingNo, incorrectBindingExpectedType, *this, structDecl,
      structDecl.getParamVarargs());
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

/// Emit a call __init__, returning an instance of the specified
/// type.  If `allowImplicitConversion` is true, the provided args are allowed
/// to implicitly convert to the expectations of the constructor signatures.
CValue ExprEmitter::emitConstructorCall(ASTType type,
                                        ArrayRef<ASTExprAnd<AnyValue>> args,
                                        const ExprNode *expr, CallSyntax syntax,
                                        ValueDest &dest,
                                        std::function<void()> errorHandler,
                                        bool allowImplicitConversion) {
  bool hasCustomErrorReporting = true;
  if (!errorHandler) {
    errorHandler = [&]() {
      auto diag = emitError(expr->getLoc(), "")
                  << type << " does not implement an '__init__' method"
                  << expr->getRange();
    };
    hasCustomErrorReporting = false;
  }

  // Check to see if we can invoke an __init__ method to convert it.
  OverloadSet callee(type, "__init__", expr, CallSyntax::kImplicitConvert,
                     shared, errorHandler);

  if (callee.isNull())
    return {};

  if (failed(callee.filterOverloadSet(
          args, allowImplicitConversion,
          /*emitDiagnosticOnFailure=*/!hasCustomErrorReporting, *this))) {
    if (hasCustomErrorReporting)
      errorHandler();
    return {};
  }

  // Ok, cool we know it will succeed; do it.
  return callee.emitCall(args, dest, *this);
}

/// Emit the specified expression as a condition, converting it to an MLIR I1
/// value that we can test directly.  This reports and error and returns null on
/// error.
RValue ExprEmitter::emitExprI1(const ExprNode *condExpr, ExprContext context) {
  CValue boolTmp; // we don't care about the intermediate Bool value.
  return emitI1({emitExprCValue(condExpr, context), condExpr}, boolTmp);
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
  if (!index)
    return {};
  auto popscalar = builder->create<POP::CastFromBuiltinOp>(
      translateLocation(source->getLoc()),
      POP::SIMDType::get(builder->getContext(), 1, KGENDType(KGENDType::index)),
      index.getIfSRValue());
  return SRValue(popscalar);
}

//===----------------------------------------------------------------------===//
// DLValue implementations
//===----------------------------------------------------------------------===//

CValue StoredAttributeRefDLValue::emitLoad(ValueDest &dest,
                                           const ExprNode *dlValueExpr,
                                           ExprEmitter &emitter) const {
  // To load x.y, we load x, then then load y out of it.
  auto base = baseVal.ir->emitLoad(ValueDest::none(), baseVal.expr, emitter);
  if (!base)
    return {};
  return AttributeRefNode::emitStoredFieldRef({base, baseVal.expr}, getField(),
                                              expr, dest, emitter);
}

void StoredAttributeRefDLValue::emitStore(ASTExprAnd<CValue> value,
                                          ExprEmitter &emitter) const {
  if (!emitter.builder) {
    emitter.emitErrorForDynamicValueInParameter(expr);
    return;
  }

  // tmp = load(base)
  // tmp.field = value
  // store(tmp -> base)
  auto loc = expr->getLocation(emitter);
  Type declIRType = POP::PointerType::get(baseVal.ir->elementType);
  auto nameAttr = StringAttr::get(loc.getContext(), "__store_tmp__");
  auto tmpDecl =
      emitter.builder->create<VarLetDeclOp>(loc, declIRType, nameAttr,
                                            /*isVar=*/true);

  // Load the entire base LValue into tmpDecl.
  ValueDest tmpValueDest(SLValue(tmpDecl), EC_AttributeRefBase);
  auto base = baseVal.ir->emitLoad(tmpValueDest, baseVal.expr, emitter);
  if (!base) {
    tmpValueDest.resetForError();
    return;
  }

  // Store into the field.
  auto fieldPtr =
      emitter.builder->create<StructGEPOp>(loc, tmpDecl, getField());
  emitter.emitStoreToLValue(value, SLValue(fieldPtr), EC_AttributeRefBase);

  // Store the whole result back.
  // TODO(moves): This could move the value.
  baseVal.ir->emitStore({SLValue(tmpDecl), expr}, emitter);
}

CValue SubscriptDLValue::emitLoad(ValueDest &dest, const ExprNode *dlValueExpr,
                                  ExprEmitter &emitter) const {
  auto methodName =
      isSubscript() ? StringRef("__getitem__") : StringRef("__getattr__");
  auto result = emitter.emitNamedMethodCall(methodName, selfAndIndicesValue,
                                            dest, CallSyntax::kSubscript, expr);
  // TODO: The result could be another LValue in the future.
  assert(!result || result.getIfRValue() || result.getIfBValue());
  return result;
}

void SubscriptDLValue::emitStore(ASTExprAnd<CValue> value,
                                 ExprEmitter &emitter) const {
  auto methodName =
      isSubscript() ? StringRef("__setitem__") : StringRef("__setattr__");
  SmallVector<ASTExprAnd<AnyValue>> operands(selfAndIndicesValue.begin(),
                                             selfAndIndicesValue.end());
  operands.push_back(value);
  emitter.emitNamedMethodCall(methodName, operands, ValueDest::none(),
                              CallSyntax::kSubscript, expr);
}
