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
  auto rep = node->emitIR(*this);
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
// Name Lookup
//===----------------------------------------------------------------------===//

/// Perform a name lookup in the specified scope and return the named
/// declaration as a LookupResult.
auto ExprEmitter::lookupAndResolveDecl(StringRef name, SMLoc loc,
                                       ASTDecl &scope) -> LookupResult {

  // Ensure the context is fully resolved, so all its members are known.  It
  // would be bad to look something up in a scope without all members known.
  // FIXME(Issue#5975): FuncOp shouldn't be special cased.
  if (!isa<FuncOp>(scope)) {
    if (failed(shared.declResolver->resolve(
            scope, DeclResolvedness::fullyResolved, loc)))
      return LookupResult::getErroneous();
  }

  // Look up the name.
  TinyPtrVector<ASTDecl *> *entry = scope.lookup(name);
  // If nothing was found, return a failure.
  if (!entry)
    return LookupResult::getFailure();

  // FIXME: Hard coded to look up the first value.
  ASTDecl *result = (*entry)[0];

  // If the lookup succeeded, make sure the signature for the referenced decl
  // is understood.
  if (failed(shared.declResolver->resolve(
          *result, DeclResolvedness::signatureResolved, loc))) {
    // If the decl was erroneous somehow, then don't form a reference to it, the
    // error has already been diagnosed.
    return LookupResult::getErroneous();
  }

  return LookupResult::getSuccess(result);
}

/// Perform a name lookup for a member in the specified type.
auto ExprEmitter::lookupAndResolveDecl(StringRef name, SMLoc loc, ASTType scope)
    -> LookupResult {
  if (auto *decl = scope.getDecl(shared))
    return lookupAndResolveDecl(name, loc, *decl);
  return LookupResult::getFailure();
}

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

/// Returns true if the insertion context is valid for implicit error
/// propagation.
static bool isValidErrorContext(Block *block) {
  for (Operation *op = block->getParentOp(); op; op = op->getParentOp()) {
    if (isa<TryOp>(op))
      return true;
    if (auto func = dyn_cast<LIT::FuncOp>(op))
      return func.getRaises();
  }
  return false;
}

/// Emit a function call to the specified callee with the specified operand
/// values.
AnyValue ExprEmitter::emitFunctionCall(CallableValue calleeVal,
                                       ArrayRef<ASTExprAnd<AnyValue>> operands,
                                       SMLoc callLoc) {
  if (!calleeVal)
    return {};

  // If the call is a direct call with a bound self, add it to the operand list
  // to simplify the logic below.
  bool isMethodInvocation = false;
  SmallVector<ASTExprAnd<AnyValue>> operandsWithSelf;
  if (calleeVal.baseVal && calleeVal.direct) {
    operandsWithSelf.reserve(operands.size() + 1);
    operandsWithSelf.push_back(calleeVal.baseVal);
    operandsWithSelf.append(operands.begin(), operands.end());
    operands = operandsWithSelf;
    calleeVal.baseVal = {};
    isMethodInvocation = true;
  }

  auto emitError = [&](const Twine &message) {
    return this->emitError(callLoc, message);
  };

  // Figure out the type of the function to call, which is either symbol or a
  // normal rvalue.
  SignatureType calleeSig;

  // This is the callee symbol constant for a direct call, or the SSA value for
  // an indirect call.
  PointerUnion<Attribute, Value> callee;
  if (calleeVal.direct) {
    SymbolConstantAttr symbol = calleeVal.direct->getBoundConstantAttr();
    if (!symbol)
      return {};

    calleeSig = symbol.getType();
    callee = symbol;
  } else {
    // Otherwise we have an indirect call, emit the callee value as a DRValue so
    // we can call it with call_indirect.
    auto calleeDRVal = emitDRValue(calleeVal.baseVal.ir, callLoc);
    if (!calleeDRVal)
      return {};

    calleeSig = dyn_cast<SignatureType>(calleeDRVal.getType());
    if (!calleeSig) {
      emitError("invalid function type to call ")
          << ASTType(calleeDRVal.getType());
      return {};
    }
    callee = calleeDRVal;
  }

  assert(calleeSig.getResultParamTypes().empty() &&
         "TODO: meta results not implemented yet");

  size_t numArgs = calleeSig.getValues().getNumInputs();
  if (numArgs != operands.size()) {
    emitError("callee expects ") << numArgs << " argument" << plural(numArgs);
    return {};
  }

  // Emit all the arguments.
  SmallVector<Value> valueArguments;
  for (auto [argAnyValueAndExpr, expectedType, convention] :
       llvm::zip(operands, calleeSig.getValueInputs(),
                 calleeSig.getValueInputConventions())) {
    auto argLoc = argAnyValueAndExpr.expr->getLoc();
    // If the callee takes the operand as a by-ref argument, we require an
    // lvalue.
    Value argVal;
    switch (ValueInputConvention(convention)) {
    case ValueInputConvention::ByRef:
      argVal = argAnyValueAndExpr.ir.getIfLValue();
      if (!argVal) {
        if (isMethodInvocation && valueArguments.empty()) {
          this->emitError(argLoc,
                          "invalid use of mutating method on rvalue of type ")
              << ASTType(argAnyValueAndExpr.ir.getType());
        } else {
          this->emitError(
              argLoc,
              "operand must be mutable in order to pass as a by-ref argument");
        }
        return {};
      }

      // If we have an lvalue of the wrong type, diagnose the error prettily.
      if (!ASTType(argVal.getType()).isEqualCanon(ASTType(expectedType))) {
        auto argRVType = argAnyValueAndExpr.ir.getRValueType();
        this->emitError(argLoc, "l-value of type ")
            << argRVType
            << " cannot be converted to reference to expected type "
            // TODO(QoI): Types are not attributes.
            << cast<POP::PointerType>(expectedType).getElementType();
        return {};
      }

      break;
    case ValueInputConvention::ByVal:
      // Otherwise, we pass as an r-value.
      argVal = emitDRValue(argAnyValueAndExpr.ir, argLoc);
      if (!argVal)
        return {};

      // Convert the argument to the expected type if needed, or diagnose if
      // incompatible.
      argVal = getAsExpectedType(argVal, argAnyValueAndExpr.expr, expectedType);
      if (!argVal)
        return {};
      break;
    }

    valueArguments.push_back(argVal);
  }

  if (!builder) {
    emitError("TODO: cannot call function in parameter context");
    return {};
  }

  // If this is a call to something representable as an attribute, we can use
  // a kgen.call_param.
  Value resultVal;
  auto loc = translateLocation(callLoc);
  // FIXME: Move result type inference into CallOp/CallIndirectOp.
  auto resultTypes = calleeSig.getValueResults();
  if (auto target = dyn_cast<Attribute>(callee)) {
    resultVal =
        builder
            ->create<CallOp>(loc, resultTypes, cast<SymbolConstantAttr>(target),
                             ArrayRef<ParamDeclAttr>(), valueArguments)
            .getResult(0);
  } else {
    // Otherwise emit calls to SSA values with call_indirect.
    auto calleeDRVal = cast<Value>(callee);
    resultVal = builder
                    ->create<CallIndirectOp>(loc, resultTypes, calleeDRVal,
                                             /*operands*/ valueArguments)
                    .getResult(0);
  }

  // If the callee can raise an error, try to unwrap it.
  if (calleeSig.getFnEffects() == FnEffects::Throws) {
    if (!isValidErrorContext(builder->getInsertionBlock())) {
      this->emitError(
          callLoc,
          "cannot call raising method within an 'fn' that does not raise");
      return {};
    }
    resultVal = builder->create<UnwrapOrPropagateOp>(
        loc, cast<POP::VariantType>(resultVal.getType()).getType(1), resultVal);
  }

  // Value returning call returns its result.
  return DRValue(resultVal);
}

/// This helper emits a method call to a special function (`kind`) on `type`
/// with the provided `operands`. This emits an error if the special function
/// is not implemented by the type and returns null.
AnyValue
ExprEmitter::emitSpecialMethodCall(ASTType type, SpecialFunctionKind kind,
                                   ArrayRef<ASTExprAnd<AnyValue>> operands,
                                   SMLoc callLoc) {
  // Look up the special function based on the SpecialFunctionKind.
  auto specialFnInfo = SpecialFunctionInfo::get(kind);
  auto nameAttr = StringAttr::get(getContext(), specialFnInfo.name);

  auto lookupResult = lookupAndResolveDecl(specialFnInfo.name, callLoc, type);
  ASTDecl *resultDecl = lookupResult.getIfSuccess();
  if (!resultDecl) {
    if (lookupResult.isFailure())
      emitError(callLoc, "") << type << " does not implement the " << nameAttr
                             << " special method";
    return {};
  }

  CallableValue callee(callLoc, *resultDecl, type.getParamBindings());
  return emitFunctionCall(callee, operands, callLoc);
}

/// Convert the specified DRValue to the expected type, invoking implicit
/// conversions if necessary.  On error, this diagnoses it and returns null.
DRValue ExprEmitter::getAsExpectedType(DRValue value, const ExprNode *expr,
                                       ASTType expectedType) {
  // If the type is already an exact match, then we are done.
  if (ASTType(value.getType()).isEqualCanon(expectedType))
    return value;

  // Check to see if we can invoke an __new__ method to convert it.
  auto lookupResult =
      lookupAndResolveDecl("__new__", expr->getLoc(), expectedType);
  ASTDecl *resultDecl = lookupResult.getIfSuccess();
  if (!resultDecl) {
    if (lookupResult.isFailure()) {
      emitError(expr->getLoc(), "value of type ")
          << ASTType(value.getType())
          << " cannot be converted to expected type " << expectedType;
    }
    return {};
  }

  CallableValue callee(expr->getLoc(), *resultDecl,
                       expectedType.getParamBindings());
  ASTExprAnd<AnyValue> newArg = {DRValue(value), expr};
  auto result = emitFunctionCall(callee, newArg, expr->getLoc());
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

  // TODO: We could look for the presence of a __lit_bool method and avoid a
  // redundant call to __bool__ for Bool types.

  // First we use the __bool__ method to convert the user defined type to
  // something that is a Bool or other type that implements __lit_bool.
  SMLoc valueLoc = value.expr->getLoc();
  boolResult =
      emitSpecialMethodCall(value.ir.getType(), SpecialFunctionKind::kBool,
                            {{value.ir, value.expr}}, valueLoc);
  if (!boolResult)
    return {};

  // Then we use __lit_bool to convert to an i1 value.
  AnyValue litBoolCall =
      emitSpecialMethodCall(boolResult.getType(), SpecialFunctionKind::kLitBool,
                            {{boolResult, value.expr}}, valueLoc);
  return DRValue(emitDRValue(litBoolCall, valueLoc));
}

/// Emit the specified expression as a condition, converting it to an MLIR I1
/// value that we can test directly.  This reports and error and returns null on
/// error.
DRValue ExprEmitter::emitConditionValueAsI1(ExprNode *condExpr) {
  AnyValue boolTmp; // we don't care about the intermediate Bool value.
  return emitConditionValueAsI1({emitRValue(condExpr), condExpr}, boolTmp);
}
