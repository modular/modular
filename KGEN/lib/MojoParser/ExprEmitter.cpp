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

#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/MojoParser/StructEmitter.h"

#include "MojoUtils.h"
#include "Traits.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ExprContext
//===----------------------------------------------------------------------===//

const char *LIT::getContextMessage(ExprContext context) {
  switch (context) {
  case EC_InvalidContext:
    assert(0 && "cannot emit an invalid context");
    return "";
  case EC_VarInit:
    return " in 'var' initializer";
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
  case EC_CallRefArgValue:
    return " in 'ref' call argument";
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
  case EC_OwnedRegArgShadow:
    return " in owned argument shadow";
  case EC_VarArgArgument:
    return " in vararg argument compiler implementation internals";
  case EC_PackArgument:
    return " in variadic pack argument compiler implementation internals";
  case EC_KWArgsArgument:
    return " in keyword arguments dict compiler implementation internals";
  case EC_DefaultParam:
    return " in default parameter";
  case EC_BoolCondition:
    return " in boolean condition";
  case EC_CondExpr:
    return " in 'if' expression value";
  case EC_BoolParamCondition:
    return " in '@parameter if' condition";
  case EC_ForParamSeq:
    return " in '@parameter for' sequence initializer";
  case EC_ForIterator:
    return " in 'for' iterator expression";
  case EC_WithContextMgr:
    return " in 'with' context manager";
  case EC_WithExitResult:
    return " in 'with' call to '__exit__' on context manager";
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
  case EC_Subscript: // y[x]
    return " in subscript";
  case EC_SliceIndex: // y[:x:]
    return " in slice index";
  case EC_ParameterList: // something[paramValue]
    return " in parameter list";
  case EC_Destructor:
    return " in '__del__' resolution";
  case EC_Capture:
    return " in capture";
  case EC_Decorator:
    return " in decorator";
  case EC_AutoDeref:
    return " in automatic dereference";
  case EC_Trait:
    return " in trait conformance checking";
  case EC_Closure:
    return " in internal closure formation";
  case EC_Lifetime:
    return " in lifetime specifier";
  }
  llvm_unreachable("invalid expr context");
}

//===----------------------------------------------------------------------===//
// ValueDest
//===----------------------------------------------------------------------===//

ValueDest::ValueDest(VarDeclOp dest, ExprContext context)
    : representation(dest.getOperation()), context(context) {}

ValueDest::ValueDest(GlobalVarDeclOp dest, ExprContext context)
    : representation(dest.getOperation()), context(context) {}

void ValueDest::dump() const { llvm::errs() << *this; }

[[maybe_unused]] raw_ostream &LIT::operator<<(raw_ostream &os,
                                              const ValueDest &value) {
  os << "ValueDest context=" << (int)value.context << " destination = ";

  auto &representation = value.representation;
  if (isa<NullRepresentation>(representation)) {
    os << "NullRepresentation";
  } else if (auto lv = dyn_cast<LValue>(representation)) {
    os << "LValue: ";
    lv.dump();
  } else if (isa<LValueBufferTaken>(representation)) {
    os << "LValueBufferTaken";
  } else if (auto expr = dyn_cast<const ExprNode *>(representation)) {
    const char *startPtr = expr->getRangeStart().getPointer();
    size_t length = expr->getRangeEnd().getPointer() - startPtr;
    os << "ExprNode: " << StringRef(startPtr, std::min(length, size_t(80)))
       << "\n---- 8< ----";

  } else if (auto *op = dyn_cast<Operation *>(representation)) {
    os << "Operation*: " << *op;
  } else if (auto type = dyn_cast<ASTType>(representation)) {
    os << "ASTType: " << type;
  } else if (isa<LValueInitializerType>(representation)) {
    os << "LValueInitializerType: "
       << cast<LValueInitializerType>(representation).type;
  } else {
    os << "UNKNOWN VALUE DEST!";
  }
  os << '\n';
  return os;
}

/// If this indicates an explicit expected RValue type, return that type.
ASTType ValueDest::getExpectedTypeIfSpecified() const {
  // Operations generally don't have implied types, except if this is global
  // variable declaration.
  if (auto op = dyn_cast<Operation *>(representation)) {
    if (auto globalVarDecl = dyn_cast<GlobalVarDeclOp>(*op)) {
      if (!isa<UnresolvedType>(globalVarDecl.getType()))
        return globalVarDecl.getType();
    }
    return {};
  }

  // These have no implied type.
  if (isa<NullRepresentation, LValueBufferTaken, const ExprNode *>(
          representation))
    return {};

  // If we just have a contextual type, return it.
  if (ASTType type = dyn_cast<ASTType>(representation))
    return type;
  if (isa<LValueInitializerType>(representation))
    return cast<LValueInitializerType>(representation).type;
  return cast<LValue>(representation).getRValueType();
}

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
  // Operations generally don't have implied types, except if this is global
  // variable declaration.
  if (auto op = dyn_cast<Operation *>(representation)) {
    if (auto globalVarDecl = dyn_cast<GlobalVarDeclOp>(*op)) {
      if (isa<UnresolvedType>(globalVarDecl.getType()))
        return existingValueType;
      return globalVarDecl.getType();
    }
    return {};
  }

  // These have no implied type.
  if (isa<NullRepresentation, LValueBufferTaken>(representation))
    return {};

  // If we just have a contextual type, return it.
  if (ASTType type = dyn_cast<ASTType>(representation))
    return type;

  assert(!isa<LValueInitializerType>(representation) &&
         "LValueInitializerType should be resolved before this");

  // If we have an un-emitted expression, emit it using our existingValueType to
  // get an LValue.
  if (auto *expr = dyn_cast<const ExprNode *>(representation)) {
    // If we have a contextual type available, pass that down to the emitter so
    // implicitly declared variables and discard patterns can know their type.
    ValueDest dest(context);
    if (existingValueType) {
      if (ASTType nmTarget = ASTType(existingValueType)
                                 .getNonmaterializableTarget(emitter.shared))
        existingValueType = nmTarget;
      dest = ValueDest(LValueInitializerType{existingValueType}, context);
    }

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

/// If this ValueDest specifies an MLValue that will be returned by
/// getMLValueForResult with the specified type, return it. Otherwise return
/// null.
///
/// NOTE: This needs to be kept in sync with getLValueForResult.
MLValue ValueDest::getDefinedMLValueIfExists(ASTType resultType,
                                             ExprEmitter &emitter) {
  // If we have an uncollapsed expression, emit it to learn more about it.
  if (const ExprNode *target = dyn_cast<const ExprNode *>(representation)) {
    ValueDest dest(LValueInitializerType{resultType}, getContext());
    if (LValue lValue = emitter.emitExprLValue(target, dest)) {
      representation = lValue;
    } else {
      dest.resetForError();
      representation = NullRepresentation(); // Error already emitted!
    }
  }

  // Check for the simple case.
  if (LValue lValue = dyn_cast<LValue>(representation)) {
    if (MLValue refValue = lValue.getIfMLValue()) {
      if (lValue.getRValueType().isEqualCanon(resultType) &&
          lValue.getMValueType().isDefaultAddrSpace())
        return refValue;
    }
  }

  // Otherwise, this would create a new buffer.
  return {};
}

/// Project a ValueDest into an lvalue with the specified underlying (RValue)
/// type.
///
/// When `allowIncompatibleTypes` is true, the method is allowed to return an
/// LValue of a different type when the underlying storage requires this. This
/// is a guarantee from the caller that it is prepared to handle a type
/// conversion on its side, eliminating a temporary buffer in register-passable
/// cases like `var x : Float32 = 1`.
///
/// When `allowIncompatibleTypes` is false, this always returns an LValue of
/// the requested type, which may return a temporary buffer.  In this case it
/// will not consume the ValueDest, so any user should reemit the ultimate
/// value through it with emitResult.
///
/// NOTE: This needs to be kept in sync with getDefinedMLValueIfExists.
LValue ValueDest::getLValueForResult(SMLoc loc, ASTType resultType,
                                     bool allowIncompatibleTypes,
                                     bool requireMLValue,
                                     ExprEmitter &emitter) {
  // Handle inference of a 'var' declaration's type.
  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    // If the result type has a non-materializable type, then we infer the var
    // to its materialized type.
    ASTType nmTarget = resultType.getNonmaterializableTarget(emitter.shared);
    ASTType materializedType = nmTarget ? nmTarget : resultType;

    // Update the VarDecl or GlobalVarDeclOp.
    Value typedRef;
    if (auto varOp = dyn_cast<VarDeclOp>(opDest)) {
      assert(isa<UnresolvedType>(varOp.getType().getElementType()) &&
             "Cannot resolve an already-resolved vardecl");
      varOp.getResult().setType(
          RefType::get(materializedType, varOp.getType().getLifetime()));
      typedRef = varOp.getResult();
    } else {
      auto globalOp = cast<GlobalVarDeclOp>(opDest);
      if (isa<UnresolvedType>(globalOp.getType()))
        globalOp.setType(materializedType);
      typedRef = emitter.builder->create<GlobalVarRefOp>(
          emitter.translateLocation(loc), globalOp);
    }
    // Now that we inferred the 'var' type, we can treat this like a normal
    // MLValue.
    representation = LValue(MLValue(typedRef));
  }

  // We have several cases where we can produce an LValue but it may have the
  // wrong type.  The client may be cool with this (when allowIncompatibleTypes
  // is true), but if not we generate a new temporary buffer.

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
      // If the client accepts any sort of LValue, then we succeed.
      if (!requireMLValue) {
        representation = LValueBufferTaken(); // Buffer taken!
        return lValue;
      }

      // Otherwise, we can only work if we have an MLValue in the correct
      // address space.
      if (auto mlVal = lValue.getIfMLValue();
          mlVal && lValue.getMValueType().isDefaultAddrSpace()) {
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
    bool isRegisterPassable =
        resultType.isRegisterPassable(loc, emitter.shared);
    emitter.emitError(loc, "cannot synthesize lvalue of ")
        << (isRegisterPassable ? "register-passable "
                               : "non-register-passable ")
        << "type " << resultType.getAsString(/*forDiag=*/true)
        << getContextMessage(emitter.paramContext);
    return {};
  }

  // If we're generating a memory location, use a required type if present or
  // the value type if not.
  ASTType slotType = resultType;
  if (auto requiredType = dyn_cast_or_null<ASTType>(representation)) {
    if (allowIncompatibleTypes || requiredType.isEqualCanon(slotType))
      slotType = requiredType;
  }

  // We model this as an mutable let value with a separately stored
  // initializer.  We return an LValue for it because this method is used
  // for the initialization.
  return MLValue(emitter.emitVarDecl("anonymous*", slotType,
                                     emitter.translateLocation(loc),
                                     VarDeclKind::Synthesized));
}

/// Return an MLValue for this destination of the specified type that we can
/// initialize. This uses and consumes the destination if it matches the type
/// of the value dest.
MLValue ValueDest::getMLValueForResult(SMLoc loc, ASTType resultType,
                                       ExprEmitter &emitter) {
  LValue lv =
      getLValueForResult(loc, resultType, /*allowIncompatibleTypes=*/false,
                         /*requireMLValue=*/true, emitter);
  if (!lv)
    return {};

  assert(lv.getIfMLValue());
  return lv.getIfMLValue();
}

/// Return true if this is an MLValue that could be in a non-default address
/// space.
bool ValueDest::isNonDefaultAddressSpace() const {
  if (LValue lValue = dyn_cast<LValue>(representation))
    if (MLValue refValue = lValue.getIfMLValue())
      if (!lValue.getMValueType().isDefaultAddrSpace())
        return true;
  return false;
}

//===----------------------------------------------------------------------===//
// ExprEmitter
//===----------------------------------------------------------------------===//

/// Emit an error about use of a dynamic value (the expression) in a context
/// that only allows parameter expressions.  This always returns a null
/// PValue.
PValue ExprEmitter::emitErrorForDynamicValueInParameter(const ExprNode *expr,
                                                        const char *message) {
  assert(paramContext != EC_InvalidContext &&
         "parameter context not set correctly");
  if (!message)
    message = "cannot use a dynamic value";
  emitError(expr->getLoc(), message)
      << getContextMessage(paramContext) << expr->getRange();
  return {};
}

/// Emit an error about use of a dynamic value (the expression) in a context
/// that only allows parameter expressions.  This always returns a null
/// PValue.
PValue ExprEmitter::emitErrorForDynamicValueInParameter(Location loc,
                                                        const char *message) {
  assert(paramContext != EC_InvalidContext &&
         "parameter context not set correctly");
  if (!message)
    message = "cannot use a dynamic value";
  emitError(loc, message) << getContextMessage(paramContext);
  return {};
}

//===----------------------------------------------------------------------===//
// Emission helpers for various value classifications.

CValue ExprEmitter::emitRValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value) // Already diagnosed error.
    return {};

  // If the value is still unresolved, materialize it.
  CValue cValue = value.ir.getIfCValue();
  if (!cValue) {
    cValue = emitCValue(value, dest);
    if (!cValue)
      return {};
  }

  // If this is already an RValue/PValue then we are done.
  if (auto rvRep = cValue.getIfRValue())
    return emitCResult(rvRep, value.expr, dest);

  // Otherwise, this is an LValue or BValue, emit a copy.
  return emitCopyOfValue({cValue, value.expr}, dest);
}

RValue ExprEmitter::emitRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  ValueDest dest(resultType, context);
  CValue result = emitRValue(value, dest);
  while (true) {
    if (!result) {
      dest.resetForError();
      return {};
    }
    // Typically emitRValue will return an RValue, but it might return a BValue.
    if (auto rv = result.getIfRValue())
      return rv;

    // It may return a BValue though (e.g. when accessing subfields with
    // computed lvalue bases), in which case we'll emit a copy of it.
    ValueDest copyDest(context);
    result = emitCopyOfValue({result, value.expr}, copyDest);
  }
}

CValue ExprEmitter::emitCValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  ValueDest dest(resultType, context);
  if (auto c = emitCValue(value, dest))
    return c;
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
  if (OverloadSetUValue overloads = value.ir.getIfOverloadSet()) {
    assert(overloads && "unknown overloaded value");
    return overloads->emitAsCValue(*this, dest);
  }

  // Otherwise we must have an initializer list.
  auto initValue = value.ir.getIfInitializer();
  assert(initValue && "Unknown UValue!");

  // We can't emit an initializer list without a contextual type.  See if we
  // have one.
  ASTType expectedType = dest.getExpectedTypeIfSpecified();
  if (!expectedType) {
    emitError(value.expr->getLoc(),
              "cannot emit initializer list without a contextual type");
    return {};
  }

  return emitConstructorCall(expectedType, CallOperands(initValue.get()),
                             value.expr, CallSyntax::kImplicitConvert, dest);
}

/// Emit an expression providing an immutable borrowed reference to a value.
BValue ExprEmitter::emitBValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value)
    return {};

  // Handle dynamic LValues by loading from them.
  if (auto lv = value.ir.getIfDLValue()) {
    value.ir = emitLoadOfLValue({lv, value.expr}, dest);
    if (!value.ir)
      return {};
  }

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (value.ir.getIfUValue()) {
    value.ir = emitCValue(value, dest);
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

  // Handle M*Value's by decaying to MBValue.
  if (value.ir.isMValue())
    value.ir = MBValue(value.ir.getMValueReference());

  // Decay RValue's into BValue's.
  if (auto srVal = value.ir.getIfSRValue()) // Decay SRValue -> SBValue
    value.ir = SBValue(srVal);
  else if (auto refVal = value.ir.getIfMRValue()) // Decay MRValue -> MBValue
    value.ir = MBValue(refVal);

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

/// Helper to check if we are trying to materialize a dynamic type value.
static bool emitErrorForMaterializingTypeValues(ExprEmitter &emitter,
                                                ASTExprAnd<PValue> value,
                                                ExprContext context) {
  TypedAttr attr = value.ir.get();
  if (isa<ModuleAttr>(attr) || !isTypeExpr(attr))
    return false;

  const ExprNode *expr = value.expr;
  InflightDiag diag = emitter.emitError(
      expr->getLoc(), "dynamic type values not permitted yet");
  if (context == EC_VarInit)
    diag << "; try creating an `alias` instead of a `var`";
  else if (context == EC_CallArgValue)
    diag << "; try passing types as a parameters instead of arguments";
  diag << expr->getRange();
  return true;
}

SRValue ExprEmitter::emitPValueToSRValue(ASTExprAnd<PValue> value,
                                         ExprContext context) {
  TypedAttr attr = value.ir.get();
  const ExprNode *expr = value.expr;

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (!builder) {
    emitErrorForDynamicValueInParameter(expr);
    return {};
  }

  // We don't allow materializing Type values yet.
  if (emitErrorForMaterializingTypeValues(*this, value, context))
    return {};

  Location location = expr->getLocation(*this);

  // If the value being materialized is itself parameterized, then we cannot
  // materialize it as an SSA value - there will be no way to bind parameters to
  // it.
  // TODO: We should have a general predicate from this provided by the KGEN
  // parameter utilities.
  if (auto signature = dyn_cast<LITSignatureType>(attr.getType())) {
    // If the value has any unbound parameters, they might be default arguments
    // or an variadic list that should be bound to an empty list.
    if (!signature.getParamTypes().empty()) {
      ParamBindings paramBindings(getScopeInfo());
      // Try to fully bind the signature, in case it can be made concrete with
      // default values, etc.
      auto [bindingAttr, _] = paramBindings.verifyBindings(signature);
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
      llvm::append_range(bindOperands, bindingAttr);
      attr = ParamOperatorAttr::get(POC::BindSignature, bindOperands);
    }

    // Reject unbound result parameters.
    if (!signature.getResultParamTypes().empty()) {
      emitError(expr->getLoc(),
                "cannot use parameterized function with result parameters ")
          << ASTType(attr.getType()) << expr->getRange();
      return {};
    }

    // Materialize signatures as closures.
    if (signature.isCapturing()) {
      emitError(
          expr->getLoc(),
          "TODO: capturing closures cannot be materialized as runtime values");
      return {};
    }
    return SRValue(builder->create<CreateClosureOp>(location, signature, attr,
                                                    ValueRange()));
  }

  // Otherwise, emit a generalized parameter constant.
  return SRValue(
      value.ir.getRValueType().isTrivial(value.expr->getLoc(), shared)
          ? Value(builder->create<ParamConstantOp>(location, value.ir))
          : builder->create<ParamMaterializeOp>(location, value.ir));
}

/// Emit any kind of PValue to an MLValue.
MBValue ExprEmitter::emitPValueToMLValue(ASTExprAnd<PValue> value, MLValue dest,
                                         ExprContext context) {
  // We don't allow materializing Type values yet.
  if (emitErrorForMaterializingTypeValues(*this, value, context))
    return {};

  // PValues don't have lifetimes and are immortal with respect to the compiler.
  // Emit a memcpy into the LValue. Creating an SSA value of the memory-only
  // type for the sake of memcpy is safe because the bulk store will ensure the
  // variable does not get promoted off the stack, and after struct lowering,
  // the type is erased down to its MLIR constituents anyways.
  Location loc = translateLocation(value.expr->getLoc());
  Value attr = emitPValueToSRValue(value, context);
  builder->create<RefStoreOp>(loc, attr, dest);
  return MBValue(dest);
}

MRValue ExprEmitter::emitPValueToMRValue(ASTExprAnd<PValue> value,
                                         ExprContext context) {
  PValue pvalue = value.ir;
  // We model this as an immutable let value with a separately stored
  // initializer.
  VarDeclOp var = emitVarDecl("anonymous*", pvalue.getType(),
                              translateLocation(value.expr->getLoc()),
                              VarDeclKind::Synthesized);
  if (!var)
    return {};
  if (!emitPValueToMLValue({pvalue, value.expr}, MLValue(var), context))
    return {};
  return MRValue(var);
}

SRValue ExprEmitter::emitSRValue(ASTExprAnd<AnyValue> anyValue,
                                 ExprContext context, ASTType resultType) {
  const ExprNode *expr = anyValue.expr;

  // Emit using resultType if present, and eliminate LValue/OverloadSetUValue's.
  RValue value = emitRValue(anyValue, context, resultType);
  if (!value)
    return {};

  if (!value.getRValueType().isRegisterPassable(expr->getLoc(), shared)) {
    emitError(expr->getLoc()) << "cannot load non-register passable type into "
                                 "SSA register (compiler bug, please report!)";
    return {};
  }

  // If we have a value in memory, use a LoadConsumeOp to load it.
  if (auto mrValue = value.getIfMRValue()) {
    if (!builder) {
      emitErrorForDynamicValueInParameter(expr);
      return {};
    }
    Value result =
        builder->create<LoadConsumeOp>(expr->getLocation(*this), mrValue);
    return SRValue(result);
  }

  // If this is already an SRValue, return it.
  if (auto rvalue = value.getIfSRValue())
    return rvalue;

  auto pValue = value.getIfPValue();
  assert(pValue && "must be PValue if register-passable and not SRValue");
  return emitPValueToSRValue({pValue, expr}, context);
}

MRValue ExprEmitter::emitMRValue(ASTExprAnd<AnyValue> value,
                                 ExprContext context) {
  auto rVal = emitRValue(value, context);
  if (!rVal)
    return {};

  if (auto mr = rVal.getIfMRValue())
    return mr;

  if (auto pv = rVal.getIfPValue())
    return emitPValueToMRValue({pv, value.expr}, context);

  // Promote SRValue to MRValue.
  if (SRValue srValue = rVal.getIfSRValue()) {
    Location argLoc = value.expr->getLocation(*this);
    VarDeclOp varOp = emitVarDecl("__mem_tmp__", srValue.getType(), argLoc,
                                  VarDeclKind::Synthesized);
    builder->create<RefStoreOp>(argLoc, srValue, varOp);
    return MRValue(varOp);
  }

  llvm_unreachable("unknown RValue");
}

/// This helper emits the specified value as an MBValue which has
/// memory-only representation, materializing PValues as needed. This
/// returns null if emission fails.
MBValue ExprEmitter::emitMBValue(ASTExprAnd<AnyValue> value,
                                 ExprContext context) {
  BValue bValue = emitBValue(value, context);
  if (!bValue)
    return {};

  if (auto mb = bValue.getIfMBValue())
    return mb;

  // Emit PValues to memory and promote to borrow.
  if (auto pValue = bValue.getIfPValue()) {
    auto mrVal = emitPValueToMRValue({pValue, value.expr}, context);
    if (!mrVal)
      return {};
    return MBValue(mrVal);
  }

  // Reject SBValue.
  emitError(value.expr->getLoc(),
            "cannot form reference to borrowed register value");
  return {};
}

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

  // Resolve any unresolved values using the result type.
  value.ir = emitCValue(value, context, resultType);

  // If this is a DLValue, see if it can be emitted as a PValue. PValues are
  // immutable, so try to load the DLValue in a parameter context.
  if (auto dl = value.ir.getIfDLValue()) {
    ValueDest dest(context);
    value.ir = dl->emitLoad(dest, *this);
    if (!value.ir) {
      dest.resetForError();
      return {};
    }
  }

  // If this is a parameter, return it.
  if (auto result = value.ir.getIfPValue())
    return result;

  // Otherwise diagnose this as "not a parameter" unless the value failed to
  // emit entirely.
  if (value.ir)
    emitErrorForDynamicValueInParameter(value.expr);
  return {};
}

/// This helper emits the specified expression as a 'ref' expression value,
/// and returns the value of RefType for the result.
/// This emits an error and returns null if emission fails.
Value ExprEmitter::emitRefValue(ASTExprAnd<AnyValue> value,
                                ExprContext context) {
  // If this is an RValue (including PValue's), put it into a memory box so
  // we can get its lifetime.
  if (auto rv = value.ir.getIfRValue()) {
    value.ir = emitMRValue(value, context);
    if (!value.ir)
      return {};
  }

  // Emit the DefArgumentWrapperDLValue as the underlying MBValue that it may
  // contain.
  if (auto dlValue = value.ir.getIfDLValue()) {
    if (MBValue underlying = dlValue->emitMBValueFromDefArgument(*this))
      value.ir = underlying;
  }

  // Otherwise we can't support other non-MValue's like borrowed registers or
  // other computed LValues.
  if (!value.ir.isMValue()) {
    emitError(value.expr->getLoc())
        << "cannot bind a non-memory value to a 'ref' argument"
        << getContextMessage(context);
    return {};
  }

  return value.ir.getMValueReference();
}

//===----------------------------------------------------------------------===//
// Emission helpers for various value classifications.

/// If needed, convert the specified value to the target destination type,
/// with a noop cast.  This is used to adjust inconsequential details of the
/// type or for simple things like upcasts.  This does not invoke constructors
/// or do other non-trivial conversions.
///
/// This produces an error and returns null on an invalid conversion.
AnyValue ExprEmitter::rebindValue(ASTExprAnd<AnyValue> value, Type destType) {
  // Materialize a parameter rebind.
  if (auto pvalue = value.ir.getIfPValue())
    return PValue(ParamOperatorAttr::get(POC::Rebind, pvalue.get(), destType));
  if (auto dlValue = value.ir.getIfDLValue()) {
    dlValue->elementType = destType;
    return dlValue;
  }

  // Cannot perform value rebind if only parameters are allowed.
  if (!builder)
    return emitErrorForDynamicValueInParameter(value.expr);

  // Materialize a rebind operation.
  auto rebind = [&](Value v) -> Value {
    if (v.getType() == destType)
      return v;

    // Reference casts use a special op for IR clarity.
    if (auto srcRefType = dyn_cast<RefType>(v.getType()))
      if (auto dstRefType = dyn_cast<RefType>(destType)) {
        // Make sure rebind isn't *introducing* reference mutability.
        assert(!(srcRefType.isMutableKnown(false) &&
                 dstRefType.isMutableKnown(true)) &&
               "Rebind is introducing mutability");
        assert(srcRefType.getAddressSpace() == dstRefType.getAddressSpace() &&
               "rebind cannot change address space");
      }
    return builder->create<RebindOp>(translateLocation(value.expr->getLoc()),
                                     destType, v);
  };

  if (auto refValue = value.ir.getIfMLValue())
    return MLValue(rebind(refValue));
  if (auto refValue = value.ir.getIfMRValue())
    return MRValue(rebind(refValue));
  if (auto refValue = value.ir.getIfMBValue())
    return MBValue(rebind(refValue));
  if (auto sbValue = value.ir.getIfSBValue())
    return SBValue(rebind(sbValue));

  auto srValue = value.ir.getIfSRValue();
  assert(srValue && "Unknown value kind");
  return SRValue(rebind(srValue));
}

PValue ExprEmitter::bindMLIRTypeToTrait(ASTExprAnd<CValue> value,
                                        TraitType trait) {
  // Only static vtables are supported right now.
  PValue typeValue = value.ir.getIfPValue();
  if (!typeValue) {
    emitError(value.expr->getLoc(), "existentials are not supported yet!");
    return {};
  }
  ASTType mlirType = typeValue.getIfTypeValue();

  SMLoc loc = value.expr->getLoc();
  ASTDecl &traitDecl = *ASTType(trait).getDecl(shared);
  // Make sure the body of the trait is resolved.
  if (failed(shared.declResolver->resolveFully(traitDecl, loc)))
    return {};

  // Use a special wrapper decl in the builtins as stubs.
  ASTDecl *wrapperDecl = shared.getBuiltinStubsMLIRType(loc).getDecl(shared);
  if (!wrapperDecl || !isa<StructDeclOp>(wrapperDecl)) {
    emitError(loc, "malformed builtin._stubs.__MLIRType");
    return {};
  }
  ASTType boundWrapper =
      cast<StructDeclOp>(wrapperDecl).bindReference({typeValue});

  SmallVector<VTableEntryAttr> vtable;
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    if (decls.empty() || !isa<LIT::FuncOp>(decls.front()))
      continue;
    for (ASTDecl *decl : decls) {
      // MLIR types are movable, copyable, and destructible only.
      switch (SpecialFunctionInfo::getKind(name)) {
      case SpecialFunctionKind::kMoveInit:
      case SpecialFunctionKind::kCopyInit:
      case SpecialFunctionKind::kDel:
        break;
      default:
        InflightDiag diag = emitError(loc, "cannot bind MLIR type ")
                            << mlirType << " to trait " << ASTType(trait);
        diag.attachNote(decl->getLoc())
            << "MLIR type cannot satisfy required trait function here";
        return {};
      }
      // We know the stub will provide exactly one overload for each allowed
      // trait requirement.
      auto ovSet = OverloadSet::lookup(getScopeInfo(), boundWrapper, name,
                                       value.expr, CallSyntax::kMethodCall);
      // Manually bind the type into the parameter list so the vtable entries
      // are specialized on the MLIR type.
      ovSet.paramBindings = ParamBindings::getForDeclaredType(
          getScopeInfo(), boundWrapper, value.expr);

      PValue callee = ovSet.getIfPValue();
      if (!callee) {
        emitError(loc, "internal error: MLIR type stub didn't resolve ")
            << name;
        return {};
      }
      vtable.push_back(VTableEntryAttr::get(name, callee));
    }
  }
  return TypeConstantAttr::get(mlirType, trait,
                               VTableAttr::get(getContext(), vtable));
}

namespace {

/// The signature for a trait requirement will have a Self parameter first whose
/// type is a TraitType for the trait it was found in.  We want to force
/// substitute a new parameter for the Self references even though it has a
/// different metatype.  This doesn't remove the parameter, that will be done
/// later.
struct TraitSelfBinder : public IndexParameterReplacer<TraitSelfBinder> {
  TypedAttr selfValue;

  TraitSelfBinder(TypedAttr selfValue) : selfValue(selfValue) {}

  // CRTP methods.
  Attribute tryReplace(Attribute attr, size_t depth) {
    // Replace a reference to $(0,0) with the new selfValue.
    auto paramRef = dyn_cast<ParamIndexRefAttr>(attr);
    if (!paramRef || paramRef.getIsResult() || paramRef.getIndex() != 0 ||
        paramRef.getDepth() + 1 != depth)
      return {};
    return selfValue;
  }
  Type tryReplace(Type type, size_t depth) { return {}; }
};
} // end anonymous namespace

PValue ExprEmitter::emitMetaTypeToTraitConversion(ASTExprAnd<CValue> value,
                                                  TraitType trait) {
  assert((isa<AnyStructType, TraitType>(value.ir.getType())) &&
         "should only be called on metatypes");

  // Only static vtables are supported right now.
  PValue typeValue = value.ir.getIfPValue();
  if (!typeValue) {
    emitError(value.expr->getLoc(), "existentials are not supported yet!");
    return {};
  }

  // Get the AnyStructType or the TraitType of the value that we're checking for
  // conversion to the trait type.
  ASTType metaType = typeValue.getRValueType();

  // Cannot bind parametric types to traits.
  if (auto anyStruct = dyn_cast<AnyStructType>(metaType)) {
    if (!anyStruct.getSignature().getParamTypes().empty()) {
      emitError(value.expr->getLoc(), "parametric type ")
          << anyStruct.getStructType()
          << " cannot bind to trait with missing parameters"
          << value.expr->getRange();
      return {};
    }
  }

  // Check that the struct or super trait implements the trait.
  ASTDecl *metaTypeDecl = metaType.getDecl(shared);
  if (!metaTypeDecl)
    return {}; // erroneous

  std::optional<InflightDiag> checkDiag;
  if (!metaTypeDecl->doesNominalTypeConformsTo(trait, checkDiag, shared)) {
    InflightDiag diag = emitError(value.expr->getLoc(), "cannot bind type ")
                        << ASTType(typeValue) << " to trait " << ASTType(trait)
                        << value.expr->getRange();
    if (checkDiag)
      diag.attachNote(metaTypeDecl->getLoc()) << std::move(*checkDiag);
    return {};
  }

  // Synthesize the vtable required for the trait from the struct. Make sure the
  // trait body is fully resolved so we know what the methods are.
  ASTDecl *traitDecl = ASTType(trait).getDecl(shared);
  if (failed(getDeclResolver().resolveFully(*traitDecl, value.expr->getLoc())))
    return {};

  // Determine if the conforming value is trivial or register passable.  If so,
  // this will affect the methods we can synthesize in conformance.  Values of
  // trait type will already have been erased to a memory type.
  bool rpTrivial = false;
  bool regPassable = false;
  if (auto structDeclOp = dyn_cast<StructDeclOp>(metaTypeDecl)) {
    rpTrivial = structDeclOp.isRegisterPassable();
    regPassable = structDeclOp.isRegisterPassableTrivial();
  }

  // Bind each trait requirement into vtable entries.
  SmallVector<VTableEntryAttr> vtable;
  for (auto &[name, requirementDecls] : traitDecl->getDeclsInScope()) {
    // Each entry can have multiple overloads in 'decls'.
    if (requirementDecls.empty() || !isa<LIT::FuncOp>(requirementDecls.front()))
      continue;

    // Find candidates in the implementing type (either a struct or trait) which
    // also may have multiple overloads.
    LookupResult result =
        shared.lookupAndResolveDecl(name, value.expr->getLoc(), *metaTypeDecl,
                                    /*searchParentScopes=*/false);
    ArrayRef<ASTDecl *> implFuncs = result.getIfSuccess();

    // Each requirement may be overloaded, resolve each individually.
    for (ASTDecl *expected : requirementDecls) {
      auto requirementFn = cast<LIT::FuncOp>(expected);

      // For any given requirement, the implementing type may have multiple
      // overloads.  Resolve which one we're using by forming an overload set
      // and filtering it.  Start by finding a set of param bindings in the
      // implementing function that get bound, including the self type if the
      // conforming type is a trait.

      // The requirement will have a Self parameter whose type will be of the
      // current trait.  In order to get types to line up, we need to force it
      // to the implementation type.  This changes the parameter value, but also
      // changes the metatype of the value.  To support this, we use a custom
      // replacer.
      LITSignatureType requirementSig =
          TraitSelfBinder(typeValue).replace(requirementFn.getFullSignature());

      // Form a set of bindings to plow into the impl signature.
      auto implBindings = ParamBindings::getForDeclaredType(
          getScopeInfo(), ASTType(typeValue), value.expr);

      // Bind the implicit T parameter on trait members to something with the
      // right metatype to keep the remapper happy.  We already replaced all
      // uses of the attr with TraitSelfBinder.
      SmallVector<TypedAttr> requirementParams;
      requirementParams.push_back(
          // NOTE: This is an UnknownAttr not an UnboundAttr.
          UnknownAttr::get(requirementSig.getParamTypes()[0]));

      ParserParamEvaluator evaluator(getDeclResolver(), requirementParams);
      for (Type type : requirementSig.getParamTypes().drop_front()) {
        auto unbound = UnboundAttr::get(evaluator.getReboundType(type));
        requirementParams.push_back(unbound);
        evaluator.addInputValue(unbound);
        implBindings.addPrechecked(value.expr, unbound);
      }
      requirementSig = requirementSig.getSpecializedSignature(
          requirementParams, value.expr->getLocation(*this));
      assert(requirementSig && "internal error substituting trait type");

      // Grab the matching function.
      OverloadSet ov(name, implFuncs, std::move(implBindings), value.expr,
                     CallSyntax::kMethodCallSynthetic);
      auto result = ov.filterOverloadSetForValueType(
          requirementSig, /*emitDiagnosticOnFailure=*/false);
      if (!result) {
        // Don't error out if name is for the thunk functions that will be
        // synthesized when conformance check happens.
        if (canSynthesizeIfMissing(name, rpTrivial, regPassable))
          continue;

        // The struct does not have the specified member and we cannot
        // synthesize it. Re-emit the error to get a diagnostic.
        (void)ov.filterOverloadSetForValueType(
            requirementSig, /*emitDiagnosticOnFailure=*/true);
        return {};
      }
      if (result.getType().mlirType != requirementSig)
        result =
            ParamOperatorAttr::get(POC::Rebind, result.get(), requirementSig);
      vtable.push_back(VTableEntryAttr::get(name, result));
    }
  }

  // Create the new type value with the vtable and the trait metatype.
  return TypeConstantAttr::get(ASTType(typeValue), trait,
                               VTableAttr::get(getContext(), vtable));
}

/// When emitting a result value, attempt to "refine" the value type by
/// evaluating 'apply' expressions in its type. Rebind the value if the type can
/// be further specialized.
static AnyValue refineResultValue(AnyValue value, const ExprNode *expr,
                                  ExprEmitter &emitter) {
  // Only CValues can be specialized. OverloadSetUValues don't have a type.
  auto cValue = value.getIfCValue();
  if (!cValue)
    return value;

  ParserParamEvaluator evaluator(emitter.getDeclResolver());
  Type valueType = cValue.getType();
  Type refinedType = evaluator.refine(valueType);
  if (refinedType == valueType)
    return value;

  return emitter.rebindValue({value, expr}, refinedType);
}

/// Emit the specified value into the current destination if present.  This
/// accepts (and silently propagates) null values.
///
/// Note that the `value` provided here may require an implicit conversion
/// into the destination slot, so the input may be memory-only and result be
/// register-passable (and visa-versa).
AnyValue ExprEmitter::emitResult(AnyValue value, const ExprNode *expr,
                                 ValueDest &dest) {
  if (!value) {
    dest.resetForError();
    return {};
  }
  ExprContext context = dest.getContext();

  // Attempt to further specialize the result value.
  value = refineResultValue(value, expr, *this);

  // If no destination is specified or it is just a contextual type hint, then
  // we can propagate the value directly.
  if (!dest.isSpecified() || isa<LValueInitializerType>(dest.representation)) {
    dest.representation = NullRepresentation();
    return value;
  }

  // If the value is still unresolved, materialize it into the destination.
  auto cValue = value.getIfCValue();
  if (!cValue)
    return emitCValue({value, expr}, dest);

  // OK, if there is a destination specified, handle them by converging the set
  // of value types we have.
  auto rvalueType = cValue.getRValueType();

  // If there is a known type for the destination but the value disagrees, emit
  // an implicit conversion directly into the destination.  This keeps values in
  // registers and avoids a "convert + clone" pair for memory->memory
  // conversions.
  if (ASTType requiredType =
          dest.resolveImpliedType(expr->getLoc(), rvalueType, *this)) {
    // If converting to a TypeCheckError type, then there is an
    // already-diagnosed error about this expression.
    if (requiredType.isTypeCheckErrorType()) {
      dest.resetForError();
      return {};
    }

    if (!requiredType.isEqualCanon(rvalueType)) {
      // If we are dealing with types that differ only pre-elaboration,
      // we insert a rebind.
      if (canConvertWithRebind(rvalueType, requiredType, shared)) {
        // The RValue types need to be rebound, but MValues have a level of
        // reference around them that we want to maintain.
        if (cValue.isMValue())
          requiredType = cValue.getMValueType().getWithElement(requiredType);

        // PValues of lifetime type have a special conversion.
        if (isa<LifetimeType>(requiredType) &&
            isa<LifetimeType>(cValue.getType()))
          if (auto pv = cValue.getIfPValue())
            value = LifetimeMutCastAttr::get(pv, requiredType);

        value = rebindValue({value, expr}, requiredType);
        return emitResult(value, expr, dest);
      }

      // If looking for a !lit.ref, allow convertible lit.ref values.
      if (isa<RefType>(requiredType) && cValue.isMValue() &&
          canConvertWithRebind(cValue.getMValueType(), requiredType, shared)) {
        value = rebindValue({value, expr}, requiredType);
        assert(value.isMValue() && "mvalues should rebind to mvalues");
        return emitResult(SRValue(value.getMValueReference()), expr, dest);
      }

      // Emit metatype conversions to trait types if the metatype implements the
      // specified trait.
      if (auto trait = dyn_cast<TraitType>(requiredType)) {
        if (isa<AnyStructType, TraitType>(rvalueType)) {
          PValue result = emitMetaTypeToTraitConversion({cValue, expr}, trait);
          return emitResult(result, expr, dest);
        }
        if (isa<TypeType>(rvalueType)) {
          PValue result = bindMLIRTypeToTrait({cValue, expr}, trait);
          return emitResult(result, expr, dest);
        }
        // If the source value is a parametric value of type 'AnyTrait[trait]'
        // then the elaborator will turn it into something that conforms to
        // 'trait' and a simple rebind is enough.
        if (auto sourceTraitMT = dyn_cast<AnyTraitType>(rvalueType)) {
          if (sourceTraitMT.getTraitType() == trait) {
            value = rebindValue({cValue, expr}, requiredType);
            return emitResult(value, expr, dest);
          }
        }
      }

      // Handle conversions between lifetimes and lifetime sets.
      if (isa<LifetimeType, LifetimeSetType>(rvalueType) &&
          isa<LifetimeType, LifetimeSetType>(requiredType)) {
        // This can only be done in the parameter domain.
        if (TypedAttr value = cValue.getIfPValue()) {
          if (isa<LifetimeType>(rvalueType)) {
            value = LifetimeSetAttr::get(value,
                                         cast<LifetimeSetType>(requiredType));
          } else {
            value = LifetimeSetUnionAttr::get(value,
                                              cast<LifetimeType>(requiredType));
          }
          return emitResult(value, expr, dest);
        }
      }

      // We disable implicit conversions to prevent converting T -> S -> U in
      // one step, and to avoid infinite conversion cycles.
      return emitConstructorCall(requiredType, CallOperands({{cValue, expr}}),
                                 expr, CallSyntax::kImplicitConvert, dest,
                                 /*allowImplicitConversion=*/false);
    }
  }

  // If the destination is just a required type, then we now know it must agree
  // and therefore don't need to do anything more.
  if (isa<ASTType>(dest.representation)) {
    dest = ValueDest(context); // Resolved the ValueDest;
    return cValue;
  }

  // If this destination was an LValue whose buffer was already taken to be
  // filled in by a client, then this is just completing the transaction.
  if (isa<LValueBufferTaken>(dest.representation)) {
    dest = ValueDest(context); // Resolved the ValueDest;
    // The client directly filled in an LValue we provided which is great, but
    // that LValue we provided took ownership of the value, so we need to return
    // the result as a borrow, not an owned reference.
    auto memValue = value.getIfMRValue();
    assert(memValue && "Must be an MRValue providing result");
    return MBValue(memValue);
  }

  // We know we have an RValue/BValue and the destination is some kind of
  // LValue.  Emit the dest to figure out where to store it.
  LValue destLV = dest.getLValueForResult(expr->getLoc(), rvalueType,
                                          /*allowIncompatibleTypes=*/true,
                                          /*requireMLValue=*/false, *this);
  if (!destLV) {
    dest.resetForError();
    return {};
  }

  // This will have completely resolved all the ValueDest possibilities.
  assert(!dest.isSpecified() || isa<LValueBufferTaken>(dest.representation));
  dest = ValueDest(context);

  // Finally, store the value into the lvalue.
  return emitStoreToLValue({cValue, expr}, destLV, context);
}

CValue ExprEmitter::emitCResult(CValue value, const ExprNode *expr,
                                ValueDest &dest) {
  // Emitting a CValue always produces a CValue.
  auto result = emitResult(value, expr, dest);
  assert((!result || result.getIfCValue()) &&
         "emitting a CValue as a result should always produce a CValue");
  return result.getIfCValue();
}

/// Emit the specified expression into the specified destination.
AnyValue ExprEmitter::emitExpr(const ExprNode *expr, ValueDest &dest) {
  assert(expr && "cannot emit a null node");
  if (auto result = expr->emitIR(dest, *this))
    return result;
  dest.resetForError();
  return {};
}

AnyValue ExprEmitter::emitExpr(const ExprNode *expr, ExprContext context,
                               ASTType resultType) {
  ValueDest dest(resultType, context);
  return emitExpr(expr, dest);
}

RValue ExprEmitter::emitExprRValue(const ExprNode *expr, ExprContext context,
                                   ASTType resultType) {
  return emitRValue({emitExpr(expr, context, resultType), expr}, context,
                    resultType);
}

CValue ExprEmitter::emitExprCValue(const ExprNode *expr, ExprContext context,
                                   ASTType resultType) {
  return emitCValue({emitExpr(expr, context, resultType), expr}, context);
}

SRValue ExprEmitter::emitExprSRValue(const ExprNode *expr, ExprContext context,
                                     ASTType resultType) {
  return emitSRValue({emitExpr(expr, context, resultType), expr}, context,
                     resultType);
}

PValue ExprEmitter::emitExprPValue(const ExprNode *expr, ExprContext context,
                                   ASTType resultType) {
  // Clear the builder to indicate that an PValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder, {});
  llvm::SaveAndRestore savedContext(paramContext, context);

  // Emit the expression using the contextual type if present.
  AnyValue rep = emitExpr(expr, context, resultType);
  return emitPValue({rep, expr}, context);
}

LValue ExprEmitter::emitExprLValue(const ExprNode *expr, ValueDest &dest) {
  AnyValue anyValue = expr->emitIR(dest, *this);
  if (!anyValue)
    return {}; // Error already diagnosed.
  return emitLValue({anyValue, expr}, dest);
}
CValue ExprEmitter::emitLoadOfLValue(ASTExprAnd<LValue> value,
                                     ValueDest &dest) {
  // If this is a computed LValue emit call to the "getter".
  if (auto dlValue = value.ir.getIfDLValue())
    return dlValue->emitLoad(dest, *this);

  // Decay a stored LValue to an MBValue.
  auto ref = value.ir.getIfMLValue();
  assert(ref && "unknown lvalue kind");
  // Emit a non-consuming __copyinit__ or load of the value.
  return emitCopyOfValue({MBValue(ref), value.expr}, dest);
}

/// Emit a copy of the specified value, producing a new owned instance of the
/// value in the specified destination.  This returns an RValue if
/// there is no consuming dest, otherwise a BValue.
CValue ExprEmitter::emitCopyOfValue(ASTExprAnd<CValue> value, ValueDest &dest) {
  ASTType valueType = value.ir.getRValueType();
  SMLoc exprLoc = value.expr->getLoc();
  if (!value.ir)
    return {};

  // Resolve away DLValue's.
  if (auto dlValue = value.ir.getIfDLValue())
    return dlValue->emitLoad(dest, *this);

  // If the value is PValue and register passable, then we can materialize a
  // unique value directly into a register.
  if (auto pValue = value.ir.getIfPValue()) {
    if (valueType.isRegisterPassable(exprLoc, shared)) {
      value.ir = emitPValueToSRValue({pValue, value.expr}, dest.context);
      return emitCResult(value.ir, value.expr, dest);
    }
  }

  // If the value's type is trivial then we don't need to do anything except
  // convert to an RValue and emit to the destination.
  if (valueType.isTrivial(exprLoc, shared)) {
    // It is ok to upgrade SBValue to SRValue for trivial types.
    if (auto sbVal = value.ir.getIfSBValue())
      value.ir = SRValue(sbVal);

    // All trivial types are register passable right now, so we can load memory
    // values and produce an SRValue.
    if (value.ir.isMValue()) {
      if (!builder) {
        emitErrorForDynamicValueInParameter(value.expr);
        return {};
      }
      Value address = value.ir.getMValueReference();
      Value result =
          builder->create<RefLoadOp>(value.expr->getLocation(*this), address);
      value.ir = SRValue(result);
    }

    return emitCResult(value.ir, value.expr, dest);
  }

  // Otherwise, we'll need to invoke the copyinit method which will take the
  // destination as inout, so we're dealing with a memory case.

  // Memory-only copyinit will take the destination as address space zero, so
  // we need to reject ValueDest's expecting it in GPU memory.
  if (dest.isNonDefaultAddressSpace()) {
    emitError(exprLoc, "value of type ")
        << valueType << " cannot be copied into a non-default address space"
        << value.expr->getRange();
    return {};
  }

  // __copyinit__ has signature: `(inout self, existing: Self)`.
  MLValue destBuffer = dest.getMLValueForResult(exprLoc, valueType, *this);
  if (!destBuffer)
    return {};

  if (auto pValue = value.ir.getIfPValue())
    return emitPValueToMLValue({pValue, value.expr}, destBuffer, dest.context);

  if (!valueType.isCopyable(exprLoc, shared)) {
    if (valueType.isMovableFrom(value, shared) &&
        !valueType.isRegisterPassable(exprLoc, shared)) {
      emitError(exprLoc, "value of type ")
          << valueType
          << " can only be moved, but source value can only be copied"
          << value.expr->getRange();
    } else {
      emitError(exprLoc) << valueType
                         << " is not copyable because it has no '__copyinit__'"
                         << value.expr->getRange();
    }
    return {};
  }

  CallOperands operands({ASTExprAnd<AnyValue>{destBuffer, value.expr}, value});
  ValueDest copyDest(dest.getContext());
  if (!emitNamedMethodCall("__copyinit__", std::move(operands), copyDest,
                           CallSyntax::kImplicitConvert, value.expr))
    return {};
  // If we required an implicit conversion, make sure it happens.
  return emitCResult(MRValue(destBuffer), value.expr, dest);
}

BValue ExprEmitter::emitStoreToLValue(ASTExprAnd<CValue> value, LValue destLV,
                                      ExprContext context) {
  // Convert nonmaterializables.
  if (auto nmTarget =
          value.ir.getRValueType().getNonmaterializableTarget(shared)) {
    if (nmTarget.isEqualCanon(destLV.getRValueType())) {
      // If the destination is an MLValue with a matching type, then just
      // materialize directly into it and return instead of allocating a
      // temporary if the conversion constructor requires one.
      MLValue destML = destLV.getIfMLValue();
      ValueDest nmConversionDest =
          destML ? ValueDest(destML, context) : ValueDest(context);
      CValue nmConversionVal =
          emitConstructorCall(nmTarget, CallOperands({value}), value.expr,
                              CallSyntax::kIndirectCall, nmConversionDest,
                              /*allowImplicitConversion=*/true);
      if (destML)
        return MBValue(destML);
      value = {nmConversionVal, value.expr};
    }
  }

  if (!value.ir.getRValueType())
    return {};

  assert(value.ir.getRValueType().isEqualCanon(destLV.getRValueType()) &&
         "Types should match");

  // If this is a computed LValue, then perform a writeback.
  if (auto dlValue = destLV.getIfDLValue()) {
    // If the value itself is an LValue, emit a load so we can call the setter.
    if (auto valueLV = value.ir.getIfLValue()) {
      ValueDest loadDest(context);
      value.ir = emitLoadOfLValue({valueLV, value.expr}, loadDest);
      if (!value)
        return {};
    }

    // Then store into the dest DLValue.
    {
      llvm::SaveAndRestore savedContext(paramContext, context);
      dlValue->emitStore(value, *this);
    }

    // Decay the input value to a BValue since ownership was taken by the store.
    return emitBValue(value, context, {});
  }

  ASTType valueType = value.ir.getRValueType();
  SMLoc exprLoc = value.expr->getLoc();

  // If the input is an LValue/BValue (incl PValue) that we don't own, or if it
  // has no __moveinit__, then copy it into the destination.
  if (!value.ir.getIfRValue() || !valueType.isMovableFrom(value, shared)) {
    // If the value isn't either copy or movable from the source, but the source
    // value is an RValue, then this is because the type isn't implementing
    // either the copy or move init.  Complain precisely, instead of just
    // complaining about copying.
    if (!valueType.isCopyable(exprLoc, shared) && value.ir.getIfRValue() &&
        !value.ir.getIfPValue()) {
      emitError(exprLoc) << valueType
                         << " is not copyable or movable because it has no "
                            "'__copyinit__' or '__moveinit__' member"
                         << value.expr->getRange();
      return {};
    }

    ValueDest dest(destLV, context);
    auto result = emitCopyOfValue(value, dest);
    assert((!result || result.getIfBValue()) &&
           "dest specified, so this should return BValue");
    dest.resetForError();
    return result.getIfBValue();
  }

  // Otherwise this is a movable RValue that we own and we have an MLValue
  // destination.
  MLValue destRef = destLV.getIfMLValue();
  assert(destRef && "No other known LValue");

  // If it is a register passable, assign with a store.
  if (valueType.isRegisterPassable(exprLoc, shared)) {
    // Materialize a PValue or load a MRValue if present.
    SRValue val = emitSRValue(value, context, valueType);
    if (!val)
      return {};
    if (!builder) {
      emitErrorForDynamicValueInParameter(value.expr);
      return {};
    }
    // Store the value to memory.  StoreOp takes ownership of the input SRValue.
    builder->create<LIT::RefStoreOp>(translateLocation(value.expr->getLoc()),
                                     val, destRef);

    return SBValue(val);
  }

  if (auto pvalue = value.ir.getIfPValue())
    return emitPValueToMLValue({pvalue, value.expr}, destRef, context);

  // Otherwise, assign with a move constructor.  We own the RValue, so prefer
  // to use __moveinit__ if present.
  if (shared.typeHasMember(valueType, "__moveinit__", value.expr->getLoc())) {
    // `__moveinit__(inout self, owned existing: Self)`.
    ASTExprAnd<AnyValue> operands[] = {
        ASTExprAnd<AnyValue>{destRef, value.expr}, value};
    ValueDest moveDest(context);
    if (!emitNamedMethodCall("__moveinit__", {operands}, moveDest,
                             CallSyntax::kImplicitConvert, value.expr))
      return {};
    return MBValue(destRef);
  }

  // Otherwise, we have to move this thing but don't have a move constructor!
  emitError(value.expr->getLoc())
      << "cannot transfer value into destination, because " << valueType
      << " doesn't implement `__moveinit__`";
  return {};
}

//===----------------------------------------------------------------------===//
// Emission helpers for specific value types.

ASTType ExprEmitter::emitExprType(const ExprNode *expr, bool allowUnbound) {
  // We have two ambiguous expressions that can either be types or dynamic
  // values: an empty tuple () and None.  In a type context, we want to treat
  // these as types, and not dynamic values.  Sniff these out to see if we have
  // them.
  const ExprNode *innerExpr = expr->getWithoutParens();
  if (innerExpr->kind == ExprNode::kNoneLiteral)
    return shared.getNoneType();
  if (innerExpr->isEmptyTuple())
    return getBuiltinTupleInstantiation(expr->getLoc(), {});

  auto value = emitExprPValue(expr, EC_Type);
  if (!value)
    return {};

  ASTType type = value.getIfTypeValue();
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

  // If the caller accepts a fully unbound type and the type is unbound, return
  // it now without verifying the bindings.
  if (allowUnbound)
    return type;

  auto structDecl = dyn_cast<StructDeclOp>(decl);
  if (!structDecl)
    return type;

  // Build up a ParamBindings set to validate and check the bindings. Skip
  // unbound values.
  ParamBindings paramBindings(getScopeInfo());
  for (TypedAttr binding : type.getParamBindings())
    paramBindings.addPrechecked(expr, binding);

  // Check the existing bindings against the full signature of the type and make
  // sure it is fully bound.
  ParameterExprArrayAttr bindingValuesAttr = paramBindings.verifyBindings(
      structDecl, structDecl.getSignature(), expr->getLoc(), /*partial=*/false);
  if (!bindingValuesAttr)
    return {};

  // If verifyBindings changed the bindings set, then we may have had an
  // empty varargs list or something.  Rebind the StructType.
  if (bindingValuesAttr.getValue() != type.getParamBindings())
    type = structDecl.bindReference(bindingValuesAttr);
  return type;
}

RValue ExprEmitter::emitI1(ASTExprAnd<CValue> value, ExprContext context) {
  if (!value.ir)
    return {};

  ASTType valueRValueType = value.ir.getRValueType();

  // If this is already an 'i1', then we're done.
  if (valueRValueType.mlirType.isInteger(1))
    return emitRValue(value, context);

  // TODO: Python manual includes this off-hand comment:
  // Also, an object that doesn’t define a __bool__() method and whose __len__()
  // method returns zero is considered to be false in a Boolean context.

  // Check for the presence of a __mlir_i1__ method.  If it exists, we can avoid
  // a redundant call to __bool__ for Bool types.
  if (!shared.typeHasMember(valueRValueType, "__mlir_i1__",
                            value.expr->getLoc())) {
    // Use the __bool__ method to convert the user defined type to
    // something that is a Bool or other type that implements __mlir_i1__.
    ValueDest boolDest(context);
    value.ir =
        emitNamedMethodCall("__bool__", {{{value.ir, value.expr}}}, boolDest,
                            CallSyntax::kImplicitConvert, value.expr);
  }

  // Then we use __mlir_i1__ to convert to an i1 value.
  ValueDest boolDest(context);
  CValue litBoolCall =
      emitNamedMethodCall("__mlir_i1__", {{{value.ir, value.expr}}}, boolDest,
                          CallSyntax::kImplicitConvert, value.expr);

  return emitRValue({litBoolCall, value.expr}, context);
}

RValue ExprEmitter::emitExprI1(const ExprNode *condExpr, ExprContext context) {
  return emitI1({emitExprCValue(condExpr, context), condExpr}, context);
}

CValue ExprEmitter::emitIndex(ASTExprAnd<AnyValue> value, ExprContext context) {
  ValueDest dest(context);
  return emitNamedMethodCall("__index__", {value}, dest,
                             CallSyntax::kMethodCall, value.expr);
}

CValue ExprEmitter::emitMLIRIndex(ASTExprAnd<AnyValue> value,
                                  ExprContext context) {
  // If the value is already of index type, just use it.
  if (CValue cvalue = value.ir.getIfCValue())
    if (isa<IndexType>(cvalue.getRValueType().mlirType))
      return cvalue;

  CValue index = emitIndex(value, context);
  if (!index)
    return {};

  // If the value is already of index type, just use it.
  if (isa<IndexType>(index.getRValueType().mlirType))
    return index;

  ValueDest dest(context);
  return emitNamedMethodCall("__mlir_index__", {{{index, value.expr}}}, dest,
                             CallSyntax::kMethodCall, value.expr);
}

CValue ExprEmitter::emitMLIRIndex(const ExprNode *expr, ExprContext context) {
  return emitMLIRIndex({emitExprCValue(expr, context), expr}, context);
}

/// This returns an instance of Tuple[...] with the specified element types
/// installed.
ASTType ExprEmitter::getBuiltinTupleInstantiation(llvm::SMLoc loc,
                                                  ArrayRef<Type> elements) {
  auto tupleType = shared.getBuiltinTupleType(declScope, loc);
  if (tupleType.isTypeCheckErrorType())
    return {};
  ASTDecl *typeDecl = ASTType(tupleType).getDecl(shared);
  auto structOp = dyn_cast_or_null<StructDeclOp>(typeDecl);
  if (!structOp) {
    emitError(loc, "internal error: Tuple type not found or not a struct");
    return {};
  }

  SyntheticNode tmpExpr(loc);
  ParamBindings bindings(getScopeInfo());
  for (ASTType elt : elements)
    bindings.add(&tmpExpr, PValue(elt));

  // Check the bindings.
  auto metaType = cast<AnyStructType>(tupleType.getMetaType());
  auto bindingsAttr = bindings.verifyBindings(structOp, metaType.getSignature(),
                                              loc, /*partial=*/false);
  if (!bindingsAttr)
    return {};

  // Ok, we succeeded at reparameterizing the type.
  return ASTType(BindTypeAttr::get(PValue(tupleType), bindingsAttr));
}

//===----------------------------------------------------------------------===//
// Return emission helpers.

MLValue ExprEmitter::findNearestErrorSlot() {
  assert(builder && "cannot raise in a context without a builder");
  Operation *opForRaise = findOpProcessingRaise(builder->getInsertionBlock());
  // Return null to indicate that the current context cannot raise.
  if (!opForRaise)
    return {};

  // In a raising function, the error slot is always the second last argument.
  if (auto func = dyn_cast<LIT::FuncOp>(opForRaise)) {
    return func.getArguments()[func.getNumArguments() -
                               func.getSignature().getErrorSlotOffset()];
  }
  // Otherwise, the error slot is carried by the surrounding try op.
  return cast<LIT::TryOp>(opForRaise).getErr();
}

void ExprEmitter::emitNormalReturn(ImplicitLocOpBuilder &builder, Value value,
                                   const ASTDecl &funcDecl) {
  auto func = cast<LIT::FuncOp>(funcDecl);
  emitNormalReturn(builder, value, func);
}

void ExprEmitter::emitNormalReturn(ImplicitLocOpBuilder &builder, Value value,
                                   LIT::FuncOp func) {
  bool markLastArgDestroyed = false;
  switch (func.getSpecialFunctionKind()) {
  default:
    break;
  /// In the __del__ method for a struct, we need to mark 'self' as being
  /// destroyed before any return operation.
  case SpecialFunctionKind::kDel:
    assert(func.getBody()->getNumArguments() == 1 &&
           "__del__ should have one argument");
    markLastArgDestroyed = true; // Mark 'self' destroyed.
    break;

  /// In the __moveinit__ method for a struct, we need to mark 'existing' as
  /// being destroyed before any return operation if it is owned convention.
  case SpecialFunctionKind::kMoveInit:
    assert(func.getBody()->getNumArguments() == 2 &&
           "__moveinit__ should have two arguments");
    markLastArgDestroyed = true; // Mark 'existing' destroyed.
    break;
  }

  if (markLastArgDestroyed) {
    Value argToDestroy = func.getBody()->getArguments().back();
    // If this is a @register_passable type, the value must be stored
    // in a box and we want to treat the box as the thing that we track.
    // CheckLifetimes doesn't track register values field sensitively, so there
    // is no way to say that the full object bit is dead in a SRValue.
    if (func.getSignature().getArgConventions().back() ==
        ArgConvention::OwnedInReg) {
      // Find the single thing that got stored to, ignoring debug.value ops.
      Value storedMem;
      for (auto user : argToDestroy.getUsers()) {
        if (isa<DebugInfo::ValueOp>(user))
          continue;
        assert(!storedMem && "Should only have a single store");
        storedMem = cast<LIT::RefStoreOp>(user).getRef();
      }
      // If we found it, then ownership has already transfered to the memory
      // object, so track it instead of the argument.
      assert(storedMem && "local value box for OwnedInReg self not found");
      argToDestroy = storedMem;
    }
    builder.create<LIT::OwnershipMarkDestroyedOp>(argToDestroy);
  }

  // Finally we emit a normal return with lit.return.
  builder.create<LIT::ReturnOp>(value);
}

//===--------------------------------------------------------------------===//
// Var/let emission helpers.

VarDeclOp ExprEmitter::emitVarDecl(const Twine &name, Type type, Location loc,
                                   VarDeclKind kind) {
  if (!builder) {
    emitErrorForDynamicValueInParameter(loc);
    return {};
  }
  StringAttr lifetimeAttr = declScope.mangleParamName(name);
  return builder->create<VarDeclOp>(loc, type, name.str(), lifetimeAttr, kind);
}

VarDeclOp ExprEmitter::emitVarDecl(StringAttr name, Type type, Location loc,
                                   VarDeclKind kind) {
  return emitVarDecl(name.strref(), type, loc, kind);
}

/// Create a mutable VarDecl for a function argument that captures its value.
/// argValue specifies the argument with the correct valuetype.
VarDeclOp ExprEmitter::makeArgLValueVarSlot(CValue argValue, StringAttr argName,
                                            SMLoc loc) {
  // Emit the initializer expression into the slot.
  VarDeclOp varDecl = emitVarDecl(argName, argValue.getRValueType(),
                                  translateLocation(loc), VarDeclKind::Arg);

  // Expr to provide location information.
  ValueDest dest(MLValue(varDecl), EC_OwnedRegArgShadow);
  if (emitBValue({argValue, SyntheticNode(loc)}, dest))
    return varDecl;

  // This can fail if not copyable/movable.
  dest.resetForError();
  return {};
}
