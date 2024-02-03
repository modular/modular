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
#include "MojoUtils.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
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
  case EC_OwnedRegArgShadow:
    return " in owned argument shadow";
  case EC_VarArgArgument:
    return " in vararg argument compiler implementation internals";
  case EC_DefaultParam:
    return " in default parameter";
  case EC_BoolCondition:
    return " in boolean condition";
  case EC_CondExpr:
    return " in 'if' expression value";
  case EC_BoolParamCondition:
    return " in '@parameter if' condition";
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
  case EC_CaptureCopy:
    return " in capture-by-copy";
  case EC_Decorator:
    return " in decorator";
  case EC_MutabilitySpec:
    return " in reference mutability specifier";
  case EC_LifetimeSpec:
    return " in lifetime specifier";
  case EC_Trait:
    return " in trait conformance checking";
  }
  llvm_unreachable("invalid expr context");
}

//===----------------------------------------------------------------------===//
// ValueDest
//===----------------------------------------------------------------------===//

ValueDest::ValueDest(VarLetDeclOp dest, ExprContext context)
    : representation(dest.getOperation()), context(context) {}

ValueDest::ValueDest(GlobalVarDeclOp dest, ExprContext context)
    : representation(dest.getOperation()), context(context) {}

void ValueDest::dump() const {
  auto &os = llvm::errs() << "ValueDest context=" << (int)context
                          << " destination = ";

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
    if (existingValueType) {
      if (ASTType nmTarget = ASTType(existingValueType)
                                 .getNonmaterializableTarget(emitter.shared))
        dest = ValueDest(LValueInitializerType{nmTarget}, context);
      else
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
      representation = NullRepresentation(); // Consumed!
    }
  }

  // Check for the simple case.
  if (LValue lValue = dyn_cast<LValue>(representation)) {
    if (MLValue refValue = lValue.getIfMLValue()) {
      if (lValue.getRValueType().isEqualCanon(resultType))
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
  // If we are inferring the type for a var or let declaration, then we can
  // always succeed and consume this ValueDest.
  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    representation = LValueBufferTaken(); // Buffer used!
    ASTType nmTarget = resultType.getNonmaterializableTarget(emitter.shared);
    ASTType materializedType = nmTarget ? nmTarget : resultType;

    if (auto varOp = dyn_cast<VarLetDeclOp>(opDest)) {
      assert(isa<UnresolvedType>(varOp.getType().getElementType()) &&
             "Cannot resolve an already-resolved vardecl");
      varOp.getResult().setType(
          RefType::get(materializedType, varOp.getType().getLifetime()));
      return MLValue(varOp);
    }
    auto globalOp = cast<GlobalVarDeclOp>(opDest);
    if (isa<UnresolvedType>(globalOp.getType()))
      globalOp.setType(materializedType);

    return MLValue(emitter.builder->create<GlobalVarRefOp>(
        emitter.translateLocation(loc), globalOp));
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
      if (!requireMLValue || lValue.getIfMLValue()) {
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
  // TODO(autopromotion).
  ASTType slotType = resultType;
  if (auto requiredType = dyn_cast_or_null<ASTType>(representation)) {
    if (allowIncompatibleTypes || requiredType.isEqualCanon(slotType))
      slotType = requiredType;
  }

  // We model this as an mutable let value with a separately stored
  // initializer.  We return an LValue for it because this method is used
  // for the initialization.
  return MLValue(emitter.emitVarLetDecl("anonymous*", slotType,
                                        emitter.translateLocation(loc),
                                        VarLetDeclKind::Synthesized));
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

//===----------------------------------------------------------------------===//
// ExprEmitter
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

//===----------------------------------------------------------------------===//
// Emission helpers for various value classifications.

CValue ExprEmitter::emitRValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value) // Already diagnosed error.
    return {};

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (auto overloads = value.ir.getIfOverloadSetUValue()) {
    value.ir = overloads->emitAsCValue(*this, dest);
    if (!value.ir)
      return {};
  }

  CValue cValue = value.ir.getIfCValue();
  assert(cValue && "OverloadSetUValue handled above");

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
  OverloadSetUValue overloads = value.ir.getIfOverloadSetUValue();
  assert(overloads && "unknown overloaded value");
  return overloads->emitAsCValue(*this, dest);
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
  // Handle M*Value's by decaying to MBValue.
  if (value.ir.isMValue())
    value.ir = MBValue(value.ir.getMValueReference());

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (auto overloads = value.ir.getIfOverloadSetUValue()) {
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
  if (auto signature = dyn_cast<LITSignatureType>(attr.getType())) {
    // If the value has any unbound parameters, they might be default arguments
    // or an variadic list that should be bound to an empty list.
    if (!signature.getParamTypes().empty()) {
      ParamBindings paramBindings(*this);
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
  }

  Location location = expr->getLocation(*this);

  // Materialize signatures as closures.
  if (auto sig = dyn_cast<SignatureType>(attr.getType())) {
    if (sig.isCapturing()) {
      emitError(
          expr->getLoc(),
          "TODO: capturing closures cannot be materialized as runtime values");
      return {};
    }
    return SRValue(
        builder->create<CreateClosureOp>(location, sig, attr, ValueRange()));
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
  // PValues don't have lifetimes and are immortal with respect to the compiler.
  // Emit a memcpy into the LValue. Creating an SSA value of the memory-only
  // type for the sake of memcpy is safe because the bulk store will ensure the
  // variable does not get promoted off the stack, and after struct lowering,
  // the type is erased down to its MLIR constituents anyways.
  Location loc = translateLocation(value.expr->getLoc());
  Value attr = value.ir.getRValueType().isTrivial(value.expr->getLoc(), shared)
                   ? Value(builder->create<ParamConstantOp>(loc, value.ir))
                   : builder->create<ParamMaterializeOp>(loc, value.ir);
  builder->create<RefStoreOp>(loc, attr, dest);
  return MBValue(dest);
}

MRValue ExprEmitter::emitPValueToMRValue(ASTExprAnd<PValue> value,
                                         ExprContext context) {
  PValue pvalue = value.ir;
  // We model this as an immutable let value with a separately stored
  // initializer.
  VarLetDeclOp var = emitVarLetDecl("anonymous*", pvalue.getType(),
                                    translateLocation(value.expr->getLoc()),
                                    VarLetDeclKind::Synthesized);
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
    VarLetDeclOp varOp = emitVarLetDecl("__mem_tmp__", srValue.getType(),
                                        argLoc, VarLetDeclKind::Synthesized);
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
  if (auto pValue = bValue.getIfPValue())
    return emitPValueToMRValue({pValue, value.expr}, context);

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

  // If this is an OverloadSetUValue, it must resolve to a single entry.
  if (auto overloads = value.ir.getIfOverloadSetUValue()) {
    ValueDest dest(context);
    value.ir = overloads->emitAsCValue(*this, dest);
    if (!value.ir)
      return {};
  }

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

  // Otherwise diagnose this as "not a parameter".
  emitErrorForDynamicValueInParameter(value.expr);
  return {};
}

//===----------------------------------------------------------------------===//
// Type conversion helpers.

/// Return true if the given type explicitly implements the trait.
static bool checkExplicitConformance(TraitType trait, ASTDecl *typeDecl) {
  ArrayRef<TypeLineageAttr> parentTypes;
  if (auto structOp = dyn_cast<StructDeclOp>(typeDecl))
    parentTypes = structOp.getParentTypes();
  else
    parentTypes = cast<TraitDeclOp>(typeDecl).getParentTypes();
  return llvm::find_if(parentTypes, [trait](TypeLineageAttr type) {
           return type.getType() == trait;
         }) != parentTypes.end();
}

/// Return true if the MLIR type can implicitly conform to the trait.
static bool checkImplicitConformance(SharedState &shared, SMLoc loc,
                                     TraitType trait) {
  ASTDecl &traitDecl = *ASTType(trait).getDecl(shared);
  // Make sure the body of the trait is resolved.
  if (failed(shared.declResolver->resolveFully(traitDecl, loc)))
    return false; // an error was emitted
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      auto traitFn = dyn_cast<LIT::FuncOp>(*decl);
      // Skip any children that aren't methods or are inherited. This could be
      // an alias.
      if (!traitFn || traitFn.getIsInherited())
        continue;
      // MLIR types are movable, copyable, and destructible only.
      if (llvm::is_contained({SpecialFunctionKind::kMoveInit,
                              SpecialFunctionKind::kCopyInit,
                              SpecialFunctionKind::kDel},
                             SpecialFunctionInfo::getKind(name)))
        continue;
      return false;
    }
  }
  return true;
}

bool ExprEmitter::canImplicitlyConvertToType(ASTExprAnd<CValue> value,
                                             ASTType requiredType) {
  return canImplicitlyConvertToType(declScope, shared, value, requiredType);
}

bool ExprEmitter::canImplicitlyConvertToType(ASTDecl &declScope,
                                             SharedState &shared,
                                             ASTExprAnd<CValue> value,
                                             ASTType requiredType) {
  assert(value.ir && "Should only query valid values");
  // If it already matches, then we're done.
  ASTType rvType = value.ir.getRValueType();
  if (rvType.isEqualCanon(requiredType) ||
      canZeroCostConvert(shared, rvType, requiredType))
    return true;

  // Metatypes can implicitly convert to any trait type they implement.
  if (auto traitType = dyn_cast<TraitType>(requiredType)) {
    if (isa<MetaTypeType, TraitType>(rvType) &&
        checkExplicitConformance(traitType, rvType.getDecl(shared)))
      return true;
    if (isa<TypeType>(rvType) &&
        checkImplicitConformance(shared, value.expr->getLoc(), traitType))
      return true;
    return false;
  }

  // Check to see if we can do an implicit conversion by invoking a `__init__`
  // method on the expected type.
  auto callee = OverloadSet::lookup(declScope, shared, requiredType, "__init__",
                                    value.expr, CallSyntax::kImplicitConvert,
                                    /*no error emission on failure */ {});

  // If there are no viable candidates for the implicit conversion, we fail.
  if (!callee)
    return false;

  // If this is a memory-only type, then we'll pass a self argument with the
  // destination when invoking the method, use a temporary so we can
  // conveniently type check this.
  SmallVector<ASTExprAnd<AnyValue>> args;
  if (!requiredType.isRegisterPassable(value.expr->getLoc(), shared)) {
    auto attr = UnknownAttr::get(PointerType::get(requiredType));
    args.push_back({PValue(attr), value.expr});
  }
  args.push_back(value);

  // If we have at least one candidate, we check to see if any of them can
  // work. We disable implicit conversions though, to prevent converting
  // T -> S -> U in one step.

  // This needs to call filterOverloadSet manually because we cannot allow
  // implicit conversions here.
  PValue calleeFn =
      callee.filterOverloadSet({args}, /*allowImplicitConversions=*/false,
                               /*emitDiagnosticOnFailure=*/false);
  return !calleeFn.isNull();
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
  ASTDecl *wrapperDecl =
      shared.getBuiltinType(declScope, "builtin._stubs", "__MLIRType", loc);
  if (!wrapperDecl)
    return {};
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
      PValue callee = OverloadSet::lookup(declScope, shared, boundWrapper, name,
                                          value.expr, CallSyntax::kMethodCall)
                          .getIfPValue();
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

PValue ExprEmitter::emitMetaTypeConversion(ASTExprAnd<CValue> value,
                                           TraitType trait) {
  // Only static vtables are supported right now.
  PValue typeValue = value.ir.getIfPValue();
  if (!typeValue) {
    emitError(value.expr->getLoc(), "existentials are not supported yet!");
    return {};
  }

  auto type = ASTType(typeValue.getRValueType());

  // Check that the struct implements the trait.
  ASTDecl *typeDecl = type.getDecl(shared);
  if (!checkExplicitConformance(trait, typeDecl)) {
    InflightDiag diag = emitError(value.expr->getLoc(), "cannot bind type ")
                        << type << " to trait " << ASTType(trait)
                        << value.expr->getRange();
    diag.attachNote(typeDecl->getLoc())
        << type << " does not implement " << ASTType(trait);
    return {};
  }

  // Synthesize the vtable required for the trait from the struct. Make sure the
  // trait body is fully resolved so we know what the methods are.
  ASTDecl *traitDecl = ASTType(trait).getDecl(shared);
  if (failed(getDeclResolver().resolveFully(*traitDecl, value.expr->getLoc())))
    return {};

  Type selfType;
  SmallVector<TypedAttr> selfParams;
  auto typeType = TypeType::get(getContext());
  if (auto metatype = dyn_cast<MetaTypeType>(type)) {
    // When converting from a concrete type, construct the self type value as
    // a declref to the metatype.
    selfType = DeclRefType::get(metatype.getSymbol(), metatype.getParamValues(),
                                metatype);
    // Substitute the implicit trait parameters.
    selfParams.assign({TypeConstantAttr::get(typeType, typeType),
                       TypeConstantAttr::get(selfType, typeType)});
  } else {
    // Otherwise, we are converting from a trait. Just rebind the types.
    selfType = ParamRefType::get(typeValue);
    selfParams.assign({TypeConstantAttr::get(type, typeType), typeValue});
  }

  StructDeclOp structDeclOp = dyn_cast<StructDeclOp>(typeDecl);
  bool rpTrivial = false;
  bool regPassable = false;
  if (structDeclOp) {
    rpTrivial = structDeclOp.isRegisterPassable();
    regPassable = structDeclOp.isRegisterPassableTrivial();
  }

  SmallVector<VTableEntryAttr> vtable;
  for (auto &[name, decls] : traitDecl->getDeclsInScope()) {
    if (decls.empty() || !isa<LIT::FuncOp>(decls.front()))
      continue;
    LookupResult result = shared.lookupAndResolveDecl(
        name, value.expr->getLoc(), *typeDecl, /*searchParentScopes=*/false);
    ArrayRef<ASTDecl *> typeFuncs = result.getIfSuccess();
    // Form an overload set of the functions and bind the type parameters.
    for (ASTDecl *expected : decls) {
      auto traitFn = cast<LIT::FuncOp>(expected);

      // Bind away the self type parameter.
      SmallVector<TypedAttr> fnParams = selfParams;
      LITSignatureType sig = traitFn.getFullSignature();
      ParameterEvaluator evaluator(selfParams);
      auto bindings = ParamBindings::getForDeclaredType(declScope, shared,
                                                        ASTType(typeValue));
      for (Type type : sig.getParamTypes().drop_front(2)) {
        fnParams.push_back(UnboundAttr::get(evaluator.getReboundType(type)));
        evaluator.addInputValue(fnParams.back());
        bindings.addPrechecked(fnParams.back());
      }
      sig =
          sig.getSpecializedSignature(fnParams, value.expr->getLocation(*this));

      // Grab the matching function.
      OverloadSet ov(name, typeFuncs, std::move(bindings), value.expr,
                     CallSyntax::kMethodCall);
      ov.baseType = ASTType(typeValue);
      PValue result = ov.filterOverloadSetForValueType(
          sig, /*emitDiagnosticOnFailure=*/false);
      if (!result) {
        // Don't error out if name is for the thunk functions that will be
        // synthesized when conformance check happens.
        if (canSynthesizeIfMissing(name, rpTrivial, regPassable))
          continue;

        // The struct does not conform to the trait. Just silently return, since
        // an error has already been emitted.
        return {};
      }
      if (result.getType().mlirType != sig)
        result = ParamOperatorAttr::get(POC::Rebind, result.get(), sig);
      vtable.push_back(VTableEntryAttr::get(name, result));
    }
  }

  // Create the new type value with the vtable and the trait metatype.
  return TypeConstantAttr::get(selfType, trait,
                               VTableAttr::get(getContext(), vtable));
}

/// When emitting a result value, attempt to "refine" the value type by
/// evaluating 'apply' expressions in its type. Rebind the value if the type can
/// be further specialized.
static AnyValue refineResultValue(AnyValue value, const ExprNode *expr,
                                  ExprEmitter &emitter) {
  Type valueType;
  // Only CValues can be specialized. OverloadSetUValues don't have a type.
  if (auto cValue = value.getIfCValue())
    valueType = cValue.getType();
  else
    return value;

  ParserParamEvaluator evaluator(emitter.getDeclResolver());
  Type refinedType = evaluator.refineType(valueType);
  if (refinedType == valueType)
    return value;

  return emitter.rebindValue({value, expr}, refinedType);
}

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

  // OK, if there is a destination specified, handle them by converging the set
  // of value types we have.

  // If the value being materialized is an unresolved overload set, try to
  // materialize it.
  if (auto overloads = value.getIfOverloadSetUValue())
    return overloads->emitAsCValue(*this, dest);

  auto cValue = value.getIfCValue();
  assert(cValue && "Must be a CValue if not an OverloadSetUValue");
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
      if (canZeroCostConvert(shared, rvalueType, requiredType)) {
        // If we are dealing with signatures that differ only in argument names,
        // we insert a rebind.
        if (cValue.isMValue()) {
          requiredType =
              cast<RefType>(cValue.getType()).getWithElement(requiredType);
        }

        // PValues of lifetime type have a special conversion.
        if (isa<LifetimeType>(requiredType))
          if (auto pv = cValue.getIfPValue())
            value = LifetimeMutCastAttr::get(pv, requiredType);

        value = rebindValue({value, expr}, requiredType);
        return emitCValue({value, expr}, dest);
      }

      // If this is a conversion to the non-materializable target of a type,
      // emit the conversion in the parameter domain.
      if (rvalueType.getNonmaterializableTarget(shared).isEqualCanon(
              requiredType) &&
          cValue.getIfPValue()) {
        CValue converted;
        {
          llvm::SaveAndRestore savedBuilder(builder, {});
          llvm::SaveAndRestore savedContext(paramContext, dest.getContext());
          ValueDest ctorDest = ValueDest(dest.getContext());
          converted =
              emitConstructorCall(requiredType, CallOperands({{cValue, expr}}),
                                  expr, CallSyntax::kImplicitConvert, ctorDest);
        }
        if (!converted) {
          dest.resetForError();
          return {};
        }
        return emitResult(converted, expr, dest);
      }

      // Emit metatype conversions to trait types if the metatype implements the
      // specified trait.
      if (auto trait = dyn_cast<TraitType>(requiredType)) {
        if (isa<MetaTypeType, TraitType>(rvalueType)) {
          PValue result = emitMetaTypeConversion({cValue, expr}, trait);
          if (!result) {
            dest.resetForError();
            return {};
          }
          return emitResult(result, expr, dest);
        }
        if (isa<TypeType>(rvalueType)) {
          PValue result = bindMLIRTypeToTrait({cValue, expr}, trait);
          if (!result) {
            dest.resetForError();
            return {};
          }
          return emitResult(result, expr, dest);
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

CValue ExprEmitter::emitExprCValue(const ExprNode *expr, ExprContext context) {
  assert(expr && "cannot emit a null node");
  return emitCValue({emitExpr(expr, context), expr}, context);
}

SRValue ExprEmitter::emitExprSRValue(const ExprNode *expr, ExprContext context,
                                     ASTType resultType) {
  assert(expr && "cannot emit a null node");
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

CValue ExprEmitter::emitCopyOfValue(ASTExprAnd<CValue> value, ValueDest &dest) {
  ASTType valueType = value.ir.getRValueType();
  SMLoc exprLoc = value.expr->getLoc();
  if (!value.ir)
    return {};

  // Resolve away DLValue's.
  if (auto dlValue = value.ir.getIfDLValue())
    return dlValue->emitLoad(dest, *this);

  switch (valueType.getRegisterPassability(exprLoc, shared)) {
  case TypeConvention::RegisterPassableTrivial:
    if (auto pValue = value.ir.getIfPValue()) {
      value.ir = emitPValueToSRValue({pValue, value.expr}, dest.context);
      if (!value.ir)
        return {};
    }
    break;
  case TypeConvention::RegisterPassable:
    if (auto pValue = value.ir.getIfPValue()) {
      value.ir = emitPValueToSRValue({pValue, value.expr}, dest.context);
      if (!value.ir)
        return {};
      break;
    }

    // Register passable __copyinit__ has signature `(self)->Self`.
    return emitNamedMethodCall("__copyinit__", CallOperands({value}), dest,
                               CallSyntax::kImplicitConvert, value.expr);

  case TypeConvention::MemoryOnly:
    // Memory-only __copyinit__ has signature: `(inout self, existing: Self)`.
    MLValue destBuffer = dest.getMLValueForResult(exprLoc, valueType, *this);
    if (!destBuffer)
      return {};

    if (auto pValue = value.ir.getIfPValue())
      return emitPValueToMLValue({pValue, value.expr}, destBuffer,
                                 dest.context);

    if (!valueType.isCopyable(exprLoc, shared)) {
      if (valueType.isMovableFrom(value, shared)) {
        emitError(exprLoc, "value of type ")
            << valueType
            << " can only be moved, but source value can only be copied"
            << value.expr->getRange();
      } else {
        emitError(exprLoc)
            << valueType << " is not copyable because it has no '__copyinit__'"
            << value.expr->getRange();
      }
      return {};
    }

    SmallVector<ASTExprAnd<AnyValue>> posOperands{
        ASTExprAnd<AnyValue>{destBuffer, value.expr}, value};
    ValueDest copyDest(dest.getContext());
    if (!emitNamedMethodCall("__copyinit__", posOperands, copyDest,
                             CallSyntax::kImplicitConvert, value.expr))
      return {};
    // If we required an implicit conversion, make sure it happens.
    return emitCResult(MRValue(destBuffer), value.expr, dest);
  }

  // Otherwise we can emit a direct use/load for trivial types.
  // It is ok to upgrade SBValue to SRValue for trivial types.
  if (auto sbVal = value.ir.getIfSBValue())
    value.ir = SRValue(sbVal);
  if (auto srVal = value.ir.getIfSRValue())
    return emitCResult(srVal, value.expr, dest);

  if (!builder) {
    emitErrorForDynamicValueInParameter(value.expr);
    return {};
  }
  Value address = value.ir.getMValueReference();
  assert(address && "Unknown value");
  Value result =
      builder->create<RefLoadOp>(value.expr->getLocation(*this), address);
  return emitCResult(SRValue(result), value.expr, dest);
}

BValue ExprEmitter::emitStoreToLValue(ASTExprAnd<CValue> value, LValue destLV,
                                      ExprContext context) {
  // Convert nonmaterializables.
  if (auto nmTarget =
          value.ir.getRValueType().getNonmaterializableTarget(shared)) {
    if (nmTarget.isEqualCanon(destLV.getRValueType())) {
      ValueDest nmConversionDest(context);
      CValue nmConversionVal =
          emitConstructorCall(nmTarget, CallOperands({value}), value.expr,
                              CallSyntax::kIndirectCall, nmConversionDest,
                              /*allowImplicitConversion=*/true);
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
  if (!valueType.isMovableFrom(value, shared)) {
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

  // Otherwise this is a movable RValue that we own.

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
    MLValue destPtr = destLV.getIfMLValue();
    assert(destPtr);
    builder->create<LIT::RefStoreOp>(translateLocation(value.expr->getLoc()),
                                     val, destPtr);

    return SBValue(val);
  }

  if (auto pvalue = value.ir.getIfPValue()) {
    auto valRef = destLV.getIfMLValue();
    assert(valRef && "Unknown LValue");
    return emitPValueToMLValue({pvalue, value.expr}, valRef, context);
  }

  // Otherwise we have an MLValue destination.
  MLValue destRef = destLV.getIfMLValue();
  assert(destRef && "No other known LValue");

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
    return shared.getBuiltinTupleInstantion(declScope, expr->getLoc(), {});

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
  ParamBindings paramBindings(*this);
  for (TypedAttr binding : type.getParamBindings())
    if (!isa<UnboundAttr>(binding))
      paramBindings.addPrechecked(binding);

  // Check the existing bindings against the full signature of the type.
  ParameterExprArrayAttr bindingValuesAttr = paramBindings.verifyBindings(
      structDecl, structDecl.getSignature(), expr->getLoc(),
      /*allowPartiallyBound=*/false);
  if (!bindingValuesAttr)
    return {};

  // If verifyBindings changed the bindings set, then we may have had an
  // empty varargs list or something.  Rebind the DeclRefType.
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

//===----------------------------------------------------------------------===//
// Return emission helpers.

LogicalResult ExprEmitter::emitRaise(SRValue errorValue, Location raiseLoc) {
  // Cannot raise in a parameter expression.
  if (!builder)
    return failure();
  // If the raise is not in a try and the parent doesn't throw, it is not valid
  // syntax.
  if (!findOpProcessingRaise(builder->getInsertionBlock()))
    return failure();

  builder->create<LIT::RaiseOp>(raiseLoc, errorValue);
  return success();
}

void ExprEmitter::emitNormalReturn(ImplicitLocOpBuilder &builder, Value value,
                                   const ASTDecl &funcDecl) {
  auto func = cast<LIT::FuncOp>(funcDecl);
  emitNormalReturn(builder, value, func);
}

void ExprEmitter::emitNormalReturn(ImplicitLocOpBuilder &builder, Value value,
                                   LIT::FuncOp func) {
  switch (func.getSpecialFunctionKind()) {
  default:
    break;

  /// In the __del__ method for a struct, we need to mark 'self' as being
  /// destroyed before any return operation.
  case SpecialFunctionKind::kDel: {
    assert(func.getBody()->getNumArguments() == 1 &&
           "__del__ should have one argument");
    Value selfArg = func.getBody()->getArgument(0);

    // If this is a @register_passable type, the value must be stored
    // in a box and we want to treat the box as the thing that we track.
    // CheckLifetimes doesn't track register values field sensitively, so there
    // is no way to say that the full object bit is dead in a SRValue.
    if (func.getSignature().getArgConvention(0) == ArgConvention::OwnedInReg) {
      // Find the single thing that got stored to, ignoring debug.value ops.
      Value storedMem;
      for (auto user : selfArg.getUsers()) {
        if (isa<DebugInfo::ValueOp>(user))
          continue;
        assert(!storedMem && "Should only have a single store");
        storedMem = cast<LIT::RefStoreOp>(user).getRef();
      }
      // If we found it, then ownership has already transfered to the memory
      // object, so track it instead of the argument.
      assert(storedMem && "local value box for OwnedInReg self not found");
      selfArg = storedMem;
    }
    builder.create<LIT::OwnershipMarkDestroyedOp>(selfArg);
    break;
  }

  /// In the __moveinit__ method for a struct, we need to mark 'existing' as
  /// being destroyed before any return operation if it is owned convention.
  case SpecialFunctionKind::kMoveInit: {
    assert(func.getBody()->getNumArguments() == 2 &&
           "__moveinit__ should have two arguments");
    Value existingArg = func.getBody()->getArgument(1);
    builder.create<LIT::OwnershipMarkDestroyedOp>(existingArg);
    break;
  }
  }

  // Finally we emit a normal return with lit.return.
  builder.create<LIT::ReturnOp>(value);
}

//===----------------------------------------------------------------------===//
// Declaration reference emission helpers.

PValue ExprEmitter::resolveAliasDeclareValue(
    AliasDeclOp param, std::optional<ArrayRef<TypedAttr>> paramValues,
    SMLoc errLoc) {
  // If the param is declared in a function, then just directly use it.
  Operation *parent = param->getParentOp();
  while (true) {
    // If this reference is within a function then keep it symbolic.
    if (parent && isa<LIT::FuncOp>(parent))
      return ParamDeclRefAttr::get(param.getName(), param.getType());

    // If this is at file scope, inline it.
    if (!parent || isa<FileModuleOp>(parent))
      return param.getValue();

    // If this is in a struct, then the value may refer to parameters declared
    // on the struct, whose values come through 'bindings'.  Remap.
    if (auto structDecl = dyn_cast<StructDeclOp>(parent)) {
      // If the reference is to a member of the struct that has bindings, remap
      // them.  This allows things like `SomeType[a,b].someAlias` to substitute
      // the a/b values into the body of `someAlias`.  If we have no bindings,
      // then we know we're in a context where the body of the alias is still
      // valid.
      if (!paramValues)
        return param.getValue();

      // Disallow accessing alias members of an unbound type.
      // TODO: This should return a parametric alias instead.
      ArrayRef<ParamDeclAttr> paramDecls = structDecl.getParams();
      size_t numParams = llvm::count_if(*paramValues, [](TypedAttr value) {
        return !isa<UnboundAttr>(value);
      });
      if (paramDecls.size() != numParams) {
        shared.emitError(errLoc,
                         "incorrect number of type parameters: expected ")
            << structDecl.getParams().size() << " but got " << numParams;
        return PValue();
      }

      ParserParamEvaluator evaluator(*shared.declResolver, paramDecls,
                                     *paramValues);
      return PValue(evaluator.getReboundAttribute(param.getValue()));
    }

    // Ignore if and other control flow things.
    parent = parent->getParentOp();
  }

  return ParamDeclRefAttr::get(param.getName(), param.getType());
}

AnyValue ExprEmitter::emitDeclReference(StringRef spelling,
                                        ArrayRef<ASTDecl *> decls,
                                        const ExprNode *expr, ValueDest &dest,
                                        std::optional<Capture> &capture) {
  shared.notifyListenerOnRef(decls, spelling, expr);

  // Functions form an address, and may be overloaded.
  if (auto firstCandidate = dyn_cast<LIT::FuncOp>(decls[0])) {
    // Form an overload set value with all the candidates.
    auto result = OverloadSetUValue::create(
        spelling, decls, ParamBindings(*this), expr, CallSyntax::kDirectCall);
    return emitResult(result, expr, dest);
  }

  assert(decls.size() == 1 && "Only functions may be overloaded");
  ASTDecl &decl = *decls[0];

  // Aliases form a PValue.
  if (auto param = dyn_cast<AliasDeclOp>(decl)) {
    PValue result =
        resolveAliasDeclareValue(param, /*bindings=*/{}, expr->getLoc());
    return emitResult(result.get(), expr, dest);
  }

  // If this is a type declaration, return it as a type.
  if (auto structOp = dyn_cast<StructDeclOp>(decl))
    return emitResult(structOp.bindReference(), expr, dest);
  if (auto traitOp = dyn_cast<TraitDeclOp>(decl))
    return emitResult(traitOp.bindReference(), expr, dest);

  // If this is a module or package declaration, form a module reference.
  if (isa<FileModuleOp, PackageOp>(decl)) {
    PValue result(ModuleAttr::get(MetaTypeType::get(
        decl.getSymbolRef(), TypeSignatureType::get(getContext()))));
    return emitResult(result, expr, dest);
  }

  if (auto pvalue = decl.getIfPValue())
    return emitResult(pvalue, expr, dest);

  // Narrow the decl to a CValue.
  CValue value;
  if (auto letDecl = dyn_cast<LetRegDeclOp>(decl)) {
    // 'let' declarations of a register passable value resolve to an SBvalue.
    value = SBValue(letDecl.getResult());
  } else if (auto var = dyn_cast<VarLetDeclOp>(decl)) {
    // Treat both 'var' and 'let' decls as mutable values and defer to check
    // lifetimes to verify 'let' decls. This allows lazy 'let' initialization.
    value = MLValue(var);
  } else if (auto globalOp = dyn_cast<GlobalVarDeclOp>(decl)) {
    // If this is a parameter context then we cannot return a dynamic field.
    if (!builder)
      return emitErrorForDynamicValueInParameter(expr);
    // Return a mutable value only if the global variable is mutable.
    auto ref = builder->create<GlobalVarRefOp>(
        translateLocation(expr->getLoc()), globalOp);
    if (globalOp.getIsVar())
      value = MLValue(ref);
    else
      value = MBValue(ref);
  } else if (auto rvalue = decl.getIfRValue()) {
    value = rvalue;
  } else if (auto bvalue = decl.getIfBValue()) {
    value = bvalue;
  } else if (auto lvalue = decl.getIfMLValue()) {
    value = lvalue;
  } else {
    emitError(expr->getLoc(), "use of declaration '")
        << spelling << "' as a value isn't supported yet" << expr->getRange();
    return {};
  }

  capture = Capture(value);
  return emitResult(value, expr, dest);
}

AnyValue ExprEmitter::emitDeclReference(StringRef spelling,
                                        ArrayRef<ASTDecl *> decls,
                                        ExprContext context) {
  SyntheticNode dummyNode({});
  std::optional<Capture> capture;
  ValueDest dest(context);
  return emitDeclReference(spelling, decls, &dummyNode, dest, capture);
}

//===----------------------------------------------------------------------===//
// DLValue implementations
//===----------------------------------------------------------------------===//

CValue DiscardDLValue::emitLoad(ValueDest &dest, ExprEmitter &emitter) const {
  // The `_` syntax stands for an unbound parameter.
  return UnboundAttr::get(elementType);
}

void DiscardDLValue::emitStore(ASTExprAnd<CValue> value,
                               ExprEmitter &emitter) const {
  // Convert to an RValue to fully evaluate it, but otherwise just discard the
  // value!
  (void)emitter.emitRValue(value, EC_Assignment, elementType);
}

CValue StoredAttributeRefDLValue::emitLoad(ValueDest &dest,
                                           ExprEmitter &emitter) const {
  // To load x.y, we load x, then then load y out of it.
  ValueDest baseDest(dest.getContext());
  auto base = baseVal.ir->emitLoad(baseDest, emitter);
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
  ASTType rvalueType = baseVal.ir->elementType;
  Value tmpDecl = emitter.emitVarLetDecl("__store_tmp__", rvalueType, loc,
                                         VarLetDeclKind::Synthesized);

  // Load the entire base LValue into tmpDecl.
  ValueDest tmpValueDest(MLValue(tmpDecl), EC_AttributeRefBase);
  auto base = baseVal.ir->emitLoad(tmpValueDest, emitter);
  if (!base) {
    tmpValueDest.resetForError();
    return;
  }

  // Store into the field.
  auto fieldPtr =
      emitter.builder->create<RefStructGEROp>(loc, tmpDecl, getField());
  emitter.emitStoreToLValue(value, MLValue(fieldPtr), EC_AttributeRefBase);

  // Store the whole result back, transfering ownership as an MRValue.
  baseVal.ir->emitStore({MRValue(tmpDecl), expr}, emitter);
}

CValue SubscriptDLValue::emitLoad(ValueDest &dest, ExprEmitter &emitter) const {
  auto methodName =
      isSubscript() ? StringRef("__getitem__") : StringRef("__getattr__");
  auto result = emitter.emitNamedMethodCall(
      methodName, CallOperands(posOperands, &kwOperands), dest,
      CallSyntax::kSubscript, expr);
  // TODO: The result could be another LValue in the future.
  assert(!result || result.getIfRValue() || result.getIfBValue());
  return result;
}

void SubscriptDLValue::emitStore(ASTExprAnd<CValue> value,
                                 ExprEmitter &emitter) const {
  // TODO(#22580): support keyword operands in __setitem__
  if (isSubscript() && !kwOperands.empty()) {
    emitter.emitError(expr->getLoc())
        << "keyword operands for __setitem__ not supported yet"
        << expr->getRange();
    return;
  }

  auto methodName =
      isSubscript() ? StringRef("__setitem__") : StringRef("__setattr__");
  SmallVector<ASTExprAnd<AnyValue>> posOperandsWithValue(posOperands);
  posOperandsWithValue.push_back(value);
  ValueDest storeDest(EC_Assignment);
  emitter.emitNamedMethodCall(methodName, posOperandsWithValue, storeDest,
                              CallSyntax::kSubscript, expr);
}

/// Loading a tuple RValue loads all the elements and returns a tuple instance.
CValue TupleDLValue::emitLoad(ValueDest &dest, ExprEmitter &emitter) const {
  // Emit a call to the tuple type constructor as an implicit conversion.
  return emitter.emitConstructorCall(elementType, eltLValues, expr,
                                     CallSyntax::kImplicitConvert, dest);
}

/// Storing to a tuple LValue extracts the elements out of the provided value
/// stores them into each component LValue.
void TupleDLValue::emitStore(ASTExprAnd<CValue> value,
                             ExprEmitter &emitter) const {
  auto emitError = [&]() -> InflightDiag {
    return (emitter.emitError(expr->getLoc())
            << value.expr->getRange() << expr->getRange());
  };

  // If the value is a type with a staticly known length, check that it agrees
  // with the # of lvalues being assigned into.  Maybe we could generalize this
  // to invoke a new static get_static_len method or something?
  // TODO(generalize): Need @parameter fn's for methods
  // https://github.com/modularml/modular/issues/14945
  ASTDecl &tupleLiteralDecl = *elementType.getDecl(emitter.shared);
  ASTType srcRValueType = value.ir.getRValueType();

  // TODO: We need to support storing anything into a tuple that can be
  // extracted from, even things with dynamic length.  For example, Python
  // allows "(a, b) = [1, 2]", we need to support PythonObject.  The correct
  // sequence is to check the len(x) of the argument and see if it is exactly
  // right, CPython produces these errors at runtime:
  //   ValueRrror: too many values to unpack (expected 2)
  //   ValueError: not enough values to unpack (expected 2, got 1)
  //
  // We currently require the input be a Tuple.
  if (srcRValueType.getDecl(emitter.shared) != &tupleLiteralDecl) {
    emitError() << "cannot unpack value of type " << srcRValueType
                << " into a tuple";
    return;
  }

  assert(srcRValueType.getParamBindings().size() == 1 &&
         "Tuple has one pack parameter");
  TypedAttr packAttr = srcRValueType.getParamBindings()[0];
  auto packVariadic = dyn_cast<VariadicAttr>(packAttr);
  if (!packVariadic) {
    emitError() << "cannot unpack value of parametric tuple type "
                << srcRValueType << " into a fixed arity";
    return;
  }
  if (packVariadic.getValues().size() != eltLValues.size()) {
    emitError() << "cannot unpack tuple value with "
                << packVariadic.getValues().size()
                << " elements into tuple binding with " << eltLValues.size()
                << " elements";
    return;
  }

  // Tuple has a get method with a signature of:
  //    get[i: Int, T: AnyRegType](self)
  // FIXME(Issue #14946): The Tuple.get's T parameter shouldn't exist!
  //   https://github.com/modularml/modular/issues/14946
  // For the dynamic case we'd use __getitem__.
  auto getDecl =
      OverloadSet::lookup(emitter.declScope, emitter.shared, elementType, "get",
                          expr, CallSyntax::kTupleGetItem,
                          /*errorHandler*/ {});

  if (getDecl.isNull()) {
    emitError() << "expected Tuple to have one get method";
    return;
  }

  // Bind the Tuple type parameters.
  getDecl.paramBindings.addPrechecked(packVariadic);

  // Ok, we have a tuple with the right number of elements, extract each element
  // and store into the corresponding lvalue.
  for (auto [index, lvalue] : llvm::enumerate(eltLValues)) {
    // Bind the i/T parameters.  Int implicitly constructs from index type.
    TypedAttr iParam =
        IntegerAttr::get(IndexType::get(emitter.getContext()), index);
    getDecl.paramBindings.posBindings.resize(1);
    getDecl.paramBindings.add(expr, iParam);
    getDecl.paramBindings.add(expr, packVariadic.getValues()[index]);

    // Emit the call to get the item from the tuple into the corresponding
    // LValue.
    LValue lv = lvalue.ir.getIfLValue();
    assert(lv && "Each dest is known to be an lvalue");
    ValueDest eltDest(lv, EC_TupleElement);

    if (!getDecl.emitCall(CallOperands({{value.ir, value.expr}}), eltDest,
                          emitter)) {
      eltDest.resetForError();
      return;
    }
  }
}

//===--------------------------------------------------------------------===//
// Var/let emission helpers.

VarLetDeclOp ExprEmitter::emitVarLetDecl(const Twine &name, Type type,
                                         Location loc, VarLetDeclKind kind) {
  StringAttr lifetimeAttr = declScope.getAnonymousLifetimeFor(name);
  return builder->create<VarLetDeclOp>(loc, type, name.str(), lifetimeAttr,
                                       kind);
}

VarLetDeclOp ExprEmitter::emitVarLetDecl(StringAttr name, Type type,
                                         Location loc, VarLetDeclKind kind) {
  StringAttr lifetimeAttr = declScope.getAnonymousLifetimeFor(name.strref());
  return builder->create<VarLetDeclOp>(loc, type, name.str(), lifetimeAttr,
                                       kind);
}
