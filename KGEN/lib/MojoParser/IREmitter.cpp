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

#include "IREmitter.h"
#include "CallEmission.h"
#include "ExprNodes.h"
#include "MojoUtils.h"
#include "ParserEvaluationContext.h"
#include "Traits.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/ScopeExit.h"
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
  case EC_CallArgDefaultValue:
    return " in default call argument";
  case EC_CallRefArgValue:
    return " in 'ref' argument";
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
  case EC_TypePattern:
    return " in type pattern";
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
  case EC_ComptimeAssert:
    return " in '__comptime_assert' expression";
  case EC_RaiseValue:
    return " in raised value";
  case EC_ReturnValue:
    return " in return value";
  case EC_Requires:
    return " in 'requires' clause";
  case EC_MLIRMagic:
    return " in MLIR magic";
  case EC_TopLevelStmt:
    return " in expression statement";
  case EC_CollectionLiteral: // [x, y], {x:y, q:r}
    return " in collection literal";
  case EC_CollectionCompElt: // [x for x in y]
    return " in comprehension expression";
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
  case EC_Trait:
    return " in trait conformance checking";
  case EC_Closure:
    return " in internal closure formation";
  case EC_Origin:
    return " in origin specifier";
  case EC_TypeOf:
    return " in type_of";
  case EC_ConformsTo:
    return " in conforms_to";
  case EC_FunctionsInModule:
    return " in __functions_in_module";
  case EC_PyBindGen:
    return " in Python binding generation";
  case EC_MergeWith:
    return " in implicit '__merge_with__' call";
  case EC_RefBinding:
    return " in 'ref' binding";
  case EC_SynthesizedMethod:
    return " in synthesized method";
  }
  llvm_unreachable("invalid expr context");
}

//===----------------------------------------------------------------------===//
// ValueDest
//===----------------------------------------------------------------------===//

ValueDest::ValueDest(VarDeclOp dest, ExprContext context)
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
    os << "ExprNode: ";
    expr->print(os);
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

  os << " patternDeclKind=";
  switch (value.getPatternDeclKind()) {
  case PatternDeclKind::kNone:
    os << "kNone";
    break;
  case PatternDeclKind::kVar:
    os << "kVar";
    break;
  case PatternDeclKind::kRef:
    os << "kRef";
    break;
  case PatternDeclKind::kBind:
    os << "kBind";
    break;
  }

  os << '\n';
  return os;
}

/// If this indicates an explicit expected RValue type, return that type.
ASTType ValueDest::getExpectedTypeIfSpecified() const {
  // These have no implied type.
  if (isa<NullRepresentation, LValueBufferTaken, Operation *, const ExprNode *>(
          representation))
    return {};

  // If we just have a contextual type, return it.
  if (ASTType type = dyn_cast<ASTType>(representation))
    return type;
  if (isa<LValueInitializerType>(representation))
    return cast<LValueInitializerType>(representation).type;
  return cast<LValue>(representation).getRValueType();
}

/// When an error is emitted instead of generating IR, this method resets the
/// ValueDest so it doesn't complain when emission is done.
void ValueDest::resetForError(IREmitter &emitter) {
  // We generally just abandon this ValueDest, but if this was set up to
  // initialize something that could infer types, we need to infer them to
  // TypeCheckErrorType to avoid downstream errors using whatever we failed to
  // initialize.

  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    if (auto varOp = dyn_cast<VarDeclOp>(opDest)) {
      assert(isa<UnresolvedType>(varOp.getType().getElementType()) &&
             "Cannot resolve an already-resolved vardecl");
      varOp.getResult().setType(varOp.getType().getWithElement(
          emitter.shared.getTypeCheckErrorType()));
    }
  } else if (auto target = dyn_cast<const ExprNode *>(representation)) {
    // If emitting the RHS failed, use a "type check error" expression as the
    // RHS so we can make sure to emit any vars declared, to silence
    // downstream errors.
    //     var x = <bad>
    //     use(x)  # Don't warn here.
    ValueDest dest(
        LValueInitializerType{emitter.shared.getTypeCheckErrorType()},
        getContext());
    (void)emitter.emitExprLValue(target, dest);
  }

  representation = NullRepresentation();
}

/// Inspect the ValueDest to see if it implies a specific type for the value
/// being computed, emitting ExprNode targets if present to get their implied
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
                                      IREmitter &emitter) {
  // These have no implied type.
  if (isa<NullRepresentation, LValueBufferTaken, Operation *>(representation))
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
    if (!existingValueType)
      return {};
    if (ASTType nmTarget = ASTType(existingValueType)
                               .getNonmaterializableTarget(emitter.shared))
      existingValueType = nmTarget;

    // Propagate var/ref context (if any) into the generated declarations.
    ValueDest dest(LValueInitializerType{existingValueType}, context);
    dest.patternDeclKind = patternDeclKind;

    // Emit the target as an LValue to understand what we're assigning into.
    LValue exprLValue = emitter.emitExprLValue(expr, dest);
    if (!exprLValue) { // Error already emitted.
      representation = NullRepresentation();
      return {};
    }
    representation = exprLValue;
  }

  // We must have an LValue at this point.
  auto lvalue = cast<LValue>(representation);

  // If this is a "bind" operation (e.g. in a for stmt) infer the type of the
  // var decl from the assignment and yield the MLValue.
  if (RLValue rlValue = lvalue.getIfRLValue()) {
    // Unbound 'bind' values will have two layers of !lit.ref on the RValue
    // type.
    VarDeclOp refOp = cast<VarDeclOp>(rlValue.getDefiningOp());
    if (refOp.getKind() == VarDeclKind::Bind)
      return cast<RefType>(refOp.getType().getElementType()).getElementType();
  }

  // If we have an lvalue already specified, return it.
  return lvalue.getRValueType();
}

/// If this ValueDest specifies an MLValue that will be returned by
/// getMLValueForResult with the specified type, return it. Otherwise return
/// null.
///
/// NOTE: This needs to be kept in sync with getLValueForResult.
MLValue ValueDest::getDefinedMLValueIfExists(ASTType resultType,
                                             IREmitter &emitter) {
  // Handle inference of a 'var' declaration's type.
  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    // If the result type has a non-materializable type, then we infer the var
    // to its materialized type.
    ASTType nmTarget = resultType.getNonmaterializableTarget(emitter.shared);
    ASTType materializedType = nmTarget ? nmTarget : resultType;

    auto varOp = cast<VarDeclOp>(opDest);
    assert(isa<UnresolvedType>(varOp.getType().getElementType()) &&
           "Cannot resolve an already-resolved vardecl");
    varOp.getResult().setType(varOp.getType().getWithElement(materializedType));

    // Now that we inferred the 'var' type, we can treat this like a normal
    // MLValue.
    representation = LValue(MLValue(varOp.getResult()));
  }

  // If we have an uncollapsed expression, emit it to learn more about it.
  if (const ExprNode *target = dyn_cast<const ExprNode *>(representation)) {
    ValueDest dest(LValueInitializerType{resultType}, getContext());
    dest.patternDeclKind = patternDeclKind;
    if (LValue lValue = emitter.emitExprLValue(target, dest)) {
      representation = lValue;
    } else {
      dest.resetForError(emitter);
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

    // If this is a "bind" operation (e.g. in a for stmt) infer the type of the
    // var decl from the assignment and yield the MLValue.
    if (RLValue rlValue = lValue.getIfRLValue()) {
      VarDeclOp refOp = cast<VarDeclOp>(rlValue.getDefiningOp());
      if (refOp.getKind() == VarDeclKind::Bind) {
        refOp.setKind(VarDeclKind::Bound);
        refOp.getResult().setType(refOp.getType().getWithElement(resultType));
        representation = LValue(MLValue(refOp));
        return MLValue(refOp);
      }
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
                                     bool requireMLValue, IREmitter &emitter) {
  // Handle inference of a 'var' declaration's type.
  if (auto *opDest = dyn_cast<Operation *>(representation)) {
    // If the result type has a non-materializable type, then we infer the var
    // to its materialized type.
    ASTType nmTarget = resultType.getNonmaterializableTarget(emitter.shared);
    ASTType materializedType = nmTarget ? nmTarget : resultType;

    auto varOp = cast<VarDeclOp>(opDest);
    assert(isa<UnresolvedType>(varOp.getType().getElementType()) &&
           "Cannot resolve an already-resolved vardecl");
    varOp.getResult().setType(varOp.getType().getWithElement(materializedType));

    // Now that we inferred the 'var' type, we can treat this like a normal
    // MLValue.
    representation = LValue(MLValue(varOp.getResult()));
  }

  // We have several cases where we can produce an LValue but it may have the
  // wrong type.  The client may be cool with this (when allowIncompatibleTypes
  // is true), but if not we generate a new temporary buffer.

  // If we have an expression node destination, then we need to bind this
  // value to a pattern (aka "target" in Python internals nomenclature).
  if (isa<const ExprNode *>(representation)) {
    // resolveImpliedType will resolve ExprNode destinations into LValues.
    (void)resolveImpliedType(loc, resultType, emitter);

    // If this is a "bind" operation (e.g. in a for stmt) then the callee will
    // fill in the produced MLValue, but subsequent accesses will need to treat
    // the value as bound (and therefore immutable).
    if (LValue lValue = dyn_cast<LValue>(representation)) {
      if (RLValue rlValue = lValue.getIfRLValue()) {
        VarDeclOp refOp = cast<VarDeclOp>(rlValue.getDefiningOp());
        if (refOp.getKind() == VarDeclKind::Bind) {
          // Switch the vardecl so that uses of it are treated as MBValue
          // instead of MLValues.
          refOp.setKind(VarDeclKind::Bound);
          refOp.getResult().setType(refOp.getType().getWithElement(resultType));
          representation = MLValue(refOp);
        }
      }
    }
  }

  // If we have an lvalue already specified, return it.
  if (LValue lValue = dyn_cast<LValue>(representation)) {
    // If asking for a buffer of the type we happen to have, or if the client
    // doesn't care if it matches, then we can directly return it.
    if (allowIncompatibleTypes ||
        emitter.canZeroCostConvert(lValue.getRValueType(), resultType,
                                   emitter.shared)) {
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
        << "type " << resultType << getContextMessage(emitter.paramContext);
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
                                       IREmitter &emitter) {
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
// IREmitter
//===----------------------------------------------------------------------===//

/// Create an IREmitter for a dynamic context with a builder.
IREmitter::IREmitter(ASTDecl &declScope, OpBuilder builder,
                     std::optional<OpBuilder> varDeclCursor)
    : SharedStateUser(declScope.getShared()), builder(builder),
      paramContext(EC_InvalidContext), declScope(declScope),
      varDeclCursor(varDeclCursor) {}

/// Create an IREmitter for a parameter context.
IREmitter::IREmitter(ASTDecl &declScope, ExprContext paramContext)
    : SharedStateUser(declScope.getShared()), builder({}),
      paramContext(paramContext), declScope(declScope) {}

/// Emit an error about use of a dynamic value (the expression) in a context
/// that only allows parameter expressions.  This always returns a null
/// PValue.
PValue IREmitter::emitErrorForDynamicValueInParameter(const ExprNode *expr,
                                                      const char *message) {
  assert(paramContext != EC_InvalidContext &&
         "parameter context not set correctly");
  if (!message)
    message = "cannot use a dynamic value";
  emitError(expr->getLoc(), message)
      << getContextMessage(paramContext) << expr->getRange();
  return {};
}

PValue
IREmitter::emitErrorForDynamicValueInParameter(llvm::SMLoc loc,
                                               const char *customMessage) {
  return emitErrorForDynamicValueInParameter(shared.translateLocation(loc),
                                             customMessage);
}

/// Emit an error about use of a dynamic value (the expression) in a context
/// that only allows parameter expressions.  This always returns a null
/// PValue.
PValue IREmitter::emitErrorForDynamicValueInParameter(Location loc,
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

CValue IREmitter::emitRValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
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

  // If the value dest expects a different result type than the lvalue or bvalue
  // that we have, then we'll need to do a conversion, and that conversion will
  // return an rvalue. Use it first which may avoid a copy of a value.
  if (auto knownDestType = dest.getExpectedTypeIfSpecified()) {
    if (!cValue.getRValueType().isEqualCanon(knownDestType)) {
      return emitConstructorCall(knownDestType, CallOperands(value.expr, value),
                                 CallSyntax::kImplicitConvert, dest);
    }
  }

  // Otherwise, this is an LValue or BValue, emit a copy.
  return emitCopyOfValue({cValue, value.expr}, dest);
}

RValue IREmitter::emitRValue(ASTExprAnd<AnyValue> value, ExprContext context,
                             ASTType resultType) {
  ValueDest dest(resultType, context);
  CValue result = emitRValue(value, dest);
  while (true) {
    if (!result) {
      dest.resetForError(*this);
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

CValue IREmitter::emitCValue(ASTExprAnd<AnyValue> value, ExprContext context,
                             ASTType resultType) {
  ValueDest dest(resultType, context);
  if (auto c = emitCValue(value, dest))
    return c;
  dest.resetForError(*this);
  return {};
}

CValue IREmitter::emitCValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
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

  if (auto initValue = value.ir.getIfInitializer())
    return initValue->emitAsCValue(*this, dest);

  llvm_unreachable("unknown UValue in emitCValue");
}

/// Emit an expression providing an immutable borrowed reference to a value.
BValue IREmitter::emitBValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value)
    return {};

  // Handle dynamic LValues by loading from them.
  if (auto dlv = value.ir.getIfDLValue()) {
    value.ir = dlv->emitLoad(dest, *this);
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
  if (value.ir.isMValue()) {
    // Maintain parametric mutability if we have it.
    if (auto mbp = value.ir.getIfMBPValue())
      return mbp;
    // Otherwise decay MLValue/MRValue to MBValue.
    return MBValue(value.ir.getMValueReference());
  }

  // Decay SRValue's into SBValue or MBValue.
  if (auto srVal = value.ir.getIfSRValue()) { // Decay => SBValue/MRValue
    if (ASTType(srVal.getType()).isTrivial(value.expr->getLoc(), shared))
      return SBValue(srVal);
    // If this is a nontrivial value, we need to create an MRValue (and decay
    // that) so we can track its lifetime correctly.
    auto mrVal = emitMRValue(value, dest.getContext());
    if (!mrVal)
      return {};
    return MBValue(mrVal);
  }

  // Finally, we know we have a BValue.
  auto resultBV = value.ir.getIfBValue();
  assert(resultBV && "unknown value kind");
  return resultBV;
}

BValue IREmitter::emitBValue(ASTExprAnd<AnyValue> value, ExprContext context,
                             ASTType resultType) {
  ValueDest dest(resultType, context);
  if (auto result = emitBValue(value, dest))
    return result;
  dest.resetForError(*this);
  return {};
}

LValue IREmitter::emitLValue(ASTExprAnd<AnyValue> value, ValueDest &dest) {
  if (!value) {
    dest.resetForError(*this);
    return {};
  }

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
  dest.resetForError(*this);
  return {};
}

/// This verifies that the specified PValue can be materialized to a runtime
/// value, emits an error if it cannot.
static LogicalResult emitErrorIfUnmaterializableValue(IREmitter &emitter,
                                                      ASTExprAnd<PValue> value,
                                                      ExprContext context) {
  TypedAttr attr = value.ir.get();
  // We cannot emit types as values yet.
  if (isTypeExpr(attr) && !isa<ModuleAttr>(attr)) {
    const ExprNode *expr = value.expr;
    MojoInflightDiag diag = emitter.emitError(
        expr->getLoc(), "dynamic type values not permitted yet");
    if (context == EC_VarInit)
      diag << "; try creating an `alias` instead of a `var`";
    else if (context == EC_CallArgValue)
      diag << "; try passing types as a parameters instead of arguments";
    diag << expr->getRange();
    return failure();
  }

  // We cannot emit a value that contains an origin in its type (e.g. a
  // StringSlice or UnsafePointer) because the origin will be incorrect -
  // referring to immortal compile-time memory.
  if (ASTType(attr.getType()).containsUnmaterializableOrigins(emitter.shared)) {
    const ExprNode *expr = value.expr;
    auto diag = emitter.emitError(
        expr->getLoc(), "cannot materialize compile-time value of type ");
    diag << ASTType(attr.getType()) << " to a runtime value"
         << expr->getRange();
    diag.attachNote(expr->getLoc())
        << "the type contains an origin referring to a compile-time value";
    return failure();
  }

  return success();
}

SRValue IREmitter::emitPValueToSRValue(ASTExprAnd<PValue> value,
                                       ExprContext context) {
  TypedAttr attr = value.ir.get();
  const ExprNode *expr = value.expr;

  // If this is a parameter, we need to materialize it, either as an
  // index.constant or as a parameter expression.
  if (!builder) {
    emitErrorForDynamicValueInParameter(expr);
    return {};
  }

  // Diagnose issues about types that cannot be comptime -> runtime
  // materialized.
  if (failed(emitErrorIfUnmaterializableValue(*this, value, context)))
    return {};

  Location location = expr->getLocation(*this);

  // If the value being materialized is itself parameterized, then we cannot
  // materialize it as an SSA value - there will be no way to bind parameters to
  // it.
  // TODO: We should have a general predicate from this provided by the KGEN
  // parameter utilities.
  if (auto signature = dyn_cast<FnTypeGeneratorType>(attr.getType())) {
    // If the value has any unbound parameters, they might be default arguments
    // or an variadic list that should be bound to an empty list.
    if (!signature.getInputParamTypes().empty()) {
      ParamBindings paramBindings(getDeclScope(), expr);
      // Try to fully bind the signature, in case it can be made concrete with
      // default values, etc.
      ParameterExprArrayAttr bindingAttr = paramBindings.tryVerifyBindings(
          signature.getInputParamTypes(), signature.getMetadata(),
          /*partial=*/true);

      // Notice if there are any unbound parameters.
      bool anyUnbound = true;
      if (bindingAttr)
        anyUnbound = llvm::any_of(
            bindingAttr, [](TypedAttr attr) { return isa<UnboundAttr>(attr); });

      if (anyUnbound) {
        // If it didn't work out, then it is an error because parameterized
        // values cannot be used in a dynamic context.
        emitError(expr->getLoc(), "cannot use parameterized function of type ")
            << ASTType(attr.getType()) << " without binding all its parameters"
            << expr->getRange();
        return {};
      }

      // Apply whatever it produced to the attr of signature type to resolve the
      // remaining arguments.
      attr = BindParamsAttr::get(attr, {bindingAttr},
                                 &shared.getEvaluationContext());
    }

    // Materialize signatures as closures.
    if (signature.isCapturing()) {
      emitError(
          expr->getLoc(),
          "TODO: capturing closures cannot be materialized as runtime values");
      return {};
    }
    return SRValue(CreateClosureOp::create(*builder, location, signature, attr,
                                           ValueRange()));
  }

  ASTType valueType = value.ir.getRValueType();

  // If the type is trivial, materialize using param.constant.
  if (valueType.isTrivial(value.expr->getLoc(), shared))
    return SRValue(ParamConstantOp::create(*builder, location, value.ir));

  // If the type is implicitly copyable, it should be cheap to be implicitly
  // materialized as well.
  //
  // NOTE: we need to leave a backdoor to allow implicit materialization for
  // default argument. This is because we parse default argument value
  // into PValue at the moment, meaning that to emit the value for default
  // argument, we will have to materialize it first. Using `EC_context` to tell
  // whether we are generating default arg value is not a typical usage of
  // `EC_context`, but it is much cleaner/simpler than passing a flag all the
  // way down from `emitPreemittedArgumentAsDynamicValue`.
  bool isDefaultArg = (context == EC_CallArgDefaultValue);
  if (isDefaultArg ||
      valueType.isImplicitlyCopyable(value.expr->getLoc(), shared))
    return SRValue(ParamMaterializeOp::create(*builder, location, value.ir));

  if (isa<ModuleType>(valueType)) {
    emitError(expr->getLoc(), "cannot use package name ")
        << valueType << " as a runtime value" << expr->getRange();
    return {};
  }

  auto diag =
      emitError(expr->getLoc(), "cannot materialize comptime value of type ")
      << value.ir.getType()
      << " to runtime because it is not 'ImplicitlyCopyable'"
      << expr->getRange();

  // Attach the fix it by wrapping materialize[]() around the expression.
  diag.attachNote(expr->getLoc())
      << "use 'materialize' to explicitly materialize the value"
      << FixIt::insertBeforeToken(expr->getRangeStart(), "materialize[")
      << FixIt::insertAfterToken(expr->getRangeEnd(), "]()", shared.diags);
  return {};
}

SRValue IREmitter::emitSRValue(ASTExprAnd<AnyValue> anyValue,
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
        LoadConsumeOp::create(*builder, expr->getLocation(*this), mrValue);
    return SRValue(result);
  }

  // If this is already an SRValue, return it.
  if (auto rvalue = value.getIfSRValue())
    return rvalue;

  auto pValue = value.getIfPValue();
  assert(pValue && "must be PValue if register-passable and not SRValue");
  return emitPValueToSRValue({pValue, expr}, context);
}

MRValue IREmitter::emitMRValue(ASTExprAnd<AnyValue> value,
                               ExprContext context) {
  auto rVal = emitRValue(value, context);
  if (!rVal)
    return {};

  if (auto mr = rVal.getIfMRValue())
    return mr;

  // Promote SRValue/PValue to MRValue.
  if (rVal.isSValue() || rVal.getIfPValue()) {
    Location argLoc = value.expr->getLocation(*this);
    VarDeclOp varOp = emitVarDecl("anonymous*", rVal.getRValueType(), argLoc,
                                  VarDeclKind::Synthesized);
    if (!varOp)
      return {};
    ValueDest dest(MLValue(varOp), context);
    if (!emitRValue({rVal, value.expr}, dest))
      dest.resetForError(*this);
    return MRValue(varOp);
  }

  llvm_unreachable("unknown RValue");
}

/// This helper emits the specified value as an MBValue which has
/// memory-only representation, materializing PValues as needed. This
/// returns null if emission fails.
MBValue IREmitter::emitMBValue(ASTExprAnd<AnyValue> value, ExprContext context,
                               ASTType resultType) {
  BValue bValue = emitBValue(value, context, resultType);
  if (!bValue)
    return {};

  if (auto mb = bValue.getIfMBValue())
    return mb;

  // Drop parametric mutability.
  if (auto mbp = bValue.getIfMBPValue())
    return MBValue(mbp);

  // Mojo can't turn an SBValue into an MBValue - the former only occurs in
  // special places, and cannot be lifetime tracked back to the original RValue
  // it was derived from.  If this assert fires, something is wrong up-stack of
  // this code.
  assert((!bValue.getIfSBValue() || bValue.getRValueType().isRegisterPassable(
                                        value.expr->getLoc(), shared)) &&
         "Cannot convert an SBValue to an MBValue");

  // PValue's and SValues need to be emitted into an owned memory temporary,
  // which we can then decay to an MBValue.
  assert(bValue.getIfPValue() || bValue.isSValue());
  auto mrVal = emitMRValue({bValue, value.expr}, context);
  if (!mrVal)
    return {};
  return MBValue(mrVal);
}

PValue IREmitter::emitPValue(ASTExprAnd<AnyValue> value, ExprContext context,
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
      dest.resetForError(*this);
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
Value IREmitter::emitRefValue(ASTExprAnd<AnyValue> value, ExprContext context) {
  // A DLValue can be a ref when its "load" operation returns a ref.  This can
  // happen when a computed getter returns a ref - e.g. for Dict.
  if (auto dlValue = value.ir.getIfDLValue()) {
    ValueDest dest(context);
    value.ir = dlValue->emitLoad(dest, *this);
    if (!value.ir) {
      dest.resetForError(*this);
      return {};
    }
  }

  // If this got resolved to an MValue then we're done.
  if (value.ir.isMValue())
    return value.ir.getMValueReference();

  // Otherwise we can't support other non-MValue's like borrowed registers or
  // RValue's.
  auto diag = emitError(value.expr->getLoc(), "value");
  if (auto cv = value.ir.getIfCValue())
    diag << " of type " << cv.getRValueType();
  diag << " doesn't have a memory origin" << getContextMessage(context)
       << value.expr->getRange();
  return {};
}

//===----------------------------------------------------------------------===//
// Emission helpers for various value classifications.

/// If the type of the specified value differs from the destination type, emit
/// a rebind operation to convert it.
Value IREmitter::emitRebindOpIfNeeded(Value value, Type destType, SMLoc loc) {
  if (!value || value.getType() == destType)
    return value;

  // Sanity check that rebind isn't *introducing* reference mutability.
  if (auto srcRefType = dyn_cast<RefType>(value.getType()))
    if (auto dstRefType = dyn_cast<RefType>(destType)) {
      assert(!(srcRefType.isMutableKnown(false) &&
               dstRefType.isMutableKnown(true)) &&
             "Rebind is introducing mutability");
      assert(getCanonicalAttr(srcRefType.getAddressSpace()) ==
                 getCanonicalAttr(dstRefType.getAddressSpace()) &&
             "rebind cannot change address space");
    }
  return RebindOp::create(*builder, translateLocation(loc), destType, value);
}

/// If needed, convert the specified value to the target destination type,
/// with a noop cast.  This is used to adjust inconsequential details of the
/// type or for simple things like upcasts.  This does not invoke constructors
/// or do other non-trivial conversions.
///
/// This produces an error and returns null on an invalid conversion.
CValue IREmitter::rebindValue(ASTExprAnd<CValue> value, Type destType) {
  // Materialize a parameter rebind.
  if (auto pvalue = value.ir.getIfPValue())
    return ParamOperatorAttr::getRebind(pvalue.get(), destType);
  if (auto dlValue = value.ir.getIfDLValue()) {
    dlValue->elementType = destType;
    return dlValue;
  }

  // Cannot perform value rebind if only parameters are allowed.
  if (!builder)
    return emitErrorForDynamicValueInParameter(value.expr);

  // Materialize a rebind operation.
  auto loc = value.expr->getLoc();
  if (auto refValue = value.ir.getIfMLValue())
    return MLValue(emitRebindOpIfNeeded(refValue, destType, loc));
  if (auto refValue = value.ir.getIfMRValue())
    return MRValue(emitRebindOpIfNeeded(refValue, destType, loc));
  if (auto refValue = value.ir.getIfMBValue())
    return MBValue(emitRebindOpIfNeeded(refValue, destType, loc));
  if (auto refValue = value.ir.getIfMBPValue())
    return MBPValue(emitRebindOpIfNeeded(refValue, destType, loc));
  if (auto sbValue = value.ir.getIfSBValue())
    return SBValue(emitRebindOpIfNeeded(sbValue, destType, loc));

  auto srValue = value.ir.getIfSRValue();
  assert(srValue && "Unknown value kind");
  return SRValue(emitRebindOpIfNeeded(srValue, destType, loc));
}

/// Emit the specified value into the current destination if present.  This
/// accepts (and silently propagates) null values.
///
/// Note that the `value` provided here may require an implicit conversion
/// into the destination slot, so the input may be memory-only and result be
/// register-passable (and visa-versa).
AnyValue IREmitter::emitResult(AnyValue value, const ExprNode *expr,
                               ValueDest &dest) {
  if (!value) {
    dest.resetForError(*this);
    return {};
  }
  ExprContext context = dest.getContext();

  // If no destination is specified or it is just a contextual type hint or this
  // is a parameter to be destructed, then we can propagate the value directly.
  if (!dest.isSpecified() || isa<LValueInitializerType>(dest.representation)) {
    dest.representation = NullRepresentation();
    return value;
  }

  // If the value is still unresolved, materialize it into the destination.
  auto cValue = value.getIfCValue();
  if (!cValue)
    return emitCValue({value, expr}, dest);
  value = {}; // Only use cValue below.

  // OK, if there is a destination specified, handle them by converging the set
  // of value types we have.
  auto rvType = cValue.getRValueType();

  // If there is a known type for the destination but the value disagrees, emit
  // an implicit conversion directly into the destination.  This keeps values in
  // registers and avoids a "convert + clone" pair for memory->memory
  // conversions.
  if (ASTType requiredType =
          dest.resolveImpliedType(expr->getLoc(), rvType, *this)) {
    if (!requiredType.isEqualCanon(rvType)) {
      cValue = emitImplicitConversionToType({cValue, expr}, requiredType, dest);
      // If this resolved the value dest, then we're done.   This handles the
      // null result case as well.
      if (!dest.isSpecified())
        return cValue;
      assert(cValue);
    }

    // At this point the canonical types line up, but the sugar may not.  Align
    // the sugar so clients don't have to deal with it.
    if (requiredType.mlirType != rvType.mlirType) {
      auto rebindType = requiredType;
      if (cValue.isMValue())
        rebindType = cValue.getMValueType().getWithElement(requiredType);
      cValue = rebindValue({cValue, expr}, rebindType);
      if (!cValue)
        return {};
    }

    rvType = cValue.getRValueType();
  }

  // If the destination is just a required type, then we now know it must agree
  // and therefore don't need to do anything more.
  if (isa<ASTType>(dest.representation)) {
    dest.representation = NullRepresentation(); // Resolved the ValueDest;
    return cValue;
  }

  // If this destination was an LValue whose buffer was already taken to be
  // filled in by a client, then this is just completing the transaction.
  if (isa<LValueBufferTaken>(dest.representation)) {
    dest.representation = NullRepresentation(); // Resolved the ValueDest;

    // The client directly filled in an LValue we provided which is great, but
    // that LValue we provided took ownership of the value, so we need to return
    // the result as a borrow, not an owned reference.
    assert(cValue.isMValue() && "Must be an MValue providing result");
    return MBValue(cValue.getMValueReference());
  }

  // We know we have an RValue/BValue and the destination is some kind of
  // LValue.  Emit the dest to figure out where to store it.
  LValue destLV = dest.getLValueForResult(expr->getLoc(), rvType,
                                          /*allowIncompatibleTypes=*/true,
                                          /*requireMLValue=*/false, *this);
  if (!destLV) {
    dest.resetForError(*this);
    return {};
  }

  // This will have completely resolved all the ValueDest possibilities.
  assert(!dest.isSpecified() || isa<LValueBufferTaken>(dest.representation));
  dest.representation = NullRepresentation(); // Resolved the ValueDest;

  // Finally, store the value into the lvalue.
  return emitStoreToLValue({cValue, expr}, destLV, context);
}

CValue IREmitter::emitCResult(CValue value, const ExprNode *expr,
                              ValueDest &dest) {
  // Emitting a CValue always produces a CValue.
  auto result = emitResult(value, expr, dest);
  assert((!result || result.getIfCValue()) &&
         "emitting a CValue as a result should always produce a CValue");
  return result.getIfCValue();
}

/// Destructuring the specific PValue against the provided target expr
/// (which specifies the pattern).
LogicalResult IREmitter::emitDestructuringPValue(PValue value,
                                                 const ExprNode *targetExpr) {
  // Clear the builder to indicate that an PValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder, {});
  return targetExpr->emitDestructuringPValue(value, *this);
}

/// Emit the specified expression into the specified destination.
AnyValue IREmitter::emitExpr(const ExprNode *expr, ValueDest &dest) {
  assert(expr && "cannot emit a null node");
  if (auto result = expr->emitIR(dest, *this))
    return result;
  dest.resetForError(*this);
  return {};
}

AnyValue IREmitter::emitExpr(const ExprNode *expr, ExprContext context,
                             ASTType resultType) {
  ValueDest dest(resultType, context);
  return emitExpr(expr, dest);
}

RValue IREmitter::emitExprRValue(const ExprNode *expr, ExprContext context,
                                 ASTType resultType) {
  return emitRValue({emitExpr(expr, context, resultType), expr}, context,
                    resultType);
}

CValue IREmitter::emitExprCValue(const ExprNode *expr, ExprContext context,
                                 ASTType resultType) {
  return emitCValue({emitExpr(expr, context, resultType), expr}, context);
}

SRValue IREmitter::emitExprSRValue(const ExprNode *expr, ExprContext context,
                                   ASTType resultType) {
  return emitSRValue({emitExpr(expr, context, resultType), expr}, context,
                     resultType);
}

PValue IREmitter::emitExprPValue(const ExprNode *expr, ExprContext context,
                                 ASTType resultType) {
  // Clear the builder to indicate that an PValue must be emitted.
  llvm::SaveAndRestore savedBuilder(builder, {});
  llvm::SaveAndRestore savedContext(paramContext, context);

  // Emit the expression using the contextual type if present.
  AnyValue rep = emitExpr(expr, context, resultType);
  return emitPValue({rep, expr}, context);
}

LValue IREmitter::emitExprLValue(const ExprNode *expr, ValueDest &dest) {
  AnyValue anyValue = expr->emitIR(dest, *this);
  if (!anyValue) {
    dest.resetForError(*this);
    return {}; // Error already diagnosed.
  }
  return emitLValue({anyValue, expr}, dest);
}

/// Emit a copy of the specified value, producing a new owned instance of the
/// value in the specified destination.  This returns an RValue if
/// there is no consuming dest, otherwise a BValue.
CValue IREmitter::emitCopyOfValue(ASTExprAnd<CValue> value, ValueDest &dest) {
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
          RefLoadOp::create(*builder, value.expr->getLocation(*this), address);
      value.ir = SRValue(result);
    }

    return emitCResult(value.ir, value.expr, dest);
  }

  // Otherwise, we'll need to invoke the copyinit method which will take the
  // destination by reference, so we're dealing with a memory case.

  // Memory-only copyinit will take the destination as address space zero, so
  // we need to reject ValueDest's expecting it in GPU memory.
  if (dest.isNonDefaultAddressSpace()) {
    emitError(exprLoc, "value of type ")
        << valueType << " cannot be copied into a non-default address space"
        << value.expr->getRange();
    return {};
  }

  // Materialize any PValue directly, so we can handle non-copyable and
  // non-movable types.
  if (auto pValue = value.ir.getIfPValue()) {
    // PValues don't have origins and are immortal with respect to the compiler.
    // Emit a memcpy into the LValue. Creating an SSA value of the memory-only
    // type for the sake of memcpy is safe because the bulk store will ensure
    // the variable does not get promoted off the stack, and after struct
    // lowering, the type is erased down to its MLIR constituents anyways.

    // FIXME: This isn't correct - it is emitting memory-only values into an
    // SSA value and then using lit.ref.store on the memory only value!
    SRValue regValue = emitPValueToSRValue({pValue, value.expr}, dest.context);
    if (!regValue)
      return {};
    MLValue destBuffer = dest.getMLValueForResult(exprLoc, valueType, *this);
    if (!destBuffer)
      return {};
    regValue = emitRebindOpIfNeeded(
        regValue, ASTType(destBuffer.getType()).getReferenceElementType(),
        exprLoc);
    RefStoreOp::create(*builder, translateLocation(exprLoc), regValue,
                       destBuffer);
    CValue result = MRValue(destBuffer);
    return emitCResult(result, value.expr, dest);
  }

  // Verify that the type is copyable in this way so we can generate tailored
  // error messages, rather than just allowing IREmitter to do it.
  if (!valueType.isImplicitlyCopyable(exprLoc, shared)) {
    // If the value is an RValue, it might be that the type isn't copyable or
    // movable at all. If so, give a specific error about this.
    if (value.ir.getIfRValue() && !valueType.isMovable(exprLoc, shared) &&
        !valueType.isExplicitlyCopyable(exprLoc, shared)) {
      emitError(exprLoc, "value of type ")
          << valueType
          << " cannot be copied or moved; consider conforming it to 'Movable'"
          << value.expr->getRange();
      return {};
    }

    auto diag = emitError(exprLoc, "value of type ")
                << valueType << " cannot be implicitly"
                << " copied, it does not conform to 'ImplicitlyCopyable'"
                << value.expr->getRange();

    // Decide if we can take ownership of the specified value.
    auto canTransferFrom = [&]() -> bool {
      // Can only transfer from an MValue.  If it is already an RValue, then
      // transferring won't help!
      if (!value.ir.isMValue() || value.ir.getIfRValue())
        return false;

      Value val = OriginTrackable::findUnderlyingValueFromField(
          value.ir.getMValueReference());
      if (!val)
        return false;
      // Can't transfer from (e.g.) read arguments.
      return cast<RefType>(val.getType()).isMutableKnown(true);
    };

    // Suggest transfer if the type is movable, or if it is a transferable
    // MValue.
    if ((valueType.isMovable(exprLoc, shared) || canTransferFrom())) {
      diag.attachNote(exprLoc)
          << "consider transferring the value with '^'"
          << FixIt::insertAfterToken(value.expr->getRangeEnd(), "^",
                                     shared.diags);
    }

    // Suggest .copy() if the type is explicitly copyable and we're trying to
    // implicitly copy it.
    if (valueType.isExplicitlyCopyable(exprLoc, shared)) {
      diag.attachNote(exprLoc)
          << "you can copy it explicitly with '.copy()'"
          << FixIt::insertAfterToken(value.expr->getRangeEnd(), ".copy()",
                                     shared.diags);
    }
    return {};
  }

  // __copyinit__ has signature: `(existing: Self) -> Self`.
  return emitNamedMethodCall("__copyinit__", CallOperands{value.expr, {value}},
                             dest, CallSyntax::kImplicitCopyInit);
}

CValue IREmitter::emitStoreToLValue(ASTExprAnd<CValue> value, LValue destLV,
                                    ExprContext context) {
  // Convert nonmaterializables.
  if (auto nmTarget =
          value.ir.getRValueType().getNonmaterializableTarget(shared)) {
    if (nmTarget.isEqualCanon(destLV.getRValueType())) {
      // If the destination is an MLValue with a matching type, then just
      // materialize directly into it and return instead of allocating a
      // temporary if the conversion constructor requires one.
      ValueDest nmConversionDest(destLV, context);
      return emitConstructorCall(nmTarget, CallOperands(value.expr, {value}),
                                 CallSyntax::kTypeCall, nmConversionDest);
    }
  }

  assert(value.ir.getRValueType().isEqualCanon(destLV.getRValueType()) &&
         "Types should match");

  // If the destination is a computed LValue, then perform a write.
  if (auto dlValue = destLV.getIfDLValue())
    return dlValue->emitStore(value, *this);

  // If the destination is a RLValue, then we are resolving a 'ref' or 'bind'
  // value into a VarDeclOp.
  if (auto rlValue = destLV.getIfRLValue()) {
    // The destination must be a VarDeclOp by construction.
    VarDeclOp refOp = cast<VarDeclOp>(rlValue.getDefiningOp());
    assert(refOp &&
           (refOp.getKind() == VarDeclKind::Ref ||
            refOp.getKind() == VarDeclKind::Bind) &&
           "not a ref or bind to initialize!");

    // Handle 'bind' by determining if this is a 'var' or immutable 'ref'.
    if (refOp.getKind() == VarDeclKind::Bind) {
      // If the value isn't a reference, we materialize it into a var binding.
      if (!value.ir.isMValue()) {
        // Switch the vardecl so that uses of it are treated as MBValue instead
        // of MLValues.
        refOp.setKind(VarDeclKind::Bound);
        refOp.getResult().setType(
            refOp.getType().getWithElement(value.ir.getRValueType()));
        // Now we store the value into the var decl.
        ValueDest bindDest(MLValue(refOp), context);
        emitBValue({value.ir, value.expr}, bindDest);
        return MBValue(refOp);
      }

      // Otherwise, handle this as an immutable ref.
      refOp.setKind(VarDeclKind::Ref);
      Value refValue = value.ir.getMValueReference();

      if (!cast<RefType>(refValue.getType()).isMutableKnown(false)) {
        refValue = RefImmutOp::create(*builder, value.expr->getLocation(*this),
                                      refValue);
      }
      value.ir = MBValue(refValue);
    }

    // If this is a 'ref', then we want non-MValues to be an error.
    Value mValue = emitRefValue(value, EC_RefBinding);
    if (!mValue)
      return {};

    // Now that we have the origin of the input, we can replace the placeholder
    // with the actual type so that uses of it will have the correct origin.
    refOp.getResult().setType(refOp.getType().getWithElement(mValue.getType()));
    RefStoreOp::create(*builder, translateLocation(value.expr->getLoc()),
                       mValue, refOp);
    return CValue::getMValueForRef(mValue); // Return the input reference.
  }

  // Otherwise, we know we have an MLValue destination.
  MLValue destRef = destLV.getIfMLValue();
  assert(destRef && "No other known LValue");
  ASTType valueType = value.ir.getRValueType();
  SMLoc exprLoc = value.expr->getLoc();

  // Verify that the result MLValue is in the right address space for a
  // __copyinit__/__moveinit__ call.
  if (!cast<RefType>(destRef.getType()).isDefaultAddrSpace() &&
      valueType.getRegisterPassability(exprLoc, shared) !=
          TypeConvention::RegisterPassableTrivial) {
    emitError(exprLoc, "value of type ")
        << valueType
        << " cannot be copied or moved into a non-default address space"
        << value.expr->getRange();
    return {};
  }

  // If the input is an LValue/BValue (incl PValue) that we don't own, or if it
  // has no __moveinit__, then copy it into the destination.
  if (!value.ir.getIfRValue() || !valueType.isMovableFrom(value, shared) ||
      value.ir.getIfPValue()) {
    ValueDest dest(destLV, context);
    auto result = emitCopyOfValue(value, dest);
    if (!result)
      dest.resetForError(*this);
    return result;
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
    // Store the value to memory after adjusting sugar.  StoreOp takes
    // ownership of the input SRValue.
    val = emitRebindOpIfNeeded(
        val, ASTType(destRef.getType()).getReferenceElementType(), exprLoc);
    RefStoreOp::create(*builder, translateLocation(exprLoc), val, destRef);
    // Must return a borrow of the result, use SBValue if we can to avoid a load
    // but otherwise we need a MBValue for non-trivial types.
    if (valueType.isTrivial(exprLoc, shared))
      return SBValue(val);
    return MBValue(destRef);
  }

  // Otherwise, assign with a move constructor.  We own the RValue, so prefer
  // to use __moveinit__ if present.
  if (shared.typeHasMember(valueType, "__moveinit__", value.expr->getLoc())) {
    // `__moveinit__(owned existing: Self) -> Self`.
    ValueDest moveDest(destRef, context);
    if (!emitNamedMethodCall("__moveinit__", CallOperands{value.expr, {value}},
                             moveDest, CallSyntax::kImplicitMoveInit))
      return {};
    return MBValue(destRef);
  }

  // Otherwise, we have to move this thing but don't have a move constructor!
  emitError(value.expr->getLoc())
      << "cannot transfer value into destination, because " << valueType
      << " doesn't implement `__moveinit__`";
  return {};
}

/// Emit IR for the specified expression without adding it to the current
/// execution context.  This even allows evaluating dynamic expressions in a
/// parameter context.  When the result is computed, evaluate the specified
/// callback on the result and then discard the result.
///
/// On failure, an error is emitted and the callback is not invoked.
///
/// This is used for evaluating expressions like `origin_of(x)` and
/// `type_of(x)` and `ref [x] T`.
void IREmitter::emitExpressionWithOutEvaluatingIt(
    const ExprNode *expr, ExprContext exprContext,
    std::function<void(CValue, IREmitter &emitter)> callback) {
  SMLoc loc = expr->getLoc();
  // The emitter indicates what context to do name lookup against, but cannot
  // be used to emit the IR into.  Find something in the declScope with an
  // Operation (e.g. a function), which will allow us to put in a Block to emit
  // into.  This is a bit of a hack, but is required because some things scan
  // up the region hierarchy.
  ASTDecl *curDecl = &declScope;
  Operation *opToInsertInto = nullptr;
  // Scan for an operation with a region.
  while (!(opToInsertInto = curDecl->getIfOperation()) ||
         opToInsertInto->getNumRegions() == 0) {
    curDecl = curDecl->getParentDecl();
    if (!curDecl) {
      emitError(loc, "INTERNAL ERROR: could not find context to emit IR "
                     "into.  Please file a bug.");
      return;
    }
  }

  auto location = expr->getLocation(*this);

  // Okay we found an operation with a region.  Abuse it :-) by adding a new
  // block, which keeps any code we're emitting contained.
  Region &r = opToInsertInto->getRegion(0);
  Block &tmpBlock = r.emplaceBlock();
  IREmitter tmpEmitter(declScope, OpBuilder::atBlockBegin(&tmpBlock));

  // Go further and add a 'try' op to it, ensuring that throwing functions are
  // allowed in this expression.
  VarDeclOp errDecl =
      tmpEmitter.emitVarDecl("__try_error__", UnresolvedType::get(getContext()),
                             location, VarDeclKind::Synthesized);
  auto tryOp = TryOp::create(*tmpEmitter.builder, location, errDecl);

  // Parse the expression into the try block.
  tmpEmitter.builder->createBlock(&tryOp.getTryRegion());

  // Emit the expression and invoke the callback on success.
  CValue subExprValue = tmpEmitter.emitExprCValue(expr, exprContext);
  if (subExprValue)
    callback(subExprValue, tmpEmitter);

  // Finally, remove our temp block
  tmpBlock.erase();
}

//===----------------------------------------------------------------------===//
// Emission helpers for specific value types.

ASTType IREmitter::emitExprType(const ExprNode *expr, bool allowUnbound) {
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
  return emitType({value, expr}, allowUnbound);
}

/// This emits the specified PValue as a type, binding defaulted parameters
/// etc if needed.
ASTType IREmitter::emitType(ASTExprAnd<PValue> value, bool allowUnbound) {
  if (!value.ir)
    return {};

  ASTType type = value.ir.getIfTypeValue();
  if (!type) {
    emitError(value.expr->getLoc(), "expected a type, not a value")
        << value.expr->getRange();
    return {};
  }

  // If the caller accepts a fully unbound type and the type is unbound, return
  // it now without verifying the bindings.
  if (allowUnbound)
    return type;

  // Check for a function type.
  if (auto sig = dyn_cast<FnTypeGeneratorType>(type)) {
    // For a fully bound type, require that the origin set is concrete.
    if (isa<UnboundAttr>(sig.getCaptureOrigins())) {
      emitError(value.expr->getLoc(),
                "function type missing required origin set parameter")
          << value.expr->getRange();
      return {};
    }
  }

  // Verify that all of the parameters for this type are bound.  We allow
  // PValues to refer to parametric type, but anything calling `emitType`
  // can only handle fully bound types.
  auto *decl = type.getDecl(shared);
  if (!decl) // MLIR types are never parameterized.
    return type;

  auto structDecl = dyn_cast_or_null<StructDeclOp>(decl->getIfOperation());
  if (!structDecl)
    return type;

  // Build up a ParamBindings set to validate and check the bindings. Skip
  // unbound values.
  ParamBindings paramBindings(getDeclScope(), value.expr);
  for (TypedAttr binding : type.getParamBindings())
    paramBindings.addPrechecked(value.expr, binding);

  // Check the existing bindings against the full signature of the type and make
  // sure it is fully bound.
  ParameterExprArrayAttr bindingValuesAttr =
      paramBindings.verifyStructBindings(*decl, structDecl.getSignature(),
                                         /*partial=*/false);
  if (!bindingValuesAttr)
    return {};

  // If verifyBindings changed the bindings set, then we may have had an
  // empty varargs list or something.  Rebind the StructType.
  if (bindingValuesAttr.getValue() != type.getParamBindings())
    type = structDecl.bindReference(bindingValuesAttr);
  return type;
}

RValue IREmitter::emitI1(ASTExprAnd<CValue> value, ExprContext context) {
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
    value.ir = emitNamedMethodCall(
        "__bool__", CallOperands{value.expr, {{value.ir, value.expr}}},
        boolDest, CallSyntax::kMethodCall);
  }

  // Then we use __mlir_i1__ to convert to an i1 value.
  ValueDest boolDest(context);
  CValue litBoolCall = emitNamedMethodCall(
      "__mlir_i1__", CallOperands{value.expr, {{value.ir, value.expr}}},
      boolDest, CallSyntax::kMethodCall);

  // If we got back a sugared PValue call to the method, then drop the sugar.
  // This reduces the size of the printed IR, making it easier to read, and the
  // user never wants to see a call to this function in a diagnostic anyway.
  if (auto pvalue = litBoolCall.getIfPValue())
    if (auto sugar = llvm::dyn_cast_or_null<SugarAttr>(pvalue.get()))
      if (sugar.getKind() == SugarKind::AlwaysInlineBuiltin)
        litBoolCall = sugar.getExpanded();

  return emitRValue({litBoolCall, value.expr}, context);
}

RValue IREmitter::emitExprI1(const ExprNode *condExpr, ExprContext context) {
  return emitI1({emitExprCValue(condExpr, context), condExpr}, context);
}

CValue IREmitter::emitIndex(ASTExprAnd<AnyValue> value, ExprContext context) {
  // If the value is already of index type, just use it.
  if (CValue cvalue = value.ir.getIfCValue())
    if (isa<IndexType>(cvalue.getRValueType().mlirType))
      return cvalue;

  ValueDest dest(context);
  auto result =
      emitNamedMethodCall("__mlir_index__", CallOperands{value.expr, {value}},
                          dest, CallSyntax::kMethodCall);

  // If we got back a sugared PValue call to the method, then drop the sugar.
  // This reduces the size of the printed IR, making it easier to read, and the
  // user never wants to see a call to this function in a diagnostic anyway.
  if (auto pvalue = result.getIfPValue())
    if (auto sugar = llvm::dyn_cast_or_null<SugarAttr>(pvalue.get()))
      if (sugar.getKind() == SugarKind::AlwaysInlineBuiltin)
        result = sugar.getExpanded();

  return result;
}

CValue IREmitter::emitIndex(const ExprNode *expr, ExprContext context) {
  return emitIndex({emitExprCValue(expr, context), expr}, context);
}

CValue IREmitter::emitBool(ASTExprAnd<PValue> value, ValueDest &dest) {
  ASTType boolType = shared.getBuiltinBoolType(declScope, value.expr->getLoc());
  return emitConstructorCall(boolType, CallOperands(value.expr, {value}),
                             CallSyntax::kImplicitConvert, dest);
}

CValue IREmitter::emitBool(ASTExprAnd<PValue> value, ExprContext context) {
  ValueDest dest(context);
  return emitBool(value, dest);
}

CValue IREmitter::emitInt(ASTExprAnd<AnyValue> indexValue, ValueDest &dest) {
  ASTType intType = shared.lookupBuiltinType("Int", getDeclScope(),
                                             indexValue.expr->getLoc());

  // Build Int from __mlir_type.index explicitly: Int.__init__(*, mlir_value=…)
  CallOperands intCtorOperands(indexValue.expr);
  intCtorOperands.add(StringAttr::get(getContext(), "mlir_value"), indexValue);
  return emitConstructorCall(intType, std::move(intCtorOperands),
                             CallSyntax::kTypeCall, dest);
}

CValue IREmitter::emitInt(ASTExprAnd<AnyValue> indexValue,
                          ExprContext context) {
  ValueDest dest(context);
  return emitInt(indexValue, dest);
}

/// This returns an instance of Tuple[...] with the specified element types
/// installed.
ASTType IREmitter::getBuiltinTupleInstantiation(llvm::SMLoc loc,
                                                ArrayRef<Type> elements) {
  auto tupleType = shared.getBuiltinTupleType(declScope, loc);
  if (tupleType.isTypeCheckErrorType())
    return {};
  ASTDecl *typeDecl = ASTType(tupleType).getDecl(shared);
  auto structOp = dyn_cast_or_null<StructDeclOp>(typeDecl->getIfOperation());
  if (!structOp) {
    emitError(loc, "internal error: Tuple type not found or not a struct");
    return {};
  }

  SyntheticNode tmpExpr(loc);
  ParamBindings bindings(getDeclScope(), &tmpExpr);
  for (ASTType elt : elements)
    bindings.add(&tmpExpr, PValue(elt));

  // Check the bindings.
  auto metaType = cast<StructMetaType>(tupleType.getMetaType());
  auto bindingsAttr =
      bindings.verifyStructBindings(*typeDecl, metaType.getSignature(),
                                    /*partial=*/false);
  if (!bindingsAttr)
    return {};

  // Ok, we succeeded at reparameterizing the type.
  return metaType.getType().bindAll(bindingsAttr.getValue());
}

//===----------------------------------------------------------------------===//
// Return emission helpers.

MLValue IREmitter::findNearestErrorSlot() {
  assert(builder && "cannot raise in a context without a builder");
  Operation *opForRaise = findOpProcessingRaise(builder->getInsertionBlock());
  // Return null to indicate that the current context cannot raise.
  if (!opForRaise)
    return {};

  // In a raising function, the error slot is always the second last argument.
  if (auto func = dyn_cast<FnOp>(opForRaise))
    return func.getArgument(func.getNumArguments() - 2);

  // Otherwise, the error slot is carried by the surrounding try op.
  return cast<LIT::TryOp>(opForRaise).getErr();
}

void IREmitter::emitNormalReturn(ImplicitLocOpBuilder &builder, Value value,
                                 bool emitEndFunc) {
  auto func = getBlockParentOfType<FnOp>(builder.getInsertionBlock());
  assert(func && "Emitting a return in a non-function?");

  auto signature = func.getFuncTypeGenerator();
  if (value) {
    // If we have a value, then make sure any sugar is adjusted.

    // Rebind away any sugar if it exists.
    if (value.getType() != func.getMLIRResultType())
      value = RebindOp::create(builder, func.getMLIRResultType(), value);
  } else {
    // If we're missing a value, then we either have a memory result that has
    // already been emitted to its slot, or a function that returns None. Either
    // way, generate a None or i1 to return with lit.return.

    // If the function returns a None type value by-reference, fill it in.  This
    // happens in throwing functions.
    if (signature.hasMemoryOnlyResult() &&
        ASTType(func.getUserResultType()).isNoneType()) {
      assert(signature.getArgConventions().back() ==
                 ArgConvention::ByRefResult &&
             "by-ref result should be the last argument");

      // This value will also get returned unless the function throws.
      value =
          ParamConstantOp::create(builder, NoneAttr::get(func.getContext()));
      RefStoreOp::create(builder, value, func.getArguments().back());
    }

    // Otherwise, the resulting actual function result must be a none-type or a
    // bool for a throwing result.
    if (signature.isThrows())
      value = ParamConstantOp::create(builder, builder.getBoolAttr(false));
    else if (!value)
      value =
          ParamConstantOp::create(builder, NoneAttr::get(func.getContext()));
  }

  // Handle any `deinit` argument by marking it destroyed.
  for (auto [conv, arg] : llvm::zip(signature.getArgConventions(),
                                    func.getBody()->getArguments())) {
    if (conv == ArgConvention::DeinitMem)
      LIT::OwnershipMarkDestroyedOp::create(builder, arg);
  }

  // Finally we emit a normal return with lit.return.
  assert(value && "Didn't specify a return value for the function");
  LIT::ReturnOp::create(builder, value);

  // If requested, emit the end func.
  if (emitEndFunc)
    EndFnOp::create(builder);
}

/// Emit a normal return (not a 'raise' return) out of the function, along
/// with any special logic that goes with it.  If the value is missing this is
/// treated as a 'return;' synthesizing a None result.
void IREmitter::emitNormalReturn(Location loc, Value value, bool emitEndFunc) {

  // If this function returns in a register, load the result value from the
  // result slot temp. We compile things like:
  //    fn example(out x: Int):
  // to have a local vardecl that can be mutated, and is loaded implicitly
  // when a "return" with no expression is used.
  if (!value) {
    auto func = getBlockParentOfType<FnOp>(builder->getInsertionBlock());
    if (func.getNamedResultAttr() &&
        !func.getFuncTypeGenerator().hasMemoryOnlyResult()) {
      auto *funcDecl = declScope.getNearestDeclOfType<FnOp>();
      assert(funcDecl && "must be in a function");
      ArrayRef<ASTDecl *> declList =
          funcDecl->lookupInCurrentScope(func.getNamedResultAttr());
      assert(declList.size() == 1 && "result temp should always be findable");
      auto irVal = declList[0]->getIfIRValue().getIfMLValue();
      assert(irVal && "result temp should always be in memory");
      SyntheticNode exprTmp(funcDecl->getLoc());
      // Move the source by interpreting the MLValue as an MRvalue.
      value = emitSRValue({MRValue(irVal), &exprTmp}, EC_ReturnValue);
      if (!value)
        return;
    }
  }

  ImplicitLocOpBuilder b(loc, *builder);
  emitNormalReturn(b, value, emitEndFunc);
}

//===--------------------------------------------------------------------===//
// Var emission helpers.
//===--------------------------------------------------------------------===//

VarDeclOp IREmitter::emitVarDecl(const Twine &name, Type type, Location loc,
                                 VarDeclKind kind) {
  if (!builder) {
    emitErrorForDynamicValueInParameter(loc);
    return {};
  }
  StringAttr originAttr = declScope.mangleParamName(name);
  return VarDeclOp::create(*builder, loc, type, name.str(), originAttr, kind);
}

VarDeclOp IREmitter::emitVarDecl(StringAttr name, Type type, Location loc,
                                 VarDeclKind kind) {
  return emitVarDecl(name.strref(), type, loc, kind);
}
