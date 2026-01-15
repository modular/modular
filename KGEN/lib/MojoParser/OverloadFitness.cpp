//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the components for overload fitness evaluation.
//
//===----------------------------------------------------------------------===//

#include "OverloadFitness.h"
#include "CallEmission.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/Constraints.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "ParamInf.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Diagnostic emission implementation
//===----------------------------------------------------------------------===//

namespace {
/// Helper class to emit errors without cluttering the evaluation logic.
struct DiagEmitter : public SharedStateUser {
  DiagEmitter(SharedState &shared, SMLoc callLoc)
      : SharedStateUser(shared), callLoc(callLoc) {}

  MojoInflightDiag
  unexpectedKwArgs(ArrayRef<StringAttr> unknownKwOperands) const;
  MojoInflightDiag wrongParamCount(size_t expectedNumParams,
                                   size_t actualNumParams) const;
  MojoInflightDiag wrongArgCountWithPack(size_t minRequiredArgs,
                                         size_t maxAllowedArgs,
                                         size_t numOperands) const;
  MojoInflightDiag unresolvedPackCount(size_t numOperands) const;
  MojoInflightDiag wrongPosOnlyCount(size_t minRequiredArgs,
                                     size_t maxAllowedArgs, size_t numOperands,
                                     const Twine &argOrParam) const;
  MojoInflightDiag resultGenericMemType(Type outputType) const;
  MojoInflightDiag argGenericMemType(size_t expectedArgIdx, Type expectedType,
                                     PogListAttr argListAttr) const;
  MojoInflightDiag missingArgs(ArrayRef<StringAttr> missingArgs,
                               const Twine &kindStr) const;
  MojoInflightDiag
  posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const;
  MojoInflightDiag tooManyPosArgs(size_t maxAllowedArgs,
                                  size_t numPosOperands) const;
  MojoInflightDiag byPosAndKw(ArrayRef<StringAttr> names) const;
  MojoInflightDiag badImplicitConversion(ASTType fromType,
                                         ASTType toType) const;

private:
  SMLoc callLoc;

  MojoInflightDiag initDiag() const { return shared.emitError(callLoc); }
};
} // namespace

MojoInflightDiag
DiagEmitter::unexpectedKwArgs(ArrayRef<StringAttr> unknownKwOperands) const {
  auto diag = initDiag();
  emitUnknownKeywords(diag, unknownKwOperands, "argument");
  return diag;
}

MojoInflightDiag DiagEmitter::wrongParamCount(size_t expectedNumParams,
                                              size_t actualNumParams) const {
  auto diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, /*minRequired=*/expectedNumParams,
                           /*maxAllowed=*/expectedNumParams, actualNumParams,
                           "parameter");
  return diag;
}

MojoInflightDiag DiagEmitter::wrongArgCountWithPack(size_t minRequiredArgs,
                                                    size_t maxAllowedArgs,
                                                    size_t numOperands) const {
  auto diag = initDiag() << "callee with non-empty variadic pack argument";
  emitWrongArgOrParamCount(diag, minRequiredArgs, maxAllowedArgs, numOperands,
                           "positional operand");
  return diag;
}

MojoInflightDiag DiagEmitter ::unresolvedPackCount(size_t numOperands) const {
  return initDiag() << "assigning " << numOperands << " operand"
                    << plural(numOperands)
                    << " to an unresolvable variadic pack argument";
}

MojoInflightDiag DiagEmitter::wrongPosOnlyCount(size_t minRequiredArgs,
                                                size_t maxAllowedArgs,
                                                size_t numOperands,
                                                const Twine &argOrParam) const {
  auto diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, minRequiredArgs, maxAllowedArgs, numOperands,
                           "positional " + argOrParam);
  return diag;
}

MojoInflightDiag DiagEmitter::resultGenericMemType(Type outputType) const {
  return initDiag()
         << "result cannot bind __TypeOfAllTypes type to memory-only type "
         << ASTType(outputType);
}

MojoInflightDiag DiagEmitter::argGenericMemType(size_t argIdx,
                                                Type expectedType,
                                                PogListAttr argListAttr) const {
  return initDiag() << "cannot bind __TypeOfAllTypes type to memory-only type "
                    << ASTType(expectedType) << " expected by argument "
                    << argListAttr.getPogs()[argIdx].getName();
}

/// Attach extra type conversion error detail or hints to the user when
/// reporting an error passing `operand` to an argument of type `argType`.
static void addTypeConversionDetail(MojoInflightDiag &diag,
                                    ASTExprAnd<AnyValue> operand,
                                    ASTType argType, SharedState &shared) {
  auto loc = operand.expr->getLoc();
  ASTType operandType = operand.ir.getRValueTypeIfResolvable();
  if (!operandType) {
    diag.attachNote(loc) << "try resolving the overloaded function first";
    return;
  }
  // Try to detect mismatched memory result type.
  auto lhsSig = sugarDynCast<FnTypeGeneratorType>(operandType);
  auto rhsSig = sugarDynCast<FnTypeGeneratorType>(argType);
  if (lhsSig && rhsSig) {
    auto getByRefResult = [](FnTypeGeneratorType sig) -> std::pair<bool, Type> {
      return {sig.getBody().hasMemoryOnlyResult(),
              ASTType(sig.getUserResultType())};
    };
    auto [lhsByRef, lhsRetType] = getByRefResult(lhsSig);
    auto [rhsByRef, rhsRetType] = getByRefResult(rhsSig);
    if (lhsByRef == rhsByRef || lhsRetType != rhsRetType)
      return;
    // Different result semantics but same result type.
    diag.attachNote(loc) << "memory-only type bound to generic result type: "
                         << (lhsByRef ? "payload" : "argument") << " returns "
                         << ASTType(lhsRetType) << " by reference";
    return;
  }
}

/// Emit a tailored diagnostic when failing to convert a value to type !lit.ref.
/// This happens when the user is forming a Reference incorrectly which happens
/// when confusion and details run the highest.
static void diagnoseFailedRefTypeConversion(MojoInflightDiag &diag,
                                            ASTExprAnd<AnyValue> operand,
                                            RefType argType,
                                            SharedState &shared) {
  diag << "ref " << ASTType(argType.getElementType());

  auto loc = operand.expr->getLoc();
  if (operand.ir.getIfRValue()) {
    diag.attachNote(loc) << "cannot bind an RValue to a reference";
    return;
  }
  if (!operand.ir.isMValue()) {
    diag.attachNote(loc) << "operand does not have a memory representation";
    return;
  }

  auto operandRefTy = operand.ir.getMValueType();
  if (!ASTType(argType.getElementType())
           .isEqualCanon(operandRefTy.getElementType())) {
    diag.attachNote(loc) << "operand element type "
                         << ASTType(operandRefTy.getElementType())
                         << " doesn't match expected element type "
                         << ASTType(argType.getElementType());
  } else if (argType.getAddressSpace() != operandRefTy.getAddressSpace()) {
    diag.attachNote(loc) << "operand address space "
                         << operandRefTy.getAddressSpace()
                         << " doesn't match expected address space "
                         << argType.getAddressSpace();
  } else if (!IREmitter::canZeroCostConvert(operandRefTy.getOriginType(),
                                            argType.getOriginType(), shared)) {
    diag.attachNote(loc) << "operand mutability " << operandRefTy.isMutable()
                         << " doesn't match expected mutability "
                         << argType.isMutable();
  } else if (!IREmitter::canZeroCostConvert(operandRefTy, argType, shared)) {
    auto operandO = operandRefTy.getOrigin();
    auto argO = argType.getOrigin();
    // Strip off mutcasts etc - if the origins still differ we can complain
    // about the simpler thing.
    auto operandOS = OriginType::stripMutCastAndFieldExtract(operandO);
    auto argOS = OriginType::stripMutCastAndFieldExtract(argO);
    if (operandOS != argOS) {
      argO = argOS;
      operandO = operandOS;
    }

    diag.attachNote(loc) << "operand origin '"
                         << ASTType::getOriginAsString(operandO, &shared)
                         << "' doesn't match expected origin '"
                         << ASTType::getOriginAsString(argO, &shared) << "'";
  }
}

namespace M::KGEN::LIT {
void printUValueTypeInfo(const AnyValue &value, MojoInflightDiag &diag) {
  if (auto initList = value.getIfInitializer()) {
    switch (initList->syntax) {
    case InitializerUValue::kSlice:
      diag << "slice initializer";
      break;
    case InitializerUValue::kListLiteral:
      diag << "list literal";
      break;
    case InitializerUValue::kDictLiteral:
      diag << "dictionary literal";
      break;
    case InitializerUValue::kSetInitLiteral:
      diag << "initializer list or set literal";
      break;
    }
  } else
    diag << "unknown overload";
}

void emitWrongTypeDiag(MojoInflightDiag &diag, ASTExprAnd<AnyValue> operand,
                       ASTType expectedType, size_t argIdx,
                       PogListAttr argListAttr, CallSyntax syntax,
                       SharedState &shared) {
  // Special case implicit conversions with a custom message.
  if (syntax == CallSyntax::kImplicitConvert) {
    if (ASTType type = operand.ir.getRValueTypeIfResolvable())
      diag << type;
    else
      printUValueTypeInfo(operand.ir, diag);
    diag << " value to " << expectedType;
    return;
  }

  diag << "value passed to " << argListAttr.getPogs()[argIdx].getName()
       << " cannot be converted from " << operand.expr->getRange();
  ASTType rValueType = operand.ir.getRValueTypeIfResolvable();
  bool isConvertingTypeValue = expectedType.getMetaType() == rValueType;
  if (rValueType) {
    if (isConvertingTypeValue)
      diag << "type value " << expectedType;
    else
      diag << rValueType;
  } else {
    printUValueTypeInfo(operand.ir, diag);
  }
  diag << " to ";

  if (auto refType = sugarDynCast<RefType>(expectedType)) {
    diagnoseFailedRefTypeConversion(diag, operand, refType, shared);
    return;
  }

  diag << (isConvertingTypeValue ? "an instance of " : "") << expectedType;
  if (isConvertingTypeValue)
    diag << "; did you mean to instantiate " << expectedType << "?";
  addTypeConversionDetail(diag, operand, expectedType, shared);
}
} // namespace M::KGEN::LIT

MojoInflightDiag DiagEmitter::missingArgs(ArrayRef<StringAttr> missingArgs,
                                          const Twine &kindStr) const {
  MojoInflightDiag diag = initDiag();
  emitMissing(diag, missingArgs, kindStr + " argument");
  return diag;
}

MojoInflightDiag
DiagEmitter::posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const {
  MojoInflightDiag diag = initDiag();
  emitPosOnlyPassedByKw(diag, posOnlyPassedByKw, "argument");
  return diag;
}

MojoInflightDiag DiagEmitter::tooManyPosArgs(size_t maxAllowedArgs,
                                             size_t numPosOperands) const {
  MojoInflightDiag diag = initDiag();
  emitTooManyPositional(diag, maxAllowedArgs, numPosOperands, "argument");
  return diag;
}

MojoInflightDiag DiagEmitter::byPosAndKw(ArrayRef<StringAttr> names) const {
  MojoInflightDiag diag = initDiag();
  emitByPosAndKw(diag, names, "argument");
  return diag;
}

MojoInflightDiag DiagEmitter::badImplicitConversion(ASTType fromType,
                                                    ASTType toType) const {
  MojoInflightDiag diag = initDiag();
  diag << "cannot implicitly convert";
  if (fromType)
    diag << " " << fromType;

  // Add target type to diag
  if (toType)
    diag << " to " << toType << ": add an explicit cast";
  return diag;
}

//===----------------------------------------------------------------------===//
// OverloadFitness
//===----------------------------------------------------------------------===//

/// Calculate the minimum required and maximum allowed number of positional
/// operands for a signature, assuming that the signature has a variadic pack;
static std::optional<std::pair<size_t, size_t>>
calculateRequiredPosOperandsForPacks(FnTypeGeneratorType signature) {
  // This function heavily assumes that a signature has at most
  // one pack variadic argument and that variadics are always the last
  // positional args.
  size_t numPosArgs = countNumPositional(signature.getArgListAttrs());

  // We don't require any positional operands (because this function does not
  // check for passing kinds).
  if (!numPosArgs)
    return std::make_pair(0, numPosArgs);

  // If we have a variadic argument, it will consume all positional operands,
  // but it does not require any.
  size_t lastPosIdx = numPosArgs - 1;
  if (signature.isPosVarArg(lastPosIdx))
    return std::make_pair(0, std::numeric_limits<size_t>::max());

  // If we have a non-empty variadic pack argument, we do require a certain
  // number of positional operands (since the value of positional packs cannot
  // be provided by keyword operands).
  // NOTE: in this case, it doesn't matter if there are preceding positional
  // arguments with default values: the pack cannot have a default value and
  // _must_ be provided positional operands explicitly, and therefore the
  // preceding defaults won't be used anyway.
  if (ASTType variadicPackType = signature.getIfVariadicPack(lastPosIdx)) {
    VariadicAttr packed = // See if resolved.
        sugarDynCast<VariadicAttr>(variadicPackType.getVariadicPackTypeList());

    // The caller should know the concrete type list unless we binded the pack
    // directly as a parameter.  This is an unpack like situation.
    // TODO: This happens in error cases and needs to be re-evaluated.
    if (!packed)
      return std::nullopt;

    // NOTE: we adjust the number of user declared pos args since that
    // includes the pack itself (hence the "-1").
    size_t packSize = packed.getValues().size();
    return std::make_pair(numPosArgs - 1 + packSize, numPosArgs - 1 + packSize);
  }

  return std::make_pair(0, numPosArgs);
}

/// Check the expected type against the provided operand. This identifies any
/// problems with the operand type.
std::optional<MojoInflightDiag> OverloadFitness::checkOneOperand(
    ASTExprAnd<AnyValue> operand, size_t argIdx,
    ArgConvention expectedConvention, ASTType expectedType,
    bool allowImplicitConversions, const OverloadSet &callable,
    PogListAttr argListAttr) {

  auto loc = operand.expr->getLoc();

  ASTType expectedRVType =
      RefType::stripRefConvention(expectedType, expectedConvention);

  /// This creates and returns an instance of a diagnostic.
  auto getDiag = [&]() -> MojoInflightDiag {
    return callable.getShared().emitError(loc);
  };

  auto emitWrongTypeDiag = [&](ASTType expectedType) -> MojoInflightDiag {
    auto diag = getDiag();
    ::emitWrongTypeDiag(diag, operand, expectedType, argIdx, argListAttr,
                        callable.syntax, callable.getShared());
    return diag;
  };

  // Allow overloading on "owned" vs "by-ref" arguments.
  // If the argument convention is owned but the operand is not an RValue then
  // we'll need to copy the value (or this is entirely invalid).  If the
  // argument convention is borrowed/ref but the value is an RValue then we have
  // an RValue decay.  Model these so that APIs can overload on owned vs
  // borrowed effectively.
  bool argTypesMatchOrIsUValue = !operand.ir.getIfCValue();
  if (!argTypesMatchOrIsUValue)
    argTypesMatchOrIsUValue =
        operand.ir.getIfCValue().getRValueType().isEqualCanon(expectedRVType);

  if (argTypesMatchOrIsUValue) {
    if (operand.ir.getIfBValue() || operand.ir.getIfLValue()) {
      // Heavily penalize implicit copies.
      if (expectedConvention == ArgConvention::OwnedMem ||
          expectedConvention == ArgConvention::DeinitMem)
        payload.numMismatchedConventions += 2;
    } else {
      assert((operand.ir.getIfUValue() || operand.ir.getIfRValue()) &&
             "UValue and RValue expressions are always owned");
      // Slightly penalize RValue->ref conversions.
      if (expectedConvention != ArgConvention::OwnedMem &&
          expectedConvention != ArgConvention::DeinitMem)
        ++payload.numMismatchedConventions;
    }
  }

  ASTDecl &declScope = callable.paramBindings.declScope;
  SharedState &shared = declScope.getShared();
  switch (expectedConvention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError: {
    // The actual value must be an lvalue if callee takes things by-ref.
    auto argVal = operand.ir.getIfLValue();
    if (!argVal) {
      auto diag = getDiag();
      if ((callable.syntax == CallSyntax::kMethodCall ||
           callable.syntax == CallSyntax::kMethodCallSynthetic) &&
          argIdx == 0) {
        diag << "invalid use of mutating method on rvalue of type ";
        if (ASTType type = operand.ir.getRValueTypeIfResolvable())
          diag << type;
        else
          printUValueTypeInfo(operand.ir, diag);
      } else {
        diag << "value passed to mutable argument "
             << argListAttr.getPogs()[argIdx].getName() << " must be mutable";
      }
      diag << operand.expr->getRange();
      return std::move(diag);
    }

    // If this is a wildcard type, we can match any operand.
    if (sugarIsa<NameLookupArgWildcardType>(argVal.getRValueType()))
      return {}; // Success.

    // ByRef argument types must exactly match, no conversions are allowed.
    if (!argVal.getRValueType().isEqualCanon(expectedRVType)) {
      auto diag = getDiag();
      diag << "l-value of type " << operand.ir.getIfLValue().getRValueType()
           << " cannot be converted to reference of type "
           << expectedType.getReferenceElementType()
           << operand.expr->getRange();
      return std::move(diag);
    }
    // Notice if a register-passable type is being passed in-memory. This allows
    // 'mut' arguments overloads to be more expensive than borrowed.
    payload.numMismatchedConventions +=
        expectedRVType.isRegisterPassable(loc, shared);
    return {}; // Success.
  }
  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    // If we are binding to something that is already a reference, check for
    // compatibility of the references and we're done.
    if (operand.ir.isMValue()) {
      RefType valueRefType = operand.ir.getMValueType();
      if (IREmitter::canZeroCostConvert(valueRefType, expectedType, shared))
        return {}; // Success.
      // Otherwise this is the wrong type for the argument.
      return emitWrongTypeDiag(expectedType);
    }

    // Otherwise, we are binding something like a PValue or SRValue to a
    // reference argument, which doesn't have a origin.  This is a problem
    // because origins can be propagated through the type system of the
    // function call to other arguments and they all need to line up.  We
    // handle this in two phases: during overload resolution we bind this to
    // an immortal origin, and then after the candidate is selected, we
    // re-emit these arguments to memory and re-infer all the parameters.
    //
    // One detail is how we do this: we bind these arguments to immutable
    // temporaries, because we specifically do NOT want 'ref' arguments with
    // parametric mutability to treat these things as mutable.
    if (sugarCast<RefType>(expectedType).isMutableKnown(true)) {
      auto diag = getDiag();
      diag << "mutable reference argument "
           << argListAttr.getPogs()[argIdx].getName()
           << "cannot bind to temporary value";
      return diag;
    }

    // Remember that this argument needs to be emitted.
    argsNeedingOrigins.resize(argIdx + 1);
    argsNeedingOrigins[argIdx] = true;

    // Handle this like a normal memory argument, since the value can undergo
    // implicit conversions etc.
    [[fallthrough]];
  }
  case ArgConvention::ReadMem:
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
    // If a register-passable type is being passed in-memory, remember this.
    payload.numMismatchedConventions +=
        expectedRVType.isRegisterPassable(loc, shared);
    [[fallthrough]];
  case ArgConvention::ReadReg:
    break;
  }

  // Get the argument if it has a concrete type.
  CValue argVal = operand.ir.getIfCValue();

  // If the argument is unresolved, see if we can resolve it with the expected
  // type.
  if (!argVal) {
    if (auto initValue = operand.ir.getIfInitializer()) {
      IREmitter emitter(declScope, ExprContext::EC_CallArgValue);
      CallOperands operands =
          initValue->getOperandsForInferredType(expectedRVType, emitter);

      // Initializer lists are good if we can construct the expected type.
      FailureOr<PValue> initFn = OverloadSet::canConstructType(
          expectedRVType, std::move(operands), declScope);
      // If there were declaration errors, assume construction is possible
      // to avoid spurious errors.
      bool valid = (bool)failed(initFn) || initFn.value();
      // If so, all is good, if not, we fail.
      if (valid)
        return {}; // Success.
      return emitWrongTypeDiag(expectedRVType);
    }

    auto orValue = operand.ir.getIfOverloadSet();
    assert(orValue && "Unknown UValue!");

    // Try to refine the OverloadSetUValue into a PValue.
    argVal = orValue->getDirectSymbol(expectedRVType, declScope);
    if (!argVal)
      return emitWrongTypeDiag(expectedRVType);

    // If we have a reference to an overloaded method like foo(a.method),
    // then we can't resolve it.
    // TODO(partial application => closures): Given we just resolved argVal,
    // we could form the "a.method" expression with a closure.
    if (orValue->baseValue) // Cannot merge base value.
      return emitWrongTypeDiag(expectedRVType);
  }

  ASTType argType = argVal.getRValueType();

  // If this is a wildcard type, we can match any operand.
  if (sugarIsa<NameLookupArgWildcardType>(argType))
    return {}; // Success.

  // Otherwise, we pass as an r-value.  If the argument types match, then
  // they are good.
  if (argType.isEqualCanon(expectedRVType))
    return {}; // Success.

  // Argument name mismatches don't count as implicit conversions.
  if (IREmitter::canZeroCostConvert(argType, expectedRVType, shared))
    return {}; // Success.

  if (auto nonmaterializableTarget =
          argType.getNonmaterializableTarget(shared)) {
    if (nonmaterializableTarget.isEqualCanon(expectedRVType)) {
      // Implicit conversion for nonmaterializable types to their target
      // type is allowed even if !allowImplicitConversions and count as half
      // as much of a mismatch as a normal implicit conversion.  This enables
      // exact matches to be more specific, and literals to be more compatible
      // than an actual conversion.
      ++payload.numImplicitConversions;
      return {}; // Success.
    }
  }

  // If implicit conversions are possible and one will work, then we succeed
  // with that conversion.
  if (allowImplicitConversions &&
      IREmitter::canImplicitlyConvertToType({argVal, operand.expr},
                                            expectedRVType, declScope)) {
    // If we had one, this bumps our # implicit conversions.
    payload.numImplicitConversions += 2;
    return {}; // Success.
  }

  // Otherwise this is the wrong type for the argument.
  return emitWrongTypeDiag(expectedRVType);
}

bool OverloadFitness::isBetter(const OverloadFitness &other) const {
  // Neither operand should be invalid or have param constraint issues
  // (they should be filtered out before comparison).
  assert(getValidity() >= Validity::kFunctionConstraintInconclusive &&
         other.getValidity() >= Validity::kFunctionConstraintInconclusive);

  // We first compare the number of implicit conversions.
  size_t numConversions = getNumImplicitConversions();
  size_t otherNumConversions = other.getNumImplicitConversions();
  if (numConversions != otherNumConversions)
    return numConversions < otherNumConversions;

  // If ambiguous, we compare the boolean metrics.
  int8_t mask = payload.getBoolMask();
  int8_t otherMask = other.payload.getBoolMask();
  if (mask != otherMask)
    return mask < otherMask;

  // Otherwise these candidates are almost identical, so we try to decide based
  // on the number of input conventions mismatches (e.g. register-passable
  // passed in memory).
  if (payload.numMismatchedConventions !=
      other.payload.numMismatchedConventions)
    return payload.numMismatchedConventions <
           other.payload.numMismatchedConventions;

  // If still ambiguous, we compare the number of bindings. This allows us to
  // treat a function with more parameters as worse than a parameter-less
  // function, so things like:
  //    fn foo(a: Int):   and  fn foo[T: AnyType](a: T):
  // resolve to the more specific function.
  return paramBindings.size() < other.paramBindings.size();
}

int8_t OverloadFitness::Payload::getBoolMask() const {
  // We consider exact matches of concrete types to be more specific than
  // those needing non-materializable conversions, both of these more
  // specific than varargs matches (for example, when overloading a
  // `foo(Int)` and `foo(Int*)` we should pick the former if both work), and
  // all of these more specific than matches with variadic parameters.
  return 2 * passesVarArgArgument + 1 * hasVariadicParams;
}

OverloadFitness OverloadFitness::evaluate(ASTDecl *candidate,
                                          const OverloadSet &callable,
                                          PValue selfPValue) {
  DeclResolver &resolver = *callable.getShared().declResolver;
  auto func = cast_or_null<FnOp>(candidate->getIfOperation());
  FnTypeGeneratorType signature = func.getFullSignature();

  if (selfPValue) {
    // TODO(MOCO-1259): Support static methods with associated aliases
    auto parentDecl = candidate->getParentDecl();
    if (dyn_cast_or_null<TraitDeclOp>(parentDecl->getIfOperation())) {
      signature = substituteTraitAliasesIntoSignature(
          resolver, *parentDecl, func, signature, selfPValue);
    }
  }

  auto [bindings, fitness, diag] =
      callable.paramBindings.verifyBindingsWithDiag(
          signature.getInputParamTypes(), signature.getMetadata(), candidate,
          /*partial=*/true);
  if (!bindings) {
    // If no diagnostics were emitted, this must be an inconclusive fitness.
    if (!diag) {
      assert(!fitness.unprovableConstraints.empty());
      return std::move(fitness.unprovableConstraints);
    }
    // Otherwise, it's a real failure, so we return an invalid fitness.
    return std::move(*diag);
  }

  return OverloadFitness(bindings);
}

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `callOperands`.
///
/// The 'funcIfDirect' member is set if this is a direct call, or null if
/// indirect.  It can be used to tune diagnostics.
OverloadFitness OverloadFitness::evaluate(FnTypeGeneratorType signature,
                                          ASTDecl *funcIfDirect,
                                          const OverloadSet &callable,
                                          const CallOperands &operands,
                                          bool allowImplicitConversions) {
  // We set up diagnostics.
  size_t numPosOperands = operands.getNumPositional();
  size_t numOperands = operands.size();

  SMLoc callLoc = callable.getExpr()->getLoc();
  SharedState &shared = callable.getShared();
  DiagEmitter emitDiagFor(shared, callLoc);

  if (!operands.empty()) {
    if (auto selfCValue = operands[0].ir.getIfCValue()) {
      if (auto selfPValue = PValue(selfCValue.getRValueType().mlirType)) {
        // TODO(MOCO-1259): Support static methods with associated aliases
        if (funcIfDirect) {
          if (auto func =
                  dyn_cast_or_null<FnOp>(funcIfDirect->getIfOperation())) {
            auto parentDecl = funcIfDirect->getParentDecl();
            if (dyn_cast_or_null<TraitDeclOp>(parentDecl->getIfOperation())) {
              signature = substituteTraitAliasesIntoSignature(
                  *shared.declResolver, *parentDecl, func, signature,
                  selfPValue);
            }
          }
        }
      }
    }
  }

  // If a variadic keyword arg is expected, we collect the unknown kw operands.
  PogListAttr argListAttr = signature.getArgListAttrs();
  OperandValueList variadicKwOperands;
  auto [kwDiagRes, kwDiagNames] =
      operands.diagnoseKeywordOperands(argListAttr, variadicKwOperands);
  switch (kwDiagRes) {
  case CallOperands::KwDiagResult::kMissingKwOnly:
    return emitDiagFor.missingArgs(kwDiagNames, "keyword-only");
  case CallOperands::KwDiagResult::kOutOfOrderInferredKw:
    llvm_unreachable("no inferred arguments");
  case CallOperands::KwDiagResult::kPosOnlyPassedByKw:
    return emitDiagFor.posOnlyPassedByKw(kwDiagNames);
  case CallOperands::KwDiagResult::kUnknownKeywords:
    return emitDiagFor.unexpectedKwArgs(kwDiagNames);
  default:
    break;
  }

  auto [posDiagRes, posDiagNames] = operands.diagnosePosOperands(argListAttr);
  switch (posDiagRes) {
  case CallOperands::PosDiagResult::kMissingPos:
    return emitDiagFor.missingArgs(posDiagNames, "positional");
  case CallOperands::PosDiagResult::kTooManyPos: {
    size_t numPosMaximum = countNumPositional(argListAttr);
    return emitDiagFor.tooManyPosArgs(numPosMaximum, numPosOperands);
  }
  case CallOperands::PosDiagResult::kByPosAndKw:
    return emitDiagFor.byPosAndKw(posDiagNames);
  default:
    break;
  }

  // Check that the signature can be rebound with this set of bindings.

  // Determine if this is an initializer that returns Self, which can be used
  // for inferring parameters on Self.
  bool returnsSelf = false;
  bool hasCTADParams = false;
  if (funcIfDirect) {
    auto fn = cast<FnOp>(funcIfDirect->getIfOperation());
    returnsSelf = fn.getSpecialFunctionInfo().hasSelfResult();
    hasCTADParams = !fn.getIsStatic() && isa<StructDeclOp>(fn->getParentOp());
  }

  std::optional<MojoInflightDiag> diag;
  auto getDiag = [&](std::optional<SMLoc> loc) -> MojoInflightDiag & {
    diag = shared.emitError(loc ? *loc : callLoc);
    return *diag;
  };

  ParamInf inference(callable.paramBindings, signature.getInputParamTypes(),
                     signature.getParamListAttrs(), allowImplicitConversions,
                     getDiag, funcIfDirect);

  // TODO: inferForCall will eventually be separated. We will eventually blend
  // parameter inference into overload resolution have something like:
  //
  //  inference.inferFromParamBinding(...)
  //
  //  for (arg in callexpr) {
  //    case call_conv:
  //       inference.inferOneOperand(...)
  //    ...
  //  }
  if (failed(inference.inferForCall(signature, operands, variadicKwOperands,
                                    returnsSelf, hasCTADParams))) {
    // FIXME: Unfortunately, `inferFailed` does not imply that there is an
    // failure reported in the `inferenceDiags` (understand why!). Besides, we
    // still have to call `verifyBindings` because mojo diagnostic will be
    // filled in `ParamBinding`. This is WRONG and need to be fixed after fully
    // migration. For now, simply erase every inferred value if inference failed
    // to make it behave the same as before (we probably need to move error
    // diagnose into ParamInfState or as a separate util?).
    if (diag)
      return std::move(*diag);
    if (!inference.unprovableConstraints.empty())
      return std::move(inference.unprovableConstraints);

    for (size_t i = 0, e = signature.getInputParamTypes().size(); i < e; i++)
      inference.evaluator.overwriteIndexBinding(i, nullptr);
  }

  auto [newBindings, bindingFitness] =
      callable.paramBindings.verifyBindings(signature, inference);

  // If there is an error, we just forward the diagnostics.
  if (diag)
    return std::move(*diag);
  if (!bindingFitness.unprovableConstraints.empty())
    return std::move(bindingFitness.unprovableConstraints);
  assert(newBindings && "expected new bindings when no diagnostic was emitted");

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  FnTypeGeneratorType originalSignature = signature;
  std::tie(signature, newBindings) = getUnboundSpecializedSignature(
      signature, newBindings, &shared.getEvaluationContext());

  // This is the result we will return if we succeed.
  OverloadFitness result(newBindings);

  // Check that the result didn't bind to a type that would require changing to
  // a different result convention.
  for (Type outputType : signature.getResults())
    if (!ASTType(outputType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.resultGenericMemType(outputType);

  // Binding the parameters would determine the type of pack varargs. Given
  // this, we need to check again if we have missing or too many arguments.

  std::optional<std::pair<size_t, size_t>> posNumBoundOr =
      calculateRequiredPosOperandsForPacks(signature);
  // This means that we can not determine a concrete number of packed
  // arguments, this is always an error.
  if (!posNumBoundOr)
    return emitDiagFor.unresolvedPackCount(numPosOperands);

  auto [minPosOperands, maxPosOperands] = *posNumBoundOr;
  if (numPosOperands < minPosOperands || maxPosOperands < numPosOperands) {
    return emitDiagFor.wrongArgCountWithPack(minPosOperands, maxPosOperands,
                                             numPosOperands);
  }

  // As we walk through the values provided as part of the argument list, we
  // match them up against arguments expected by the signature of the callee,
  // take note if variadic arguments are passed, and accumulate implicit
  // conversions required for a match.  This value indicates the next operand
  // to consider as a positional argument.
  size_t posOperandIdx = 0;

  argListAttr = signature.getArgListAttrs();
  DefaultValueHandler defaultHandler(argListAttr);
  // FiXME: This should be substituting implicit parameters as the arguments are
  // bound to be able to handle things like
  // https://github.com/modular/modular/issues/3855 correctly.
  for (auto [expectedArgIdx, expectedTypeX, expectedConvention] :
       llvm::enumerate(signature.getArguments(),
                       signature.getArgConventions())) {
    Type expectedType = expectedTypeX;
    // Ignore the return slot if present.
    if (expectedConvention == ArgConvention::ByRefError)
      continue;
    if (expectedConvention == ArgConvention::ByRefResult) {
      result.payload.numMismatchedConventions +=
          ASTType(expectedType)
              .getReferenceElementType()
              .isRegisterPassable(callLoc, shared);
      continue;
    }

    if (signature.isKwVarArg(expectedArgIdx)) {
      expectedType = ASTType(expectedType).getKwargsDictRefValueType();
      auto refExpType = RefType::getAnyOrigin(expectedType, /*isMut=*/true);
      for (auto operand : variadicKwOperands) {
        // TODO: Passing OwnedMem is a hack that is needed because the value
        // type is not a reference type (and doesn't have a origin), but we
        // still want to type check it. So, passing it as if it was reg-passable
        // happens to just work, until we rectify this. Right now the reason the
        // value type cannot be a reference type is because `Reference` does not
        // (and in fact cannot) conform to `Copyable & Movable`.
        if (auto diag = result.checkOneOperand(
                operand, expectedArgIdx, ArgConvention::OwnedMem, refExpType,
                allowImplicitConversions, callable, argListAttr))
          return std::move(diag).value();
      }
      // This comes after all the positionals.
      posOperandIdx = numOperands;
      continue;
    }

    // If the arguments or results got bound to a memory-only type then their
    // argument convention needs to change.  We cannot support this until we get
    // proper type traits.
    // TODO: Don't let memory types bind to __TypeOfAllTypes.
    if (!ASTType(expectedType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.argGenericMemType(expectedArgIdx, expectedType,
                                           argListAttr);

    // Figure out the next positional argument to process.
    while (posOperandIdx < numOperands && operands[posOperandIdx].keyword)
      ++posOperandIdx;

    // Handle case when there are no more provided positional operands.
    StringAttr argName = argListAttr.getName(expectedArgIdx);
    if (posOperandIdx == numOperands) {
      // If the argument is a varargs argument list or pack, then it can be
      // initialized with zero values no problem.
      if (signature.isPosVarArg(expectedArgIdx) ||
          signature.isPack(expectedArgIdx)) {
        // We consider an empty varargs list to be an implicit conversion,
        // so an exact signature match takes precedence.
        ++result.payload.numImplicitConversions;
        continue;
      }

      // Check if the argument was passed as a keyword operand.
      if (const OperandValue *kwOperandOr = operands.findKwArg(argName)) {
        // If we found a keyword argument, we check it normally.
        if (auto diag = result.checkOneOperand(
                *kwOperandOr, expectedArgIdx, expectedConvention, expectedType,
                allowImplicitConversions, callable, argListAttr))
          return std::move(diag).value();
        continue;
      }

      // We ensured earlier that that can be no missing positional arguments.
      assert(defaultHandler.getDefault(expectedArgIdx) &&
             "missing positional argument not caught by diagnostics");

      continue;
    }

    /// Check and process a single positional operand and advance the operand
    /// index.
    auto processPositionalOperand =
        [&](ASTType expectedType,
            ArgConvention conv) -> std::optional<MojoInflightDiag> {
      if (auto diag = result.checkOneOperand(
              operands[posOperandIdx], expectedArgIdx, conv, expectedType,
              allowImplicitConversions, callable, argListAttr))
        return std::move(diag).value();
      ++posOperandIdx;
      return {};
    };

    // If we have a varargs argument, then it will eat the rest of the
    // positional arguments, but we have to check each of them.
    if (signature.isPosVarArg(expectedArgIdx)) {
      auto expectedVariadic = sugarCast<VariadicType>(expectedType);
      auto varArgsEltType = expectedVariadic.getElementType();
      while (posOperandIdx != numPosOperands) {
        if (auto result = processPositionalOperand(
                varArgsEltType,
                signature.getPosVarArgConvention(expectedArgIdx)))
          return std::move(*result);
        result.payload.passesVarArgArgument = true;
      }
      continue;
    }

    // If we have a pack type, it must have a known number of elements, and so
    // consume exactly that many positional operands.
    if (ASTType variadicPackType =
            signature.getIfVariadicPack(expectedArgIdx)) {
      auto actualArgConvention =
          signature.getPackVarArgConvention(expectedArgIdx);
      RefPackType packType = variadicPackType.getVariadicPackInfo(shared);
      for (TypedAttr element : packType.getVariadicIfResolved().getValues()) {
        auto refType = packType.getElementRefTypeFor(ASTType(element).mlirType);
        if (auto result =
                processPositionalOperand(refType, actualArgConvention))
          return std::move(*result);
        result.payload.passesVarArgArgument = true;
      }
      continue;
    }

    // Otherwise, we have an ordinary positional argument that is not varargs or
    // a pack. We ensured earlier that it is not also passed as a keyword
    // operand, so we process it as usual.
    assert(
        (argListAttr.getPassingKind(expectedArgIdx) == PassingKind::PosOnly ||
         (!argName.empty() && !operands.findKwArg(argName))) &&
        "redundant argument not caught by diagnostics");
    if (auto result =
            processPositionalOperand(expectedType, expectedConvention))
      return std::move(*result);
  }

  assert(posOperandIdx == numOperands &&
         "should handle argument mismatch above");

  // Fail if this is a constructor call that returns the wrong result type. This
  // can happen with weird things like this:
  //     struct A[X: Int]: fn __init__(out self: A[4]): pass
  //     var a = A[1]()  # Infers to A[4]; error!
  //     var b = A[4]()  # Ok!
  if (callable.syntax == CallSyntax::kTypeCall) {
    // Check to see if any of the parameter bound to the result type disagree
    // with the 'Self' parameters, which are bound into newBindings.
    auto resultType = ASTType(signature.getUserResultType());
    auto numBindings = resultType.getParamBindings().size();
    for (auto [paramIdx, actual, expected] :
         llvm::enumerate(resultType.getParamBindings(),
                         newBindings.getValue().take_front(numBindings))) {
      if (!isEqualCanon(actual, expected)) {
        DeclResolver::DeclScopeChanger x(funcIfDirect);
        auto declOp =
            cast<StructDeclOp>(resultType.getDecl(shared)->getIfOperation());
        auto paramType = declOp.getSignature().getInputParamTypes()[paramIdx];
        MojoInflightDiag resultTypeDiag = shared.emitError(callLoc);
        resultTypeDiag << "return type " << resultType << " parameter "
                       << ParamDeclRefAttr::get(
                              declOp.getParams()[paramIdx].getName(), paramType)
                       << " has value " << actual
                       << " that doesn't match expected " << expected;
        return std::move(resultTypeDiag);
      }
    }
  }

  // Fail if this is an implicit conversion but the ctor is not marked @implicit
  if (funcIfDirect && callable.syntax == CallSyntax::kImplicitConvert &&
      !cast<FnOp>(funcIfDirect->getIfOperation()).isImplicitConversion()) {
    ASTType fromType = operands[0].ir.getRValueTypeIfResolvable();
    return emitDiagFor.badImplicitConversion(fromType,
                                             signature.getUserResultType());
  }

  // Check that all fn constraints are satisfied.
  SmallVector<ConstraintAttr> fnUnprovableConstraints;
  checkConstraints(callable.paramBindings.declScope,
                   originalSignature.getMetadata(),
                   signature.getFnMetadata().getConstraints(),
                   originalSignature.getFnMetadata().getConstraints(), getDiag,
                   &fnUnprovableConstraints,
                   /*evaluator=*/nullptr);
  if (diag)
    return std::move(*diag);
  if (!fnUnprovableConstraints.empty())
    result.unprovableConstraints = std::move(fnUnprovableConstraints);

  // Otherwise we succeeded!
  return result;
}
