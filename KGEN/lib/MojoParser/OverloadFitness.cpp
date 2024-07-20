//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the components for overload fitness evaluation.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/OverloadFitness.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParameterInference.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "OperandDiagnostics.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"

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
  DiagEmitter(SharedState &shared, SMLoc callLoc, size_t numOperands,
              CallSyntax callSyntax)
      : SharedStateUser(shared), callLoc(callLoc), numOperands(numOperands),
        callSyntax(callSyntax) {}

  InflightDiag unexpectedKwArgs(ArrayRef<StringAttr> unknownKwOperands) const;
  InflightDiag wrongParamType(ASTExprAnd<AnyValue> actualBinding,
                              size_t paramIdx, ASTType expectedType) const;
  InflightDiag wrongParamCount(size_t expectedNumParams,
                               size_t actualNumParams) const;
  InflightDiag wrongArgCountWithPack(size_t minRequiredArgs,
                                     size_t maxAllowedArgs,
                                     size_t numOperands) const;
  InflightDiag wrongPosOnlyCount(size_t minRequiredArgs, size_t numOperands,
                                 const Twine &argOrParam) const;
  InflightDiag resultGenericMemType(Type outputType) const;
  InflightDiag argGenericMemType(size_t expectedArgIdx,
                                 Type expectedType) const;
  InflightDiag argTypeMismatch(OverloadFitness::ArgTypeMismatchKind kind,
                               ASTType ty, ASTExprAnd<AnyValue> operand,
                               size_t argIdx) const;
  InflightDiag missingArgs(ArrayRef<StringAttr> missingArgs,
                           const Twine &kindStr) const;
  InflightDiag posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const;
  InflightDiag tooManyPosArgs(size_t maxAllowedArgs,
                              size_t numPosOperands) const;
  InflightDiag byPosAndKw(ArrayRef<StringAttr> names) const;

private:
  SMLoc callLoc;
  size_t numOperands;
  CallSyntax callSyntax;

  /// Wrapper around pretty printing logic for an argument given by index.
  void describeArgumentNo(InflightDiag &diag, size_t argIdx) const;

  InflightDiag initDiag() const { return shared.emitError(callLoc); }
};
} // namespace

void DiagEmitter::describeArgumentNo(InflightDiag &diag, size_t argIdx) const {
  // If this is a method syntax call, don't count the receiver.
  if (callSyntax == CallSyntax::kMethodCall ||
      callSyntax == CallSyntax::kMethodCallSynthetic) {
    // It is probably possible for this assert to fire, if it does we should
    // tailor the error message.
    if (argIdx != 0)
      diag << "method argument #" << (argIdx - 1);
    else
      diag << "self argument";
  } else if (callSyntax == CallSyntax::kOperator && argIdx == 1) {
    diag << "right side";
  } else if (callSyntax == CallSyntax::kReversedOperator && argIdx == 0) {
    diag << "left side";
  } else if (callSyntax == CallSyntax::kSubscript && argIdx != 0) {
    if (argIdx == 1 && numOperands == 2)
      diag << "index";
    else
      diag << "index #" << (argIdx - 1);
  } else if (callSyntax == CallSyntax::kAttribute && argIdx != 0) {
    diag << "attribute name";
  } else {
    diag << "argument #" << argIdx;
  }
}

InflightDiag
DiagEmitter::unexpectedKwArgs(ArrayRef<StringAttr> unknownKwOperands) const {
  InflightDiag diag = initDiag();
  emitUnknownKeywords(diag, unknownKwOperands, "argument");
  return diag;
}

InflightDiag DiagEmitter::wrongParamType(ASTExprAnd<AnyValue> actualBinding,
                                         size_t paramIdx,
                                         ASTType expectedType) const {
  return initDiag() << "callee parameter #" << paramIdx << " has "
                    << ASTType(expectedType) << " type, but value has type "
                    << actualBinding.ir.getIfPValue().getType()
                    << actualBinding.expr->getRange();
}

InflightDiag DiagEmitter::wrongParamCount(size_t expectedNumParams,
                                          size_t actualNumParams) const {
  InflightDiag diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, /*minRequired=*/expectedNumParams,
                           /*maxAllowed=*/expectedNumParams, actualNumParams,
                           "parameter");
  return diag;
}

InflightDiag DiagEmitter::wrongArgCountWithPack(size_t minRequiredArgs,
                                                size_t maxAllowedArgs,
                                                size_t numOperands) const {
  InflightDiag diag = initDiag()
                      << "callee with non-empty variadic pack argument";
  emitWrongArgOrParamCount(diag, minRequiredArgs, maxAllowedArgs, numOperands,
                           "positional operand");
  return diag;
}

InflightDiag DiagEmitter::wrongPosOnlyCount(size_t minRequiredArgs,
                                            size_t numOperands,
                                            const Twine &argOrParam) const {
  InflightDiag diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, minRequiredArgs,
                           /*maxAllowed=*/numOperands, numOperands,
                           "positional " + argOrParam);
  return diag;
}

InflightDiag DiagEmitter::resultGenericMemType(Type outputType) const {
  return initDiag()
         << "result cannot bind AnyTrivialRegType type to memory-only type "
         << outputType;
}

InflightDiag DiagEmitter::argGenericMemType(size_t expectedArgIdx,
                                            Type expectedType) const {
  InflightDiag diag = initDiag();
  describeArgumentNo(diag, expectedArgIdx);
  return std::move(diag)
         << " cannot bind AnyTrivialRegType type to memory-only type "
         << expectedType;
}

/// Attach extra type conversion error detail or hints to the user when
/// reporting an error passing `operand` to an argument of type `argType`.
static void addTypeConversionDetail(InflightDiag &diag,
                                    ASTExprAnd<AnyValue> operand,
                                    ASTType argType) {
  auto loc = operand.expr->getLoc();
  ASTType operandType = operand.ir.getRValueTypeIfResolvable();
  if (!operandType) {
    diag.attachNote(loc) << "try resolving the overloaded function first";
    return;
  }
  // Try to detect mismatched inout result type.
  auto lhsSig = dyn_cast<SignatureType>(operandType);
  auto rhsSig = dyn_cast<SignatureType>(argType);
  if (lhsSig && rhsSig) {
    auto getByRefResult = [](SignatureType sig) -> std::pair<bool, Type> {
      return {sig.hasMemoryOnlyResult(),
              ASTType(sig).getSignatureUserResultType()};
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
static void diagnoseFailedRefTypeConversion(InflightDiag &diag,
                                            ASTExprAnd<AnyValue> operand,
                                            RefType argType,
                                            SharedState &shared) {
  diag << "'Reference[" << ASTType(argType.getElementType()) << ", ...]";

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
  } else if (!canConvertWithRebind(operandRefTy.getLifetimeType(),
                                   argType.getLifetimeType(), shared)) {
    diag.attachNote(loc) << "operand mutability " << operandRefTy.isMutable()
                         << " doesn't match expected mutability "
                         << argType.isMutable();
  } else if (!canConvertWithRebind(operandRefTy, argType, shared)) {
    diag.attachNote(loc) << "operand lifetime " << operandRefTy.getLifetime()
                         << " doesn't match expected lifetime "
                         << argType.getLifetime();
  }
}

static void printRValueTypeInfo(const AnyValue &value, InflightDiag &diag) {
  if (ASTType type = value.getRValueTypeIfResolvable())
    diag << type;
  else if (value.getIfInitializer())
    diag << "initializer list";
  else
    diag << "unknown overload";
}

InflightDiag
DiagEmitter::argTypeMismatch(OverloadFitness::ArgTypeMismatchKind kind,
                             ASTType ty, ASTExprAnd<AnyValue> operand,
                             size_t argIdx) const {
  using ArgTypeMismatchKind = OverloadFitness::ArgTypeMismatchKind;
  InflightDiag diag = initDiag();
  switch (kind) {
  case ArgTypeMismatchKind::kNotLValue:
    if ((callSyntax == CallSyntax::kMethodCall ||
         callSyntax == CallSyntax::kMethodCallSynthetic) &&
        argIdx == 0) {
      diag << "invalid use of mutating method on rvalue of type ";
      printRValueTypeInfo(operand.ir, diag);
    } else {
      describeArgumentNo(diag, argIdx);
      diag << " must be mutable in order to pass to a mutating argument";
    }
    diag << operand.expr->getRange();
    return diag;
  case ArgTypeMismatchKind::kWrongLVType:
    return std::move(diag) << "l-value of type "
                           << operand.ir.getIfLValue().getRValueType()
                           << " cannot be converted to reference of type "
                           << ty.getReferenceElementType()
                           << operand.expr->getRange();
  case ArgTypeMismatchKind::kWrongType: {
    // Special case implicit conversions with a custom message.
    if (callSyntax == CallSyntax::kImplicitConvert) {
      printRValueTypeInfo(operand.ir, diag);
      diag << " value to " << ty;
      return diag;
    }

    describeArgumentNo(diag, argIdx);
    diag << " cannot be converted from " << operand.expr->getRange();
    ASTType rValueType = operand.ir.getRValueTypeIfResolvable();
    bool isConvertingTypeValue = ty.getMetaType() == rValueType;
    if (rValueType) {
      if (isConvertingTypeValue)
        diag << "type value " << ty;
      else
        diag << rValueType;
    } else if (operand.ir.getIfInitializer()) {
      diag << "initializer list";
    } else {
      diag << "unknown overload";
    }
    diag << " to ";

    if (auto refType = dyn_cast<RefType>(ty)) {
      diagnoseFailedRefTypeConversion(diag, operand, refType, shared);
      return diag;
    }

    diag << (isConvertingTypeValue ? "an instance of " : "") << ty;
    if (isConvertingTypeValue)
      diag << "; did you mean to instantiate " << ty << "?";
    addTypeConversionDetail(diag, operand, ty);
    return diag;
  }
  default:
    llvm_unreachable("unexpected ArgTypeMismatchKind");
  }
}

InflightDiag DiagEmitter::missingArgs(ArrayRef<StringAttr> missingArgs,
                                      const Twine &kindStr) const {
  InflightDiag diag = initDiag();
  emitMissing(diag, missingArgs, kindStr + " argument");
  return diag;
}

InflightDiag
DiagEmitter::posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const {
  InflightDiag diag = initDiag();
  emitPosOnlyPassedByKw(diag, posOnlyPassedByKw, "argument");
  return diag;
}

InflightDiag DiagEmitter::tooManyPosArgs(size_t maxAllowedArgs,
                                         size_t numPosOperands) const {
  InflightDiag diag = initDiag();
  emitTooManyPositional(diag, maxAllowedArgs, numPosOperands, "argument");
  return diag;
}

InflightDiag DiagEmitter::byPosAndKw(ArrayRef<StringAttr> names) const {
  InflightDiag diag = initDiag();
  emitByPosAndKw(diag, names, "argument");
  return diag;
}

//===----------------------------------------------------------------------===//
// OverloadFitness
//===----------------------------------------------------------------------===//

/// Calculate the minimum required and maximum allowed number of positional
/// operands for a signature, assuming that the signature has a variadic pack;
static std::pair<size_t, size_t>
calculateRequiredPosOperandsForPacks(LITSignatureType signature) {
  // This function heavily assumes that a signature has at most
  // one pack variadic argument and that variadics are always the last
  // positional args.
  size_t numPosArgs = countNumPositional(signature.getArgListAttrs());

  // We don't require any positional operands (because this function does not
  // check for passing kinds).
  if (!numPosArgs)
    return {0, numPosArgs};

  // If we have a variadic argument, it will consume all positional operands,
  // but it does not require any.
  size_t lastPosIdx = numPosArgs - 1;
  if (signature.isPosVarArg(lastPosIdx))
    return {0, std::numeric_limits<size_t>::max()};

  // If we have a non-empty variadic pack argument, we do require a certain
  // number of positional operands (since the value of positional packs cannot
  // be provided by keyword operands).
  // NOTE: in this case, it doesn't matter if there are preceding positional
  // arguments with default values: the pack cannot have a default value and
  // _must_ be provided positional operands explicitly, and therefore the
  // preceding defaults won't be used anyway.
  if (ASTType variadicPackType = signature.getIfVariadicPack(lastPosIdx)) {
    RefPackType packType = variadicPackType.getVariadicPackInfo();
    VariadicAttr packed = packType.getVariadicIfResolved();
    // The caller should know the concrete type list unless we binded the pack
    // directly as a parameter.  This is an unpack like situation.
    // TODO: This happens in error cases and needs to be re-evaluated.
    if (!packed)
      return {0, numPosArgs - 1};

    // NOTE: we adjust the number of user declared pos args since that
    // includes the pack itself (hence the "-1").
    size_t packSize = packed.getValues().size();
    return {numPosArgs - 1 + packSize, numPosArgs - 1 + packSize};
  }

  return {0, numPosArgs};
}

/// Check the expected type against the provided operand. This identifies any
/// problems with the operand type and also returns the type to be used for
/// error propagation.
///
/// This ties into parameter inference, but is only called on the top level
/// function operands being matched up, not anything in recursive functiontype
/// positions.
std::pair<OverloadFitness::ArgTypeMismatchKind, ASTType>
OverloadFitness::checkOneOperand(ASTExprAnd<AnyValue> operand,
                                 ArgConvention expectedConvention,
                                 ASTType expectedType,
                                 size_t &numImplicitConversions,
                                 size_t &numMismatchedConventions,
                                 bool allowImplicitConversions, SMLoc loc,
                                 const TypeCheckScopeInfo &scopeInfo) {
  SharedState &shared = scopeInfo.shared;
  switch (expectedConvention) {
  case ArgConvention::InitSelf:
    // If this is an UnknownAttr, then it is a placeholder for 'self' which will
    // ultimately be an lvalue of the indicated type.
    if (auto pValue = operand.ir.getIfPValue())
      if (isa<UnknownAttr>(pValue.get())) {
        // FIXME: This should be modeled as an LValue to merge into the normal
        // logic.  We don't have LValue's that are PValue's though.
        ASTType expElementType = expectedType.getReferenceElementType();
        ASTType argElementType =
            pValue.getRValueType().getReferenceElementType();

        // This is valid if the types obviously match or if the arg type has
        // unbound parameters that are inferred.
        if (argElementType.isEqualAllowingUnknownAttr(expElementType,
                                                      scopeInfo.shared))
          return {kValidType, expectedType};
        return {kWrongType, argElementType};
      }
    [[fallthrough]];
  case ArgConvention::InOut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError: {
    // The actual value must be an lvalue if callee takes things by-ref.
    auto argVal = operand.ir.getIfLValue();
    if (!argVal)
      return {kNotLValue, expectedType};

    // ByRef argument types must exactly match, no conversions are allowed.
    ASTType elementType = expectedType.getReferenceElementType();
    if (!argVal.getRValueType().isEqualCanon(elementType))
      return {kWrongLVType, expectedType};
    // Notice if a register-passable type is being passed in-memory. This allows
    // 'inout' arguments overloads to be more expensive than borrowed.
    numMismatchedConventions += elementType.isRegisterPassable(loc, shared);
    return {kValidType, expectedType};
  }
  case ArgConvention::Ref: {
    // Element type and address have to match and the mutability has to be
    // compatible.
    RefType valueRefType;
    if (operand.ir.isMValue())
      valueRefType = cast<RefType>(operand.ir.getMValueReference().getType());
    else if (auto pv = operand.ir.getIfPValue(); pv && scopeInfo.isParamContext)
      valueRefType = RefType::getImmortal(pv.getType(), /*isMut=*/true);

    // As a special hack, look through DefArgumentWrapperDLValue to the
    // underlying MBValue that it may contain.  This is for two reasons:
    //  1) We don't want to infer mutability from the argument even though
    //     it is a DLValue, because we'd force copy-out + writeback,
    //     materializing the def argument box.
    //  2) We have significant bugs around lifetime inference from SBValues
    //     and DLValues because we're not materializing the box in time.  This
    //     is tracked by MOCO-684.
    // Solve this by hacking this important case specifically.
    if (auto dlValue = operand.ir.getIfDLValue())
      if (auto refType = dlValue->getMBValueTypeFromDefArgument())
        valueRefType = refType;

    // The argument must be an MValue in the case of a dynamic call.
    if (valueRefType &&
        canConvertWithRebind(valueRefType, expectedType, shared))
      return {kValidType, expectedType};

    // Otherwise this is the wrong type for the argument.
    return {kWrongType, expectedType};
  }

    [[fallthrough]];
  case ArgConvention::BorrowedInMem:
  case ArgConvention::OwnedInMem:
    // Ignore the pointer type on memory conventions when matching types.
    // Note: We do not support overloading on borrow/owned currently,
    // but we could add this if there is a reason to.
    expectedType = expectedType.getReferenceElementType();
    // If a register-passable type is being passed in-memory, remember this.
    numMismatchedConventions += expectedType.isRegisterPassable(loc, shared);
    [[fallthrough]];
  case ArgConvention::BorrowedInReg:
  case ArgConvention::OwnedInReg: {
    // Get the argument if it has a concrete type.
    CValue argVal = operand.ir.getIfCValue();

    // If the argument is unresolved, see if we can resolve it with the expected
    // type.
    if (!argVal) {
      if (auto initValue = operand.ir.getIfInitializer()) {
        // Initializer lists are good if we can construct the expected type.
        FailureOr<PValue> initFn = OverloadSet::canConstructType(
            expectedType, initValue.get(), operand.expr, scopeInfo);
        // If there were declaration errors, assume construction is possible to
        // avoid spurious errors.
        bool valid = (bool)failed(initFn) || initFn.value();
        // If so, all is good, if not, we fail.
        return {valid ? kValidType : kWrongType, expectedType};
      }

      auto orValue = operand.ir.getIfOverloadSet();
      assert(orValue && "Unknown UValue!");

      // Try to refine the OverloadSetUValue into a PValue.
      argVal = orValue->getDirectSymbol(expectedType);
      if (!argVal)
        return {kWrongType, expectedType};

      // If we have a reference to an overloaded method like foo(a.method),
      // then we can't resolve it.
      // TODO(partial application => closures): Given we just resolved argVal,
      // we could form the "a.method" expression with a closure.
      if (orValue->baseValue) // Cannot merge base value.
        return {kWrongType, expectedType};
    }

    ASTType argType = argVal.getRValueType();
    // Otherwise, we pass as an r-value.  If the argument types match, then
    // they are good.
    if (argType.isEqualCanon(expectedType))
      return {kValidType, expectedType};

    if (auto nonmaterializableTarget =
            argType.getNonmaterializableTarget(shared)) {
      if (nonmaterializableTarget.isEqualCanon(expectedType)) {
        // Implicit conversion for nonmaterializable types to their target
        // type is allowed even if !allowImplicitConversions and count as half
        // as much of a mismatch as a normal implicit conversion.  This enables
        // exact matches to be more specific, and literals to be more compatible
        // than an actual conversion.
        ++numImplicitConversions;
        return {kValidType, expectedType};
      }
    }

    // Argument name mismatches don't count as implicit conversions.
    if (canConvertWithRebind(argType, expectedType, shared))
      return {kValidType, expectedType};

    // If implicit conversions are possible and one will work, then we succeed
    // with that conversion.
    if (allowImplicitConversions &&
        OverloadSet::canImplicitlyConvertToType({argVal, operand.expr},
                                                expectedType, scopeInfo)) {
      // If we had one, this bumps our # implicit conversions.
      numImplicitConversions += 2;
      return {kValidType, expectedType};
    }

    // Otherwise this is the wrong type for the argument.
    return {kWrongType, expectedType};
  }
  }

  llvm_unreachable("unknown case");
}

bool OverloadFitness::isBetter(const OverloadFitness &other) const {
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

  // If still ambiguous, we compare the number of bindings.
  if (paramBindings.size() != other.paramBindings.size())
    return paramBindings.size() < other.paramBindings.size();

  // Otherwise these candidates are almost identical, so we try to decide based
  // on the number of input conventions mismatches (e.g. register-passable
  // passed in memory).
  return payload.numMismatchedConventions <
         other.payload.numMismatchedConventions;
}

int8_t OverloadFitness::Payload::getBoolMask() const {
  // We consider exact matches of concrete types to be more specific than
  // those needing non-materializable conversions, both of these more
  // specific than varargs matches (for example, when overloading a
  // `foo(Int)` and `foo(Int*)` we should pick the former if both work), and
  // all of these more specific than matches with variadic parameters.
  return 2 * passesVarArgArgument + 1 * hasVariadicParams;
}

OverloadFitness OverloadFitness::evaluate(ArrayRef<Type> paramTypes,
                                          PogListAttr paramListAttr,
                                          const OverloadSet &callable,
                                          bool allowImplicitConversions) {
  auto [bindings, fitness, diag] = callable.paramBindings.verifyBindings(
      paramTypes, paramListAttr, callable.baseName, callable.expr->getLoc(),
      /*opLoc=*/{}, /*partial=*/true);
  if (!bindings)
    return std::move(*diag);
  return {bindings, Payload{fitness.numImplicitConversions, 0, 0,
                            fitness.hasVariadicParams}};
}

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `callOperands`.
///
/// The 'funcIfDirect' member is set if this is a direct call, or null if
/// indirect.  It can be used to tune diagnostics.
OverloadFitness OverloadFitness::evaluate(LITSignatureType signature,
                                          ASTDecl *funcIfDirect,
                                          const OverloadSet &callable,
                                          const OperandContainer &callOperands,
                                          bool allowImplicitConversions) {
  // We set up diagnostics.
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  size_t numPosOperands = posOperands.size();

  size_t numOperands = numPosOperands + callOperands.getNumKwOperands();
  SMLoc callLoc = callable.expr->getLoc();
  SharedState &shared = callable.getShared();
  DiagEmitter emitDiagFor(shared, callLoc, numOperands, callable.syntax);

  // If a variadic keyword arg is expected, we collect the unknown kw operands.
  KeywordOperandContainer variadicKwOperands;
  auto [kwDiagRes, kwDiagNames] = diagnoseKeywordOperands(
      signature.getArgListAttrs(), variadicKwOperands, callOperands);
  switch (kwDiagRes) {
  case KwDiagResult::kMissingKwOnly:
    return emitDiagFor.missingArgs(kwDiagNames, "keyword-only");
  case KwDiagResult::kPosOnlyPassedByKw:
    return emitDiagFor.posOnlyPassedByKw(kwDiagNames);
  case KwDiagResult::kUnknownKeywords:
    return emitDiagFor.unexpectedKwArgs(kwDiagNames);
  default:
    break;
  }

  PogListAttr argListAttr = signature.getArgListAttrs();
  auto [posDiagRes, posDiagNames] =
      diagnosePosOperands(argListAttr, callOperands);
  switch (posDiagRes) {
  case PosDiagResult::kMissingPos:
    return emitDiagFor.missingArgs(posDiagNames, "positional");
  case PosDiagResult::kTooManyPos: {
    size_t numPosMaximum = countNumPositional(argListAttr);
    return emitDiagFor.tooManyPosArgs(numPosMaximum, numPosOperands);
  }
  case PosDiagResult::kByPosAndKw:
    return emitDiagFor.byPosAndKw(posDiagNames);
  default:
    break;
  }

  // Check that the signature can be rebound with this set of bindings. We use
  // diagnostic handlers to capture any issues.
  InflightDiag diag = shared.emitError(callLoc);
  ParameterInferenceDiagnostics inferenceDiags;
  PogListAttr paramListAttr = signature.getParamListAttrs();
  ParamBindings::DiagEmitter bindingDiag{
      /*emitParamCount=*/
      [&](size_t numActual, bool posOnly) {
        if (posOnly) {
          size_t numPosOnly = countNumPosOnly(paramListAttr);
          diag =
              emitDiagFor.wrongPosOnlyCount(numPosOnly, numActual, "parameter");
        } else {
          // Hide the implicit trait parameter from the diagnostic.
          size_t hidden = 0;
          if (funcIfDirect &&
              isa<TraitDeclOp>(cast<LIT::FuncOp>(*funcIfDirect)->getParentOp()))
            hidden = 1;
          size_t numExpected = signature.getNumParams() - hidden -
                               countNumImplicitKinds(paramListAttr) -
                               countNumInferredKinds(paramListAttr);
          diag = emitDiagFor.wrongParamCount(numExpected, numActual - hidden);
        }
        // For each of the missing parameters, attach any parameter inference
        // diagnostics.
        inferenceDiags.attach(paramListAttr, diag, numActual);
      },
      /*emitPosType=*/
      [&](size_t paramIdx, ASTExprAnd<AnyValue> binding, ASTType expectedType) {
        diag = emitDiagFor.wrongParamType(binding, paramIdx, expectedType);
      },
      /*emitKwType=*/
      [&](StringAttr paramName, ASTExprAnd<AnyValue> binding,
          ASTType expectedType) {
        diag << "callee parameter " << paramName << " has " << expectedType
             << " type, but value has type "
             << binding.ir.getIfPValue().getType() << binding.expr->getRange();
      },
      /*emitUnknownKeywords=*/
      [&](ArrayRef<StringAttr> unknownKeywords) {
        emitUnknownKeywords(diag, unknownKeywords, "parameter");
      },
      /*emitRedundantKeywords=*/
      [&](ArrayRef<StringAttr> names) {
        emitByPosAndKw(diag, names, "parameter");
      },
      /*emitPosOnlyPassedByKw=*/
      [&](ArrayRef<StringAttr> names) {
        emitPosOnlyPassedByKw(diag, names, "parameter");
      },
      /*emitDeductionFailure=*/
      [&](size_t paramIdx) {
        auto emitMessage = [&](auto sig) {
          diag << "could not deduce ";
          if (StringAttr name = sig.getParamName(paramIdx); !name.empty())
            diag << "parameter " << name;
          else
            diag << nameForPosOnly(paramIdx, "parameter");
        };

        // If this is a method on a struct and we couldn't infer something from
        // its self parameters, complain about the struct.
        if (funcIfDirect) {
          if (auto structOp = dyn_cast<StructDeclOp>(
                  cast<LIT::FuncOp>(*funcIfDirect)->getParentOp())) {
            auto structSig = structOp.getSignature();
            if (paramIdx < structSig.getNumParams()) {
              emitMessage(structSig);
              diag << " of parent struct '" << structOp.getDeclName().getValue()
                   << "'";
              diag.attachNote(structOp.getLoc()) << " struct declared here";
              inferenceDiags.attach(paramListAttr, diag);
              return;
            }
          }
        }
        emitMessage(signature);
        diag << " of callee '" << callable.baseName << "'";
        inferenceDiags.attach(paramListAttr, diag);
      },
      /*emitUnboundPackInVariadic=*/
      [&](ASTExprAnd<AnyValue> binding) {
        diag << "unbound pack syntax (i.e. `*_`) cannot be used where variadic "
                "parameters are expected"
             << binding.expr->getRange();
      },
      /*emitUnboundPackNotEnd=*/
      [&](ASTExprAnd<AnyValue> binding) {
        diag << "unbound pack must be at the end of the parameter list"
             << binding.expr->getRange();
      },
      /*emitInferOnlyFailure=*/
      [&](size_t paramIdx) {
        if (signature.getParamListAttrs().getPassingKind(paramIdx) ==
            PassingKind::Inferred) {
          diag << "failed to infer parameter ";
          printNameOrIdx(signature.getParamName(paramIdx), paramIdx, diag);
          inferenceDiags.attach(paramListAttr, diag);
          return;
        }

        // Find the parameter number and potentially name of the type of the
        // argument that failed to be inferred.
        mlir::AttrTypeWalker walker;
        size_t idx;
        walker.addWalk([&](StructType type) {
          for (auto [i, value] : llvm::enumerate(type.getParamValues())) {
            if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value);
                indexRef && !indexRef.getDepth() &&
                indexRef.getIndex() == paramIdx) {
              diag << "failed to infer implicit parameter ";
              auto structDecl =
                  cast<StructDeclOp>(ASTType(type).getDecl(shared));
              printNameOrIdx(structDecl.getSignature().getParamName(i), i,
                             diag);
              diag << " of argument ";
              printNameOrIdx(signature.getArgName(idx), idx, diag);
              diag << " type '" << structDecl.getSymName() << "'";
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
        for (auto [i, argType] : llvm::enumerate(signature.getArguments())) {
          idx = i;
          if (walker.walk(argType).wasInterrupted())
            break;
        }
        inferenceDiags.attach(paramListAttr, diag);
      },
      /*emitMissing=*/
      [&](ArrayRef<StringAttr> names, const Twine &kindStr) {
        emitMissing(diag, names, kindStr + " parameter");
      },
      /*emitTooManyPositional=*/
      [&](size_t numMaxAllowed, size_t numActual) {
        emitTooManyPositional(diag, numMaxAllowed, numActual, "parameter");
      },
  };

  auto parameterInferenceHook = [&](ArrayRef<TypedAttr> bindingsSoFar,
                                    const ParserParamEvaluator &evaluator) {
    ParameterInferenceState inferrence(
        callable.paramBindings, callable.paramBindings.getPosBindings(),
        &callable.paramBindings.getKWBindings(), bindingsSoFar, evaluator,
        inferenceDiags, allowImplicitConversions);

    // Infer information from this signature holistically.
    if (failed(inferrence.infer(signature, callOperands, variadicKwOperands)))
      return PValue();

    // See if we inferred information about the next value.
    if (auto result = inferrence.getInferredValue(bindingsSoFar.size()))
      return PValue(result);

    // Check to see if this is a CTAD parameter - a parameter on the struct
    // that encloses the method.  Consider "conditional conformance" cases like:
    //     struct X[A: AnyType]:
    //       fn foo[B: Movable](self: X[B]): ...
    // When resolving a function call like `someX.foo()`, we install the
    // bindings for 'A' from the typeof(someX) when resolving the
    // AttributeRefExpr and then infer 'B' from someX again.
    //
    // However, when we have something like `X.foo(someX)` we cannot install the
    // bindings for 'A' at AttributeRef resolution time, and 'someX' is only
    // bound by parameter inference to 'B'.  Notice this and infer the parameter
    // directly from A.  This is also important for operator resolution, which
    // works effectively the same way.
    //
    // TODO: Provide a first class representation for conditional conformance
    // that doesn't have us shadowing parameters like this!
    if (funcIfDirect) {
      auto func = cast<LIT::FuncOp>(*funcIfDirect);
      if (!func.getIsStatic() && isa<StructDeclOp>(func->getParentOp())) {
        if (failed(inferrence.inferCTADParams(signature, callOperands)))
          return PValue();
        if (auto result = inferrence.getInferredValue(bindingsSoFar.size()))
          return PValue(result);
      }
    }

    // If we succeeded inference but didn't get a value for this parameter, then
    // the parameter must not be present: complain.
    inferenceDiags.addFailure(bindingsSoFar.size(), callable.expr,
                              InferenceFailure::NotFoundFailure());
    return PValue();
  };
  auto [newBindings, bindingFitness] = callable.paramBindings.verifyBindings(
      signature, &bindingDiag, parameterInferenceHook, /*partial=*/false);

  // If there is an error, we just forward the diagnostics.
  if (!newBindings)
    return std::move(diag);
  diag.abandon();

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  std::tie(signature, newBindings) =
      getUnboundSpecializedSignature(signature, newBindings);

  // Check that the result didn't bind to a type that would require changing to
  // a different result convention.
  for (Type outputType : signature.getResults())
    if (!ASTType(outputType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.resultGenericMemType(outputType);

  // Binding the parameters would determine the type of pack varargs. Given
  // this, we need to check again if we have missing or too many arguments.
  auto [minPosOperands, maxPosOperands] =
      calculateRequiredPosOperandsForPacks(signature);
  if (numPosOperands < minPosOperands || maxPosOperands < numPosOperands) {
    return emitDiagFor.wrongArgCountWithPack(minPosOperands, maxPosOperands,
                                             numPosOperands);
  }

  SMLoc loc = callable.expr->getLoc();

  // We will accumulate the implicit conversion in arguments to those counted
  // for the parameter bindings.
  size_t numImplicitConversions = bindingFitness.numImplicitConversions;

  // As we walk through the values provided as part of the argument list, we
  // match them up against arguments expected by the signature of the callee,
  // take note if variadic arguments are passed, and accumulate implicit
  // conversions required for a match.
  size_t posOperandIdx = 0;
  bool passesVarArgArgument = false;

  // For each mismatch in "preferred" argument convention, penalize the
  // overload. This is to resolve ambiguities that can arise from synthesized
  // thunks for converting calling conventions.
  size_t numMismatchedConventions = 0;

  auto checkAnOperand = [&](ASTExprAnd<AnyValue> operand,
                            ArgConvention expectedConvention,
                            ASTType expectedType) {
    return checkOneOperand(operand, expectedConvention, expectedType,
                           numImplicitConversions, numMismatchedConventions,
                           allowImplicitConversions, loc,
                           callable.paramBindings);
  };

  // Use a ParserParamEvaluator to substitute 'apply' expressions in the
  // argument types.
  ParserParamEvaluator evaluator(*shared.declResolver);
  argListAttr = signature.getArgListAttrs();
  DefaultValueHandler defaultHandler(argListAttr);
  for (auto [expectedArgIdx, unboundExpectedType, expectedConvention] :
       llvm::enumerate(signature.getArguments(),
                       signature.getArgConventions())) {
    // Ignore the return slot if present.
    Type expectedType = evaluator.refine(unboundExpectedType);
    if (expectedConvention == ArgConvention::ByRefError)
      continue;
    if (expectedConvention == ArgConvention::ByRefResult) {
      numMismatchedConventions += ASTType(expectedType)
                                      .getReferenceElementType()
                                      .isRegisterPassable(loc, shared);
      continue;
    }

    if (signature.isKwVarArg(expectedArgIdx)) {
      expectedType = ASTType(expectedType).getKwargsDictRefValueType();

      for (auto [name, operand] : variadicKwOperands) {
        // TODO: Passing OwnedInReg is a hack that is needed because the value
        // type is not a reference type (and doesn't have a lifetime), but we
        // still want to type check it. So, passing it as if it was reg-passable
        // happens to just work, until we rectify this. Right now the reason the
        // value type cannot be a reference type is because `Reference` does not
        // (and in fact cannot) conform to `CollectionElement`.
        auto [kind, ty] =
            checkAnOperand(operand, ArgConvention::OwnedInReg, expectedType);
        if (kind != kValidType)
          return emitDiagFor.argTypeMismatch(kind, ty, operand, expectedArgIdx);
      }
      continue;
    }

    // If the arguments or results got bound to a memory-only type then their
    // argument convention needs to change.  We cannot support this until we get
    // proper type traits.
    // TODO: Don't let memory types bind to AnyTrivialRegType.
    if (!ASTType(expectedType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.argGenericMemType(expectedArgIdx, expectedType);

    // Handle case when there are no more provided positional operands.
    StringAttr argName = argListAttr.getName(expectedArgIdx);
    if (posOperandIdx == numPosOperands) {
      // If the argument is a varargs argument list or pack, then it can be
      // initialized with zero values no problem.
      if (signature.isPosVarArg(expectedArgIdx) ||
          signature.isPackVarArg(expectedArgIdx)) {
        // We consider an empty varargs list to be an implicit conversion,
        // so an exact signature match takes precedence.
        ++numImplicitConversions;
        continue;
      }

      // Check if the argument was passed as a keyword operand.
      if (std::optional<ASTExprAnd<AnyValue>> kwOperandOr =
              callOperands.findKwArg(argName)) {
        // If we found a keyword argument, we check it normally.
        auto [kind, ty] =
            checkAnOperand(*kwOperandOr, expectedConvention, expectedType);
        if (kind != kValidType) {
          return emitDiagFor.argTypeMismatch(kind, ty, *kwOperandOr,
                                             expectedArgIdx);
        }
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
            ArgConvention conv) -> std::optional<InflightDiag> {
      ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx];
      auto [kind, ty] = checkAnOperand(operand, conv, expectedType);
      if (kind != kValidType)
        return emitDiagFor.argTypeMismatch(kind, ty, operand, posOperandIdx);
      ++posOperandIdx;
      return std::nullopt;
    };

    // If we have a varargs argument, then it will eat the rest of the
    // positional arguments, but we have to check each of them.
    if (signature.isPosVarArg(expectedArgIdx)) {
      auto expectedVariadic = cast<VariadicType>(expectedType);
      auto varArgsEltType = expectedVariadic.getElementType();
      while (posOperandIdx != numPosOperands) {
        if (auto result = processPositionalOperand(
                varArgsEltType, expectedVariadic.getConvention()))
          return std::move(*result);
        passesVarArgArgument = true;
      }
      continue;
    }

    // If we have a pack type, it must have a known number of elements, and so
    // consume exactly that many positional operands.
    if (ASTType variadicPackType =
            signature.getIfVariadicPack(expectedArgIdx)) {
      auto actualArgConvention =
          signature.getPackVarArgConvention(expectedArgIdx);
      RefPackType packType = variadicPackType.getVariadicPackInfo();
      for (TypedAttr element : packType.getVariadicIfResolved().getValues()) {
        auto refType = packType.getElementRefTypeFor(ASTType(element).mlirType);
        if (auto result =
                processPositionalOperand(refType, actualArgConvention))
          return std::move(*result);
        passesVarArgArgument = true;
      }
      continue;
    }

    // Otherwise, we have an ordinary positional argument that is not varargs or
    // a pack. We ensured earlier that it is not also passed as a keyword
    // operand, so we process it as usual.
    assert(
        (argListAttr.getPassingKind(expectedArgIdx) == PassingKind::PosOnly ||
         (!argName.empty() && !callOperands.findKwArg(argName))) &&
        "redundant argument not caught by diagnostics");
    if (auto result =
            processPositionalOperand(expectedType, expectedConvention))
      return std::move(*result);
  }

  assert(posOperandIdx == numPosOperands &&
         "should handle argument mismatch above");

  // Otherwise we succeeded!
  return {newBindings,
          Payload{numImplicitConversions, numMismatchedConventions,
                  passesVarArgArgument, bindingFitness.hasVariadicParams}};
}
