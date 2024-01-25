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
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "MojoUtils.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "LITEXPRCALLS"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ParameterInferenceState Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParameterInferenceState {
public:
  ParameterInferenceState(SharedState &shared, size_t index,
                          ParserParamEvaluator &evaluator)
      : shared(shared), parameterIndex(index), evaluator(evaluator) {}

  /// Given an incomplete parameter binding set for a call to the specified
  /// signature, try to infer the value of the next 'decl' parameter.  This
  /// should always return null /without/ an error if it cannot be inferred, and
  /// return a specific value if unambiguously determined.
  PValue infer(LITSignatureType signature, ArrayRef<TypedAttr> bindingsSoFar,
               const CallOperands &callOperands);

private:
  void matchTypes(Type actualType, Type expectedType);
  void matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);
  LogicalResult checkOneOperand(AnyValue value, ASTType expectedType,
                                ValueInputConvention expectedConvention);
  LogicalResult checkOneOperand(ASTExprAnd<AnyValue> operand,
                                ASTType expectedType,
                                ValueInputConvention expectedConvention);

  SharedState &shared;
  size_t parameterIndex;
  ParserParamEvaluator &evaluator;
  SmallVector<PValue> inferredValues;
};
} // namespace

void ParameterInferenceState::matchTypes(Type actualType, Type expectedType) {
  // If the expected type is a parameter ref, then we're binding the specified
  // type to an attribute parameter.
  if (auto expectedParamRef = dyn_cast<ParamRefType>(expectedType)) {
    ASTType type = actualType;
    if (ASTType nmTarget = type.getNonmaterializableTarget(shared))
      type = nmTarget;
    if (Type metatype = type.getMetaType()) {
      matchParams(TypeConstantAttr::get(type, metatype),
                  expectedParamRef.getParam());
    } else {
      // Otherwise, this is an MLIR type.
      matchParams(TypeConstantAttr::get(actualType,
                                        TypeType::get(actualType.getContext())),
                  expectedParamRef.getParam());
    }
    return;
  }

  // If the types trivially match then there is no inference to do.
  if (actualType == expectedType)
    return;

  // Handle when both are DeclRefTypes.
  if (auto actualDRT = dyn_cast<DeclRefType>(actualType)) {
    if (auto expectedDRT = dyn_cast<DeclRefType>(expectedType)) {
      // Ignore if these are two fundamentally different symbols.
      if (actualDRT.getSymbol() != expectedDRT.getSymbol())
        return;

      // Fail if the parameter lists fundamentally mismatch.
      // TODO: Defaulted parameters could make this ok?
      if (actualDRT.getParamValues().size() !=
          expectedDRT.getParamValues().size())
        return;

      // Match up the parameter bindings.
      for (auto [actual, expected] :
           llvm::zip(actualDRT.getParamValues(), expectedDRT.getParamValues()))
        matchParams(actual, expected);
      return;
    }
  }

  // Handle various common POP types for convenience, starting with SIMDType.
  if (auto actual = dyn_cast<POP::SIMDType>(actualType))
    if (auto expected = dyn_cast<POP::SIMDType>(expectedType)) {
      matchParams(actual.getSize(), expected.getSize());
      matchParams(actual.getDType(), expected.getDType());
      return;
    }

  // POP::ArrayType.
  if (auto actual = dyn_cast<POP::ArrayType>(actualType))
    if (auto expected = dyn_cast<POP::ArrayType>(expectedType)) {
      matchParams(actual.getSize(), expected.getSize());
      matchTypes(actual.getElementType(), expected.getElementType());
      return;
    }

  // Handle RefType.
  if (auto actual = dyn_cast<RefType>(actualType))
    if (auto expected = dyn_cast<RefType>(expectedType)) {
      matchTypes(actual.getElementType(), expected.getElementType());
      matchParams(actual.getLifetime(), expected.getLifetime());
      matchParams(actual.getAddressSpace(), expected.getAddressSpace());
      return;
    }

  // Handle LifetimeType.
  if (auto actual = dyn_cast<LifetimeType>(actualType))
    if (auto expected = dyn_cast<LifetimeType>(expectedType)) {
      matchParams(actual.isMutable(), expected.isMutable());
      return;
    }

  // Handle PointerType.
  if (auto actual = dyn_cast<PointerType>(actualType))
    if (auto expected = dyn_cast<PointerType>(expectedType))
      return matchTypes(actual.getElementType(), expected.getElementType());

  // Handle VariadicType.
  if (auto actual = dyn_cast<VariadicType>(actualType))
    if (auto expected = dyn_cast<VariadicType>(expectedType))
      return matchTypes(actual.getElementType(), expected.getElementType());

  // Handle PackType.
  if (auto actual = dyn_cast<PackType>(actualType))
    if (auto expected = dyn_cast<PackType>(expectedType))
      return matchParams(actual.getVariadic(), expected.getVariadic());

  // TODO: Could do StructType?
  LLVM_DEBUG(llvm::errs() << "CANNOT INFER MISMATCH TYPES:\n";
             actualType.dump(); expectedType.dump();
             llvm::errs() << parameterIndex);
}

void ParameterInferenceState::matchParams(TypedAttr actualAttr,
                                          TypedAttr expectedAttr) {

  // We can only match up these values if their types match.
  if (actualAttr.getType() != expectedAttr.getType())
    matchTypes(actualAttr.getType(), expectedAttr.getType());

  auto actualOp = dyn_cast<ParamOperatorAttr>(actualAttr);
  auto expectedOp = dyn_cast<ParamOperatorAttr>(expectedAttr);
  if (actualOp && expectedOp &&
      actualOp.getOpcode() == expectedOp.getOpcode() &&
      actualOp.getNumOperands() == expectedOp.getNumOperands()) {
    for (auto [a, b] :
         llvm::zip(actualOp.getOperands(), expectedOp.getOperands()))
      matchParams(a, b);
  }

  // If the expected value is the parameter declaration in question, remember
  // this value!
  if (auto ire = dyn_cast<ParamIndexRefAttr>(expectedAttr)) {
    if (ire.getDepth() == 0 && !ire.getIsResult() &&
        ire.getIndex() == parameterIndex) {
      Type expectedType = expectedAttr.getType();
      if (actualAttr.getType() == expectedType) {
        // Microoptimization: first just check the common case of the types
        // matching exactly, so that we don't always need to rebound.
        inferredValues.push_back(actualAttr);
        return;
      }
      // Otherwise, compare the rebound types to handle dependent types.
      expectedType = evaluator.getReboundType(expectedType);
      if (actualAttr.getType() == expectedType) {
        inferredValues.push_back(actualAttr);
        return;
      }
      // Otherwise, attempt an implicit conversion between the inferred type and
      // the expected type.
      ExprEmitter emitter(shared, shared.getTopLevelDecl(), EC_TypeParamValue);
      SyntheticNode node(shared.getTopLevelDecl().getLoc());
      if (emitter.canImplicitlyConvertToType({actualAttr, node},
                                             expectedType)) {
        PValue result = emitter.emitPValue({actualAttr, node},
                                           EC_TypeParamValue, expectedType);
        if (result)
          inferredValues.push_back(result);
      }
    }
    return;
  }

  // If the attrs trivial match then we're done and there is no inference to do.
  if (actualAttr == expectedAttr)
    return;

  LLVM_DEBUG(llvm::errs() << "CANNOT INFER MISMATCHING ATTRS:\n";
             actualAttr.dump(); expectedAttr.dump();
             llvm::errs() << parameterIndex << "\n");
}

LogicalResult ParameterInferenceState::checkOneOperand(
    AnyValue value, ASTType expectedType,
    ValueInputConvention expectedConvention) {
  // We'll bind the next provided value.
  switch (expectedConvention) {
  case ValueInputConvention::InitSelf:
    // If this is an UnknownAttr, then it is a placeholder for type
    // checking, just let it pass.
    if (PValue pValue = value.getIfPValue())
      if (isa<UnknownAttr>(pValue.get()))
        return success();
    [[fallthrough]];
  case ValueInputConvention::ByRef:
  case ValueInputConvention::ByRefResult: {
    // The actual value must be an lvalue if callee takes things by-ref.
    LValue argVal = value.getIfLValue();
    if (!argVal)
      return failure();

    // By-ref argument types must exactly match, no conversions are allowed.
    matchTypes(argVal.getRValueType(), expectedType.getReferenceElementType());
    return success();
  }

  case ValueInputConvention::OwnedInMem:
  case ValueInputConvention::BorrowedInMem:
    // Otherwise,we expect an r-value to match up, ignoring the reference type
    // from the convention.
    expectedType = expectedType.getReferenceElementType();
    [[fallthrough]];
  case ValueInputConvention::OwnedInReg:
  case ValueInputConvention::BorrowedInReg: {
    Type actualType;
    // TODO: Consider implicit conversions?
    if (CValue cValue = value.getIfCValue())
      actualType = cValue.getRValueType();
    else if (ORValue orValue = value.getIfORValue())
      if (PValue pValue = orValue->getIfPValue())
        actualType = pValue.getType();

    if (!actualType)
      return success();

    // If the argument is an explicit low-level reference type passed as a
    // borrowed register value, then we allow matching it to its underlying
    // element type.
    if (auto expectedRef = dyn_cast<RefType>(expectedType.mlirType)) {
      if (expectedConvention == ValueInputConvention::BorrowedInReg &&
          !isa<RefType>(actualType)) {
        // Infer element, addrspace, and lifetime.
        if (value.isMValue()) {
          auto valueRefType =
              cast<RefType>(value.getMValueReference().getType());
          // If the MValue is an MBValue specifically, make sure to strip off
          // any mutability from the reference.  There are lots of things that
          // are mutable that have borrowed references.
          if (value.getIfMBValue() && !valueRefType.isMutableKnown(false))
            valueRefType = valueRefType.getWithMutability(false);

          matchTypes(valueRefType, expectedRef);
        } else {
          // In the case of a SValue / PValue, we can do an MBValue conversion
          // to expose the address, but we can't infer a lifetime or address
          // space.
          matchTypes(actualType, expectedRef.getElementType());
        }
        return success();
      }
    }

    // Otherwise, we pass as an r-value if we know the type.
    matchTypes(actualType, expectedType);
    return success();
  }
  case ValueInputConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }
  llvm_unreachable("invalid value input convention");
};

LogicalResult ParameterInferenceState::checkOneOperand(
    ASTExprAnd<AnyValue> operand, ASTType expectedType,
    ValueInputConvention expectedConvention) {
  return checkOneOperand(operand.ir, expectedType, expectedConvention);
}

PValue ParameterInferenceState::infer(LITSignatureType signature,
                                      ArrayRef<TypedAttr> bindingsSoFar,
                                      const CallOperands &callOperands) {
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  size_t numPosOperands = posOperands.size();

  // TODO: Apply the bindings so far (plus a distinct new attribute relating
  // back to the original decls for ones that are missing) to the signature with
  // getSpecializedSignature so we benefit from the already-fixed substitutions
  // being applied to the input types.  This can make them more concrete and
  // help with inferring dependent types based on already-bound parameters.

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t posOperandIdx = 0;
  for (auto [expectedArgIdx, expectedType, expectedConvention, argName] :
       llvm::enumerate(signature.getValueInputs(),
                       signature.getInputConventions(),
                       signature.getArgNames())) {

    // There is no provided operand for a by-ref result.
    if (expectedConvention == ValueInputConvention::ByRefResult)
      continue;

    // Handle case when there are no more provided positional operands.
    if (posOperandIdx == numPosOperands) {
      // If the argument is a varargs argument list, then it can be initialized
      // with zero values no problem.
      if (signature.isVarArg(expectedArgIdx) ||
          signature.isPackVarArg(expectedArgIdx))
        break;

      // Check if a keyword operand was provided for this argument
      if (std::optional<ASTExprAnd<AnyValue>> kwOperandOr =
              callOperands.findKwArg(argName)) {
        if (failed(checkOneOperand(*kwOperandOr, expectedType,
                                   expectedConvention)))
          return {};
        continue;
      }

      // If available, we check the default argument value.
      // NOTE: The type of the default argument has to match the argument type,
      // meaning there can't be anything to infer here directly, but we still
      // check to make sure that the default value doesn't contradict already
      // inferred parameters.
      ArrayRef<TypedAttr> defaultArgs = signature.getDefaultPosArgs();
      size_t numArgs = signature.getNumInputs();
      if (expectedArgIdx >= numArgs - defaultArgs.size()) {
        PValue defaultVal =
            defaultArgs[expectedArgIdx + defaultArgs.size() - numArgs];
        if (failed(
                checkOneOperand(defaultVal, expectedType, expectedConvention)))
          return {};
        continue;
      }

      // Otherwise we have an argument count mismatch, just fail.
      return {};
    }

    // Otherwise we'll check the expected type against one (or more in the case
    // of varargs) provided values.

    // If we have a varargs argument, then it will eat the rest of the
    // arguments, but we have to check each of them.
    if (signature.isVarArg(expectedArgIdx)) {
      auto expectedVariadic = cast<VariadicType>(expectedType);
      auto varArgsEltType = expectedVariadic.getElementType();
      while (posOperandIdx != numPosOperands)
        if (failed(checkOneOperand(posOperands[posOperandIdx++], varArgsEltType,
                                   expectedVariadic.getConvention())))
          return {};
      continue;
    }

    // If we have a pack argument, then we're binding a variadic parameter with
    // multiple type values.  We need to consume all remaining arguments and use
    // their types as bindings.
    if (auto packType = getIfPackType(signature, expectedArgIdx)) {
      SmallVector<TypedAttr> types;
      auto variadicType = cast<VariadicType>(packType.getVariadic().getType());
      Type elementType = variadicType.getElementType();
      ExprEmitter emitter(shared, shared.getTopLevelDecl(), EC_TypeParamValue);
      SyntheticNode node(shared.getTopLevelDecl().getLoc());
      while (posOperandIdx != numPosOperands) {
        ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx++];
        CValue value = operand.ir.getIfCValue();
        if (!value) {
          shared.emitWarning(operand.expr->getLoc(),
                             "could not infer parameter type for this value, "
                             "because it is not concrete");
          return {};
        }
        ASTType toPush = value.getRValueType();
        // Infer nonmaterializable types as their materialization target.
        if (ASTType nmTarget = toPush.getNonmaterializableTarget(shared))
          toPush = nmTarget;
        Type metatype = toPush.getMetaType();
        TypedAttr actualAttr = TypeConstantAttr::get(
            toPush, metatype ? metatype : TypeType::get(shared.getContext()));
        if (!emitter.canImplicitlyConvertToType({actualAttr, node},
                                                elementType))
          return {};
        PValue result = emitter.emitPValue({actualAttr, node},
                                           EC_TypeParamValue, elementType);
        if (!result)
          return {};
        types.push_back(result);
      }

      matchTypes(PackType::get(VariadicAttr::get(types, variadicType)),
                 expectedType);
      continue;
    }

    // In the typical case, this argument isn't varargs or a pack, so just check
    // it.  If there was a problem, report it, otherwise continue on to the next
    // expected argument to check.
    if (failed(checkOneOperand(posOperands[posOperandIdx++], expectedType,
                               expectedConvention)))
      return {};
  }

  // If we have left over operands, then this signature cannot match.
  if (posOperandIdx != numPosOperands && !signature.hasParamVarArgs())
    return {};

  // We succeed iff we were able to infer a single (unique) value.
  if (!inferredValues.empty()) {
    PValue first = inferredValues.front();
    auto sameAsFirst = [&](PValue v) { return v.get() == first.get(); };
    if (llvm::all_of(inferredValues, sameAsFirst))
      return first;
  }

  return {};
}

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

  InflightDiag unexpectedKwArgs(StringSet<> &unknownKwOperands) const;
  InflightDiag wrongParamType(const InputParamBindings::Binding &actualBinding,
                              size_t paramIdx, ASTType expectedType) const;
  InflightDiag wrongParamCount(size_t expectedNumParams, size_t actualNumParams,
                               StringRef inputOrResult) const;
  InflightDiag wrongArgCount(size_t minRequiredArgs, size_t maxAllowedArgs,
                             size_t numOperands) const;
  InflightDiag wrongPosOnlyCount(size_t minRequiredArgs, size_t numOperands,
                                 const Twine &argOrParam) const;
  InflightDiag resultGenericMemType(Type outputType) const;
  InflightDiag argGenericMemType(size_t expectedArgIdx,
                                 Type expectedType) const;
  InflightDiag redundantArg(size_t argIdx, StringAttr argName) const;
  InflightDiag argTypeMismatch(OverloadFitness::ArgTypeMismatchKind kind,
                               ASTType ty, ASTExprAnd<AnyValue> operand,
                               size_t argIdx) const;
  InflightDiag missingArgs(ArrayRef<StringAttr> missingArgs,
                           const Twine &kindStr) const;
  InflightDiag posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const;

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
  if (callSyntax == CallSyntax::kMethodCall) {
    // It is probably possible for this assert to fire, if it does we should
    // tailor the error message.
    assert(argIdx != 0 && "TODO: unexpected self mismatch");
    diag << "method argument #" << (argIdx - 1);
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
DiagEmitter::unexpectedKwArgs(StringSet<> &unknownKwOperands) const {
  InflightDiag diag = initDiag();
  SmallVector<StringRef> keywords = llvm::map_to_vector(
      unknownKwOperands, [](auto &it) { return it.getKey(); });
  emitUnknownKeywords(diag, std::move(keywords), "argument");
  return diag;
}

InflightDiag
DiagEmitter::wrongParamType(const InputParamBindings::Binding &actualBinding,
                            size_t paramIdx, ASTType expectedType) const {
  return initDiag() << "callee parameter #" << paramIdx << " has "
                    << ASTType(expectedType) << " type, but value has type "
                    << ASTType(actualBinding.getType())
                    << actualBinding.expr->getRange();
}

InflightDiag DiagEmitter::wrongParamCount(size_t expectedNumParams,
                                          size_t actualNumParams,
                                          StringRef inputOrResult) const {
  InflightDiag diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, /*minRequired=*/expectedNumParams,
                           /*maxAllowed=*/expectedNumParams, actualNumParams,
                           Twine(inputOrResult) + " parameter");
  return diag;
}

InflightDiag DiagEmitter::wrongArgCount(size_t minRequiredArgs,
                                        size_t maxAllowedArgs,
                                        size_t numOperands) const {
  InflightDiag diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, minRequiredArgs, maxAllowedArgs, numOperands,
                           "argument");
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
  return initDiag() << "result cannot bind AnyRegType type to memory-only type "
                    << outputType;
}

InflightDiag DiagEmitter::argGenericMemType(size_t expectedArgIdx,
                                            Type expectedType) const {
  InflightDiag diag = initDiag();
  describeArgumentNo(diag, expectedArgIdx);
  return std::move(diag) << " cannot bind AnyRegType type to memory-only type "
                         << expectedType;
}

InflightDiag DiagEmitter::redundantArg(size_t argIdx,
                                       StringAttr argName) const {
  InflightDiag diag = initDiag();
  describeArgumentNo(diag, argIdx);
  return std::move(diag) << " (" << argName
                         << ") passed both as positional and keyword operand";
}

/// Attach extra type conversion error detail or hints to the user.
static void addTypeConversionDetail(InflightDiag &diag, SourceRange payloadLoc,
                                    ASTType payloadType, ASTType argType) {
  if (!payloadType) {
    diag.attachNote(payloadLoc.getStart())
        << "try resolving the overloaded function first" << payloadLoc;
    return;
  }
  // Try to detect mismatched byref result type.
  auto lhsSig = dyn_cast<SignatureType>(payloadType);
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
    diag.attachNote(payloadLoc.getStart())
        << "memory-only type bound to generic result type: "
        << (lhsByRef ? "payload" : "argument") << " returns "
        << ASTType(lhsRetType) << " by reference";
  }
}

/// Helper to get the RValueType from an operand.
static ASTType getRValueType(ASTExprAnd<AnyValue> operand) {
  AnyValue value = operand.ir;
  if (auto cValue = value.getIfCValue())
    return cValue.getRValueType();
  // Otherwise, try to narrow an overload set to a PValue.
  if (auto pValue = value.getIfORValue()->getIfPValue())
    return pValue.getType();
  return ASTType();
}

InflightDiag
DiagEmitter::argTypeMismatch(OverloadFitness::ArgTypeMismatchKind kind,
                             ASTType ty, ASTExprAnd<AnyValue> operand,
                             size_t argIdx) const {
  using ArgTypeMismatchKind = OverloadFitness::ArgTypeMismatchKind;
  InflightDiag diag = initDiag();
  switch (kind) {
  case ArgTypeMismatchKind::kNotLValue:
    if (callSyntax == CallSyntax::kMethodCall && argIdx == 0) {
      diag << "invalid use of mutating method on rvalue of type ";
      if (ASTType type = getRValueType(operand))
        diag << type;
      else
        diag << "unknown overload";
    } else {
      describeArgumentNo(diag, argIdx);
      diag << " must be mutable in order to pass as a by-ref argument";
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
    describeArgumentNo(diag, argIdx);
    diag << " cannot be converted from ";
    ASTType rValueType = getRValueType(operand);
    if (rValueType)
      diag << rValueType;
    else
      diag << "unknown overload";
    SourceRange payloadLoc = operand.expr->getRange();
    diag << " to " << ty << payloadLoc;
    addTypeConversionDetail(diag, payloadLoc, rValueType, ty);
    return diag;
  }
  default:
    llvm_unreachable("");
  }
}

InflightDiag DiagEmitter::missingArgs(ArrayRef<StringAttr> missingArgs,
                                      const Twine &kindStr) const {
  InflightDiag diag = initDiag() << "missing " << missingArgs.size()
                                 << " required " << kindStr << " argument"
                                 << plural(missingArgs.size()) << ": ";
  llvm::interleave(
      missingArgs, [&](StringAttr str) { diag << str; },
      [&]() { diag << ", "; });
  return diag;
}

InflightDiag
DiagEmitter::posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const {
  size_t num = posOnlyPassedByKw.size();
  InflightDiag diag = initDiag() << "got " << num << " positional-only argument"
                                 << plural(num) << " passed as keyword operand"
                                 << plural(num) << ": ";
  llvm::interleave(
      posOnlyPassedByKw, [&](StringAttr str) { diag << str; },
      [&]() { diag << ", "; });
  return diag;
}

//===----------------------------------------------------------------------===//
// OverloadFitness
//===----------------------------------------------------------------------===//

std::pair<size_t, size_t>
OverloadFitness::calculateMinMaxArgs(LITSignatureType signature) {
  size_t minRequiredArgs = 0;
  size_t maxAllowedArgs = 0;
  for (auto [idx, convention] :
       llvm::enumerate(signature.getInputConventions())) {
    // Ignore the return slot if present.
    if (convention == ValueInputConvention::ByRefResult)
      continue;

    // VarArgs arguments don't require a value, but allow any number of them.
    if (signature.isVarArg(idx)) {
      maxAllowedArgs = std::numeric_limits<size_t>::max();
      continue;
    }

    // Arguments with a pack type must have a known number of element types,
    // and so they require exactly that many arguments.
    if (auto packType = getIfPackType(signature, idx)) {
      size_t numValues = packType.getVariadicAttr().getValues().size();
      minRequiredArgs += numValues;
      maxAllowedArgs += numValues;
      continue;
    }

    // Otherwise, we have an ordinary argument that requires a value.
    ++minRequiredArgs;
    ++maxAllowedArgs;
  }

  // One less required argument for each argument that has a default value we
  // can use instead.
  minRequiredArgs -= signature.getDefaultPosArgs().size();

  return {minRequiredArgs, maxAllowedArgs};
}

std::pair<OverloadFitness::ArgTypeMismatchKind, ASTType>
OverloadFitness::checkOneOperand(ASTExprAnd<AnyValue> operand,
                                 ValueInputConvention expectedConvention,
                                 ASTType expectedType,
                                 size_t &numImplicitConversions,
                                 size_t &numMismatchedConventions,
                                 bool &hasNonmaterializableConversion,
                                 bool allowImplicitConversions, SMLoc loc,
                                 ASTDecl &declScope, SharedState &shared) {
  switch (expectedConvention) {
  case ValueInputConvention::InitSelf:
    // If this is an UnknownAttr, then it is a placeholder for type
    // checking, just let it pass.
    if (auto pValue = operand.ir.getIfPValue())
      if (isa<UnknownAttr>(pValue.get()))
        break;
    [[fallthrough]];
  case ValueInputConvention::ByRef:
  case ValueInputConvention::ByRefResult: {
    // The actual value must be an lvalue if callee takes things by-ref.
    auto argVal = operand.ir.getIfLValue();
    if (!argVal)
      return {kNotLValue, expectedType};

    // By-ref argument types must exactly match, no conversions are allowed.
    ASTType elementType = expectedType.getReferenceElementType();
    if (!argVal.getRValueType().isEqualCanon(elementType))
      return {kWrongLVType, expectedType};
    // If a register-passable type is being returned in-memory, remember this.
    numMismatchedConventions += elementType.isRegisterPassable(loc, shared);
    break;
  }
  case ValueInputConvention::BorrowedInMem:
  case ValueInputConvention::OwnedInMem:
    // Ignore the pointer type on memory conventions when matching types.
    // Note: Should do not support overloading on borrow/owned currently,
    // but we could add this if there is a reason to.
    expectedType = expectedType.getReferenceElementType();
    // If a register-passable type is being passed in-memory, remember this.
    numMismatchedConventions += expectedType.isRegisterPassable(loc, shared);
    [[fallthrough]];
  case ValueInputConvention::BorrowedInReg:
  case ValueInputConvention::OwnedInReg: {
    // If the argument is an overload set, see if it can be resolve to the
    // right type.
    CValue argVal;
    if (auto orValue = operand.ir.getIfORValue()) {
      if (!orValue->baseValue) { // Cannot merge base value.
        // Try to refine the ORValue into a PValue.
        argVal = orValue->getDirectSymbol(expectedType);
        if (!argVal)
          return {kWrongType, expectedType};
      }
    } else {
      argVal = operand.ir.getIfCValue();
      assert(argVal && "we handled ORValue above");
    }

    auto argType = argVal.getRValueType();
    // Otherwise, we pass as an r-value.  If the argument types match, then
    // they are good.
    if (argType.isEqualCanon(expectedType))
      break;
    if (auto nonmaterializableTarget =
            argType.getNonmaterializableTarget(shared)) {
      if (nonmaterializableTarget.isEqualCanon(expectedType)) {
        // Implicit conversion for nonmaterializable types to their target
        // type is allowed even if !allowImplicitConversions.  Even though
        // this may be an implicit conversion, don't increment the
        // numImplicitConversions count so that it will win against other
        // implicit conversions.  However, we keep track of whether
        // nonmaterializable autoconversion has happened so that functions
        // that literally take the nonmaterializable type can still win
        // instead of autoconverting if their signature matches exactly.
        hasNonmaterializableConversion = true;
        break;
      }
    }

    // Argument name mismatches don't count as implicit conversions.
    if (canZeroCostConvert(shared, argType, expectedType))
      break;

    // If implicit conversions are possible and one will work, then we succeed
    // with that conversion.
    if (allowImplicitConversions &&
        ExprEmitter::canImplicitlyConvertToType(
            declScope, shared, {argVal, operand.expr}, expectedType)) {
      // If we had one, this bumps our # implicit conversions.
      ++numImplicitConversions;
      break;
    }

    // Check value -> reference conversion is allowed in an argument.  This can
    // be performed by passing the existing address of dropping something into a
    // memory box.
    // TODO(references): if we had lifetimeof(self) and inout/borrowed
    // overloading, we could get rid of this implicit conversion.
    if (auto expectedRef = dyn_cast<RefType>(expectedType)) {
      // Element type and address have to be exactly equal, the mutability just
      // has to be compatible.
      if (ASTType(argType).isEqualCanon(expectedRef.getElementType()) &&
          // We don't currently support non-MValues.  We could dump them into
          // memory with an MBValue conversion if there is a need to.
          argVal.isMValue()) {
        auto argRefType = cast<RefType>(argVal.getMValueReference().getType());
        if (canZeroCostConvert(shared, argRefType, expectedRef))
          break;
      }
    }

    // Otherwise this is the wrong type for the argument.
    return {kWrongType, expectedType};
  }
  case ValueInputConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }

  return {kValidType, expectedType};
};

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
  return 4 * hasNonmaterializableConversion + 2 * passesVarArgArgument +
         1 * hasVariadicParams;
}

/// Helper to diagnose common cases of candidate mismatch related to keyword
/// arguments/operands (unexpected kw-operands, pos-only argument provided by
/// kw-operand, missing kw-only arguments).
static std::optional<InflightDiag>
diagnoseKeywordOperands(LITSignatureType signature,
                        const CallOperands &callOperands,
                        const DiagEmitter &emitDiagFor) {
  // First, we collect any (named) pos-only arguments passed by keyword operand,
  // and missing kw-only arguments. We also collect all argument names that
  // might be specified by keyword.
  StringSet<> argNames;
  SmallVector<StringAttr> posOnlyPassedByKw;
  SmallVector<StringAttr> missingKwOnly;
  for (auto [argIdx, argName, argPassingKind] : llvm::enumerate(
           signature.getArgNames(), signature.getArgPassingKinds())) {
    if (argPassingKind == PassingKind::PosOnly) {
      if (callOperands.findKwArg(argName))
        posOnlyPassedByKw.emplace_back(argName);
      continue;
    }
    if (argPassingKind == PassingKind::KwOnly &&
        !callOperands.findKwArg(argName)) {
      missingKwOnly.emplace_back(argName);
      continue;
    }
    if (signature.isVarArg(argIdx) || signature.isPackVarArg(argIdx))
      continue; // Variadic/pack args cannot be specified by keyword.
    auto [_, addedNew] = argNames.insert(argName);
    assert(addedNew && "duplicate argument name in signature");
  }

  if (!missingKwOnly.empty())
    return emitDiagFor.missingArgs(missingKwOnly, "keyword-only");
  if (!posOnlyPassedByKw.empty())
    return emitDiagFor.posOnlyPassedByKw(posOnlyPassedByKw);

  // TODO(#21295): handle variadic keyword arguments.
  // Then we find all the keyword operands with unknown names.
  StringSet<> unknownKwOperands;
  if (callOperands.hasKwOperands()) {
    for (auto [name, operandVal] : *callOperands.kwOperands)
      if (!argNames.contains(name))
        unknownKwOperands.insert(name);
  }

  if (!unknownKwOperands.empty())
    return emitDiagFor.unexpectedKwArgs(unknownKwOperands);

  return std::nullopt;
}

OverloadFitness OverloadFitness::evaluate(LITSignatureType signature,
                                          const OverloadSet &callable,
                                          const CallOperands &callOperands,
                                          bool allowImplicitConversions) {
  // We set up diagnostics.
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  size_t numPosOperands = posOperands.size();
  size_t numOperands = numPosOperands + callOperands.getNumKwOperands();
  SMLoc callLoc = callable.expr->getLoc();
  SharedState &shared = callable.getShared();
  DiagEmitter emitDiagFor(shared, callLoc, numOperands, callable.syntax);

  if (auto diag = diagnoseKeywordOperands(signature, callOperands, emitDiagFor))
    return std::move(*diag);

  // Check that the signature can be rebound with this set of bindings. We use
  // diagnostic handlers to capture any issues.
  InflightDiag diag = shared.emitError(callLoc);
  InputParamBindings::DiagEmitter bindingDiag{
      /*emitParamCount=*/
      [&](size_t numActual, bool posOnly) {
        if (posOnly) {
          size_t numPosOnly = countNumPosOnly(signature.getParamPassingKinds());
          diag = emitDiagFor.wrongPosOnlyCount(numPosOnly, numActual,
                                               "input parameter");
        } else {
          // Hide the implicit trait parameters from the diagnostic.
          // FIXME(#25492): This is awkward and the model should be reworked.
          size_t hidden = 0;
          if (ASTType type = callable.baseType)
            if (isa_and_nonnull<TraitType>(type.getMetaType()))
              hidden = 2;
          size_t numExpected = signature.getNumInputParams() - hidden;
          diag = emitDiagFor.wrongParamCount(numExpected, numActual - hidden,
                                             "input");
        }
      },
      /*emitPosType=*/
      [&](size_t paramIdx, const InputParamBindings::Binding &binding,
          ASTType expectedType) {
        diag = emitDiagFor.wrongParamType(binding, paramIdx, expectedType);
      },
      /*emitKwType=*/
      [&](StringAttr paramName, const InputParamBindings::Binding &binding,
          ASTType expectedType) {
        diag << "callee parameter '" << paramName << "' has "
             << ASTType(expectedType) << " type, but value has type "
             << ASTType(binding.getType()) << binding.expr->getRange();
      },
      /*emitUnknownKw=*/
      [&](SmallVectorImpl<StringRef> &&unknownKeywords) {
        emitUnknownKeywords(diag, std::move(unknownKeywords), "parameter");
      },
      /*emitRedundantKw=*/
      [&](size_t paramIdx, StringAttr paramName) {
        diag << "parameter #" << paramIdx << " (" << paramName
             << ") passed both as positional and keyword operand";
      },
      /*emitPosOnlyPassedByKw=*/
      [&](SmallVectorImpl<StringRef> &&names) {
        emitPosOnlyPassedByKw(diag, std::move(names), "parameter");
      },
      /*emitCtadFailure=*/
      [&](size_t paramIdx) {
        ASTDecl &decl = *callable.baseType.getDecl(shared);
        auto structOp = cast<StructDeclOp>(decl);
        diag << "could not deduce parameter #" << paramIdx << " ("
             << structOp.getSignature().getParamNames()[paramIdx]
             << ") of parent struct " << *decl.getNameIfOperation();
        diag.attachNote(decl.getLoc()) << "struct declared here";
      },
  };

  auto parameterInferenceHook =
      [&](size_t index, ArrayRef<TypedAttr> bindingsSoFar,
          TypedAttr defaultParam, ParserParamEvaluator &evaluator) -> PValue {
    if (PValue inferred = ParameterInferenceState(shared, index, evaluator)
                              .infer(signature, bindingsSoFar, callOperands))
      return inferred;
    return PValue(defaultParam);
  };
  auto [newBindings, bindingFitness] =
      callable.inputParamBindings.verifyBindings(
          signature, bindingDiag, parameterInferenceHook,
          /*isPackVarArg=*/signature.hasPackVarArgs() && !posOperands.empty());

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
  for (Type outputType : signature.getValueResults()) {
    if (!ASTType(outputType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.resultGenericMemType(outputType);
    // `!kgen.variant` is a special case.  We use it to wrap the result
    // type for functions that `raise`, and in the case of returning a
    // parametric type we can't rule out a user trying to pass a memory-only
    // type.  This came up in issue
    // https://github.com/modularml/mojo/issues/910.  So we need a deep check
    // to prevent memory-only types being used as parameters.
    ASTDecl *decl = ASTType(outputType).getDecl(shared);
    if (!decl) {
      if (auto variant = dyn_cast<VariantType>(outputType)) {
        auto isMemoryOnly = [&](Type variant) {
          return ASTType(variant).getRegisterPassability(callLoc, shared) ==
                 TypeConvention::MemoryOnly;
        };
        if (llvm::any_of(variant.getTypes(), isMemoryOnly))
          return emitDiagFor.resultGenericMemType(outputType);
      }
    }
  }

  // Ok, the parameters all line up, check the argument list.  We generally want
  // to diagnose problems where too few or too many arguments are passed if that
  // is the problem, rather than complaining about a type error of some argument
  // that doesn't work out.  Check for that first.
  auto [minRequiredArgs, maxAllowedArgs] = calculateMinMaxArgs(signature);
  if (numOperands < minRequiredArgs || maxAllowedArgs < numOperands) {
    return emitDiagFor.wrongArgCount(minRequiredArgs, maxAllowedArgs,
                                     numOperands);
  }
  auto isPosOnlyArg = [](auto p) {
    auto [kind, convention] = p;
    if (convention == ValueInputConvention::ByRefResult)
      return false;
    return kind == PassingKind::PosOnly;
  };
  size_t numPosOnlyArgs =
      llvm::count_if(llvm::zip(signature.getArgPassingKinds(),
                               signature.getInputConventions()),
                     isPosOnlyArg);
  if (numPosOperands < numPosOnlyArgs) {
    return emitDiagFor.wrongPosOnlyCount(numPosOnlyArgs, numPosOperands,
                                         "argument");
  }

  SMLoc loc = callable.expr->getLoc();

  // We will accumulate the implicit conversion in arguments to those counted
  // for the parameter bindings.
  size_t numImplicitConversions = bindingFitness.numImplicitConversions;
  bool hasNonmaterializableConversion = false;

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
                            ValueInputConvention expectedConvention,
                            ASTType expectedType) {
    return checkOneOperand(operand, expectedConvention, expectedType,
                           numImplicitConversions, numMismatchedConventions,
                           hasNonmaterializableConversion,
                           allowImplicitConversions, loc,
                           callable.inputParamBindings.declScope, shared);
  };

  // We collect missing arguments (in case the argument count check didn't catch
  // this).
  SmallVector<StringAttr> missingArgs;

  // Use a ParserParamEvaluator to substitute 'apply' expressions in the
  // argument types.
  ParserParamEvaluator evaluator(*shared.declResolver);
  for (auto [expectedArgIdx, unboundExpectedType, expectedConvention, argName,
             passingKind] :
       llvm::enumerate(signature.getValueInputs(),
                       signature.getInputConventions(), signature.getArgNames(),
                       signature.getArgPassingKinds())) {
    assert(!signature.isKWVarArg(expectedArgIdx) &&
           "`**arg` variadics not supported yet");

    // Ignore the return slot if present.
    Type expectedType = evaluator.refineType(unboundExpectedType);
    if (expectedConvention == ValueInputConvention::ByRefResult) {
      numMismatchedConventions += ASTType(expectedType)
                                      .getReferenceElementType()
                                      .isRegisterPassable(loc, shared);
      continue;
    }

    // If the arguments or results got bound to a memory-only type then their
    // argument convention needs to change.  We cannot support this until we get
    // proper type traits.  Note that the PointerType is considered a valid
    // register passable type, so things passed byref are ok.
    if (!ASTType(expectedType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.argGenericMemType(expectedArgIdx, expectedType);

    // Handle case when there are no more provided positional operands.
    if (posOperandIdx == numPosOperands) {
      // If the argument is a varargs argument list or pack, then it can be
      // initialized with zero values no problem.
      if (signature.isVarArg(expectedArgIdx) ||
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

      // We don't need to provide a value for this argument if it has a default.
      if (expectedArgIdx >=
          signature.getNumInputs() - signature.getDefaultPosArgs().size()) {
        // Arguments with default values must be followed only by other
        // arguments with default values, or by keyword arguments.
        continue;
      }

      // We have an argument that was not passed a value. We collect these.
      missingArgs.push_back(argName);
      continue;
    }

    /// Check and process a single positional operand and advance the operand
    /// index.
    auto processPositionalOperand =
        [&](ASTType expectedType,
            ValueInputConvention conv) -> std::optional<InflightDiag> {
      ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx];
      auto [kind, ty] = checkAnOperand(operand, conv, expectedType);
      if (kind != kValidType)
        return emitDiagFor.argTypeMismatch(kind, ty, operand, posOperandIdx);
      ++posOperandIdx;
      return std::nullopt;
    };

    // If we have a varargs argument, then it will eat the rest of the
    // positional arguments, but we have to check each of them.
    if (signature.isVarArg(expectedArgIdx)) {
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
    if (PackType packType = getIfPackType(signature, expectedArgIdx)) {
      for (TypedAttr element : packType.getVariadicAttr().getValues()) {
        if (auto result =
                processPositionalOperand(ASTType(element), expectedConvention))
          return std::move(*result);
        passesVarArgArgument = true;
      }
      continue;
    }

    // Otherwise, we have an ordinary positional argument that is not varargs or
    // a pack. Ensure that it is not also passed as a keyword operand, then
    // process it as usual.
    if (passingKind != PassingKind::PosOnly) {
      assert(!argName.empty());
      if (callOperands.findKwArg(argName))
        return emitDiagFor.redundantArg(expectedArgIdx, argName);
    }
    if (auto result =
            processPositionalOperand(expectedType, expectedConvention))
      return std::move(*result);
  }

  // If any arguments were missing, we emit diagnostics.
  if (!missingArgs.empty())
    return emitDiagFor.missingArgs(missingArgs, "positional");

  assert(posOperandIdx == numPosOperands &&
         "should handle argument mismatch above");

  // Otherwise we succeeded!
  return {newBindings,
          Payload{numImplicitConversions, numMismatchedConventions,
                  hasNonmaterializableConversion, passesVarArgArgument,
                  bindingFitness.hasVariadicParams}};
}
