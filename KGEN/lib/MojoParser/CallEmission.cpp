//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "Utils.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

#include <limits>

#define DEBUG_TYPE "LITEXPRCALLS"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

void InputParamBindings::addPrechecked(TypedAttr precheckedBinding) {
  posBindings.push_back({nullptr, precheckedBinding, /*typeChecked=*/true});
}

void InputParamBindings::add(const ExprNode *expr, TypedAttr value) {
  posBindings.push_back({expr, value, /*typeChecked=*/false});
}

//===----------------------------------------------------------------------===//
// ParameterInferenceState Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParameterInferenceState {
public:
  ParameterInferenceState(SharedState &state, size_t index, Type type)
      : state(state), parameterIndex(index) {}

  /// Given an incomplete parameter binding set for a call to the specified
  /// signature, try to infer the value of the next 'decl' parameter.  This
  /// should always return null /without/ an error if it cannot be inferred, and
  /// return a specific value if unambiguously determined.
  PValue infer(LITSignatureType signature, ArrayRef<TypedAttr> bindingsSoFar,
               const CallOperands &callOperands);

private:
  LogicalResult matchTypes(Type actualType, Type expectedType);
  LogicalResult matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);
  LogicalResult checkOneOperand(ASTExprAnd<AnyValue> operand,
                                ASTType expectedType,
                                ValueInputConvention expectedConvention);

  SharedState &state;
  size_t parameterIndex;
  SmallVector<PValue> inferredValues;
};
} // namespace

LogicalResult ParameterInferenceState::matchTypes(Type actualType,
                                                  Type expectedType) {
  // If the expected type is a parameter ref, then we're binding the specified
  // type to an attribute parameter.
  if (auto expectedParamRef = dyn_cast<ParamRefType>(expectedType))
    return matchParams(ParameterizedTypeConstantAttr::get(actualType),
                       expectedParamRef.getParam());

  // If the types trivially match then there is no inference to do.
  if (actualType == expectedType)
    return success();

  // Handle when both are DeclRefTypes.
  if (auto actualDRT = dyn_cast<DeclRefType>(actualType)) {
    if (auto expectedDRT = dyn_cast<DeclRefType>(expectedType)) {
      // Ignore if these are two fundamentally different symbols.
      if (actualDRT.getSymbol() != expectedDRT.getSymbol())
        return success();

      // Fail if the parameter lists fundamentally mismatch.
      // TODO: Defaulted parameters could make this ok?
      if (actualDRT.getParamValues().size() !=
          expectedDRT.getParamValues().size())
        return failure();

      // Match up the parameter bindings.
      for (auto [actual, expected] : llvm::zip(actualDRT.getParamValues(),
                                               expectedDRT.getParamValues())) {
        assert(actual.getName() == expected.getName());
        if (failed(matchParams(actual.getValue(), expected.getValue())))
          return failure();
      }
      return success();
    }
  }

  // Handle various common POP types for convenience, starting with SIMDType.
  if (auto actual = dyn_cast<POP::SIMDType>(actualType))
    if (auto expected = dyn_cast<POP::SIMDType>(expectedType))
      return failure(
          failed(matchParams(actual.getSize(), expected.getSize())) ||
          failed(matchParams(actual.getDType(), expected.getDType())));

  // POP::ArrayType.
  if (auto actual = dyn_cast<POP::ArrayType>(actualType))
    if (auto expected = dyn_cast<POP::ArrayType>(expectedType))
      return failure(
          failed(matchParams(actual.getSize(), expected.getSize())) ||
          failed(
              matchParams(actual.getElementType(), expected.getElementType())));

  // Handle PointerType.
  if (auto actual = dyn_cast<PointerType>(actualType))
    if (auto expected = dyn_cast<PointerType>(expectedType))
      return matchParams(actual.getElementType(), expected.getElementType());

  // Handle VariadicType
  if (auto actual = dyn_cast<VariadicType>(actualType))
    if (auto expected = dyn_cast<VariadicType>(expectedType))
      return matchParams(actual.getElementType(), expected.getElementType());

  // TODO: Could do StructType?
  LLVM_DEBUG(llvm::errs() << "CANNOT INFER MISMATCH TYPES:\n";
             actualType.dump(); expectedType.dump();
             llvm::errs() << parameterIndex);
  return success();
}

LogicalResult ParameterInferenceState::matchParams(TypedAttr actualAttr,
                                                   TypedAttr expectedAttr) {

  // We can only match up these values if their types match.
  if (actualAttr.getType() != expectedAttr.getType() &&
      failed(matchTypes(actualAttr.getType(), expectedAttr.getType())))
    return failure();

  // If the expected value is the parameter declaration in question, remember
  // this value!
  if (auto ire = dyn_cast<ParamIndexRefAttr>(expectedAttr)) {
    if (ire.getDepth() == 0 && !ire.getIsResult() &&
        ire.getIndex() == parameterIndex)
      inferredValues.push_back(actualAttr);
    return success();
  }

  // If the attrs trivial match then we're done and there is no inference to do.
  if (actualAttr == expectedAttr)
    return success();

  LLVM_DEBUG(llvm::errs() << "CANNOT INFER MISMATCHING ATTRS:\n";
             actualAttr.dump(); expectedAttr.dump();
             llvm::errs() << parameterIndex << "\n");
  return success();
}

LogicalResult ParameterInferenceState::checkOneOperand(
    ASTExprAnd<AnyValue> operand, ASTType expectedType,
    ValueInputConvention expectedConvention) {
  // We'll bind the next provided value.
  switch (expectedConvention) {
  case ValueInputConvention::InitSelf:
    // If this is an UnknownAttr, then it is a placeholder for type
    // checking, just let it pass.
    if (PValue pValue = operand.ir.getIfPValue())
      if (isa<UnknownAttr>(pValue.get()))
        return success();
    [[fallthrough]];
  case ValueInputConvention::ByRef:
  case ValueInputConvention::ByRefResult: {
    // The actual value must be an lvalue if callee takes things by-ref.
    LValue argVal = operand.ir.getIfLValue();
    if (!argVal)
      return failure();

    // By-ref argument types must exactly match, no conversions are allowed.
    return matchTypes(argVal.getRValueType(),
                      expectedType.getReferenceElementType());
  }

  case ValueInputConvention::OwnedInMem:
  case ValueInputConvention::BorrowedInMem:
    // Otherwise,we expect an r-value to match up, ignoring the pointer type
    // from the convention.
    expectedType = expectedType.getReferenceElementType();
    [[fallthrough]];
  case ValueInputConvention::OwnedInReg:
  case ValueInputConvention::BorrowedInReg:
    // Otherwise, we pass as an r-value if we know the type.
    // TODO: Consider implicit conversions?
    if (CValue cValue = operand.ir.getIfCValue())
      return matchTypes(cValue.getRValueType(), expectedType);
    // Consider the types of ORValues with single candidates.
    if (ORValue orValue = operand.ir.getIfORValue())
      if (PValue pValue = orValue->emitAsPValue())
        return matchTypes(pValue.getType(), expectedType);
    return success();
  }
  llvm_unreachable("invalid value input convention");
};

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
      if (signature.isVarArg(expectedArgIdx))
        break;

      // If we have a pack argument, then we're binding zero type values to it.
      if (POP::PackType packType = getIfPackType(signature, expectedArgIdx)) {
        if (!inferredValues.empty())
          break;
        inferredValues.push_back(VariadicAttr::get(
            {}, cast<VariadicType>(packType.getVariadic().getType())));
        continue;
      }

      // Check if a keyword operand was provided for this argument
      if (std::optional<ASTExprAnd<AnyValue>> kwOperandOr =
              callOperands.findKwArg(argName)) {
        if (failed(checkOneOperand(*kwOperandOr, expectedType,
                                   expectedConvention)))
          return {};
        continue;
      }

      // TODO: If this argument is defaulted, infer against it.

      // Otherwise we have an argument count mismatch, just fail.
      return {};
    }

    // Otherwise we'll check the expected type against one (or more in the case
    // of varargs) provided values.

    // If we have a varargs argument, then it will eat the rest of the
    // arguments, but we have to check each of them.
    if (signature.isVarArg(expectedArgIdx)) {
      auto varArgsEltType = ASTType(expectedType).getVariadicElementType();
      while (posOperandIdx != numPosOperands)
        if (failed(checkOneOperand(posOperands[posOperandIdx++], varArgsEltType,
                                   expectedConvention)))
          return {};
      continue;
    }

    // If we have a pack argument, then we're binding a variadic parameter with
    // multiple type values.  We need to consume all remaining arguments and use
    // their types as bindings.
    if (auto packType = getIfPackType(signature, expectedArgIdx)) {
      if (!inferredValues.empty())
        break;
      SmallVector<TypedAttr> types;
      while (posOperandIdx != numPosOperands) {
        ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx++];
        CValue value = operand.ir.getIfCValue();
        if (!value) {
          state.emitWarning(operand.expr->getLoc(),
                            "could not infer parameter type for this value, "
                            "because it is not concrete");
          return {};
        }
        ASTType toPush = value.getRValueType();
        // Infer nonmaterializable types as their materialization target.
        if (ASTType nmTarget = toPush.getNonmaterializableTarget(state))
          toPush = nmTarget;
        types.push_back(ParameterizedTypeConstantAttr::get(toPush));
      }

      inferredValues.push_back(VariadicAttr::get(
          types, cast<VariadicType>(packType.getVariadic().getType())));
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
    if (llvm::all_of(inferredValues, sameAsFirst)) {
      // Infer nonmaterializable types as their materialization target.
      if (ASTType typeVal = first.getIfTypeValue()) {
        if (ASTType nmTarget = typeVal.getNonmaterializableTarget(state))
          return PValue(nmTarget);
      }
      return first;
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// InputParamBindings Implementation
//===----------------------------------------------------------------------===//

/// Check a single binding and emit a parameter value if possible. If an
/// implicit conversion is required, the provided counter is incremented.
static PValue emitSingleParameterValue(size_t index,
                                       InputParamBindings::Binding binding,
                                       ASTType expectedType,
                                       size_t &numImplicitConversions,
                                       ExprEmitter &emitter,
                                       ParserParamEvaluator &evaluator) {
  assert(binding.expr &&
         "should always have an expr tree for unchecked bindings");

  // Check the type matches what is expected, and perform an implicit
  // conversion if needed.
  expectedType = ASTType(evaluator.getReboundType(expectedType.mlirType));

  PValue bindingPVal = PValue(binding.getValue());

  // If the parameter already has the right type, then we're good.
  if (expectedType.isEqualCanon(binding.getValue().getType()))
    return bindingPVal;

  // If the parameter can be implicitly converted, do so.
  if (emitter.canImplicitlyConvertToType({bindingPVal, binding.expr},
                                         expectedType)) {
    auto argValue = emitter.emitPValue({bindingPVal, binding.expr},
                                       EC_CallParamValue, expectedType);
    assert(argValue && "Already checked this would succeed");
    ++numImplicitConversions;
    return argValue;
  }

  return {};
};

std::pair<ParameterExprArrayAttr, InputParamBindings::Fitness>
InputParamBindings::verifyBindings(
    ArrayRef<Type> expectedParamTypes, ArrayRef<TypedAttr> defaultParams,
    ExprEmitter &emitter, bool hasParamVarArgs,
    ParameterInferenceHookTy parameterInferenceHook, bool isPackVarArg,
    SetEvaluatorHookTy setEvaluator, function_ref<void()> emitParamCountDiag,
    function_ref<void(size_t, Binding &, ASTType)> emitParamTypeDiag) const {

  // If we have bound parameters, type check them now and bind names to them.
  size_t numParams = expectedParamTypes.size();
  SmallVector<TypedAttr> newBindings;
  newBindings.reserve(numParams);

  // We use the contextual emitter to perform implicit conversions, but these
  // conversions must be done within a parameter context.  Make sure we don't
  // have a builder from the caller, this indicates that an PValue is required.
  llvm::SaveAndRestore savedBuilder(emitter.builder, {});
  llvm::SaveAndRestore savedContext(emitter.paramContext, EC_ParameterList);

  // Parameters defined at the beginning of the parameter list may be used by
  // the types of other parameters defined later in the list, e.g. in:
  //    [rank: Int, indices: StaticTuple[rank]]
  // the value provided to 'indices' should actually depend on the specified
  // value of 'rank'.  We use a ParameterEvaluator to keep track of the mapping
  // so far and remap types on demand.
  ParserParamEvaluator evaluator(emitter.getDeclResolver());

  // This lambda installs the decl's value in the parameter evaluator and new
  // binding array.
  auto setParamValue = [&](TypedAttr value) {
    if (setEvaluator)
      setEvaluator(newBindings.size(), value, evaluator);
    else
      evaluator.addInputValue(value);
    newBindings.push_back(value);
  };

  // This lambda hides the diagnostic and error handling logic for checking a
  // single parameter bdingin.
  Fitness fitness{0, false};
  auto handleSingleBinding = [&](size_t index, Binding binding,
                                 ASTType expectedType) -> PValue {
    PValue pValue = emitSingleParameterValue(index, binding, expectedType,
                                             fitness.numImplicitConversions,
                                             emitter, evaluator);
    if (!pValue) {
      // Set the diagnostic metadata and call the custom diagnostic handler.
      fitness.expectedBinding = std::make_pair(index, expectedType);
      emitParamTypeDiag(index, binding, expectedType);
    }

    return pValue;
  };

  size_t bindingIdx = 0;
  size_t numPosBindings = posBindings.size();
  for (auto [idx, type] : llvm::enumerate(expectedParamTypes)) {
    bool isVarArg = idx + 1 == numParams && hasParamVarArgs;

    // Check to see if we ran out of bindings to provide to this param decl.
    if (bindingIdx == numPosBindings) {
      // Determine what type we expect next.
      Type requestedType = evaluator.getReboundType(type);
      Type expectedType = requestedType;
      // If this is a vararg parameter, infer using the element type.
      if (isVarArg)
        if (auto varType = dyn_cast<VariadicType>(expectedType))
          expectedType = ASTType(varType.getElementType());

      // If we have a method to infer parameter values, invoke it to see if we
      // can get an inferred value for the parameter.
      if (parameterInferenceHook) {
        if (PValue pValue =
                parameterInferenceHook(idx, type, expectedType, newBindings)) {
          assert(pValue.getType().mlirType == requestedType &&
                 "inferred a parameter value of wrong type");
          setParamValue(pValue);
          continue;
        }
      }

      // If the parameter decl is a variadic parameter list, and do not have
      // pack operands that could be used to infer those parameters, then we can
      // fulfill it with an empty list.  We know it must be the last parameter
      // decl. If this isn't actually a variadic type, then we simply reached
      // the end of the parameter list.
      if (isVarArg && !isPackVarArg) {
        if (auto varType = dyn_cast<VariadicType>(type)) {
          setParamValue(VariadicAttr::get({}, varType));
          fitness.lastExpectedType = expectedType;
          continue;
        }
      }

      // If available, we use a default parameter value.
      if (idx >= numParams - defaultParams.size()) {
        setParamValue(defaultParams[idx + defaultParams.size() - numParams]);
        continue;
      }

      // Otherwise, we're simply missing bindings.
      fitness.lastExpectedType = expectedType;
      emitParamCountDiag();
      return {{}, fitness};
    }

    Binding binding = posBindings[bindingIdx];
    // If this value was already bound and checked, use it.
    if (binding.typeChecked) {
      setParamValue(binding.value);
      ++bindingIdx;
      continue;
    }

    // Scalar parameter values are installed directly. Or, if we have a variadic
    // of the same type, we can use it as the value of the parameter directly.
    // FIXME: This allows passing a variadic `Ts` directly. Do we want a new
    // PValue classification for `*Ts`, which is required to pass this legally?
    if (!isVarArg || binding.getValue().getType() == type) {
      PValue paramValue = handleSingleBinding(idx, binding, type);
      if (!paramValue)
        return {{}, fitness};
      setParamValue(paramValue);
      ++bindingIdx;
      continue;
    }

    // If the parameter is a variadic list, it may consume many values, and they
    // all get packed up into a VariadicAttr.
    fitness.hasVariadicParams = true;
    SmallVector<TypedAttr> elements;
    Type expectedType = ASTType(type).getVariadicElementType();
    do {
      binding = posBindings[bindingIdx++];
      PValue pValue = handleSingleBinding(idx, binding, expectedType);
      if (!pValue)
        return {{}, fitness};
      elements.emplace_back(pValue);
    } while (bindingIdx != numPosBindings);
    setParamValue(VariadicAttr::get(
        elements, VariadicType::get(evaluator.getReboundType(expectedType))));
  }

  // Check and complain if we have bindings that didn't get used.
  if (bindingIdx != numPosBindings) {
    emitParamCountDiag();
    return {{}, fitness};
  }

  return {ParameterExprArrayAttr::get(emitter.getContext(), newBindings),
          fitness};
}

/// Helper to produce a consistent error message for incorrect argument and
/// parameter counts.
static void emitWrongArgOrParamCount(InflightDiag &diag, size_t minRequired,
                                     size_t maxAllowed, size_t numActual,
                                     Twine argOrParam) {
  diag << " expects ";

  // Tailor the diagnostic if the exact number of expected args is known.
  if (minRequired == maxAllowed && numActual != minRequired) {
    diag << minRequired << " " << argOrParam << plural(minRequired);
  } else if (numActual < minRequired) {
    diag << "at least " << minRequired << " " << argOrParam
         << plural(minRequired);
  } else {
    assert(numActual > maxAllowed);
    diag << "at most " << maxAllowed << " " << argOrParam << plural(maxAllowed);
  }

  diag << ", but " << numActual << plural(numActual, " was", " were")
       << " specified";
}

std::pair<ParameterExprArrayAttr, InputParamBindings::Fitness>
InputParamBindings::verifyBindings(ArrayRef<Type> expectedParamTypes,
                                   ArrayRef<TypedAttr> defaultParams,
                                   ExprEmitter &emitter, bool hasParamVarArgs,
                                   StringRef baseName, Location opLoc,
                                   llvm::SMLoc exprLoc,
                                   SetEvaluatorHookTy setEvaluator) const {
  return verifyBindings(
      expectedParamTypes, defaultParams, emitter, hasParamVarArgs,
      /*parameterInferenceHook=*/{},
      /*isPackVarArg=*/false, setEvaluator, /*emitParamCountDiag=*/
      [&]() {
        size_t minRequired = expectedParamTypes.size() - defaultParams.size();
        size_t maxAllowed = expectedParamTypes.size();
        size_t actualNumParams = posBindings.size();
        InflightDiag diag = emitter.emitError(exprLoc, "'") << baseName << "'";
        emitWrongArgOrParamCount(diag, minRequired, maxAllowed, actualNumParams,
                                 "input parameter");
        diag.attachNote(opLoc) << "'" << baseName << "' declared here";
      },
      /*emitParamTypeDiag=*/
      [&](size_t index, Binding &binding, ASTType expectedType) {
        auto diag = emitter.emitError(binding.expr->getLoc(), "'")
                    << baseName << "' parameter #" << index << " has "
                    << expectedType << " type, but value has type "
                    << ASTType(binding.getValue().getType())
                    << binding.expr->getRange();
        diag.attachNote(opLoc) << "'" << baseName << "' declared here";
      });
}

std::pair<ParameterExprArrayAttr, InputParamBindings::Fitness>
InputParamBindings::verifyBindings(
    LITSignatureType sig, ExprEmitter &emitter,
    ParameterInferenceHookTy parameterInferenceHook, bool isPackVarArg) const {
  return verifyBindings(
      sig.getInputParamTypes(), sig.getDefaultParameters(), emitter,
      sig.hasParamVarArgs(), parameterInferenceHook, isPackVarArg,
      /*setEvaluator=*/{},
      /*emitParamCountDiag=*/[]() {},
      /*emitParamTypeDiag=*/[](size_t, Binding &, ASTType) {});
}

ParameterExprArrayAttr
InputParamBindings::verifyBindings(StructDeclOp structOp, ExprEmitter &emitter,
                                   llvm::SMLoc exprLoc) const {
  SmallVector<Type> paramTypes =
      llvm::map_to_vector(structOp.getInputParams(),
                          [](ParamDeclAttr decl) { return decl.getType(); });
  auto setParamValue = [&](size_t declIdx, TypedAttr value,
                           ParserParamEvaluator &evaluator) {
    evaluator.setParameterValue(structOp.getInputParams()[declIdx], value);
  };
  auto [bindingValuesAttr, _] =
      verifyBindings(paramTypes, structOp.getDefaultParameters(), emitter,
                     structOp.getParamVarArgs(), structOp.getName(),
                     structOp.getLoc(), exprLoc, setParamValue);
  return bindingValuesAttr;
}

ParameterExprArrayAttr
InputParamBindings::verifyBindings(LITSignatureType sig, ExprEmitter &emitter,
                                   StringRef baseName, Location opLoc,
                                   llvm::SMLoc exprLoc) const {
  auto [newBindings, _] =
      verifyBindings(sig.getInputParamTypes(), sig.getDefaultParameters(),
                     emitter, sig.hasParamVarArgs(), baseName, opLoc, exprLoc);
  return newBindings;
}

//===----------------------------------------------------------------------===//
// OverloadFitness Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This struct indicates whether a signature can be successfully applied to a
/// parameter binding and argument list.  If so, it keeps track of the number of
/// implicit conversions required to make the call, and if not, it indicates the
/// reason for the mismatch.
struct OverloadFitness {
  OverloadFitness(OverloadFitness &&other)
      : paramBindings(other.paramBindings),
        numImplicitConversions(other.numImplicitConversions),
        diag(other.diag ? std::optional<InflightDiag>(other.takeDiag())
                        : std::nullopt) {}

  ~OverloadFitness() {
    if (diag)
      takeDiag().abandon();
  }

  /// Return the parameter bindings if the candidate is valid.
  ParameterExprArrayAttr getParamBindings() const {
    assert(isValid());
    return paramBindings;
  }

  /// Return the number of implicit conversions if the candidate is valid.
  size_t getNumImplicitConversions() const {
    assert(isValid());
    return numImplicitConversions;
  }

  /// Consume the diagnostic if the candidate is not valid.
  InflightDiag takeDiag() {
    assert(!isValid());
    return std::move(*diag);
  }

  /// Return whether the candidate was valid.
  bool isValid() const { return !diag; }

  /// Determine whether the specified signature can be invoked with the
  /// parameter bindings specified in `callable` and the arguments specified in
  /// `operands`.
  static OverloadFitness evaluate(LITSignatureType signature,
                                  const OverloadSet &callable,
                                  const CallOperands &callOperands,
                                  bool allowImplicitConversions,
                                  ExprEmitter &emitter);

  enum ArgTypeMismatchKind {
    kValidType,   //< No argument type mismatch.
    kNotLValue,   //< By-ref argument requires an lvalue, but got an rvalue.
    kWrongLVType, //< By-ref argument and provided l-value types mismatch.
    kWrongType,   //< An argument value not convertible to the expected type.
  };

private:
  /// For valid candidates, this defines the parameter bindings to use.
  ParameterExprArrayAttr paramBindings;
  /// The number of implicit conversions required;
  size_t numImplicitConversions;
  /// The diagnostic for invalid candidates, or null for valid ones.
  std::optional<InflightDiag> diag = std::nullopt;

  OverloadFitness(InflightDiag &&diag) : diag(std::move(diag)) {}
  OverloadFitness(ParameterExprArrayAttr paramBindings,
                  size_t numImplicitConversions)
      : paramBindings(paramBindings),
        numImplicitConversions(numImplicitConversions) {}

  /// Calculate the minimum required and maximum allowed number of arguments
  /// from a signature.
  static std::pair<size_t, size_t>
  calculateMinMaxArgs(LITSignatureType signature);

  /// Check the expected type against the provided operand. This identifies any
  /// problems with the operand type and also returns the type to be used for
  /// error propagation.
  static std::pair<ArgTypeMismatchKind, ASTType>
  checkOneOperand(ASTExprAnd<AnyValue> operand,
                  ValueInputConvention expectedConvention, ASTType expectedType,
                  size_t &numImplicitConversions,
                  bool &hasNonmaterializableConversion,
                  bool allowImplicitConversions, ExprEmitter &emitter);
};
} // namespace

/// Helper class to emit errors without cluttering the evaluation logic.
struct DiagEmitter {
  DiagEmitter(SMLoc callLoc, size_t numOperands, CallSyntax callSyntax,
              ExprEmitter &emitter)
      : callLoc(callLoc), numOperands(numOperands), callSyntax(callSyntax),
        emitter(emitter) {}

  InflightDiag unexpectedKwArgs(StringSet<> &unknownKwOperands) {
    size_t numUnknownKws = unknownKwOperands.size();
    InflightDiag diag = initDiag() << "unexpected keyword argument"
                                   << plural(numUnknownKws) << ": ";

    // We need to sort the unknown keywords to have reproducible errors.
    SmallVector<StringRef> sorted;
    for (auto &it : unknownKwOperands)
      sorted.emplace_back(it.getKey());
    llvm::sort(sorted);
    llvm::interleave(
        sorted, [&](StringRef str) { diag << "'" << str << "'"; },
        [&]() { diag << ", "; });
    return diag;
  }

  InflightDiag wrongParamType(const InputParamBindings::Binding &actualBinding,
                              size_t paramIdx, ASTType expectedType) {
    return initDiag() << "callee parameter #" << paramIdx << " has "
                      << ASTType(expectedType) << " type, but value has type "
                      << ASTType(actualBinding.getType())
                      << actualBinding.expr->getRange();
  }

  InflightDiag wrongParamCount(size_t expectedNumParams, size_t actualNumParams,
                               StringRef inputOrResult) {
    InflightDiag diag = initDiag() << "callee";
    emitWrongArgOrParamCount(diag, /*minRequired=*/expectedNumParams,
                             /*maxAllowed=*/expectedNumParams, actualNumParams,
                             Twine(inputOrResult) + " parameter");
    return diag;
  }

  InflightDiag wrongArgCount(size_t minRequiredArgs, size_t maxAllowedArgs,
                             size_t numOperands) {
    InflightDiag diag = initDiag() << "callee";
    emitWrongArgOrParamCount(diag, minRequiredArgs, maxAllowedArgs, numOperands,
                             "argument");
    return diag;
  }

  InflightDiag resultGenericMemType(Type outputType) {
    return initDiag()
           << "result cannot bind generic !mlirtype to memory-only type "
           << outputType;
  }

  InflightDiag argGenericMemType(size_t expectedArgIdx, Type expectedType) {
    InflightDiag diag = initDiag();
    describeArgumentNo(diag, expectedArgIdx);
    return std::move(diag)
           << " cannot bind generic !mlirtype to memory-only type "
           << expectedType;
  }

  InflightDiag redundantArg(size_t argIdx, StringAttr argName) {
    InflightDiag diag = initDiag();
    describeArgumentNo(diag, argIdx);
    return std::move(diag) << " (" << argName
                           << ") passed both as positional and keyword operand";
  }

  InflightDiag argTypeMismatch(OverloadFitness::ArgTypeMismatchKind kind,
                               ASTType ty, ASTExprAnd<AnyValue> operand,
                               size_t argIdx) {
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
      return std::move(diag)
             << "l-value of type " << operand.ir.getIfLValue().getRValueType()
             << " cannot be converted to reference of type "
             << ty.getReferenceElementType() << operand.expr->getRange();
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

private:
  SMLoc callLoc;
  size_t numOperands;
  CallSyntax callSyntax;
  ExprEmitter &emitter;

  /// Attach extra type conversion error detail or hints to the user.
  static void addTypeConversionDetail(InflightDiag &diag,
                                      SourceRange payloadLoc,
                                      ASTType payloadType, ASTType argType) {
    if (!payloadType) {
      diag.attachNote(payloadLoc.getStart())
          << "try resolving the overloaded function first" << payloadLoc;
      return;
    }
    // Try to detect mismatched byref result type.
    auto lhsSig = dyn_cast<SignatureType>(payloadType.mlirType);
    auto rhsSig = dyn_cast<SignatureType>(argType.mlirType);
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
    if (auto pValue = value.getIfORValue()->emitAsPValue())
      return pValue.getType();
    return ASTType();
  }

  /// Wrapper around pretty printing logic for an argument given by index.
  void describeArgumentNo(InflightDiag &diag, size_t argIdx) {
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

  InflightDiag initDiag() { return emitter.emitError(callLoc); }
};

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
  minRequiredArgs -= signature.getDefaultArguments().size();

  return {minRequiredArgs, maxAllowedArgs};
}

std::pair<OverloadFitness::ArgTypeMismatchKind, ASTType>
OverloadFitness::checkOneOperand(ASTExprAnd<AnyValue> operand,
                                 ValueInputConvention expectedConvention,
                                 ASTType expectedType,
                                 size_t &numImplicitConversions,
                                 bool &hasNonmaterializableConversion,
                                 bool allowImplicitConversions,
                                 ExprEmitter &emitter) {
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
    if (!argVal.getRValueType().isEqualCanon(
            expectedType.getReferenceElementType()))
      return {kWrongLVType, expectedType};
    break;
  }
  case ValueInputConvention::BorrowedInMem:
  case ValueInputConvention::OwnedInMem:
    // Ignore the pointer type on memory conventions when matching types.
    // Note: Should do not support overloading on borrow/owned currently,
    // but we could add this if there is a reason to.
    expectedType = expectedType.getReferenceElementType();
    [[fallthrough]];
  case ValueInputConvention::BorrowedInReg:
  case ValueInputConvention::OwnedInReg:
    // If the argument is an overload set, see if it can be resolve to the
    // right type.
    CValue argVal;
    if (auto orValue = operand.ir.getIfORValue()) {
      // Try to refine the ORValue into a PValue.
      argVal = orValue->emitAsPValue(&emitter, expectedType);
      if (!argVal)
        return {kWrongType, expectedType};
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
            argType.getNonmaterializableTarget(emitter.shared))
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

    // Argument name mismatches don't count as implicit conversions.
    auto expectedSig = dyn_cast<SignatureType>(expectedType.mlirType);
    auto argSig = dyn_cast<SignatureType>(argType.mlirType);
    if (expectedSig && argSig &&
        canZeroCostConvertSignature(expectedSig, argSig))
      break;

    // If we lack an exact match and conversions are disabled, this
    // candidate fails.
    if (!allowImplicitConversions || !emitter.canImplicitlyConvertToType(
                                         {argVal, operand.expr}, expectedType,
                                         /*allowArgNameCheck=*/false))
      return {kWrongType, expectedType};

    // If we had one, this bumps our # implicit conversions.
    ++numImplicitConversions;
    break;
  }

  return {kValidType, expectedType};
};

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `posOperands`.
OverloadFitness OverloadFitness::evaluate(LITSignatureType signature,
                                          const OverloadSet &callable,
                                          const CallOperands &callOperands,
                                          bool allowImplicitConversions,
                                          ExprEmitter &emitter) {
  // Before we do anything, we check if there were any unexpected keyword
  // operands passed. This keeps the subsequent code much simpler.

  // First, we collect all real argument names.
  StringSet<> argNames;
  for (auto [argIdx, argName] :
       llvm::enumerate(signature.getMetadata().getArgNames())) {
    if (argName.empty())
      continue; // Positional-only argument.
    if (signature.isVarArg(argIdx) || signature.isPackVarArg(argIdx))
      continue; // Variadic/pack args cannot be specified by keyword.
    auto [_, addedNew] = argNames.insert(argName);
    assert(addedNew && "duplicate argument name in signature");
  }

  // TODO(#21295): handle variadic keyword arguments.
  // Then we find all the keyword operands with unknown names.
  StringSet<> unknownKwOperands;
  if (callOperands.hasKwOperands()) {
    for (auto [name, operandVal] : *callOperands.kwOperands)
      if (!argNames.contains(name))
        unknownKwOperands.insert(name);
  }

  // We set up diagnostics.
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  size_t numPosOperands = posOperands.size();
  size_t numOperands = numPosOperands + callOperands.getNumKwOperands();
  SMLoc callLoc = callable.expr->getLoc();
  DiagEmitter emitDiagFor(callLoc, numOperands, callable.syntax, emitter);

  if (!unknownKwOperands.empty())
    return emitDiagFor.unexpectedKwArgs(unknownKwOperands);

  // Check that the signature can be rebound with this set of bindings.
  auto [newBindings, bindingFitness] =
      callable.inputParamBindings.verifyBindings(
          signature, emitter,
          [&](size_t index, Type type, ASTType expectedParamType,
              ArrayRef<TypedAttr> bindingsSoFar) -> PValue {
            return ParameterInferenceState(emitter.shared, index, type)
                .infer(signature, bindingsSoFar, callOperands);
          },
          /*isPackVarArg=*/signature.hasPackVarArgs() && !posOperands.empty());

  // If there is an error, return the problem.
  if (!newBindings) {
    ArrayRef<InputParamBindings::Binding> posBindings =
        callable.inputParamBindings.posBindings;
    if (auto expectedBinding = bindingFitness.expectedBinding) {
      auto &[paramIdx, expectedType] = *expectedBinding;
      return emitDiagFor.wrongParamType(posBindings[paramIdx], paramIdx,
                                        expectedType);
    }
    return emitDiagFor.wrongParamCount(signature.getNumInputParams(),
                                       posBindings.size(), "input");
  }

  // Check the result parameter count.
  if (size_t expectedNumResultParams = signature.getNumResultParams(),
      actualNumResultParams = callable.resultParams.size();
      expectedNumResultParams != actualNumResultParams) {
    return emitDiagFor.wrongParamCount(expectedNumResultParams,
                                       actualNumResultParams, "result");
  }

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  std::tie(signature, newBindings) =
      getUnboundSpecializedSignature(signature, newBindings);

  // Check that the result didn't bind to a type that would require changing to
  // a different result convention.
  for (Type outputType : signature.getValueResults())
    if (!ASTType(outputType).isRegisterPassable(callLoc, emitter.shared))
      return emitDiagFor.resultGenericMemType(outputType);

  // Ok, the parameters all line up, check the argument list.  We generally want
  // to diagnose problems where too few or too many arguments are passed if that
  // is the problem, rather than complaining about a type error of some argument
  // that doesn't work out.  Check for that first.
  auto [minRequiredArgs, maxAllowedArgs] = calculateMinMaxArgs(signature);
  if (numOperands < minRequiredArgs || maxAllowedArgs < numOperands) {
    return emitDiagFor.wrongArgCount(minRequiredArgs, maxAllowedArgs,
                                     numOperands);
  }

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

  // Use a ParserParamEvaluator to substitute 'apply' expressions in the
  // argument types.
  ParserParamEvaluator evaluator(emitter.getDeclResolver());
  for (auto [expectedArgIdx, unboundExpectedType, expectedConvention, argName] :
       llvm::enumerate(signature.getValueInputs(),
                       signature.getInputConventions(),
                       signature.getMetadata().getArgNames())) {
    assert(!signature.isKWVarArg(expectedArgIdx) &&
           "`**arg` variadics not supported yet");

    // Ignore the return slot if present.
    if (expectedConvention == ValueInputConvention::ByRefResult)
      continue;

    // If the arguments or results got bound to a memory-only type then their
    // argument convention needs to change.  We cannot support this until we get
    // proper type traits.  Note that the PointerType is considered a valid
    // register passable type, so things passed byref are ok.
    Type expectedType = evaluator.refineType(unboundExpectedType);
    if (!ASTType(expectedType).isRegisterPassable(callLoc, emitter.shared))
      return emitDiagFor.argGenericMemType(expectedArgIdx, expectedType);

    // Handle case when there are no more provided positional arguments.
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
        auto [kind, ty] = checkOneOperand(*kwOperandOr, expectedConvention,
                                          expectedType, numImplicitConversions,
                                          hasNonmaterializableConversion,
                                          allowImplicitConversions, emitter);
        if (kind != kValidType) {
          return emitDiagFor.argTypeMismatch(kind, ty, *kwOperandOr,
                                             expectedArgIdx);
        }
        continue;
      }

      // We don't need to provide value for this argument if it has a default
      // value.
      if (expectedArgIdx >=
          signature.getNumInputs() - signature.getDefaultArguments().size()) {
        // Arguments with default values must be followed only by other
        // arguments with default values, or by keyword argument.
        continue;
      }

      llvm_unreachable("argument had no corresponding operand");
    }

    /// Check and process a single positional operand and advance the operand
    /// index.
    auto processPositionalOperand =
        [&, expectedConvention = expectedConvention,
         newBindings = std::ref(newBindings)](
            ASTType expectedType) -> std::optional<InflightDiag> {
      ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx];
      auto [kind, ty] = checkOneOperand(
          operand, expectedConvention, expectedType, numImplicitConversions,
          hasNonmaterializableConversion, allowImplicitConversions, emitter);
      if (kind != kValidType)
        return emitDiagFor.argTypeMismatch(kind, ty, operand, posOperandIdx);
      ++posOperandIdx;
      return std::nullopt;
    };

    // If we have a varargs argument, then it will eat the rest of the
    // positional arguments, but we have to check each of them.
    if (signature.isVarArg(expectedArgIdx)) {
      auto varArgsEltType = ASTType(expectedType).getVariadicElementType();
      while (posOperandIdx != numPosOperands) {
        if (auto result = processPositionalOperand(varArgsEltType))
          return std::move(*result);
        passesVarArgArgument = true;
      }
      continue;
    }

    // If we have a pack type, it must have a known number of elements, and so
    // consume exactly that many positional operands.
    if (POP::PackType packType = getIfPackType(signature, expectedArgIdx)) {
      for (TypedAttr element : packType.getVariadicAttr().getValues()) {
        if (auto result = processPositionalOperand(ASTType(element)))
          return std::move(*result);
        passesVarArgArgument = true;
      }
      continue;
    }

    // Otherwise, we have an ordinary positional argument that is not varargs or
    // a pack. Ensure that it is not also passed as a keyword operand, then
    // process it as usual.
    if (!argName.empty())
      if (callOperands.findKwArg(argName))
        return emitDiagFor.redundantArg(expectedArgIdx, argName);
    if (auto result = processPositionalOperand(expectedType))
      return std::move(*result);
  }

  assert(posOperandIdx == numPosOperands &&
         "should handle argument mismatch above");

  // Otherwise we succeeded!  For our payload, indicate the number of implicit
  // conversions, whether there were (even more implicit) nonmaterializable
  // conversions, and whether anything was passed through varargs.  We consider
  // exact matches of concrete types to be more specific than varargs matches,
  // and both of these more specific than matches with variadic parameters.
  size_t payload = numImplicitConversions * 8;
  payload += (hasNonmaterializableConversion ? 4 : 0);
  payload += (passesVarArgArgument ? 2 : 0);
  payload += (bindingFitness.hasVariadicParams ? 1 : 0);
  return {newBindings, payload};
}

//===----------------------------------------------------------------------===//
// OverloadSet Implementation
//===----------------------------------------------------------------------===//

/// Get a symbol for a direct reference to the specified function in its
/// enclosing context.  This does not bind any values to arguments.
OverloadSet::OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                         ParameterExprArrayAttr bindingsAttr,
                         const ExprNode *expr, CallSyntax syntax)
    : baseName(baseName), fnDecls(fnDecls.begin(), fnDecls.end()), expr(expr),
      syntax(syntax) {
  if (bindingsAttr) {
    for (TypedAttr precheckedBinding : bindingsAttr)
      inputParamBindings.addPrechecked(precheckedBinding);
  }
}

OverloadSet::OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                         ParamBindArrayAttr bindingsAttr, const ExprNode *expr,
                         CallSyntax syntax)
    : baseName(baseName), fnDecls(fnDecls.begin(), fnDecls.end()), expr(expr),
      syntax(syntax) {
  if (bindingsAttr) {
    for (ParamBindAttr precheckedBinding : bindingsAttr)
      inputParamBindings.addPrechecked(precheckedBinding.getValue());
  }
}

// Assuming we have at least one valid candidate, filter the candidate list to
// the ones with the lowest number of implicit conversions required. If there is
// more than one candidate with minimal implicit conversions, we filter for the
// ones with the fewest number of parameter bindings. If there is still
// ambiguity, we filter for non-static methods.
//
// To aid downstream diganostics, the function returns the number of conversions
// needed for the best candidates, and a pointer to the fitness of one of them.
// All diagnostics from erroneous candidates are dropped.
static std::pair<const OverloadFitness *, size_t>
selectBestCandidates(ArrayRef<ASTDecl *> fnDecls,
                     MutableArrayRef<OverloadFitness> evaluations,
                     SmallVectorImpl<ASTDecl *> &newFnDecls) {
  assert(newFnDecls.empty());
  size_t minConversions = std::numeric_limits<size_t>::max();
  size_t minBindings = std::numeric_limits<size_t>::max();
  bool areTheBestCandidatesStatic = true;
  const OverloadFitness *oneFitness = &evaluations[0];
  for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
    // Ignore failures.
    if (!eval.isValid()) {
      eval.takeDiag().abandon();
      continue;
    }

    // Ignore candidates that have more conversions.
    size_t numConversions = eval.getNumImplicitConversions();
    if (numConversions > minConversions)
      continue;

    // Ignore candidates that have too many bindings.
    size_t numBindings = eval.getParamBindings().size();
    if ((numConversions == minConversions) && (numBindings > minBindings))
      continue;

    // If we found a new floor to the number of conversions needed, or a new
    // candidate with minimal conversions with a new floor for the number of
    // bindings, clear the list.
    if (numConversions < minConversions || numBindings < minBindings) {
      newFnDecls.clear();
      minConversions = numConversions;
      minBindings = numBindings;
      areTheBestCandidatesStatic = true;
    }

    auto func = cast<LIT::FuncOp>(*candidate);

    // If the current best candidates are not static, we ignore new static
    // candidates.
    if (!areTheBestCandidatesStatic && func.getIsStatic())
      continue;

    // If the current best candidates are static, and we just found a non-static
    // one, we clear the list.
    if (areTheBestCandidatesStatic && !func.getIsStatic()) {
      newFnDecls.clear();
      areTheBestCandidatesStatic = false;
    }

    newFnDecls.push_back(candidate);
    oneFitness = &eval;
  }

  // The numConversions value computed by OverloadFitness includes the number of
  // implicit conversions required but also uses the three lowest bits to track
  // whether a nonmaterializable conversion was needed, and if variadic
  // conversion was used in the parameters or arguments. Among other things,
  // this allows us to treat varargs as a less-specific match than an exact
  // signature match (for example, when overloading a `foo(Int)` and `foo(Int*)`
  // we should pick the former if both work). That said, we don't want to
  // complain about the wrong number in diagnostics, so we adjust for this.
  return {oneFitness, minConversions / 8};
}

PValue OverloadSet::filterOverloadSet(const CallOperands &operands,
                                      bool allowImplicitConversions,
                                      bool emitDiagnosticOnFailure,
                                      ExprEmitter &emitter) const {
  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<OverloadFitness> evaluations;
  bool anyValid = false;
  for (ASTDecl *candidate : fnDecls) {
    auto func = cast<LIT::FuncOp>(*candidate);

    // If we are dealing with a static method, we check if the operands include
    // a self operand and remove it, otherwise the signature might not match.
    CallOperands callOperands(operands);
    if (func.getIsStatic() && operands.hasSelfOperand)
      callOperands.posOperands = callOperands.posOperands.drop_front();

    evaluations.push_back(
        OverloadFitness::evaluate(func.getFullSignature(), *this, callOperands,
                                  allowImplicitConversions, emitter));
    anyValid |= evaluations.back().isValid();
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (!anyValid) {
    if (emitDiagnosticOnFailure) {
      // If there is a single callee, emit a specific error about the call.
      if (fnDecls.size() == 1) {
        auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
        auto diag = emitter.emitError(expr->getLoc(), "invalid call to '")
                    << baseName << "': " << expr->getRange()
                    << evaluations[0].takeDiag();
        diag.attachNote(fnDecl.getLoc()) << "function declared here";
        return {};
      }

      // Otherwise emit an error, and a note for what is wrong with each
      // candidate.
      auto diag =
          emitter.emitError(expr->getLoc(), "no matching function in call to '")
          << baseName << "': " << expr->getRange();
      for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
        auto fnDecl = cast<LIT::FuncOp>(*candidate);
        diag.attachNote(fnDecl->getLoc())
            << "candidate not viable: " << eval.takeDiag();
      }
      return {};
    }
    return {};
  }

  // Ok, we have at least one valid candidate, so filter for the best matches.
  SmallVector<ASTDecl *, 1> newFnDecls;
  auto [oneFitness, minConversions] =
      selectBestCandidates(fnDecls, evaluations, newFnDecls);

  // Notify the listener of the updated decl references for the call now that
  // invalid candidates have been filtered out.
  if (!newFnDecls.empty())
    emitter.shared.notifyListenerOnRef(newFnDecls, baseName, expr, syntax);

  // If we found exactly one viable candidate, or all the overloads are marked
  // as adaptive, then we succeed.
  auto allMarkedAdaptive = [&]() -> bool {
    return llvm::all_of(newFnDecls, [](ASTDecl *decl) {
      return cast<LIT::FuncOp>(*decl).getIsAdaptive();
    });
  };
  if (newFnDecls.size() == 1 || (!newFnDecls.empty() && allMarkedAdaptive())) {
    // On success, wrap things up into one callee.
    InputParamBindings newBindings;
    for (TypedAttr bind : oneFitness->getParamBindings())
      newBindings.addPrechecked(bind);
    return getCallee(newFnDecls, baseName, newBindings, expr, emitter);
  }

  // Otherwise, we have multiple viable candidates that are ambiguous because
  // they all require the same number of implicit conversions.
  if (emitDiagnosticOnFailure) {
    // We only want to suggest adding @adaptive if at least one in the set is
    // marked adaptive.
    bool anyMarkedAdaptive = llvm::any_of(newFnDecls, [](ASTDecl *decl) {
      return cast<LIT::FuncOp>(*decl).getIsAdaptive();
    });
    if (anyMarkedAdaptive) {
      auto diag = emitter.emitError(expr->getLoc(), "ambiguous call to '")
                  << baseName
                  << "', multiple implementations detected but not all are "
                     "marked adaptive, add @adaptive to all overloads"
                  << expr->getRange();
      for (LIT::FuncOp candidate : llvm::map_range(
               newFnDecls, [](ASTDecl *d) { return cast<LIT::FuncOp>(*d); })) {
        if (!candidate.getIsAdaptive())
          diag.attachNote(candidate.getLoc()) << "non-adaptive candidate here";
      }
    } else {
      auto diag = emitter.emitError(expr->getLoc(), "ambiguous call to '")
                  << baseName << "', each candidate requires " << minConversions
                  << " implicit conversion" << plural(minConversions)
                  << ", disambiguate with an explicit cast" << expr->getRange();
      for (ASTDecl *candidate : newFnDecls)
        diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
            << "candidate declared here";
    }
  }
  return {};
}

/// Filter down and complete this overload set based on knowledge that we need
/// to produce a function pointer with the specified type.
PValue OverloadSet::filterOverloadSetForValueType(ASTType functionType,
                                                  bool emitDiagnosticOnFailure,
                                                  ExprEmitter &emitter) const {
  // If the target type is something weird then don't filter.  Let the error be
  // reported another way.
  if (!isa<SignatureType>(functionType.mlirType)) {
    if (emitDiagnosticOnFailure) {
      auto diag = emitter.emitError(expr->getLoc())
                  << "cannot convert function to non-function type "
                  << functionType;
      for (ASTDecl *candidate : fnDecls)
        diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
            << "candidate declared here with type "
            << ASTType(cast<LIT::FuncOp>(*candidate).getFullSignature());
    }
    return {};
  }

  // TODO: This is using an exact match which is perhaps too specific of a
  // check.  We could do some amount of parameter inference to support cases
  // like:
  //
  //    fn foo[Type: mlirtype]() -> Type
  //    var f : ()-> Int = foo
  //
  // We could also support generating a lambda for fancy implicit conversions
  // and subtyping some day.
  auto getBindingsForSignature =
      [&](LITSignatureType candidateType) -> ParameterExprArrayAttr {
    // Apply any bound parameters to the candidate's type since they will be
    // applied when a reference is made.
    // TODO: Parameter inference.
    auto [newBindings, _] =
        inputParamBindings.verifyBindings(candidateType, emitter);
    return newBindings;
  };

  auto isValidCandidate = [&](LITSignatureType candidateType) -> bool {
    // Apply any bound parameters to the candidate's type since they will be
    // applied when a reference is made.  We only do this if there are some
    // bindings present, because (unlike normal function calls) the result type
    // may have unbound parameters that we are trying to match, e.g. when in a
    // parameter expression context.
    if (!inputParamBindings.posBindings.empty()) {
      auto newBindings = getBindingsForSignature(candidateType);
      if (!newBindings)
        return false; // If there is an error, return the problem.

      // If anything was bound, apply it to the signature so the expected
      // argument types are updated.
      if (!newBindings.empty()) {
        candidateType = candidateType.getSpecializedSignature(
            newBindings, [&]() -> InFlightDiagnostic {
              llvm_unreachable("bad bindings went undetected");
            });
      }
    }

    return functionType.isEqualCanon(candidateType) ||
           canZeroCostConvertSignature(
               cast<SignatureType>(functionType.mlirType), candidateType);
  };

  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<ASTDecl *> validCandidates;
  for (ASTDecl *candidate : fnDecls) {
    LITSignatureType candidateType =
        cast<LIT::FuncOp>(*candidate).getFullSignature();
    if (isValidCandidate(candidateType))
      validCandidates.push_back(candidate);
  }

  // Notify the listener of the updated decl references for the call now that
  // invalid candidates have been filtered out.
  if (!validCandidates.empty())
    emitter.shared.notifyListenerOnRef(validCandidates, baseName, expr, syntax);

  // If we have exactly one viable candidate, then we succeed.
  auto allMarkedAdaptive = [&]() -> bool {
    return llvm::all_of(validCandidates, [](ASTDecl *decl) {
      return cast<LIT::FuncOp>(*decl).getIsAdaptive();
    });
  };

  // If we resolved to a single candidate or an adaptive set, then we succeed.
  if (validCandidates.size() == 1 ||
      (!validCandidates.empty() && allMarkedAdaptive())) {
    if (inputParamBindings.posBindings.empty())
      return getCallee(validCandidates, baseName, inputParamBindings, expr,
                       emitter);

    auto candidateType = cast<LIT::FuncOp>(*fnDecls.front()).getFullSignature();

    InputParamBindings newBindings;
    for (TypedAttr bind : getBindingsForSignature(candidateType))
      newBindings.addPrechecked(bind);
    return getCallee(validCandidates, baseName, newBindings, expr, emitter);
  }

  // If we aren't to emit a diagnostic, just return the failure.
  if (!emitDiagnosticOnFailure)
    return {};

  auto diag = emitter.emitError(expr->getLoc());
  if (validCandidates.empty()) {
    diag << "no '" << baseName << "' candidates have type " << functionType
         << expr->getRange();
  } else {
    diag << "ambiguous use of '" << baseName << "' as type " << functionType
         << expr->getRange();
  }

  for (ASTDecl *candidate : fnDecls)
    diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
        << "candidate declared here with type "
        << ASTType(cast<LIT::FuncOp>(*candidate).getFullSignature());

  return {};
}

/// Utility function to perform substitutions of the specified callable bindings
/// into the symbol for the given function declaration. It returns the resultant
/// SymbolConstantAttr or produces an error message and returns null.
static TypedAttr
getBoundConstAttrFor(LIT::FuncOp funcOp, StringRef baseName,
                     const InputParamBindings &inputParamBindings,
                     const ExprNode *expr, ExprEmitter &emitter) {

  // If there are no input parameters specified and if we allow unbound
  // symbols, just return the unbound symbol.
  if (inputParamBindings.posBindings.empty())
    return funcOp.getBoundReference();

  // Check that the signature can be rebound with our set of bindings.
  LITSignatureType signature = funcOp.getFullSignature();
  ParameterExprArrayAttr newBindings = inputParamBindings.verifyBindings(
      signature, emitter, baseName, funcOp.getLoc(), expr->getLoc());
  if (!newBindings)
    return {};

  // Now that we checked the types match, form the binding.
  return funcOp.getBoundReference(newBindings);
}

/// Perform substitutions of the specified bindings into the symbol, returning,
/// in symConstAttrs, the resultant SymbolConstant attr for each adaptive
/// function overload. On failure it produces an error message and returns null.
static VariadicAttr getAdaptiveSet(ArrayRef<ASTDecl *> fnDecls,
                                   StringRef baseName,
                                   const InputParamBindings &inputParamBindings,
                                   const ExprNode *expr, ExprEmitter &emitter) {
  SmallVector<TypedAttr> symConstAttrs;
  for (ASTDecl *fnDecl : fnDecls) {
    auto funcOp = cast<LIT::FuncOp>(*fnDecl);
    if (!funcOp.getIsAdaptive()) {
      auto diag = emitter.emitError(expr->getLoc(),
                                    "cannot form a reference to non @adaptive "
                                    "declaration of '")
                  << baseName << "'" << expr->getRange();
      diag.attachNote(funcOp.getLoc()) << "declared here";
      return {};
    }
    TypedAttr symbolAttr = getBoundConstAttrFor(
        funcOp, baseName, inputParamBindings, expr, emitter);
    if (!symbolAttr)
      return {};
    symConstAttrs.push_back(symbolAttr);
  }

  return VariadicAttr::get(emitter.getContext(), symConstAttrs,
                           VariadicType::get(symConstAttrs.front().getType()));
}

/// Resolve the callee into either a single PValue callee (if there's only one
/// decl provided) or a variadic that contains all the possible adaptive
/// overloads.
PValue OverloadSet::getAdaptiveSet(ExprEmitter &emitter) {
  return ::getAdaptiveSet(fnDecls, baseName, inputParamBindings, expr, emitter);
}

/// Resolve the callee into either a single PValue callee (if there's only one
/// decl provided) or a variadic that contains all the possible adaptive
/// overloads. Because adaptive overloads must all have the same signature, this
/// also returns the signature type that they all share.
PValue OverloadSet::getCallee(ArrayRef<ASTDecl *> fnDecls, StringRef baseName,
                              const InputParamBindings &inputParamBindings,
                              const ExprNode *expr, ExprEmitter &emitter) {
  assert(!fnDecls.empty() &&
         "cannot get the callee when no callees have been resolved");
  if (fnDecls.size() == 1) {
    auto funcOp = cast<LIT::FuncOp>(*fnDecls.front());
    return getBoundConstAttrFor(funcOp, baseName, inputParamBindings, expr,
                                emitter);
  }

  VariadicAttr variadicSetAttr =
      ::getAdaptiveSet(fnDecls, baseName, inputParamBindings, expr, emitter);
  if (!variadicSetAttr)
    return {};

  // If the callee is a list, create a param.fork op and create a
  // CallParam on that. Mangle the declared parameter name with the line and
  // column number to ensure uniqueness.
  auto [line, col] = emitter.getSourceMgr().getLineAndColumn(expr->getLoc());

  if (!emitter.builder)
    return emitter.emitErrorForDynamicValueInParameter(
        expr, "TODO: cannot call adaptive function in parameter contexts");

  StringRef name = cast<LIT::FuncOp>(*fnDecls.front()).getName();
  StringAttr declName = emitter.builder->getStringAttr(
      Twine("(adaptive)") + name + Twine(line) + "_" + Twine(col));
  auto decl = ParamDeclAttr::get(declName,
                                 variadicSetAttr.getType().getElementAsType());
  emitter.builder->create<ParamForkOp>(
      emitter.translateLocation(expr->getLoc()), decl, variadicSetAttr);
  return PValue(ParamDeclRefAttr::get(decl));
}

/// Perform substitutions of the specified bindings into the symbol, returning
/// the resultant LITSymbolConstant attr or producing an error message and
/// returning null. This allows producing a reference to a parameterized
/// function without the parameters specified.  They can be bound later.
TypedAttr OverloadSet::getBoundConstantAttr(ExprEmitter &emitter) const {
  if (fnDecls.size() != 1) {
    assert(!fnDecls.empty() && "DirectCallable malformed");
    auto diag = emitter.emitError(
                    expr->getLoc(),
                    "cannot form a reference to overloaded declaration of '")
                << baseName << "'" << expr->getRange();
    for (ASTDecl *candidate : fnDecls) {
      auto funcOp = cast<LIT::FuncOp>(*candidate);
      diag.attachNote(funcOp.getLoc()) << "candidate declared here";
    }

    return {};
  }

  return getBoundConstAttrFor(cast<LIT::FuncOp>(*fnDecls[0]), baseName,
                              inputParamBindings, expr, emitter);
}

/// Get a OverloadSet for a lookup of a named method on the specified type.
/// If successful, this provides a non-null OverloadSet.
///
/// On failure, this returns a null OverloadSet and invokes errorHandler if
/// the problem hasn't already been diagnosed. This does not emit an error on
/// failure.
OverloadSet::OverloadSet(ASTType type, StringRef methodName,
                         const ExprNode *expr, CallSyntax syntax,
                         SharedState &shared, function_ref<void()> errorHandler)
    : expr(expr), syntax(syntax) {

  // If this is a previously-reported error, ignore and don't report an
  // additional error.
  if (type.isTypeCheckErrorType())
    return;

  SMLoc callLoc = expr->getLoc();

  // First perform a lookup to see if there are any candidates.
  auto lookupResult = shared.lookupAndResolveDecl(methodName, callLoc, type,
                                                  /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
  if (resultDecls.empty()) {
    if (!lookupResult.isErroneous() && errorHandler) // Already diagnosed?
      errorHandler();
    return;
  }

  // If we find a vardecl or any other thing, then fail because it cannot be
  // called.
  if (!isa<LIT::FuncOp>(*resultDecls[0]))
    return;

  // Handle method references, which might be overloaded.
  SmallVector<TypedAttr> parentBindings;
  for (ParamBindAttr binding : type.getParamBindings())
    parentBindings.push_back(binding.getValue());
  *this = OverloadSet(
      methodName, resultDecls,
      ParameterExprArrayAttr::get(shared.getContext(), parentBindings), expr,
      syntax);
}

/// Lookup of a named named method on the specified type, filtered to match a
/// concrete operand set. If successful, this provides a non-null PValue for a
/// single callee.
PValue OverloadSet::lookup(ASTType type, StringRef methodName,
                           const CallOperands &callOperands,
                           const ExprNode *callExpr, CallSyntax syntax,
                           ExprEmitter &emitter,
                           function_ref<void()> errorHandler) {
  ASTType nmTarget = type.getNonmaterializableTarget(emitter.shared);
  bool shouldPrintError = bool(errorHandler);
  auto doLookup = [&](ASTType type, bool shouldPrintError) -> PValue {
    OverloadSet ovSet(type, methodName, callExpr, syntax, emitter.shared,
                      errorHandler);

    // If the core lookup failed, don't filter.
    if (ovSet.isNull())
      return {};

    // Filter the overload set with the actual operands list.  If this
    // fails, report an error (if we have an error handler) and reset to a
    // null state so the client can check this.
    return ovSet.filterOverloadSet(
        callOperands, /*allowImplicitConversions=*/true,
        /*emitDiagnosticOnFailure=*/shouldPrintError, emitter);
  };

  // If there is a nonmaterializableTarget, try using the original type first,
  // then falling back on the target.
  if (nmTarget) {
    PValue ret = doLookup(type, false);
    if (ret)
      return ret;
    type = nmTarget;
  }
  return doLookup(type, shouldPrintError);
}

PValue OverloadSet::getDirectSymbol(ExprEmitter *emitter,
                                    ASTType expectedType) const {
  // Verify that the target has no result parameters.  We have no way to bind
  // these indirectly.
  if (!resultParams.empty()) {
    if (emitter) {
      emitter->emitError(
          expr->getLoc(),
          "calls with result parameter bindings must be called directly")
          << expr->getRange();
    }
    return {};
  }

  // Handle the case of a single candidate.
  if (fnDecls.size() == 1) {
    // This is an unbound function. Just return a reference.
    if (inputParamBindings.posBindings.empty())
      return cast<LIT::FuncOp>(*fnDecls.front()).getBoundReference();
    if (!emitter)
      return {};
    // Bind the parameters.
    return getBoundConstantAttr(*emitter);
  }

  // With an emitter and an expected type, the overload set can definitely be
  // resolved to a single candidate or not.
  if (expectedType && emitter) {
    return filterOverloadSetForValueType(
        expectedType, /*emitDiagnosticOnFailure=*/true, *emitter);
  }
  // Otherwise, emit the bind if possible.
  if (!emitter)
    return {};
  return getBoundConstantAttr(*emitter);
}

PValue OverloadSet::emitAsPValue(ExprEmitter *emitter,
                                 ASTType expectedType) const {
  // Overload sets with base values cannot be emitted as PValues since they
  // depend on a dynamic value.
  // TODO: A conversion can be emitted if the base value is a PValue.
  if (baseValue)
    return {};

  return getDirectSymbol(emitter, expectedType);
}

/// Emit this as a CRValue if it can be resolved, otherwise emit an ambiguity
/// error and return null.
CValue OverloadSet::emitAsCValue(ExprEmitter &emitter, ValueDest &dest) {
  // If we have an overload set with multiple possibilities, we'll fail to emit
  // this as a CRValue.  Try to resolve it based on the destination's type.
  ASTType expectedType;
  if (fnDecls.size() > 1) {
    expectedType = dest.resolveImpliedType(expr->getLoc(),
                                           /*no implied type*/ Type(), emitter);
  }

  // We allow unbound symbols here which can be emitted as an PValue.  In the
  // case where we are partially applying, that will force the unbound symbol
  // into a SRValue which will catch symbols that are not fully bound.
  PValue directSymbolAttr = getDirectSymbol(&emitter, expectedType);
  if (!directSymbolAttr)
    return {};

  // If we have no base value, then we are just a symbol, return it.
  if (!baseValue)
    return emitter.emitCResult(directSymbolAttr, expr, dest);

  auto loc = baseValue.expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  auto calleeSignature =
      cast<SignatureType>(directSymbolAttr.getType().mlirType);
  Type firstArgIRType = calleeSignature.getValueInputs()[0];
  ValueInputConvention selfConvention = calleeSignature.getInputConvention(0);
  Value firstArgValue;

  assert(!calleeSignature.isVarArg(0) && !calleeSignature.isKWVarArg(0) &&
         "Error: self shouldn't be varargs");

  switch (selfConvention) {
  case ValueInputConvention::ByRefResult:
  case ValueInputConvention::OwnedInMem:
  case ValueInputConvention::BorrowedInMem: {
    auto diag =
        emitter.emitError(
            loc, "TODO: partial application requires closure generation ")
        << baseValue.expr->getRange();
    if (auto cValue = baseValue.ir.getIfCValue())
      diag << cValue.getRValueType();
    return {};
  }

  case ValueInputConvention::ByRef:
  case ValueInputConvention::InitSelf: {
    LValue baseLV = emitter.emitLValue(baseValue, ValueDest::none());
    if (!baseLV)
      return {};

    // Using partial application over an lvalue isn't safe until we support an
    // ownership models with mutable borrows.
    emitter.emitError(loc, "TODO: partial application to mutable base isn't "
                           "supportable without a lifetime model")
        << baseValue.expr->getRange();
    return {};
  }
  case ValueInputConvention::BorrowedInReg:
  case ValueInputConvention::OwnedInReg:
    // Otherwise we can have either an lvalue or rvalue, but we need to convert
    // to an rvalue if we have an lvalue.
    firstArgValue = emitter.emitSRValue(baseValue, EC_CallArgValue);
    if (!firstArgValue)
      return {};

    // TODO: Partial application isn't handling ownership right at all, we
    // should probably disable it.
    break;
  }

  assert(firstArgIRType == firstArgValue.getType() &&
         "base types should always structurally line up");

  // Partial apply wants to know what operands to bind, we always bind the first
  // one.
  auto result = SRValue(emitter.builder->create<CreateClosureOp>(
      expr->getLocation(emitter), directSymbolAttr, firstArgValue));
  return emitter.emitCResult(result, expr, dest);
}

//===----------------------------------------------------------------------===//
// Call Emission Implementation
//===----------------------------------------------------------------------===//

/// Emit a function call to the specified callee with the specified operand
/// values.  This emits an error and returns null on failure.
CValue OverloadSet::emitCall(const CallOperands &callOperands, ValueDest &dest,
                             ExprEmitter &emitter) {
  if (isNull()) // Base was already diagnosed as an error.
    return {};

  // Used in some cases below, lifetime needs to exist for this whole method.
  SmallVector<ASTExprAnd<AnyValue>> posOperandsWithSelf;

  // If we have a bound self, add it to the operand list to simplify the logic
  // below.
  CallOperands operands = callOperands;
  if (baseValue) {
    ArrayRef<ASTExprAnd<AnyValue>> posOperands = operands.posOperands;
    posOperandsWithSelf.reserve(posOperands.size() + 1);
    posOperandsWithSelf.push_back(baseValue);
    posOperandsWithSelf.append(posOperands.begin(), posOperands.end());
    baseValue = {};
    assert(syntax == CallSyntax::kMethodCall && "Unexpected syntax form");
    operands.posOperands = posOperandsWithSelf;
    operands.hasSelfOperand = true;
  }

  // Check the direct callees to see if they can be unambiguously resolved
  // with the bindings list and specified arguments.
  PValue callee = filterOverloadSet(operands,
                                    /*allowImplicitConversions=*/true,
                                    /*emitDiagnosticOnFailure=*/true, emitter);
  if (!callee)
    return {};

  SignatureType calleeSig = cast<SignatureType>(callee.getType().mlirType);

  // Check declarations for the result parameters and collect them here.
  assert(calleeSig.getNumResultParams() == resultParams.size() &&
         "We know that the callee is type checked");

  // Verify completion of forward declared alias declarations.  We know the
  // decl exists, but we don't know if the type is compatible or it has been
  // multiply defined.
  //
  // TODO: We don't remap input parameters types into output parameter types.
  // We surely handle this wrong: `fn x[a: type -> a]():` for example.
  SmallVector<ParamDeclAttr> resultParamDecls;
  for (auto [type, declAndLoc] :
       llvm::zip(calleeSig.getResultParamTypes(), resultParams)) {
    auto forwardDecl = cast<AliasForwardDeclOp>(*declAndLoc.first);

    // Verify the types match.
    // TODO: Move this to overload resolution.
    if (!ASTType(forwardDecl.getType()).isEqualCanon(type)) {
      auto diag =
          emitter.emitError(declAndLoc.second, "result parameter returns type ")
          << type << " but forward declaration is of type "
          << ASTType(forwardDecl.getType());
      diag.attachNote(forwardDecl.getLoc()) << "alias forward declared here";
      return {};
    }
    resultParamDecls.push_back(ParamDeclAttr::get(forwardDecl.getName(), type));
  }

  // Calls in parameter context cannot have result parameters.
  if (!emitter.builder && !calleeSig.getResultParamTypes().empty()) {
    auto diag =
        emitter.emitError(expr->getLoc(), "cannot call '")
        << baseName
        << "' in parameter expression because it has a parameter result";
    for (auto &resultParam : resultParams) {
      diag << SourceRange(resultParam.second, resultParam.second);
      resultParam.first->hasReferenceError = true;
    }
    return {};
  }

  return emitter.emitCallUnchecked(callee, operands, resultParamDecls, dest,
                                   expr);
}

CValue ExprEmitter::emitIndirectCall(CValue callee,
                                     const CallOperands &callOperands,
                                     ValueDest &dest,
                                     const ExprNode *callExpr) {
  auto calleeSig = dyn_cast<SignatureType>(callee.getRValueType().mlirType);
  if (!calleeSig) {
    // If we are invoking something other than a SignatureType, try to invoke
    // its `__call__` method.
    SmallVector<ASTExprAnd<AnyValue>> posOperandsWithCallee;
    posOperandsWithCallee.push_back({callee, callExpr});
    llvm::append_range(posOperandsWithCallee, callOperands.posOperands);
    return emitNamedMethodCall(
        "__call__",
        CallOperands(posOperandsWithCallee, callOperands.kwOperands), dest,
        CallSyntax::kDirectCall, callExpr);
  }

  if (calleeSig.getNumResultParams()) {
    emitError(callExpr->getLoc(), "invalid indirect call: callee has ")
        << calleeSig.getNumResultParams() << " unbound result parameter"
        << plural(calleeSig.getNumResultParams()) << callExpr->getRange();
    return {};
  }

  // If we have a function pointer, resolve it to an RValue.
  CRValue calleeRV = emitCRValue({callee, callExpr}, EC_CallCalleeValue);
  if (!calleeRV)
    return {};

  // Check to see if we can apply these operands to the callee signature.
  OverloadSet bindings{"callee", /*fnDecls=*/{}, ParamBindArrayAttr(), callExpr,
                       CallSyntax::kIndirectCall};
  auto fitness =
      OverloadFitness::evaluate(calleeSig, bindings, callOperands,
                                /*allowImplicitConversions=*/true, *this);
  if (!fitness.isValid()) {
    // If not, diagnose it with an error.
    emitError(callExpr->getLoc(), "invalid indirect call: ")
        << fitness.takeDiag();
    return {};
  }

  return emitCallUnchecked(calleeRV, callOperands,
                           /*resultParams=*/{}, dest, callExpr);
}

CValue ExprEmitter::emitNamedMethodCall(StringRef methodName,
                                        const CallOperands &callOperands,
                                        ValueDest &dest, CallSyntax syntax,
                                        const ExprNode *callNode) {
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  assert(!posOperands.empty() &&
         "Cannot emit a method call without a receiver!");

  // Emit the first/self operand to a CValue so we can figure out which type to
  // lookup on.
  CValue selfVal = posOperands[0].ir.getIfCValue();
  SmallVector<ASTExprAnd<AnyValue>> updatedPosOperands;
  if (!selfVal) {
    selfVal = emitCValue(posOperands[0], ValueDest::none());
    if (!selfVal)
      return {};
    // We can't mutate posOperands because it's an ArrayRef.  If something
    // changed, recurse with a temporary buffer.
    updatedPosOperands.append(posOperands.begin(), posOperands.end());
    updatedPosOperands[0].ir = selfVal;
    posOperands = updatedPosOperands;
  }

  CallOperands operands(posOperands, callOperands.kwOperands);

  ASTType type = selfVal.getRValueType();

  auto emitNoMethodError = [&]() {
    auto diag = emitError(callNode->getLoc(), "")
                << type << " does not implement the '" << methodName
                << "' method";
    switch (syntax) {
    case CallSyntax::kMethodCall:
      [[fallthrough]];
    case CallSyntax::kOperator:
      diag << posOperands[0].expr->getRange();
      break;
    case CallSyntax::kReversedOperator:
      diag << posOperands[1].expr->getRange();
      break;
    default:
      break;
    }
  };

  PValue callee = {};
  if (ASTType nmTarget = type.getNonmaterializableTarget(shared)) {
    // If the type doesn't have the specified method, but it's
    // nonmaterializable, give it a second chance with the materialized type.
    // If the type doesn't have the specified method, emit an error.
    callee = OverloadSet::lookup(type, methodName, operands, callNode, syntax,
                                 *this);
    if (!callee) {
      CValue convertedSelf = emitConstructorCall(
          nmTarget, CallOperands({{selfVal, posOperands[0].expr}}), callNode,
          CallSyntax::kImplicitConvert, ValueDest::none(),
          /*allowImplicitConversion=*/true);
      if (!convertedSelf)
        return {};
      updatedPosOperands.clear();
      updatedPosOperands.append(posOperands.begin(), posOperands.end());
      updatedPosOperands[0].ir = convertedSelf;
      posOperands = updatedPosOperands;
      type = nmTarget;
    }
  }

  // If the type doesn't have the specified method, emit an error.
  if (!callee)
    callee = OverloadSet::lookup(type, methodName, operands, callNode, syntax,
                                 *this, emitNoMethodError);
  if (!callee)
    return {};

  return emitIndirectCall(callee, operands, dest, callNode);
}

CValue ExprEmitter::emitConstructorCall(ASTType type,
                                        const CallOperands &callOperands,
                                        const ExprNode *expr, CallSyntax syntax,
                                        ValueDest &dest,
                                        bool allowImplicitConversion) {
  // If the dest type is invalid, then an error has already been reported.
  if (type.isTypeCheckErrorType())
    return {};

  // Check to see if we can invoke an __init__ method to convert it.
  OverloadSet callee(type, "__init__", expr, syntax, shared);
  return emitConstructorCall(type, callee, callOperands, expr, syntax, dest,
                             allowImplicitConversion);
}

CValue ExprEmitter::emitConstructorCall(ASTType type, const OverloadSet &callee,
                                        const CallOperands &callOperands,
                                        const ExprNode *expr, CallSyntax syntax,
                                        ValueDest &dest,
                                        bool allowImplicitConversion) {
  // Init for memory-only types get their self argument implicitly initialized
  // and passed in as the first argument.
  ArrayRef<ASTExprAnd<AnyValue>> origPosOperands = callOperands.posOperands;
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = origPosOperands;
  CallOperands operands = callOperands;
  bool isMemoryOnly = !type.isRegisterPassable(expr->getLoc(), shared);
  SmallVector<ASTExprAnd<AnyValue>> posOperandsWithSelf;
  auto argsAddSelf = [&]() {
    posOperandsWithSelf.clear();
    if (isMemoryOnly) {
      posOperandsWithSelf.reserve(posOperands.size() + 1);

      // Unfortunately, we can't just use 'type' or the dest LValue as the
      // buffer to initialize, because the concrete result type might need
      // parameters to be inferred, and those may depend on other value
      // arguments.  Handle this by setting up a placeholder with the type
      // we know so far, and use that to filter the overload set.
      auto attr = UnknownAttr::get(PointerType::get(type));
      posOperandsWithSelf.push_back({PValue(attr), expr});
      posOperandsWithSelf.append(posOperands.begin(), posOperands.end());
      operands.posOperands = posOperandsWithSelf;
      operands.hasSelfOperand = true;
    }
  };
  argsAddSelf();

  // Try to resolve the overload set to exactly one candidate, but don't emit an
  // error on failure (we typically want to customize the error).
  PValue calleeFn =
      callee.filterOverloadSet(operands, allowImplicitConversion,
                               /*emitDiagnosticOnFailure=*/false, *this);

  ASTType operandType;
  if (callOperands.posOperands.size() == 1 &&
      callOperands.posOperands[0].ir.getIfCValue()) {
    operandType = callOperands.posOperands[0].ir.getIfCValue().getRValueType();
  }

  CValue autoNonmaterializableConversion;
  SmallVector<ASTExprAnd<AnyValue>> autoConvertedArgs;
  if (!calleeFn) {
    // If we are converting from a nonmaterializable struct, always allow an
    // extra implicit conversion to the nonmaterializable target.  Then try
    // again to find a constructor.
    if (ASTType nonmaterializableTarget =
            operandType.getNonmaterializableTarget(shared)) {
      if (!nonmaterializableTarget.isEqualCanon(type)) {
        ValueDest autoDest(nonmaterializableTarget, EC_CallArgValue);
        autoNonmaterializableConversion = emitConstructorCall(
            nonmaterializableTarget, origPosOperands, origPosOperands[0].expr,
            syntax, autoDest, /*allowImplicitConversion=*/false);
        autoConvertedArgs.push_back(
            {autoNonmaterializableConversion, origPosOperands[0].expr});
        operands.posOperands = autoConvertedArgs;
        argsAddSelf();
        calleeFn =
            callee.filterOverloadSet(operands, allowImplicitConversion,
                                     /*emitDiagnosticOnFailure=*/false, *this);
      }
    }
  }

  if (!calleeFn) {
    // If we failed to resolve the set, then try to emit a tailored error.  If
    // constructing from one value, then this is a type conversion (either
    // implicit or explicit).
    if (operandType) {
      // Reject Int(x) where x is already an Int with an error + fixit.
      if (syntax == CallSyntax::kTypeCall && operandType.isEqualCanon(type) &&
          isa<CallNode>(expr)) {
        const CallNode &callNode = *cast<CallNode>(expr);
        // This removes the constructor call, but does not remove the parens
        // because we don't want to introduce precedence problems.
        emitError(expr->getLoc())
            << "cannot construct " << type
            << " with itself, you can remove the constructor call"
            << posOperands[0].expr->getRange()
            << FixIt::remove(callNode.callee->getRange());
        return {};
      }

      if (syntax != CallSyntax::kImplicitConvert) {
        emitError(expr->getLoc())
            << "cannot construct " << type << " from " << operandType
            << " value" << getContextMessage(dest.getContext())
            << expr->getRange();
        return {};
      }

      // Handle common type mismatches with a tailored error.
      if (dest.getContext() == EC_CallParamValue ||
          dest.getContext() == EC_CallArgValue) {
        auto diag = emitError(expr->getLoc())
                    << "cannot pass " << operandType << " value, "
                    << ((dest.getContext() == EC_CallParamValue) ? "parameter"
                                                                 : "argument")
                    << " expected " << type << expr->getRange();
        return {};
      }

      emitError(expr->getLoc())
          << "cannot implicitly convert " << operandType << " value to " << type
          << getContextMessage(dest.getContext()) << expr->getRange();
      return {};
    }

    // If the type has no candidates, complain about that.
    if (callee.isNull()) {
      if (!type.getDecl(shared)) {
        emitError(expr->getLoc(), "MLIR type ")
            << type
            << " must be created with an MLIR operation, not constructor "
               "syntax"
            << getContextMessage(dest.getContext()) << expr->getRange();
        return {};
      }

      emitError(expr->getLoc(), "")
          << type << " does not implement any '__init__' methods"
          << getContextMessage(dest.getContext()) << expr->getRange();
      return {};
    }

    // Otherwise, do it again to emit a generic overload set error.
    calleeFn =
        callee.filterOverloadSet(operands, allowImplicitConversion,
                                 /*emitDiagnosticOnFailure=*/true, *this);
    assert(!calleeFn && "This should fail if it failed before");
    return {};
  }

  // If we successfully resolve the overload set, we know the call will succeed,
  // do it. Register-passable and parameter constructor calls do not require
  // result slot allocation.
  if (!isMemoryOnly)
    return emitCallUnchecked(calleeFn, operands, {}, dest, expr);
  if (!builder) {
    operands = callOperands;
    return emitCallUnchecked(calleeFn, operands, {}, dest, expr);
  }

  // We need to invoke memory-only constructors specially since the buffer is
  // exposed.
  auto calleeSig = cast<SignatureType>(calleeFn.getType().mlirType);
  auto firstArgRVType =
      ASTType(calleeSig.getValueInputs()[0]).getReferenceElementType();

  // For a memory-only call, we need to replace the destination buffer with the
  // actual destination lvalue to use.
  MLValue destMLValue =
      dest.getMLValueForResult(expr->getLoc(), firstArgRVType, *this);
  posOperandsWithSelf[0].ir = destMLValue;
  if (!destMLValue)
    return {};

  // Emit the call, but not into 'dest', typically init will return None.
  CValue result = emitIndirectCall(calleeFn, operands, ValueDest::none(), expr);
  if (!result)
    return {};

  // Now that we've emitted the result into the result buffer, emit a conversion
  // if the expected type and the actual type differ.  This can happen when the
  // ValueDest isn't the same as the result, e.g. "var x: MemFloat = MemInt()".
  return emitCResult(MRValue(destMLValue), expr, dest);
}
