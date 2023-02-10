//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#include "LitExprCalls.h"
#include "ASTDecl.h"
#include "LitExprEmitter.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "LITEXPRCALLS"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given the MLIR type for a variadic argument, return the element type as an
/// MLIR type.
static Type getVariadicElementType(Type variadicType) {
  auto mValue = MValue(cast<KGEN::VariadicType>(variadicType).getElementType());
  // KGEN::VariadicType allows arbitrary parameter expressions, but we only ever
  // use concrete types for variadic syntax.
  assert(mValue.getIfTypeValue() &&
         "variadic convention never has parameteric element");
  return mValue.getIfTypeValue();
}

//===----------------------------------------------------------------------===//
// Parameter Inference Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParameterInferenceState {
public:
  ParameterInferenceState(ParamDeclAttr decl) : parameterName(decl.getName()) {}

  /// Given an incomplete parameter binding set for a call to the specified
  /// signature, try to infer the value of the next 'decl' parameter.  This
  /// should always return null /without/ an error if it cannot be inferred, and
  /// return a specific value if unambiguously determined.
  MValue infer(SignatureType signature, ArrayRef<ParamBindAttr> bindingsSoFar,
               ArrayRef<ASTExprAnd<AnyValue>> operands);

private:
  LogicalResult matchTypes(Type actualType, Type expectedType);
  LogicalResult matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);

  StringAttr parameterName;
  SmallVector<MValue> inferredValues;
};
} // namespace

LogicalResult ParameterInferenceState::matchTypes(Type actualType,
                                                  Type expectedType) {
  // If the expected type is a parameter ref, then we're binding the specified
  // type to an attribute parameter.
  if (auto expectedParamRef = dyn_cast<ParamRefType>(expectedType))
    return matchParams(ParameterizedTypeConstantAttr::get(actualType),
                       expectedParamRef.getParam());

  // Handle when both are DeclRefTypes.
  if (auto actualDRT = dyn_cast<DeclRefType>(actualType))
    if (auto expectedDRT = dyn_cast<DeclRefType>(expectedType)) {
      // Fail if this is to two fundamentally different symbols.
      if (actualDRT.getSymbol() != expectedDRT.getSymbol())
        return failure();

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

  // Handle POP::PointerType.
  if (auto actual = dyn_cast<POP::PointerType>(actualType))
    if (auto expected = dyn_cast<POP::PointerType>(expectedType))
      return matchParams(actual.getElementType(), expected.getElementType());

  // If the types trivial match then we're done and there is no inference to do.
  if (actualType == expectedType)
    return success();

  // TODO: Could do StructType and VariantType?
  LLVM_DEBUG(llvm::errs() << "CANNOT INFER MISMATCH TYPES:\n";
             actualType.dump(); expectedType.dump(); parameterName.dump());
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
  if (auto dre = dyn_cast<ParamDeclRefAttr>(expectedAttr)) {
    // If the name mismatches, then it is some other parameter, assume it is
    // fine.
    if (dre.getName() == parameterName)
      inferredValues.push_back(actualAttr);
    return success();
  }

  // If the attrs trivial match then we're done and there is no inference to do.
  if (actualAttr == expectedAttr)
    return success();

  LLVM_DEBUG(llvm::errs() << "CANNOT INFER MISMATCHING ATTRS:\n";
             actualAttr.dump(); expectedAttr.dump(); parameterName.dump());
  return success();
}

/// Given an incomplete parameter binding set for a call to the specified
/// signature, try to infer the value of the next 'decl' parameter.  This should
/// always return null /without/ an error if it cannot be inferred, and return
/// a specific value if unambiguously determined.
///
MValue ParameterInferenceState::infer(SignatureType signature,
                                      ArrayRef<ParamBindAttr> bindingsSoFar,
                                      ArrayRef<ASTExprAnd<AnyValue>> operands) {
  // TODO: Apply the bindings so far (plus a distinct new attribute relating
  // back to the original decls for ones that are missing) to the signature with
  // getSpecializedSignature so we benefit from the already-fixed subsitutions
  // being applied to the input types.  This can make them more concrete and
  // help with inferring dependent types based on already-bound parameters.
  //
  // signature = signature.getSpecializedSignature(bindingsSoFar + placeholders)

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t providedValueIdx = 0;
  for (auto [expectedArgIdx, expectedType] :
       llvm::enumerate(signature.getValueInputs())) {
    auto expectedConvention = signature.getInputConvention(expectedArgIdx);

    // Handle case when there are no more provided arguments.
    if (providedValueIdx == operands.size()) {
      // If the argument is a varargs argument list, then it can be initialized
      // with zero values no problem.
      if (uint8_t(expectedConvention & ValueInputConvention::VarArg))
        break;

      // TODO: If this argument is defaulted, infer against it.

      // Otherwise we have an argument count mismatch, just fail.
      return {};
    }

    // Otherwise we'll check the expected type against one (or more in the case
    // of varargs) provided values.
    auto checkOneOperand = [&](ASTType expectedType) -> LogicalResult {
      // We'll bind the next provided value.
      auto operand = operands[providedValueIdx++];
      switch (expectedConvention & ~ValueInputConvention::VarArg) {
      default:
        llvm_unreachable("not reachable");
      case ValueInputConvention::ByRef: {
        // The actual value must be an lvalue if callee takes things by-ref.
        auto argVal = operand.ir.getIfLValue();
        if (!argVal)
          return failure();

        // By-ref argument types must exactly match, no conversions are allowed.
        return matchTypes(argVal.getType(), expectedType);
      }
      case ValueInputConvention::ByVal:
        // Otherwise, we pass as an r-value.
        // TODO: Consider implicit conversions?
        return matchTypes(operand.ir.getRValueType(), expectedType);
      }
    };

    // In the typical case, this argument isn't varargs, just check it.
    if (!uint8_t(expectedConvention & ValueInputConvention::VarArg)) {
      // If there was a problem, report it, otherwise continue on to the next
      // expected argument to check.
      if (failed(checkOneOperand(expectedType)))
        return {};
    } else {
      // If we have a varargs argument, then it will eat the rest of the
      // arguments, but we have to check each of them.
      auto varArgsEltType = getVariadicElementType(expectedType);
      while (providedValueIdx != operands.size()) {
        if (failed(checkOneOperand(varArgsEltType)))
          return {};
      }
    }
  }

  // If we have left over operands, then this signature cannot match.
  if (providedValueIdx != operands.size())
    return {};

  // If we have no inferred values or if they disagree, then we fail to infer.
  if (inferredValues.empty() ||
      !llvm::all_of(inferredValues, [&](MValue v) -> bool {
        return v.get() == inferredValues.front().get();
      }))
    return {};

  return inferredValues.front();
}

//===----------------------------------------------------------------------===//
// InputParamBindings Implementation
//===----------------------------------------------------------------------===//

/// Check that our set of parameter bindings work with the specified input
/// parameters, returning a checked ParamBindArrayAttr if so.  If the parameters
/// do not work, this emits an diagnostic (if `declOp` is non-null) and set
/// `incorrectBindingNo/Expectedtype` to the bad binding (or -1 if there is a
/// count mismatch).
ParamBindArrayAttr InputParamBindings::verifyBindings(
    ParamDeclArrayAttr actualParamDecls, StringRef baseName, SMLoc loc,
    ssize_t &incorrectBindingNo, ASTType &incorrectBindingExpectedType,
    LitSharedState &shared, Operation *declOp,
    ParameterInferenceHookTy parameterInferenceHook) const {

  // If we have an incorrect number of bindings specified, this lambda reports
  // the problem.
  auto complainAboutParameterCount = [&]() {
    // Tell the caller what went wrong.
    incorrectBindingNo = -1;
    if (!declOp)
      return;
    auto expectedNumParams = actualParamDecls.size();
    auto actualNumParams = bindings.size();
    auto diag = shared.emitError(loc, "'")
                << baseName << "' expects " << expectedNumParams
                << " input parameter" << plural(expectedNumParams) << " but "
                << actualNumParams << plural(actualNumParams, " was", " were")
                << " provided";
    diag.attachNote(declOp->getLoc()) << "'" << baseName << "' declared here";
  };

  // If we have bound parameters, type check them now and bind names to them.
  SmallVector<ParamBindAttr> newBindings;
  newBindings.reserve(actualParamDecls.size());

  // This is the IR emitter we use for emitting implicit conversions when
  // needed.
  IREmitter emitter(shared, /*no builder*/ {});

  // Parameters defined at the beginning of the parameter list may be used by
  // the types of other parameters defined later in the list, e.g. in:
  //    [rank: Int, indices: StaticTuple[rank]]
  // the value provided to 'indices' should actually depend on the specified
  // value of 'rank'.  We use a ParameterEvaluator to keep track of the mapping
  // so far and remap types on demand.
  ParameterEvaluator evaluator;
  size_t nextBinding = 0;
  for (ParamDeclAttr decl : actualParamDecls) {
    // This lambda installs the decl's value in the parameter evaluator and new
    // binding array.
    auto setParamValue = [&](TypedAttr value) {
      evaluator.setParameterValue(decl, value);
      newBindings.push_back(ParamBindAttr::get(decl.getName(), value));
    };

    // Check to see if we ran out of bindings to provide to this param decl.
    if (nextBinding == bindings.size()) {
      // If the parameter decl is a variadic parameter list, we can fulfill it
      // with an empty list.  We know it must be the last parameter decl.
      if (auto variadicType = dyn_cast<KGEN::VariadicType>(decl.getType())) {
        auto emptyVariadic =
            KGEN::VariadicAttr::get(ArrayRef<TypedAttr>(), variadicType);
        setParamValue(emptyVariadic);
        continue;
      }

      // If we have a method to infer parameter values, invoke it to see if we
      // can get an inferred value for the parameter.
      if (parameterInferenceHook) {
        if (auto value = parameterInferenceHook(decl, newBindings)) {
          assert(value.getType() == evaluator.getReboundType(decl.getType()) &&
                 "inferred a default parameter value of wrong type");
          setParamValue(value);
          continue;
        }
      }

      // TODO: Apply default values for parameters.

      // Otherwise, we're simply missing bindings.
      complainAboutParameterCount();
      return {};
    }

    auto binding = bindings[nextBinding++];
    // If this value was already bound and checked, use it.
    if (auto prebound = dyn_cast<ParamBindAttr>(binding.bindingOrValue)) {
      assert(decl.getName() == prebound.getName());
      setParamValue(prebound.getValue());
      continue;
    }

    auto handleSingleParameterValue = [&](Binding binding,
                                          ASTType expectedType) -> MValue {
      assert(binding.expr &&
             "should always have an expr tree for unchecked bindings");

      // Check the type matches what is expected, and perform an implicit
      // conversion if needed.
      expectedType = ASTType(evaluator.getReboundType(expectedType.mlirType));

      auto errorHandler = [&]() {
        if (declOp) {
          auto diag = shared.emitError(binding.expr->getLoc(), "'")
                      << baseName << "' parameter " << decl.getName() << " has "
                      << expectedType << " type, but value has type "
                      << ASTType(binding.getValue().getType())
                      << binding.expr->getRange();
          diag.attachNote(declOp->getLoc())
              << "'" << baseName << "' declared here";
        }
        incorrectBindingNo = newBindings.size();
        incorrectBindingExpectedType = expectedType;
      };

      auto argValue = emitter.getAsExpectedType(
          MValue(binding.getValue()), binding.expr, expectedType, errorHandler);
      if (!argValue)
        return {};

      assert(argValue.getIfMValue() &&
             "cannot emit a dynamic value in parameter context");
      return argValue.getIfMValue();
    };

    // Scalar parameter values are installed directly.
    MValue paramValue;
    auto variadicType = dyn_cast<KGEN::VariadicType>(decl.getType());
    if (!variadicType) {
      // Otherwise we get a single value.
      MValue paramValue = handleSingleParameterValue(binding, decl.getType());
      if (!paramValue)
        return {};
      setParamValue(paramValue);
      continue;
    }

    // If the parameter is a variadic list, it may consume many values, and they
    // all get packed up into a VariadicAttr.
    SmallVector<TypedAttr> elements;
    Type expectedType = ParamRefType::get(variadicType.getElementType());
    elements.push_back(handleSingleParameterValue(binding, expectedType));
    if (!elements.back())
      return {};
    while (nextBinding != bindings.size()) {
      binding = bindings[nextBinding++];
      elements.push_back(handleSingleParameterValue(binding, expectedType));
      if (!elements.back())
        return {};
    }
    setParamValue(KGEN::VariadicAttr::get(elements, variadicType));
  }

  // Check and complain if we have bindings that didn't get used.
  if (nextBinding != bindings.size()) {
    complainAboutParameterCount();
    return {};
  }

  return ParamBindArrayAttr::get(shared.getContext(), newBindings);
}

//===----------------------------------------------------------------------===//
// DirectCallable Implementation
//===----------------------------------------------------------------------===//

/// Get a symbol for a direct reference to the specified function in its
/// enclosing context.  This does not bind any values to arguments.
DirectCallable::DirectCallable(SMLoc nameLoc, StringRef baseName,
                               ArrayRef<ASTDecl *> fnDecls,
                               ParamBindArrayAttr bindingsAttr)
    : nameLoc(nameLoc), baseName(baseName),
      fnDecls(fnDecls.begin(), fnDecls.end()) {
  if (bindingsAttr) {
    for (ParamBindAttr bind : bindingsAttr)
      inputParamBindings.add(bind);
  }
}

namespace {
/// This struct indicates whether a signature can be successfully applied to a
/// parameter binding and argument list.  If so, it keeps track of the number of
/// implicit conversions required to make the call, and if not, it indicates the
/// reason for the mismatch.
struct OverloadFitness {
  enum Kind {
    kValid,            //< This is a valid candidate.
    kParamCount,       //< Invalid due to a parameter count mismatch
    kParamWrongType,   //< A parameter value not convertible to expected type
    kResultParamCount, //< Incorrect number of result params.
    kArgCount,         //< Incorrect number of arguments passed.
    kArgTooFewAtLeast, //< Variadic but too few values were specified.
    kArgTooManyAtMost, //< Default args, but too many values were specified.
    kArgNotLValue,     //< By-ref argument requires an lvalue, but got an rvalue
    kArgWrongLVType,   //< By-ref argument and provided l-value types mismatch.
    kArgWrongType,     //< An argument value not convertible to expected type
  } kind;

  /// The interpretation of this payload depends on the 'kind' field:
  ///  kValid:            number of implicit conversions required.
  ///  kParamCount:       not used.
  ///  kParamWrongType:   the parameter # that mismatches.
  ///  kResultParamCount: not used.
  ///  kArgCount:         the number of arguments expected.
  ///  kArgTooFewAtLeast: the minimum number of arguments expected.
  ///  kArgTooManyAtMost: the maximum number of arguments allowed.
  ///  kArgNotLValue:     the argument # that mismatches.
  ///  kArgWrongLVType:   the argument # that mismatches.
  ///  kArgWrongType:     the argument # that mismatches.
  size_t payload;

  /// For type mismatches, this is the actual or expected type, otherwise null.
  ASTType type;

  /// For valid candidates, this defines the parameter bindings to use.
  ParamBindArrayAttr paramBindings;

  /// Determine whether the specified signature can be invoked with the
  /// parameter bindings specified in `callable` and the arguments specified in
  /// `operands`.
  static OverloadFitness evaluate(SignatureType signature,
                                  const DirectCallable &callable,
                                  ArrayRef<ASTExprAnd<AnyValue>> operands,
                                  LitSharedState &shared);

  /// Add explaination for why this candidate doesn't work to the specified
  /// diagnostic.
  void diagnose(SignatureType signature, const DirectCallable &callable,
                ArrayRef<ASTExprAnd<AnyValue>> operands, CallSyntax syntax,
                LitDiagnostic &diag);
};
} // namespace

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `operands`.
OverloadFitness OverloadFitness::evaluate(
    SignatureType signature, const DirectCallable &callable,
    ArrayRef<ASTExprAnd<AnyValue>> operands, LitSharedState &shared) {

  // Check that the signature can be rebound with this set of bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;
  auto newBindings = callable.inputParamBindings.verifyBindings(
      signature.getInputParams(), callable.baseName, callable.nameLoc,
      incorrectBindingNo, incorrectBindingExpectedType, shared,
      /*don't emit diagnostics*/ nullptr,
      [&](ParamDeclAttr decl, ArrayRef<ParamBindAttr> bindingsSoFar) -> MValue {
        return ParameterInferenceState(decl).infer(signature, bindingsSoFar,
                                                   operands);
      });

  // If there is an error, return the problem.
  if (!newBindings) {
    if (incorrectBindingNo == -1)
      return {kParamCount, 0, ASTType(), newBindings};
    return {kParamWrongType, static_cast<size_t>(incorrectBindingNo),
            incorrectBindingExpectedType, newBindings};
  }

  // Check the result parameter count.
  if (signature.getResultParams().size() != callable.resultParams.size())
    return {kResultParamCount, 0, ASTType(), newBindings};

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  if (!newBindings.empty()) {
    signature = signature.getSpecializedSignature(
        newBindings, [&]() -> InFlightDiagnostic {
          llvm_unreachable("bad bindings went undetected");
        });
    assert(signature && "bad bindings went undetected");
  }

  // Ok, the parameters all line up, check the argument list.  We generally want
  // to diagnose problems where too few or too many arguments are passed if that
  // is the problem, rather than complaining about a type error of some argument
  // that doesn't work out.  Check for that first.
  size_t minRequiredArgs = 0;
  size_t maxAllowedargs = 0;
  for (auto convention : signature.getValueInputConventions()) {
    // Varargs arguments don't require a value, but allow any number of them.
    if (uint8_t(convention & ValueInputConvention::VarArg)) {
      maxAllowedargs = ~size_t(0);
      continue;
    }

    ++minRequiredArgs;
    ++maxAllowedargs;
  }

  // One less required argument for each argument that has a default value we
  // can use instead.
  if (DefaultArgumentArrayAttr defaults = signature.getDefaultArguments())
    minRequiredArgs -= defaults.size();

  if (operands.size() < minRequiredArgs) {
    // Tailor the diagnostic when more args are allowed.
    auto problem =
        minRequiredArgs != maxAllowedargs ? kArgTooFewAtLeast : kArgCount;
    return {problem, minRequiredArgs, ASTType(), newBindings};
  }
  if (operands.size() > maxAllowedargs) {
    // Tailor the diagnostic when more args are allowed.
    auto problem =
        minRequiredArgs != maxAllowedargs ? kArgTooManyAtMost : kArgCount;
    return {problem, maxAllowedargs, ASTType(), newBindings};
  }

  // As we walk through the values provided as part of the argument list, we
  // match them up against arguments expected by the signature of the callee and
  // count how many implicit conversions are required for a match.
  size_t providedValueIdx = 0;
  size_t numImplicitConversions = 0;
  for (auto [expectedArgIdx, expectedType] :
       llvm::enumerate(signature.getValueInputs())) {
    auto expectedConvention = signature.getInputConvention(expectedArgIdx);

    // Handle case when there are no more provided arguments.
    if (providedValueIdx == operands.size()) {
      // If the argument is a varargs argument list, then it can be initialized
      // with zero values no problem.
      if (uint8_t(expectedConvention & ValueInputConvention::VarArg))
        break;
      // We don't need a provided value for this index if we can use a default
      // value, which has already been converted to the expected type.
      if (signature.getDefaultArguments() &&
          providedValueIdx ==
              signature.getDefaultArguments().front().getIndexValue())
        // In the callee, arguments with default values must be followed only by
        // other arguments with default values, so we do not need to enumerate
        // any more of the callee arguments.
        break;
    }

    // Otherwise we'll check the expected type against one (or more in the case
    // of varargs) provided values.
    auto checkOneOperand = [&](ASTType expectedType) -> OverloadFitness {
      // We'll bind the next provided value.
      auto operand = operands[providedValueIdx];
      switch (expectedConvention & ~ValueInputConvention::VarArg) {
      case ValueInputConvention::KWVarArg:
        assert(0 &&
               "keyword arguments and `**arg` variadics not supported yet");
        break;
      case ValueInputConvention::VarArg:
        llvm_unreachable("not reachable");
      case ValueInputConvention::ByRef: {
        // The actual value must be an lvalue if callee takes things by-ref.
        auto argVal = operand.ir.getIfLValue();
        if (!argVal)
          return {kArgNotLValue, providedValueIdx, operand.ir.getType(),
                  newBindings};

        // By-ref argument types must exactly match, no conversions are allowed.
        if (!ASTType(argVal.getType()).isEqualCanon(expectedType))
          return {kArgWrongLVType, providedValueIdx, expectedType, newBindings};
        break;
      }
      case ValueInputConvention::ByVal:
        auto argType = operand.ir.getRValueType();
        // Otherwise, we pass as an r-value.  If the argument types match, then
        // they are good.
        if (argType.isEqualCanon(expectedType))
          break;

        // If we lack an exact match and conversions are disabled, this
        // candidate fails.
        if (callable.disableImplicitConversions ||
            !CallableValue::canImplicitlyConvertToType(operand, expectedType,
                                                       shared))
          return {kArgWrongType, providedValueIdx, expectedType, newBindings};

        // If we had one, this bumps our # implicit conversions.
        ++numImplicitConversions;
        break;
      }

      // This provided value has been used up.
      ++providedValueIdx;
      return {kValid, 0, ASTType(), newBindings};
    };

    // In the typical case, this argument isn't varargs, just check it.
    if (!uint8_t(expectedConvention & ValueInputConvention::VarArg)) {
      // If there was a problem, report it, otherwise continue on to the next
      // expected argument to check.
      auto result = checkOneOperand(expectedType);
      if (result.kind != kValid)
        return result;
    } else {
      // If we have a varargs argument, then it will eat the rest of the
      // arguments, but we have to check each of them.
      auto varArgsEltType = getVariadicElementType(expectedType);
      while (providedValueIdx != operands.size()) {
        auto result = checkOneOperand(varArgsEltType);
        if (result.kind != kValid)
          return result;
      }
    }
  }

  assert(providedValueIdx == operands.size() &&
         "should handle argument mismatch above");

  // Otherwise we succeeded!
  return {kValid, numImplicitConversions, ASTType(), newBindings};
}

/// Add explaination for why this candidate doesn't work to the specified
/// diagnostic. isMethodCall indicates whether the call was written with
/// `foo(x,y)` syntax or `x.foo(y)` syntax.
void OverloadFitness::diagnose(SignatureType signature,
                               const DirectCallable &callable,
                               ArrayRef<ASTExprAnd<AnyValue>> operands,
                               CallSyntax syntax, LitDiagnostic &diag) {
  // TODO: Would be really nice to range underline the operand in question!
  switch (kind) {
  case kValid:
    diag << "candidate is viable";
    return;
  case kParamCount: {
    size_t actualNumBindings = callable.inputParamBindings.bindings.size();
    diag << "callee expects " << signature.getInputParams().size()
         << " input parameter" << plural(signature.getInputParams().size())
         << " but " << actualNumBindings
         << plural(actualNumBindings, " was", " were") << " provided";
    return;
  }
  case kParamWrongType: {
    auto decl = signature.getInputParams()[payload];
    auto binding = callable.inputParamBindings.bindings[payload];
    diag << "callee parameter " << decl.getName() << " has " << ASTType(type)
         << " type, but value has type " << ASTType(binding.getType())
         << binding.expr->getRange();
    return;
  }
  case kResultParamCount:
    diag << "callee expects " << signature.getResultParams().size()
         << " result parameter" << plural(signature.getResultParams().size())
         << " but " << callable.resultParams.size()
         << plural(callable.resultParams.size(), " was", " were")
         << " provided";
    return;
  case kArgCount:
    diag << "callee expects " << payload << " arguments, but "
         << operands.size() << " specified";
    return;
  case kArgTooFewAtLeast:
    diag << "callee expects at least " << payload << " arguments, but only "
         << operands.size() << " specified";
    return;
  case kArgTooManyAtMost:
    diag << "callee expects at most " << payload << " arguments, but "
         << operands.size() << " were specified";
    return;
  case kArgNotLValue:
    if (syntax == CallSyntax::kMethodCall && payload == 0) {
      diag << "invalid use of mutating method on rvalue of type "
           << ASTType(type) << operands[0].expr->getRange();
      return;
    }
    diag << "argument #" << payload
         << " must be mutable in order to pass as a by-ref argument"
         << operands[0].expr->getRange();
    return;
  case kArgWrongLVType: {
    MValue eltTypeAttr = cast<POP::PointerType>(Type(type)).getElementType();
    assert(eltTypeAttr.getIfTypeValue() &&
           "unwrapped value should be a direct type, not a parameter");
    diag << "l-value of type " << operands[payload].ir.getRValueType()
         << " cannot be converted to reference of type "
         << eltTypeAttr.getIfTypeValue() << operands[payload].expr->getRange();
    return;
  }

  case kArgWrongType:
    // If this is a method syntax call, don't count the receiver.
    if (syntax == CallSyntax::kMethodCall) {
      // it is probably possible for this assert to fire, if it does we should
      // tailor the error message.
      assert(payload != 0 && "TODO: unexpected self mismatch");
      diag << "method argument #" << (payload - 1);
    } else if (syntax == CallSyntax::kOperator && payload == 1) {
      diag << "right side";
    } else if (syntax == CallSyntax::kReversedOperator && payload == 0) {
      diag << "left side";
    } else if (syntax == CallSyntax::kSubscript && payload != 0) {
      if (payload == 1 && operands.size() == 2)
        diag << "index";
      else
        diag << "index #" << (payload - 1);
    } else {
      diag << "argument #" << payload;
    }
    diag << " cannot be converted from " << operands[payload].ir.getRValueType()
         << " to " << type << operands[payload].expr->getRange();
    break;
  }
}

/// Evaluate the fnDecls candidates and see if there is an unambiguous
/// candidate that works with the specified parameter bindings and provided
/// arguments.  If so, replace fnDecls with a single entry that works and
/// return success.  If not, generate a diagnostic and return failure.
LogicalResult DirectCallable::filterOverloadSet(
    ArrayRef<ASTExprAnd<AnyValue>> operands, CallSyntax syntax,
    bool emitDiagnosticOnFailure, LitSharedState &shared,
    SymbolConstantAttr *validCandidate) {
  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<OverloadFitness> evaluations;
  bool anyValid = false;
  for (ASTDecl *candidate : fnDecls) {
    auto signature = cast<LIT::FuncOp>(*candidate).getFullSignature();
    evaluations.push_back(
        OverloadFitness::evaluate(signature, *this, operands, shared));
    anyValid |= evaluations.back().kind == OverloadFitness::kValid;
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (!anyValid) {
    if (emitDiagnosticOnFailure) {
      // If there is a single callee, emit a specific error about the call.
      if (fnDecls.size() == 1) {
        auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
        auto diag = shared.emitError(nameLoc, "invalid call to '")
                    << baseName << "': ";
        evaluations[0].diagnose(fnDecl.getFullSignature(), *this, operands,
                                syntax, diag);
        diag.attachNote(fnDecl.getLoc()) << "function declared here";
        return failure();
      }

      // Otherwise emit an error, and a note for what is wrong with each
      // candidate.
      auto diag = shared.emitError(nameLoc, "no matching function in call to '")
                  << baseName << "': ";
      for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
        auto fnDecl = cast<LIT::FuncOp>(*candidate);
        diag.attachNote(fnDecl->getLoc()) << "candidate not viable: ";
        eval.diagnose(fnDecl.getFullSignature(), *this, operands, syntax, diag);
      }
      return failure();
    }
  }

  // Ok, we have at least one valid candidate, filter the list to the ones with
  // the lowest number of implicit conversions required.
  size_t minConversions = ~0U;
  SmallVector<ASTDecl *, 1> newFnDecls;
  OverloadFitness oneFitness = evaluations[0];
  for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
    // Ignore failures or candidates that have more conversions.
    if (eval.kind != OverloadFitness::kValid || eval.payload > minConversions)
      continue;

    // If we found a new floor to the # conversions needed, clear the list.
    if (eval.payload < minConversions) {
      newFnDecls.clear();
      minConversions = eval.payload;
    }
    newFnDecls.push_back(candidate);
    oneFitness = eval;
  }

  // If we found exactly one viable candidate, then we succeed.
  if (newFnDecls.size() == 1) {
    // If the caller wanted to know about the valid symbol, return it.
    if (validCandidate)
      *validCandidate = cast<LIT::FuncOp>(*newFnDecls[0])
                            .getBoundReference(oneFitness.paramBindings);
    // Mutate our state to represent what we've learned.  We have one callee
    // and we have valid predetermined parameter bindings.
    fnDecls = std::move(newFnDecls);
    inputParamBindings.bindings.clear();
    for (auto bind : oneFitness.paramBindings)
      inputParamBindings.add(bind);

    return success();
  }

  // Otherwise, we have multiple viable candidates that are ambiguous because
  // they all require the same number of implicit conversions.
  if (emitDiagnosticOnFailure) {
    auto diag = shared.emitError(nameLoc, "ambiguous call to '")
                << baseName << "', each candidate requires " << minConversions
                << " implicit conversion" << plural(minConversions)
                << ", disambiguate with an explicit cast";
    for (ASTDecl *candidate : newFnDecls)
      diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
          << "candidate declared here";
  }
  return failure();
}

/// Perform subsitutions of the specified bindings into the symbol, returning
/// the resultant LITSymbolConstant attr or producing an error message and
/// returning null. This allows producing a reference to a parameterized
/// function without the parmaeters specified.  They can be bound later.
SymbolConstantAttr
DirectCallable::getBoundConstantAttr(LitSharedState &shared) const {
  if (fnDecls.size() != 1) {
    assert(!fnDecls.empty() && "DirectCallable malformed");
    auto diag =
        shared.emitError(
            nameLoc, "cannot form a reference to overloaded declaration of '")
        << baseName << "'";
    for (ASTDecl *candidate : fnDecls) {
      auto funcOp = cast<LIT::FuncOp>(*candidate);
      diag.attachNote(funcOp.getLoc()) << "candidate declared here";
    }

    return {};
  }

  auto funcOp = cast<LIT::FuncOp>(*fnDecls[0]);

  // If there are no input parameters specified and if we allow unbound symbols,
  // just return the unbound symbol.
  if (inputParamBindings.bindings.empty())
    return funcOp.getBoundReference();

  // Check that the signature can be rebound with our set of bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;

  auto newBindings = inputParamBindings.verifyBindings(
      funcOp.getFullSignature().getInputParams(), baseName, nameLoc,
      incorrectBindingNo, incorrectBindingExpectedType, shared,
      /*emit diagnostics*/ funcOp);
  if (!newBindings)
    return {};

  // Now that we checked the types match, form the binding.
  return funcOp.getBoundReference(newBindings);
}

/// Check declarations for the result parameters and add them to
/// resultParamDecls.  This emits and error and returns failure if an error is
/// detected.
LogicalResult DirectCallable::getResultParamDecls(
    SignatureType signature, SmallVectorImpl<ParamDeclAttr> &resultParamDecls,
    IREmitter &emitter) {
  assert(signature.getResultParams().size() == resultParams.size() &&
         "We know that the callee is type checked");

  // If there is nothing to do, then we are done.
  if (resultParams.empty())
    return success();

  // Verify completion of forward declared alias declarations.  We know the
  // decl exists, but we don't know if the type is compatible or it has been
  // multiply defined.
  //
  // TODO: We don't remap input parameters types into output parameter types.
  // We surely handle this wrong: `fn x[a: type -> a]():` for example.
  for (auto [param, declAndLoc] :
       llvm::zip(signature.getResultParams(), resultParams)) {
    auto forwardDecl = cast<AliasForwardDeclOp>(*declAndLoc.first);

    // Verify the types match.
    // TODO: Move this to overload resolution.
    if (!ASTType(forwardDecl.getType()).isEqualCanon(param.getType())) {
      auto diag =
          emitter.emitError(declAndLoc.second, "result parameter returns type ")
          << param.getType() << " but forward declaration is of type "
          << ASTType(forwardDecl.getType());
      diag.attachNote(forwardDecl.getLoc()) << "alias forward declared here";
      return failure();
    }
    resultParamDecls.push_back(
        ParamDeclAttr::get(forwardDecl.getName(), param.getType()));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallableValue Implementation
//===----------------------------------------------------------------------===//

/// Get a CallableValue for a lookup of a named method on the specified type.
/// If successful, this provides a non-null CallableValue.
///
/// On failure, this returns a null CallableValue and sets 'erroneousDecl' to
/// indicate whether there was a problem with the callee that has already been
/// diagnosed (allowing the client to squish downstream error messages).  This
/// does not emit an error on failure.
CallableValue::CallableValue(ASTType type, StringRef methodName, SMLoc callLoc,
                             bool &erroneousDecl, LitSharedState &shared) {

  erroneousDecl = false;
  // First perform a lookup to see if there are any candidates.
  auto lookupResult = shared.lookupAndResolveDecl(methodName, callLoc, type,
                                                  /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
  if (resultDecls.empty()) {
    if (lookupResult.isErroneous())
      erroneousDecl = true;
    return;
  }

  // If we find a vardecl or any other thing, then fail because it cannot be
  // called.
  if (!isa<LIT::FuncOp>(*resultDecls[0]))
    return;

  // Handle method references, which might be overloaded.
  direct =
      DirectCallable{callLoc, methodName, resultDecls, type.getParamBindings()};
}

/// Emit this as a flattened RValue or LValue with no additional parameter
/// context.  This returns null on failure.
AnyValue CallableValue::emitAsValue(IREmitter &emitter) const {
  // If we have no bound symbol, return the normal lvalue or rvalue we
  // represent.
  if (!direct)
    return baseVal.ir;

  // We allow unbound symbols here which can be emitted as an MValue.  In the
  // case where we are partially applying, that will force the unbound symbol
  // into a DRValue which will catch symbols that are not fully bound.
  auto directSymbolAttr = direct->getBoundConstantAttr(emitter.shared);
  if (!directSymbolAttr)
    return {};

  // Verify that the target has no result parameters.  We have no way to bind
  // these indirectly.
  SignatureType calleeSignature = directSymbolAttr.getType();
  if (!calleeSignature.getResultParams().empty()) {
    emitter.emitError(direct->nameLoc,
                      "calls with result parameters must be called directly");
    return {};
  }

  // If we have no base value, then we are just a symbol, return it.
  if (!baseVal)
    return MValue(directSymbolAttr);

  auto loc = baseVal.expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  Type firstArgIRType = calleeSignature.getValueInputs()[0];
  Value firstArgValue;
  ValueInputConvention selfConvention = calleeSignature.getInputConvention(0);

  assert(!uint8_t(selfConvention & ValueInputConvention::VarArg) &&
         "Error: self shouldn't be able to be varargs");

  switch (selfConvention) {
  case ValueInputConvention::KWVarArg:
    emitter.emitError(
        loc, "keyword arguments and `**arg` variadics not supported yet");
    return {};
  case ValueInputConvention::VarArg:
    llvm_unreachable("unreachable");
  case ValueInputConvention::ByRef: {
    LValue baseLV = baseVal.ir.getIfLValue();
    if (!baseLV) {
      emitter.emitError(loc,
                        "invalid use of mutating method on rvalue of type ")
          << ASTType(baseVal.ir.getType()) << baseVal.expr->getRange();
      return {};
    }
    firstArgValue = baseLV;

    // Using partial application over an lvalue isn't safe until we support an
    // ownership models with mutable borrows.
    emitter.emitError(loc, "TODO: partial application to mutable base isn't "
                           "supportable without a lifetime model")
        << baseVal.expr->getRange();
    return {};
  }
  case ValueInputConvention::ByVal:
    // Otherwise we can have either an lvalue or rvalue, but we need to convert
    // to an rvalue if we have an lvalue.
    firstArgValue = emitter.emitDRValue(baseVal);
    if (!firstArgValue)
      return {};
    break;
  }

  assert(firstArgIRType == firstArgValue.getType() &&
         "base types should always structurally line up");

  // For an instance value, we have to partially apply the callee to the first
  // argument of the reference.  Materialize callee as a DRValue for
  // partial_apply.
  auto calleeDRVal =
      emitter.emitDRValue({AnyValue(directSymbolAttr), baseVal.expr});

  // Partial apply wants to know what operands to bind, we always bind the first
  // one.
  auto zeroAttr = emitter.builder->getAttr<mlir::DenseI64ArrayAttr>(0);
  return DRValue(emitter.builder->create<POP::PartialApplyOp>(
      baseVal.expr->getLocation(emitter), calleeDRVal,
      mlir::ValueRange(firstArgValue), zeroAttr));
}

/// Return true if 'value' may be implicitly converted to 'requiredType'
/// by invoking (one level of) conversion operations.  This does not generate
/// any IR.
bool CallableValue::canImplicitlyConvertToType(ASTExprAnd<AnyValue> value,
                                               ASTType requiredType,
                                               LitSharedState &shared) {
  // If it already matches, then we're done.
  if (value.ir.getRValueType().isEqualCanon(requiredType))
    return true;

  // Otherwise, check to see if we can do an implicit conversion by invoking a
  // `__new__` method on the expected type.
  bool isErroneousDecl = false;
  CallableValue callee(requiredType, "__new__", value.expr->getLoc(),
                       isErroneousDecl, shared);

  // If there are no viable candidates for the implicit conversion, we fail.
  if (!callee.direct)
    return false;

  // If we have at least one candidate, we check to see if any of them can
  // work. We disable implicit conversions though, to prevent converting
  // T -> S -> U in one step.
  callee.direct->disableImplicitConversions = true;
  return succeeded(callee.direct->filterOverloadSet(
      {value}, CallSyntax::kImplicitConvert,
      /*emitDiagnosticOnFailure=*/false, shared));
}

//===----------------------------------------------------------------------===//
// Call Emission Implementation
//===----------------------------------------------------------------------===//

/// Returns true if the insertion context is valid for implicit error
/// propagation.
static bool isValidErrorContext(Block *block) {
  for (Operation *op = block->getParentOp(); op; op = op->getParentOp()) {
    if (auto tryOp = dyn_cast<TryOp>(op);
        tryOp && tryOp.getTryRegion().isAncestor(block->getParent()))
      return true;
    if (auto func = dyn_cast<LIT::FuncOp>(op))
      return func.isThrows();
  }
  llvm_unreachable("block outside of function?");
}

/// Emit a function call to the specified callee with the specified operand
/// values.  This emits an error and returns null on failure.
AnyValue
CallableValue::emitFunctionCall(ArrayRef<ASTExprAnd<AnyValue>> operands,
                                CallSyntax syntax, const ExprNode *callNode,
                                IREmitter &emitter) {
  if (isNull()) // Base was already diagnosed as an error.
    return {};

  // Used in some cases below, lifetime needs to exist for this whole method.
  SmallVector<ASTExprAnd<AnyValue>> operandsWithSelf;
  SMLoc callLoc = callNode->getLoc();

  auto emitError = [&](const Twine &message) {
    return emitter.emitError(callLoc, message);
  };

  // Figure out the type of the function to call, which is either symbol or a
  // normal rvalue.
  SignatureType calleeSig;

  // If the callee defines parameters, this is the definitions to use.
  SmallVector<ParamDeclAttr> resultParamDecls;

  // This is the callee symbol constant for a direct call, or the SSA value for
  // an indirect call.
  AnyValue callee;
  if (direct) {
    // If we have a bound self, add it to the operand list to simplify the logic
    // below.
    if (baseVal) {
      operandsWithSelf.reserve(operands.size() + 1);
      operandsWithSelf.push_back(baseVal);
      operandsWithSelf.append(operands.begin(), operands.end());
      operands = operandsWithSelf;
      baseVal = {};
      assert(syntax == CallSyntax::kMethodCall && "Unexpected syntax form");
    }

    // Check the direct callees to see if they can be unambiguously resolved
    // with the bindings list and specified arguments.
    SymbolConstantAttr symbol;
    if (failed(direct->filterOverloadSet(operands, syntax,
                                         /*emitDiagnosticOnFailure=*/true,
                                         emitter.shared, &symbol)))
      return {};

    calleeSig = symbol.getType();
    callee = MValue(symbol);
    if (failed(
            direct->getResultParamDecls(calleeSig, resultParamDecls, emitter)))
      return {};

  } else {
    // Otherwise we have an indirect call. If the callee is an MValue, emit a
    // `call_param`. Otherwise, emit the callee value as a DRValue so we can
    // call it with call_indirect.
    callee = baseVal.ir.getIfMValue();
    if (!callee) {
      callee = emitter.emitDRValue(baseVal);
      if (!callee)
        return {};
    }

    calleeSig = dyn_cast<SignatureType>(callee.getType());
    if (!calleeSig) {
      emitError("invalid function type to call ")
          << ASTType(callee.getType()) << baseVal.expr->getRange();
      return {};
    }

    // Check to see if we can apply these operands to the callee signature.
    DirectCallable bindings{callLoc, "callee", /*params*/ {}, {}};
    auto fitness = OverloadFitness::evaluate(calleeSig, bindings, operands,
                                             emitter.shared);
    if (fitness.kind != OverloadFitness::kValid) {
      // If not, diagnose it with an error.
      auto diag = emitError("invalid indirect call: ");
      fitness.diagnose(calleeSig, bindings, operands, syntax, diag);
      return {};
    }
  }

  assert(calleeSig.getResultParams().size() == resultParamDecls.size() &&
         "Type checking should be done");

  // Emit all the arguments.  We iterate by expected arguments since we're
  // building the argument list of the call.  Default arguments and variadics
  // get filled in here.
  SmallVector<ASTExprAnd<AnyValue>> argumentValues;
  size_t nextOperandIdx = 0;
  size_t nextDefaultIdx = 0;
  for (auto [expectedTypeX, conventionX] : llvm::zip(
           calleeSig.getValueInputs(), calleeSig.getValueInputConventions())) {
    // Work around lambda not being able to reference bindings.
    auto expectedType = expectedTypeX;
    auto convention = conventionX;
    // If we ran out of operands, fulfill this with a default value or empty
    // variadic list.
    if (nextOperandIdx == operands.size()) {
      // Varargs arguments are fulfilled with an empty !pop.variadic list.
      if (uint8_t(convention & ValueInputConvention::VarArg)) {
        auto variadic = KGEN::VariadicAttr::get(
            ArrayRef<TypedAttr>(), expectedType.cast<KGEN::VariadicType>());
        argumentValues.push_back({MValue(variadic), callNode});
        continue;
      }
      // Otherwise, apply the default argument. We've ensured above that we have
      // a default argument for each missing operand.
      argumentValues.push_back(
          {MValue(calleeSig.getDefaultArguments()[nextDefaultIdx].getValue()),
           callNode});
      ++nextDefaultIdx;
      continue;
    }

    // Otherwise, we're applying one or more arguments to this.
    auto emitOneArgVal = [&](ASTExprAnd<AnyValue> operand) -> AnyValue {
      switch (convention & ~ValueInputConvention::VarArg) {
      case ValueInputConvention::KWVarArg:
        llvm_unreachable("keyword args and `**arg` not supported yet");
        break;
      case ValueInputConvention::VarArg:
        llvm_unreachable("varargs handled separately");
      case ValueInputConvention::ByRef:
        // By-ref arguments, must be lvalues.
        assert(operand.ir.getIfLValue() &&
               "Call should already be type checked");
        return operand.ir;
        break;
      case ValueInputConvention::ByVal:
        // by-val arguments are converted to the expected r-value type.
        auto argVal = emitter.emitRValue(operand);
        // In the case of a variadic argument, we need to remove the
        // !pop.varadic<> wrapper to get the type to convert to.
        Type expectedArgType = expectedType;
        if (uint8_t(convention & ValueInputConvention::VarArg))
          expectedArgType = getVariadicElementType(expectedArgType);

        return emitter.getAsExpectedType(argVal, operand.expr, expectedArgType,
                                         " in argument");
      }
    };

    // For a normal non-vararg argument, we just emit it and add it to our list.
    if (!uint8_t(convention & ValueInputConvention::VarArg)) {
      auto operand = operands[nextOperandIdx++];
      AnyValue argVal = emitOneArgVal(operand);
      if (!argVal)
        return {};
      argumentValues.push_back({argVal, operand.expr});
      continue;
    }

    // For variadic list, we need to emit all of the remaining operands.
    // Emit all of the remaining values to make sure they're converted to the
    // right type.
    SmallVector<ASTExprAnd<AnyValue>> variadicOperands(
        operands.begin() + nextOperandIdx, operands.end());
    for (auto &operand : variadicOperands) {
      auto emittedArg = emitOneArgVal(operand);
      if (!emittedArg)
        return {};
      operand.ir = emittedArg;
    }
    nextOperandIdx = operands.size();

    // If all of the operands are compile-time values, then we can represent
    // the variadic sequence as an attribute.
    if (std::all_of(variadicOperands.begin(), variadicOperands.end(),
                    [](auto operand) { return operand.ir.getIfMValue(); })) {
      SmallVector<TypedAttr> variadicArgs;
      for (auto operand : variadicOperands)
        variadicArgs.push_back(operand.ir.getIfMValue().get());
      auto argAttr = KGEN::VariadicAttr::get(
          variadicArgs, expectedType.cast<KGEN::VariadicType>());
      argumentValues.push_back({MValue(argAttr), variadicOperands[0].expr});
      continue;
    }

    // If not all operands are compile-time values, use an operation to create a
    // variadic sequence.
    SmallVector<Value> variadicArgs;
    for (auto &operand : variadicOperands) {
      DRValue argVal =
          emitter.emitDRValue({emitOneArgVal(operand), operand.expr});
      if (!argVal)
        return {};
      variadicArgs.push_back(argVal);
    }

    Location loc = emitter.translateLocation(callLoc);
    Value argVal = emitter.builder->create<POP::VariadicCreateOp>(
        loc, expectedType, variadicArgs);
    argumentValues.push_back({RValue(argVal), variadicOperands[0].expr});
  }

  assert(nextOperandIdx == operands.size() &&
         "typechecking confirmed that we would use up all operands");

  // If this is a call to a @always_inline function, see if we can fold its
  // entire body into an MValue.  This can fail for a number of reasons, in
  // which case we fall back to emitting normally.
  if (direct) {
    auto calleeFunc = cast<LIT::FuncOp>(*direct->fnDecls[0]);
    if (calleeFunc.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled) {
      auto calleeSym = cast<SymbolConstantAttr>(callee.getIfMValue().get());
      ParamBindArrayAttr inputParams = calleeSym.getParamValues();
      if (auto result = inlineFunctionCallIntoMValue(
              callLoc, *direct->fnDecls[0], inputParams, argumentValues,
              emitter))
        return result;
    }
  }

  auto &builder = emitter.builder;
  if (!builder) {
    // Emitting a call in a meta context. Generate an apply operator.
    SmallVector<TypedAttr> operands({callee.getIfMValue().get()});
    for (auto argValAndExpr : argumentValues) {
      if (!argValAndExpr.ir.getIfMValue()) {
        emitter.emitError(argValAndExpr.expr->getLoc(),
                          "cannot use a dynamic value in meta context")
            << argValAndExpr.expr->getRange();
        return {};
      }
      operands.push_back(argValAndExpr.ir.getIfMValue().get());
    }

    // Calls in parameter context cannot have result parameters.
    if (!calleeSig.getResultParams().empty()) {
      assert(direct && "can only have result parameters in direct calls");
      auto diag =
          emitter.emitError(callLoc, "cannot call '")
          << direct->baseName
          << "' in parameter expression because it has a parameter result";
      for (auto &resultParam : direct->resultParams) {
        diag << LitSourceRange(resultParam.second, resultParam.second);
        resultParam.first->hasReferenceError = true;
      }
      return {};
    }

    return MValue(ParamOperatorAttr::get(POC::Apply, operands));
  }

  // Otherwise, materialize MValue arguments as DRValues.
  SmallVector<Value> callArgs;
  for (auto argValAndExpr : argumentValues) {
    if (auto lv = argValAndExpr.ir.getIfLValue())
      callArgs.push_back(lv);
    else
      callArgs.push_back(emitter.emitDRValue(argValAndExpr));
    if (!callArgs.back())
      return {};
  }

  ArrayRef<Type> resultTypes = calleeSig.getValueResults();
  Operation *callOp;
  Location loc = emitter.translateLocation(callLoc);
  if (auto target = callee.getIfMValue()) {
    if (cast<SignatureType>(target.getType()).isAsync()) {
      // If the callee is an async function, emit an async call.
      callOp = builder->create<AsyncCallOp>(loc, target.get(), resultParamDecls,
                                            callArgs);
    } else if (auto symbol = dyn_cast<SymbolConstantAttr>(target.get())) {
      // If the callee is a symbol constant, directly emit a call.
      callOp = builder->create<CallOp>(loc, resultTypes, symbol,
                                       resultParamDecls, callArgs);
    } else {
      callOp = builder->create<CallParamOp>(loc, resultTypes, target.get(),
                                            resultParamDecls, callArgs);
    }
  } else {
    // Otherwise emit calls to SSA values with call_indirect.
    callOp = builder->create<POP::CallIndirectOp>(
        loc, resultTypes, callee.getIfDRValue(), callArgs);
  }

  // If the callee can raise an error, try to unwrap it.
  if (calleeSig.isThrows() && !calleeSig.isAsync() &&
      !isValidErrorContext(builder->getInsertionBlock())) {
    emitError(
        "cannot call function that may raise in a context that cannot raise");
    return {};
  }

  // Value returning call returns its result.
  return DRValue(callOp->getResult(0));
}

/// Given a call to an alwaysinline function, check to see if we can resolve it
/// all the way down to an MValue.  We do this when in a parameter context
/// (since there is no debug info to ever generate) and when calling a "nodebug"
/// function.  This is best-effort: when it fails, we fall back to emitting a
/// normal call or "apply" parameter expression.
AnyValue CallableValue::inlineFunctionCallIntoMValue(
    SMLoc callLoc, ASTDecl &callee, ParamBindArrayAttr inputParams,
    ArrayRef<ASTExprAnd<AnyValue>> argumentValues, IREmitter &emitter) {
  auto funcOp = cast<LIT::FuncOp>(callee);

  // TODO: We currently cannot handle calls to parameterized functions, we
  // aren't doing the substitution yet.
  if (!inputParams.empty())
    return {};

  // Resolve the body to type check and generate the IR we need for inlining.
  if (failed(emitter.getDeclResolver().resolveFully(callee, callLoc)))
    return {};

  // We aren't allowed to toss away debug information.  If we have "nodebug" or
  // are emitting into a parameter context, then we are allowed to try this.
  if (funcOp.getAlwaysInlineLevel() != AlwaysInlineLevel::EnabledNoDebug &&
      emitter.builder)
    return {};

  // Perform parameter substitution if there are input parameters.
  // TODO: ParameterEvaluator paramEvaluator(inputParams);

  // Keep track of a mapping from the arguments (and interior results of
  // operations) to their representation.
  SmallDenseMap<Value, RValue> valueMapping;

  // Prime the arguments of the callee.
  auto &block = *funcOp.getBody();
  for (auto [blockArg, value] :
       llvm::zip(block.getArguments(), argumentValues)) {
    auto rValue = value.ir.getIfRValue();
    // We don't support folding by-ref arguments, we don't have an attribute
    // model for them.
    if (!rValue)
      return {};
    valueMapping[blockArg] = rValue;
  }

  SmallVector<Attribute> operandAttrs;
  SmallVector<OpFoldResult> foldResults;
  SmallVector<ParamConstantOp> materializedConstants;
  for (auto &op : block) {
    // First, check for our special cases.

    // If this is is the return operation then we're done.
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      assert(returnOp.getNumOperands() == 1 &&
             "Lit functions always return one value");
      return valueMapping[returnOp.getOperand(0)];
    }

    // 'let' declarations are noops.
    if (auto letDecl = dyn_cast<LetDeclOp>(op)) {
      // Note: Do not inline these two C++ statements, the hash table lookups
      // can invalidate each other.
      auto entry = valueMapping[letDecl.getValue()];
      valueMapping[letDecl.getResult()] = entry;
      continue;
    }

    // Ignore debuginfo.value operations entirely since we're dropping debug
    // info.
    if (isa<DebugInfo::ValueOp>(op))
      continue;

    // Clear all the vectors that are local state.  We define them outside the
    // loop just to avoid unneeded reallocation.
    materializedConstants.clear();
    operandAttrs.clear();
    foldResults.clear();

    // TODO: Add support for parameter substitution, how do we call fold
    // though?

    // Check to see if we can fold this operation.
    operandAttrs.reserve(op.getNumOperands());
    for (auto operand : op.getOperands()) {
      auto &entry = valueMapping[operand];
      assert(entry && "Value mapping broken");
      // If the input isn't an MValue then it is an error, let the caller
      // diagnose it.
      if (!entry.getIfMValue())
        return {};
      operandAttrs.push_back(entry.getIfMValue().get());
    }

    // Otherwise, bail out and allow the normal call procssing logic to
    // produce an apply of the original function.
    if (failed(op.fold(operandAttrs, foldResults)))
      return {};

    // We successfully folded this: remember the results.
    assert(foldResults.size() == op.getNumResults());
    for (auto [result, value] : llvm::zip(op.getResults(), foldResults)) {
      PointerUnion<Attribute, Value> puValue = value;
      if (auto drVal = dyn_cast<Value>(puValue))
        valueMapping[result] = DRValue(drVal);
      else {
        auto attr = dyn_cast<TypedAttr>(cast<Attribute>(puValue));
        assert(attr &&
               "Folding operation with typed result made untyped attr?");
        valueMapping[result] = MValue(attr);
      }
    }
  }

  // If we fell off the bottom of the function without finding a return, then
  // there is something wrong.  Don't fold it.
  return {};
}
