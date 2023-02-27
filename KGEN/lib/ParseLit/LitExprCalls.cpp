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
#include "LitParameterEvaluator.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"

#define DEBUG_TYPE "LITEXPRCALLS"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given the MLIR type for a variadic argument, return the element type as an
/// MLIR type.
static Type getVariadicElementType(Type variadicType) {
  auto mValue = PRValue(cast<VariadicType>(variadicType).getElementType());
  // VariadicType allows arbitrary parameter expressions, but we only ever
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
  PRValue infer(SignatureType signature, ArrayRef<ParamBindAttr> bindingsSoFar,
                ArrayRef<ASTExprAnd<AnyValue>> operands);

private:
  LogicalResult matchTypes(Type actualType, Type expectedType);
  LogicalResult matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);

  StringAttr parameterName;
  SmallVector<PRValue> inferredValues;
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

  // Handle VariadicType
  if (auto actual = dyn_cast<KGEN::VariadicType>(actualType))
    if (auto expected = dyn_cast<KGEN::VariadicType>(expectedType))
      return matchParams(actual.getElementType(), expected.getElementType());

  // If the types trivial match then we're done and there is no inference to do.
  if (actualType == expectedType)
    return success();

  // TODO: Could do StructType?
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
PRValue
ParameterInferenceState::infer(SignatureType signature,
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
    ValueInputConvention expectedConvention =
        signature.getInputConvention(expectedArgIdx);

    // Handle case when there are no more provided arguments.
    if (providedValueIdx == operands.size()) {
      // If the argument is a varargs argument list, then it can be initialized
      // with zero values no problem.
      if (signature.isVararg(expectedArgIdx))
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
      switch (expectedConvention) {
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
    if (!signature.isVararg(expectedArgIdx)) {
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
      !llvm::all_of(inferredValues, [&](PRValue v) -> bool {
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
    ExprEmitter &emitter, Operation *declOp, bool paramVarargs,
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
    auto diag = emitter.emitError(loc, "'")
                << baseName << "' expects " << expectedNumParams
                << " input parameter" << plural(expectedNumParams) << " but "
                << actualNumParams << plural(actualNumParams, " was", " were")
                << " provided";
    diag.attachNote(declOp->getLoc()) << "'" << baseName << "' declared here";
  };

  // If we have bound parameters, type check them now and bind names to them.
  SmallVector<ParamBindAttr> newBindings;
  newBindings.reserve(actualParamDecls.size());

  // We use the contextual emitter to perform implicit conversions, but these
  // conversions must be done within a parameter context.  Make sure we don't
  // have a builder from the caller, this indicates that an PRValue is required.
  llvm::SaveAndRestore savedBuilder(emitter.builder);
  emitter.builder.reset();

  // Parameters defined at the beginning of the parameter list may be used by
  // the types of other parameters defined later in the list, e.g. in:
  //    [rank: Int, indices: StaticTuple[rank]]
  // the value provided to 'indices' should actually depend on the specified
  // value of 'rank'.  We use a ParameterEvaluator to keep track of the mapping
  // so far and remap types on demand.
  LitParameterEvaluator evaluator(emitter.shared);
  size_t nextBinding = 0;
  for (auto [idx, declX] : llvm::enumerate(actualParamDecls)) {
    ParamDeclAttr decl = declX;
    bool isVararg = idx + 1 == actualParamDecls.size() && paramVarargs;

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
      if (isVararg) {
        auto emptyVariadic = KGEN::VariadicAttr::get(
            ArrayRef<TypedAttr>(), cast<KGEN::VariadicType>(decl.getType()));
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
                                          ASTType expectedType) -> PRValue {
      assert(binding.expr &&
             "should always have an expr tree for unchecked bindings");

      // Check the type matches what is expected, and perform an implicit
      // conversion if needed.
      expectedType = ASTType(evaluator.getReboundType(expectedType.mlirType));

      auto errorHandler = [&]() {
        if (declOp) {
          auto diag = emitter.emitError(binding.expr->getLoc(), "'")
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

      auto argValue =
          emitter.getAsExpectedType({PRValue(binding.getValue()), binding.expr},
                                    expectedType, ValueDest(), errorHandler);
      if (!argValue)
        return {};

      assert(argValue.getIfPRValue() &&
             "cannot emit a dynamic value in parameter context");
      return argValue.getIfPRValue();
    };

    // Scalar parameter values are installed directly.
    PRValue paramValue;
    if (!isVararg) {
      // Otherwise we get a single value.
      PRValue paramValue = handleSingleParameterValue(binding, decl.getType());
      if (!paramValue)
        return {};
      setParamValue(paramValue);
      continue;
    }

    // If the parameter is a variadic list, it may consume many values, and they
    // all get packed up into a VariadicAttr.
    SmallVector<TypedAttr> elements;
    auto variadicType = cast<VariadicType>(decl.getType());
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
    setParamValue(VariadicAttr::get(elements, variadicType));
  }

  // Check and complain if we have bindings that didn't get used.
  if (nextBinding != bindings.size()) {
    complainAboutParameterCount();
    return {};
  }

  return ParamBindArrayAttr::get(emitter.getContext(), newBindings);
}

//===----------------------------------------------------------------------===//
// DirectCallable Implementation
//===----------------------------------------------------------------------===//

/// Get a symbol for a direct reference to the specified function in its
/// enclosing context.  This does not bind any values to arguments.
DirectCallable::DirectCallable(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                               ParamBindArrayAttr bindingsAttr)
    : baseName(baseName), fnDecls(fnDecls.begin(), fnDecls.end()) {
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
                                  const ExprNode *callExpr,
                                  ExprEmitter &emitter);

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
OverloadFitness
OverloadFitness::evaluate(SignatureType signature,
                          const DirectCallable &callable,
                          ArrayRef<ASTExprAnd<AnyValue>> operands,
                          const ExprNode *callExpr, ExprEmitter &emitter) {

  // Check that the signature can be rebound with this set of bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;
  auto newBindings = callable.inputParamBindings.verifyBindings(
      signature.getInputParams(), callable.baseName, callExpr->getLoc(),
      incorrectBindingNo, incorrectBindingExpectedType, emitter,
      /*don't emit diagnostics*/ nullptr, signature.hasParamVarargs(),
      [&](ParamDeclAttr decl,
          ArrayRef<ParamBindAttr> bindingsSoFar) -> PRValue {
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
  for (auto [idx, convention] :
       llvm::enumerate(signature.getValueInputConventions())) {
    // Varargs arguments don't require a value, but allow any number of them.
    if (signature.isVararg(idx)) {
      maxAllowedargs = ~size_t(0);
      continue;
    }

    ++minRequiredArgs;
    ++maxAllowedargs;
  }

  // One less required argument for each argument that has a default value we
  // can use instead.
  minRequiredArgs -= signature.getDefaultArguments().size();

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
    ValueInputConvention expectedConvention =
        signature.getInputConvention(expectedArgIdx);
    unsigned argIdx = expectedArgIdx;

    // Handle case when there are no more provided arguments.
    if (providedValueIdx == operands.size()) {
      // If the argument is a varargs argument list, then it can be initialized
      // with zero values no problem.
      if (signature.isVararg(expectedArgIdx))
        break;
      // We don't need a provided value for this index if we can use a default
      // value, which has already been converted to the expected type.
      if (providedValueIdx >= signature.getValueInputs().size() -
                                  signature.getDefaultArguments().size())
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
      assert(!signature.isKWVararg(argIdx) &&
             "keyword arguments and `**arg` variadics not supported yet");
      switch (expectedConvention) {
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
                                                       emitter))
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
    if (!signature.isVararg(expectedArgIdx)) {
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
    PRValue eltTypeAttr = cast<POP::PointerType>(Type(type)).getElementType();
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
LogicalResult
DirectCallable::filterOverloadSet(ArrayRef<ASTExprAnd<AnyValue>> operands,
                                  CallSyntax syntax, const ExprNode *callExpr,
                                  bool emitDiagnosticOnFailure,
                                  ExprEmitter &emitter) {
  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<OverloadFitness> evaluations;
  bool anyValid = false;
  for (ASTDecl *candidate : fnDecls) {
    auto signature = cast<LIT::FuncOp>(*candidate).getFullSignature();
    evaluations.push_back(OverloadFitness::evaluate(signature, *this, operands,
                                                    callExpr, emitter));
    anyValid |= evaluations.back().kind == OverloadFitness::kValid;
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (!anyValid) {
    if (emitDiagnosticOnFailure) {
      // If there is a single callee, emit a specific error about the call.
      if (fnDecls.size() == 1) {
        auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
        auto diag = emitter.emitError(callExpr->getLoc(), "invalid call to '")
                    << baseName << "': " << callExpr->getRange();
        evaluations[0].diagnose(fnDecl.getFullSignature(), *this, operands,
                                syntax, diag);
        diag.attachNote(fnDecl.getLoc()) << "function declared here";
        return failure();
      }

      // Otherwise emit an error, and a note for what is wrong with each
      // candidate.
      auto diag = emitter.emitError(callExpr->getLoc(),
                                    "no matching function in call to '")
                  << baseName << "': " << callExpr->getRange();
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

  // If we found exactly one viable candidate, or all the overloads are marked
  // as adaptive, then we succeed.
  bool allMarkedAdaptive = llvm::all_of(newFnDecls, [](ASTDecl *decl) {
    return cast<LIT::FuncOp>(*decl).getIsAdaptive();
  });
  if (newFnDecls.size() == 1 || (!newFnDecls.empty() && allMarkedAdaptive)) {
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
    // We only want to suggest adding @adaptive if at least one in the set is
    // marked adaptive.
    bool anyMarkedAdaptive = llvm::any_of(newFnDecls, [](ASTDecl *decl) {
      return cast<LIT::FuncOp>(*decl).getIsAdaptive();
    });
    if (anyMarkedAdaptive) {
      auto diag = emitter.emitError(callExpr->getLoc(), "ambiguous call to '")
                  << baseName
                  << "', multiple implementations detected but not all are "
                     "marked adaptive, add @adaptive to all overloads"
                  << callExpr->getRange();
      for (LIT::FuncOp candidate : llvm::map_range(
               newFnDecls, [](ASTDecl *d) { return cast<LIT::FuncOp>(*d); })) {
        if (!candidate.getIsAdaptive())
          diag.attachNote(candidate.getLoc()) << "non-adaptive candidate here";
      }
    } else {
      auto diag = emitter.emitError(callExpr->getLoc(), "ambiguous call to '")
                  << baseName << "', each candidate requires " << minConversions
                  << " implicit conversion" << plural(minConversions)
                  << ", disambiguate with an explicit cast"
                  << callExpr->getRange();
      for (ASTDecl *candidate : newFnDecls)
        diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
            << "candidate declared here";
    }
  }
  return failure();
}

/// Resolve the callee into either a single PRValue callee (if there's only one
/// decl provided) or a variadic that contains all the possible adaptive
/// overloads. Because adaptive overloads must all have the same signature, this
/// also returns the signature type that they all share.
std::pair<PRValue, SignatureType> DirectCallable::getCallee() {
  assert(!fnDecls.empty() &&
         "cannot get the callee when no callees have been resolved");
  // Get the parameter bindings, if there are any.
  ParamBindArrayAttr bindArray = {};
  if (!inputParamBindings.bindings.empty()) {
    SmallVector<ParamBindAttr> binds;
    for (const InputParamBindings::Binding &b : inputParamBindings.bindings)
      binds.push_back(cast<ParamBindAttr>(b.bindingOrValue));
    assert(binds.size() == inputParamBindings.bindings.size() &&
           "some bindings were not bindings?");
    bindArray = ParamBindArrayAttr::get(binds.front().getContext(), binds);
  }
  if (fnDecls.size() == 1) {
    SymbolConstantAttr callee =
        cast<LIT::FuncOp>(*fnDecls.front()).getBoundReference(bindArray);
    return {PRValue(callee), callee.getType()};
  }

  // Otherwise, we have to construct a list to be called.
  SmallVector<TypedAttr> symbols =
      llvm::to_vector(llvm::map_range(fnDecls, [&](ASTDecl *decl) {
        return cast<TypedAttr>(
            cast<LIT::FuncOp>(*decl).getBoundReference(bindArray));
      }));
  // Pull out the type, and construct a list attr to be returned.
  SignatureType calleeType =
      cast<SymbolConstantAttr>(symbols.front()).getType();
  auto calleeList = VariadicAttr::get(symbols, VariadicType::get(calleeType));
  return {PRValue(calleeList), calleeType};
}

/// Utility function to perform subsitutions of the specified callable bindings
/// into the symbol for the given function declaration. It returns the resultant
/// SymbolConstantAttr or produces an error message and returns null.
static SymbolConstantAttr getBoundConstAttrFor(const DirectCallable *callable,
                                               const ExprNode *callExpr,
                                               ASTDecl *fnDecl,
                                               ExprEmitter &emitter) {
  auto funcOp = cast<LIT::FuncOp>(*fnDecl);

  // If there are no input parameters specified and if we allow unbound symbols,
  // just return the unbound symbol.
  if (callable->inputParamBindings.bindings.empty())
    return funcOp.getBoundReference();

  // Check that the signature can be rebound with our set of bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;

  auto newBindings = callable->inputParamBindings.verifyBindings(
      funcOp.getFullSignature().getInputParams(), callable->baseName,
      callExpr->getLoc(), incorrectBindingNo, incorrectBindingExpectedType,
      emitter, /*emit diagnostics*/ funcOp,
      funcOp.getSignature().hasParamVarargs());
  if (!newBindings)
    return {};

  // Now that we checked the types match, form the binding.
  return funcOp.getBoundReference(newBindings);
}

/// Perform subsitutions of the specified bindings into the symbol, returning
/// the resultant LITSymbolConstant attr or producing an error message and
/// returning null. This allows producing a reference to a parameterized
/// function without the parmaeters specified.  They can be bound later.
SymbolConstantAttr
DirectCallable::getBoundConstantAttr(const ExprNode *callExpr,
                                     ExprEmitter &emitter) const {
  if (fnDecls.size() != 1) {
    assert(!fnDecls.empty() && "DirectCallable malformed");
    auto diag = emitter.emitError(
                    callExpr->getLoc(),
                    "cannot form a reference to overloaded declaration of '")
                << baseName << "'" << callExpr->getRange();
    for (ASTDecl *candidate : fnDecls) {
      auto funcOp = cast<LIT::FuncOp>(*candidate);
      diag.attachNote(funcOp.getLoc()) << "candidate declared here";
    }

    return {};
  }
  return getBoundConstAttrFor(this, callExpr, fnDecls[0], emitter);
}

LogicalResult DirectCallable::getBoundConstantAttrsAdaptiveSet(
    SmallVectorImpl<TypedAttr> &symConstAttrs, const ExprNode *callExpr,
    ExprEmitter &emitter) const {
  for (ASTDecl *fnDecl : fnDecls) {
    auto funcOp = cast<LIT::FuncOp>(*fnDecl);
    if (!funcOp.getIsAdaptive()) {
      auto diag = emitter.emitError(callExpr->getLoc(),
                                    "cannot form a reference to non @adaptive "
                                    "declaration of '")
                  << baseName << "'" << callExpr->getRange();
      diag.attachNote(funcOp.getLoc()) << "declared here";
      return failure();
    }
    SymbolConstantAttr symConstAttr =
        getBoundConstAttrFor(this, callExpr, fnDecl, emitter);
    if (!symConstAttr)
      return failure();
    symConstAttrs.push_back(symConstAttr);
  }
  return success();
}

/// Check declarations for the result parameters and add them to
/// resultParamDecls.  This emits and error and returns failure if an error is
/// detected.
LogicalResult DirectCallable::getResultParamDecls(
    SignatureType signature, SmallVectorImpl<ParamDeclAttr> &resultParamDecls,
    ExprEmitter &emitter) {
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
CallableValue::CallableValue(ASTType type, StringRef methodName,
                             const ExprNode *callExpr, bool &erroneousDecl,
                             LitSharedState &shared)
    : expr(callExpr) {

  SMLoc callLoc = callExpr->getLoc();

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
  direct = DirectCallable{methodName, resultDecls, type.getParamBindings()};
}

/// Emit this as a flattened RValue or LValue with no additional parameter
/// context.  This returns null on failure.
AnyValue CallableValue::emitAsValue(ExprEmitter &emitter,
                                    ValueDest dest) const {
  // If we have no bound symbol, return the normal lvalue or rvalue we
  // represent.
  if (!direct)
    return emitter.emitResult(baseVal, expr, dest);

  // We allow unbound symbols here which can be emitted as an PRValue.  In the
  // case where we are partially applying, that will force the unbound symbol
  // into a SRValue which will catch symbols that are not fully bound.
  auto directSymbolAttr = direct->getBoundConstantAttr(expr, emitter);
  if (!directSymbolAttr)
    return {};

  // Verify that the target has no result parameters.  We have no way to bind
  // these indirectly.
  SignatureType calleeSignature = directSymbolAttr.getType();
  if (!calleeSignature.getResultParams().empty()) {
    emitter.emitError(expr->getLoc(),
                      "calls with result parameters must be called directly")
        << expr->getRange();
    return {};
  }

  // If we have no base value, then we are just a symbol, return it.
  if (!baseVal)
    return emitter.emitResult(directSymbolAttr, expr, dest);

  auto loc = expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  Type firstArgIRType = calleeSignature.getValueInputs()[0];
  Value firstArgValue;
  ValueInputConvention selfConvention = calleeSignature.getInputConvention(0);

  assert(!calleeSignature.isVararg(0) && !calleeSignature.isKWVararg(0) &&
         "Error: self shouldn't be able to be varargs");

  switch (selfConvention) {
  case ValueInputConvention::ByRef: {
    LValue baseLV = baseVal.getIfLValue();
    if (!baseLV) {
      emitter.emitError(loc,
                        "invalid use of mutating method on rvalue of type ")
          << ASTType(baseVal.getType()) << expr->getRange();
      return {};
    }
    firstArgValue = baseLV;

    // Using partial application over an lvalue isn't safe until we support an
    // ownership models with mutable borrows.
    emitter.emitError(loc, "TODO: partial application to mutable base isn't "
                           "supportable without a lifetime model")
        << expr->getRange();
    return {};
  }
  case ValueInputConvention::ByVal:
    // Otherwise we can have either an lvalue or rvalue, but we need to convert
    // to an rvalue if we have an lvalue.
    // TODO(memory_primary): Emit into memory directly.
    firstArgValue = emitter.emitSRValue({baseVal, expr});
    if (!firstArgValue)
      return {};
    break;
  }

  assert(firstArgIRType == firstArgValue.getType() &&
         "base types should always structurally line up");

  // For an instance value, we have to partially apply the callee to the first
  // argument of the reference.  Materialize callee as a SRValue for
  // partial_apply.
  // TODO(memory_primary): Emit into memory directly.
  auto calleeDRVal = emitter.emitSRValue({AnyValue(directSymbolAttr), expr});

  // Partial apply wants to know what operands to bind, we always bind the first
  // one.
  auto zeroAttr = emitter.builder->getAttr<mlir::DenseI64ArrayAttr>(0);
  auto result = SRValue(emitter.builder->create<POP::PartialApplyOp>(
      expr->getLocation(emitter), calleeDRVal, mlir::ValueRange(firstArgValue),
      zeroAttr));

  return emitter.emitResult(result, expr, dest);
}

LogicalResult CallableValue::emitAdaptiveSet(SmallVectorImpl<TypedAttr> &values,
                                             ExprEmitter &emitter) const {
  // If we have no bound symbol, bail out.
  if (!direct)
    return failure();

  return direct->getBoundConstantAttrsAdaptiveSet(values, expr, emitter);
}

/// Return true if 'value' may be implicitly converted to 'requiredType'
/// by invoking (one level of) conversion operations.  This does not generate
/// any IR.
bool CallableValue::canImplicitlyConvertToType(ASTExprAnd<AnyValue> value,
                                               ASTType requiredType,
                                               ExprEmitter &emitter) {
  // If it already matches, then we're done.
  if (value.ir.getRValueType().isEqualCanon(requiredType))
    return true;

  // Otherwise, check to see if we can do an implicit conversion by invoking a
  // `__new__` method on the expected type.
  bool isErroneousDecl = false;
  CallableValue callee(requiredType, "__new__", value.expr, isErroneousDecl,
                       emitter.shared);

  // If there are no viable candidates for the implicit conversion, we fail.
  if (!callee.direct)
    return false;

  // If we have at least one candidate, we check to see if any of them can
  // work. We disable implicit conversions though, to prevent converting
  // T -> S -> U in one step.
  callee.direct->disableImplicitConversions = true;
  return succeeded(callee.direct->filterOverloadSet(
      {value}, CallSyntax::kImplicitConvert, callee.expr,
      /*emitDiagnosticOnFailure=*/false, emitter));
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
                                ValueDest dest, CallSyntax syntax,
                                ExprEmitter &emitter) {
  if (isNull()) // Base was already diagnosed as an error.
    return {};

  // Used in some cases below, lifetime needs to exist for this whole method.
  SmallVector<ASTExprAnd<AnyValue>> operandsWithSelf;
  SMLoc callLoc = expr->getLoc();

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
      operandsWithSelf.push_back({baseVal, expr});
      operandsWithSelf.append(operands.begin(), operands.end());
      operands = operandsWithSelf;
      baseVal = {};
      assert(syntax == CallSyntax::kMethodCall && "Unexpected syntax form");
    }

    // Check the direct callees to see if they can be unambiguously resolved
    // with the bindings list and specified arguments.
    if (failed(direct->filterOverloadSet(operands, syntax, expr,
                                         /*emitDiagnosticOnFailure=*/true,
                                         emitter)))
      return {};

    std::tie(callee, calleeSig) = direct->getCallee();
    if (failed(
            direct->getResultParamDecls(calleeSig, resultParamDecls, emitter)))
      return {};

  } else {
    // Otherwise we have an indirect call. If the callee is an PRValue, emit a
    // `call_param`. Otherwise, emit the callee value as a SRValue so we can
    // call it with call_indirect.
    callee = baseVal.getIfPRValue();
    if (!callee) {
      callee = emitter.emitSRValue({baseVal, expr});
      if (!callee)
        return {};
    }

    calleeSig = dyn_cast<SignatureType>(callee.getType());
    if (!calleeSig) {
      emitError("invalid function type to call ")
          << ASTType(callee.getType()) << expr->getRange();
      return {};
    }

    // Check to see if we can apply these operands to the callee signature.
    DirectCallable bindings{"callee", /*params*/ {}, {}};
    auto fitness =
        OverloadFitness::evaluate(calleeSig, bindings, operands, expr, emitter);
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
  for (auto [idx, expectedTypeX, conventionX] : llvm::zip(
           llvm::seq<unsigned>(0, calleeSig.getValueInputs().size()),
           calleeSig.getValueInputs(), calleeSig.getValueInputConventions())) {
    // Work around lambda not being able to reference bindings.
    unsigned argIdx = idx;
    Type expectedType = expectedTypeX;
    ValueInputConvention convention = conventionX;
    // If we ran out of operands, fulfill this with a default value or empty
    // variadic list.
    if (nextOperandIdx == operands.size()) {
      // Varargs arguments are fulfilled with an empty !pop.variadic list.
      if (calleeSig.isVararg(argIdx)) {
        auto variadic = VariadicAttr::get(ArrayRef<TypedAttr>(),
                                          expectedType.cast<VariadicType>());
        argumentValues.push_back({PRValue(variadic), expr});
        continue;
      }
      // Otherwise, apply the default argument. We've ensured above that we have
      // a default argument for each missing operand.
      argumentValues.push_back(
          {PRValue(calleeSig.getDefaultArguments()[nextDefaultIdx]), expr});
      ++nextDefaultIdx;
      continue;
    }

    // Otherwise, we're applying one or more arguments to this.
    auto emitOneArgVal = [&](ASTExprAnd<AnyValue> operand) -> AnyValue {
      switch (convention) {
      case ValueInputConvention::ByRef:
        // By-ref arguments, must be lvalues.
        assert(operand.ir.getIfLValue() &&
               "Call should already be type checked");
        return operand.ir;
      case ValueInputConvention::ByVal:
        // by-val arguments are converted to the expected r-value type.
        // In the case of a variadic argument, we need to remove the
        // !pop.varadic<> wrapper to get the type to convert to.
        Type expectedArgType = expectedType;
        if (calleeSig.isVararg(argIdx))
          expectedArgType = getVariadicElementType(expectedArgType);

        operand.ir = emitter.getAsExpectedType(operand, expectedArgType,
                                               // TODO(memory-primary)
                                               ValueDest(), " in argument");
        return emitter.emitRValue(
            operand,
            // TODO(memory-primary): emit into the argument slot.
            ValueDest());
      }
    };

    // For a normal non-vararg argument, we just emit it and add it to our list.
    if (!calleeSig.isVararg(argIdx)) {
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
                    [](auto operand) { return operand.ir.getIfPRValue(); })) {
      SmallVector<TypedAttr> variadicArgs;
      for (auto operand : variadicOperands)
        variadicArgs.push_back(operand.ir.getIfPRValue().get());
      auto argAttr =
          VariadicAttr::get(variadicArgs, expectedType.cast<VariadicType>());
      argumentValues.push_back({PRValue(argAttr), variadicOperands[0].expr});
      continue;
    }

    // If not all operands are compile-time values, use an operation to create a
    // variadic sequence.
    SmallVector<Value> variadicArgs;
    for (auto &operand : variadicOperands) {
      // TODO(memory_primary): Emit into memory directly.
      SRValue argVal =
          emitter.emitSRValue({emitOneArgVal(operand), operand.expr});
      if (!argVal)
        return {};
      variadicArgs.push_back(argVal);
    }

    Location loc = emitter.translateLocation(callLoc);
    Value argVal = emitter.builder->create<POP::VariadicCreateOp>(
        loc, expectedType, variadicArgs);
    argumentValues.push_back({SRValue(argVal), variadicOperands[0].expr});
  }

  assert(nextOperandIdx == operands.size() &&
         "typechecking confirmed that we would use up all operands");

  // If this is a call to a @always_inline function (and there's only one
  // possible callee), see if we can fold its entire body into an PRValue. This
  // can fail for a number of reasons, in which case we fall back to emitting
  // normally.
  if (direct && direct->fnDecls.size() == 1) {
    auto calleeFunc = cast<LIT::FuncOp>(*direct->fnDecls[0]);
    if (calleeFunc.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled) {
      auto calleeSym = cast<SymbolConstantAttr>(callee.getIfPRValue().get());
      ParamBindArrayAttr inputParams = calleeSym.getParamValues();
      if (auto result = inlineFunctionCallIntoPRValue(
              *direct->fnDecls[0], inputParams, argumentValues, emitter))
        return emitter.emitResult(result.get(), expr, dest);
    }
  }

  auto &builder = emitter.builder;
  if (!builder) {
    // Emitting a call in a parameter context. Generate an apply operator.
    SmallVector<TypedAttr> operands({callee.getIfPRValue().get()});
    for (auto argValAndExpr : argumentValues) {
      if (!argValAndExpr.ir.getIfPRValue()) {
        emitter.emitError(argValAndExpr.expr->getLoc(),
                          "cannot use a dynamic value in parameter context")
            << argValAndExpr.expr->getRange();
        return {};
      }
      operands.push_back(argValAndExpr.ir.getIfPRValue().get());
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

    auto result = ParamOperatorAttr::get(POC::Apply, operands);
    return emitter.emitResult(result, expr, dest);
  }

  // Otherwise, materialize PRValue arguments as SRValues.
  SmallVector<Value> callArgs;
  for (auto argValAndExpr : argumentValues) {
    if (auto lv = argValAndExpr.ir.getIfLValue())
      callArgs.push_back(lv);
    else
      // TODO(memory_primary): Emit into memory directly.
      callArgs.push_back(emitter.emitSRValue(argValAndExpr));
    if (!callArgs.back())
      return {};
  }

  ArrayRef<Type> resultTypes = calleeSig.getValueResults();
  Operation *callOp;
  Location loc = emitter.translateLocation(callLoc);
  if (auto target = callee.getIfPRValue()) {
    if (auto sig = dyn_cast<SignatureType>(target.getType());
        sig && sig.isAsync()) {
      // If the callee is an async function, emit an async call.
      callOp = builder->create<AsyncCallOp>(loc, target.get(), resultParamDecls,
                                            callArgs);
    } else if (auto symbol = dyn_cast<SymbolConstantAttr>(target.get())) {
      // If the callee is a symbol constant, directly emit a call.
      callOp = builder->create<CallOp>(loc, resultTypes, symbol,
                                       resultParamDecls, callArgs);
    } else if (auto variadic = dyn_cast<VariadicAttr>(target.get())) {
      // If the callee is a list, create a param.fork op and create a CallParam
      // on that. We want to get the name of the function that is being called
      // and mangle it into the parameter name to ensure uniqueness.
      StringRef mangledCall(expr->getRangeStart().getPointer(),
                            expr->getRangeEnd().getPointer() -
                                expr->getRangeStart().getPointer() - 1);
      auto decl =
          ParamDeclAttr::get(builder->getStringAttr("(adaptive)" + mangledCall),
                             variadic.getType().getResolvedElementType());
      builder->create<ParamForkOp>(loc, decl, variadic);
      callOp = builder->create<CallParamOp>(loc, resultTypes,
                                            ParamDeclRefAttr::get(decl),
                                            resultParamDecls, callArgs);
    } else {
      callOp = builder->create<CallParamOp>(loc, resultTypes, target.get(),
                                            resultParamDecls, callArgs);
    }
  } else {
    // Otherwise emit calls to SSA values with call_indirect.
    callOp = builder->create<POP::CallIndirectOp>(
        loc, resultTypes, callee.getIfSRValue(), callArgs);
  }

  // If the callee can raise an error, try to unwrap it.
  if (calleeSig.isThrows() && !calleeSig.isAsync() &&
      !isValidErrorContext(builder->getInsertionBlock())) {
    emitError(
        "cannot call function that may raise in a context that cannot raise");
    return {};
  }

  // Value returning call returns its result.
  auto result = SRValue(callOp->getResult(0));
  return emitter.emitResult(result, expr, dest);
}

/// Given a call to an alwaysinline function that is invoked with simple
/// parameter constants, check to see if we can resolve it all the way down to
/// an PRValue.  We do this when in a parameter context (since there is no debug
/// info to ever generate) and when calling a "nodebug" function.
///
/// This is best-effort: when it fails, we fall back to emitting a normal call
/// or "apply" parameter expression.
PRValue CallableValue::inlineFunctionCallIntoPRValue(
    ASTDecl &callee, ParamBindArrayAttr inputParams,
    ArrayRef<ASTExprAnd<AnyValue>> argumentValues, ExprEmitter &emitter) {
  auto funcOp = cast<LIT::FuncOp>(callee);

  // TODO: We currently cannot handle calls to parameterized functions, we
  // aren't doing the substitution yet.
  if (!inputParams.empty())
    return {};

  // We aren't allowed to toss away debug information.  If we have "nodebug" or
  // are emitting into a parameter context, then we are allowed to try this.
  if (funcOp.getAlwaysInlineLevel() != AlwaysInlineLevel::EnabledNoDebug &&
      emitter.builder)
    return {};

  // We don't support folding by-ref or dynamic arguments and we only support
  // folding simple constants because we don't want to build massive parameter
  // expressions based on the internals of function calls.
  for (auto argValue : argumentValues) {
    auto mValue = argValue.ir.getIfPRValue();
    if (!mValue || !ParameterAttr::isSimpleConstant(mValue.get()))
      return {};
  }

  // Keep track of a mapping from the arguments (and interior results of
  // operations) to their representation, start with the input arguments.
  SmallDenseMap<Value, PRValue> valueMapping;
  auto &block = *funcOp.getBody();
  for (auto [blockArg, value] : llvm::zip(block.getArguments(), argumentValues))
    valueMapping[blockArg] = value.ir.getIfPRValue();

  // Resolve the body to type check and generate the IR we need for inlining.
  if (failed(emitter.getDeclResolver().resolveFully(callee, expr->getLoc())))
    return {};

  // Perform parameter substitution if there are input parameters.
  // TODO: ParameterEvaluator paramEvaluator(inputParams);
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
      operandAttrs.push_back(entry.get());
    }

    // Otherwise, bail out and allow the normal call procssing logic to
    // produce an apply of the original function.
    if (failed(op.fold(operandAttrs, foldResults)))
      return {};

    // We successfully folded this: remember the results.
    assert(foldResults.size() == op.getNumResults());
    for (auto [result, value] : llvm::zip(op.getResults(), foldResults)) {
      PointerUnion<Attribute, Value> puValue = value;
      // If the fold says the result is equal to one of the inputs, use the
      // known value from our mapping.
      if (auto drVal = dyn_cast<Value>(puValue))
        valueMapping[result] = valueMapping[drVal];
      else {
        auto attr = dyn_cast<TypedAttr>(cast<Attribute>(puValue));
        assert(attr &&
               "Folding operation with typed result made untyped attr?");
        valueMapping[result] = PRValue(attr);
      }
    }
  }

  // If we fell off the bottom of the function without finding a return, then
  // there is something wrong.  Don't fold it.
  return {};
}
