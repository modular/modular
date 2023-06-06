//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#include "CallEmission.h"
#include "ASTDecl.h"
#include "ExprEmitter.h"
#include "ParserParamEvaluator.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LifetimeTrackable.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include <limits>

#define DEBUG_TYPE "LITEXPRCALLS"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

void InputParamBindings::addPrechecked(TypedAttr precheckedBinding) {
  bindings.push_back({nullptr, precheckedBinding, /*typeChecked=*/true});
}

//===----------------------------------------------------------------------===//
// Parameter Inference Implementation
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
  PValue infer(SignatureType signature, ArrayRef<TypedAttr> bindingsSoFar,
               ArrayRef<ASTExprAnd<AnyValue>> operands);

private:
  LogicalResult matchTypes(Type actualType, Type expectedType);
  LogicalResult matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);

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

  // Handle POP::PointerType.
  if (auto actual = dyn_cast<POP::PointerType>(actualType))
    if (auto expected = dyn_cast<POP::PointerType>(expectedType))
      return matchParams(actual.getElementType(), expected.getElementType());

  // Handle VariadicType
  if (auto actual = dyn_cast<VariadicType>(actualType))
    if (auto expected = dyn_cast<VariadicType>(expectedType))
      return matchParams(actual.getElementType(), expected.getElementType());

  // If the types trivial match then we're done and there is no inference to do.
  if (actualType == expectedType)
    return success();

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

/// If the argument at the given index is of pack type, returns that type.
/// therwise, returns null.
static POP::PackType getIfPackType(SignatureType sig, size_t index) {
  return sig.isPackVararg(index)
             ? ::cast<POP::PackType>(sig.getValueInputs()[index])
             : nullptr;
}

/// Given an incomplete parameter binding set for a call to the specified
/// signature, try to infer the value of the next 'decl' parameter.  This should
/// always return null /without/ an error if it cannot be inferred, and return
/// a specific value if unambiguously determined.
PValue ParameterInferenceState::infer(SignatureType signature,
                                      ArrayRef<TypedAttr> bindingsSoFar,
                                      ArrayRef<ASTExprAnd<AnyValue>> operands) {
  // TODO: Apply the bindings so far (plus a distinct new attribute relating
  // back to the original decls for ones that are missing) to the signature with
  // getSpecializedSignature so we benefit from the already-fixed substitutions
  // being applied to the input types.  This can make them more concrete and
  // help with inferring dependent types based on already-bound parameters.

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t providedValueIdx = 0;
  for (auto [expectedArgIdx, expectedType] :
       llvm::enumerate(signature.getValueInputs())) {
    ValueInputConvention expectedConvention =
        signature.getInputConvention(expectedArgIdx);

    // There is no provided operand for a by-ref result.
    if (expectedConvention == ValueInputConvention::ByRefResult)
      continue;

    // Handle case when there are no more provided arguments.
    if (providedValueIdx == operands.size()) {
      // If the argument is a varargs argument list, then it can be initialized
      // with zero values no problem.
      if (signature.isVararg(expectedArgIdx))
        break;

      // TODO: If this argument is defaulted, infer against it.

      // If we have a pack argument, then we're binding zero type values to it.
      if (auto packType = getIfPackType(signature, expectedArgIdx)) {
        if (!inferredValues.empty())
          break;
        inferredValues.push_back(VariadicAttr::get(
            {}, cast<VariadicType>(packType.getVariadic().getType())));
        continue;
      }

      // Otherwise we have an argument count mismatch, just fail.
      return {};
    }

    // Otherwise we'll check the expected type against one (or more in the case
    // of varargs) provided values.
    auto checkOneOperand = [&](ASTType expectedType) -> LogicalResult {
      // We'll bind the next provided value.
      auto operand = operands[providedValueIdx++];
      switch (expectedConvention) {
      case ValueInputConvention::InitSelf:
        // If this is an UnknownAttr, then it is a placeholder for type
        // checking, just let it pass.
        if (auto pValue = operand.ir.getIfPValue())
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
                          expectedType.getPointerElementType());
      }

      case ValueInputConvention::OwnedInMem:
      case ValueInputConvention::BorrowedInMem:
        // Otherwise,we expect an r-value to match up, ignoring the pointer type
        // from the convention.
        expectedType = expectedType.getPointerElementType();
        [[fallthrough]];
      case ValueInputConvention::OwnedInReg:
      case ValueInputConvention::BorrowedInReg:
        // Otherwise, we pass as an r-value if we know the type.
        // TODO: Consider implicit conversions?
        if (auto c = operand.ir.getIfCValue())
          return matchTypes(c.getRValueType(), expectedType);
        // Consider the types of ORValues with single candidates.
        if (auto o = operand.ir.getIfORValue()) {
          if (o->fnDecls.size() == 1) {
            return matchTypes(
                cast<LIT::FuncOp>(*o->fnDecls.front()).getSignature(),
                expectedType);
          }
        }
        return success();
      }
    };

    // If we have a varargs argument, then it will eat the rest of the
    // arguments, but we have to check each of them.
    if (signature.isVararg(expectedArgIdx)) {
      auto varArgsEltType = ASTType(expectedType).getVariadicElementType();
      while (providedValueIdx != operands.size()) {
        if (failed(checkOneOperand(varArgsEltType)))
          return {};
      }
      continue;
    }

    // If we have a pack argument, then we're binding a variadic parameter with
    // multiple type values.  We need to consume all remaining arguments and use
    // their types as bindings.
    if (auto packType = getIfPackType(signature, expectedArgIdx)) {
      if (!inferredValues.empty())
        break;
      SmallVector<TypedAttr> types;
      while (providedValueIdx != operands.size()) {
        ASTExprAnd<AnyValue> operand = operands[providedValueIdx++];
        CValue value = operand.ir.getIfCValue();
        if (!value) {
          state.emitWarning(operand.expr->getLoc(),
                            "could not infer parameter type for this value, "
                            "because it is not concrete");
          return {};
        }
        types.push_back(
            ParameterizedTypeConstantAttr::get(value.getRValueType()));
      }

      inferredValues.push_back(VariadicAttr::get(
          types, cast<VariadicType>(packType.getVariadic().getType())));
      continue;
    }

    // In the typical case, this argument isn't varargs or a pack, so just check
    // it.  If there was a problem, report it, otherwise continue on to the next
    // expected argument to check.
    if (failed(checkOneOperand(expectedType)))
      return {};
  }

  // If we have left over operands, then this signature cannot match.
  if (providedValueIdx != operands.size() &&
      !bitEnumContainsAny(signature.getFnEffects(), FnEffects::ParamVararg))
    return {};

  // If we have no inferred values or if they disagree, then we fail to infer.
  if (inferredValues.empty() ||
      !llvm::all_of(inferredValues, [&](PValue v) -> bool {
        return v.get() == inferredValues.front().get();
      }))
    return {};

  return inferredValues.front();
}

//===----------------------------------------------------------------------===//
// InputParamBindings Implementation
//===----------------------------------------------------------------------===//

/// Check that our set of parameter bindings work with the specified input
/// parameters and call operands (if any), returning a checked
/// ParamBindArrayAttr if so.  If the parameters do not work, this emits an
/// diagnostic (if `declOp` is non-null) and sets
/// `incorrectBindingNo/Expectedtype` to the bad binding (or -1 if there is a
/// count mismatch).
///
/// This rejects the signature list if all the parameters are not bound.
ParameterExprArrayAttr InputParamBindings::verifyBindings(
    ArrayRef<Type> actualParamTypes, ParamDeclArrayAttr actualParamDecls,
    StringRef baseName, SMLoc loc, ssize_t &incorrectBindingNo,
    ASTType &incorrectBindingExpectedType, ExprEmitter &emitter,
    Operation *declOp, bool paramVarargs, bool packVarargs,
    ArrayRef<ASTExprAnd<AnyValue>> callOperands,
    ParameterInferenceHookTy parameterInferenceHook) const {

  // If we have an incorrect number of bindings specified, this lambda reports
  // the problem.
  auto complainAboutParameterCount = [&]() {
    // Tell the caller what went wrong.
    incorrectBindingNo = -1;
    if (!declOp)
      return;
    auto expectedNumParams = actualParamTypes.size();
    auto actualNumParams = bindings.size();
    auto diag = emitter.emitError(loc, "'")
                << baseName << "' expects " << expectedNumParams
                << " input parameter" << plural(expectedNumParams) << " but "
                << actualNumParams << plural(actualNumParams, " was", " were")
                << " provided";
    diag.attachNote(declOp->getLoc()) << "'" << baseName << "' declared here";
  };

  // If we have bound parameters, type check them now and bind names to them.
  SmallVector<TypedAttr> newBindings;
  newBindings.reserve(actualParamTypes.size());

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
  size_t nextBinding = 0;
  bool isPackVararg = packVarargs && !callOperands.empty();
  for (auto [idx, typeX] : llvm::enumerate(actualParamTypes)) {
    Type type = typeX;
    size_t index = idx;
    bool isVararg = idx + 1 == actualParamTypes.size() && paramVarargs;

    // This lambda installs the decl's value in the parameter evaluator and new
    // binding array.
    auto setParamValue = [&](TypedAttr value) {
      if (actualParamDecls)
        evaluator.setParameterValue(actualParamDecls[newBindings.size()],
                                    value);
      else
        evaluator.addInputValue(value);
      newBindings.push_back(value);
    };

    // Check to see if we ran out of bindings to provide to this param decl.
    if (nextBinding == bindings.size()) {
      // If we have a method to infer parameter values, invoke it to see if we
      // can get an inferred value for the parameter.
      if (parameterInferenceHook) {
        Type requestedType = evaluator.getReboundType(type);
        Type expectedType = requestedType;
        // If this is a vararg parameter, infer using the element type.
        if (isVararg && isa<VariadicType>(requestedType)) {
          expectedType =
              ASTType(cast<VariadicType>(expectedType).getElementType());
        }
        if (auto value = parameterInferenceHook(index, type, expectedType,
                                                newBindings)) {
          assert(value.getType().mlirType == requestedType &&
                 "inferred a default parameter value of wrong type");
          setParamValue(value);
          continue;
        }
      }

      // If the parameter decl is a variadic parameter list, and do not have
      // pack operands that could be used to infer those parameters, then we can
      // fulfill it with an empty list.  We know it must be the last parameter
      // decl.
      if (isVararg && !isPackVararg) {
        // If this isn't actually a variadic type, then we simply reached the
        // end of the parameter list.
        if (!isa<VariadicType>(type))
          continue;
        auto emptyVariadic =
            VariadicAttr::get(ArrayRef<TypedAttr>(), cast<VariadicType>(type));
        setParamValue(emptyVariadic);
        continue;
      }

      // TODO: Apply default values for parameters.

      // Otherwise, we're simply missing bindings.
      complainAboutParameterCount();
      return {};
    }

    auto binding = bindings[nextBinding++];
    // If this value was already bound and checked, use it.
    if (binding.typeChecked) {
      setParamValue(binding.value);
      continue;
    }

    auto handleSingleParameterValue = [&](Binding binding,
                                          ASTType expectedType) -> PValue {
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
        return argValue;
      }

      // Handle conversion failure with a custom error.
      incorrectBindingNo = newBindings.size();
      incorrectBindingExpectedType = expectedType;
      if (!declOp)
        return {};
      auto diag = emitter.emitError(binding.expr->getLoc(), "'")
                  << baseName << "' parameter #" << index << " has "
                  << expectedType << " type, but value has type "
                  << ASTType(binding.getValue().getType())
                  << binding.expr->getRange();
      diag.attachNote(declOp->getLoc()) << "'" << baseName << "' declared here";
      return {};
    };

    // Scalar parameter values are installed directly. Or, if we have a variadic
    // of the same type, we can use it as the value of the parameter directly.
    // FIXME: This allows passing a variadic `Ts` directly. Do we want a new
    // PValue classification for `*Ts`, which is required to pass this legally?
    if (!isVararg || binding.getValue().getType() == type) {
      PValue paramValue = handleSingleParameterValue(binding, type);
      if (!paramValue)
        return {};
      setParamValue(paramValue);
      continue;
    }

    // If the parameter is a variadic list, it may consume many values, and they
    // all get packed up into a VariadicAttr.
    SmallVector<TypedAttr> elements;
    Type expectedType = ASTType(type).getVariadicElementType();
    elements.push_back(handleSingleParameterValue(binding, expectedType));
    if (!elements.back())
      return {};
    while (nextBinding != bindings.size()) {
      binding = bindings[nextBinding++];
      elements.push_back(handleSingleParameterValue(binding, expectedType));
      if (!elements.back())
        return {};
    }
    setParamValue(VariadicAttr::get(
        elements, VariadicType::get(evaluator.getReboundType(expectedType))));
  }

  // Check and complain if we have bindings that didn't get used.
  if (nextBinding != bindings.size()) {
    complainAboutParameterCount();
    return {};
  }

  return ParameterExprArrayAttr::get(emitter.getContext(), newBindings);
}

/// Given a candidate that may or may not be compatible with the given
/// parameter set so far, indicate what the next parameter's expected type
/// should be, or return null if the current parameters are incompatible with
/// it.
ASTType
InputParamBindings::getNextExpectedBindingType(SignatureType candidateType,
                                               ExprEmitter &emitter) const {

  // We can get the next expected type by calling verifyBindings and seeing what
  // it queries for parameterInferenceHook.
  ASTType nextExpectedType;

  ssize_t incorrectBindingNo;
  ASTType incorrectBindingExpectedType;
  (void)verifyBindings(candidateType.getInputParamTypes(), {},
                       /*no diagnostics*/ "xx", SMLoc(), incorrectBindingNo,
                       incorrectBindingExpectedType, emitter,
                       /*don't emit diagnostics*/ nullptr,
                       candidateType.hasParamVarargs(),
                       candidateType.hasPackVarargs(), /*callOperands=*/{},
                       [&](size_t index, Type type, ASTType expectedType,
                           ArrayRef<TypedAttr> bindingsSoFar) -> PValue {
                         nextExpectedType = expectedType;
                         return {};
                       });
  return nextExpectedType;
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
    kArgGenericMem,    //< Argument bound from mlirtype to a memory-only type.
    kResultGenericMem, //< Result bound from mlirtype to a memory-only type.
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
  ///  kArgGenericMem:    the argument # that is a problem.
  ///  kResultGenericMem: the result #, always 0.
  size_t payload;

  /// For type mismatches, this is the actual or expected type, otherwise null.
  ASTType type;

  /// For valid candidates, this defines the parameter bindings to use.
  ParameterExprArrayAttr paramBindings;

  /// Determine whether the specified signature can be invoked with the
  /// parameter bindings specified in `callable` and the arguments specified in
  /// `operands`.
  static OverloadFitness evaluate(SignatureType signature,
                                  const OverloadSet &callable,
                                  ArrayRef<ASTExprAnd<AnyValue>> operands,
                                  bool allowImplicitConversions,
                                  ExprEmitter &emitter);

  /// Add explanation for why this candidate doesn't work to the specified
  /// diagnostic.
  void diagnose(SignatureType signature, const OverloadSet &callable,
                ArrayRef<ASTExprAnd<AnyValue>> operands, InflightDiag &diag);
};
} // namespace

/// Determine whether the specified signature can be invoked with the
/// parameter bindings specified in `callable` and the arguments specified in
/// `operands`.
OverloadFitness
OverloadFitness::evaluate(SignatureType signature, const OverloadSet &callable,
                          ArrayRef<ASTExprAnd<AnyValue>> operands,
                          bool allowImplicitConversions, ExprEmitter &emitter) {

  const ExprNode *callExpr = callable.expr;

  // Check that the signature can be rebound with this set of bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;
  ParameterExprArrayAttr newBindings =
      callable.inputParamBindings.verifyBindings(
          signature.getInputParamTypes(), {}, callable.baseName,
          callExpr->getLoc(), incorrectBindingNo, incorrectBindingExpectedType,
          emitter,
          /*don't emit diagnostics*/ nullptr, signature.hasParamVarargs(),
          signature.hasPackVarargs(), operands,
          [&](size_t index, Type type, ASTType expectedParamType,
              ArrayRef<TypedAttr> bindingsSoFar) -> PValue {
            return ParameterInferenceState(emitter.shared, index, type)
                .infer(signature, bindingsSoFar, operands);
          });

  // If there is an error, return the problem.
  if (!newBindings) {
    if (incorrectBindingNo == -1)
      return {kParamCount, 0, ASTType(), newBindings};
    return {kParamWrongType, static_cast<size_t>(incorrectBindingNo),
            incorrectBindingExpectedType, newBindings};
  }

  // Check the result parameter count.
  if (signature.getResultParamTypes().size() != callable.resultParams.size())
    return {kResultParamCount, 0, ASTType(), newBindings};

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  std::tie(signature, newBindings) =
      getUnboundSpecializedSignature(signature, newBindings);

  // Check that the result didn't bind to a type that would require changing to
  // a different result convention.
  for (auto output : signature.getValueResults())
    if (!ASTType(output).isRegisterPassable(callable.expr->getLoc(),
                                            emitter.shared))
      return {kResultGenericMem, 0, output, newBindings};

  // Ok, the parameters all line up, check the argument list.  We generally want
  // to diagnose problems where too few or too many arguments are passed if that
  // is the problem, rather than complaining about a type error of some argument
  // that doesn't work out.  Check for that first.
  size_t minRequiredArgs = 0;
  size_t maxAllowedargs = 0;
  for (auto [idx, convention] :
       llvm::enumerate(signature.getValueInputConventions())) {
    // Ignore the return slot if present.
    if (convention == ValueInputConvention::ByRefResult)
      continue;

    // Varargs arguments don't require a value, but allow any number of them.
    if (signature.isVararg(idx)) {
      maxAllowedargs = std::numeric_limits<size_t>::max();
      continue;
    }

    // Arguments with a pack type must have a known number of element types,
    // and so they require exactly that many arguments.
    if (auto packType = getIfPackType(signature, idx)) {
      size_t numValues = packType.getVariadicAttr().getValues().size();
      minRequiredArgs += numValues;
      maxAllowedargs += numValues;
      continue;
    }

    // Otherwise, we have an ordinary argument that requires a value.
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
  bool passesVarargArgument = false;

  // Use a ParserParamEvaluator to substitute 'apply' expressions in the
  // argument types.
  ParserParamEvaluator evaluator(emitter.getDeclResolver());
  for (auto [expectedArgIdxX, unboundExpectedType] :
       llvm::enumerate(signature.getValueInputs())) {
    size_t expectedArgIdx = expectedArgIdxX; // Workaround lambda problem.
    ValueInputConvention expectedConvention =
        signature.getInputConvention(expectedArgIdx);

    // Ignore the return slot if present.
    if (expectedConvention == ValueInputConvention::ByRefResult)
      continue;

    Type expectedType = evaluator.refineType(unboundExpectedType);

    // If the arguments or results got bound to a memory-only type then their
    // argument convention needs to change.  We cannot support this until we get
    // proper type traits.  Note that the POP::PointerType is considered a valid
    // register passable type, so things passed byref are ok.
    if (!ASTType(expectedType)
             .isRegisterPassable(callable.expr->getLoc(), emitter.shared))
      return {kArgGenericMem, expectedArgIdx, expectedType, newBindings};

    // Handle case when there are no more provided arguments.
    if (providedValueIdx == operands.size()) {
      // If the argument is a varargs argument list or pack, then it can be
      // initialized with zero values no problem.
      if (signature.isVararg(expectedArgIdx) ||
          signature.isPackVararg(expectedArgIdx)) {
        // We consider an empty varargs list to be an implicit conversion,
        // so an exact signature match takes precedence.
        ++numImplicitConversions;
        break;
      }
      // We don't need to provide value for this argument if it has a default
      // value.
      if (expectedArgIdx >= signature.getValueInputs().size() -
                                signature.getDefaultArguments().size())
        // In the callee, arguments with default values must be followed only by
        // other arguments with default values, so we do not need to enumerate
        // any more of the callee arguments.
        break;
    }

    // Otherwise we'll check the expected type against one (or more in the case
    // of varargs or packs) of the provided values. This reports any problems
    // with the operand type, or otherwise continues on to the next expected
    // argument to check.
    auto checkOneOperand = [&](ASTType expectedType) -> OverloadFitness {
      // We'll bind the next provided value.
      auto operand = operands[providedValueIdx];
      assert(!signature.isKWVararg(expectedArgIdx) &&
             "keyword arguments and `**arg` variadics not supported yet");
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
          return {kArgNotLValue, providedValueIdx, Type(), newBindings};

        // By-ref argument types must exactly match, no conversions are allowed.
        if (!argVal.getRValueType().isEqualCanon(
                expectedType.getPointerElementType()))
          return {kArgWrongLVType, providedValueIdx, expectedType, newBindings};
        break;
      }
      case ValueInputConvention::BorrowedInReg:
      case ValueInputConvention::BorrowedInMem:
      case ValueInputConvention::OwnedInReg:
      case ValueInputConvention::OwnedInMem:
        // Ignore the pointer type on memory conventions when matching types.
        // Note: Should do not support overloading on borrow/owned currently,
        // but we could add this if there is a reason to.
        if (expectedConvention == ValueInputConvention::OwnedInMem ||
            expectedConvention == ValueInputConvention::BorrowedInMem)
          expectedType = expectedType.getPointerElementType();

        // If the argument is an overload set, see if it can be resolve to the
        // right type.
        CValue argVal;
        if (auto orValue = operand.ir.getIfORValue()) {
          // If the overload set contains just a single candidate, it can be
          // used in implicit conversions. Materialize the function as a PValue.
          if (orValue->fnDecls.size() == 1) {
            argVal = orValue->fnDecls.front()->getFuncAsPValue();
          } else {
            argVal = orValue->filterOverloadSetForValueType(
                expectedType, /*emitDiagnosticOnFailure=*/false, emitter);
            if (!argVal)
              return {kArgWrongType, providedValueIdx, expectedType,
                      newBindings};
            break;
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

        // If we lack an exact match and conversions are disabled, this
        // candidate fails.
        if (!allowImplicitConversions ||
            !emitter.canImplicitlyConvertToType({argVal, operand.expr},
                                                expectedType))
          return {kArgWrongType, providedValueIdx, expectedType, newBindings};

        // If we had one, this bumps our # implicit conversions.
        ++numImplicitConversions;
        break;
      }

      // This provided value has been used up.
      ++providedValueIdx;
      return {kValid, 0, ASTType(), newBindings};
    };

    // If we have a varargs argument, then it will eat the rest of the
    // arguments, but we have to check each of them.
    if (signature.isVararg(expectedArgIdx)) {
      auto varArgsEltType = ASTType(expectedType).getVariadicElementType();
      while (providedValueIdx != operands.size()) {
        auto result = checkOneOperand(varArgsEltType);
        if (result.kind != kValid)
          return result;
        passesVarargArgument = true;
      }
      continue;
    }

    // If we have a pack type, it must have a known number of elements, and so
    // consumes exactly that number of arguments.
    if (auto packType = getIfPackType(signature, expectedArgIdx)) {
      for (TypedAttr element : packType.getVariadicAttr().getValues()) {
        OverloadFitness result = checkOneOperand(ASTType(element));
        if (result.kind != kValid)
          return result;
        passesVarargArgument = true;
      }
      continue;
    }

    // Otherwise, we have an ordinary argument that is not varargs or a pack.
    // Check it and move on to the next one.
    auto result = checkOneOperand(expectedType);
    if (result.kind != kValid)
      return result;
  }

  assert(providedValueIdx == operands.size() &&
         "should handle argument mismatch above");

  // Otherwise we succeeded!  For our payload, indicate the number of implicit
  // conversions and whether anything was passed through varargs.  We consider
  // exact matches of concrete types to be more specific than varargs matches.
  return {kValid, numImplicitConversions * 2 + (passesVarargArgument ? 1 : 0),
          ASTType(), newBindings};
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
        << "memory-primary type bound to generic result type: "
        << (lhsByRef ? "payload" : "argument") << " returns "
        << ASTType(lhsRetType) << " by reference";
  }
}

/// Add explanation for why this candidate doesn't work to the specified
/// diagnostic. isMethodCall indicates whether the call was written with
/// `foo(x,y)` syntax or `x.foo(y)` syntax.
void OverloadFitness::diagnose(SignatureType signature,
                               const OverloadSet &callable,
                               ArrayRef<ASTExprAnd<AnyValue>> operands,
                               InflightDiag &diag) {
  auto describePayloadArgumentNo = [&]() {
    // If this is a method syntax call, don't count the receiver.
    if (callable.syntax == CallSyntax::kMethodCall) {
      // it is probably possible for this assert to fire, if it does we should
      // tailor the error message.
      assert(payload != 0 && "TODO: unexpected self mismatch");
      diag << "method argument #" << (payload - 1);
    } else if (callable.syntax == CallSyntax::kOperator && payload == 1) {
      diag << "right side";
    } else if (callable.syntax == CallSyntax::kReversedOperator &&
               payload == 0) {
      diag << "left side";
    } else if (callable.syntax == CallSyntax::kSubscript && payload != 0) {
      if (payload == 1 && operands.size() == 2)
        diag << "index";
      else
        diag << "index #" << (payload - 1);
    } else if (callable.syntax == CallSyntax::kAttribute && payload != 0) {
      diag << "attribute name";
    } else {
      diag << "argument #" << payload;
    }
  };

  // This adds a string describing the type of the payload operand to the
  // diagnostic.
  auto getPayloadRValueType = [&] {
    if (auto cValue = operands[payload].ir.getIfCValue())
      return cValue.getRValueType();
    // If this is a single element overload set, then we can use the only
    // candidates type since it must not have worked out.
    const OverloadSet &ovset = *operands[payload].ir.getIfORValue();
    if (ovset.fnDecls.size() == 1)
      return ASTType(cast<LIT::FuncOp>(*ovset.fnDecls[0]).getSignature());
    return ASTType();
  };
  auto addPayloadRValueTypeName = [&]() {
    if (ASTType type = getPayloadRValueType())
      diag << type;
    else
      diag << "unknown overload";
  };

  switch (kind) {
  case kValid:
    diag << "candidate is viable";
    return;
  case kParamCount: {
    size_t actualNumBindings = callable.inputParamBindings.bindings.size();
    diag << "callee expects " << signature.getInputParamTypes().size()
         << " input parameter" << plural(signature.getInputParamTypes().size())
         << " but " << actualNumBindings
         << plural(actualNumBindings, " was", " were") << " provided";
    return;
  }
  case kParamWrongType: {
    auto binding = callable.inputParamBindings.bindings[payload];
    diag << "callee parameter #" << payload << " has " << ASTType(type)
         << " type, but value has type " << ASTType(binding.getType())
         << binding.expr->getRange();
    return;
  }
  case kResultParamCount:
    diag << "callee expects " << signature.getResultParamTypes().size()
         << " result parameter"
         << plural(signature.getResultParamTypes().size()) << " but "
         << callable.resultParams.size()
         << plural(callable.resultParams.size(), " was", " were")
         << " provided";
    return;
  case kArgCount:
    diag << "callee expects " << payload << " argument" << plural(payload)
         << ", but " << operands.size() << " "
         << plural(operands.size(), "was", "were") << " specified";
    return;
  case kArgTooFewAtLeast:
    diag << "callee expects at least " << payload << " argument"
         << plural(payload) << ", but " << operands.size() << " "
         << plural(operands.size(), "was", "were") << " specified";
    return;
  case kArgTooManyAtMost:
    diag << "callee expects at most " << payload << " argument"
         << plural(payload) << ", but " << operands.size() << " "
         << plural(operands.size(), "was", "were") << " specified";
    return;
  case kArgNotLValue:
    if (callable.syntax == CallSyntax::kMethodCall && payload == 0) {
      diag << "invalid use of mutating method on rvalue of type ";
      addPayloadRValueTypeName();
    } else {
      describePayloadArgumentNo();
      diag << " must be mutable in order to pass as a by-ref argument";
    }
    diag << operands[payload].expr->getRange();
    return;
  case kArgWrongLVType:
    diag << "l-value of type "
         << operands[payload].ir.getIfLValue().getRValueType()
         << " cannot be converted to reference of type "
         << type.getPointerElementType() << operands[payload].expr->getRange();
    return;

  case kArgWrongType: {
    describePayloadArgumentNo();
    diag << " cannot be converted from ";
    addPayloadRValueTypeName();
    SourceRange payloadLoc = operands[payload].expr->getRange();
    diag << " to " << type << payloadLoc;
    addTypeConversionDetail(diag, payloadLoc, getPayloadRValueType(), type);
    break;
  }

  case kArgGenericMem:
    describePayloadArgumentNo();
    diag << " cannot bind generic !mlirtype to memory-only type " << type;
    break;
  case kResultGenericMem:
    diag << "result cannot bind generic !mlirtype to memory-only type " << type;
    break;
  }
}

/// Evaluate the fnDecls candidates and see if there is an unambiguous
/// candidate that works with the specified parameter bindings and provided
/// arguments.  If so, return the single entry that works.  If not, generate a
/// diagnostic (when `emitDiagnosticOnFailure` is true) and return null.
PValue OverloadSet::filterOverloadSet(ArrayRef<ASTExprAnd<AnyValue>> operands,
                                      bool allowImplicitConversions,
                                      bool emitDiagnosticOnFailure,
                                      ExprEmitter &emitter) const {
  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<OverloadFitness> evaluations;
  bool anyValid = false;
  for (ASTDecl *candidate : fnDecls) {
    auto signature = cast<LIT::FuncOp>(*candidate).getFullSignature();
    evaluations.push_back(OverloadFitness::evaluate(
        signature, *this, operands, allowImplicitConversions, emitter));
    anyValid |= evaluations.back().kind == OverloadFitness::kValid;
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (!anyValid) {
    if (emitDiagnosticOnFailure) {
      // If there is a single callee, emit a specific error about the call.
      if (fnDecls.size() == 1) {
        auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
        auto diag = emitter.emitError(expr->getLoc(), "invalid call to '")
                    << baseName << "': " << expr->getRange();
        evaluations[0].diagnose(fnDecl.getFullSignature(), *this, operands,
                                diag);
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
        diag.attachNote(fnDecl->getLoc()) << "candidate not viable: ";
        eval.diagnose(fnDecl.getFullSignature(), *this, operands, diag);
      }
      return {};
    }
    return {};
  }

  // Ok, we have at least one valid candidate, filter the list to the ones with
  // the lowest number of implicit conversions required.
  size_t minConversions = std::numeric_limits<size_t>::max();
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
  auto allMarkedAdaptive = [&]() -> bool {
    return llvm::all_of(newFnDecls, [](ASTDecl *decl) {
      return cast<LIT::FuncOp>(*decl).getIsAdaptive();
    });
  };
  if (newFnDecls.size() == 1 || (!newFnDecls.empty() && allMarkedAdaptive())) {
    // On success, wrap things up into one callee.
    InputParamBindings newBindings;
    for (TypedAttr bind : oneFitness.paramBindings)
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
      // The numConversions field computed for kValue includes the number of
      // implicit conversions required but also uses the low bit to track the
      // whether a varargs conversion was used.  This allows us to treat varargs
      // as a less-specific match than an exact signature match (for example,
      // when overloading a `foo(Int)` and `foo(Int*)` we should pick the former
      // if both work.  That said, when we get here we don't want to complain
      // about the wrong number.
      size_t numConversions = minConversions >> 1;

      auto diag = emitter.emitError(expr->getLoc(), "ambiguous call to '")
                  << baseName << "', each candidate requires " << numConversions
                  << " implicit conversion" << plural(numConversions)
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
      [&](SignatureType candidateType) -> ParameterExprArrayAttr {
    // Apply any bound parameters to the candidate's type since they will be
    // applied when a reference is made.
    // TODO: Parameter inference.
    ssize_t incorrectBindingNo = 0;
    ASTType incorrectBindingExpectedType;
    return inputParamBindings.verifyBindings(
        candidateType.getInputParamTypes(), {}, baseName, expr->getLoc(),
        incorrectBindingNo, incorrectBindingExpectedType, emitter,
        /*don't emit diagnostics*/ nullptr, candidateType.hasParamVarargs());
  };

  auto isValidCandidate = [&](SignatureType candidateType) -> bool {
    // Apply any bound parameters to the candidate's type since they will be
    // applied when a reference is made.  We only do this if there are some
    // bindings present, because (unlike normal function calls) the result type
    // may have unbound parameters that we are trying to match, e.g. when in a
    // parameter expression context.
    if (!inputParamBindings.bindings.empty()) {
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

    return functionType.isEqualCanon(candidateType);
  };

  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<ASTDecl *> validCandidates;
  for (ASTDecl *candidate : fnDecls) {
    auto candidateType = cast<LIT::FuncOp>(*candidate).getFullSignature();
    if (isValidCandidate(candidateType))
      validCandidates.push_back(candidate);
  }

  // If we have exactly one viable candidate, then we succeed.
  auto allMarkedAdaptive = [&]() -> bool {
    return llvm::all_of(validCandidates, [](ASTDecl *decl) {
      return cast<LIT::FuncOp>(*decl).getIsAdaptive();
    });
  };

  // If we resolved to a single candidate or an adaptive set, then we succeed.
  if (validCandidates.size() == 1 ||
      (!validCandidates.empty() && allMarkedAdaptive())) {
    if (inputParamBindings.bindings.empty())
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
static TypedAttr getBoundConstAttrFor(LIT::FuncOp funcOp, StringRef baseName,
                                      InputParamBindings inputParamBindings,
                                      const ExprNode *expr,
                                      ExprEmitter &emitter) {

  // If there are no input parameters specified and if we allow unbound
  // symbols, just return the unbound symbol.
  if (inputParamBindings.bindings.empty())
    return funcOp.getBoundReference();

  // Check that the signature can be rebound with our set of bindings.
  ssize_t incorrectBindingNo = 0;
  ASTType incorrectBindingExpectedType;

  auto newBindings = inputParamBindings.verifyBindings(
      funcOp.getFullSignature().getInputParamTypes(), {}, baseName,
      expr->getLoc(), incorrectBindingNo, incorrectBindingExpectedType, emitter,
      /*emit diagnostics*/ funcOp, funcOp.getSignature().hasParamVarargs());
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
                                   InputParamBindings inputParamBindings,
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
                              InputParamBindings inputParamBindings,
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
  unsigned bufferID =
      emitter.getSourceMgr().FindBufferContainingLoc(expr->getLoc());
  auto [line, col] =
      emitter.getSourceMgr().getLineAndColumn(expr->getLoc(), bufferID);

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

//===----------------------------------------------------------------------===//
// OverloadSet Implementation
//===----------------------------------------------------------------------===//

/// Get a OverloadSet for a lookup of a named method on the specified type.
/// If successful, this provides a non-null OverloadSet.
///
/// On failure, this returns a null OverloadSet and invokes errorHandler if
/// the problem hasn't already been diagnosed. This does not emit an error on
/// failure.
OverloadSet::OverloadSet(ASTType type, StringRef methodName,
                         const ExprNode *expr, CallSyntax syntax,
                         SharedState &shared,
                         std::function<void()> errorHandler)
    : expr(expr), syntax(syntax) {

  // If this is a previously-reported error, ignore and don't report an
  // additional error.
  if (isa<TypeCheckErrorType>(type.mlirType))
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
                           ArrayRef<ASTExprAnd<AnyValue>> operands,
                           const ExprNode *callExpr, CallSyntax syntax,
                           ExprEmitter &emitter,
                           std::function<void()> errorHandler) {
  OverloadSet ovSet(type, methodName, callExpr, syntax, emitter.shared,
                    errorHandler);

  // If the core lookup failed, don't filter.
  if (ovSet.isNull())
    return {};

  // Filter the overload set with the actual operands list.  If this fails,
  // report an error (if we have an error handler) and reset to a null state so
  // the client can check this.
  bool shouldPrintError = bool(errorHandler);
  return ovSet.filterOverloadSet(operands, /*allowImplicitConversions=*/true,
                                 /*emitDiagnosticOnFailure=*/shouldPrintError,
                                 emitter);
}

/// Emit this as a CRValue if it can be resolved, otherwise emit an ambiguity
/// error and return null.
CValue OverloadSet::emitAsCValue(ExprEmitter &emitter, ValueDest &dest) {
  // If we have an overload set with multiple possibilities, we'll fail to emit
  // this as a CRValue.  Try to resolve it based on the destination's type.
  PValue directSymbolAttr;
  if (fnDecls.size() > 1) {
    if (ASTType expectedType = dest.resolveImpliedType(
            expr->getLoc(), /*no implied type*/ Type(), emitter)) {
      directSymbolAttr = filterOverloadSetForValueType(
          expectedType, /*emitDiagnosticOnFailure=*/true, emitter);
      if (!directSymbolAttr)
        return {};
    }
  }

  // We allow unbound symbols here which can be emitted as an PValue.  In the
  // case where we are partially applying, that will force the unbound symbol
  // into a SRValue which will catch symbols that are not fully bound.
  if (!directSymbolAttr) {
    directSymbolAttr = getBoundConstantAttr(emitter);
    if (!directSymbolAttr)
      return {};
  }

  // Verify that the target has no result parameters.  We have no way to bind
  // these indirectly.
  auto calleeSignature =
      cast<SignatureType>(directSymbolAttr.getType().mlirType);
  if (!calleeSignature.getResultParamTypes().empty()) {
    emitter.emitError(expr->getLoc(),
                      "calls with result parameters must be called directly")
        << expr->getRange();
    return {};
  }

  // If we have no base value, then we are just a symbol, return it.
  if (!baseValue)
    return emitter.emitCResult(directSymbolAttr, expr, dest);

  auto loc = baseValue.expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  Type firstArgIRType = calleeSignature.getValueInputs()[0];
  ValueInputConvention selfConvention = calleeSignature.getInputConvention(0);
  Value firstArgValue;

  assert(!calleeSignature.isVararg(0) && !calleeSignature.isKWVararg(0) &&
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
CValue OverloadSet::emitCall(ArrayRef<ASTExprAnd<AnyValue>> operands,
                             ValueDest &dest, ExprEmitter &emitter) {
  if (isNull()) // Base was already diagnosed as an error.
    return {};

  // Used in some cases below, lifetime needs to exist for this whole method.
  SmallVector<ASTExprAnd<AnyValue>> operandsWithSelf;

  // If we have a bound self, add it to the operand list to simplify the logic
  // below.
  if (baseValue) {
    operandsWithSelf.reserve(operands.size() + 1);
    operandsWithSelf.push_back(baseValue);
    operandsWithSelf.append(operands.begin(), operands.end());
    operands = operandsWithSelf;
    baseValue = {};
    assert(syntax == CallSyntax::kMethodCall && "Unexpected syntax form");
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
  assert(calleeSig.getResultParamTypes().size() == resultParams.size() &&
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

/// Emit an indirect call to a resolved value.
CValue ExprEmitter::emitIndirectCall(CValue callee,
                                     ArrayRef<ASTExprAnd<AnyValue>> operands,
                                     ValueDest &dest,
                                     const ExprNode *callExpr) {
  auto calleeSig = dyn_cast<SignatureType>(callee.getRValueType().mlirType);
  if (!calleeSig) {
    // If we are invoking something other than a SignatureType, try to invoke
    // its `__call__` method.
    SmallVector<ASTExprAnd<AnyValue>> callOperands;
    callOperands.push_back({callee, callExpr});
    llvm::append_range(callOperands, operands);
    return emitNamedMethodCall("__call__", callOperands, dest,
                               CallSyntax::kDirectCall, callExpr);
  }

  assert(calleeSig.getResultParamTypes().empty());

  // If we have a function pointer, resolve it to an RValue.
  CRValue calleeRV = emitCRValue({callee, callExpr}, EC_CallCalleeValue);
  if (!calleeRV)
    return {};

  // Check to see if we can apply these operands to the callee signature.
  OverloadSet bindings{"callee", /*params=*/{}, ParamBindArrayAttr(), callExpr,
                       CallSyntax::kIndirectCall};
  auto fitness =
      OverloadFitness::evaluate(calleeSig, bindings, operands,
                                /*allowImplicitConversions=*/true, *this);
  if (fitness.kind != OverloadFitness::kValid) {
    // If not, diagnose it with an error.
    auto diag = emitError(callExpr->getLoc(), "invalid indirect call: ");
    fitness.diagnose(calleeSig, bindings, operands, diag);
    return {};
  }

  return emitCallUnchecked(calleeRV, operands, /*resultParams=*/{}, dest,
                           callExpr);
}

/// folded into a PValue.
static FailureOr<TypedAttr>
inlineFunctionCallIntoPValue(AnyValue callee,
                             ArrayRef<ASTExprAnd<AnyValue>> argumentValues,
                             ParserParamEvaluator &evaluator) {
  auto calleePR = callee.getIfPValue();
  if (!calleePR)
    return failure();
  auto calleeSymbolCst = dyn_cast<SymbolConstantAttr>(calleePR.get());
  if (!calleeSymbolCst)
    return failure();
  SmallVector<Attribute> arguments;
  for (auto argValue : argumentValues) {
    auto mValue = argValue.ir.getIfPValue();
    if (!mValue || !ParameterAttr::isSimpleConstant(mValue.get()))
      return failure();
    arguments.push_back(mValue.get());
  }
  return evaluator.evaluateFunctionCall(calleeSymbolCst.getSymbol(), arguments);
}

/// Given a call to a function with a memory only result and the desired value
/// destination, decide if it is safe to directly emit into the slot.  Doing so
/// requires a form of alias analysis to determine whether any input arguments
/// could alias the result slot.  We cannot emit into the result slot when
/// passing the value as an argument like 'x = foo(x)' or 'x = x + 1'.
///
/// At this point, we've already applied implicit conversions and converted
/// things to RValues or BValues as required by the argument convention, but
/// things may still be in parameter space.
static bool isSafeToUseValueDestForDirectResult(
    ASTType destRValueType, ValueDest &dest,
    ArrayRef<ASTExprAnd<AnyValue>> argValues,
    ArrayRef<ValueInputConvention> argConventions, ExprEmitter &emitter) {
  // Drop the first argument which is the return slot.
  assert(argConventions[0] == ValueInputConvention::ByRefResult);
  argValues = argValues.drop_front();
  argConventions = argConventions.drop_front();

  // Check to see if the destination provides a buffer.  If not, it is safe to
  // emit into it, but it doesn't actually matter.
  Value destBuffer = dest.getDefinedSLValueIfExists(destRValueType, emitter);
  if (!destBuffer)
    return true;

  // See if the destination buffer is something that ownership can track.  If
  // not, we cannot make reliable determinations about aliasing.
  Value underlyingDest =
      LifetimeTrackable::findUnderlyingValueFromField(destBuffer);
  if (!underlyingDest)
    return false;

  // Check to see if the specified argument value pointer could alias with the
  // destination buffer, returning true if it might.  We can only disambiguate
  // this safely when we can prove that the pointer points to a different
  // distinguishable object than the result slot.
  // TODO: This will need to be extended to support lifetimes.
  auto ptrGuaranteedNoAlias = [&](Value ptrVal) -> bool {
    Value underlyingPtr =
        LifetimeTrackable::findUnderlyingValueFromField(ptrVal);
    return underlyingPtr && underlyingPtr != underlyingDest;
  };

  // If any of the arguments might alias, then we need to use a temporary
  // buffer.
  for (auto [value, convention] : llvm::zip(argValues, argConventions)) {
    switch (convention) {
    case ValueInputConvention::OwnedInReg:
    case ValueInputConvention::BorrowedInReg:
      // Register conventions can never alias the result.
      continue;

    case ValueInputConvention::OwnedInMem:
    case ValueInputConvention::BorrowedInMem:
    case ValueInputConvention::ByRefResult:
    case ValueInputConvention::ByRef:
    case ValueInputConvention::InitSelf:
      // Parameter values will never alias.
      if (value.ir.getIfPValue())
        continue;
      if (auto sl = value.ir.getIfSLValue()) {
        if (ptrGuaranteedNoAlias(sl))
          continue;
        return false;
      }
      if (auto mb = value.ir.getIfMBValue()) {
        if (ptrGuaranteedNoAlias(mb))
          continue;
        return false;
      }
      if (auto mb = value.ir.getIfMRValue()) {
        if (ptrGuaranteedNoAlias(mb))
          continue;
        return false;
      }
      // Dynamic variadic memory values are passed with a pop.variadic.create,
      // check each field.
      if (auto sr = value.ir.getIfSRValue()) {
        if (auto variadic = sr.getDefiningOp<POP::VariadicCreateOp>()) {
          for (auto operand : variadic.getOperands()) {
            if (!ptrGuaranteedNoAlias(operand))
              return false;
          }
          continue;
        }
      }
      llvm_unreachable("Unknown value kind for memory convention");
    }
  }

  // If no problems are found, it is safe!
  return true;
}

CValue ExprEmitter::emitCallUnchecked(CRValue callee,
                                      ArrayRef<ASTExprAnd<AnyValue>> operands,
                                      ArrayRef<ParamDeclAttr> resultParams,
                                      ValueDest &dest,
                                      const ExprNode *callExpr) {
  SignatureType calleeSig = cast<SignatureType>(callee.getType().mlirType);
  Location loc = translateLocation(callExpr->getLoc());
  SmallVector<ASTExprAnd<AnyValue>> argumentValues;

  assert(calleeSig.getResultParamTypes().size() == resultParams.size() &&
         "Type checking should be done");

  /// This struct accumulates information about IR to emit after the call, e.g.
  /// writebacks for computed inout lvalues, and lifetime markers.
  struct AfterCallActions {
    const ExprNode *expr;
    Location loc;

    // The first entry of this is a ValueDest for a DLValue that we can invoke
    // for the setter.
    SmallVector<std::pair<ValueDest, SLValue>> lvalueWritebacks;

    /// This is a list of values that we need to keep alive across the duration
    /// of the call.  They will get lit.ownership.use operations at the end of
    /// the call.
    SmallVector<Value> valuesToKeepAlive;

    AfterCallActions(const ExprNode *expr, Location loc)
        : expr(expr), loc(loc) {}

    void emit(ExprEmitter &emitter) {
      // Emit the elements and clear the writebacks so the ValueDest's get
      // destroyed when they are emitted into.
      while (!lvalueWritebacks.empty()) {
        auto elt = lvalueWritebacks.pop_back_val();
        if (!emitter.emitResult(MRValue(elt.second), expr, elt.first))
          elt.first.resetForError();
      }

      // Emit all the lit.ownership.use ops.
      for (auto value : valuesToKeepAlive)
        emitter.builder->create<OwnershipUseOp>(loc, value);
    }

    // If an error happens before we emit the write backs, make sure to nuke
    // them so they don't crash the compiler.
    ~AfterCallActions() {
      // If any error occurs during IR emission, these won't be emitted.
      while (!lvalueWritebacks.empty())
        lvalueWritebacks.pop_back_val().first.resetForError();
    }
  } afterCallActions(callExpr, loc);

  /// This function emits the specified pre-emitted argument into a single MLIR
  /// Value suitable for passing to the callee with the specified convention.
  /// This handles promotion of PValues to dynamic values as needed.
  auto emitPreemittedArgumentAsDynamicValue =
      [&](ASTExprAnd<AnyValue> argValAndExpr,
          ValueInputConvention convention) -> Value {
    Value arg;
    switch (convention) {
    case ValueInputConvention::OwnedInReg:
      // Promote PValue's if needed.
      return emitSRValue(argValAndExpr, EC_CallArgValue);
    case ValueInputConvention::OwnedInMem:
      arg = argValAndExpr.ir.getIfMRValue();
      break;
    case ValueInputConvention::BorrowedInReg:
      if (auto pVal = argValAndExpr.ir.getIfPValue())
        return arg = emitSRValue(argValAndExpr, EC_CallArgValue);

      // If this is an MBValue, the element must be register passable but not
      // loaded.
      if (auto mbVal = argValAndExpr.ir.getIfMBValue()) {
        const ExprNode *expr = argValAndExpr.expr;
        // TODO: Factor this into a helper.
        if (!builder) {
          emitErrorForDynamicValueInParameter(expr);
          return {};
        }
        auto load =
            builder->create<POP::LoadOp>(expr->getLocation(*this), mbVal,
                                         /*alignment=*/std::nullopt);
        argValAndExpr.ir = SBValue(load);
      }

      arg = argValAndExpr.ir.getIfSBValue();
      break;
    case ValueInputConvention::BorrowedInMem:
      arg = argValAndExpr.ir.getIfMBValue();
      break;
    case ValueInputConvention::ByRefResult: {
      auto tmpSlotAddr = argValAndExpr.ir.getIfSLValue();
      assert(tmpSlotAddr && "byref_result value start in a temp slot");
      auto rvalueType = ASTType(tmpSlotAddr.getType()).getPointerElementType();

      // Often the result of the call will be directly assigned into a
      // user-defined var or other location with existing storage.  In these
      // cases, we really want to assign directly into the existing slot.
      //
      // However, we cannot do that if the destination slot is also being passed
      // into the call as an input value, as in: `x = foo(x)` or `x = x + 1`.
      // In these cases we really do need a temporary+copy in the var slot.
      // At this point we've got enough information about the arguments to make
      // that assessment in a correct way.
      if (!isSafeToUseValueDestForDirectResult(
              rvalueType, dest, argumentValues,
              calleeSig.getValueInputConventions(), *this))
        return tmpSlotAddr;

      // Okay it is safe to use, so remove the temporary allocation we aren't
      // going to use.
      tmpSlotAddr.getDefiningOp<VarLetDeclOp>()->erase();
      // Get the SLValue of the destination slot.
      return dest.getSLValueForResult(callExpr->getLoc(), rvalueType, *this);
    }
    case ValueInputConvention::ByRef:
    case ValueInputConvention::InitSelf: {
      // We know that the operand is an LValue, but it might be
      // dynamic/computed.
      LValue lv = argValAndExpr.ir.getIfLValue();
      assert(lv && "type checking ensures we will have an lvalue");
      if (auto sl = lv.getIfSLValue())
        return sl;

      // If dynamic, we need to generate a temporary slot, emit a 'get' into
      // that slot, pass the address, then write it back when we're done.
      ValueDest dlvBuffer(lv, EC_CallArgValue);
      SLValue slvBuffer = dlvBuffer.getSLValueForResult(
          argValAndExpr.expr->getLoc(), lv.getRValueType(), *this);
      // Emit the 'get' into the buffer.
      ValueDest bufferDest(slvBuffer, EC_CallArgValue);
      if (!emitLoadOfLValue({lv, argValAndExpr.expr}, bufferDest)) {
        bufferDest.resetForError();
        dlvBuffer.resetForError();
        return {};
      }
      afterCallActions.lvalueWritebacks.push_back(
          {std::move(dlvBuffer), slvBuffer});
      return slvBuffer;
    }
    }
    if (!arg) {
      llvm::errs() << "CALL ARG MISMATCH: " << int(convention) << " ";
      argValAndExpr.ir.dump();
      llvm_unreachable("didn't get a value as expected");
    }
    return arg;
  };

  // Emit all the arguments.  We iterate by expected arguments since we're
  // building the argument list of the call.  Default arguments and
  // variadics get filled in here.
  size_t nextOperandIdx = 0;
  size_t nextDefaultIdx = 0;

  // Use a ParserParamEvaluator to fold only 'apply' expressions. Emit a rebind
  // if the refined type is different than the expected type.
  ParserParamEvaluator evaluator(getDeclResolver());
  for (auto [idx, expectedTypeX, conventionX] : llvm::zip(
           llvm::seq<unsigned>(0, calleeSig.getValueInputs().size()),
           calleeSig.getValueInputs(), calleeSig.getValueInputConventions())) {
    // Work around lambda not being able to reference bindings.
    unsigned argIdx = idx;
    Type expectedType = evaluator.refineType(expectedTypeX);
    ValueInputConvention convention = conventionX;

    // If this is the return slot for a call, we want to propagate the ValueDest
    // into this, but we need information about each argument being emitted
    // before we can do that.  As such, we just use a var decl and replace it
    // opportunistically later if we can.
    if (convention == ValueInputConvention::ByRefResult) {
      if (!builder) {
        // TODO: Support memory-primary results in parameter expressions
        emitError(callExpr->getLoc(), "TODO: memory-primary results are not "
                                      "supported in parameter expressions.");
        return {};
      }
      assert(idx == 0 && calleeSig.hasMemoryOnlyResult());
      auto resultTmp = builder->create<VarLetDeclOp>(
          loc, expectedType, "__call_result_tmp__", /*isVar=*/true,
          /*isSynth=*/true);
      argumentValues.push_back({SLValue(resultTmp), callExpr});
      continue;
    }

    // If we ran out of operands, fulfill this with a default value, empty
    // variadic list, or empty pack.
    if (nextOperandIdx == operands.size()) {
      // Varargs arguments are fulfilled with an empty !kgen.variadic list.
      if (calleeSig.isVararg(argIdx)) {
        auto variadic = VariadicAttr::get(ArrayRef<TypedAttr>(),
                                          expectedType.cast<VariadicType>());
        argumentValues.push_back({PValue(variadic), callExpr});
        continue;
      }

      // Pack arguments are fulfilled with an empty !pop.pack sequence.
      if (auto packType = getIfPackType(calleeSig, argIdx)) {
        assert(packType.isEmpty() &&
               "pack type already checked against operand count");
        auto pack = POP::PackAttr::get(ArrayRef<TypedAttr>(), packType);
        argumentValues.push_back({PValue(pack), callExpr});
        continue;
      }

      // Otherwise, apply the default argument. We've ensured above that we
      // have a default argument for each missing operand.
      argumentValues.push_back(
          {PValue(calleeSig.getDefaultArguments()[nextDefaultIdx]), callExpr});
      ++nextDefaultIdx;
      continue;
    }

    // Otherwise, we're applying one or more arguments to this.
    auto emitOneArgVal = [&](ASTExprAnd<AnyValue> operand,
                             size_t sequenceIndex = 0) -> AnyValue {
      switch (convention) {
      case ValueInputConvention::ByRef:
      case ValueInputConvention::ByRefResult:
      case ValueInputConvention::InitSelf:
        // By-ref arguments, must be lvalues.
        assert(operand.ir.getIfLValue() &&
               "Call should already be type checked");
        return operand.ir;
      case ValueInputConvention::OwnedInReg:
      case ValueInputConvention::OwnedInMem:
      case ValueInputConvention::BorrowedInReg:
      case ValueInputConvention::BorrowedInMem:
        // by-val arguments are converted to the expected r-value type.
        ASTType expectedArgType = expectedType;
        if (calleeSig.isVararg(argIdx))
          // In the case of a variadic argument, we need to remove the
          // !pop.variadic<> wrapper to get the type to convert to.
          expectedArgType = expectedArgType.getVariadicElementType();
        else if (auto packType = getIfPackType(calleeSig, argIdx))
          // Operands being applied to a concrete pack type argument must be
          // converted to the pack element type at that index.
          expectedArgType =
              packType.getVariadicAttr().getValues()[sequenceIndex];

        if (convention == ValueInputConvention::OwnedInMem ||
            convention == ValueInputConvention::BorrowedInMem)
          expectedArgType = expectedArgType.getPointerElementType();

        if (convention == ValueInputConvention::OwnedInReg ||
            convention == ValueInputConvention::OwnedInMem)
          return emitRValue(operand, EC_CallArgValue, expectedArgType);
        return emitBValue(operand, EC_CallArgValue, expectedArgType);
      }
      llvm_unreachable("unknown value input convention");
    };

    // For a normal (not a vararg or a pack) argument, we just emit it and add
    // it to our list.
    if (!calleeSig.isVararg(argIdx) && !isa<POP::PackType>(expectedType)) {
      auto operand = operands[nextOperandIdx++];
      AnyValue argVal = emitOneArgVal(operand);
      if (!argVal)
        return {};
      argumentValues.push_back({argVal, operand.expr});
      continue;
    }

    // For a variadic or pack sequence, we need to emit all of the remaining
    // operands. Emit all of the remaining values to make sure they're converted
    // to the right type.
    SmallVector<ASTExprAnd<AnyValue>> remainingOperands(
        operands.begin() + nextOperandIdx, operands.end());
    for (auto [idx, operand] : llvm::enumerate(remainingOperands)) {
      auto emittedArg = emitOneArgVal(operand, idx);
      if (!emittedArg)
        return {};
      operand.ir = emittedArg;
    }
    nextOperandIdx = operands.size();

    // If all of the operands are compile-time values, then we can represent
    // the sequence as an attribute.
    if (std::all_of(remainingOperands.begin(), remainingOperands.end(),
                    [](auto operand) { return operand.ir.getIfPValue(); })) {
      SmallVector<TypedAttr> args;
      for (auto operand : remainingOperands)
        args.push_back(operand.ir.getIfPValue().get());
      Attribute attr;
      if (calleeSig.isVararg(argIdx))
        attr = VariadicAttr::get(args, expectedType.cast<VariadicType>());
      else
        attr = POP::PackAttr::get(args, expectedType.cast<POP::PackType>());
      argumentValues.push_back({PValue(attr), remainingOperands[0].expr});
      continue;
    }

    // If not all operands are compile-time values, use an operation to
    // create a variadic or pack sequence.
    SmallVector<Value> args;
    for (auto &operand : remainingOperands) {
      Value argVal = emitPreemittedArgumentAsDynamicValue(operand, convention);
      if (!argVal)
        return {};
      args.push_back(argVal);

      // Make sure the values in the pack stay live across the entire call, not
      // just the pop.variadic.create op.
      bool isTrivial = false;
      if (auto cv = operand.ir.getIfCValue())
        isTrivial = cv.getRValueType().isTrivial(callExpr->getLoc(), shared);
      if (!isTrivial)
        afterCallActions.valuesToKeepAlive.push_back(argVal);
    }

    Location loc = translateLocation(callExpr->getLoc());
    Value argVal;
    if (calleeSig.isVararg(argIdx))
      argVal = builder->create<POP::VariadicCreateOp>(loc, expectedType, args);
    else
      argVal = builder->create<POP::PackCreateOp>(loc, expectedType, args);
    argumentValues.push_back({SRValue(argVal), remainingOperands[0].expr});
  }

  assert(nextOperandIdx == operands.size() &&
         "typechecking confirmed that we would use up all operands");

  // If this is a call to a @always_inline function (and there's only one
  // possible callee), see if we can fold its entire body into an PValue.
  // This can fail for a number of reasons, in which case we fall back to
  // emitting normally.
  if (!calleeSig.isThrows()) {
    // We don't handle peeling off the variant and rethrowing for throws
    // functions yet.
    if (FailureOr<TypedAttr> resultPR =
            inlineFunctionCallIntoPValue(callee, argumentValues, evaluator);
        succeeded(resultPR))
      return emitCResult(*resultPR, callExpr, dest);
  }

  if (!builder) {
    // TODO: We can support throwing parameter calls by inserting a 'force to
    // normal value' check which aborts (at compile time) if interpretation
    // throws an error.
    if (calleeSig.isThrows()) {
      emitErrorForDynamicValueInParameter(
          callExpr, "TODO: cannot call potentially raising function");
      return {};
    }
    if (calleeSig.isAsync()) {
      emitErrorForDynamicValueInParameter(callExpr,
                                          "cannot call async function");
      return {};
    }

    // Emitting a call in a parameter context. Generate an apply operator.
    SmallVector<TypedAttr> operands({callee.getIfPValue().get()});
    for (auto [argValAndExpr, calleeArgType] :
         llvm::zip(argumentValues, calleeSig.getValueInputs())) {
      if (!argValAndExpr.ir.getIfPValue()) {
        emitError(argValAndExpr.expr->getLoc(),
                  "cannot use a dynamic value in parameter context")
            << argValAndExpr.expr->getRange();
        return {};
      }
      TypedAttr arg = argValAndExpr.ir.getIfPValue().get();
      // Emit a rebind if the refined type does not match the callee arg type.
      if (arg.getType() != calleeArgType)
        arg = ParamOperatorAttr::get(POC::Rebind, arg, calleeArgType);
      operands.push_back(arg);
    }

    TypedAttr result = ParamOperatorAttr::get(POC::Apply, operands);
    return emitCResult(result, callExpr, dest);
  }

  // Otherwise, materialize PValue and DLValue's as SSA values for emission.
  SmallVector<Value> callArgs;

  for (auto [argValAndExpr, conventionX, calleeArgTypeAndIdx] :
       llvm::zip(argumentValues, calleeSig.getValueInputConventions(),
                 llvm::enumerate(calleeSig.getValueInputs()))) {
    auto calleeArgType = calleeArgTypeAndIdx.value();
    auto argIdx = calleeArgTypeAndIdx.index();
    ValueInputConvention convention = conventionX;

    // If this is a variadic operation, the N operands have already been emitted
    // together and consolidated into a pop.variadic.create/pop.variadic.attr,
    // which is emitted as an SRValue instead of whatever the underlying type
    // is.
    if (calleeSig.isVararg(argIdx) || isa<POP::PackType>(calleeArgType))
      convention = ValueInputConvention::OwnedInReg;

    Value arg = emitPreemittedArgumentAsDynamicValue(argValAndExpr, convention);
    if (!arg)
      return {};
    if (arg.getType() != calleeArgType)
      arg = builder->create<RebindOp>(loc, calleeArgType, arg);
    callArgs.push_back(arg);
  }

  ArrayRef<Type> resultTypes = calleeSig.getValueResults();
  Operation *callOp;
  if (auto target = callee.getIfPValue()) {
    if (auto sig = dyn_cast<SignatureType>(target.getType().mlirType);
        sig && sig.isAsync()) {
      // If the callee is an async function, emit an async call.
      callOp = builder->create<AsyncCallOp>(loc, target.get(), resultParams,
                                            callArgs);
    } else if (auto symbol = dyn_cast<SymbolConstantAttr>(target.get())) {
      // If the callee is a symbol constant, directly emit a call.
      callOp = builder->create<CallOp>(loc, resultTypes, symbol, resultParams,
                                       callArgs);
    } else {
      callOp = builder->create<CallParamOp>(loc, resultTypes, target.get(),
                                            resultParams, callArgs);
    }
  } else {
    callOp = builder->create<CallSignatureOp>(loc, resultTypes,
                                              callee.getIfSRValue(), callArgs);
  }
  Value callResult = callOp->getResult(0);

  // If there were any writebacks to handle, emit them before handling raised
  // errors.
  afterCallActions.emit(*this);

  // If the callee can raise an error, it will be represented as a variant: try
  // to unwrap it.
  if (calleeSig.isThrows()) {
    // Put the insertion point back after we're done building the 'if'.
    OpBuilder::InsertionGuard builderGuard(*builder);

    auto callResultTy = cast<POP::VariantType>(callResult.getType());
    auto normalType = callResultTy.getType(1);
    auto ifOp = builder->create<HLCF::IfOp>(
        loc, normalType,
        builder->create<POP::VariantIsOp>(loc, callResult, normalType));

    // If this a normal value, yield it.
    builder->createBlock(&ifOp.getThenRegion());
    Value value =
        builder->create<POP::VariantGetOp>(loc, normalType, callResult);
    builder->create<HLCF::YieldOp>(loc, value);

    // Otherwise, this is an error, extract the error and throw it.
    builder->createBlock(&ifOp.getElseRegion());
    Value err = builder->create<POP::VariantGetOp>(loc, callResultTy.getType(0),
                                                   callResult);

    if (failed(emitRaise(err, loc))) {
      emitError(callExpr->getLoc(),
                "cannot call function that may raise in a context that "
                "cannot raise");
      return {};
    }
    builder->create<UnreachableOp>(loc);

    // Ok, the call result is the result of the HLCF::If.
    callResult = ifOp.getResult(0);
  }

  // If there is a memory result slot, the value we filled in is our MRValue
  // result and we've already handled the ValueDest by emitting into it.
  if (calleeSig.hasMemoryOnlyResult()) {
    // Re-emit the value in case a conversion was required or if the result was
    // a dynamic-lvalue.  In both case we will have emitted into a temporary
    // slot and 'dest' will have the ultimate location to write to.
    return emitCResult(MRValue(callArgs[0]), callExpr, dest);
  }

  // Otherwise, register-passable results are the call result which may need to
  // be emitted into a ValueDest.
  return emitCResult(SRValue(callResult), callExpr, dest);
}
