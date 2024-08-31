//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARAMETERINFERENCE_H
#define KGEN_MOJOPARSER_PARAMETERINFERENCE_H

#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"

namespace M::KGEN::LIT {
class ExprNode;
class LITSignatureType;

//===----------------------------------------------------------------------===//
// InferenceFailure
//===----------------------------------------------------------------------===//

/// These are the different failure modes that we know happen.
struct InferenceFailure {
  /// This failure happens when a parameter is found of the wrong type.
  struct TypeConflictFailure {
    ASTType paramType, argParamType;
  };

  /// This failure happens when a parameter is inferred to two different values.
  struct ValueConflictFailure {
    TypedAttr v1, v2;
  };

  /// This failure happens when the parameter isn't found at all.
  struct NotFoundFailure {};

  template <typename Failure>
  InferenceFailure(Failure info) : info(info) {}

  // Describe what went wrong.
  void emitSpecificNote(function_ref<InflightDiag &()> attachNote) const;

private:
  SmartVariant<TypeConflictFailure, ValueConflictFailure, NotFoundFailure> info;
};

//===----------------------------------------------------------------------===//
// ParameterInferenceDiagnostics
//===----------------------------------------------------------------------===//

class ParameterInferenceDiagnostics {
public:
  /// Indicate that parameter inference failed to infer the parameter at
  /// `paramIdx` from the argument at `argPos`.
  void addFailure(size_t paramIdx, const ExprNode *argExpr,
                  InferenceFailure &&info) {
    diags.push_back({paramIdx, argExpr, std::move(info)});
  }

  /// Attach failed parameter inference diagnostics for parameters with no
  /// values to the overload resolution diagnostic.
  void attach(PogListAttr params, InflightDiag &diag, size_t numActual = 0);

  struct FailedInference {
    size_t paramIdx;
    const ExprNode *argExpr;
    InferenceFailure info;
  };
  using DiagStorage = SmallVector<FailedInference, 1>;

  DiagStorage saveDiags() { return diags; }
  void resetDiags(DiagStorage &&newDiags) { diags = std::move(newDiags); }

private:
  DiagStorage diags;
};

//===----------------------------------------------------------------------===//
// ParameterInferenceState
//===----------------------------------------------------------------------===//

/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParameterInferenceState : public TypeCheckScopeInfo {
public:
  ParameterInferenceState(const TypeCheckScopeInfo &scopeInfo,
                          const CallOperands &givenBindings,
                          ArrayRef<TypedAttr> bindingsSoFar,
                          const ParserParamEvaluator &evaluator,
                          ParameterInferenceDiagnostics &diags,
                          bool allowImplicitConversions)
      : TypeCheckScopeInfo(scopeInfo), givenBindings(givenBindings),
        evaluator(evaluator),
        inferredParams(bindingsSoFar.begin(), bindingsSoFar.end()),
        diags(diags), allowImplicitConversions(allowImplicitConversions) {}

  /// Given an incomplete parameter binding set for a parameter list, try to
  /// infer the value of the next parameter. We only do this if there are any
  /// inferred parameters present.
  void infer(ArrayRef<Type> paramTypes, PogListAttr paramListAttr);

  /// Given an incomplete parameter binding set and the arguments for a call to
  /// the specified signature, try to infer the value of the next 'decl'
  /// parameter. This should always return failure /without/ an error if it
  /// cannot be inferred, and return success if a value was determined.
  LogicalResult infer(LITSignatureType signature,
                      const CallOperands &callOperands,
                      const OperandValueList &variadicKwOperands);

  /// Given an incomplete parameter binding set, try to infer parameters on Self
  /// of a method from the first argument.
  LogicalResult inferCTADParams(LITSignatureType signature,
                                const CallOperands &callOperands);

  /// After inferring parameter values, this allows access to the results.
  TypedAttr getInferredValue(size_t idx) const {
    return idx < inferredParams.size() ? inferredParams[idx] : TypedAttr();
  }

private:
  LogicalResult matchTypes(Type actualType, Type expectedType);
  LogicalResult matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);
  LogicalResult matchSingleEltStruct(TypedAttr actualAddrSpace,
                                     TypedAttr expectedAddrSpace);
  LogicalResult inferInitSelfTypes(Type actualType, Type expectedType);

  /// Infer parameters from an operand being passed into this function. This is
  /// only called on the top level function operands being matched up, not
  /// anything in recursive functiontype positions.
  LogicalResult inferOneOperand(ASTExprAnd<AnyValue> operand,
                                ASTType expectedType,
                                ArgConvention expectedConvention);
  void addFailure(size_t parameterIndex, InferenceFailure &&info) {
    diags.addFailure(parameterIndex, curArgExpr, std::move(info));
  }

  /// Infer parameters from a single parameter binding.
  void inferOneParam(ASTExprAnd<AnyValue> binding, Type expectedType);

  /// These are the bindings originally provided to the callable. These are used
  /// to infer parameters from other parameter values.
  const CallOperands &givenBindings;

  /// This is the evaluator instance parameter inference uses to progressively
  /// refine dependent types as we infer parameters.
  ParserParamEvaluator evaluator;

  /// One entry for each parameter from the original binding list.  If
  /// non-null, we've already inferred a value for that parameter.
  SmallVector<TypedAttr> inferredParams;

  /// The signature type of parameter infernece. This is how many signature
  /// types deep inference is inside parameter expressions and determines which
  /// index references we match against.
  size_t paramIndexRefDepth = 0;

  /// The current set of parameter inference diagnostics.
  ParameterInferenceDiagnostics &diags;

  /// True if implicit conversions in argument lists are permitted.
  bool allowImplicitConversions;

  /// The expression of the current argument being used for parameter inference.
  const ExprNode *curArgExpr = nullptr;

  /// These are parameters that were inferred from a more specific Self type
  /// argument in an initializer call. These parameters can forward reference
  /// non-Self parameters. We need to refine them again at the end of inference.
  SmallVector<unsigned> initSelfParams;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMETERINFERENCE_H
