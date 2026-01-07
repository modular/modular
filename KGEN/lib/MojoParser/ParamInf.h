//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARAMETERINFERENCE_H
#define KGEN_MOJOPARSER_PARAMETERINFERENCE_H

#include "CallEmission.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "ParserEvaluationContext.h"

namespace M::KGEN::LIT {
class ExprNode;

//===----------------------------------------------------------------------===//
// InferenceFailure
//===----------------------------------------------------------------------===//

/// These are the different failure modes that we know happen.
struct InferenceFailure {
  /// This failure happens when a parameter is found of the wrong type.
  struct TypeConflictFailure {
    size_t paramIdx; // TODO: Render this name.
    ASTType paramType, argParamType;
  };

  /// This failure happens when a parameter is inferred to two different values.
  struct ValueConflictFailure {
    size_t paramIdx;
    TypedAttr v1, v2;
  };

  /// This failure happens when the parameter isn't found at all.
  struct NotFoundFailure {
    size_t paramIdx;
  };

  template <typename Failure>
  InferenceFailure(Failure info) : info(info) {}

  // Describe what went wrong.
  void addExplanation(MojoInflightDiag &diag) const;

private:
  SmartVariant<TypeConflictFailure, ValueConflictFailure, NotFoundFailure> info;
};

//===----------------------------------------------------------------------===//
// ParamInfDiags
//===----------------------------------------------------------------------===//

class ParamInfDiags {
public:
  /// Indicate that parameter inference failed to infer the parameter at
  /// `paramIdx` from the argument at `argPos`.
  void addFailure(InferenceFailure &&info) {
    // only report the first error;
    if (hasFailure())
      return;
    diags = std::move(info);
  }

  void addExplanation(MojoInflightDiag &diag) {
    if (hasFailure())
      diags->addExplanation(diag);
  }

  bool hasFailure() const { return diags.has_value(); }

  using DiagStorage = std::optional<InferenceFailure>;

  DiagStorage saveDiags() { return diags; }
  void resetDiags(DiagStorage &&newDiags) { diags = std::move(newDiags); }

private:
  DiagStorage diags;
};

//===----------------------------------------------------------------------===//
// ParamInfState
//===----------------------------------------------------------------------===//

/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParamInfState {
public:
  /// This is the declaration that we do name lookup against.
  ASTDecl &declScope;
  SharedState &shared;

  ParamInfState(ASTDecl &declScope, const CallOperands &givenBindings,
                size_t numPreCheckedBindings, ArrayRef<Type> declaredParamTypes,
                PogListAttr declaredParamPogs, ParamInfDiags &diags,
                bool allowImplicitConversions);

  /// Given an incomplete parameter binding set for a parameter list, try to
  /// infer the value of the next parameter. We only do this if there are any
  /// inferred parameters present.  The 'hasArguments' field specifies whether
  /// there are arguments that can be used to infer parameters from (which are
  /// not handled by this call). When `installParam` is set, the parameter will
  /// be installed into evaluator.
  ///
  /// TODO: remove `installParam` and make it always true.
  void inferFromParamList(bool hasArguments);

  /// Given an incomplete parameter binding set and the arguments for a call to
  /// the specified signature, try to infer the value of the next 'decl'
  /// parameter. This should always return failure /without/ an error if it
  /// cannot be inferred, and return success if a value was determined.
  ///
  /// returnsSelf is True if this is performing inference on a function like
  /// __init__ that returns Self, which might be specialized.
  LogicalResult inferForCall(FnTypeGeneratorType signature,
                             const CallOperands &callOperands,
                             const OperandValueList &variadicKwOperands,
                             bool returnsSelf, bool hasCTADParams);

  /// Given an incomplete parameter binding set, try to infer parameters on Self
  /// of a method from the first argument.
  LogicalResult inferCTADParams(FnTypeGeneratorType signature,
                                const CallOperands &callOperands);

  /// After inferring parameter values, this allows access to the results.
  TypedAttr getInferredValue(size_t idx) const {
    return evaluator.getIndexBindings()[idx];
  }

  void dump() const;

  void addFailure(InferenceFailure &&info) {
    diags.addFailure(std::move(info));
  }

  /// This is the evaluator instance parameter inference uses to progressively
  /// refine dependent types as we infer parameters.
  ParserParameterEvaluator evaluator;

private:
  LogicalResult inferSelfFromInitResult(FnTypeGeneratorType signature);

  /// Infer parameters from an operand being passed into this function. This is
  /// only called on the top level function operands being matched up, not
  /// anything in recursive functiontype positions.
  LogicalResult inferOneOperand(ASTExprAnd<AnyValue> operand,
                                ASTType expectedType,
                                ArgConvention expectedConvention);
  /// Infer parameters from a single parameter binding.
  void inferOneParam(ASTExprAnd<AnyValue> binding, Type expectedType);

  /// These are the bindings originally provided to the callable. These are used
  /// to infer parameters from other parameter values.
  const CallOperands &givenBindings;

  /// This describes the number of type of all of the parameters we're trying to
  /// resolve for this entire declaration.
  ArrayRef<Type> declaredParamTypes;

  /// This describes the nature of the parameter list we're inferring for.
  PogListAttr declaredParamPogs;

  /// The current set of parameter inference diagnostics.
  ParamInfDiags &diags;

  /// True if implicit conversions in argument lists are permitted.
  const bool allowImplicitConversions;

  /// The expression of the current argument being used for parameter inference.
  const ExprNode *curArgExpr = nullptr;

  /// Cached finder to identify types that contains unbound ParamIndexRefAttrs.
  ParamIndexRefAttrFinder paramFinder;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMETERINFERENCE_H
