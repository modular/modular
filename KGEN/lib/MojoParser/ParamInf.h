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
class ParamInf;

//===----------------------------------------------------------------------===//
// InferenceFailure
//===----------------------------------------------------------------------===//

/// These are the different failure modes that we know happen.
struct InferenceFailure {
  /// This failure happens when a parameter is found of the wrong type.
  struct TypeConflict {
    size_t paramIdx; // TODO: Render this name.
    ASTType paramType, argParamType;
  };

  /// This failure happens when a parameter is inferred to two different values.
  struct ValueConflict {
    size_t paramIdx;
    TypedAttr v1, v2;
  };

  /// This failure happens when merge* is called, but the expected type/value
  /// still has an unresolved dependent type which can't be inferred.
  struct DependsOnUnresolved {
    size_t paramIdx;
  };

  /// This failure happens when parameter is inferred, yet the constraint
  /// attached on it can not be proved.
  struct UnprovableConstraints {
    size_t paramIdx;
  };

  /// This failure hasn't been categorized yet.
  /// FIXME: Remove this.
  struct Unclassified {};

  template <typename Failure>
  InferenceFailure(Failure info) : info(info) {}

  // Describe what went wrong.
  void addExplanation(MojoInflightDiag &diag) const;

  /// If this failure is due to an unresolved parameter, return the index of the
  /// parameter.
  std::optional<size_t> getIfDependentOnUnresolved() const {
    if (isa<DependsOnUnresolved>(info)) {
      return cast<DependsOnUnresolved>(info).paramIdx;
    }
    return std::nullopt;
  }

private:
  SmartVariant<TypeConflict, ValueConflict, DependsOnUnresolved, Unclassified,
               UnprovableConstraints>
      info;
};

//===----------------------------------------------------------------------===//
// ParamInf
//===----------------------------------------------------------------------===//

/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParamInf {
public:
  /// This is the declaration that we do name lookup against.
  ASTDecl &declScope;
  SharedState &shared;

  /// If we're inferring the parameters for a declaration like a fn or struct,
  /// maintain a pointer to it so we can emit better diagnostics.  This will be
  /// null when binding a parametric value, like a parametric alias.
  ASTDecl *const declIfKnown;

  /// This is the callback to report diagnostics through.
  llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag;

  ParamInf(
      ASTDecl &declScope, const CallOperands &givenBindings,
      size_t numPreCheckedBindings, ArrayRef<Type> declaredParamTypes,
      PogListAttr declaredParamPogs, bool allowImplicitConversions,
      llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag,
      ASTDecl *declIfDirect);

  /// Infer all of the parameters we can from 'givenBindings'.
  ///
  /// The 'partial' field specifies this is
  /// performing a partial binding - e.g. because this is not a full type
  /// binding, or because more params can be inferred from arguments to the
  /// call.
  ///
  /// On failure, this will emit a diagnostic through the 'getDiag' callback.
  LogicalResult inferFromParamList(bool partial);

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

  /// FIXME: This is a temporary flag that will soon go away. This is used to
  /// distinguish parameter inference for overload resolution or struct
  /// parameter binding. We are migrating one part at a time.
  std::optional<LogicalResult> inferredForCallRet = std::nullopt;

  /// Given an incomplete parameter binding set, try to infer parameters on Self
  /// of a method from the first argument.
  LogicalResult inferCTADParams(FnTypeGeneratorType signature,
                                const CallOperands &callOperands);

  /// After inferring parameter values, this allows access to the results.
  TypedAttr getInferredValue(size_t idx) const {
    return evaluator.getIndexBindings()[idx];
  }

  void dump() const;

  /// This is the evaluator instance parameter inference uses to progressively
  /// refine dependent types as we infer parameters.
  ParserParameterEvaluator evaluator;

  // A simple wrapper around `overwriteIndexBinding` to ensure sugar is aligned
  // before overwriting parameter value.
  // Notable, this method does not check there is no existing parameter inferred
  // and unconditional overwrite everything.
  //
  // Return failure when the constraint attached to the parameter can not be
  // satisfied, it populates unprovableConstraints too.
  SmallVector<ConstraintAttr> unprovableConstraints;
  LogicalResult setInferredValue(size_t paramIdx, TypedAttr paramVal);

  /// Cached finder to identify types that contains unbound ParamIndexRefAttrs.
  ParamIndexRefAttrFinder paramFinder;

private:
  // This says that a parameter with an unresolved dependent type was seen
  // during initial parameter binding application, so resolution of it was
  // deferred.
  //
  // This is to handle cases like:
  //
  // fn foo[rank : Int, coord : IndexList[rank, Int]](i : SomeThing[rank]):
  //    pass
  //
  // fn foo_user():
  //    var i = SomeThing[2]
  //    foo[coord = Tuple(1, 2)](i)
  //
  // When binding `coord` with `coord: Tuple[Int, Int]`, we do not know that
  // `rank=2` nor `Tuple[Int, Int]` is convertible to `IndexList[2, Int]`, we
  // have to postpone the binding till `rank` is resolved.
  //
  // Do we need to support this? I don't think this is too crazy to require user
  // to type `foo[rank = 2, coord = (1, 2)]` here?
  bool hasDeferredGivenParam = false;

  LogicalResult inferSelfFromInitResult(FnTypeGeneratorType signature);

  /// Infer parameters from an operand being passed into this function. This is
  /// only called on the top level function operands being matched up, not
  /// anything in recursive functiontype positions.
  LogicalResult inferOneOperand(ASTExprAnd<AnyValue> operand, size_t argIdx,
                                ASTType expectedType,
                                ArgConvention expectedConvention,
                                PogListAttr argPogs, CallSyntax syntax);

  /// Infer and emit a single value for a parameter binding. This returns
  /// failure if it emits a diagnostic, otherwise is returns a parameter value
  /// if resolved, or null if deferred.
  FailureOr<TypedAttr> inferAndEmitOneParam(ASTExprAnd<AnyValue> binding,
                                            ASTType expectedType,
                                            size_t paramIdx);

  /// These are the bindings originally provided to the callable. These are used
  /// to infer parameters from other parameter values.
  const CallOperands &givenBindings;

  /// This describes the number of type of all of the parameters we're trying to
  /// resolve for this entire declaration.
  ArrayRef<Type> declaredParamTypes;

  /// This describes the nature of the parameter list we're inferring for.
  PogListAttr declaredParamPogs;

  /// True if implicit conversions in argument lists are permitted.
  const bool allowImplicitConversions;

  /// The number "givenBindings" that are pre-checked and just need to be
  /// installed, instead of treated as things specified in the [] list.
  const size_t numPreCheckedParam;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMETERINFERENCE_H
