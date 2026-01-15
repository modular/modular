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

//===----------------------------------------------------------------------===//
// ParamInf
//===----------------------------------------------------------------------===//

/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParamInf {
public:
  /// These are the bindings originally provided to the callable.
  const ParamBindings &paramBindings;

  /// If we're inferring the parameters for a declaration like a fn or struct,
  /// maintain a pointer to it so we can emit better diagnostics.  This will be
  /// null when binding a parametric value, like a parametric alias.
  ASTDecl *const declIfKnown;

  /// This is the callback to report diagnostics through.
  llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag;

  ParamInf(
      const ParamBindings &paramBinding, ArrayRef<Type> declaredParamTypes,
      PogListAttr declaredParamPogs, bool allowImplicitConversions,
      bool partial,
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
  LogicalResult inferFromParamList();

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

  // Infer any missing parameter from defaulted value (this is supposed to be
  // invoked after both parameter list and argument list has been scanned).
  LogicalResult inferFromDefaults();

  // Finalize the inference by making any remaining uninferred parameter to
  // UnboundAttr.
  void finalizeWithUnbound();

  /// After inferring parameter values, this allows access to the results.
  TypedAttr getInferredValue(size_t idx) const {
    return evaluator.getIndexBindings()[idx];
  }

  /// Convenience getters for fields from paramBindings.
  ASTDecl &getDeclScope() const { return paramBindings.declScope; }
  SharedState &getShared() const { return paramBindings.shared; }
  const CallOperands &getGivenBindings() const {
    return paramBindings.getParameters();
  }
  size_t getNumPreCheckedParam() const {
    return paramBindings.getNumPreCheckedParams();
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

  /// This describes the number of type of all of the parameters we're trying to
  /// resolve for this entire declaration.
  ArrayRef<Type> declaredParamTypes;

  /// This describes the nature of the parameter list we're inferring for.
  PogListAttr declaredParamPogs;

  /// True if implicit conversions in argument lists are permitted.
  const bool allowImplicitConversions;
  /// True if the inference can lead to unbound attribute.
  const bool partial;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARAMETERINFERENCE_H
