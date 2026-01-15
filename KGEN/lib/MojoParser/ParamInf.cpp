//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParamInf.h"
#include "CallEmission.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "ParamBindings.h"
#include "ParamMatcher.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/Constraints.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/SharedState.h"

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"

#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

#define DEBUG_TYPE "PARAMINF"

//===----------------------------------------------------------------------===//
// ParameterInference
//===----------------------------------------------------------------------===//
ParamInf::ParamInf(
    const ParamBindings &paramBinding, ArrayRef<Type> declaredParamTypes,
    PogListAttr declaredParamPogs, bool allowImplicitConversions, bool partial,
    llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag,
    ASTDecl *declIfDirect)
    : paramBindings(paramBinding), declIfKnown(declIfDirect),
      getDiag(std::move(getDiag)), evaluator(paramBinding.shared),
      declaredParamTypes(declaredParamTypes),
      declaredParamPogs(declaredParamPogs),
      allowImplicitConversions(allowImplicitConversions), partial(partial) {
  size_t finalSize = declaredParamTypes.size();

  // Pre-install any "prechecked" bindings.  These come from self arguments like
  // `x: T[1, 2]; x.foo()`: we'll have 1,2 as prechecked bindings due to 'x' as
  // the self argument of the call.  This is pretty gross, but we need to do
  // something like this because we have variadics, specified values for
  // infer-only parameters etc.  We are also dealing with two concatenated
  // parameter lists: the Self parameters have keywords before the method
  // parameters etc.
  if (getNumPreCheckedParam()) {
    ArrayRef preChecked =
        ArrayRef(getGivenBindings().values).take_front(getNumPreCheckedParam());
    for (auto &preCheckedOperand : preChecked) {
      auto preCheckParamVal = preCheckedOperand.ir.getIfPValue().get();
      if (sugarIsa<UnboundAttr>(preCheckParamVal))
        evaluator.appendIndexBinding(TypedAttr());
      else
        evaluator.appendIndexBinding(preCheckParamVal);
    }
  }
  // Fills in with nullptr.
  while (evaluator.getNumIndexBindings() < finalSize)
    evaluator.appendIndexBinding(TypedAttr());
}

void ParamInf::dump() const {
  auto &os = llvm::errs() << "ParamInf:\n";
  for (auto [idx, value] : llvm::enumerate(evaluator.getIndexBindings())) {
    os << "  *(0," << idx << ") = ";
    if (value)
      os << value;
    else
      os << "<not yet set> : "
         << const_cast<ParamInf *>(this)->evaluator.getReboundType(
                declaredParamTypes[idx]);
    os << "\n";
  }
}

/// Try to infer parameters of Self from an initializer if specialized.
///
/// Consider:
///    struct S[a: Int]:
///      fn __init__(out self): ...
///      fn __init__(out self: S[1], x: Int): ...
///
/// When constructed with no arguments, the first constructor must be used and
/// it is impossible to infer the value of 'a', so you must use `S[1]()`.  This
/// is the usual case.
///
/// However the second initializer is more specialized due to its custom Self -
/// it only applies when 'a' is 1, so we can infer that would be the value to
/// use if it is selected because one arg is passed to the initializer `S(42)`.
///
/// This function helps to infer the 'a' parameter when more specialized.  This
/// custom logic is required because often (eg in this case) the "actual" type
/// will have UnboundAttr parameters, instead of fully bound ones like a normal
/// argument.
LogicalResult ParamInf::inferSelfFromInitResult(FnTypeGeneratorType signature) {
  DeclResolver::DeclScopeChanger x(declIfKnown);

  // When a parameter gets bound, we re-evaluate the result type to see the
  // fully concretized parameters that the parameter may be computing.
RetryLabel:
  ASTType returnedType =
      evaluator.getReboundType(signature.getUserResultType());

  auto reportConflict = [&](size_t paramIdx, TypedAttr actual,
                            TypedAttr expected) -> LogicalResult {
    getDiag(getGivenBindings().callExpr->getLoc())
        << "return type " << returnedType << " parameter "
        << ParamIndexRefAttr::get(/*depth*/ 0, paramIdx, actual.getType())
        << " value " << actual << " doesn't match expected value " << expected;
    return failure();
  };

  // Match up the parameter bindings if the 'actual' param is an UnboundAttr and
  // the expected has something more specific than a reference to the contextual
  // parameter.
  for (auto [idx, retParam] :
       llvm::enumerate(returnedType.getParamBindings())) {
    // If this is simply a reference to the enclosing parameter (as in a normal
    // Self) init, then we can't infer anything from it.  In the example above,
    // this ignores the "a" parameter in "fn __init__() -> S[a]:" which is what
    // "out self" desugars to.
    auto selfParam = evaluator.getIndexBindings()[idx];
    if (retParam == selfParam)
      continue;

    // Otherwise, if the self parameter got inferred, propagate the result
    // from it to the returned parameter.  This handles things like:
    //   struct X[A: AnyType]:
    //     fn __init__[T: Movable](arg: Int, out self: X[T]):
    // which gets used as X[String](42) inferring T and A.
    ParamMatcher matcher(getGivenBindings().callExpr, *this);
    if (selfParam) {
      // TODO: Macro'ize this when error handling logic is fixed.
      switch (matcher.matchParams(selfParam, retParam)) {
      case ParamMatcher::Retry:
        goto RetryLabel;
      case ParamMatcher::Match:
        break;
      case ParamMatcher::Error:
        return reportConflict(idx, retParam, selfParam);
      }
    } else if (!paramFinder.hasReferences(retParam)) {
      // Otherwise if the the returned parameter has no unbound parameter
      // references then we infer the self parameter from it. This infers X=42:
      //   struct X[A: Int]:
      //     fn __init__(out self: X[42]):
      auto selfType =
          evaluator.getReboundType(signature.getInputParamTypes()[idx]);
      auto selfParam = ParamIndexRefAttr::get(/*depth*/ 0, idx, selfType);
      switch (matcher.matchParams(retParam, selfParam)) {
      case ParamMatcher::Retry:
        goto RetryLabel;
      case ParamMatcher::Match:
        break;
      case ParamMatcher::Error:
        return reportConflict(idx, selfParam, retParam);
      }
    }
  }

  return success();
}

static Type inferInitializerType(ASTDecl &declScope, InitializerUValue *init,
                                 ASTExprAnd<AnyValue> operand,
                                 ASTType defaultType) {
  IREmitter emitter(declScope, ExprContext::EC_CallArgValue);
  if (!defaultType)
    return {};
  ASTType inferredType =
      defaultType.getWithUnknownParametersReplaced(declScope.getShared());
  CallOperands operands =
      init->getOperandsForInferredType(inferredType, emitter);

  // We expect the initializer to return the constructed type.
  // Infer the parameters of this overload candidate against the computed
  // result type of the initializer.
  FailureOr<PValue> initFn = OverloadSet::canConstructType(
      inferredType, std::move(operands), declScope);
  if (failed(initFn) || !initFn.value())
    return {};
  return sugarCast<FnTypeGeneratorType>(initFn.value().getType())
      .getUserResultType();
}

// TODO: Reconsolidate this.
namespace M::KGEN::LIT {
void printUValueTypeInfo(const AnyValue &value, MojoInflightDiag &diag);
void emitWrongTypeDiag(MojoInflightDiag &diag, ASTExprAnd<AnyValue> operand,
                       ASTType expectedType, size_t argIdx,
                       PogListAttr argListAttr, CallSyntax syntax,
                       SharedState &shared);
} // namespace M::KGEN::LIT

/// Check the expected type against the provided operand. This identifies any
/// problems with the operand type, which it handled by emitting a diagnostic
/// and returning failure.
///
/// This can be called on a function signature with incomplete bindings, which
/// means that 'origExpectedType' may have unbound parameters.  As such, this
/// will infer parameters from the operand and return the inferred type.
///
/// TODO: This is a more general mirror of 'OverloadFitness::checkOneOperand':
/// unify it into this.
LogicalResult ParamInf::inferOneOperand(ASTExprAnd<AnyValue> operand,
                                        size_t argIdx, ASTType origExpectedType,
                                        ArgConvention expectedConvention,
                                        PogListAttr argPogs,
                                        CallSyntax syntax) {
  // Make sure the diagnostic machinery knows about our getDeclScope() so
  // parameter names get emitted correctly.
  DeclResolver::DeclScopeChanger x(declIfKnown);

  auto emitWrongTypeDiag = [&](ASTType expectedType) -> MojoInflightDiag & {
    auto &diag = getDiag(operand.expr->getLoc());
    ::emitWrongTypeDiag(diag, operand, expectedType, argIdx, argPogs, syntax,
                        getShared());
    return diag;
  };

  // Whenever a parameter is bound, we need to re-evaluate the expected type and
  // try again.
RetryLabel:
  ASTType expectedType = evaluator.getReboundType(origExpectedType);

  // TODO: Calculate OverloadFitness's fitness (# implicit conversions etc).
  ParamMatcher matcher(operand.expr, *this);

  // We'll bind the next provided value.
  switch (expectedConvention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError: {
    // The actual value must be an lvalue if callee takes things by-ref.
    auto argVal = operand.ir.getIfLValue();
    if (!argVal) {
      auto &diag = getDiag(operand.expr->getLoc());
      if ((syntax == CallSyntax::kMethodCall ||
           syntax == CallSyntax::kMethodCallSynthetic) &&
          argIdx == 0) {
        diag << "invalid use of mutating method on rvalue of type ";
        if (ASTType type = operand.ir.getRValueTypeIfResolvable())
          diag << type;
        else
          printUValueTypeInfo(operand.ir, diag);
      } else {
        diag << "value passed to mutable argument " << argPogs.getName(argIdx)
             << " must be mutable";
      }
      diag << operand.expr->getRange();
      return failure();
    }

    // If this is a wildcard type, we can match any operand.
    if (sugarIsa<NameLookupArgWildcardType>(argVal.getRValueType()))
      return success();

    // Ok we have an LValue.  The reference element types must match.
    switch (matcher.matchTypes(argVal.getRValueType(),
                               expectedType.getReferenceElementType())) {
    case ParamMatcher::Retry:
      goto RetryLabel;
    case ParamMatcher::Match:
      break;
    case ParamMatcher::Error:
      // ByRef argument types must exactly match, no conversions are allowed.
      auto &diag = getDiag(operand.expr->getLoc());
      diag << "l-value of type " << operand.ir.getIfLValue().getRValueType()
           << " cannot be converted to reference of type "
           << expectedType.getReferenceElementType()
           << operand.expr->getRange();
      matcher.failureReason->addExplanation(diag);
      return failure();
    }
    return success();
  }
  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    auto expectedRef = sugarCast<RefType>(expectedType);

    // If we are binding the reference to a value in memory directly, check for
    // reference compatibility directly.
    if (operand.ir.isMValue()) {
      RefType valueRefType = operand.ir.getMValueType();
      // If the IRValue type is MBValue or MRValue then we need infer an
      // immutable ref, to match behavior where we don't allow passing an
      // MBValue or MRValue as 'mut'.
      if (!operand.ir.getIfMLValue() && !operand.ir.getIfMBPValue() &&
          !valueRefType.isMutableKnown(false))
        valueRefType = valueRefType.getWithMutability(false);

      // If the origin is already specified, allow implicit conversions,
      // allowing you to pass a concrete origin to something expecting a union
      // or AnyOrigin.  This check happens here (instead of in matchTypes)
      // because function arguments can be rebound when origins disagree, but
      // this isn't correct/possible in arbitrary nested positions.
      if (!paramFinder.hasReferences(expectedType)) {
        if (IREmitter::canZeroCostConvert(valueRefType, expectedType,
                                          getShared()))
          return success();
        emitWrongTypeDiag(expectedType);
        return failure();
      }

      // Otherwise, match the origins up to infer from the value.
      switch (matcher.matchTypes(valueRefType, expectedType)) {
      case ParamMatcher::Retry:
        goto RetryLabel;
      case ParamMatcher::Match:
        return success();
      case ParamMatcher::Error:
        emitWrongTypeDiag(expectedType);
        return failure();
      }
    }
    // Otherwise, we are binding something like a PValue or SRValue to a
    // reference argument, which doesn't have a origin.  This is a problem
    // because origins can be propagated through the type system of the
    // function call to other arguments and they all need to line up.  We
    // handle this in two phases: during overload resolution we bind this to
    // an immortal origin, and then after the candidate is selected, we
    // re-emit these arguments to memory and re-infer all the parameters.
    //
    // One detail is how we do this: we bind these arguments to immutable
    // temporaries, because we specifically do NOT want 'ref' arguments with
    // parametric mutability to treat these things as mutable.
    if (sugarCast<RefType>(expectedType).isMutableKnown(true)) {
      auto &diag = getDiag(operand.expr->getLoc());
      diag << "mutable reference argument " << argPogs.getName(argIdx)
           << "cannot bind to temporary value";
      return diag;
    }

    // Otherwise, we'll need to drop this value into a temporary.  For now, we
    // infer it as AnyOrigin.  We bind the origin directly and then handle
    // it like any other argument because we can support implicit conversions.
    auto anyOrigin =
        AnyOriginAttr::get(expectedRef.getContext(), /*isMut=*/false);
    switch (matcher.matchSingleEltStruct(anyOrigin, expectedRef.getOrigin())) {
    case ParamMatcher::Retry:
      goto RetryLabel;
    case ParamMatcher::Match:
      break;
    case ParamMatcher::Error:
      // Ignore failures because we only want to set a value if none is already
      // known so things aren't ambiguous.
      matcher.resetError();
      break;
    }

    // The address space of the temp will be the default.
    auto addrSpace =
        IntegerAttr::get(IndexType::get(expectedRef.getContext()), 0);
    switch (matcher.matchSingleEltStruct(addrSpace,
                                         expectedRef.getAddressSpace())) {
    case ParamMatcher::Retry:
      goto RetryLabel;
    case ParamMatcher::Match:
      break;
    case ParamMatcher::Error:
      matcher.resetError();
      break;
    }

    // Handle the element type compatibility check below to allow implicit
    // conversions etc.
    [[fallthrough]];
  }
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
  case ArgConvention::ReadMem:
    // Otherwise, we expect an r-value to match up, ignoring the reference type
    // from the convention.
    expectedType = expectedType.getReferenceElementType();
    break;
  case ArgConvention::ReadReg:
    break;
  }

  // Okay, we got a normal value argument convention and stripped off any
  // ArgConvention-related !lit.ref from the expected type.  See if we can
  // resolve the argument to a CValue.
  CValue argVal = operand.ir.getIfCValue();

  // Check to see if the expected type has an initializer with the
  // specified operands.  Remove any parameters from the expected type
  // since those are what we're inferring from the arguments.  The result
  // 'actualType' will have those newly inferred parameters.
  if (!argVal) {
    if (auto initValue = operand.ir.getIfInitializer()) {
      // If we have a type like List[$0] replace it with List[?] so we can
      // infer the unbound parameter.
      auto unbound = expectedType.getWithUnknownParametersReplaced(getShared());
      Type initType =
          inferInitializerType(getDeclScope(), &(*initValue), operand, unbound);
      // If the literal cannot bind to the inferred type, try binding it to the
      // default literal type and matching the inferred type against that.
      if (!initType)
        initType = inferInitializerType(getDeclScope(), &(*initValue), operand,
                                        initValue->getDefaultType(getShared()));

      // If there were declaration errors, assume success to not raise
      // spurious errors due to not resolving to those erroneous
      // declarations.
      if (!initType) { // TODO: Could improve this error to talk about inits.
        emitWrongTypeDiag(expectedType);
        return failure();
      }
      // If we found one, we resolve our value to the inferred type.
      switch (matcher.matchTypes(initType, expectedType)) {
      case ParamMatcher::Retry:
        goto RetryLabel;
      case ParamMatcher::Match:
        return success();
      case ParamMatcher::Error:
        // TODO: Could improve this to talk about initializers.
        auto &diag = emitWrongTypeDiag(expectedType);
        matcher.failureReason->addExplanation(diag);
        return failure();
      }
    }

    auto orValue = operand.ir.getIfOverloadSet();
    assert(orValue && "Unknown UValue!");

    // Try to refine the OverloadSetUValue into a PValue.
    argVal = orValue->getDirectSymbol(expectedType, getDeclScope());
    if (!argVal) { // TODO: Could improve this to talk about overload sets.
      emitWrongTypeDiag(expectedType);
      return failure();
    }

    // If we have a reference to an overloaded method like foo(a.method),
    // then we can't resolve it.
    // TODO(partial application => closures): Given we just resolved argVal,
    // we could form the "a.method" expression with a closure.
    if (orValue->baseValue) { // Cannot merge base value.
      emitWrongTypeDiag(expectedType);
      return failure(); // TODO: Improve this.
    }
    // Otherwise, success, fallthrough.
  }

  // If the argument types exactly match, then they are good.
  ASTType argType = argVal.getRValueType();
  if (argType.isEqualCanon(expectedType) ||
      // If this is a wildcard type, we can match any operand.
      sugarIsa<NameLookupArgWildcardType>(argType))
    return success();

  // We're speculatively trying different options.  If we have errors on one
  // path we need to roll them back.
  std::optional<MatchFailure> savedFailureReason;

  // If the expected type has unresolved bindings, try to infer them from the
  // argument first, before trying implicit conversions etc.
  if (paramFinder.hasReferences(expectedType)) {
    switch (matcher.matchTypes(argType, expectedType)) {
    case ParamMatcher::Retry:
      goto RetryLabel;
    case ParamMatcher::Match:
      return success(); // Types were equal after matching.
    case ParamMatcher::Error:
      savedFailureReason = matcher.failureReason;
      matcher.resetError();
      break;
    }
  } else {
    // Zero cost conversions don't count as implicit conversions. We attempt
    // this after trying to match the types to try to infer values first.
    if (IREmitter::canZeroCostConvert(argType, expectedType, getShared()))
      return success();
  }

  // Handle values of nonmaterializable types.  These freely convert to their
  // nonmaterializable target type: even when implicit conversions are disabled.
  // We can accept this argument if that converted type is compatible with
  // our expected type.
  if (syntax != CallSyntax::kParamBindings) {
    if (auto nonmaterializableTarget =
            argType.getNonmaterializableTarget(getShared())) {

      // Infer the parameters of this overload candidate against the computed
      // result type of the initializer.
      switch (matcher.matchTypes(nonmaterializableTarget, expectedType)) {
      case ParamMatcher::Retry:
        goto RetryLabel;
      case ParamMatcher::Match:
        return success();
      case ParamMatcher::Error:
        matcher.resetError();
        break;
      }
    }
  }

  // If implicit conversions are enabled and the target type is known, then
  // we can check to see if any of the constructors for the result type can
  // work.  If disabled, then we have a failure.
  if (!allowImplicitConversions) {
    auto &diag = emitWrongTypeDiag(expectedType);
    if (savedFailureReason)
      savedFailureReason->addExplanation(diag);
    return failure();
  }

  // If the expected type has been fully resolved, check it for implicit
  // conversions using the normal type machinery.  This will handle things like
  // function pointer conversions that the code below doesn't.
  if (!paramFinder.hasReferences(expectedType)) {
    if (IREmitter::canImplicitlyConvertToType({argVal, operand.expr},
                                              expectedType, getDeclScope()))
      return success();
    auto &diag = emitWrongTypeDiag(expectedType);
    if (savedFailureReason)
      savedFailureReason->addExplanation(diag);
    return failure();
  }

  /// When checking if an implicit conversion is possible, apply the bindings
  /// inferred so far (plus a distinct new attribute relating back to the
  /// original decls for ones that are missing) to the signature with
  /// getSpecializedSignature so we benefit from the already-fixed substitutions
  /// being applied to the input types.  This can make them more concrete and
  /// help with inferring dependent types based on already-bound parameters.  If
  /// we inferred a value for the parameter from previous arguments, substitute
  /// it into the expected types of subsequent arguments.  This allows us to
  /// handle dependent argument types like:
  ///     fn foo[dt: DType](p: UnsafePointer[Scalar[dt]], v:
  ///     Scalar[p.type.type]):
  /// where the type of 'v' depends on 'dt' being inferred.

  // Determine if we can construct the requested type given the existing value
  // we have.  If so, get the type inferred signature of the init method that
  // would make it work.
  IREmitter emitter(getDeclScope(), ExprContext::EC_CallArgValue);

  // If this is a struct type, try to infer by implicit conversion. Non-struct
  // type should have been handled above by `canZeroCostConvert` if possible.
  // `canConstructType` call below looks up `__init__`, which does not make
  // sense on non-struct type either.
  if (sugarIsa<StructType>(expectedType)) {
    // The expected type may be parameterized, and that type may both have
    // parameters that we are trying to infer as well as parameters that are
    // already known.  For example, if expectedType is known to be
    // 'SIMD[uint8, 1]', then we can infer which constructor to use when the
    // input is an IntLiteral.
    //
    // On the other hand, if expectedType is something like 'SIMD[?, 1]' and the
    // argument is an Int8, then we need the implicit conversion to infer the
    // base element.  Our solution to this is to rip and replace parameters that
    // contain unbound parameters, replacing them with UnboundAttr so inference
    // can find them.
    auto nonParamType =
        expectedType.getWithUnknownParametersReplaced(emitter.shared);
    FailureOr<PValue> pValue = OverloadSet::canConstructType(
        nonParamType,
        CallOperands(CallSyntax::kImplicitConvert, operand.expr,
                     {{argVal, operand.expr}}),
        emitter.getDeclScope());
    if (llvm::failed(pValue)) {
      auto &diag = getDiag(operand.expr->getLoc());
      diag << "cannot convert to type with a previously diagnosed error";
      return failure();
    }

    // If we found one, we succeed if the returned type is compatible with the
    // expected type.  Infer the parameters of this overload candidate against
    // the computed result type of the initializer.
    if (auto callee = pValue.value()) {
      auto initSig = sugarCast<FnTypeGeneratorType>(callee.getType());
      switch (matcher.matchTypes(initSig.getUserResultType(), expectedType)) {
      case ParamMatcher::Retry:
        goto RetryLabel;
      case ParamMatcher::Match:
        return success();
      case ParamMatcher::Error:
        matcher.resetError();
        break;
      }
    }
  }

  // Otherwise, none of that worked. We aren't sure what to do here - it could
  // be any of these things, so we need to emit an error.  If out failure is
  // due to an uninferred parameter, and if that parameter had a default, then
  // we can bind it.
  if (savedFailureReason && savedFailureReason->getIfDependentOnUnresolved()) {
    // If we're in the parameter binding list for a call then we can re-evaluate
    // this binding after the arguments of the call are resolved.
    if (syntax == CallSyntax::kParamBindings) {
      hasDeferredGivenParam = true;
      return success();
    }

    // At this point, if we still have an unresolvable dependent type, give it
    // one last shot and try to pull default parameter value
    //
    // fn store[
    //     dtype: DType
    //     width: Int = 1,
    // ](
    //     self: UnsafePointer[Scalar[dtype], ...],
    //     val: SIMD[dtype, width],
    // )
    //
    // # here Int(8) need to be implicitly converted to SIMD[dtype, 1],
    // store(ptr, Int(8))
    //
    // Otherwise, check to see if this is due to an uninferred param with a
    // default value.  If so, bind the default and try again.
    size_t paramIdx = savedFailureReason->getIfDependentOnUnresolved().value();
    DefaultValueHandler defaultHandler(declaredParamPogs);
    if (auto value = defaultHandler.getDefault(paramIdx)) {
      assert(!evaluator.getIndexBindings()[paramIdx] &&
             "shouldn't have inferred this if we failed because of it");
      value = evaluator.getReboundAttribute(value);
      if (failed(setInferredValue(paramIdx, value)))
        return failure();
      goto RetryLabel;
    }
  }

  auto &diag = emitWrongTypeDiag(expectedType);
  if (savedFailureReason)
    savedFailureReason->addExplanation(diag);
  return failure();
}

/// Infer and emit a single value for a parameter binding. This returns
/// failure if it emits a diagnostic, otherwise is returns a parameter value
/// if resolved, or null if deferred.
FailureOr<TypedAttr>
ParamInf::inferAndEmitOneParam(ASTExprAnd<AnyValue> binding,
                               ASTType expectedType, size_t paramIdx) {
  TypedAttr bindingVal = binding.ir.getIfPValue();
  assert(bindingVal && "Parameters are always PValue's");

  // We don't typecheck the '_' magic parameter, we propagate it.
  //
  // NOTE: we have to return a `_` here to mark the parameter has been
  // explicitly unbound instead of `nullptr` (maybe unless we know this is not a
  // partial binding?). Consider the following cases
  //
  // struct T[a : Int = 1] : pass
  // comptime T1 = T[_]
  // comptime T2 = T[]
  //
  // if we return nullptr here, ParamInf can not distinguish between T1 and T2,
  // and in both cases, `a` will be bound with the default value.
  //
  // NOTE: in a non-partial binding context, `_` can be also used as a place
  // holder, in this case we don't infer it to `_`
  if (isa<UnboundAttr>(bindingVal)) {
    if (partial)
      return TypedAttr(UnboundAttr::get(expectedType));
    // TODO: should we even allow using `_` in a concrete binding context? maybe
    // just raise an error here to be less ambiguous.
    return TypedAttr(); // Deferred
  }

  // If the expected type has unresolved bindings, try to infer them from the
  // argument.  This is a non-trivial operation because we support inferring
  // from the value directly, but also inferring as a result of implicit
  // conversions.
  if (paramFinder.hasReferences(expectedType)) {
    if (failed(inferOneOperand(binding, paramIdx, expectedType,
                               ArgConvention::ReadReg, declaredParamPogs,
                               CallSyntax::kParamBindings)))
      return failure();
  }

  // We might have inferred more parameter after `inferOneOperand`.
  expectedType = evaluator.getReboundType(expectedType);

  if (paramFinder.hasReferences(expectedType)) {
    hasDeferredGivenParam = true;
    return TypedAttr(); // Deferred.
  }

  // Check the type matches what is expected, and perform an implicit
  // conversion if needed.
  if (expectedType.isEqualCanon(bindingVal.getType()))
    // Align sugar if necessary.
    return ParamOperatorAttr::getRebind(bindingVal, expectedType);

  // If the parameter can be implicitly converted, do so.
  IREmitter emitter(getDeclScope(), EC_TypeParamValue);
  if (IREmitter::canImplicitlyConvertToType(
          {bindingVal, binding.expr}, expectedType, emitter.getDeclScope())) {
    ValueDest tmpDest(EC_CallParamValue);
    CValue converted = emitter.emitImplicitConversionToType(
        {bindingVal, binding.expr}, expectedType, tmpDest);
    return converted.getIfPValue().get();
  }

  // Otherwise, the parameter is simply the wrong type, emit an error about this
  // problem.
  DeclResolver::DeclScopeChanger x(&(getDeclScope()));
  MojoInflightDiag &diag = getDiag({});
  if (declIfKnown) // Why only structs? Seems arbitrary, push higher?
    diag << "'" << *declIfKnown->getUserNameIfOperation() << "' ";
  diag << "parameter "
       << ParamDeclRefAttr::get(declaredParamPogs.getName(paramIdx),
                                declaredParamTypes[paramIdx])
       << " has " << expectedType << " type, but value has type "
       << bindingVal.getType() << binding.expr->getRange();

  return failure();
}

// A simple wrapper around `overwriteIndexBinding` to ensure sugar is aligned
// before overwriting parameter value.
// Notable, this method does not check there is no existing parameter inferred
// and unconditional overwrite everything.
LogicalResult ParamInf::setInferredValue(size_t paramIdx, TypedAttr paramVal) {
  paramVal = evaluator.getReboundAttribute(paramVal);
  ASTType targetType = evaluator.getReboundType(declaredParamTypes[paramIdx]);
  // Type must be equal
  assert(targetType.isEqualCanon(paramVal.getType()));

  // now align sugar
  if (paramVal.getType() != targetType)
    paramVal = ParamOperatorAttr::getRebind(paramVal, targetType);

  evaluator.overwriteIndexBinding(paramIdx, paramVal);

  if (isa<UnboundAttr>(paramVal))
    return success();

  ArrayRef<ConstraintAttr> constraints =
      declaredParamPogs.getPogs()[paramIdx].getConstraints();
  if (constraints.empty())
    return success();

  // Verify all constraints are satisfied, collecting unprovable constraints.
  ConstraintResult result = checkConstraints(
      getDeclScope(), declaredParamPogs, constraints, /*origConstraints=*/{},
      getDiag, &unprovableConstraints, &evaluator);

  // TODO: how about we just emitting unprovable error here right away?
  return success(result == ConstraintResult::Satisfied);
}

/// Infer all of the parameters we can from 'givenBindings'.
///
/// The 'partial' field specifies this is
/// performing a partial binding - e.g. because this is not a full type
/// binding, or because more params can be inferred from arguments to the
/// call.
///
/// On failure, this will emit a diagnostic through the 'getDiag' callback.
LogicalResult ParamInf::inferFromParamList() {
  // Notice, but strip out, the ellipsis if present.
  bool hasEllipsis = false;
  CallOperands tmpOperands(getGivenBindings().syntax,
                           getGivenBindings().callExpr);
  if (llvm::any_of(getGivenBindings().values, [](const OperandValue &binding) {
        return isa<EllipsisAttr>(binding.ir.getIfPValue().get());
      })) {
    hasEllipsis = true;
    // Rebuild the operands list without it.  We only do this if present as a
    // micro-optimization.
    for (auto binding : getGivenBindings().values) {
      if (!isa<EllipsisAttr>(binding.ir.getIfPValue().get()))
        tmpOperands.values.push_back(binding);
      else if (!partial) {
        getDiag(binding.expr->getLoc())
            << "'...' is not allowed in concrete parameter bindings";
        return failure();
      }
    }
  }

  // Use the temporary operands list if we had to remove an ellipsis, otherwise
  // use the original operands list.
  auto &givenBindings = hasEllipsis ? tmpOperands : this->getGivenBindings();

  // Do basic validation of the argument list using shared logic.
  // TODO: Integrate this into the logic below.
  OperandValueList variadicKwOperands;
  auto [kwDiagRes, kwDiagNames] = givenBindings.diagnoseKeywordOperands(
      declaredParamPogs, variadicKwOperands, /*allowMissingKwOnly=*/true);
  if (kwDiagRes != CallOperands::KwDiagResult::kValid) {
    MojoInflightDiag &diag = getDiag({});
    switch (kwDiagRes) {
    case CallOperands::KwDiagResult::kMissingKwOnly:
      emitMissing(diag, kwDiagNames, "keyword-only parameter");
      break;
    case CallOperands::KwDiagResult::kOutOfOrderInferredKw:
      emitOutOfOrderInferredKw(diag, kwDiagNames);
      break;
    case CallOperands::KwDiagResult::kPosOnlyPassedByKw:
      emitPosOnlyPassedByKw(diag, kwDiagNames, "parameter");
      break;
    case CallOperands::KwDiagResult::kUnknownKeywords:
      emitUnknownKeywords(diag, kwDiagNames, "parameter");
      break;
    default:
      llvm_unreachable("unknown KwDiagResult");
    }
    return failure();
  }

  auto [posDiagRes, posDiagNames] = givenBindings.diagnosePosOperands(
      declaredParamPogs, /*allowCountMismatch=*/true);
  if (posDiagRes == CallOperands::PosDiagResult::kByPosAndKw) {
    emitByPosAndKw(getDiag({}), posDiagNames, "parameter");
    return failure();
  }

  // Parameter inference and call emission rely on this function not failing
  // early due to missing or too many positional parameters.
  assert(posDiagRes == CallOperands::PosDiagResult::kValid &&
         "positional parameter operand check failed unexpectedly");

  // We may have pre-checked and out-of-order inferred parameters.  Avoid
  // stomping on them.
  auto applyBinding = [&](size_t idx, TypedAttr paramVal) -> LogicalResult {
    // Ignore this if the parameter value is deferred.
    if (!paramVal)
      return success();

    auto existing = evaluator.getIndexBindings()[idx];
    if (!existing)
      return setInferredValue(idx, paramVal);

    assert(isEqualCanon(existing, paramVal) &&
           "inferred to different values but didn't notice");

    return success();
  };

  size_t posIdx = 0, numParams = givenBindings.size();
  DefaultValueHandler defaultHandler(declaredParamPogs);
  for (auto [idx, pog] : llvm::enumerate(declaredParamPogs.getPogs())) {
    if (idx < getNumPreCheckedParam()) {
      ++posIdx; // Prechecked, already installed (or not, if _).
      continue;
    }

    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = evaluator.getReboundType(declaredParamTypes[idx]);

    // Skip over any provided keyword parameters when matching things up, we
    // handle them separately below.
    while (posIdx < numParams && givenBindings[posIdx].keyword)
      ++posIdx;

    // If we have a varargs parameters, then it will eat the rest of the
    // parameters, but we have to check each of them.
    if (declaredParamPogs.isPosVarArg(idx)) {
      // If there are no parameter values, then leave the parameter uninferred
      // for now.  It could be inferred from an call-argument or be left
      // unbound.
      if (posIdx == numParams)
        continue;

      // Unpacked variadics (`Tuple[*elts]` where elts is a variadic list) can
      // be passed directly as a whole variadic parameter.
      auto expectedVA = sugarCast<VariadicType>(expectedType);
      if (auto unpacked = dyn_cast<UnpackedAttr>(
              givenBindings[posIdx].ir.getIfPValue().get())) {
        // FIXME: Make sure to only unpack *x in pos varargs and **x in kw
        // varargs.
        FailureOr<TypedAttr> paramVal = inferAndEmitOneParam(
            {unpacked.getValue(), givenBindings[posIdx].expr}, expectedVA, idx);
        // Exit if an error was already emitted.
        if (failed(paramVal) || failed(applyBinding(idx, *paramVal)))
          return failure();
        ++posIdx;
        continue;
      }

      // Otherwise, we infer the variadic to be the elements of the variadic
      // list being passed in.
      Type varArgsEltType = expectedVA.getElementType();
      SmallVector<TypedAttr> elements;
      bool isDeferred = false;
      while (posIdx != numParams) {
        // This pass just skips keyword parameters, they are handled later.
        if (givenBindings[posIdx].keyword) {
          ++posIdx;
          continue;
        }

        // Passing `_` to a variadic is not allowed. Users should pass `*_` to
        // unbind a variadic parameter.
        if (isa<UnboundAttr>(givenBindings[posIdx].ir.getIfPValue().get())) {
          auto &diag = getDiag(givenBindings[posIdx].expr->getLoc());
          diag << "unbound syntax (i.e. `_`) cannot be passed as a variadic "
                  "parameter";
          return failure();
        }

        // FIXME: pack and install variadics parameter correctly.
        FailureOr<TypedAttr> paramVal =
            inferAndEmitOneParam(givenBindings[posIdx], varArgsEltType, idx);
        if (failed(paramVal)) // Exit if an error was already emitted.
          return failure();

        ++posIdx;
        if (!*paramVal) {
          isDeferred = true;
          continue;
        }

        varArgsEltType = evaluator.getReboundType(varArgsEltType);
        // Realign sugar.
        if (paramVal->getType() != varArgsEltType)
          paramVal = ParamOperatorAttr::getRebind(*paramVal, varArgsEltType);
        elements.push_back(*paramVal);
      }

      if (!isDeferred) {
        expectedVA = cast<VariadicType>(evaluator.getReboundType(expectedVA));
        auto paramVA = VariadicAttr::get(elements, expectedVA);
        if (failed(applyBinding(idx, paramVA)))
          return failure();
      }
      continue;
    }

    // If we have a non-kw param value, it binds to this parameter if it accepts
    // it.
    if (posIdx < numParams && (pog.getPassingKind() == PassingKind::PosOrKw ||
                               pog.getPassingKind() == PassingKind::PosOnly)) {
      FailureOr<TypedAttr> paramVal =
          inferAndEmitOneParam(givenBindings[posIdx], expectedType, idx);
      // Exit if an error was already emitted.
      if (failed(paramVal) || failed(applyBinding(idx, *paramVal)))
        return failure();
      ++posIdx;
      continue;
    }

    // If we're out of positional bindings, or this works with a keyword, try
    // looking for a provided keyword parameter binding.
    if ((pog.getPassingKind() != PassingKind::PosOnly &&
         pog.getPassingKind() != PassingKind::Implicit)) {
      if (const OperandValue *param =
              givenBindings.findKwArg(declaredParamPogs.getName(idx))) {

        FailureOr<TypedAttr> paramVal =
            inferAndEmitOneParam(*param, expectedType, idx);
        // Exit if an error was already emitted.
        if (failed(paramVal) || failed(applyBinding(idx, *paramVal)))
          return failure();
        continue;
      }
    }

    // If this parameter is unspecified but we have a ... in the parameter list,
    // leave it unbound even if it has a default.
    if (hasEllipsis)
      continue;

    // TODO: Handle default parameters etc when not "partial".
  }

  return success();
}

LogicalResult ParamInf::inferForCall(FnTypeGeneratorType signature,
                                     const CallOperands &operands,
                                     const OperandValueList &variadicKwOperands,
                                     bool returnsSelf, bool hasCTADParams) {
  // First try to infer parameters from the already provided bindings.
  if (failed(inferFromParamList()))
    return failure();

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t posOperandIdx = 0;
  size_t numOperands = operands.size();
  PogListAttr argPogs = signature.getArgListAttrs();
  DefaultValueHandler defaultHandler(argPogs);
  for (auto [expectedArgIdx, expectedConvention] :
       llvm::enumerate(signature.getArgConventions())) {

    // There is no provided operand for a by-ref result.
    if (isResultSlot(expectedConvention))
      continue;

    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType =
        evaluator.getReboundType(signature.getArgument(expectedArgIdx));
    if (signature.isKwVarArg(expectedArgIdx)) {
      Type valTy = ASTType(expectedType).getKwargsDictRefValueType();
      auto refValType = RefType::getAnyOrigin(valTy, /*isMut=*/true);
      for (auto operand : variadicKwOperands) {
        // TODO: Passing OwnedMem is a hack that is needed because the value
        // type is not a reference type (and doesn't have a origin), but we
        // still want to type check it. So, passing it as if it was reg-passable
        // happens to just work, until we rectify this. Right now the reason the
        // value type cannot be a reference type is because `Pointer` does not
        // (and in fact cannot) conform to `Copyable & Movable`.
        if (failed(inferOneOperand(operand, expectedArgIdx, refValType,
                                   ArgConvention::OwnedMem, argPogs,
                                   operands.syntax)))
          return failure();
      }
      // This is always last in the operand list.
      posOperandIdx = numOperands;
      continue;
    }

    // If we have a varargs argument, then it will eat the rest of the
    // arguments, but we have to check each of them.
    if (signature.isPosVarArg(expectedArgIdx)) {
      auto expectedVariadic = sugarCast<VariadicType>(expectedType);
      auto varArgsEltType = expectedVariadic.getElementType();
      while (posOperandIdx != numOperands) {
        auto &operand = operands[posOperandIdx];
        if (!operand.keyword &&
            failed(inferOneOperand(
                operand, expectedArgIdx, varArgsEltType,
                signature.getPosVarArgConvention(expectedArgIdx), argPogs,
                operands.syntax)))
          return failure();
        ++posOperandIdx;
      }
      continue;
    }

    // If we have a pack argument, then we're binding a variadic parameter with
    // multiple type values.  We need to consume all remaining arguments and use
    // their RValue types as bindings.
    if (ASTType variadicPackType =
            signature.getIfVariadicPack(expectedArgIdx)) {
      size_t origPosOperandIdx = posOperandIdx;
    RetryLabel:
      // Reset the index before retry.
      posOperandIdx = origPosOperandIdx;
      variadicPackType = evaluator.getReboundType(variadicPackType);
      RefPackType packType = variadicPackType.getVariadicPackInfo(getShared());

      // Figure out that the element type of the list is, e.g. AnyType or
      // Stringable.
      Type elementType = packType.getVariadicElementType();

      // It is possible the pack element types are not being inferred - for
      // example, they could have been explicitly specified.  If this is the
      // case, then we need to perform an implicit conversion to the element
      // type that was explicitly specified.  Be careful though, it is possible
      // the specified type list is completely wrong in length or content.
      VariadicAttr eltsTypesIfResolved =
          dyn_cast<VariadicAttr>(packType.getVariadic());

      SmallVector<TypedAttr> types;
      IREmitter emitter(getDeclScope(), EC_TypeParamValue);
      const ExprNode *packArgExpr = nullptr;
      while (posOperandIdx != numOperands) {
        const auto &operand = operands[posOperandIdx++];
        if (operand.keyword) // Ignore keyword operands.
          continue;

        // Remember the first argument expression for the pack.
        if (packArgExpr == nullptr)
          packArgExpr = operand.expr;

        // If the element types for the pack were specified, convert the value
        // to that type.
        TypedAttr attrForElementType;
        if (eltsTypesIfResolved &&
            types.size() < eltsTypesIfResolved.getValues().size()) {
          attrForElementType = eltsTypesIfResolved.getValues()[types.size()];
        } else {
          // Otherwise, infer the variadic element type from the value's type.
          ASTType toPush = operand.ir.getRValueTypeIfResolvable();
          if (!toPush) {
            getDiag(operand.expr->getLoc())
                << "could not infer type of parameter pack "
                << argPogs.getName(expectedArgIdx)
                << " given value with unresolved type";
            return failure();
          }

          // Infer nonmaterializable types as their materialization target.
          if (ASTType nmTarget = toPush.getNonmaterializableTarget(getShared()))
            toPush = nmTarget;
          Type metatype = toPush.getMetaType();
          attrForElementType = TypeParamAttr::get(
              toPush,
              metatype ? metatype : TypeType::get(getShared().getContext()));
          // Make sure the value is compatible with the expected trait, this
          // produces better error messages.  It would be great to sink this
          // into matchType at some point!
          if (!IREmitter::canImplicitlyConvertToType(
                  {attrForElementType, operand.expr}, elementType,
                  emitter.getDeclScope())) {
            getDiag(operand.expr->getLoc())
                << "could not convert element of "
                << argPogs.getName(expectedArgIdx) << " with type " << toPush
                << " to expected type " << elementType;
            return failure();
          }

          // Perform a conversion (e.g. from a concrete to trait type) as
          // needed.
          attrForElementType =
              emitter.emitPValue({attrForElementType, operand.expr},
                                 EC_TypeParamValue, elementType);
          assert(attrForElementType && "just checked this failure");
        }
        types.push_back(attrForElementType);
      }

      // Infer the value of type list from the types we have.
      auto variadicType =
          sugarCast<VariadicType>(packType.getVariadic().getType());

      // If there are no arguments for the pack, use the location of the call.
      if (packArgExpr)
        packArgExpr = getGivenBindings().getExpr();
      ParamMatcher matcher(packArgExpr, *this);
      switch (matcher.matchParams(VariadicAttr::get(types, variadicType),
                                  packType.getVariadic())) {
      case ParamMatcher::Retry:
        goto RetryLabel;
      case ParamMatcher::Match:
        continue;
      case ParamMatcher::Error:
        return failure();
      }
    }

    // Check for any more positional operands.
    while (posOperandIdx != numOperands && operands[posOperandIdx].keyword)
      ++posOperandIdx;

    // Handle positional arguments.
    if (posOperandIdx < numOperands) {
      if (failed(inferOneOperand(operands[posOperandIdx++], expectedArgIdx,
                                 expectedType, expectedConvention, argPogs,
                                 operands.syntax)))
        return failure();
      continue;
    }

    // Handle case when there are no more provided positional operands.
    // Check if a keyword operand was provided for this argument
    if (const OperandValue *kwOperandOr =
            operands.findKwArg(signature.getArgName(expectedArgIdx))) {
      if (failed(inferOneOperand(*kwOperandOr, expectedArgIdx, expectedType,
                                 expectedConvention, argPogs, operands.syntax)))
        return failure();
      continue;
    }

    // If not, and this argument has a default value, then just skip it. We
    // can't infer from default values since its type already matches the
    // argument type. If its type is dependent, we already know the value is
    // well-formed regardless of the parameter's value.
    if (defaultHandler.getDefault(expectedArgIdx))
      continue;

    // Otherwise we have an argument count mismatch, just fail.
    return failure();
  }

  // If we have left over operands, then this signature cannot match.
  if (posOperandIdx != numOperands && !signature.getMetadata().hasAnyVarArg())
    return failure();

  // If this is a result in a returnsSelf function like an __init__, infer
  // self parameters (which could be specialized and shadowed).
  //   struct Example[T: AnyType]:
  //      fn __init__[U: Movable](owned value: U) -> Example[U]:
  //         pass
  // All of the arguments have been resolved here so all parameters must be
  // inferred (or not able to).
  if (returnsSelf && failed(inferSelfFromInitResult(signature)))
    return failure();

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
  if (hasCTADParams && failed(inferCTADParams(signature, operands)))
    return failure();

  if (hasDeferredGivenParam) {
    // Simply try it again now that more parameter has been inferred.
    if (failed(inferFromParamList()))
      return failure();
  }

  // Lastly, See if we can fulfill any missing parameters with default values
  // for their type (variadic attr always have a default empty value if not
  // inferable).
  if (failed(inferFromDefaults()))
    return failure();

  // We succeed iff we inferred a value for this parameter.
  return success();
}

/// Given an incomplete parameter binding set, try to infer parameters on Self
/// of a method from the first argument.
LogicalResult ParamInf::inferCTADParams(FnTypeGeneratorType signature,
                                        const CallOperands &operands) {
  // Consider "conditional conformance" cases like:
  //     struct X[A: AnyType]:
  //       fn foo[B: Movable](self: X[B]): ...
  //
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

  // We can only do this if we have an argument.
  assert(!operands.empty() && !operands[0].keyword &&
         "init should have positional self argument");

  auto selfConvention = signature.getArgConventions()[0];
  ASTType declaredSelfType =
      RefType::stripRefConvention(signature.getArgument(0), selfConvention);

  // Get the ASTDecl for the declared self type.  This will give us the struct
  // that we are referring to without bound parameters.
  ASTDecl *decl = declaredSelfType.getDecl(getShared());
  if (!decl)
    return success();

  // Get the Self type, with parameters bound to the structs CTAD parameters.
  ASTType selfType = decl->getTypeDeclSelf();
  if (!selfType)
    return success();

  // We need to convert named parameters like "T", which are ParamDeclRefAttr
  // into ParamIndexRefAttr(0) style of representation.
  if (auto structDecl = dyn_cast<StructDeclOp>(decl->getIfOperation())) {
    IndexRefRemapper remapper(structDecl.getParams(), /*resultParams*/ {});
    selfType = remapper.replace(selfType.mlirType);
  }

  // If passing self by reference, wrap the Self type with the RefType
  // paraphernalia like origins.
  if (hasAddress(selfConvention))
    selfType =
        sugarCast<RefType>(signature.getArgument(0)).getWithElement(selfType);

  // Infer the first operand against this type - it was presumably already
  // inferred against the methods declared type of 'self' as well.
  auto argPogs = signature.getArgListAttrs();
  return inferOneOperand(operands[0], /*argIdx*/ 0, selfType, selfConvention,
                         argPogs, operands.syntax);
}

// Infer any missing parameter from defaulted value (this is supposed to be
// invoked after both parameter list and argument list has been scanned).
LogicalResult ParamInf::inferFromDefaults() {
  // Lastly, See if we can fulfill any missing parameters with default values
  // for their type (variadic attr always have a default empty value if not
  // inferable).

  DefaultValueHandler defaultHandler(declaredParamPogs);
  for (size_t idx = 0, e = declaredParamTypes.size(); idx != e; ++idx) {
    if (evaluator.getIndexBindings()[idx])
      continue;

    // If available, we use a default parameter value.
    // FIXME: Shouldn't this go into inference itself like empty variadic
    // binding is?
    if (TypedAttr defaultParam = defaultHandler.getDefault(idx);
        defaultParam && !sugarIsa<UnknownAttr>(defaultParam)) {

      // Skip anything that is prechecked
      // TODO: move this out the if condition: this is crazy that we still need
      // to infer a empty variadic (even when there is a prechecked unbound
      // attribute). This probably means that something is wrong in
      // ParamBindings.
      if (idx < paramBindings.getNumPreCheckedParams())
        continue;

      // Default parameter values may reference other parameter values, so we
      // need to evaluate these.
      // If the default value is dependent, and we can not fully resolve all its
      // dependencies, do not try to set the value of it.
      defaultParam = evaluator.getReboundAttribute(defaultParam);
      if (!paramFinder.hasReferences(defaultParam)) {
        if (failed(setInferredValue(idx, defaultParam)))
          return failure();
      }
    }

    // FIXME: this need a more systematical fix.
    // Determine if we can use a default parameter for CTAD
    if (paramBindings.ctadPogs.size() > idx) {
      PassingKind passingKind = paramBindings.ctadPogs[idx].getPassingKind();
      ArrayRef<TypedAttr> defaults;
      unsigned numCtadParams;
      unsigned normalizedIdx;
      if (passingKind == PassingKind::KwOnly) {
        defaults = paramBindings.defaultKwTypeParams;
        numCtadParams = paramBindings.numKwOnlyCtadParams;
        normalizedIdx = idx - paramBindings.numPosCtadParams;
      } else {
        defaults = paramBindings.defaultPosTypeParams;
        numCtadParams = paramBindings.numPosCtadParams;
        normalizedIdx = idx;
      }

      size_t defaultStartIdx = numCtadParams - defaults.size();
      if (normalizedIdx < numCtadParams && normalizedIdx >= defaultStartIdx) {
        TypedAttr defaultCTAD = defaults[normalizedIdx - defaultStartIdx];
        if (failed(setInferredValue(idx, defaultCTAD)))
          return failure();
      }
    }

    // If not specified/inferrable, variadic always have a default empty value.
    if (!partial && declaredParamPogs.isPosVarArg(idx)) {
      // FIXME: This isn't rewriting the variadic list for dependent types.
      auto type = declaredParamTypes[idx];
      auto empty = VariadicAttr::get({}, sugarCast<VariadicType>(type));
      if (failed(setInferredValue(idx, empty)))
        return failure();
    }
  }

  return success();
}

// TODO: We probably don't have to do this? This is just to make sure we reached
// the same end state as the old parameter inference. Understand why.
void ParamInf::finalizeWithUnbound() {
  // This is the end of parameter inference, replace any fail-to-infer parameter
  // to unboundAttr.
  for (size_t idx = 0, e = declaredParamTypes.size(); idx != e; ++idx) {
    TypedAttr inferred = evaluator.getIndexBindings()[idx];
    if (!inferred || sugarIsa<UnboundAttr>(inferred)) {
      Type targetType = evaluator.getReboundType(declaredParamTypes[idx]);
      inferred = UnboundAttr::get(targetType);

      evaluator.overwriteIndexBinding(idx, inferred);
    }
  }
}
