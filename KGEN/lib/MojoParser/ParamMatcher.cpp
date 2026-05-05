//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParamMatcher.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamBindings.h"

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
#include "KGEN/POPDialect/POPAttrs.h"

#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

#define DEBUG_TYPE "PARAMINF"

extern bool checkConventionsConvertible(ArgConvention expectedConv,
                                        ArgConvention actualConv);

//===----------------------------------------------------------------------===//
// InferenceFailure
//===----------------------------------------------------------------------===//

void MatchFailure::addExplanation(MojoInflightDiag &diag) const {
  SharedState *shared = diag.getSharedIfActive();
  if (!shared)
    return;

  if (isa<Unclassified, UnprovableConstraints>(info))
    return;

  if (isa<DependsOnUnresolved>(info)) {
    auto details = cast<DependsOnUnresolved>(info);
    diag << ", it depends on an unresolved parameter "
         << ParamIndexRefAttr::get(/*depth*/ 0, details.paramIdx,
                                   UnresolvedType::get(shared->getContext()));
    return;
  }

  if (isa<UnboundButInferrable>(info)) {
    auto details = cast<UnboundButInferrable>(info);
    diag << ", it overwrites an explicitly unbound parameter '_' at #"
         << details.unboundParamIdx;
    return;
  }

  if (isa<ValueConflict>(info)) {
    auto failure = cast<ValueConflict>(info);
    diag << ", it inferred to two different values: " << failure.v1 << " and "
         << failure.v2;
    diag.attachNote(diag.getLastLoc())
        << "try `rebind` them to one type if they will be "
           "concretized to the same type";
    return;
  }

  auto failure = cast<TypeConflict>(info);
  if (sugarIsa<NonStructTypeType>(failure.paramType)) {
    if (auto anyStruct = sugarDynCast<StructMetaType>(failure.argParamType)) {
      diag << ", argument type " << ASTType(anyStruct.getType())
           << " is a user-defined struct type, so it is not "
              "a 'non_struct_type'";
      return;
    }
  }

  if (sugarIsa<TraitType>(failure.paramType)) {
    if (auto anyStruct = sugarDynCast<StructMetaType>(failure.argParamType)) {
      diag << ", argument type " << ASTType(anyStruct.getType())
           << " does not conform to trait " << failure.paramType;
      return;
    }
    if (sugarIsa<TraitType>(failure.argParamType)) {
      diag << ", argument type " << failure.argParamType
           << " is not a child trait of " << failure.paramType;
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// ParamMatcher
//===----------------------------------------------------------------------===//

// This macro is used to propagate the non-success codes.
#define PROP(EXPR)                                                             \
  do {                                                                         \
    if (failed(EXPR))                                                          \
      return failure();                                                        \
  } while (0)

ParamMatcher::ParamMatcher(const ExprNode *expr, InferenceState &state,
                           bool allowImplicitConversions)
    : expr(expr), state(state), shared(state.shared),
      allowImplicitConversions(allowImplicitConversions) {}

ParamMatcher::FailableScope::FailableScope(ParamMatcher &matcher)
    : matcher(matcher),
      inferredIdx(matcher.state.evaluator.getNumIndexBindings()) {
  // Keep track of which bindings are inferred at the start of the scope.
  for (unsigned i = 0, e = inferredIdx.size(); i != e; ++i)
    if (matcher.state.evaluator.getIndexBindings()[i])
      inferredIdx.set(i);
}

// If we find an error and want to try another possible way to match, we need to
// undo any falsely inferred parameters while running the experiment. Consider
// something like:
//
// @fieldwise_init
// struct NDBuffer[
//     mut: Bool,
//     //,
//     dtype: DType,
//     rank: Int,
//     origin: Origin[mut=mut],
//     ...
// ](TrivialRegisterPassable):
//   @implicit
//   def __init__(
//       out self,
//       # note that `other` here does NOT uses `Self.origin`
//       other: NDBuffer[Self.dtype, Self.rank, ...],
//   ):
//        pass
//
// In this case, we can should NOT infer `Self.origin` to `other.origin`.
// This could happens without the reset since we can successfully match
// `other` against `Self` all the way till we pass `origin`. At that point,
// although matcher returns failure, `origin` has been inferred, and the
// implicit convertibility is then tested against parameter inferred from a
// failed match.
void ParamMatcher::FailableScope::revert() {
  assert(matcher.failureReason && "not in error state");

  // Reset any inferred bindings that were not inferred when the scope was
  // created.
  for (unsigned i = 0, e = inferredIdx.size(); i != e; ++i)
    if (!inferredIdx[i])
      matcher.state.evaluator.overwriteIndexBinding(i, nullptr);

  matcher.failureReason.reset();
}

static void appendLocallyDefinedParam(Type paramType,
                                      ParameterEvaluator &evaluator) {
  auto paramIdx = evaluator.getIndexBindings().size();
  auto name = StringAttr::get(paramType.getContext(),
                              "#.ParamMatcher.#" + Twine(paramIdx));
  evaluator.appendIndexBinding(ParamDeclRefAttr::get(name, paramType));
}

LogicalResult ParamMatcher::matchFunctionTypes(FnTypeGeneratorType actual,
                                               FnTypeGeneratorType expected) {
  // FIXME: "actual" ends up with parameter names sometimes, not always index
  // references. If this happens we need to convert to make this work correctly.

  // See paramIndexRefDepth's comments for what this increment is for.
  // The increment was solely for the matchFunctionTypes call, so undo it.
  llvm::SaveAndRestore depth(paramIndexRefDepth, paramIndexRefDepth + 1);

  // Functions with different parameterization cannot be converted between each
  // other. If the types are equal but the passing conventions are different,
  // then the conversion is allowed.
  // TODO: Consider default parameter values and enable parameter inference to
  // reconcile differences. For now we do a hack to allow VariadicPack to work,
  // allowing inferred params only.
  size_t nextActualParamIdx = 0;

  if (expected.getInputParamTypes().size() < actual.getInputParamTypes().size())
    return error(MatchFailure::Unclassified{});

  ParameterEvaluator actualEvaluator(shared.getParameterEvaluator());
  ParameterEvaluator expectedEvaluator(shared.getParameterEvaluator());

  // Match up and shadow the parameter defined by the FnTypeGenerator
  for (auto [idx, pType] : llvm::enumerate(expected.getInputParamTypes())) {
    // These always exist in 'expected', install a binding for the param to make
    // the two signature lists consistent.
    appendLocallyDefinedParam(expectedEvaluator.getReboundType(pType),
                              expectedEvaluator);

    // Ran out of params or mismatch on types.
    if (nextActualParamIdx >= actual.getInputParamTypes().size() ||
        actual.getInputParamTypes()[nextActualParamIdx] != pType) {

      if (expected.getParamListAttrs().getPassingKind(idx) !=
              PassingKind::Inferred &&
          expected.getParamListAttrs().getPassingKind(idx) !=
              PassingKind::Implicit)
        return error(MatchFailure::Unclassified{});
      // Make sure to put a parameter binding in even though it doesn't exist in
      // 'actual'.
      continue;
    }

    // If the types match and the parameter exists in 'actual', install the same
    // binding we used for expected.
    actualEvaluator.appendIndexBinding(
        expectedEvaluator.getIndexBindings()[idx]);
    ++nextActualParamIdx;
  }
  if (nextActualParamIdx != actual.getInputParamTypes().size())
    return error(MatchFailure::Unclassified{});

  auto actualFnTp =
      cast<FnType>(actualEvaluator.getReboundType(actual.getBody()));
  auto expectedFnTp =
      cast<FnType>(expectedEvaluator.getReboundType(expected.getBody()));

  // Past this point we don't want to look at actual/expected anymore. They
  // don't have the parameters bound right. Clear them to avoid errors in the
  // future.
  actual = {};
  expected = {};

  if (!allowImplicitConversions) {
    // Allow signature types to be converted for free if they differ only in
    // argument names, parameter names, passing kinds, or implicit origins.
    // The must matches `IREmitter::canZeroCostConvert`
    if (actualFnTp.getNumArguments() != expectedFnTp.getNumArguments() ||
        actualFnTp.getArgConventions() != expectedFnTp.getArgConventions() ||
        actualFnTp.getFnEffects() != expectedFnTp.getFnEffects())
      return error(MatchFailure::Unclassified{});
  }

  auto actualEffects = actualFnTp.getFnEffects();
  auto expectedEffects = expectedFnTp.getFnEffects();

  // The function result types must be the same or implicitly convertible.
  ASTType actualResultType = actualFnTp.getUserResultType();
  ASTType expectedResultType = expectedFnTp.getUserResultType();

  // Otherwise, either we allow implicit conversion, or we already ensured
  // zeroCostConversion above.
  bool actualMemResult = actualFnTp.hasMemoryOnlyResult();
  ArrayRef<Type> actualArgTypes =
      actualFnTp.getArguments().drop_back(actualMemResult);
  bool expectedMemResult = expectedFnTp.hasMemoryOnlyResult();
  ArrayRef<Type> expectedArgTypes =
      expectedFnTp.getArguments().drop_back(expectedMemResult);

  // If the actual function is not throwing, and the expected function is,
  // then we can infer the Error type to be Never.
  if (!actualEffects.isThrows() && expectedEffects.isThrows()) {
    FailableScope failableScope(*this);

    // Match the expected error type to Never, but allow this to fail: it may
    // already be some concrete type like Error and that is ok.
    if (failed(matchTypes(NeverType::get(expectedFnTp.getContext()),
                          expectedFnTp.getUserThrownType()))) {
      failableScope.revert();
    }
    expectedArgTypes = expectedArgTypes.drop_back();
    actualEffects.setThrows(true);
  }

  // If the actual result type returns a ref result and the expected returns a
  // value, then we can copy the value to the ref result.  Otherwise, the
  // ref-result-ness needs to line up.
  if (actualFnTp.isRefResult() != expectedFnTp.isRefResult()) {
    // A function returning a value can't promote to a ref result.
    if (expectedFnTp.isRefResult())
      return error(MatchFailure::Unclassified{});
    // We can look through the reference to implicitly copy.
    actualResultType = actualResultType.getReferenceElementType();
    actualEffects.setRefResult(false);

    // If the result types mismatch then we need to do an implicit conversion to
    // succeed, that is checked below.  If they match, we have to do an implicit
    // copy.
    // TODO: Do this the right ParamMatcher way.
    if (isEqualCanon(actualResultType, expectedResultType) &&
        // EXPLICIT-COPY-REF-RETURN: Allow /explicit-only/ copyability here as a
        // hack to support __next__ in iterators to work where we want them to
        // be able to return ref but still conform to Iterator.  Remove this
        // when we have more strong origin support in traits.
        !actualResultType.isCopyable(expr->getLoc(), shared,
                                     /*isImplicit=*/false))
      return error(MatchFailure::Unclassified{});
  }

  // Match the result. When implicit conversions are allowed we allow them here.
  {
    FailableScope failableScope(*this);
    if (failed(matchTypes(actualResultType, expectedResultType))) {
      if (!allowImplicitConversions)
        return failure();

      // TODO: Do this the right ParamMatcher way.
      if (!IREmitter::canImplicitlyConvertToType(
              {UnknownAttr::get(actualResultType), expr}, expectedResultType,
              state.declScope))
        return failure();
      failableScope.revert();
    }
  }

  if (actualEffects != expectedEffects)
    return error(MatchFailure::Unclassified{});

  PROP(matchParams(actualFnTp.getCaptureOrigins(),
                   expectedFnTp.getCaptureOrigins()));

  // Functions with an incompatible number of arguments cannot be converted
  // between each other. The number of arguments should be equal, unless the
  // expected function is variadic.
  // TODO: Consider default argument values.
  std::optional<size_t> expectedVariadicArgIndexOpt =
      expectedFnTp.findPackVarArgIndex();
  if (expectedVariadicArgIndexOpt.has_value()) {
    size_t expectedVariadicArgIndex = expectedVariadicArgIndexOpt.value();
    if (actualArgTypes.size() < expectedVariadicArgIndex) {
      // Caller didn't supply enough arguments.
      return error(MatchFailure::Unclassified{});
    }
  } else { // No variadic
    if (actualArgTypes.size() != expectedArgTypes.size()) {
      // Caller didn't supply the expected number of arguments.
      return error(MatchFailure::Unclassified{});
    }
  }

  bool expectedHasVariadic = expectedVariadicArgIndexOpt.has_value();
  bool actualHasVariadic = actualFnTp.findPackVarArgIndex().has_value();
  // If this is true, then we need to collect a bunch of `actual`'s args into a
  // variadic for `expected`.
  bool collectIntoVariadic = expectedHasVariadic && !actualHasVariadic;

  // "Normal" here means it won't be received by a variadic arg in the expected
  // function.
  size_t numNormalArgs = actualArgTypes.size();
  if (collectIntoVariadic) {
    numNormalArgs = expectedVariadicArgIndexOpt.value();
  }

  // Check all the normal args (which aren't going into a variadic arg).
  for (size_t actualArgIndex = 0; actualArgIndex < numNormalArgs;
       ++actualArgIndex) {
    auto actualConv = actualFnTp.getArgConvention(actualArgIndex);
    ArgConvention expectedConv = expectedFnTp.getArgConvention(actualArgIndex);
    ASTType actualAstType = actualArgTypes[actualArgIndex];
    ASTType expectedAstType = expectedArgTypes[actualArgIndex];

    if (!checkConventionsConvertible(expectedConv, actualConv))
      return error(MatchFailure::Unclassified{});

    Type expectedValueAstType =
        RefType::stripRefConvention(expectedAstType, expectedConv);
    Type actualValueAstType =
        RefType::stripRefConvention(actualAstType, actualConv);
    // Now check that the argument types line up.
    PROP(matchTypes(actualValueAstType, expectedValueAstType));

    // If the argument has a required keyword, then the two must match names.
    if (actualFnTp.getArgListAttrs().getPassingKind(actualArgIndex) ==
            PassingKind::KwOnly ||
        expectedFnTp.getArgListAttrs().getPassingKind(actualArgIndex) ==
            PassingKind::KwOnly) {
      if (actualFnTp.getArgName(actualArgIndex) !=
          expectedFnTp.getArgName(actualArgIndex))
        return error(MatchFailure::Unclassified{});
    }
  }

  // If the expected def has a variadic arg, check all the actual args that will
  // go into it.
  if (collectIntoVariadic) {
    auto expectedVariadicArgIndex = expectedVariadicArgIndexOpt.value();

    ArgConvention expectedConv =
        expectedFnTp.getArgConvention(expectedVariadicArgIndex);

    // Get the variadic pack's element trait.
    ASTType expectedArgVariadicPackType =
        expectedFnTp.getIfVariadicListOrPack(expectedVariadicArgIndex);
    RefPackType refPackType =
        expectedArgVariadicPackType.getVariadicPackInfo(shared);
    ASTType variadicElType = refPackType.getParamListElementType();

    // This works because VariadicPack's element type is always a trait.
    auto expectedTraitType = cast<TraitType>(variadicElType.mlirType);

    TypedAttr variadic = refPackType.getVariadic();
    // As we do our checks, we'll also be calculating the actual kgen.variadic
    // parameter value.
    SmallVector<TypedAttr> elements;
    for (size_t actualArgIndex = numNormalArgs, e = actualArgTypes.size();
         actualArgIndex < e; ++actualArgIndex) {
      auto actualConv = actualFnTp.getArgConvention(actualArgIndex);
      ASTType actualAstType = actualArgTypes[actualArgIndex];

      if (!checkConventionsConvertible(expectedConv, actualConv))
        return error(MatchFailure::Unclassified{});

      Type actualValueAstType =
          RefType::stripRefConvention(actualAstType, actualConv);

      FailableScope failableScope(*this);
      // If the argument types line up, then we can skip the rest of this.
      if (succeeded(matchTypes(actualValueAstType, variadicElType)))
        continue;
      failableScope.revert();

      // We can convert a more general `actual` function (that takes in a trait
      // argument) to a more specific `expected` function that takes in a struct
      // argument, as long as that struct conforms to that trait.
      // In other words, here we're handling function conversions with covariant
      // arguments (see TTSMFS).
      IREmitter emitter(state.getDeclScope(), EC_TypeParamValue);
      // Now, check if the actual arg can be converted to the expected trait.
      PValue actualAstTypeAsVariadicElTrait =
          emitter.emitMetaTypeToTraitConversion(
              {CValue(actualValueAstType), expr}, expectedTraitType);
      if (!actualAstTypeAsVariadicElTrait)
        return error(MatchFailure::Unclassified{});

      // And since we have it, let's use it to build up a kgen.variadic
      // parameter value.
      elements.push_back(actualAstTypeAsVariadicElTrait);
    }

    // Now assemble the kgen.variadic parameter value and match it against the
    // expected one.
    auto varType = ParamListType::get(variadicElType);
    PROP(matchParams(ParamListAttr::get(elements, varType), variadic));

    // Now that the Variadic has a value, infer the TypeList for the
    // VariadicPack as well.
    auto typeListParam =
        expectedArgVariadicPackType.getVariadicPackInfo().typeListStruct;
    auto typeListType = typeListParam.getType();
    // We need to adjust the depth because we took it out of the function type
    // for the expected function type.
    typeListType = IndexDepthAdjuster(/*adjustDepth=*/-1).replace(typeListType);
    typeListType = state.evaluator.getReboundType(typeListType);
    PROP(matchParams(UnknownAttr::get(typeListType), typeListParam));
  }

  // The function types are convertible.
  return success();
}

LogicalResult ParamMatcher::matchTypes(Type actualType, Type expectedType) {
  // Rebind expected type such that a dependent parameter is updated properly in
  // cases like:
  // ParamType[x, x + 1] and x is inferred.
  //
  // TODO: This does not support out-of-order dependencies: ParamType[x + 1, x],
  // we need a more sophisticated worklist to handle it properly.
  expectedType = state.evaluator.getReboundType(expectedType);

  // If the types trivially match then there is no inference to do.
  if (isEqualCanon(actualType, expectedType))
    return success();

  // Parameter type wrappers such as `downcast(T, Trait)` do not change the
  // runtime representation.  Accept those identity-preserving conversions here
  // before trying to infer through the parameter expression itself.
  if (sugarIsa<ParamType>(actualType) && sugarIsa<ParamType>(expectedType) &&
      IREmitter::canZeroCostConvert(actualType, expectedType, shared))
    return success();

  // If the expected type is a parameter ref, then we're binding the specified
  // type to an attribute parameter.
  if (auto expectedParamRef = dyn_cast<ParamType>(expectedType))
    return matchParams(PValue(actualType).get(), expectedParamRef.getParam());

  // Handle when both are metatypes.
  // For example, when we match a Tuple[Int, Bool] against a
  // T: type_of(Tuple[*ArgTypes]), this will infer that the ArgTypes variadic
  // is [Int, Bool].
  if (auto actualMetaType = dyn_cast<StructMetaType>(actualType)) {
    auto actualDRT = actualMetaType.getType();
    if (auto expectedMetaType = dyn_cast<StructMetaType>(expectedType)) {
      auto expectedDRT = expectedMetaType.getType();
      // Ignore if these are two fundamentally different symbols.
      if (actualDRT.getSymbol() != expectedDRT.getSymbol())
        return error(MatchFailure::Unclassified{});

      // Fail if the parameter lists fundamentally mismatch.
      assert(actualDRT.getParamValues().size() ==
                 expectedDRT.getParamValues().size() &&
             "two instances of same struct must have same length param lists");

      // Match up the parameter bindings.
      for (auto [actual, expected] : llvm::zip(actualDRT.getParamValues(),
                                               expectedDRT.getParamValues())) {
        PROP(matchParams(actual, expected));
      }
      return success();
    }
  }

  // Handle when both are StructTypes.
  if (auto actualDRT = dyn_cast<LIT::StructType>(actualType)) {
    if (auto expectedDRT = dyn_cast<LIT::StructType>(expectedType)) {
      // Ignore if these are two fundamentally different symbols.
      if (actualDRT.getSymbol() == expectedDRT.getSymbol()) {
        // Fail if the parameter lists fundamentally mismatch.
        assert(
            actualDRT.getParamValues().size() ==
                expectedDRT.getParamValues().size() &&
            "two instances of same struct must have same length param lists");

        // Match up the parameter bindings.
        for (auto [actual, expected] : llvm::zip(
                 actualDRT.getParamValues(), expectedDRT.getParamValues())) {
          PROP(matchParams(actual, expected));
        }
        return success();
      }
    }
  }

  // Handle various common POP types for convenience, starting with SIMDType.
  if (auto actual = dyn_cast<SIMDType>(actualType))
    if (auto expected = dyn_cast<SIMDType>(expectedType)) {
      PROP(matchParams(actual.getSize(), expected.getSize()));
      return matchParams(actual.getDType(), expected.getDType());
    }

  // POP::ArrayType.
  if (auto actual = dyn_cast<POP::ArrayType>(actualType))
    if (auto expected = dyn_cast<POP::ArrayType>(expectedType)) {
      PROP(matchParams(actual.getSize(), expected.getSize()));
      return matchTypes(actual.getElementType(), expected.getElementType());
    }

  // Handle RefType.
  if (auto actual = dyn_cast<RefType>(actualType))
    if (auto expected = dyn_cast<RefType>(expectedType)) {
      PROP(matchTypes(actual.getElementType(), expected.getElementType()));
      PROP(matchSingleEltStruct(actual.getOrigin(), expected.getOrigin()));
      return matchSingleEltStruct(actual.getAddressSpace(),
                                  expected.getAddressSpace());
    }

  // Handle OriginType.
  if (auto actual = dyn_cast<OriginType>(actualType))
    if (auto expected = dyn_cast<OriginType>(expectedType)) {
      // If the mutable bit is resolved, check for conversions from mut=true to
      // mut=false.
      if (!state.paramFinder.hasReferences(expectedType)) {
        if (!IREmitter::canZeroCostConvert(actualType, expectedType, shared))
          return error(MatchFailure::Unclassified{});
        return success();
      }

      // Otherwise infer it.
      return matchSingleEltStruct(actual.isMutable(), expected.isMutable());
    }

  // Handle PointerType.
  if (auto actual = dyn_cast<PointerType>(actualType))
    if (auto expected = dyn_cast<PointerType>(expectedType)) {
      PROP(matchTypes(actual.getElementType(), expected.getElementType()));
      return matchSingleEltStruct(actual.getAddressSpace(),
                                  expected.getAddressSpace());
    }

  // Handle ParamListType.
  if (auto actual = dyn_cast<ParamListType>(actualType))
    if (auto expected = dyn_cast<ParamListType>(expectedType))
      return matchTypes(actual.getElementType(), expected.getElementType());

  // Handle RefPackType.
  if (auto actual = dyn_cast<RefPackType>(actualType))
    if (auto expected = dyn_cast<RefPackType>(expectedType)) {
      PROP(matchParams(actual.getVariadic(), expected.getVariadic()));
      PROP(matchSingleEltStruct(actual.getOrigin(), expected.getOrigin()));
      return matchParams(actual.getAddressSpace(), expected.getAddressSpace());
    }

  // Handle FuncTypeGeneratorType
  if (auto actual = dyn_cast<FnTypeGeneratorType>(actualType))
    if (auto expected = dyn_cast<FnTypeGeneratorType>(expectedType))
      return matchFunctionTypes(actual, expected);

  // Allowing match a FnLiteralType To a FnType, but not vice versa.
  if (auto actual = dyn_cast<FnLiteralTypeGeneratorType>(actualType))
    if (auto expected = dyn_cast<FnTypeGeneratorType>(expectedType))
      return matchFunctionTypes(actual.getSymbolConstantAttr().getType(),
                                expected);

  // Handle GeneratorType
  if (auto actual = dyn_cast<GeneratorType>(actualType)) {
    if (auto expected = dyn_cast<GeneratorType>(expectedType)) {
      // See paramIndexRefDepth's comments for what this increment is for.
      llvm::SaveAndRestore depth(paramIndexRefDepth, paramIndexRefDepth + 1);

      if (isa<FnType>(actual.getBody()) || isa<FnType>(expected.getBody())) {
        // success()ing two FnTypeGeneratorType should have been handled above
        assert(!isa<FnType>(actual.getBody()) ||
               !isa<FnType>(expected.getBody()));
        return error(MatchFailure::Unclassified{});
      }
      // This a simple type generator, match the input parameter types and body
      // type.
      ArrayRef<Type> actInputs = actual.getInputParamTypes();
      ArrayRef<Type> expInputs = expected.getInputParamTypes();
      if (actInputs.size() == expInputs.size()) {
        // Match up the types of the generators, binding the to consistent
        // values to see if the body lines up after substitution. This isn't
        // related to the top level parameters we're trying to infer, this is
        // relating to the parameters defined in the generator.
        ParameterEvaluator genEvaluator(shared.getParameterEvaluator());
        for (auto [ai, ei] : llvm::zip_equal(actInputs, expInputs)) {
          PROP(matchTypes(ai, ei));
          // Shadow the parameter defined by the generator
          appendLocallyDefinedParam(genEvaluator.getReboundType(ai),
                                    genEvaluator);
        }
        return matchTypes(genEvaluator.getReboundType(actual.getBody()),
                          genEvaluator.getReboundType(expected.getBody()));
      }
    }
  }

  // If the actual type is a reference to a parameter, it might be a local
  // parameter within a function.  The type checker will resolve this using the
  // metatype of the parameter, something like AnyStruct[someType].  This tells
  // us the actual type of the parameter.
  // TODO: Why isn't this a general solution?
  if (auto actualParamRef = dyn_cast<ParamType>(actualType)) {
    if (auto actualMetaType = ASTType(actualType).extractMetaType()) {
      if (auto structMeta = sugarDynCast<StructMetaType>(actualMetaType))
        return matchTypes(structMeta.getType(), expectedType);
      if (auto traitMeta = sugarDynCast<AnyTraitType>(actualMetaType))
        return matchTypes(traitMeta.getTraitType(), expectedType);
    }
  }

  // Handle meta type upcasting.
  // Assumptions needed: overload resolution for e.g.
  // repr[T: Writable](Tuple[*Ts]) inside a fn with `where AllWritable[*Ts]`.
  FailureOr<bool> typeUpCastable = IREmitter::canMetaTypeUpCastTo(
      shared, state.declScope.getLoc(), actualType, expectedType,
      &state.declScope);
  if (succeeded(typeUpCastable) && typeUpCastable.value())
    return success();

  // Ok we have a failure, let's figure out why.

  // If the expected type has unresolved bindings that can't be inferred, then
  // we may have some other parameter that needs to be inferred before this
  // type can be matched.  Report that failure so the caller can decide what
  // to do about it.
  Type adjustedExpectedType = expectedType;
  if (paramIndexRefDepth) {
    IndexDepthAdjuster adjuster(/*adjustDepth=*/-paramIndexRefDepth);
    adjustedExpectedType = adjuster.replace(expectedType);
  }
  if (auto paramIdx = state.paramFinder.findOneReference(adjustedExpectedType))
    return error(MatchFailure::DependsOnUnresolved{*paramIdx});

  // Otherwise we have a generalized mismatch.
  LLVM_DEBUG(llvm::errs() << "CANNOT INFER UNKNOWN TYPES:\n"; actualType.dump();
             expectedType.dump(); llvm::errs() << "\n");
  return error(MatchFailure::Unclassified{});
}

LogicalResult ParamMatcher::matchParams(TypedAttr actualAttr,
                                        TypedAttr expectedAttr) {
  // Rebind expected attr such that a dependent parameter is updated properly in
  // cases like:
  // ParamType[x, x + 1] and x is inferred.
  //
  // TODO: This does not support out-of-order dependencies: ParamType[x + 1, x],
  // we need a more sophisticated worklist to handle it properly.
  expectedAttr = state.evaluator.getReboundAttribute(expectedAttr);

  // If the attrs trivial match then we're done and there is no inference to do.
  if (isEqualCanon(actualAttr, expectedAttr))
    return success();

  // Look through type upcasts to the more derived type.
  actualAttr = UpcastAttr::strip(actualAttr);
  expectedAttr = UpcastAttr::strip(expectedAttr);

  auto getTargetMetaTypeForTypeValue = [](Type targetMetaTp) -> Type {
    if (auto vaTp = sugarDynCast<ParamListType>(targetMetaTp))
      targetMetaTp = vaTp.getElementType();

    if (auto paramTp = sugarDynCast<ParamType>(targetMetaTp)) {
      // If the expected type is parameterized, we strip the meta type.
      // E.g.,
      //
      // def foo[
      //  elt_trait : type_of(AnyType & Foo),
      //  *elt : elt_trait
      // ] : ...
      //
      // foo() # we infer elt : !kgen.param_list<!AnyType & !Foo>
      auto metaType = paramTp.getParam().getType();
      // TODO: should we make AnyTraitType a `MetaType`?.
      if (auto anyTrait = sugarDynCast<AnyTraitType>(metaType))
        return anyTrait.getTraitType();

      return sugarCast<MetaType>(metaType).getType();
    };
    return targetMetaTp;
  };

  // Figure out how to realign the types of the actual/expected attrs.
  if (isEqualCanon(actualAttr.getType(), expectedAttr.getType())) {
    // If the types of both attributes are the same, no adjustment is needed.
  } else {
    FailableScope failableScope(*this);
    auto result = matchTypes(actualAttr.getType(), expectedAttr.getType());
    if (succeeded(result)) {
      // If they are different types but compatible then upcast actualAttr to
      // the expected type.
      IREmitter emitter(state.declScope, EC_TypeParamValue);

      // FIXME: We are running into problems because we have Actual values of
      // "FnTypeGeneratorType" that have named parameters in them, but expected
      // values that want index-based ones.  matchFunctionTypes should convert
      // the former to the latter and we should remove this redundant check for
      // implicit convertibility.
      expectedAttr = state.evaluator.getReboundAttribute(expectedAttr);
      auto expectedType = getCanonicalType(expectedAttr.getType());
      if (IREmitter::canImplicitlyConvertToType(
              {actualAttr, expr}, expectedType, emitter.getDeclScope())) {
        actualAttr = emitter.emitPValue({actualAttr, expr}, EC_TypeParamValue,
                                        expectedType);
        assert(actualAttr && "conversion is double checked");

        if (isEqualCanon(actualAttr, expectedAttr))
          return success();
      }
    } else {
      // Ok something failed, swallow the error and try the next option.
      failableScope.revert();

      // If this is a type expression, try align the type (if possible) before
      // concluding type inconvertibility. This turns things like:
      // #kgen.type<!Int> : !lit.trait<!AnyType> to
      // #kgen.type<!Int> : !lit.trait<!Copyable>
      // Then we can correctly check type value convertibility between
      // #kgen.type<!Int> : !lit.trait<!AnyType> and
      // #param.ref<....> : !lit.trait<!Copyable>
      bool fixableByUpCast = false;
      if (LIT::isTypeExpr(actualAttr) ||
          LIT::isVariadicOfTypeExpr(actualAttr)) {
        auto targetMT = getTargetMetaTypeForTypeValue(expectedAttr.getType());
        ArrayRef<TypedAttr> toCheck(actualAttr);
        if (auto va = sugarDynCast<ParamListAttr>(actualAttr))
          toCheck = va.getValues();

        fixableByUpCast = llvm::all_of(toCheck, [&](TypedAttr typeExpr) {
          assert(LIT::isTypeExpr(typeExpr));
          // Try get the tightest possible metatype bound.
          Type tightestBound = ASTType(typeExpr).extractMetaType();
          if (!tightestBound) {
            // `struct __MLIRType` is the corner case here :(.
            tightestBound = typeExpr.getType();
          }
          FailureOr<bool> upCastable = IREmitter::canMetaTypeUpCastTo(
              shared, state.declScope.getLoc(), tightestBound, targetMT,
              &state.declScope);
          return succeeded(upCastable) && upCastable.value();
        });

        if (fixableByUpCast) {
          SmallVector<TypedAttr> casted =
              llvm::map_to_vector(toCheck, [targetMT](TypedAttr toMap) {
                return TypeParamAttr::get(ASTType(UpcastAttr::strip(toMap)),
                                          targetMT);
              });

          if (auto va = sugarDynCast<ParamListAttr>(actualAttr))
            actualAttr =
                ParamListAttr::get(casted, ParamListType::get(targetMT));
          else
            actualAttr = casted.front();

          auto matchFixed =
              matchTypes(actualAttr.getType(), expectedAttr.getType());
          assert(succeeded(matchFixed));

          // If we found an exact match, we're done.
          if (isEqualCanon(actualAttr, expectedAttr))
            return success();
        }
      }

      if (!fixableByUpCast) {
        if (auto ire = dyn_cast<ParamIndexRefAttr>(expectedAttr)) {
          return error(MatchFailure::TypeConflict{
              ire.getIndex(), expectedAttr.getType(), actualAttr.getType()});
        }
        return error(MatchFailure::Unclassified{});
      }
    }
  }

  // If the actual value is a ? then we never bind to it.
  if (isa<UnboundAttr>(actualAttr))
    return success();

  // If we are dealing with two type constants, we match their values.
  auto actualTypeConst = dyn_cast<TypeParamAttr>(actualAttr);
  auto expectedTypeConst = dyn_cast<TypeParamAttr>(expectedAttr);
  if (actualTypeConst && expectedTypeConst)
    return matchTypes(actualTypeConst.getMlirType(),
                      expectedTypeConst.getMlirType());

  // If both parameters are operator expressions, match them up lexically.
  auto actualOp = dyn_cast<ParamOperatorAttr>(actualAttr);
  auto expectedOp = dyn_cast<ParamOperatorAttr>(expectedAttr);
  if (actualOp && expectedOp &&
      actualOp.getOpcode() == expectedOp.getOpcode() &&
      actualOp.getNumOperands() == expectedOp.getNumOperands()) {
    for (auto [a, b] :
         llvm::zip(actualOp.getOperands(), expectedOp.getOperands())) {
      PROP(matchParams(a, b));
    }
    return success();
  }

  // If one or the other is a rebind, look through it if just processing sugar.
  if (actualOp && actualOp.getOpcode() == POC::Rebind &&
      ASTType(actualOp.getType())
          .isEqualCanon(actualOp.getOperand(0).getType()))
    return matchParams(actualOp.getOperand(0), expectedAttr);
  if (expectedOp && expectedOp.getOpcode() == POC::Rebind &&
      ASTType(expectedOp.getType())
          .isEqualCanon(expectedOp.getOperand(0).getType()))
    return matchParams(actualAttr, expectedOp.getOperand(0));

  // If both parameters are GetWitnessAttrs, match up the insides.
  if (auto actualGetWitness = dyn_cast<GetWitnessAttr>(actualAttr)) {
    if (auto expectedGetWitness = dyn_cast<GetWitnessAttr>(expectedAttr)) {
      // The trait name and witness name are immediates, not parameters, so they
      // must match exactly.
      if (actualGetWitness.getTraitName() ==
              expectedGetWitness.getTraitName() &&
          actualGetWitness.getWitnessName() ==
              expectedGetWitness.getWitnessName()) {
        return matchParams(actualGetWitness.getTypeValue(),
                           expectedGetWitness.getTypeValue());
      }
    }
  }

  // If the expected value is the parameter declaration remember the binding!
  if (auto ire = dyn_cast<ParamIndexRefAttr>(expectedAttr)) {
    // Check if this ParamIndexRefAttr is referring to a param-decl in the
    // ParameterInferenceState's original scope. See paramIndexRefDepth's
    // comments for more about this.
    if (ire.getDepth() == paramIndexRefDepth) {
      Type expectedType = expectedAttr.getType();
      // We are at `paramIndexRefDepth`, but all parameters have been inferred
      // as if at level 0.  Readjust if needed.
      if (paramIndexRefDepth) {
        IndexDepthAdjuster adjuster(/*adjustDepth=*/-paramIndexRefDepth);
        expectedType = adjuster.replace(expectedType);
      }

      expectedType = state.evaluator.getReboundType(expectedType);
      // If the types don't agree, attempt an implicit conversion between the
      // actual value and the expected type.
      if (!isEqualCanon(actualAttr.getType(), expectedType)) {
        // We can only see subtypes in type values, all other values must be
        // aligned perfectly.
        IREmitter emitter(state.declScope, EC_TypeParamValue);
        ASTExprAnd<CValue> toConvert = {actualAttr, expr};
        actualAttr =
            emitter.emitPValue(toConvert, EC_TypeParamValue, expectedType);
        // FIXME: Figure out why this is happening in invalid code.  Something
        // else not propagating failures aggressively?
        if (!actualAttr)
          return error(MatchFailure::Unclassified{});

        assert(actualAttr && "Already checked implicit convertibility");
        assert(isEqualCanon(actualAttr.getType(), expectedType));
      }

      size_t parameterIndex = ire.getIndex();
      assert(parameterIndex < state.evaluator.getNumIndexBindings() &&
             "out-of-bound parameter reference");

      TypedAttr inferredValue =
          state.evaluator.getIndexBindings()[parameterIndex];
      // If this is a new parameter we've inferred, huzzah, remember it.
      if (!inferredValue) {
        if (state.isExplicitlyUnbound(parameterIndex))
          return error(MatchFailure::UnboundButInferrable{parameterIndex});

        if (failed(state.setInferredValue(parameterIndex, actualAttr)))
          return error(MatchFailure::UnprovableConstraints{parameterIndex});
        return success();
      }

      // If we saw this parameter before, make sure it is compatible with
      // (or more specific than) the other values we've inferred.
      if (!isEqualCanon(inferredValue, actualAttr)) {
        return error(MatchFailure::ValueConflict{parameterIndex, inferredValue,
                                                 actualAttr});
      }
      return success();
    }
    // If this is some parameter other than the one we're inferring, assume it
    // will work out.
    return success();
  }

  if (auto actualVar = dyn_cast<ParamListAttr>(actualAttr)) {
    if (auto expectedVar = dyn_cast<ParamListAttr>(expectedAttr)) {
      if (actualVar.getValues().size() != expectedVar.getValues().size())
        return error(MatchFailure::Unclassified{});
      for (auto [act, exp] :
           llvm::zip(actualVar.getValues(), expectedVar.getValues())) {
        PROP(matchParams(act, exp));
      }
      return success();
    }
  }

  if (auto actual = dyn_cast<POP::ArrayAttr>(actualAttr)) {
    if (auto expected = dyn_cast<POP::ArrayAttr>(expectedAttr)) {
      if (actual.getValues().size() != expected.getValues().size())
        return error(MatchFailure::Unclassified{});
      for (auto [act, exp] :
           llvm::zip(actual.getValues(), expected.getValues())) {
        PROP(matchParams(act, exp));
      }
      return success();
    }
  }

  if (auto actualSym = dyn_cast<SymbolConstantAttr>(actualAttr)) {
    if (auto expectedSym = dyn_cast<SymbolConstantAttr>(expectedAttr)) {
      if (actualSym.getSymbol() != expectedSym.getSymbol() ||
          actualSym.getParamValues().size() !=
              expectedSym.getParamValues().size())
        return error(MatchFailure::Unclassified{});
      for (auto [act, exp] : llvm::zip(actualSym.getParamValues(),
                                       expectedSym.getParamValues())) {
        PROP(matchParams(act, exp));
      }
      return success();
    }
  }

  // StoreToMem occurs in parameter expressions in types.
  if (auto actualStore = dyn_cast<StoreToMemAttr>(actualAttr)) {
    if (auto expectedStore = dyn_cast<StoreToMemAttr>(expectedAttr))
      return matchParams(actualStore.getValue(), expectedStore.getValue());
  }

  // StructExtractAttr can also line up.
  if (auto actualExtract = dyn_cast<LIT::StructExtractAttr>(actualAttr)) {
    if (auto expectedExtract = dyn_cast<LIT::StructExtractAttr>(expectedAttr)) {
      if (actualExtract.getField() == expectedExtract.getField())
        return matchParams(actualExtract.getStructValue(),
                           expectedExtract.getStructValue());
    }
  }

  if (auto actualSet = dyn_cast<OriginSetAttr>(actualAttr)) {
    if (auto expectedSet = dyn_cast<OriginSetAttr>(expectedAttr)) {
      // HACK: To phase this in, permit implicitly downcasting to the empty set.
      // This will go away when the default capturing syntax is changed to the
      // any set.
      if (expectedSet.getOperands().empty())
        return success();
      if (actualSet.getOperands().size() != expectedSet.getOperands().size())
        return error(MatchFailure::Unclassified{});
      for (auto [actual, expected] : llvm::zip_equal(
               actualSet.getOperands(), expectedSet.getOperands())) {
        PROP(matchParams(actual, expected));
      }
      return success();
    }
  }

  // Check struct values fieldwise.
  if (auto actualStruct = dyn_cast<LITStructAttr>(actualAttr)) {
    if (auto expectedStruct = dyn_cast<LITStructAttr>(expectedAttr)) {
      if (actualStruct.getType() == expectedStruct.getType()) {
        assert(actualStruct.getValues().size() ==
                   expectedStruct.getValues().size() &&
               "struct of same types disagree on fields");
        for (auto [act, exp] :
             llvm::zip(actualStruct.getValues(), expectedStruct.getValues())) {
          assert(std::get<0>(act) == std::get<0>(exp) && "field name mismatch");
          PROP(matchParams(std::get<1>(act), std::get<1>(exp)));
        }
        return success();
      }
    }
  }

  if (auto actualTab = dyn_cast<ParamListTabulateAttr>(actualAttr)) {
    if (auto expectedTab = dyn_cast<ParamListTabulateAttr>(expectedAttr)) {
      PROP(matchParams(actualTab.getCount(), expectedTab.getCount()));
      PROP(matchParams(actualTab.getGenerator(), expectedTab.getGenerator()));
      return success();
    }
  }

  // We look through non-builtin-apply sugar always.  For builtin apply's, we
  // want to do inference on the sugar itself, because we want to treat x+y
  // as unequal to y+x unless canonically equal.
  auto actualSugar = dyn_cast<SugarAttr>(actualAttr);
  auto expectedSugar = dyn_cast<SugarAttr>(expectedAttr);
  if (actualSugar && expectedSugar &&
      expectedSugar.getKind() == SugarKind::AlwaysInlineBuiltin) {
    if (actualSugar.getCanonical() == expectedSugar.getCanonical())
      return success();
    return matchParams(actualSugar.getSugared(), expectedSugar.getSugared());
  }
  if (actualSugar)
    return matchParams(actualSugar.getExpanded(), expectedAttr);
  if (expectedSugar)
    return matchParams(actualAttr, expectedSugar.getExpanded());

  // If we are matching a type against a downcast (which is from some parametric
  // type) check to see if we can match the actual type against the downcasted
  // type. The actualAttr will have upcasts already stripped from it.
  if (auto down = dyn_cast<DowncastAttr>(expectedAttr)) {
    if (isEqualCanon(actualAttr.getType(), down.getType()))
      return matchParams(actualAttr, down.getInputTypeValue());
  }

  // Ok we have a failure, let's figure out why.

  // If the expected value has unresolved bindings that can't be inferred, then
  // we may have some other parameter that needs to be inferred before this
  // value can be matched.  Report that failure so the caller can decide what
  // to do about it.
  TypedAttr adjustedExpectedAttr = expectedAttr;
  if (paramIndexRefDepth) {
    IndexDepthAdjuster adjuster(/*adjustDepth=*/-paramIndexRefDepth);
    adjustedExpectedAttr = adjuster.replace(expectedAttr);
  }
  if (auto paramIdx =
          state.paramFinder.findOneReference(adjustedExpectedAttr)) {
    assert(!isa<ParamIndexRefAttr>(adjustedExpectedAttr) &&
           "should have inferred this above");
    return error(MatchFailure::DependsOnUnresolved{*paramIdx});
  }

  // Otherwise we have a generalized mismatch.
  LLVM_DEBUG(llvm::errs() << "CANNOT INFER UNKNOWN ATTRS:\n"; actualAttr.dump();
             expectedAttr.dump(); llvm::errs() << "\n");
  return error(MatchFailure::Unclassified{});
}

// Special Hack (tm) for matching mlir values for pointers and references to
// single-element structs.
//
// Things like pointers are defined like this:
//   struct YourPointer[type: ..., address_space: AddressSpace]:
//     def __init__(out self, ref [_] address: type):
//
// When inferring the "address_space" constructor from a !lit.ref that holds it.
// In the !lit.ref, we have (e.g. "3" or some parameter) as an index value.
// However, we need to realize a value for the address_space parameter,
// which is a struct of a struct of an index.
//
// This ends up looking like:
//   ActualAttr = 3 : index
//   Expected=#lit.struct.extract<:@Int #lit.struct.extract<:@AddressSpace
//          *(0,3), "_value">, "_mlir_value"> : index
//
// The "right" solution is to change pointer and reference to take an
// AddressSpace directly.  Until then we do a special hack for these things.
LogicalResult ParamMatcher::matchSingleEltStruct(TypedAttr actualOrig,
                                                 TypedAttr expectedOrig) {
  expectedOrig = state.evaluator.getReboundAttribute(expectedOrig);
  auto actual = ParamOperatorAttr::stripRebind(actualOrig);
  auto expected = ParamOperatorAttr::stripRebind(expectedOrig);

  if (actual == expected)
    return success();

  // If it is an extract from a known struct, then we know there is one field in
  // the struct - we can form a StructAttr around our actual value and recurse.
  if (auto expExtract = sugarDynCast<LIT::StructExtractAttr>(expected)) {
    // If these are two lined up extracts, look through them.
    if (auto actExtract = sugarDynCast<LIT::StructExtractAttr>(actual)) {
      if (expExtract.getField() == actExtract.getField())
        return matchSingleEltStruct(actExtract.getStructValue(),
                                    expExtract.getStructValue());
    }

    // See if we can infer anything from the types, this allows us to infer
    // 'is_mut' parameter from "origin<1>" and "origin<is_mut>".
    PROP(matchTypes(actual.getType(), expected.getType()));
    expExtract = cast<LIT::StructExtractAttr>(
        state.evaluator.getReboundAttribute(expExtract));

    // Ok, we have a struct that seems like it could line up.  See if we can
    // implicitly construct this from a value of this type.  If so, then we
    // assume it is a value-wise initializer that we can infer from.
    //
    // TODO: We could make this more strict by using a keyword argument for the
    // argument value instead of an implicit conv.
    auto expStruct = expExtract.getStructValue();
    // Figure out if the struct is something we can handle.
    auto expDRT = sugarCast<LIT::StructType>(expStruct.getType());

    // Conservatively only handle the types we know have a single field.  We
    // special case these ones to avoid name lookup in common cases.
    if (expDRT.getName().strref() != "AddressSpace" &&
        expDRT.getName().strref() != "Int" &&
        expDRT.getName().strref() != "Bool") {
      // For non-trivial types like Origin[IsMut], we may need to infer
      // parameters, so do a full conversion check.

      // Convert Origin[*(0, 0)] to Origin[?] so we can infer the parameter(s).
      auto nonParamDRT =
          ASTType(expDRT).getWithUnknownParametersReplaced(shared);
      CallOperands ctorOperands(CallSyntax::kImplicitConvert, expr,
                                EC_TypeParamValue, {{actual, expr}});
      FailureOr<PValue> pValue = OverloadSet::canConstructType(
          nonParamDRT, ctorOperands, state.declScope);
      if (failed(pValue) || !pValue.value())
        return error(MatchFailure::Unclassified{});

      // If we succeeded, figure out what the concrete type being inferred would
      // be with any parameters bound.
      auto initSig = sugarCast<FnTypeGeneratorType>(pValue.value().getType());
      // The constructed type is the result of the initializer.
      assert(initSig.getNumArguments() != 0);
      expDRT = sugarCast<LIT::StructType>(initSig.getUserResultType());

      // Finally, perform any implicit conversion of the actual value to
      // whatever the 'value' would provide.
      auto argRVType = RefType::stripRefConvention(initSig.getArgument(0),
                                                   initSig.getArgConvention(0));

      if (actual.getType() != argRVType &&
          IREmitter::canZeroCostConvert(actual.getType(), argRVType, shared)) {
        actual = IREmitter::emitZeroCostConvert(actual, argRVType, shared);
      }
    }

    // Now that we know the actual type, we can infer against a wrapped struct,
    // which can then infer from nested items etc.
    std::tuple<StringAttr, TypedAttr> actualField(expExtract.getField(),
                                                  actual);
    auto wrappedActual = LITStructAttr::get(actualField, expDRT);
    return matchSingleEltStruct(wrappedActual, expStruct);
  }

  return matchParams(actualOrig, expectedOrig);
}
