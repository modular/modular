//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParameterInference.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "ParamBindings.h"

#include "KGEN/MojoParser/ASTDecl.h"
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

#define DEBUG_TYPE "LITEXPRCALLS"

// TODO(MOCO-1106): Put this in a better place.
extern bool checkConventionsConvertible(ArgConvention expectedConv,
                                        ArgConvention actualConv);

//===----------------------------------------------------------------------===//
// InferenceFailure
//===----------------------------------------------------------------------===//

void InferenceFailure::addExplanation(MojoInflightDiag &diag) const {
  if (isa<NotFoundFailure>(info)) {
    diag << ", it isn't used in any argument";
    return;
  }

  if (isa<ValueConflictFailure>(info)) {
    auto failure = cast<ValueConflictFailure>(info);
    diag << ", it inferred to two different values: " << failure.v1 << " and "
         << failure.v2;
    diag.attachNote(diag.getLastLoc())
        << "try `rebind` them to one type if they will be "
           "concretized to the same type";
    return;
  }

  auto failure = cast<TypeConflictFailure>(info);
  if (sugarIsa<TypeType>(failure.paramType)) {
    if (auto anyStruct = sugarDynCast<StructMetaType>(failure.argParamType)) {
      diag << ", argument type " << ASTType(anyStruct.getType())
           << " is not a '@register_passable(\"trivial\")' type, so "
              "does not satisfy AnyTrivialRegType";
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
// ParameterInferenceDiagnostics
//===----------------------------------------------------------------------===//

void ParameterInferenceDiagnostics::addExplanation(MojoInflightDiag &diag) {
  // Pick the first diagnostic for the earliest parameter after numActual.
  const FailedInference *best = nullptr;
  for (const FailedInference &failure : diags) {
    // Don't report diagnostics when failure occurred from a default value,
    // we need a location.
    if (!failure.argExpr)
      continue;
    // If we have a best match for an earlier parameter, ignore this one.
    if (best && best->paramIdx <= failure.paramIdx)
      continue;
    // Otherwise this is the best we've found.
    best = &failure;
  }

  if (best)
    best->info.addExplanation(diag);
}

//===----------------------------------------------------------------------===//
// ParameterInferenceState
//===----------------------------------------------------------------------===//

ParameterInferenceState::ParameterInferenceState(
    ASTDecl &declScope, const CallOperands &givenBindings,
    ArrayRef<Type> declaredParamTypes, PogListAttr declaredParamPogs,
    ArrayRef<TypedAttr> bindingsSoFar, ParameterInferenceDiagnostics &diags,
    bool allowImplicitConversions)
    : declScope(declScope), shared(declScope.getShared()),
      givenBindings(givenBindings), declaredParamTypes(declaredParamTypes),
      declaredParamPogs(declaredParamPogs), evaluator(declScope.getShared()),
      diags(diags), allowImplicitConversions(allowImplicitConversions) {

  // Maintain the invariant that 'evaluator' has the full size of the set of
  // parameters we're trying to infer.  Add null entries for any uninferred
  // values.
  size_t finalSize = declaredParamTypes.size();
  assert(bindingsSoFar.size() <= finalSize &&
         "too many params inferred already?");
  for (auto paramValue : bindingsSoFar)
    evaluator.appendIndexBinding(paramValue);
  while (evaluator.getNumIndexBindings() < finalSize)
    evaluator.appendIndexBinding(TypedAttr());
}

void ParameterInferenceState::dump() const {
  llvm::errs() << "ParameterInferenceState:\n";
  for (auto [idx, value] : llvm::enumerate(evaluator.getIndexBindings())) {
    llvm::errs() << "  *(0," << idx << ") = " << value << "\n";
  }
}

LogicalResult
ParameterInferenceState::matchFunctionTypes(FnTypeGeneratorType actual,
                                            FnTypeGeneratorType expected) {

  // Functions with different parameterization cannot be converted between each
  // other. If the types are equal but the passing conventions are different,
  // then the conversion is allowed.
  // TODO: Consider default parameter values and enable parameter inference to
  // reconcile differences.
  if (actual.getInputParamTypes() != expected.getInputParamTypes())
    return failure();

  // If the functions differ in return type conventions, check if the nominal
  // types are equal.
  bool actualMemResult = actual.hasMemoryOnlyResult();
  bool expectedMemResult = expected.hasMemoryOnlyResult();
  // TODO: We allow implicit conversions here.
  if (failed(
          matchTypes(actual.getUserResultType(), expected.getUserResultType())))
    return failure();

  ArrayRef<Type> actualArgTypes =
      actual.getArguments().drop_back(actualMemResult);
  ArrayRef<Type> expectedArgTypes =
      expected.getArguments().drop_back(expectedMemResult);

  auto actualEffects = actual.getFnEffects();
  auto expectedEffects = expected.getFnEffects();
  // If the actual function is not throwing, and the expected function is,
  // then we can infer the Error type to be Never.
  if (!actualEffects.isThrows() && expectedEffects.isThrows()) {
    // Match the expected error type to Never, but allow this to fail: it may
    // already be some concrete type like Error and that is ok.
    (void)matchTypes(NeverType::get(expected.getContext()),
                     expected.getUserThrownType());
    expectedArgTypes = expectedArgTypes.drop_back();
    actualEffects.setThrows(true);
  }

  if (actualEffects != expectedEffects)
    return failure();

  if (failed(matchParams(actual.getCaptureOrigins(),
                         expected.getCaptureOrigins()))) {
    return failure();
  }

  // Functions with an incompatible number of arguments cannot be converted
  // between each other. The number of arguments should be equal, unless the
  // expected function is variadic.
  // TODO: Consider default argument values.
  std::optional<size_t> expectedVariadicArgIndexOpt =
      expected.findPackVarArgIndex();
  if (expectedVariadicArgIndexOpt.has_value()) {
    size_t expectedVariadicArgIndex = expectedVariadicArgIndexOpt.value();
    if (actualArgTypes.size() < expectedVariadicArgIndex) {
      // Caller didn't supply enough arguments.
      return failure();
    }
  } else { // No variadic
    if (actualArgTypes.size() != expectedArgTypes.size()) {
      // Caller didn't supply the expected number of arguments.
      return failure();
    }
  }

  bool expectedHasVariadic = expectedVariadicArgIndexOpt.has_value();
  bool actualHasVariadic = actual.findPackVarArgIndex().has_value();
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
       actualArgIndex++) {
    auto actualConv = actual.getArgConvention(actualArgIndex);
    ArgConvention expectedConv = expected.getArgConvention(actualArgIndex);
    ASTType actualAstType = actualArgTypes[actualArgIndex];
    ASTType expectedAstType = expectedArgTypes[actualArgIndex];

    if (!checkConventionsConvertible(expectedConv, actualConv))
      return failure();

    Type expectedValueAstType =
        RefType::stripRefConvention(expectedAstType, expectedConv);
    Type actualValueAstType =
        RefType::stripRefConvention(actualAstType, actualConv);
    // Now check that the argument types line up.
    if (!succeeded(matchTypes(actualValueAstType, expectedValueAstType)))
      return failure();
  }

  // If the expected fn has a variadic arg, check all the actual args that will
  // go into it.
  if (collectIntoVariadic) {
    auto expectedVariadicArgIndex = expectedVariadicArgIndexOpt.value();

    ArgConvention expectedConv =
        expected.getArgConvention(expectedVariadicArgIndex);

    // Get the variadic pack's element trait.
    ASTType expectedArgVariadicPackType =
        expected.getIfVariadicPack(expectedVariadicArgIndex);
    RefPackType refPackType =
        expectedArgVariadicPackType.getVariadicPackInfo(shared);
    ASTType variadicElType = refPackType.getVariadicElementType();

    // This works because VariadicPack's element type is always a trait.
    auto expectedTraitType = cast<TraitType>(variadicElType.mlirType);

    TypedAttr variadic = refPackType.getVariadic();
    // As we do our checks, we'll also be calculating the actual kgen.variadic
    // parameter value.
    SmallVector<TypedAttr> elements;

    for (size_t actualArgIndex = numNormalArgs, e = actualArgTypes.size();
         actualArgIndex < e; ++actualArgIndex) {
      auto actualConv = actual.getArgConvention(actualArgIndex);
      ASTType actualAstType = actualArgTypes[actualArgIndex];

      if (!checkConventionsConvertible(expectedConv, actualConv))
        return failure();

      Type actualValueAstType =
          RefType::stripRefConvention(actualAstType, actualConv);

      // If the argument types line up, then we can skip the rest of this.
      if (succeeded(matchTypes(actualValueAstType, variadicElType.mlirType)))
        continue;

      // We can convert a more general `actual` function (that takes in a trait
      // argument) to a more specific `expected` function that takes in a struct
      // argument, as long as that struct conforms to that trait.
      // In other words, here we're handling function conversions with covariant
      // arguments (see TTSMFS).
      IREmitter emitter(declScope, EC_TypeParamValue);
      SyntheticNode synthNode(declScope.getLoc());
      // Now, check if the actual arg can be converted to the expected trait.
      PValue actualAstTypeAsVariadicElTrait =
          emitter.emitMetaTypeToTraitConversion(
              {CValue(actualValueAstType), synthNode}, expectedTraitType);
      if (!actualAstTypeAsVariadicElTrait) {
        return failure();
      }
      // And since we have it, let's use it to build up a kgen.variadic
      // parameter value.
      elements.push_back(actualAstTypeAsVariadicElTrait);
    }

    // Now assemble the kgen.variadic parameter value and match it against the
    // expected one.
    auto varType = VariadicType::get(variadicElType);
    auto variadicAttr = VariadicAttr::get(elements, varType);
    if (failed(matchParams(variadicAttr, variadic))) {
      return failure();
    }
  }

  // The function types are convertible.
  return success();
}

LogicalResult ParameterInferenceState::matchTypes(Type actualType,
                                                  Type expectedType) {
  // If the types trivially match then there is no inference to do.
  if (isEqualCanon(actualType, expectedType))
    return success();

  // If the expected type is a parameter ref, then we're binding the specified
  // type to an attribute parameter.
  if (auto expectedParamRef = dyn_cast<ParamType>(expectedType)) {
    // If this is a non-materializable type (like IntLiteral), infer it like its
    // materializable type (like Int), for example:
    //    fn example[T: AnyTrivialRegType](a: T): ...
    //    example(1) # T should be Int, not IntLiteral.
    // TODO: Why is this here?  Seems like a strange place to do this.
    if (ASTType nmTarget =
            ASTType(actualType).getNonmaterializableTarget(shared))
      actualType = nmTarget;

    return matchParams(PValue(actualType).get(), expectedParamRef.getParam());
  }

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
        return failure();

      // Fail if the parameter lists fundamentally mismatch.
      assert(actualDRT.getParamValues().size() ==
                 expectedDRT.getParamValues().size() &&
             "two instances of same struct must have same length param lists");

      // Match up the parameter bindings.
      for (auto [actual, expected] :
           llvm::zip(actualDRT.getParamValues(), expectedDRT.getParamValues()))
        if (failed(matchParams(actual, expected)))
          return failure();
      return success();
    }
  }

  // Handle when both are StructTypes.
  if (auto actualDRT = dyn_cast<StructType>(actualType)) {
    if (auto expectedDRT = dyn_cast<StructType>(expectedType)) {
      // Ignore if these are two fundamentally different symbols.
      if (actualDRT.getSymbol() != expectedDRT.getSymbol())
        return failure();

      // Fail if the parameter lists fundamentally mismatch.
      assert(actualDRT.getParamValues().size() ==
                 expectedDRT.getParamValues().size() &&
             "two instances of same struct must have same length param lists");

      // Match up the parameter bindings.
      for (auto [actual, expected] :
           llvm::zip(actualDRT.getParamValues(), expectedDRT.getParamValues()))
        if (failed(matchParams(actual, expected)))
          return failure();
      return success();
    }
  }

  // Handle various common POP types for convenience, starting with SIMDType.
  if (auto actual = dyn_cast<POP::SIMDType>(actualType))
    if (auto expected = dyn_cast<POP::SIMDType>(expectedType)) {
      if (failed(matchParams(actual.getSize(), expected.getSize())))
        return failure();
      return matchParams(actual.getDType(), expected.getDType());
    }

  // POP::ArrayType.
  if (auto actual = dyn_cast<POP::ArrayType>(actualType))
    if (auto expected = dyn_cast<POP::ArrayType>(expectedType)) {
      if (failed(matchParams(actual.getSize(), expected.getSize())))
        return failure();
      return matchTypes(actual.getElementType(), expected.getElementType());
    }

  // Handle RefType.
  if (auto actual = dyn_cast<RefType>(actualType))
    if (auto expected = dyn_cast<RefType>(expectedType)) {
      if (failed(
              matchTypes(actual.getElementType(), expected.getElementType())))
        return failure();
      if (failed(
              matchSingleEltStruct(actual.getOrigin(), expected.getOrigin())))
        return failure();

      return matchSingleEltStruct(actual.getAddressSpace(),
                                  expected.getAddressSpace());
    }

  // Handle OriginType.
  if (auto actual = dyn_cast<OriginType>(actualType))
    if (auto expected = dyn_cast<OriginType>(expectedType)) {
      // Try to match up the types so we infer parameters properly.
      if (succeeded(
              matchSingleEltStruct(actual.isMutable(), expected.isMutable())))
        return success();
      // If that fails, check compatibility, actualType might be mutable=true,
      // and expected might be mutable=false, and this is fine.
      return success(
          IREmitter::canZeroCostConvert(actualType, expectedType, shared));
    }

  // Handle PointerType.
  if (auto actual = dyn_cast<PointerType>(actualType))
    if (auto expected = dyn_cast<PointerType>(expectedType)) {
      if (failed(
              matchTypes(actual.getElementType(), expected.getElementType())))
        return failure();
      return matchSingleEltStruct(actual.getAddressSpace(),
                                  expected.getAddressSpace());
    }

  // Handle VariadicType.
  if (auto actual = dyn_cast<VariadicType>(actualType))
    if (auto expected = dyn_cast<VariadicType>(expectedType))
      return matchTypes(actual.getElementType(), expected.getElementType());

  // Handle RefPackType.
  if (auto actual = dyn_cast<RefPackType>(actualType))
    if (auto expected = dyn_cast<RefPackType>(expectedType)) {
      if (failed(matchParams(actual.getVariadic(), expected.getVariadic())) ||
          failed(
              matchSingleEltStruct(actual.getOrigin(), expected.getOrigin())))
        return failure();
      return matchParams(actual.getAddressSpace(), expected.getAddressSpace());
    }

  // Handle FuncTypeGeneratorType
  if (auto actual = dyn_cast<FnTypeGeneratorType>(actualType)) {
    if (auto expected = dyn_cast<FnTypeGeneratorType>(expectedType)) {
      // See paramIndexRefDepth's comments for what this increment is for.
      // The increment was solely for the matchFunctionTypes call, so undo it.
      llvm::SaveAndRestore depth(paramIndexRefDepth, paramIndexRefDepth + 1);
      return matchFunctionTypes(actual, expected);
    }
  }

  // Handle GeneratorType
  if (auto actual = dyn_cast<GeneratorType>(actualType)) {
    if (auto expected = dyn_cast<GeneratorType>(expectedType)) {
      // See paramIndexRefDepth's comments for what this increment is for.
      llvm::SaveAndRestore depth(paramIndexRefDepth, paramIndexRefDepth + 1);

      if (isa<FnType>(actual.getBody()) || isa<FnType>(expected.getBody())) {
        // Matching two FnTypeGeneratorType should have been handled above
        assert(!isa<FnType>(actual.getBody()) ||
               !isa<FnType>(expected.getBody()));
        return failure();
      }
      // This a simple type generator, match the input parameter types and body
      // type.
      ArrayRef<Type> actInputs = actual.getInputParamTypes();
      ArrayRef<Type> expInputs = expected.getInputParamTypes();
      if (actInputs.size() != expInputs.size())
        return failure();
      for (auto [ai, ei] : llvm::zip_equal(actInputs, expInputs))
        if (failed(matchTypes(ai, ei)))
          return failure();
      return matchTypes(actual.getBody(), expected.getBody());
    }
  }

  // If the actual type is a reference to a parameter, it might be a local
  // parameter within a function.  The type checker will resolve this using the
  // metatype of the parameter, something like AnyStruct[someType].  This tells
  // us the actual type of the parameter.
  // TODO: Why isn't this a general solution?
  if (auto actualParamRef = dyn_cast<ParamType>(actualType)) {
    if (auto actualMetaType = ASTType(actualType).getMetaType()) {
      if (auto structMeta = sugarDynCast<StructMetaType>(actualMetaType))
        return matchTypes(structMeta.getType(), expectedType);
      if (auto traitMeta = sugarDynCast<AnyTraitType>(actualMetaType))
        return matchTypes(traitMeta.getTraitType(), expectedType);
    }
  }

  // Handle meta type upcasting.
  FailureOr<bool> typeUpCastable = IREmitter::canMetaTypeUpCastTo(
      shared, declScope.getLoc(), actualType, expectedType);
  if (succeeded(typeUpCastable) && typeUpCastable.value())
    return success();

  // TODO: We're not handling a lot of important things, e.g., implicit
  // conversions that cause us to see i1->Bool and similar things here, etc. as
  // such, we can't treat conversion errors for unknown things as failures.
  LLVM_DEBUG(llvm::errs() << "CANNOT INFER UNKNOWN TYPES:\n"; actualType.dump();
             expectedType.dump(); llvm::errs() << "\n");
  return failure();
}

LogicalResult ParameterInferenceState::matchParams(TypedAttr actualAttr,
                                                   TypedAttr expectedAttr) {
  // If the attrs trivial match then we're done and there is no inference to do.
  if (isEqualCanon(actualAttr, expectedAttr))
    return success();

  // Look through type upcasts to the more derived type.
  actualAttr = UpcastAttr::strip(actualAttr);
  expectedAttr = UpcastAttr::strip(expectedAttr);

  auto getTargetMetaTypeForTypeValue = [](Type targetMetaTp) -> Type {
    if (auto vaTp = sugarDynCast<VariadicType>(targetMetaTp))
      targetMetaTp = vaTp.getElementType();

    if (auto paramTp = sugarDynCast<ParamType>(targetMetaTp)) {
      // If the expected type is parameterized, we strip the meta type.
      // E.g.,
      //
      // fn foo[
      //  elt_trait : type_of(AnyType & Foo),
      //  *elt : elt_trait
      // ] : ...
      //
      // foo() # we infer elt : !kgen.variadic<!AnyType & !Foo>
      auto metaType = paramTp.getParam().getType();
      // TODO: should we make AnyTraitType a `MetaType`?.
      if (auto anyTrait = sugarDynCast<AnyTraitType>(metaType))
        return anyTrait.getTraitType();

      return sugarCast<MetaType>(metaType).getType();
    };
    return targetMetaTp;
  };

  if (failed(matchTypes(actualAttr.getType(), expectedAttr.getType()))) {
    // If this is a type expression, try align the type (if possible) before
    // concluding type inconvertibility. This turns things like:
    // #kgen.type<!Int> : !lit.trait<!AnyType> to
    // #kgen.type<!Int> : !lit.trait<!Copyable>
    // Then we can correctly check type value convertibility between
    // #kgen.type<!Int> : !lit.trait<!AnyType> and
    // #param.ref<....> : !lit.trait<!Copyable>
    bool fixableByUpCast = false;
    if (LIT::isTypeExpr(actualAttr) || LIT::isVariadicOfTypeExpr(actualAttr)) {
      auto targetMT = getTargetMetaTypeForTypeValue(expectedAttr.getType());
      ArrayRef<TypedAttr> toCheck(actualAttr);
      if (auto va = sugarDynCast<VariadicAttr>(actualAttr))
        toCheck = va.getValues();

      fixableByUpCast = llvm::all_of(toCheck, [&](TypedAttr typeExpr) {
        assert(LIT::isTypeExpr(typeExpr));
        // Try get the tightest possible metatype bound.
        Type tightestBound = ASTType(typeExpr).getMetaType();
        if (!tightestBound) {
          // `struct __MLIRType` is the corner case here :(.
          tightestBound = typeExpr.getType();
        }
        FailureOr<bool> upCastable = IREmitter::canMetaTypeUpCastTo(
            shared, declScope.getLoc(), tightestBound, targetMT);
        return succeeded(upCastable) && upCastable.value();
      });

      if (fixableByUpCast) {
        SmallVector<TypedAttr> casted =
            llvm::map_to_vector(toCheck, [targetMT](TypedAttr toMap) {
              return TypeParamAttr::get(ASTType(toMap), targetMT);
            });

        if (auto va = sugarDynCast<VariadicAttr>(actualAttr))
          actualAttr = VariadicAttr::get(casted, VariadicType::get(targetMT));
        else
          actualAttr = casted.front();

        LogicalResult matchFixed =
            matchTypes(actualAttr.getType(), expectedAttr.getType());
        assert(succeeded(matchFixed));
        (void)matchFixed;
      }
    }

    if (!fixableByUpCast) {
      if (auto ire = dyn_cast<ParamIndexRefAttr>(expectedAttr)) {
        addFailure(ire.getIndex(),
                   InferenceFailure::TypeConflictFailure{
                       evaluator.getReboundType(expectedAttr.getType()),
                       actualAttr.getType()});
      }
      return failure();
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
         llvm::zip(actualOp.getOperands(), expectedOp.getOperands()))
      if (failed(matchParams(a, b)))
        return failure();
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
      if (actualGetWitness.getTraitName() !=
              expectedGetWitness.getTraitName() ||
          actualGetWitness.getWitnessName() !=
              expectedGetWitness.getWitnessName())
        return failure();
      return matchParams(actualGetWitness.getTypeValue(),
                         expectedGetWitness.getTypeValue());
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
      expectedType = evaluator.getReboundType(expectedType);

      // If the types don't agree, attempt an implicit conversion between the
      // actual value and the expected type.
      if (!isEqualCanon(actualAttr.getType(), expectedType)) {
        // We can only see subtypes in type values, all other values must be
        // aligned perfectly.
        IREmitter emitter(declScope, EC_TypeParamValue);
        SyntheticNode node(declScope.getLoc());
        ASTExprAnd<CValue> toConvert = {actualAttr, node};
        // We should have ensured type convertibility
        assert(IREmitter::canImplicitlyConvertToType(toConvert, expectedType,
                                                     emitter.getDeclScope()) &&
               "Unconvertible parameter should have been detected");
        actualAttr =
            emitter.emitPValue(toConvert, EC_TypeParamValue, expectedType);
        assert(actualAttr && "Already checked implicit convertibility");
        assert(isEqualCanon(actualAttr.getType(), expectedType));
      }

      size_t parameterIndex = ire.getIndex();
      TypedAttr inferredValue = evaluator.getIndexBindings()[parameterIndex];

      // If this is a new parameter we've inferred, huzzah, remember it.
      if (!inferredValue) {
        evaluator.overwriteIndexBinding(parameterIndex, actualAttr);
        evaluator.clearCache();
        return success();
      }

      // If we saw this parameter before, make sure it is compatible with
      // (or more specific than) the other values we've inferred.
      if (failed(matchParams(inferredValue, actualAttr))) {
        addFailure(parameterIndex, InferenceFailure::ValueConflictFailure{
                                       inferredValue, actualAttr});
        return failure();
      }
      return success();
    }
    // If this is some parameter other than the one we're inferring, assume it
    // will work out.
    return success();
  }

  if (auto actualVar = dyn_cast<VariadicAttr>(actualAttr)) {
    if (auto expectedVar = dyn_cast<VariadicAttr>(expectedAttr)) {
      if (actualVar.getValues().size() != expectedVar.getValues().size())
        return failure();
      for (auto [act, exp] :
           llvm::zip(actualVar.getValues(), expectedVar.getValues())) {
        if (failed(matchParams(act, exp)))
          return failure();
      }
      return success();
    }
  }

  if (auto actualSym = dyn_cast<SymbolConstantAttr>(actualAttr)) {
    if (auto expectedSym = dyn_cast<SymbolConstantAttr>(expectedAttr)) {
      if (actualSym.getSymbol() != expectedSym.getSymbol() ||
          actualSym.getParamValues().size() !=
              expectedSym.getParamValues().size())
        return failure();
      for (auto [act, exp] : llvm::zip(actualSym.getParamValues(),
                                       expectedSym.getParamValues())) {
        if (failed(matchParams(act, exp)))
          return failure();
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
      if (actualExtract.getField() != expectedExtract.getField())
        return failure();
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
        return failure();
      for (auto [actual, expected] : llvm::zip_equal(
               actualSet.getOperands(), expectedSet.getOperands())) {
        if (failed(matchParams(actual, expected)))
          return failure();
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
          if (failed(matchParams(std::get<1>(act), std::get<1>(exp))))
            return failure();
        }
        return success();
      }
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

  LLVM_DEBUG(llvm::errs() << "CANNOT INFER UNKNOWN ATTRS:\n"; actualAttr.dump();
             expectedAttr.dump(); llvm::errs() << "\n");
  return failure();
}

// Special Hack (tm) for matching mlir values for pointers and references to
// single-element structs.
//
// Things like pointers are defined like this:
//   struct YourPointer[type: ..., address_space: AddressSpace]:
//     fn __init__(out self, ref [_] address: type):
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
LogicalResult
ParameterInferenceState::matchSingleEltStruct(TypedAttr actualOrig,
                                              TypedAttr expectedOrig) {
  auto actual = ParamOperatorAttr::stripRebind(actualOrig);
  auto expected = ParamOperatorAttr::stripRebind(expectedOrig);

  if (actual == expected)
    return success();

  // If it is an extract from a known struct, then we know there is one field in
  // the struct - we can form a StructAttr around our actual value and recurse.
  if (auto expExtract = sugarDynCast<LIT::StructExtractAttr>(expected)) {
    // If these are two lined up extracts, look through them.
    if (auto actExtract = sugarDynCast<LIT::StructExtractAttr>(actual)) {
      if (expExtract.getField() != actExtract.getField())
        return failure();
      return matchSingleEltStruct(actExtract.getStructValue(),
                                  expExtract.getStructValue());
    }

    // If the types mismatch, it might be due to an origin mutability
    // conversion, which we can handle.
    if (actual.getType() != expected.getType()) {
      // See if we can infer anything from the types, this allows us to infer
      // 'is_mut' parameter from "origin<1>" and "origin<is_mut>".
      if (failed(matchTypes(actual.getType(), expected.getType())))
        return failure();
    }

    // Ok, we have a struct that seems like it could line up.  See if we can
    // implicitly construct this from a value of this type.  If so, then we
    // assume it is a value-wise initializer that we can infer from.
    //
    // TODO: We could make this more strict by using a keyword argument for the
    // argument value instead of an implicit conv.
    auto expStruct = expExtract.getStructValue();
    // Figure out if the struct is something we can handle.
    auto expDRT = sugarCast<StructType>(expStruct.getType());

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
      FailureOr<PValue> pValue = OverloadSet::canConstructType(
          nonParamDRT, CallOperands({{actual, curArgExpr}}), curArgExpr,
          declScope, /*isImplicitConversion=*/true);
      if (failed(pValue) || !pValue.value())
        return failure();

      // If we succeeded, figure out what the concrete type being inferred would
      // be with any parameters bound.
      auto initSig = sugarCast<FnTypeGeneratorType>(pValue.value().getType());
      // The constructed type is the result of the initializer.
      assert(initSig.getNumArguments() != 0);
      expDRT = sugarCast<StructType>(initSig.getUserResultType());

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
LogicalResult ParameterInferenceState::inferSelfFromInitResult(
    FnTypeGeneratorType signature) {
  ASTType returnedType =
      evaluator.getReboundType(signature.getUserResultType());

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
    if (selfParam) {
      if (failed(matchParams(selfParam, retParam))) {
        addFailure(idx,
                   InferenceFailure::ValueConflictFailure{selfParam, retParam});
        return failure();
      }
    } else if (!paramFinder.hasReferences(retParam)) {
      // Otherwise if the the returned parameter has no unbound parameter
      // references then we infer the self parameter from it. This infers X=42:
      //   struct X[A: Int]:
      //     fn __init__(out self: X[42]):
      auto selfType =
          evaluator.getReboundType(signature.getInputParamTypes()[idx]);
      auto selfParam = ParamIndexRefAttr::get(/*depth*/ 0, idx, selfType);
      if (failed(matchParams(retParam, selfParam))) {
        addFailure(idx,
                   InferenceFailure::ValueConflictFailure{retParam, selfParam});
        return failure();
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
      inferredType, std::move(operands), operand.expr, declScope,
      /*isImplicitConversion=*/false);
  if (failed(initFn) || !initFn.value())
    return {};
  return sugarCast<FnTypeGeneratorType>(initFn.value().getType())
      .getUserResultType();
}

/// Infer parameters from an operand being passed into this function. This is
/// only called on the top level function operands being matched up, not
/// anything in recursive functiontype positions.
LogicalResult
ParameterInferenceState::inferOneOperand(ASTExprAnd<AnyValue> operand,
                                         ASTType expectedType,
                                         ArgConvention expectedConvention) {
  // Early return if this operand will not help with inferring parameters. This
  // avoids unnecessary checks & dealing with errors unrelated to parameter
  // inference here. The only operands that can contribute to param inference
  // are either those whose expected types contain param references.
  if (!paramFinder.hasReferences(expectedType.mlirType))
    return success();

  AnyValue value = operand.ir;
  curArgExpr = operand.expr;

  auto resolveOperandCValue = [&](Type expectedTypeOfOperand) -> CValue {
    if (auto argVal = value.getIfCValue())
      return argVal;

    // Handle collection literals.
    if (auto init = value.getIfInitializer()) {
      ASTType inferredType = evaluator.getReboundType(expectedTypeOfOperand);
      // If we have a type like List[$0] replace it with List[?] so we can
      // infer the unbound parameter.
      inferredType = inferredType.getWithUnknownParametersReplaced(shared);
      Type initType =
          inferInitializerType(declScope, &(*init), operand, inferredType);
      // If we could not infer the type from the inferred type (in the case
      // where the inferred type is a parameter with trait metatype and no
      // initialier, try the default type.)
      if (!initType)
        initType =
            inferInitializerType(declScope, &(*init), operand,
                                 init->getDefaultType(declScope.getShared()));
      return PValue(initType);
    }

    OverloadSetUValue orValue = value.getIfOverloadSet();
    if (!orValue)
      return {};

    // Try to refine the OverloadSetUValue into a PValue.
    CValue argVal = orValue->getDirectSymbol(expectedType, declScope);
    if (!argVal)
      return {};
    // If we have a reference to an overloaded method like foo(a.method),
    // then we can't resolve it.
    // TODO(partial application => closures): Given we just resolved argVal,
    // we could form the "a.method" expression with a closure.
    if (orValue->baseValue) // Cannot merge base value.
      return {};
    return argVal;
  };

  // We'll bind the next provided value.
  switch (expectedConvention) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::Mut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError: {
    // The actual value must be an lvalue if callee takes things by-ref, but we
    // don't want to force this - we want parameter inference to infer argument
    // types etc instead of producing a "failed to infer" message.
    CValue argVal = value.getIfCValue();
    if (!argVal)
      return failure();

    // By-ref argument types must exactly match, no conversions are allowed.
    return matchTypes(argVal.getRValueType(),
                      expectedType.getReferenceElementType());
  }

  case ArgConvention::Ref:
  case ArgConvention::MutRef: {
    auto expectedRef = sugarCast<RefType>(expectedType);
    // Infer the origin and address space before inferring the element type.
    CValue argVal = resolveOperandCValue(expectedRef.getElementType());
    if (!argVal)
      return failure();

    // If we are binding the reference to a value in memory directly, check for
    // reference compatibility directly.
    if (argVal.isMValue()) {
      RefType valueRefType = value.getMValueType();
      // If the IRValue type is MBValue or MRValue then we need infer an
      // immutable ref, to match behavior where we don't allow passing an
      // MBValue or MRValue as 'mut'.
      if (!argVal.getIfMLValue() && !argVal.getIfMBPValue() &&
          !valueRefType.isMutableKnown(false))
        valueRefType = valueRefType.getWithMutability(false);

      return matchTypes(valueRefType, expectedType);
    }

    // Otherwise, we'll need to drop this value into a temporary.  For now, we
    // infer it as AnyOrigin.  We bind the origin directly and then handle
    // it like any other argument because we can support implicit conversions.
    RefType valueRefType =
        RefType::getAnyOrigin(argVal.getRValueType(), /*isMut=*/false);

    (void)matchSingleEltStruct(valueRefType.getOrigin(),
                               expectedRef.getOrigin());
    (void)matchSingleEltStruct(valueRefType.getAddressSpace(),
                               expectedRef.getAddressSpace());

    // Handle the element type compatibility check below to allow implicit
    // conversions etc.
    expectedType = expectedType.getReferenceElementType();
    break;
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

  // Check to see if the expected type has an initializer with the
  // specified operands.  Remove any parameters from the expected type
  // since those are what we're inferring from the arguments.  The result
  // 'actualType' will have those newly inferred parameters.
  if (auto initValue = operand.ir.getIfInitializer()) {
    ASTType inferredType = evaluator.getReboundType(expectedType);
    // If we have a type like List[$0] replace it with List[?] so we can
    // infer the unbound parameter.
    inferredType = inferredType.getWithUnknownParametersReplaced(shared);
    Type initType =
        inferInitializerType(declScope, &(*initValue), operand, inferredType);
    // If the literal cannot bind to the inferred type, try binding it to the
    // default literal type and matching the inferred type against that.
    if (!initType)
      initType = inferInitializerType(declScope, &(*initValue), operand,
                                      initValue->getDefaultType(shared));

    // If there were declaration errors, assume success to not raise
    // spurious errors due to not resolving to those erroneous
    // declarations.
    if (!initType)
      return failure();
    // If we found one, we recursively call inferOneOperand (but with implicit
    // conversions disabled of course) to resolve our value as the init
    // methods argument.  This allows us to infer parameters from it.
    return matchTypes(initType, expectedType);
  }

  // Okay, we got a normal value argument convention and stripped off any
  // ArgConvention-related !lit.ref from the expected type.  See if we can
  // resolve the argument to a CValue.
  CValue argVal = resolveOperandCValue(expectedType);
  if (!argVal)
    return failure();

  // If the argument types exactly match, then they are good.
  ASTType argType = argVal.getRValueType();
  if (argType.isEqualCanon(expectedType))
    return success();

  // We're speculatively trying different options.  If we have errors on one
  // path we need to roll them back.
  auto savedDiags = diags.saveDiags();

  // See if the types match with inference, if not, remember why.
  if (succeeded(matchTypes(argType, expectedType)))
    return success();

  // Before we check with the implicit conversions, save any diagnostics
  // accumulated without it.  If both fail, we default to the non-implicit
  // conversion diagnostics.
  auto noImplicitConversionDiags = diags.saveDiags();

  // Go back to diagnostics before we did the thing that failed.
  diags.resetDiags(std::move(savedDiags));
  savedDiags = diags.saveDiags();

  // Zero cost conversions don't count as implicit conversions. We attempt this
  // after trying to match the types to try to infer values first.
  if (IREmitter::canZeroCostConvert(argType, expectedType, shared))
    return success();

  // Handle values of nonmaterializable types.  These freely convert to their
  // nonmaterializableTarget type even when implicit conversions are disabled,
  // so we can accept this argument if that converted type is compatible with
  // our expected type.
  if (auto nonmaterializableTarget =
          argType.getNonmaterializableTarget(shared)) {

    // Infer the parameters of this overload candidate against the computed
    // result type of the initializer.
    if (succeeded(matchTypes(nonmaterializableTarget, expectedType)))
      return success();

    // If that didn't work out, keep going, but with the original
    // diagnostics.
    diags.resetDiags(std::move(savedDiags));
    savedDiags = diags.saveDiags();
  }

  // If implicit conversions are enabled and the target type is known, then
  // we can check to see if any of the constructors for the result type can
  // work.

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

  ASTType knownExpectedType = evaluator.getReboundType(expectedType);
  ASTDecl *expectedDecl = knownExpectedType.getDecl(shared);
  if (!allowImplicitConversions || !expectedDecl) {
    diags.resetDiags(std::move(noImplicitConversionDiags));
    return failure();
  }

  // Determine if we can construct the requested type given the existing value
  // we have.  If so, get the type inferred signature of the init method that
  // would make it work.
  IREmitter emitter(declScope, ExprContext::EC_CallArgValue);

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
      knownExpectedType.getWithUnknownParametersReplaced(emitter.shared);
  FailureOr<PValue> pValue = OverloadSet::canConstructType(
      nonParamType, CallOperands({{argVal, curArgExpr}}), curArgExpr,
      emitter.getDeclScope(), /*isImplicitConversion=*/true);
  if (llvm::failed(pValue))
    return success(); // Issue already diagnosed.

  if (!pValue.value()) {
    // If we had a fully formed type that we were inferring into, then this is
    // a failure.
    if (!noImplicitConversionDiags.empty() ||
        nonParamType.mlirType == expectedType.mlirType) {
      diags.resetDiags(std::move(noImplicitConversionDiags));
      return failure();
    }

    // Otherwise, it could be because it is using a later parameter that we
    // haven't bound or that is defaulting.  We aren't currently inferring the
    // entire set of parameters all at once, so we just treat that as "not a
    // failure" and assume it will work out.
    return success();
  }

  // If we found one, we recursively call inferOneOperand (but with implicit
  // conversions disabled of course) to resolve our value as the init
  // methods argument.  This allows us to infer parameters from it.
  auto initSig = sugarCast<FnTypeGeneratorType>(pValue.value().getType());
  // We expect the initializer to return the constructed type.
  // Infer the parameters of this overload candidate against the computed
  // result type of the initializer.
  auto result = matchTypes(initSig.getUserResultType(), knownExpectedType);

  // If the implicit conversion worked then we're good.
  if (succeeded(result))
    return success();

  // Otherwise restore the diags from the non-implicit conversion path,
  // they'll be less confusing.
  diags.resetDiags(std::move(noImplicitConversionDiags));
  return failure();
}

void ParameterInferenceState::inferOneParam(ASTExprAnd<AnyValue> binding,
                                            Type expectedType) {
  (void)inferOneOperand(binding, expectedType, ArgConvention::ReadReg);
}

/// Given an incomplete parameter binding set for a parameter list, try to
/// infer the value of the next parameter. We only do this if there are any
/// inferred parameters present.
void ParameterInferenceState::inferFromParamList(bool hasArguments) {
  // If the parameter list has any inferred parameters, then we have to infer
  // against the provided binding list, since we might infer parameters from
  // other parameters. Otherwise, just exit early.
  if (declaredParamTypes.empty() ||
      (!declaredParamPogs.hasInferredParams() &&
       !declaredParamPogs.isPosVarArg(declaredParamTypes.size() - 1)))
    return;

  // Partially specialize the pogs so any default values are specialized to
  // include any already-inferred values.
  SmallVector<Type> types;
  for (auto type : declaredParamTypes)
    types.push_back(evaluator.getReboundType(type));
  declaredParamPogs =
      cast<PogListAttr>(evaluator.getReboundAttribute(declaredParamPogs));

  size_t posIdx = 0, numParams = givenBindings.size();
  DefaultValueHandler defaultHandler(declaredParamPogs);
  for (auto [idx, pog] : llvm::enumerate(declaredParamPogs.getPogs())) {
    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = types[idx];

    // Skip over any provided keyword parameters when matching things up, we
    // handle them separately below.
    while (posIdx < numParams && givenBindings[posIdx].keyword)
      ++posIdx;

    // If we have a varargs parameters, then it will eat the rest of the
    // parameters, but we have to check each of them.
    if (declaredParamPogs.isPosVarArg(idx)) {
      auto expectedVariadic = sugarCast<VariadicType>(expectedType);
      Type varArgsEltType = expectedVariadic.getElementType();
      while (posIdx != numParams) {
        if (!givenBindings[posIdx].keyword)
          inferOneParam(givenBindings[posIdx], varArgsEltType);
        ++posIdx;
      }
      continue;
    }

    // If we have a non-kw param value, it binds to this parameter if it accepts
    // it.
    if (posIdx < numParams && (pog.getPassingKind() == PassingKind::PosOrKw ||
                               pog.getPassingKind() == PassingKind::PosOnly)) {
      inferOneParam(givenBindings[posIdx], expectedType);
      ++posIdx;
      continue;
    }

    // If we're out of positional bindings, or this works with a keyword, try
    // looking for a provided keyword parameter binding.
    if ((pog.getPassingKind() != PassingKind::PosOnly &&
         pog.getPassingKind() != PassingKind::Implicit)) {
      if (const OperandValue *param =
              givenBindings.findKwArg(declaredParamPogs.getName(idx))) {
        inferOneParam(*param, expectedType);
        continue;
      }
    }

    // If not, and this parameter has a default value, then just skip it. We
    // can't infer from default values.
    if (defaultHandler.getDefault(idx))
      continue;

    // Otherwise, this is a missing parameter. Just skip it.
    // TODO: Seems like we should reject??
  }

  // If we had a variadic parameter that is unspecified, and no arguments to
  // infer it from, it must be because of an empty variadic list.
  if (!hasArguments) {
    for (size_t i = 0, e = declaredParamTypes.size(); i != e; ++i) {
      if (declaredParamPogs.isPosVarArg(i) &&
          !evaluator.getIndexBindings()[i]) {
        // FIXME: variadic element type may be dependent on an earlier inferred
        // parameter, we should get the rebound type.
        auto type = types[i];
        auto empty = VariadicAttr::get({}, sugarCast<VariadicType>(type));
        evaluator.overwriteIndexBinding(i, empty);
        evaluator.clearCache();
      }
    }
  }
}

LogicalResult ParameterInferenceState::inferForCall(
    FnTypeGeneratorType signature, const CallOperands &operands,
    const OperandValueList &variadicKwOperands, bool returnsSelf) {
  // First try to infer parameters from the already provided bindings.
  inferFromParamList(/*hasArguments*/ true);

  {
    // Substitute the already inferred parameters into the signature, without
    // removing their parameter decls.
    //
    // For example, if given this signature,
    //     fn [N: Int, S: SIMD[DType.int8, N]]() -> ()
    // and we know that N=1, it should become:
    //     fn [N: Int, S: SIMD[DType.int8, 1]]() -> ()
    // Note the N became a 1 right here ---^
    //
    // We can't use `getSpecializedGenerator` for this as it removes the
    // parameter-decls. For example, giving `getSpecializedGenerator` this:
    //     fn [N: Int, S: SIMD[DType.int8, N]]() -> ()
    // and N=1, it would produce this signature:
    //     fn [S: SIMD[DType.int8, 1]]() -> ()
    // which we don't want, because the rest of the logic expects
    // inputParamTypes to be intact.
    //
    // All this must be done by slicing out the nested types & attrs from the
    // generator type so that the depths of index references are correct.
    FnType bodyType = signature.getBody();
    PogListAttr metadata = signature.getMetadata();
    SmallVector<Type> paramTypes;
    for (auto ty : signature.getInputParamTypes())
      paramTypes.push_back(evaluator.getReboundType(ty));

    bodyType = sugarCast<FnType>(evaluator.getReboundType(bodyType));
    metadata = cast<PogListAttr>(evaluator.getReboundAttribute(metadata));
    signature = sugarCast<FnTypeGeneratorType>(
        GeneratorType::get(paramTypes, bodyType, metadata));
  }

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t posOperandIdx = 0;
  size_t numOperands = operands.size();
  DefaultValueHandler defaultHandler(signature.getArgListAttrs());
  for (auto [expectedArgIdx, expectedConvention] :
       llvm::enumerate(signature.getArgConventions())) {

    // There is no provided operand for a by-ref result.
    if (isResultSlot(expectedConvention))
      continue;

    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = signature.getArgument(expectedArgIdx);
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
        if (failed(
                inferOneOperand(operand, refValType, ArgConvention::OwnedMem)))
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
                operand, varArgsEltType,
                signature.getPosVarArgConvention(expectedArgIdx))))
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
      RefPackType packType = variadicPackType.getVariadicPackInfo(shared);

      // Figure out that the element type of the list is, e.g. AnyType or
      // Stringable.
      Type elementType = packType.getVariadicElementType();

      SmallVector<TypedAttr> types;
      IREmitter emitter(declScope, EC_TypeParamValue);
      while (posOperandIdx != numOperands) {
        const auto &operand = operands[posOperandIdx++];
        if (operand.keyword) // Ignore keyword operands.
          continue;
        curArgExpr = operand.expr;

        ASTType toPush = operand.ir.getRValueTypeIfResolvable();
        if (!toPush) {
          shared.emitWarning(operand.expr->getLoc(),
                             "could not infer parameter type for this value, "
                             "because it is not concrete");
          return failure();
        }

        // Infer nonmaterializable types as their materialization target.
        if (ASTType nmTarget = toPush.getNonmaterializableTarget(shared))
          toPush = nmTarget;
        Type metatype = toPush.getMetaType();
        TypedAttr actualAttr = TypeParamAttr::get(
            toPush, metatype ? metatype : TypeType::get(shared.getContext()));
        SyntheticNode node(operand.expr->getLoc());
        if (!IREmitter::canImplicitlyConvertToType(
                {actualAttr, node}, elementType, emitter.getDeclScope())) {

          // If that didn't work, then we fail due to the type mismatch.  If the
          // variadic type is due to a parameter mismatch, record it.
          if (auto ire =
                  sugarDynCast<ParamIndexRefAttr>(packType.getVariadic());
              ire && ire.getDepth() == paramIndexRefDepth) {
            // Otherwise, we failed to infer the parameter. Record this failure.
            addFailure(ire.getIndex(), InferenceFailure::TypeConflictFailure{
                                           elementType, actualAttr.getType()});
          }
          return failure();
        }

        // Perform a conversion (e.g. from a concrete to trait type) as needed.
        PValue result = emitter.emitPValue({actualAttr, node},
                                           EC_TypeParamValue, elementType);
        if (!result)
          return failure();
        types.push_back(result);
      }

      // Infer the value of type list from the types we have.
      auto variadicType =
          sugarCast<VariadicType>(packType.getVariadic().getType());
      if (failed(matchParams(VariadicAttr::get(types, variadicType),
                             packType.getVariadic())))
        return failure();
      continue;
    }

    // Check for any more positional operands.
    while (posOperandIdx != numOperands && operands[posOperandIdx].keyword)
      ++posOperandIdx;

    // Handle positional arguments.
    if (posOperandIdx < numOperands) {
      if (failed(inferOneOperand(operands[posOperandIdx++], expectedType,
                                 expectedConvention)))
        return failure();
      continue;
    }

    // Handle case when there are no more provided positional operands.
    // Check if a keyword operand was provided for this argument
    if (const OperandValue *kwOperandOr =
            operands.findKwArg(signature.getArgName(expectedArgIdx))) {
      if (failed(
              inferOneOperand(*kwOperandOr, expectedType, expectedConvention)))
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
  if (returnsSelf) {
    if (failed(inferSelfFromInitResult(signature)))
      return failure();
  }

  // See if we can fulfill any missing parameters with default values for their
  // type.
  for (size_t i = 0, e = declaredParamTypes.size(); i != e; ++i) {
    if (!evaluator.getIndexBindings()[i] &&
        signature.getParamListAttrs().isPosVarArg(i)) {
      // FIXME: This isn't rewriting the variadic list for dependent types.
      auto type = declaredParamTypes[i];
      auto empty = VariadicAttr::get({}, sugarCast<VariadicType>(type));
      evaluator.overwriteIndexBinding(i, empty);
      evaluator.clearCache();
      break;
    }
  }
  // We succeed iff we inferred a value for this parameter.
  return success();
}

/// Given an incomplete parameter binding set, try to infer parameters on Self
/// of a method from the first argument.
LogicalResult
ParameterInferenceState::inferCTADParams(FnTypeGeneratorType signature,
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
  ASTDecl *decl = declaredSelfType.getDecl(shared);
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
  return inferOneOperand(operands[0], selfType, selfConvention);
}
