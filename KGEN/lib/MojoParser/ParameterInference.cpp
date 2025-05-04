//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParameterInference.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/SharedState.h"
#include "ParamBindings.h"

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"

#include "MojoUtils.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

#define DEBUG_TYPE "LITEXPRCALLS"

// DO NOT SUBMIT, thoughts on where this should go?
//     also we should probably return something other than a bool.
extern bool checkConventionsConvertible(ArgConvention expectedConv,
                                        ArgConvention actualConv);

//===----------------------------------------------------------------------===//
// InferenceFailure
//===----------------------------------------------------------------------===//

void InferenceFailure::emitSpecificNote(
    function_ref<InflightDiag &()> attachNote) const {

  if (isa<NotFoundFailure>(info)) {
    attachNote() << "parameter isn't used in any argument";
    return;
  }

  if (isa<ValueConflictFailure>(info)) {
    auto failure = cast<ValueConflictFailure>(info);
    attachNote() << "parameter inferred to two different values: " << failure.v1
                 << " and " << failure.v2;
    return;
  }

  auto failure = cast<TypeConflictFailure>(info);
  if (isa<TypeType>(failure.paramType)) {
    if (auto anyStruct = dyn_cast<StructMetaType>(failure.argParamType)) {
      attachNote() << "argument type " << anyStruct.getType()
                   << " is not a '@register_passable(\"trivial\")' type, so "
                      "does not satisfy AnyTrivialRegType";
      return;
    }
  }

  if (isa<TraitType>(failure.paramType)) {
    if (auto anyStruct = dyn_cast<StructMetaType>(failure.argParamType)) {
      attachNote() << "argument type " << anyStruct.getType()
                   << " does not conform to trait " << failure.paramType;
      return;
    }
    if (isa<TraitType>(failure.argParamType)) {
      attachNote() << "argument type " << failure.argParamType
                   << " is not a child trait of " << failure.paramType;
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// ParameterInferenceDiagnostics
//===----------------------------------------------------------------------===//

void ParameterInferenceDiagnostics::attach(PogListAttr params,
                                           InflightDiag &diag,
                                           size_t numActual) {
  // Pick the first diagnostic for the earliest parameter after numActual.
  const FailedInference *best = nullptr;
  for (const FailedInference &failure : diags) {
    // Ignore failures for things we know.  Why?
    if (failure.paramIdx < numActual)
      continue;
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

  if (!best)
    return;

  best->info.emitSpecificNote([&]() -> InflightDiag & {
    diag.attachNote(best->argExpr->getLoc())
        << best->argExpr->getRange() << "failed to infer parameter ";
    printNameOrIdx(params.getName(best->paramIdx), best->paramIdx, diag);
    return diag << ", ";
  });
}

//===----------------------------------------------------------------------===//
// ParameterInferenceState
//===----------------------------------------------------------------------===//

ParameterInferenceState::ParameterInferenceState(
    ASTDecl &declScope, const CallOperands &givenBindings,
    ArrayRef<TypedAttr> bindingsSoFar, const ParameterEvaluator &evaluator,
    ParameterInferenceDiagnostics &diags, bool allowImplicitConversions)
    : declScope(declScope), shared(declScope.getShared()),
      givenBindings(givenBindings), evaluator(evaluator),
      inferredParams(bindingsSoFar.begin(), bindingsSoFar.end()), diags(diags),
      allowImplicitConversions(allowImplicitConversions) {}

LogicalResult
ParameterInferenceState::matchFunctionTypes(FnTypeGeneratorType actual,
                                            FnTypeGeneratorType expected) {
  // TODO: Enable non-raising to raising conversions.
  if (actual.getFnEffects() != expected.getFnEffects())
    return failure();

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
  // TODO: We could allow implicit conversions here.
  if (failed(
          matchTypes(actual.getUserResultType(), expected.getUserResultType())))
    return failure();

  for (auto [actualResult, expectedResult] :
       llvm::zip(actual.getResults(), expected.getResults()))
    if (failed(matchTypes(actualResult, expectedResult))) {
      return failure();
    }

  if (failed(matchParams(actual.getCaptureOrigins(),
                         expected.getCaptureOrigins()))) {
    return failure();
  }

  ArrayRef<Type> actualArgTypes =
      actual.getArguments().drop_back(actualMemResult);
  ArrayRef<Type> expectedArgTypes =
      expected.getArguments().drop_back(expectedMemResult);

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

    ASTType expectedValueAstType =
        getFunctionArgumentRValueType(expectedAstType, expectedConv);
    ASTType actualValueAstType =
        getFunctionArgumentRValueType(actualAstType, actualConv);
    // Now check that the argument types line up.
    if (!succeeded(matchTypes(actualValueAstType.mlirType,
                              expectedValueAstType.mlirType)))
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

    for (size_t actualArgIndex = numNormalArgs;
         actualArgIndex < actualArgTypes.size(); actualArgIndex++) {
      auto actualConv = actual.getArgConvention(actualArgIndex);
      ASTType actualAstType = actualArgTypes[actualArgIndex];

      if (!checkConventionsConvertible(expectedConv, actualConv))
        return failure();

      ASTType actualValueAstType =
          getFunctionArgumentRValueType(actualAstType, actualConv);

      // If the argument types line up, then we can skip the rest of this.
      if (succeeded(
              matchTypes(actualValueAstType.mlirType, variadicElType.mlirType)))
        continue;

      // We can convert a more general `actual` function (that takes in a trait
      // argument) to a more specific `expected` function that takes in a struct
      // argument, as long as that struct conforms to that trait.
      // In other words, here we're handling function conversions with covariant
      // arguments (see TTSMFS).
      ExprEmitter emitter(declScope, EC_TypeParamValue);
      SyntheticNode synthNode(declScope.getLoc());
      CValue actualAstTypeCValue = CValue(actualValueAstType.mlirType);
      // Now, check if the actual arg can be converted to the expected trait.
      PValue actualAstTypeAsVariadicElTrait =
          emitter.emitMetaTypeToTraitConversion(
              {actualAstTypeCValue, synthNode}, expectedTraitType);
      if (!actualAstTypeAsVariadicElTrait) {
        return failure();
      }
      // And since we have it, let's use it to build up a kgen.variadic
      // parameter value.
      elements.push_back(actualAstTypeAsVariadicElTrait);
    }

    // Now assemble the kgen.variadic parameter value and match it against the
    // expected one.
    auto varType = VariadicType::get(variadicElType, ArgConvention::ReadReg);
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
  if (actualType == expectedType)
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
  // T: __type_of(Tuple[*ArgTypes]), this will infer that the ArgTypes variadic
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
          ExprEmitter::canZeroCostConvert(actualType, expectedType, shared));
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
      ++paramIndexRefDepth;

      if (succeeded(matchFunctionTypes(actual, expected))) {
        --paramIndexRefDepth;
        return success();
      } else {
        --paramIndexRefDepth;
        return failure();
      }
    }
  }

  // If the actual type is a reference to a parameter, it might be a local
  // parameter within a function.  The type checker will resolve this using the
  // metatype of the parameter, something like AnyStruct[someType].  This tells
  // us the actual type of the parameter.
  // TODO: Why isn't this a general solution?
  if (auto actualParamRef = dyn_cast<ParamType>(actualType)) {
    if (auto actualMetaType = ASTType(actualType).getMetaType()) {
      if (auto structMeta = dyn_cast<StructMetaType>(actualMetaType))
        return matchTypes(structMeta.getType(), expectedType);
      if (auto traitMeta = dyn_cast<AnyTraitType>(actualMetaType))
        return matchTypes(traitMeta.getTraitType(), expectedType);
    }
  }

  // TODO: We're not handling a lot of important things, e.g. conversion from
  // AnyStruct -> TraitType; conversion from AnyStruct -> AnyTrivialRegType;
  // implicit conversions that cause us to see i1->Bool and similar things here,
  // etc. as such, we can't treat conversion errors for unknown things as
  // failures.
  LLVM_DEBUG(llvm::errs() << "CANNOT INFER UNKNOWN TYPES:\n"; actualType.dump();
             expectedType.dump(); llvm::errs() << "\n");
  return failure();
}

LogicalResult ParameterInferenceState::matchParams(TypedAttr actualAttr,
                                                   TypedAttr expectedAttr) {
  // If the attrs trivial match then we're done and there is no inference to do.
  if (actualAttr == expectedAttr)
    return success();

  // Look through type upcasts to the more derived type.
  actualAttr = UpcastAttr::strip(actualAttr);
  expectedAttr = UpcastAttr::strip(expectedAttr);

  // We can only match up these values if their types match.
  if (actualAttr.getType() != expectedAttr.getType()) {
    // FIXME: Enforce attribute type convertibility.
    // This breaks, e.g.:
    // TypeParamAttr(T, SomeStruct) <-> TypeParamAttr(Param,
    // AnyTrivialRegType)
    (void)matchTypes(actualAttr.getType(), expectedAttr.getType());
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

  // If the expected value is the parameter declaration remember the binding!
  if (auto ire = dyn_cast<ParamIndexRefAttr>(expectedAttr)) {
    if (ire.getDepth() == paramIndexRefDepth &&
        // We need to infer in lexical order because we may have dependent types
        // between parameters.  The evaluator implicitly keeps track of how many
        // we have inferred.
        ire.getIndex() <= evaluator.getNumInputParams()) {
      // Compare the rebound types to handle dependent types.
      Type expectedType = evaluator.getReboundType(expectedAttr.getType());
      size_t parameterIndex = ire.getIndex();

      // If the types don't agree, attempt an implicit conversion between the
      // actual value and the expected type.
      if (actualAttr.getType() != expectedType) {
        ExprEmitter emitter(declScope, EC_TypeParamValue);
        SyntheticNode node(declScope.getLoc());
        if (ExprEmitter::canImplicitlyConvertToType(
                {actualAttr, node}, expectedType, emitter.getDeclScope())) {
          if (PValue result = emitter.emitPValue(
                  {actualAttr, node}, EC_TypeParamValue, expectedType))
            actualAttr = result;
        }
      }
      // If that didn't work, then we fail due to the type mismatch.
      if (actualAttr.getType() != expectedType) {
        // Otherwise, we failed to infer the parameter. Record this failure.
        addFailure(parameterIndex, InferenceFailure::TypeConflictFailure{
                                       expectedType, actualAttr.getType()});
        return failure();
      }

      // If we didn't already have a slot for this, make space.
      if (inferredParams.size() <= parameterIndex)
        inferredParams.resize(parameterIndex + 1);
      TypedAttr &inferredValue = inferredParams[parameterIndex];

      // Otherwise we succeeded in finding a value, see if it is compatible with
      // or more specific than the other values we've inferred.
      if (inferredValue && failed(matchParams(inferredValue, actualAttr))) {
        addFailure(parameterIndex, InferenceFailure::ValueConflictFailure{
                                       inferredValue, actualAttr});
        return failure();
      }

      inferredValue = actualAttr;

      // If we found the next missing parameter value for the evaluator, install
      // it so we can remap dependent types more effectively.
      if (parameterIndex == evaluator.getNumInputParams())
        evaluator.addInputValue(inferredValue);

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
      return failure();
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
//          *(0,3), "_value">, "value"> : index
//
// The "right" solution is to change pointer and reference to take an
// AddressSpace directly.  Until then we do a special hack for these things.
LogicalResult
ParameterInferenceState::matchSingleEltStruct(TypedAttr actual,
                                              TypedAttr expected) {
  if (actual == expected)
    return success();

  // If it is an extract from a known struct, then we know there is one field in
  // the struct - we can form a StructAttr around our actual value and recurse.
  if (auto expExtract = dyn_cast<LIT::StructExtractAttr>(expected)) {
    // If these are two lined up extracts, look through them.
    if (auto actExtract = dyn_cast<LIT::StructExtractAttr>(actual)) {
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
    auto expDRT = cast<StructType>(expStruct.getType());

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
      auto initSig = cast<FnTypeGeneratorType>(pValue.value().getType());
      // The constructed type is the result of the initializer.
      assert(initSig.getNumArguments() != 0);
      expDRT = cast<StructType>(initSig.getUserResultType());

      // Finally, perform any implicit conversion of the actual value to
      // whatever the 'value' would provide.
      auto argRVType = initSig.getArguments()[0];
      if (hasAddress(initSig.getArgConvention(0)))
        argRVType = ASTType(argRVType).getReferenceElementType();

      if (actual.getType() != argRVType &&
          ExprEmitter::canZeroCostConvert(actual.getType(), argRVType,
                                          shared)) {
        actual = ExprEmitter::emitZeroCostConvert(actual, argRVType, shared);
      }
    }

    // Now that we know the actual type, we can infer against a wrapped struct,
    // which can then infer from nested items etc.
    std::tuple<StringAttr, TypedAttr> actualField(expExtract.getField(),
                                                  actual);
    auto wrappedActual = LITStructAttr::get(actualField, expDRT);
    return matchSingleEltStruct(wrappedActual, expStruct);
  }

  return matchParams(actual, expected);
}

/// Return true if the specified parameter expression contains a reference to an
/// parameter that isn't yet bound in bindingsSoFar.
static bool usesUnboundParameters(TypedAttr paramValue,
                                  ArrayRef<TypedAttr> bindingsSoFar) {
  return paramValue
      .walk([&](ParamIndexRefAttr ref) -> WalkResult {
        if (ref.getDepth() == 0 && ref.getIndex() >= bindingsSoFar.size())
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
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
LogicalResult
ParameterInferenceState::inferSelfFromInitResult(Type returnedType) {
  // We can only support struct inference right now.
  auto returnedDRT = dyn_cast<StructType>(returnedType);
  if (!returnedDRT)
    return success();

  // Match up the parameter bindings if the 'actual' param is an UnboundAttr and
  // the expected has something more specific than a reference to the contextual
  // parameter.
  for (auto [idx, param] : llvm::enumerate(returnedDRT.getParamValues())) {
    // If this is simply a reference to the enclosing parameter (as in a normal
    // Self) init, then we can't infer anything from it.
    if (auto indexRef = dyn_cast<ParamIndexRefAttr>(param))
      if (indexRef.getDepth() == 0 && indexRef.getIndex() == idx)
        continue;

    // Notice that this is an explicitly bound Self parameter that we are going
    // to try to infer a more specific value for.  We need to remember this so
    // we can come back and refine it later. This is because we could have
    // inferred a forward reference, such as in:
    //   struct Foo[T: AnyType]:
    //     fn __init__[U: Movable](out self: Foo[U], x: U):
    selfResultParams.push_back(idx);

    // Otherwise, this is a more specialized parameter bound on Self for this
    // method.  Form the parameter that we need to infer.
    auto toInfer = ParamIndexRefAttr::get(/*depth*/ 0, idx, param.getType());

    // Try to infer this parameter from the expected (declared) type.
    if (failed(matchParams(param, toInfer))) {
      // If the parameter value depends on any uninferred (yet) parameters then
      // ignore the error.  Not doing so would cause a conflict with the correct
      // value eventually inferred.
      //
      // This is to enable us to handle things like:
      //   struct Foo[T: Int]:
      //     fn __init__[X: Int](out self: Foo[X+1], arg: Foo[X]):
      // Where we initially infer T = "X+1" (which isn't even valid because it
      // is referring the the X parameter), and then later refing it after we
      // discover what the value of X is when inferring parameter 1.  It is
      // gross that the value of parameter #0 can depend on parameter #1.  We
      // need out-of-order resolution.
      if (usesUnboundParameters(param, inferredParams))
        continue;
      return failure();
    }
  }

  return success();
}

/// Given a signature type that has some of its parameter bindings known, burn
/// the values for those parameters in, leaving the rest untouched so we can
/// infer them.
template <typename... Ts>
static std::tuple<Ts...>
getPartiallySpecializedSignature(ArrayRef<TypedAttr> bindingsSoFar,
                                 ParameterEvaluator &evaluator,
                                 bool signatureScoped, Ts... args) {
  if (bindingsSoFar.empty())
    return std::make_tuple(args...);

  struct Substitutor : IndexParameterReplacer<Substitutor> {
    Substitutor(ArrayRef<TypedAttr> bindingsSoFar,
                ParameterEvaluator &evaluator, bool signatureScoped)
        : bindingsSoFar(bindingsSoFar), evaluator(evaluator),
          signatureScoped(signatureScoped) {}

    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      // Depth-1 because we're matching against the signature parameters,
      // and that pushes level of depth immediately.
      auto ref = ::dyn_cast<ParamIndexRefAttr>(attr);
      if (!ref || ref.getDepth() != depth - signatureScoped ||
          ref.getIndex() >= bindingsSoFar.size())
        return {};
      TypedAttr result = bindingsSoFar[ref.getIndex()];

      [[maybe_unused]] auto getExpectedType = [&]() -> ASTType {
        // Since we're at a depth of `depth - signatureScoped`, while the
        // `evaluator` expects a depth of 0, we need to adjust any depths in the
        // type before running it through `evaluator`.
        IndexDepthAdjuster depthAdjuster(depth - signatureScoped);
        Type adjustedType = depthAdjuster.replace(result.getType());
        return evaluator.getReboundType(adjustedType);
      };
      assert(result.getType() == getExpectedType() &&
             "Parameter type mismatch");
      return result;
    }

    ArrayRef<TypedAttr> bindingsSoFar;
    ParameterEvaluator &evaluator;
    bool signatureScoped;
  } substitutor(bindingsSoFar, evaluator, signatureScoped);

  return std::make_tuple(substitutor.replace(args)...);
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

  auto resolveOperandCValue = [&]() -> CValue {
    if (auto argVal = value.getIfCValue())
      return argVal;

    OverloadSetUValue orValue = value.getIfOverloadSet();
    assert(orValue && "Unknown UValue!");
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
    // Infer the origin and address space before inferring the element type.
    CValue argVal = resolveOperandCValue();
    if (!argVal)
      return failure();

    RefType valueRefType;
    if (argVal.isMValue()) {
      valueRefType = cast<RefType>(value.getMValueReference().getType());
      // If the IRValue type is MBValue or MRValue then we need infer an
      // immutable ref, to match behavior where we don't allow passing an
      // MBValue or MRValue as 'mut'.
      if (!argVal.getIfMLValue() && !argVal.getIfMBPValue() &&
          !valueRefType.isMutableKnown(false))
        valueRefType = valueRefType.getWithMutability(false);

    } else {
      // If this is a def argument box, infer the reference from the underlying
      // def argument.
      if (auto dlv = argVal.getIfDLValue())
        valueRefType = dlv->getMBValueTypeFromDefArgument();
    }

    // If we are binding the reference to a value in memory directly, check for
    // reference compatibility.
    if (valueRefType)
      return matchTypes(valueRefType, expectedType);

    // Otherwise, we'll need to drop this value into a temporary.  For now, we
    // infer it as AnyOrigin.  We bind the origin directly and then handle
    // it like any other argument because we can support implicit conversions.
    valueRefType =
        RefType::getAnyOrigin(argVal.getRValueType(), /*isMut=*/false);
    auto expectedRef = cast<RefType>(expectedType);

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
  case ArgConvention::ReadMem:
    // Otherwise, we expect an r-value to match up, ignoring the reference type
    // from the convention.
    expectedType = expectedType.getReferenceElementType();
    break;
  case ArgConvention::ReadReg:
    break;
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
  auto getPartiallySpecializedType = [&]() -> ASTType {
    SmallVector<TypedAttr> currentParams;
    for (TypedAttr param : inferredParams) {
      if (param)
        currentParams.push_back(param);
      else
        break;
    }
    auto [type] = getPartiallySpecializedSignature(currentParams, evaluator,
                                                   /*signatureScoped=*/false,
                                                   Type(expectedType));
    return type;
  };

  // Check to see if the expected type has an initializer with the
  // specified operands.  Remove any parameters from the expected type
  // since those are what we're inferring from the arguments.  The result
  // 'actualType' will have those newly inferred parameters.
  if (auto initValue = operand.ir.getIfInitializer()) {
    FailureOr<PValue> initFn = OverloadSet::canConstructType(
        getPartiallySpecializedType().getWithoutParameters(shared),
        CallOperands(initValue.get()), operand.expr, declScope,
        /*isImplicitConversion=*/false);
    // If there were declaration errors, assume success to not raise
    // spurious errors due to not resolving to those erroneous
    // declarations.
    return success(failed(initFn) || bool(initFn.value()));
  }

  // Okay, we got a normal value argument convention and stripped off any
  // ArgConvention-related !lit.ref from the expected type.  See if we can
  // resolve the argument to a CValue.
  CValue argVal = resolveOperandCValue();
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
  if (ExprEmitter::canZeroCostConvert(argType, expectedType, shared))
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
  ASTType knownExpectedType = getPartiallySpecializedType();
  ASTDecl *expectedDecl = knownExpectedType.getDecl(shared);
  if (!allowImplicitConversions || !expectedDecl) {
    diags.resetDiags(std::move(noImplicitConversionDiags));
    return failure();
  }

  // Determine if we can construct the requested type given the existing value
  // we have.  If so, get the type inferred signature of the init method that
  // would make it work.
  ExprEmitter emitter(declScope, ExprContext::EC_CallArgValue);

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
  auto initSig = cast<FnTypeGeneratorType>(pValue.value().getType());
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
void ParameterInferenceState::infer(ArrayRef<Type> paramTypes,
                                    PogListAttr paramListAttr,
                                    bool hasArguments) {
  // If the parameter list has any inferred parameters, then we have to infer
  // against the provided binding list, since we might infer parameters from
  // other parameters. Otherwise, just exit early.
  if (paramTypes.empty() || (!paramListAttr.hasInferredParams() &&
                             !paramListAttr.isVariadic(paramTypes.size() - 1)))
    return;

  auto types = TypeArrayAttr::get(paramListAttr.getContext(), paramTypes);

  DefaultValueHandler defaultHandler(paramListAttr);
  std::tie(types, paramListAttr) = getPartiallySpecializedSignature(
      inferredParams, evaluator, /*signatureScoped=*/false, types,
      paramListAttr);

  size_t posIdx = 0, numParams = givenBindings.size();
  for (auto [idx, pog] : llvm::enumerate(paramListAttr.getPogs())) {
    // Inferred parameters won't have supplied values because they cannot be
    // specified by the user. We want to infer them from other parameters.
    if (pog.getPassingKind() == PassingKind::Inferred)
      continue;

    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = types[idx];

    // Zoom up to the next positional param.
    while (posIdx < numParams && givenBindings[posIdx].keyword)
      ++posIdx;

    // If we have a varargs parameters, then it will eat the rest of the
    // parameters, but we have to check each of them.
    if (paramListAttr.isVariadic(idx)) {
      auto expectedVariadic = cast<VariadicType>(expectedType);
      Type varArgsEltType = expectedVariadic.getElementType();
      while (posIdx != numParams) {
        if (!givenBindings[posIdx].keyword)
          inferOneParam(givenBindings[posIdx], varArgsEltType);
        ++posIdx;
      }
      continue;
    }

    // This must be a positional binding.
    if (posIdx < numParams) {
      inferOneParam(givenBindings[posIdx], expectedType);
      ++posIdx;
      continue;
    }

    // If we're out of positional bindings, try looking for a provided keyword
    // parameter binding.
    if (const OperandValue *param =
            givenBindings.findKwArg(paramListAttr.getName(idx))) {
      inferOneParam(*param, expectedType);
      continue;
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
    size_t nextParamNo = evaluator.getNumInputParams();
    if (nextParamNo < types.size() && paramListAttr.isVariadic(nextParamNo)) {
      // If we didn't already have a slot for this, make space.
      if (inferredParams.size() <= nextParamNo)
        inferredParams.resize(nextParamNo + 1);
      auto type = types[evaluator.getNumInputParams()];
      auto empty = VariadicAttr::get({}, cast<VariadicType>(type));
      inferredParams[nextParamNo] = empty;
      evaluator.addInputValue(empty);
    }
  }
}

LogicalResult ParameterInferenceState::infer(
    FnTypeGeneratorType signature, const CallOperands &operands,
    const OperandValueList &variadicKwOperands, bool returnsSelf) {
  // First try to infer parameters from parameters.
  infer(signature.getInputParamTypes(), signature.getParamListAttrs(),
        /*hasArguments*/ true);

  size_t numOperands = operands.size();

  DefaultValueHandler defaultHandler(signature.getArgListAttrs());
  std::tie(signature) = getPartiallySpecializedSignature(
      inferredParams, evaluator, /*signatureScoped=*/true, signature);

  // If this is a result in a returnsSelf function like an __init__, infer
  // self parameters (which could be specialized and shadowed).
  // NOTE: This has to happen early due to crazy cases like this:
  //   struct Example[T: AnyType]:
  //      fn __init__[U: Movable](owned value: U) -> Example[U]:
  //         pass
  // The way this works is that we infer "T = $0" here, then go on to analyze
  // the argument to infer that U = Int (or whatever), and then at the end of
  // this we go ahead and resolve $0 = Int.  This is crazily circuitous but is
  // because we have to infer parameter 0 before we can infer param #1.
  if (returnsSelf) {
    if (failed(inferSelfFromInitResult(signature.getUserResultType())))
      return failure();
  }

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t posOperandIdx = 0;
  for (auto [expectedArgIdx, expectedConvention] :
       llvm::enumerate(signature.getArgConventions())) {

    // There is no provided operand for a by-ref result.
    if (isResultSlot(expectedConvention))
      continue;

    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = signature.getArguments()[expectedArgIdx];

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
      auto expectedVariadic = cast<VariadicType>(expectedType);
      auto varArgsEltType = expectedVariadic.getElementType();
      while (posOperandIdx != numOperands) {
        auto &operand = operands[posOperandIdx];
        if (!operand.keyword &&
            failed(inferOneOperand(operand, varArgsEltType,
                                   expectedVariadic.getConvention())))
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
      ExprEmitter emitter(declScope, EC_TypeParamValue);
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
        SyntheticNode node(shared.getTopLevelDecl().getLoc());
        if (!ExprEmitter::canImplicitlyConvertToType(
                {actualAttr, node}, elementType, emitter.getDeclScope())) {

          // If that didn't work, then we fail due to the type mismatch.  If the
          // variadic type is due to a parameter mismatch, record it.
          if (auto ire = dyn_cast<ParamIndexRefAttr>(packType.getVariadic());
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
      auto variadicType = cast<VariadicType>(packType.getVariadic().getType());
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
  if (posOperandIdx != numOperands && !signature.getMetadata().hasVariadic())
    return failure();

  // If we had a variadic parameter that is unspecified, it must be because of
  // an empty variadic list.
  size_t nextParamNo = evaluator.getNumInputParams();
  if (nextParamNo < signature.getInputParamTypes().size() &&
      signature.getParamListAttrs().isVariadic(nextParamNo)) {
    // If we didn't already have a slot for this, make space.
    if (inferredParams.size() <= nextParamNo)
      inferredParams.resize(nextParamNo + 1);
    auto type = signature.getInputParamTypes()[evaluator.getNumInputParams()];
    auto empty = VariadicAttr::get({}, cast<VariadicType>(type));
    inferredParams[nextParamNo] = empty;
    evaluator.addInputValue(empty);
  }

  // Make sure to rebind any selfResultParams if they've been inferred already.
  // This is because we have to support things like:
  //
  //     struct Foo[T: AnyType]:
  //         fn __init__[U: Movable](x: U, out self: Foo[U]):
  //
  if (!selfResultParams.empty()) {
    // Need to first populate the evaluator with unbound attrs in case some
    // Self params were not deduced.
    ArrayRef<Type> paramTypes = signature.getInputParamTypes();
    for (size_t paramIdx = evaluator.getNumInputParams(),
                e = signature.getInputParamTypes().size();
         paramIdx < e; ++paramIdx)
      evaluator.addInputValue(UnboundAttr::get(paramTypes[paramIdx]));

    for (unsigned idx : selfResultParams) {
      if (idx < inferredParams.size()) {
        TypedAttr &param = inferredParams[idx];
        param = evaluator.getReboundAttribute(param);
      }
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
  ASTType declaredSelfType = signature.getArguments()[0];
  if (hasAddress(selfConvention))
    declaredSelfType = declaredSelfType.getReferenceElementType();

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
  if (auto structDecl = dyn_cast<StructDeclOp>(decl)) {
    IndexRefRemapper remapper(structDecl.getParams(), /*resultParams*/ {});
    selfType = remapper.replace(selfType.mlirType);
  }

  // If passing self by reference, wrap the Self type with the RefType
  // paraphernalia like origins.
  if (hasAddress(selfConvention)) {
    auto selfRefType = cast<RefType>(signature.getArguments()[0]);
    selfType = selfRefType.getWithElement(selfType);
  }

  // Infer the first operand against this type - it was presumably already
  // inferred against the methods declared type of 'self' as well.
  return inferOneOperand(operands[0], selfType, selfConvention);
}
