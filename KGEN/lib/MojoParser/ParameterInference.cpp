//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ParameterInference.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/ParamBindings.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/MojoParser/SharedState.h"

#include "MojoUtils.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

#define DEBUG_TYPE "LITEXPRCALLS"

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
  if (isa<TraitType>(failure.paramType)) {
    if (auto anyStruct = dyn_cast<AnyStructType>(failure.argParamType)) {
      attachNote() << "argument type " << anyStruct.getStructType()
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

LogicalResult ParameterInferenceState::matchTypes(Type actualType,
                                                  Type expectedType) {
  // If the types trivially match then there is no inference to do.
  if (actualType == expectedType)
    return success();

  // If the expected type is a parameter ref, then we're binding the specified
  // type to an attribute parameter.
  if (auto expectedParamRef = dyn_cast<ParamRefType>(expectedType)) {
    if (auto actualParamRef = dyn_cast<ParamRefType>(actualType)) {
      auto actualParam = actualParamRef.getParam();
      auto expectedParam = expectedParamRef.getParam();
      // If this type is a rebind of another type, then this is a downcast that
      // type erases, e.g. because it passed through some generic function which
      // had a looser type bound.  Remove the downcast to infer from the
      // super-type bound.
      if (auto rebind = dyn_cast<ParamOperatorAttr>(actualParam);
          rebind && rebind.getOpcode() == POC::Rebind)
        actualParam = rebind.getOperand(0);
      if (auto rebind = dyn_cast<ParamOperatorAttr>(expectedParam);
          rebind && rebind.getOpcode() == POC::Rebind)
        expectedParam = rebind.getOperand(0);

      return matchParams(actualParam, expectedParam);
    }

    ASTType type = actualType;
    if (ASTType nmTarget = type.getNonmaterializableTarget(shared))
      type = nmTarget;
    Type metatype = type.getMetaType();
    if (!metatype) // Otherwise, this is an MLIR type.
      metatype = TypeType::get(actualType.getContext());

    return matchParams(TypeConstantAttr::get(type, metatype),
                       expectedParamRef.getParam());
  }

  // Handle when both are DeclRefTypes.
  if (auto actualDRT = dyn_cast<DeclRefType>(actualType)) {
    if (auto expectedDRT = dyn_cast<DeclRefType>(expectedType)) {
      // Ignore if these are two fundamentally different symbols.
      if (actualDRT.getSymbol() != expectedDRT.getSymbol())
        return failure();

      // Fail if the parameter lists fundamentally mismatch.
      // TODO: Defaulted parameters could make this ok?
      if (actualDRT.getParamValues().size() !=
          expectedDRT.getParamValues().size())
        return failure();

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
      if (failed(matchParams(actual.getLifetime(), expected.getLifetime()))) {
        // The lifetimes are allowed to mismatch due to mut->immut casts.
        if (!canConvertWithRebind(actual.getLifetimeType(),
                                  expected.getLifetimeType(), shared))
          return failure();
      }
      return matchSingleEltStruct(actual.getAddressSpace(),
                                  expected.getAddressSpace());
    }

  // Handle LifetimeType.
  if (auto actual = dyn_cast<LifetimeType>(actualType))
    if (auto expected = dyn_cast<LifetimeType>(expectedType)) {
      // Try to match up the types so we infer parameters properly.
      if (succeeded(
              matchSingleEltStruct(actual.isMutable(), expected.isMutable())))
        return success();
      // If that fails, check compatibility, actualType might be mutable=true,
      // and expected might be mutable=false, and this is fine.
      return success(canConvertWithRebind(actualType, expectedType, shared));
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
          failed(matchParams(actual.getLifetime(), expected.getLifetime())))
        return failure();
      return matchParams(actual.getAddressSpace(), expected.getAddressSpace());
    }

  // Handle SignatureType
  if (auto actual = dyn_cast<SignatureType>(actualType))
    if (auto expected = dyn_cast<SignatureType>(expectedType)) {
      // When checking SignatureTypes, we have to keep track of
      // paramIndexRefDepth to be sure we are binding the right parameters.
      if (actual.getArguments().size() == expected.getArguments().size() &&
          actual.getResults().size() == expected.getResults().size()) {
        ++paramIndexRefDepth;
        for (auto [actualArgument, expectedArgument] :
             llvm::zip(actual.getArguments(), expected.getArguments()))
          if (failed(matchTypes(actualArgument, expectedArgument)))
            return failure();

        for (auto [actualResult, expectedResult] :
             llvm::zip(actual.getResults(), expected.getResults()))
          if (failed(matchTypes(actualResult, expectedResult)))
            return failure();

        --paramIndexRefDepth;
        return success();
      }
    }

  // If the actual type is a reference to a parameter, it might be a local
  // parameter within a function.  The type checker will resolve this using the
  // metatype of the parameter, something like AnyStruct[someType].  This tells
  // us the actual type of the parameter.
  // TODO: Why isn't this a general solution?
  if (auto actualParamRef = dyn_cast<ParamRefType>(actualType)) {
    if (auto actualMetaType = ASTType(actualType).getMetaType()) {
      if (auto structMeta = dyn_cast<AnyStructType>(actualMetaType))
        return matchTypes(structMeta.getStructType(), expectedType);
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

  // We can only match up these values if their types match.
  if (actualAttr.getType() != expectedAttr.getType()) {
    // FIXME: Enforce attribute type convertibility.
    // This breaks, e.g.:
    // TypeConstantAttr(T, SomeStruct) <-> TypeConstantAttr(Param,
    // AnyTrivialRegType)
    (void)matchTypes(actualAttr.getType(), expectedAttr.getType());
  }

  // If the actual value is a ? then we never bind to it.
  if (isa<UnboundAttr>(actualAttr))
    return success();

  // If we are dealing with two type constants, we match their values.
  auto actualTypeConst = dyn_cast<TypeConstantAttr>(actualAttr);
  auto expectedTypeConst = dyn_cast<TypeConstantAttr>(expectedAttr);
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
    if (ire.getDepth() == paramIndexRefDepth && !ire.getIsResult() &&
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
        ExprEmitter emitter(shared, declScope, EC_TypeParamValue);
        SyntheticNode node(declScope.getLoc());
        if (emitter.canImplicitlyConvertToType({actualAttr, node},
                                               expectedType)) {
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

      // Otherwise we succeeded in finding a value, see if it is compatible
      // with other values we've inferred.
      if (inferredValue && inferredValue != actualAttr) {
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

  LLVM_DEBUG(llvm::errs() << "CANNOT INFER UNKNOWN ATTRS:\n"; actualAttr.dump();
             expectedAttr.dump(); llvm::errs() << "\n");
  return failure();
}

// Special Hack (tm) for address space matching for pointers and references.
// Both are defined like this:
//   struct YourPointer[type: ..., address_space: AddressSpace]:
//     alias _mlir_type =
//        `!kgen.pointer<`, type, `,`, address_space._value.value, `>`
//     fn __init__(inout self, address: Self._mlir_type):
//
// When inferring the "address_space" constructor from a concrete pointer type,
// we end up seeing a concrete value in the !kgen.pointer, e.g. "3" as an index
// value.  However, we need to realize a value for the address_space parameter,
// which is a struct of a struct of an index.
//
// This ends up looking like:
//   ActualAttr = 0 : index
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

    if (actual.getType() != expected.getType())
      return failure();

    auto expStruct = expExtract.getStructValue();
    // Figure out if the struct is something we can handle.
    auto expDRT = cast<DeclRefType>(expStruct.getType());
    // Conservatively only handle the types we know have a single field.
    if (expDRT.getName().strref() != "AddressSpace" &&
        expDRT.getName().strref() != "Int" &&
        expDRT.getName().strref() != "Bool")
      return failure();
    std::tuple<StringAttr, TypedAttr> actualField(expExtract.getField(),
                                                  actual);
    auto wrappedActual = LITStructAttr::get(actualField, expDRT);
    return matchSingleEltStruct(wrappedActual, expStruct);
  }

  return matchParams(actual, expected);
}

/// Infer parameters from an operand being passed into this function. This is
/// only called on the top level function operands being matched up, not
/// anything in recursive functiontype positions.
LogicalResult
ParameterInferenceState::inferOneOperand(ASTExprAnd<AnyValue> operand,
                                         ASTType expectedType,
                                         ArgConvention expectedConvention) {
  AnyValue value = operand.ir;
  curArgExpr = operand.expr;

  auto resolveOperandCValue = [&]() -> CValue {
    if (auto argVal = value.getIfCValue())
      return argVal;

    OverloadSetUValue orValue = value.getIfOverloadSet();
    assert(orValue && "Unknown UValue!");
    // Try to refine the OverloadSetUValue into a PValue.
    CValue argVal = orValue->getDirectSymbol(expectedType);
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
  case ArgConvention::InitSelf:
    // If this is an UnknownAttr, then it is a placeholder for type
    // checking, match up the types, but otherwise let it pass.
    if (PValue pValue = value.getIfPValue())
      if (isa<UnknownAttr>(pValue.get())) {
        ASTType argType(pValue.get().getType());
        return matchTypes(argType.getReferenceElementType(),
                          expectedType.getReferenceElementType());
      }
    [[fallthrough]];
  case ArgConvention::InOut:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError: {
    // The actual value must be an lvalue if callee takes things by-ref.
    LValue argVal = value.getIfLValue();
    if (!argVal)
      return failure();

    // By-ref argument types must exactly match, no conversions are allowed.
    return matchTypes(argVal.getRValueType(),
                      expectedType.getReferenceElementType());
  }

  case ArgConvention::Ref: {
    // Infer the lifetime and address space before inferring the element type.
    CValue argVal = resolveOperandCValue();
    if (!argVal)
      return failure();

    RefType valueRefType;
    if (value.isMValue())
      valueRefType = cast<RefType>(value.getMValueReference().getType());
    else if (value.getIfPValue() && isParamContext)
      valueRefType =
          RefType::getImmortal(argVal.getRValueType(), /*isMut=*/true);

    // As a special hack, look through DefArgumentWrapperDLValue to the
    // underlying MBValue that it may contain.  This is for two reasons:
    //  1) We don't want to infer mutability from the argument even though
    //     it is a DLValue, because we'd force copy-out + writeback,
    //     materializing the def argument box.
    //  2) We have significant bugs around lifetime inference from SBValues
    //     and DLValues because we're not materializing the box in time.  This
    //     is tracked by MOCO-684.
    // Solve this by hacking this important case specifically.
    if (auto dlValue = value.getIfDLValue())
      if (auto refType = dlValue->getMBValueTypeFromDefArgument())
        valueRefType = refType;

    if (valueRefType)
      return matchTypes(valueRefType, expectedType);
    return success();
  }
  case ArgConvention::OwnedInMem:
  case ArgConvention::BorrowedInMem:
    // Otherwise, we expect an r-value to match up, ignoring the reference type
    // from the convention.
    expectedType = expectedType.getReferenceElementType();
    break;
  case ArgConvention::OwnedInReg:
  case ArgConvention::BorrowedInReg:
    break;
  case ArgConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }

  // Check to see if the expected type has an initializer with the
  // specified operands.  Remove any parameters from the expected type
  // since those are what we're inferring from the arguments.  The result
  // 'actualType' will have those newly inferred parameters.
  if (auto initValue = operand.ir.getIfInitializer()) {
    ExprEmitter emitter(shared, declScope, ExprContext::EC_CallArgValue);
    auto [initFn, erroneousDecl] = emitter.canConstructType(
        expectedType.getWithoutParameters(emitter.shared), initValue.get(),
        operand.expr);
    // If there were declaration errors, assume success to not raise
    // spurious errors due to not resolving to those erroneous
    // declarations.
    return success(bool(initFn) || erroneousDecl);
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

  // Zero cost conversions don't count as implicit conversions.
  if (canConvertWithRebind(argType, expectedType, shared))
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
  ASTDecl *expectedDecl = expectedType.getDecl(shared);
  if (!allowImplicitConversions || !expectedDecl) {
    diags.resetDiags(std::move(noImplicitConversionDiags));
    return failure();
  }

  // Determine if we can construct the requested type given the existing value
  // we have.  If so, get the type inferred signature of the init method that
  // would make it work.
  ExprEmitter emitter(shared, declScope, ExprContext::EC_CallArgValue);

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
  auto [pValue, _] = emitter.canConstructType(
      nonParamType, CallOperands({{argVal, curArgExpr}}), curArgExpr,
      /*allowImplicitConversions=*/false);
  if (!pValue) {
    // If we had a fully formed type that we were inferring into, then this is
    // a failure.
    if (nonParamType.mlirType == expectedType.mlirType) {
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
  auto initSig = cast<LITSignatureType>(pValue.getType());
  // We expected to args: 0=self, 1=value we're converting from.
  ASTType inferredSelf;
  if (initSig.getArgConvention(0) == ArgConvention::InitSelf)
    inferredSelf = ASTType(initSig.getArguments()[0]).getReferenceElementType();
  else // FIXME: get rid of -> Self initializers.
    inferredSelf = initSig.getResultType();

  // Infer the parameters of this overload candidate against the computed
  // result type of the initializer.
  auto result = matchTypes(inferredSelf, expectedType);

  // If the implicit conversion worked then we're good.
  if (succeeded(result))
    return success();

  // Otherwise restore the diags from the non-implicit conversion path,
  // they'll be less confusing.
  diags.resetDiags(std::move(noImplicitConversionDiags));
  return failure();
}

void ParameterInferenceState::inferOneParam(
    const ParamBindings::Binding &binding, Type expectedType) {
  // Don't infer from unpacked parameters.
  if (isa<UnpackedAttr>(binding.value))
    return;
  curArgExpr = binding.expr;
  (void)matchTypes(binding.getType(), expectedType);
}

/// Given a signature type that has some of its parameter bindings known, burn
/// the values for those parameters in, leaving the rest untouched so we can
/// infer them.
template <typename... Ts>
static std::tuple<Ts...>
getPartiallySpecializedSignature(ArrayRef<TypedAttr> bindingsSoFar,
                                 ParserParamEvaluator &evaluator, Ts... args) {
  if (bindingsSoFar.empty())
    return std::make_tuple(args...);

  struct Substitutor : IndexParameterReplacer<Substitutor> {
    Substitutor(ArrayRef<TypedAttr> bindingsSoFar,
                ParserParamEvaluator &evaluator)
        : bindingsSoFar(bindingsSoFar), evaluator(evaluator) {}

    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      // Depth-1 because we're matching against the signature parameters,
      // and that pushes level of depth immediately.
      auto ref = ::dyn_cast<ParamIndexRefAttr>(attr);
      if (!ref || ref.getDepth() != depth - 1 ||
          ref.getIndex() >= bindingsSoFar.size())
        return {};
      auto result = bindingsSoFar[ref.getIndex()];
      assert(result.getType() == evaluator.getReboundType(ref.getType()) &&
             "Parameter type mismatch");
      return result;
    }

    ArrayRef<TypedAttr> bindingsSoFar;
    ParserParamEvaluator &evaluator;
  } substitutor(bindingsSoFar, evaluator);

  auto refine = [&](auto arg) {
    auto newArg = substitutor.replace(arg);
    if (newArg == arg)
      return arg;
    // If we changed something, then we substituted constants into the type
    // tree. This can cause some expressions to fold with the interpreter, so
    // see if we can simplify the result.
    return cast<decltype(arg)>(evaluator.refine(newArg));
  };

  return std::make_tuple(refine(args)...);
}

/// Apply the bindings so far (plus a distinct new attribute relating
/// back to the original decls for ones that are missing) to the signature with
/// getSpecializedSignature so we benefit from the already-fixed substitutions
/// being applied to the input types.  This can make them more concrete and
/// help with inferring dependent types based on already-bound parameters.
template <typename... Ts>
static bool partiallySpecializeIfNeeded(ArrayRef<TypedAttr> inferredParams,
                                        ParserParamEvaluator &evaluator,
                                        size_t &numAlreadySpecialized,
                                        Ts &...args) {
  // If we inferred a value for the parameter from previous arguments,
  // substitute it into the expected types of subsequent arguments.  This
  // allows us to handle dependent argument types like:
  //    fn foo[dt: DType](p: DTypePointer[dt], v: Scalar[p.type]):
  // where the type of 'v' depends on 'dt' being inferred.
  //
  // FIXME: Don't do this, it makes it more difficult to diagnose conflicting
  // values.  We should switch over to using the evaluator instead.
  if (numAlreadySpecialized < inferredParams.size() &&
      inferredParams[numAlreadySpecialized]) {
    // Take all the bindings that are now known.  Be careful about gaps.
    SmallVector<TypedAttr> effectiveBindings(inferredParams);
    for (auto it = effectiveBindings.begin(), e = effectiveBindings.end();
         it != e; ++it) {
      // Drop any unknown parameter values and everything after it.
      if (!*it) {
        effectiveBindings.erase(it, effectiveBindings.end());
        break;
      }
    }

    std::tie(args...) =
        getPartiallySpecializedSignature(effectiveBindings, evaluator, args...);
    numAlreadySpecialized = effectiveBindings.size();
    return true;
  }
  return false;
}

/// Helper that returns true if the parameter list has any inferred parameters.
static bool hasInferredParams(PogListAttr paramListAttr) {
  ArrayRef<PogMetadataAttr> params = paramListAttr.getPogs();
  return !params.empty() &&
         params.front().getPassingKind() == PassingKind::Inferred;
}

void ParameterInferenceState::infer(ArrayRef<Type> paramTypes,
                                    PogListAttr paramListAttr) {
  // If the parameter list has any inferred parameters, then we have to infer
  // against the provided binding list, since we might infer parameters from
  // other parameters. Otherwise, just exit early.
  if (!hasInferredParams(paramListAttr))
    return;

  auto types = TypeArrayAttr::get(paramListAttr.getContext(), paramTypes);

  size_t numAlreadySpecialized = inferredParams.size();
  DefaultValueHandler defaultHandler(paramListAttr);
  std::tie(types, paramListAttr) = getPartiallySpecializedSignature(
      inferredParams, evaluator, types, paramListAttr);
  auto rebindPartialTypes = [&]() {
    if (partiallySpecializeIfNeeded(inferredParams, evaluator,
                                    numAlreadySpecialized, types,
                                    paramListAttr))
      defaultHandler = DefaultValueHandler(paramListAttr);
  };

  size_t posIdx = 0, numPosParams = givenBindings.posOperands.size();
  for (auto [idx, pog] : llvm::enumerate(paramListAttr.getPogs())) {
    // Inferred parameters won't have supplied values because they cannot be
    // specified by the user. We want to infer them from other parameters.
    if (pog.getPassingKind() == PassingKind::Inferred)
      continue;
    rebindPartialTypes();

    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = types[idx];

    // If we have a varargs parameters, then it will eat the rest of the
    // parameters, but we have to check each of them.
    if (paramListAttr.isVariadic(idx)) {
      auto expectedVariadic = cast<VariadicType>(expectedType);
      Type varArgsEltType = expectedVariadic.getElementType();
      while (posIdx != numPosParams)
        inferOneParam(givenBindings.posOperands[posIdx++], varArgsEltType);
      continue;
    }

    // If we're out of positional bindings, try looking for a provided keyword
    // parameter binding.
    if (posIdx == numPosParams) {
      if (std::optional<ParamBindings::Binding> param =
              givenBindings.findKwArg(paramListAttr.getName(idx))) {
        inferOneParam(*param, expectedType);
        continue;
      }

      // If not, and this parameter has a default value, then just skip it. We
      // can't infer from default values.
      if (defaultHandler.getDefault(idx))
        continue;

      // Otherwise, this is a missing parameter. Just skip it.
      continue;
    }

    // In the typical case, this isn't a variadic or keyword parameter. It
    // must be a positional binding.
    inferOneParam(givenBindings.posOperands[posIdx++], expectedType);
  }
}

LogicalResult
ParameterInferenceState::infer(LITSignatureType signature,
                               const CallOperands &callOperands,
                               const KeywordOperands &variadicKwOperands) {
  // First try to infer parameters from parameters.
  infer(signature.getParamTypes(), signature.getParamListAttrs());

  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  size_t numPosOperands = posOperands.size();

  size_t numAlreadySpecialized = inferredParams.size();
  DefaultValueHandler defaultHandler(signature.getArgListAttrs());
  std::tie(signature) =
      getPartiallySpecializedSignature(inferredParams, evaluator, signature);
  auto rebindPartialSignature = [&](bool isParam = false) {
    if (partiallySpecializeIfNeeded(inferredParams, evaluator,
                                    numAlreadySpecialized, signature)) {
      defaultHandler =
          DefaultValueHandler(isParam ? signature.getParamListAttrs()
                                      : signature.getArgListAttrs());
    }
  };

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t posOperandIdx = 0;
  for (auto [expectedArgIdx, expectedConvention] :
       llvm::enumerate(signature.getArgConventions())) {

    // There is no provided operand for a by-ref result.
    if (SignatureType::isResultSlot(expectedConvention))
      continue;
    rebindPartialSignature();

    // Note that 'signature' changes the type as we go, so don't use
    // llvm::enumerate on the argument type list!
    Type expectedType = signature.getArguments()[expectedArgIdx];

    if (signature.isKwVarArg(expectedArgIdx)) {
      Type valTy = ASTType(expectedType).getKwargsDictRefValueType();
      for (auto [name, operand] : variadicKwOperands) {
        // TODO: Passing OwnedInReg is a hack that is needed because the value
        // type is not a reference type (and doesn't have a lifetime), but we
        // still want to type check it. So, passing it as if it was reg-passable
        // happens to just work, until we rectify this. Right now the reason the
        // value type cannot be a reference type is because `Reference` does not
        // (and in fact cannot) conform to `CollectionElement`.
        if (failed(inferOneOperand(operand, valTy, ArgConvention::OwnedInReg)))
          return failure();
      }
      continue;
    }

    // If we have a varargs argument, then it will eat the rest of the
    // arguments, but we have to check each of them.
    if (signature.isPosVarArg(expectedArgIdx)) {
      auto expectedVariadic = cast<VariadicType>(expectedType);
      auto varArgsEltType = expectedVariadic.getElementType();
      while (posOperandIdx != numPosOperands)
        if (failed(inferOneOperand(posOperands[posOperandIdx++], varArgsEltType,
                                   expectedVariadic.getConvention())))
          return failure();
      continue;
    }

    // If we have a pack argument, then we're binding a variadic parameter with
    // multiple type values.  We need to consume all remaining arguments and use
    // their RValue types as bindings.
    if (ASTType variadicPackType =
            signature.getIfVariadicPack(expectedArgIdx)) {
      RefPackType packType = variadicPackType.getVariadicPackInfo();

      // Figure out that the element type of the list is, e.g. AnyType or
      // Stringable.
      Type elementType = packType.getVariadicElementType();

      SmallVector<TypedAttr> types;
      ExprEmitter emitter(shared, declScope, EC_TypeParamValue);
      while (posOperandIdx != numPosOperands) {
        ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx++];
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
        TypedAttr actualAttr = TypeConstantAttr::get(
            toPush, metatype ? metatype : TypeType::get(shared.getContext()));
        SyntheticNode node(shared.getTopLevelDecl().getLoc());
        if (!emitter.canImplicitlyConvertToType({actualAttr, node},
                                                elementType))
          return failure();
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

    // Handle case when there are no more provided positional operands.
    if (posOperandIdx == numPosOperands) {
      // Check if a keyword operand was provided for this argument
      if (std::optional<ASTExprAnd<AnyValue>> kwOperandOr =
              callOperands.findKwArg(signature.getArgName(expectedArgIdx))) {
        if (failed(inferOneOperand(*kwOperandOr, expectedType,
                                   expectedConvention)))
          return failure();
        continue;
      }

      // If available, we check the default argument value.
      // NOTE: The type of the default argument has to match the argument type,
      // meaning there can't be anything to infer here directly, but we still
      // check to make sure that the default value doesn't contradict already
      // inferred parameters.
      if (TypedAttr defaultOr = defaultHandler.getDefault(expectedArgIdx)) {
        if (failed(inferOneOperand({defaultOr, curArgExpr}, expectedType,
                                   expectedConvention)))
          return failure();
        continue;
      }

      // Otherwise we have an argument count mismatch, just fail.
      return failure();
    }

    // In the typical case, this argument isn't varargs or a pack, so just check
    // it.  If there was a problem, report it, otherwise continue on to the next
    // expected argument to check.
    if (failed(inferOneOperand(posOperands[posOperandIdx++], expectedType,
                               expectedConvention)))
      return failure();
  }

  // If we have left over operands, then this signature cannot match.
  if (posOperandIdx != numPosOperands && !signature.hasParamVarArgs())
    return failure();

  // We succeed iff we inferred a value for this parameter.
  return success();
}
