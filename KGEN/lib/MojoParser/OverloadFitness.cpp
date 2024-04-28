//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the components for overload fitness evaluation.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/OverloadFitness.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "OperandDiagnostics.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "LITEXPRCALLS"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Helper to get the RValueType from an operand.
static ASTType getRValueTypeIfResolvable(AnyValue value) {
  if (auto cValue = value.getIfCValue())
    return cValue.getRValueType();
  // Otherwise, try to narrow an overload set to a PValue.
  if (auto ovSet = value.getIfOverloadSet())
    if (auto pValue = ovSet->getIfPValue())
      return pValue.getType();
  // Initializer lists have no implied type.
  return ASTType();
}

namespace {
/// This failure happens when a parameter is found of the wrong type.
struct TypeConflictFailure {
  ASTType paramType, argParamType;
};

// This failure happens when a parameter is inferred to two different values.
struct ValueConflictFailure {
  TypedAttr v1, v2;
};

// This failure happens when the parameter isn't found at all.
struct NotFoundFailure {};

// These are the different failure modes that we know happen.
struct InferenceFailure {
  template <typename Failure>
  InferenceFailure(Failure info) : info(info) {}
  // Describe what went wrong.
  void emitSpecificNote(function_ref<InflightDiag &()> attachNote) const;

private:
  SmartVariant<TypeConflictFailure, ValueConflictFailure, NotFoundFailure> info;
};
} // namespace

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

namespace {
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
  void attach(LITSignatureType signature, InflightDiag &diag, size_t numActual,
              const CallOperands &operands);

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
} // namespace

/// Helper to consistently print a parameter/argument name or index (if the name
/// is empty) in diagnostics.
static void printNameOrIdx(StringAttr name, size_t idx, InflightDiag &diag) {
  if (!name.empty())
    diag << "'" << name.getValue() << "'";
  else
    diag << "#" << idx;
}

void ParameterInferenceDiagnostics::attach(LITSignatureType signature,
                                           InflightDiag &diag, size_t numActual,
                                           const CallOperands &operands) {
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
    printNameOrIdx(signature.getParamName(best->paramIdx), best->paramIdx,
                   diag);
    return diag << ", ";
  });
}

//===----------------------------------------------------------------------===//
// ParameterInferenceState Implementation
//===----------------------------------------------------------------------===//

namespace {
/// This class provides the implementation details that help to infer
/// information about the specified parameter.
class ParameterInferenceState {
public:
  ParameterInferenceState(ASTDecl &declScope, SharedState &shared,
                          ArrayRef<TypedAttr> bindingsSoFar,
                          const ParserParamEvaluator &evaluator,
                          ParameterInferenceDiagnostics &diags,
                          bool allowImplicitConversions)
      : declScope(declScope), shared(shared), evaluator(evaluator),
        inferredParams(bindingsSoFar.begin(), bindingsSoFar.end()),
        diags(diags), allowImplicitConversions(allowImplicitConversions) {}

  LogicalResult infer(LITSignatureType signature,
                      const CallOperands &callOperands,
                      const KeywordOperands &variadicKwOperands);

  /// After inferring parameter values, this allows access to the results.
  TypedAttr getInferredValue(size_t idx) const {
    return idx < inferredParams.size() ? inferredParams[idx] : TypedAttr();
  }

private:
  LogicalResult matchTypes(Type actualType, Type expectedType);
  LogicalResult matchParams(TypedAttr actualAttr, TypedAttr expectedAttr);
  LogicalResult matchAddressSpace(TypedAttr actualAddrSpace,
                                  TypedAttr expectedAddrSpace);

  /// Infer parameters from an operand being passed into this function. This is
  /// only called on the top level function operands being matched up, not
  /// anything in recursive functiontype positions.
  LogicalResult inferOneOperand(ASTExprAnd<AnyValue> operand,
                                ASTType expectedType,
                                ArgConvention expectedConvention);
  void addFailure(size_t parameterIndex, InferenceFailure &&info) {
    diags.addFailure(parameterIndex, curArgExpr, std::move(info));
  }

  ASTDecl &declScope;
  SharedState &shared;
  ParserParamEvaluator evaluator;

  /// One entry for each parameter from the original binding list.  If
  /// non-null, we've already inferred a value for that parameter.
  SmallVector<TypedAttr> inferredParams;

  size_t paramIndexRefDepth = 0;
  ParameterInferenceDiagnostics &diags;

  // True if implicit conversions in argument lists are permitted.
  bool allowImplicitConversions;

  const ExprNode *curArgExpr = nullptr;
};
} // namespace

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
      // If this type is a rebind of another type, then this is a downcast that
      // type erases, e.g. because it passed through some generic function which
      // had a looser type bound.  Remove the downcast to infer from the
      // super-type bound.
      if (auto rebind = dyn_cast<ParamOperatorAttr>(actualParam);
          rebind && rebind.getOpcode() == POC::Rebind)
        actualParam = rebind.getOperand(0);

      return matchParams(actualParam, expectedParamRef.getParam());
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
      return matchAddressSpace(actual.getAddressSpace(),
                               expected.getAddressSpace());
    }

  // Handle LifetimeType.
  if (auto actual = dyn_cast<LifetimeType>(actualType))
    if (auto expected = dyn_cast<LifetimeType>(expectedType)) {
      // Try to match up the types so we infer parameters properly.
      if (succeeded(matchParams(actual.isMutable(), expected.isMutable())))
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
      return matchAddressSpace(actual.getAddressSpace(),
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
  // AnyStruct -> TraitType; conversion from AnyStruct -> AnyRegType; implicit
  // conversions that cause us to see i1->Bool and similar things here, etc.
  // as such, we can't treat conversion errors for unknown things as failures.
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
    // TypeConstantAttr(T, SomeStruct) <-> TypeConstantAttr(Param, AnyRegType)
    (void)matchTypes(actualAttr.getType(), expectedAttr.getType());
  }

  // If the actual value is a ? then we never bind to it.
  if (isa<UnboundAttr>(actualAttr))
    return success();

  // If we are dealing with two type constants, we match their values.
  auto actualTypeConst = dyn_cast<TypeConstantAttr>(actualAttr);
  auto expectedTypeConst = dyn_cast<TypeConstantAttr>(expectedAttr);
  if (actualTypeConst && expectedTypeConst)
    return matchTypes(actualTypeConst.getValue(), expectedTypeConst.getValue());

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
        addFailure(parameterIndex,
                   TypeConflictFailure{expectedType, actualAttr.getType()});
        return failure();
      }

      // If we didn't already have a slot for this, make space.
      if (inferredParams.size() <= parameterIndex)
        inferredParams.resize(parameterIndex + 1);
      TypedAttr &inferredValue = inferredParams[parameterIndex];

      // Otherwise we succeeded in finding a value, see if it is compatible
      // with other values we've inferred.
      if (inferredValue && inferredValue != actualAttr) {
        addFailure(parameterIndex,
                   ValueConflictFailure{inferredValue, actualAttr});
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
LogicalResult ParameterInferenceState::matchAddressSpace(TypedAttr actual,
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
      return matchAddressSpace(actExtract.getStructValue(),
                               expExtract.getStructValue());
    }

    if (actual.getType() != expected.getType())
      return failure();

    auto expStruct = expExtract.getStructValue();
    // Figure out if the struct is something we can handle.
    auto expDRT = cast<DeclRefType>(expStruct.getType());
    // Conservatively only handle the types we know have a single field.
    if (expDRT.getName().strref() != "AddressSpace" &&
        expDRT.getName().strref() != "Int")
      return failure();
    std::tuple<StringAttr, TypedAttr> actualField(expExtract.getField(),
                                                  actual);
    auto wrappedActual = LITStructAttr::get(actualField, expDRT);
    return matchAddressSpace(wrappedActual, expStruct);
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
  case ArgConvention::ByRef:
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

  case ArgConvention::OwnedInMem:
  case ArgConvention::BorrowedInMem:
    // Otherwise,we expect an r-value to match up, ignoring the reference type
    // from the convention.
    expectedType = expectedType.getReferenceElementType();
    break;
  case ArgConvention::OwnedInReg:
  case ArgConvention::BorrowedInReg:
    break;
  case ArgConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }

  // Okay, we got a normal value argument convention and stripped off any
  // ArgConvention-related !lit.ref from the expected type.
  CValue argVal = value.getIfCValue();
  if (!argVal) {
    if (auto initValue = operand.ir.getIfInitializer()) {
      // Check to see if the expected type has an initializer with the
      // specified operands.  Remove any parameters from the expected type
      // since those are what we're inferring from the arguments.  The result
      // 'actualType' will have those newly inferred parameters.
      ExprEmitter emitter(shared, declScope, ExprContext::EC_CallArgValue);
      auto [initFn, erroneousDecl] = emitter.canConstructType(
          expectedType.getWithoutParameters(emitter.shared), initValue.get(),
          operand.expr);
      // If there were declaration errors, assume success to not raise
      // spurious errors due to not resolving to those erroneous
      // declarations.
      return success(bool(initFn) || erroneousDecl);
    }

    OverloadSetUValue orValue = value.getIfOverloadSet();
    assert(orValue && "Unknown UValue!");
    // Try to refine the OverloadSetUValue into a PValue.
    argVal = orValue->getDirectSymbol(expectedType);
    if (!argVal)
      return failure();
    // If we have a reference to an overloaded method like foo(a.method),
    // then we can't resolve it.
    // TODO(partial application => closures): Given we just resolved argVal,
    // we could form the "a.method" expression with a closure.
    if (orValue->baseValue) // Cannot merge base value.
      return failure();
  }

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

  // If the argument is an explicit !lit.ref type and the argument value is an
  // MValue, then we allow matching it to its underlying element type,
  // addrspace, mutability, lifetime etc.
  //
  // This is magic used by Reference.__init__, allowing Reference(someMValue)
  // to infer the lifetime and mutability of the MValue.
  if (auto expectedRef = dyn_cast<RefType>(expectedType.mlirType)) {
    if (expectedConvention == ArgConvention::BorrowedInReg &&
        !isa<RefType>(argType) && value.isMValue()) {
      auto valueRefType = cast<RefType>(value.getMValueReference().getType());
      // If the MValue is an MBValue specifically, make sure to strip off
      // any mutability from the reference.  The parser allows the IR
      // representation of an MBValue to be mutable, but we don't want to
      // infer mutability of a reference from that.
      if (value.getIfMBValue() && !valueRefType.isMutableKnown(false))
        valueRefType = valueRefType.getWithMutability(false);

      if (succeeded(matchTypes(valueRefType, expectedRef)))
        return success();

      // If that didn't work out, keep going, but with the original
      // diagnostics.
      diags.resetDiags(std::move(savedDiags));
      savedDiags = diags.saveDiags();
    }
  }

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
};

/// Given a signature type that has some of its parameter bindings known, burn
/// the values for those parameters in, leaving the rest untouched so we can
/// infer them.
static LITSignatureType
getPartiallySpecializedSignature(LITSignatureType signature,
                                 ArrayRef<TypedAttr> bindingsSoFar,
                                 ParserParamEvaluator &evaluator) {
  if (bindingsSoFar.empty())
    return signature;

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

  auto newSignature = substitutor.replace(signature);
  if (newSignature == signature)
    return signature;

  // If we changed something, then we substituted constants into the type tree.
  // This can cause some expressions to fold with the interpreter, so see if we
  // can simplify the result.
  return cast<LITSignatureType>(evaluator.refineType(newSignature));
}

/// Given an incomplete parameter binding set for a call to the specified
/// signature, try to infer the value of the next 'decl' parameter.  This
/// should always return null /without/ an error if it cannot be inferred, and
/// return a specific value if unambiguously determined.
LogicalResult
ParameterInferenceState::infer(LITSignatureType signature,
                               const CallOperands &callOperands,
                               const KeywordOperands &variadicKwOperands) {
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  size_t numPosOperands = posOperands.size();

  // Apply the bindings so far (plus a distinct new attribute relating
  // back to the original decls for ones that are missing) to the signature with
  // getSpecializedSignature so we benefit from the already-fixed substitutions
  // being applied to the input types.  This can make them more concrete and
  // help with inferring dependent types based on already-bound parameters.
  signature =
      getPartiallySpecializedSignature(signature, inferredParams, evaluator);
  size_t numAlreadySpecialized = inferredParams.size();

  // Match up the operands provided by the call to the input arguments.  Keep in
  // mind that the callee signature might not match at all, so we have to be
  // careful here!
  size_t posOperandIdx = 0;
  DefaultValueHandler defaultHandler(signature.getArgListAttrs());
  for (auto [expectedArgIdx, expectedConvention] :
       llvm::enumerate(signature.getArgConventions())) {

    // There is no provided operand for a by-ref result.
    if (SignatureType::isResultSlot(expectedConvention))
      continue;

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

      signature = getPartiallySpecializedSignature(signature, effectiveBindings,
                                                   evaluator);
      defaultHandler = DefaultValueHandler(signature.getArgListAttrs());
      numAlreadySpecialized = effectiveBindings.size();
    }

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
      SyntheticNode node(shared.getTopLevelDecl().getLoc());
      while (posOperandIdx != numPosOperands) {
        ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx++];
        ASTType toPush = getRValueTypeIfResolvable(operand.ir);
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

//===----------------------------------------------------------------------===//
// Diagnostic emission implementation
//===----------------------------------------------------------------------===//

namespace {
/// Helper class to emit errors without cluttering the evaluation logic.
struct DiagEmitter : public SharedStateUser {
  DiagEmitter(SharedState &shared, SMLoc callLoc, size_t numOperands,
              CallSyntax callSyntax)
      : SharedStateUser(shared), callLoc(callLoc), numOperands(numOperands),
        callSyntax(callSyntax) {}

  InflightDiag unexpectedKwArgs(ArrayRef<StringAttr> unknownKwOperands) const;
  InflightDiag wrongParamType(const ParamBindings::Binding &actualBinding,
                              size_t paramIdx, ASTType expectedType) const;
  InflightDiag wrongParamCount(size_t expectedNumParams,
                               size_t actualNumParams) const;
  InflightDiag wrongArgCountWithPack(size_t minRequiredArgs,
                                     size_t maxAllowedArgs,
                                     size_t numOperands) const;
  InflightDiag wrongPosOnlyCount(size_t minRequiredArgs, size_t numOperands,
                                 const Twine &argOrParam) const;
  InflightDiag resultGenericMemType(Type outputType) const;
  InflightDiag argGenericMemType(size_t expectedArgIdx,
                                 Type expectedType) const;
  InflightDiag argTypeMismatch(OverloadFitness::ArgTypeMismatchKind kind,
                               ASTType ty, ASTExprAnd<AnyValue> operand,
                               size_t argIdx) const;
  InflightDiag missingArgs(ArrayRef<StringAttr> missingArgs,
                           const Twine &kindStr) const;
  InflightDiag posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const;
  InflightDiag tooManyPosArgs(size_t maxAllowedArgs,
                              size_t numPosOperands) const;
  InflightDiag byPosAndKw(ArrayRef<StringAttr> names) const;

private:
  SMLoc callLoc;
  size_t numOperands;
  CallSyntax callSyntax;

  /// Wrapper around pretty printing logic for an argument given by index.
  void describeArgumentNo(InflightDiag &diag, size_t argIdx) const;

  InflightDiag initDiag() const { return shared.emitError(callLoc); }
};
} // namespace

void DiagEmitter::describeArgumentNo(InflightDiag &diag, size_t argIdx) const {
  // If this is a method syntax call, don't count the receiver.
  if (callSyntax == CallSyntax::kMethodCall ||
      callSyntax == CallSyntax::kMethodCallSynthetic) {
    // It is probably possible for this assert to fire, if it does we should
    // tailor the error message.
    assert(argIdx != 0 && "TODO: unexpected self mismatch");
    diag << "method argument #" << (argIdx - 1);
  } else if (callSyntax == CallSyntax::kOperator && argIdx == 1) {
    diag << "right side";
  } else if (callSyntax == CallSyntax::kReversedOperator && argIdx == 0) {
    diag << "left side";
  } else if (callSyntax == CallSyntax::kSubscript && argIdx != 0) {
    if (argIdx == 1 && numOperands == 2)
      diag << "index";
    else
      diag << "index #" << (argIdx - 1);
  } else if (callSyntax == CallSyntax::kAttribute && argIdx != 0) {
    diag << "attribute name";
  } else {
    diag << "argument #" << argIdx;
  }
}

InflightDiag
DiagEmitter::unexpectedKwArgs(ArrayRef<StringAttr> unknownKwOperands) const {
  InflightDiag diag = initDiag();
  emitUnknownKeywords(diag, unknownKwOperands, "argument");
  return diag;
}

InflightDiag
DiagEmitter::wrongParamType(const ParamBindings::Binding &actualBinding,
                            size_t paramIdx, ASTType expectedType) const {
  return initDiag() << "callee parameter #" << paramIdx << " has "
                    << ASTType(expectedType) << " type, but value has type "
                    << ASTType(actualBinding.getType())
                    << actualBinding.expr->getRange();
}

InflightDiag DiagEmitter::wrongParamCount(size_t expectedNumParams,
                                          size_t actualNumParams) const {
  InflightDiag diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, /*minRequired=*/expectedNumParams,
                           /*maxAllowed=*/expectedNumParams, actualNumParams,
                           "parameter");
  return diag;
}

InflightDiag DiagEmitter::wrongArgCountWithPack(size_t minRequiredArgs,
                                                size_t maxAllowedArgs,
                                                size_t numOperands) const {
  InflightDiag diag = initDiag()
                      << "callee with non-empty variadic pack argument";
  emitWrongArgOrParamCount(diag, minRequiredArgs, maxAllowedArgs, numOperands,
                           "positional operand");
  return diag;
}

InflightDiag DiagEmitter::wrongPosOnlyCount(size_t minRequiredArgs,
                                            size_t numOperands,
                                            const Twine &argOrParam) const {
  InflightDiag diag = initDiag() << "callee";
  emitWrongArgOrParamCount(diag, minRequiredArgs,
                           /*maxAllowed=*/numOperands, numOperands,
                           "positional " + argOrParam);
  return diag;
}

InflightDiag DiagEmitter::resultGenericMemType(Type outputType) const {
  return initDiag() << "result cannot bind AnyRegType type to memory-only type "
                    << outputType;
}

InflightDiag DiagEmitter::argGenericMemType(size_t expectedArgIdx,
                                            Type expectedType) const {
  InflightDiag diag = initDiag();
  describeArgumentNo(diag, expectedArgIdx);
  return std::move(diag) << " cannot bind AnyRegType type to memory-only type "
                         << expectedType;
}

/// Attach extra type conversion error detail or hints to the user when
/// reporting an error passing `operand` to an argument of type `argType`.
static void addTypeConversionDetail(InflightDiag &diag,
                                    ASTExprAnd<AnyValue> operand,
                                    ASTType argType, SharedState &shared) {
  auto loc = operand.expr->getLoc();
  ASTType operandType = getRValueTypeIfResolvable(operand.ir);
  if (!operandType) {
    diag.attachNote(loc) << "try resolving the overloaded function first";
    return;
  }
  // Try to detect mismatched byref result type.
  auto lhsSig = dyn_cast<SignatureType>(operandType);
  auto rhsSig = dyn_cast<SignatureType>(argType);
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
    diag.attachNote(loc) << "memory-only type bound to generic result type: "
                         << (lhsByRef ? "payload" : "argument") << " returns "
                         << ASTType(lhsRetType) << " by reference";
    return;
  }
}

/// Emit a tailored diagnostic when failing to convert a value to type !lit.ref.
/// This happens when the user is forming a Reference incorrectly which happens
/// when confusion and details run the highest.
static void diagnoseFailedRefTypeConversion(InflightDiag &diag,
                                            ASTExprAnd<AnyValue> operand,
                                            RefType argType,
                                            SharedState &shared) {
  diag << "'Reference[" << ASTType(argType.getElementType()) << ", ...]";

  auto loc = operand.expr->getLoc();
  if (operand.ir.getIfRValue()) {
    diag.attachNote(loc) << "cannot bind an RValue to a reference";
    return;
  }
  if (!operand.ir.isMValue()) {
    diag.attachNote(loc) << "operand does not have a memory representation";
    return;
  }

  auto operandRefTy = cast<RefType>(operand.ir.getMValueReference().getType());
  if (!ASTType(argType.getElementType())
           .isEqualCanon(operandRefTy.getElementType())) {
    diag.attachNote(loc) << "operand element type "
                         << ASTType(operandRefTy.getElementType())
                         << " doesn't match expected element type "
                         << ASTType(argType.getElementType());
  } else if (argType.getAddressSpace() != operandRefTy.getAddressSpace()) {
    diag.attachNote(loc) << "operand address space "
                         << operandRefTy.getAddressSpace()
                         << " doesn't match expected address space "
                         << argType.getAddressSpace();
  } else if (!canConvertWithRebind(operandRefTy.getLifetimeType(),
                                   argType.getLifetimeType(), shared)) {
    diag.attachNote(loc) << "operand mutability " << operandRefTy.isMutable()
                         << " doesn't match expected mutability "
                         << argType.isMutable();
  } else if (!canConvertWithRebind(operandRefTy, argType, shared)) {
    diag.attachNote(loc) << "operand lifetime " << operandRefTy.getLifetime()
                         << " doesn't match expected lifetime "
                         << argType.getLifetime();
  }
}

InflightDiag
DiagEmitter::argTypeMismatch(OverloadFitness::ArgTypeMismatchKind kind,
                             ASTType ty, ASTExprAnd<AnyValue> operand,
                             size_t argIdx) const {
  using ArgTypeMismatchKind = OverloadFitness::ArgTypeMismatchKind;
  InflightDiag diag = initDiag();
  switch (kind) {
  case ArgTypeMismatchKind::kNotLValue:
    if ((callSyntax == CallSyntax::kMethodCall ||
         callSyntax == CallSyntax::kMethodCallSynthetic) &&
        argIdx == 0) {
      diag << "invalid use of mutating method on rvalue of type ";
      if (ASTType type = getRValueTypeIfResolvable(operand.ir))
        diag << type;
      else if (operand.ir.getIfInitializer())
        diag << "initializer list";
      else
        diag << "unknown overload";
    } else {
      describeArgumentNo(diag, argIdx);
      diag << " must be mutable in order to pass as a by-ref argument";
    }
    diag << operand.expr->getRange();
    return diag;
  case ArgTypeMismatchKind::kWrongLVType:
    return std::move(diag) << "l-value of type "
                           << operand.ir.getIfLValue().getRValueType()
                           << " cannot be converted to reference of type "
                           << ty.getReferenceElementType()
                           << operand.expr->getRange();
  case ArgTypeMismatchKind::kWrongType: {
    describeArgumentNo(diag, argIdx);
    diag << " cannot be converted from " << operand.expr->getRange();
    ASTType rValueType = getRValueTypeIfResolvable(operand.ir);
    bool isConvertingTypeValue = ty.getMetaType() == rValueType;
    if (rValueType) {
      if (isConvertingTypeValue)
        diag << "type value " << ty;
      else
        diag << rValueType;
    } else if (operand.ir.getIfInitializer()) {
      diag << "initializer list";
    } else {
      diag << "unknown overload";
    }
    diag << " to ";

    if (auto refType = dyn_cast<RefType>(ty)) {
      diagnoseFailedRefTypeConversion(diag, operand, refType, shared);
      return diag;
    }

    diag << (isConvertingTypeValue ? "an instance of " : "") << ty;
    if (isConvertingTypeValue)
      diag << "; did you mean to instantiate " << ty << "?";
    addTypeConversionDetail(diag, operand, ty, shared);
    return diag;
  }
  default:
    llvm_unreachable("unexpected ArgTypeMismatchKind");
  }
}

InflightDiag DiagEmitter::missingArgs(ArrayRef<StringAttr> missingArgs,
                                      const Twine &kindStr) const {
  InflightDiag diag = initDiag();
  emitMissing(diag, missingArgs, kindStr + " argument");
  return diag;
}

InflightDiag
DiagEmitter::posOnlyPassedByKw(ArrayRef<StringAttr> posOnlyPassedByKw) const {
  InflightDiag diag = initDiag();
  emitPosOnlyPassedByKw(diag, posOnlyPassedByKw, "argument");
  return diag;
}

InflightDiag DiagEmitter::tooManyPosArgs(size_t maxAllowedArgs,
                                         size_t numPosOperands) const {
  InflightDiag diag = initDiag();
  emitTooManyPositional(diag, maxAllowedArgs, numPosOperands, "argument");
  return diag;
}

InflightDiag DiagEmitter::byPosAndKw(ArrayRef<StringAttr> names) const {
  InflightDiag diag = initDiag();
  emitByPosAndKw(diag, names, "argument");
  return diag;
}

//===----------------------------------------------------------------------===//
// OverloadFitness
//===----------------------------------------------------------------------===//

/// Calculate the minimum required and maximum allowed number of positional
/// operands for a signature, assuming that the signature has a variadic pack;
static std::pair<size_t, size_t>
calculateRequiredPosOperandsForPacks(LITSignatureType signature) {
  // This function heavily assumes that a signature has at most
  // one pack variadic argument and that variadics are always the last
  // positional args.
  size_t numPosArgs = countNumPositional(signature.getArgListAttrs());

  // We don't require any positional operands (because this function does not
  // check for passing kinds).
  if (!numPosArgs)
    return {0, numPosArgs};

  // If we have a variadic argument, it will consume all positional operands,
  // but it does not require any.
  size_t lastPosIdx = numPosArgs - 1;
  if (signature.isPosVarArg(lastPosIdx))
    return {0, std::numeric_limits<size_t>::max()};

  // If we have a non-empty variadic pack argument, we do require a certain
  // number of positional operands (since the value of positional packs cannot
  // be provided by keyword operands).
  // NOTE: in this case, it doesn't matter if there are preceding positional
  // arguments with default values: the pack cannot have a default value and
  // _must_ be provided positional operands explicitly, and therefore the
  // preceding defaults won't be used anyway.
  if (ASTType variadicPackType = signature.getIfVariadicPack(lastPosIdx)) {
    RefPackType packType = variadicPackType.getVariadicPackInfo();
    VariadicAttr packed = packType.getVariadicIfResolved();
    // The caller should know the concrete type list unless we binded the pack
    // directly as a parameter.  This is an unpack like situation.
    // TODO: This happens in error cases and needs to be re-evaluated.
    if (!packed)
      return {0, numPosArgs - 1};

    // NOTE: we adjust the number of user declared pos args since that
    // includes the pack itself (hence the "-1").
    size_t packSize = packed.getValues().size();
    return {numPosArgs - 1 + packSize, numPosArgs - 1 + packSize};
  }

  return {0, numPosArgs};
}

/// Check the expected type against the provided operand. This identifies any
/// problems with the operand type and also returns the type to be used for
/// error propagation.
///
/// This ties into parameter inference, but is only called on the top level
/// function operands being matched up, not anything in recursive functiontype
/// positions.
std::pair<OverloadFitness::ArgTypeMismatchKind, ASTType>
OverloadFitness::checkOneOperand(ASTExprAnd<AnyValue> operand,
                                 ArgConvention expectedConvention,
                                 ASTType expectedType,
                                 size_t &numImplicitConversions,
                                 size_t &numMismatchedConventions,
                                 bool &hasNonmaterializableConversion,
                                 bool allowImplicitConversions, SMLoc loc,
                                 ASTDecl &declScope, SharedState &shared) {
  switch (expectedConvention) {
  case ArgConvention::InitSelf:
    // If this is an UnknownAttr, then it is a placeholder for type
    // checking, just let it pass.
    if (auto pValue = operand.ir.getIfPValue())
      if (isa<UnknownAttr>(pValue.get()))
        return {kValidType, expectedType};
    [[fallthrough]];
  case ArgConvention::ByRef:
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError: {
    // The actual value must be an lvalue if callee takes things by-ref.
    auto argVal = operand.ir.getIfLValue();
    if (!argVal)
      return {kNotLValue, expectedType};

    // By-ref argument types must exactly match, no conversions are allowed.
    ASTType elementType = expectedType.getReferenceElementType();
    if (!argVal.getRValueType().isEqualCanon(elementType))
      return {kWrongLVType, expectedType};
    // If a register-passable type is being returned in-memory, remember this.
    numMismatchedConventions += elementType.isRegisterPassable(loc, shared);
    return {kValidType, expectedType};
  }
  case ArgConvention::BorrowedInMem:
  case ArgConvention::OwnedInMem:
    // Ignore the pointer type on memory conventions when matching types.
    // Note: We do not support overloading on borrow/owned currently,
    // but we could add this if there is a reason to.
    expectedType = expectedType.getReferenceElementType();
    // If a register-passable type is being passed in-memory, remember this.
    numMismatchedConventions += expectedType.isRegisterPassable(loc, shared);
    [[fallthrough]];
  case ArgConvention::BorrowedInReg:
  case ArgConvention::OwnedInReg: {
    // Get the argument if it has a concrete type.
    CValue argVal = operand.ir.getIfCValue();

    // If the argument is unresolved, see if we can resolve it with the expected
    // type.
    if (!argVal) {
      if (auto initValue = operand.ir.getIfInitializer()) {
        // Initializer lists are good if we can construct the expected type.
        ExprEmitter emitter(shared, declScope, ExprContext::EC_CallArgValue);
        auto [initFn, erroneousDecl] = emitter.canConstructType(
            expectedType, initValue.get(), operand.expr);
        // If there were declaration errors, assume construction is possible to
        // avoid spurious errors.
        bool valid = (bool)initFn || erroneousDecl;
        // If so, all is good, if not, we fail.
        return {valid ? kValidType : kWrongType, expectedType};
      }

      auto orValue = operand.ir.getIfOverloadSet();
      assert(orValue && "Unknown UValue!");

      // Try to refine the OverloadSetUValue into a PValue.
      argVal = orValue->getDirectSymbol(expectedType);
      if (!argVal)
        return {kWrongType, expectedType};

      // If we have a reference to an overloaded method like foo(a.method),
      // then we can't resolve it.
      // TODO(partial application => closures): Given we just resolved argVal,
      // we could form the "a.method" expression with a closure.
      if (orValue->baseValue) // Cannot merge base value.
        return {kWrongType, expectedType};
    }

    ASTType argType = argVal.getRValueType();
    // Otherwise, we pass as an r-value.  If the argument types match, then
    // they are good.
    if (argType.isEqualCanon(expectedType))
      return {kValidType, expectedType};

    if (auto nonmaterializableTarget =
            argType.getNonmaterializableTarget(shared)) {
      if (nonmaterializableTarget.isEqualCanon(expectedType)) {
        // Implicit conversion for nonmaterializable types to their target
        // type is allowed even if !allowImplicitConversions.  Even though
        // this may be an implicit conversion, don't increment the
        // numImplicitConversions count so that it will win against other
        // implicit conversions.  However, we keep track of whether
        // nonmaterializable autoconversion has happened so that functions
        // that literally take the nonmaterializable type can still win
        // instead of autoconverting if their signature matches exactly.
        hasNonmaterializableConversion = true;
        return {kValidType, expectedType};
      }
    }

    // Argument name mismatches don't count as implicit conversions.
    if (canConvertWithRebind(argType, expectedType, shared))
      return {kValidType, expectedType};

    // If implicit conversions are possible and one will work, then we succeed
    // with that conversion.
    if (allowImplicitConversions &&
        ExprEmitter(shared, declScope, ExprContext::EC_CallArgValue)
            .canImplicitlyConvertToType({argVal, operand.expr}, expectedType)) {
      // If we had one, this bumps our # implicit conversions.
      ++numImplicitConversions;
      return {kValidType, expectedType};
    }

    // If this is a low-level !lit.ref passed by value, we support binding an
    // MValue of the element type.  Parameter inference will infer the lifetime
    // and mutability of the reference in the common case that they are params.
    //
    // This is magic used by Reference.__init__, allowing Reference(someMValue)
    // to infer the lifetime and mutability of the MValue.
    if (auto expectedRef = dyn_cast<RefType>(expectedType)) {
      // Element type and address have to be exactly equal, the mutability just
      // has to be compatible.
      if (ASTType(argType).isEqualCanon(expectedRef.getElementType()) &&
          argVal.isMValue()) {
        auto argRefType = cast<RefType>(argVal.getMValueReference().getType());
        if (canConvertWithRebind(argRefType, expectedRef, shared))
          return {kValidType, expectedType};
      }
    }

    // Otherwise this is the wrong type for the argument.
    return {kWrongType, expectedType};
  }
  case ArgConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }

  llvm_unreachable("unknown case");
};

bool OverloadFitness::isBetter(const OverloadFitness &other) const {
  // We first compare the number of implicit conversions.
  size_t numConversions = getNumImplicitConversions();
  size_t otherNumConversions = other.getNumImplicitConversions();
  if (numConversions != otherNumConversions)
    return numConversions < otherNumConversions;

  // If ambiguous, we compare the boolean metrics.
  int8_t mask = payload.getBoolMask();
  int8_t otherMask = other.payload.getBoolMask();
  if (mask != otherMask)
    return mask < otherMask;

  // If still ambiguous, we compare the number of bindings.
  if (paramBindings.size() != other.paramBindings.size())
    return paramBindings.size() < other.paramBindings.size();

  // Otherwise these candidates are almost identical, so we try to decide based
  // on the number of input conventions mismatches (e.g. register-passable
  // passed in memory).
  return payload.numMismatchedConventions <
         other.payload.numMismatchedConventions;
}

int8_t OverloadFitness::Payload::getBoolMask() const {
  // We consider exact matches of concrete types to be more specific than
  // those needing non-materializable conversions, both of these more
  // specific than varargs matches (for example, when overloading a
  // `foo(Int)` and `foo(Int*)` we should pick the former if both work), and
  // all of these more specific than matches with variadic parameters.
  return 4 * hasNonmaterializableConversion + 2 * passesVarArgArgument +
         1 * hasVariadicParams;
}

OverloadFitness OverloadFitness::evaluate(LITSignatureType signature,
                                          ASTDecl *funcIfDirect,
                                          const OverloadSet &callable,
                                          const CallOperands &callOperands,
                                          bool allowImplicitConversions) {
  // We set up diagnostics.
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  size_t numPosOperands = posOperands.size();
  size_t numOperands = numPosOperands + callOperands.getNumKwOperands();
  SMLoc callLoc = callable.expr->getLoc();
  SharedState &shared = callable.getShared();
  DiagEmitter emitDiagFor(shared, callLoc, numOperands, callable.syntax);

  // If a variadic keyword arg is expected, we collect the unknown kw operands.
  KeywordOperands variadicKwOperands;
  auto [kwDiagRes, kwDiagNames] = diagnoseKeywordOperands(
      signature.getArgListAttrs(), variadicKwOperands, callOperands);
  switch (kwDiagRes) {
  case KwDiagResult::kMissingKwOnly:
    return emitDiagFor.missingArgs(kwDiagNames, "keyword-only");
  case KwDiagResult::kPosOnlyPassedByKw:
    return emitDiagFor.posOnlyPassedByKw(kwDiagNames);
  case KwDiagResult::kUnknownKeywords:
    return emitDiagFor.unexpectedKwArgs(kwDiagNames);
  default:
    break;
  }

  PogListAttr argListAttr = signature.getArgListAttrs();
  auto [posDiagRes, posDiagNames] =
      diagnosePosOperands(argListAttr, callOperands);
  switch (posDiagRes) {
  case PosDiagResult::kMissingPos:
    return emitDiagFor.missingArgs(posDiagNames, "positional");
  case PosDiagResult::kTooManyPos: {
    size_t numPosMaximum = countNumPositional(argListAttr);
    return emitDiagFor.tooManyPosArgs(numPosMaximum, numPosOperands);
  }
  case PosDiagResult::kByPosAndKw:
    return emitDiagFor.byPosAndKw(posDiagNames);
  default:
    break;
  }

  // Check that the signature can be rebound with this set of bindings. We use
  // diagnostic handlers to capture any issues.
  InflightDiag diag = shared.emitError(callLoc);
  ParameterInferenceDiagnostics inferenceDiags;
  ParamBindings::DiagEmitter bindingDiag{
      /*emitParamCount=*/
      [&](size_t numActual, bool posOnly) {
        PogListAttr paramListAttr = signature.getParamListAttrs();
        if (posOnly) {
          size_t numPosOnly = countNumPosOnly(paramListAttr);
          diag =
              emitDiagFor.wrongPosOnlyCount(numPosOnly, numActual, "parameter");
        } else {
          // Hide the implicit trait parameter from the diagnostic.
          size_t hidden = 0;
          if (funcIfDirect &&
              isa<TraitDeclOp>(cast<LIT::FuncOp>(*funcIfDirect)->getParentOp()))
            hidden = 1;
          size_t numExpected = signature.getNumParams() - hidden -
                               countNumImplicitKinds(paramListAttr);
          diag = emitDiagFor.wrongParamCount(numExpected, numActual - hidden);
        }
        // For each of the missing parameters, attach any parameter inference
        // diagnostics.
        inferenceDiags.attach(signature, diag, numActual, callOperands);
      },
      /*emitPosType=*/
      [&](size_t paramIdx, const ParamBindings::Binding &binding,
          ASTType expectedType) {
        diag = emitDiagFor.wrongParamType(binding, paramIdx, expectedType);
      },
      /*emitKwType=*/
      [&](StringAttr paramName, const ParamBindings::Binding &binding,
          ASTType expectedType) {
        diag << "callee parameter " << paramName << " has "
             << ASTType(expectedType) << " type, but value has type "
             << ASTType(binding.getType()) << binding.expr->getRange();
      },
      /*emitUnknownKeywords=*/
      [&](ArrayRef<StringAttr> unknownKeywords) {
        emitUnknownKeywords(diag, unknownKeywords, "parameter");
      },
      /*emitRedundantKeywords=*/
      [&](ArrayRef<StringAttr> names) {
        emitByPosAndKw(diag, names, "parameter");
      },
      /*emitPosOnlyPassedByKw=*/
      [&](ArrayRef<StringAttr> names) {
        emitPosOnlyPassedByKw(diag, names, "parameter");
      },
      /*emitDeductionFailure=*/
      [&](size_t paramIdx) {
        auto emitMessage = [&](auto sig) {
          diag << "could not deduce ";
          if (StringAttr name = sig.getParamName(paramIdx); !name.empty())
            diag << "parameter " << name;
          else
            diag << nameForPosOnly(paramIdx, "parameter");
        };

        if (funcIfDirect) {
          if (auto structOp = dyn_cast<StructDeclOp>(
                  cast<LIT::FuncOp>(*funcIfDirect)->getParentOp())) {
            emitMessage(structOp.getSignature());
            diag << " of parent struct '" << structOp.getDeclName().getValue()
                 << "'";
            diag.attachNote(structOp.getLoc()) << " struct declared here";
            return;
          }
        }
        emitMessage(signature);
        diag << " of callee '" << callable.baseName << "'";
      },
      /*emitUnboundPackInVariadic=*/
      [&](const ParamBindings::Binding &binding) {
        diag << "unbound pack syntax (i.e. `*_`) cannot be used where variadic "
                "parameters are expected"
             << binding.expr->getRange();
        ;
      },
      /*emitUnboundPackNotEnd=*/
      [&](const ParamBindings::Binding &binding) {
        diag << "unbound pack must be at the end of the parameter list"
             << binding.expr->getRange();
      },
      /*emitInferOnlyFailure=*/
      [&](size_t paramIdx) {
        // Find the parameter number and potentially name of the type of the
        // argument that failed to be inferred.
        for (auto [idx, argType] : llvm::enumerate(signature.getArguments())) {
          if (auto type = dyn_cast<DeclRefType>(argType)) {
            for (auto [i, value] : llvm::enumerate(type.getParamValues())) {
              if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value);
                  indexRef && !indexRef.getDepth() &&
                  indexRef.getIndex() == paramIdx) {
                diag << "failed to infer implicit parameter ";
                auto structDecl =
                    cast<StructDeclOp>(ASTType(type).getDecl(shared));
                printNameOrIdx(structDecl.getSignature().getParamName(i), i,
                               diag);
                diag << " of argument ";
                printNameOrIdx(signature.getArgName(idx), idx, diag);
                diag << " type '" << structDecl.getSymName() << "'";
                return;
              }
            }
          }
        }
      },
      /*emitMissing=*/
      [&](ArrayRef<StringAttr> names, const Twine &kindStr) {
        emitMissing(diag, names, kindStr + " parameter");
      },
      /*emitTooManyPositional=*/
      [&](size_t numMaxAllowed, size_t numActual) {
        emitTooManyPositional(diag, numMaxAllowed, numActual, "parameter");
      },
  };

  auto parameterInferenceHook = [&](ArrayRef<TypedAttr> bindingsSoFar,
                                    const ParserParamEvaluator &evaluator) {
    // Infer information from this signature holistically.
    ParameterInferenceState inferrence(callable.paramBindings.declScope, shared,
                                       bindingsSoFar, evaluator, inferenceDiags,
                                       allowImplicitConversions);
    if (failed(inferrence.infer(signature, callOperands, variadicKwOperands)))
      return PValue();

    // See if we inferred information about the next value.
    if (auto result = inferrence.getInferredValue(bindingsSoFar.size()))
      return PValue(result);

    // If we succeeded inference but didn't get a value for this parameter, then
    // the parameter must not be present: complain.
    inferenceDiags.addFailure(bindingsSoFar.size(), callable.expr,
                              NotFoundFailure());
    return PValue();
  };
  auto [newBindings, bindingFitness] = callable.paramBindings.verifyBindings(
      signature, bindingDiag, parameterInferenceHook);

  // If there is an error, we just forward the diagnostics.
  if (!newBindings)
    return std::move(diag);
  diag.abandon();

  // If anything was bound, apply it to the signature so the expected argument
  // types are updated.
  std::tie(signature, newBindings) =
      getUnboundSpecializedSignature(signature, newBindings);

  // Check that the result didn't bind to a type that would require changing to
  // a different result convention.
  for (Type outputType : signature.getResults())
    if (!ASTType(outputType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.resultGenericMemType(outputType);

  // Binding the parameters would determine the type of pack varargs. Given
  // this, we need to check again if we have missing or too many arguments.
  auto [minPosOperands, maxPosOperands] =
      calculateRequiredPosOperandsForPacks(signature);
  if (numPosOperands < minPosOperands || maxPosOperands < numPosOperands) {
    return emitDiagFor.wrongArgCountWithPack(minPosOperands, maxPosOperands,
                                             numPosOperands);
  }

  SMLoc loc = callable.expr->getLoc();

  // We will accumulate the implicit conversion in arguments to those counted
  // for the parameter bindings.
  size_t numImplicitConversions = bindingFitness.numImplicitConversions;
  bool hasNonmaterializableConversion = false;

  // As we walk through the values provided as part of the argument list, we
  // match them up against arguments expected by the signature of the callee,
  // take note if variadic arguments are passed, and accumulate implicit
  // conversions required for a match.
  size_t posOperandIdx = 0;
  bool passesVarArgArgument = false;

  // For each mismatch in "preferred" argument convention, penalize the
  // overload. This is to resolve ambiguities that can arise from synthesized
  // thunks for converting calling conventions.
  size_t numMismatchedConventions = 0;

  auto checkAnOperand = [&](ASTExprAnd<AnyValue> operand,
                            ArgConvention expectedConvention,
                            ASTType expectedType) {
    return checkOneOperand(operand, expectedConvention, expectedType,
                           numImplicitConversions, numMismatchedConventions,
                           hasNonmaterializableConversion,
                           allowImplicitConversions, loc,
                           callable.paramBindings.declScope, shared);
  };

  // Use a ParserParamEvaluator to substitute 'apply' expressions in the
  // argument types.
  ParserParamEvaluator evaluator(*shared.declResolver);
  argListAttr = signature.getArgListAttrs();
  DefaultValueHandler defaultHandler(argListAttr);
  for (auto [expectedArgIdx, unboundExpectedType, expectedConvention] :
       llvm::enumerate(signature.getArguments(),
                       signature.getArgConventions())) {
    // Ignore the return slot if present.
    Type expectedType = evaluator.refineType(unboundExpectedType);
    if (expectedConvention == ArgConvention::ByRefError)
      continue;
    if (expectedConvention == ArgConvention::ByRefResult) {
      numMismatchedConventions += ASTType(expectedType)
                                      .getReferenceElementType()
                                      .isRegisterPassable(loc, shared);
      continue;
    }

    if (signature.isKwVarArg(expectedArgIdx)) {
      expectedType = ASTType(expectedType).getKwargsDictRefValueType();

      for (auto [name, operand] : variadicKwOperands) {
        // TODO: Passing OwnedInReg is a hack that is needed because the value
        // type is not a reference type (and doesn't have a lifetime), but we
        // still want to type check it. So, passing it as if it was reg-passable
        // happens to just work, until we rectify this. Right now the reason the
        // value type cannot be a reference type is because `Reference` does not
        // (and in fact cannot) conform to `CollectionElement`.
        auto [kind, ty] =
            checkAnOperand(operand, ArgConvention::OwnedInReg, expectedType);
        if (kind != kValidType)
          return emitDiagFor.argTypeMismatch(kind, ty, operand, expectedArgIdx);
      }
      continue;
    }

    // If the arguments or results got bound to a memory-only type then their
    // argument convention needs to change.  We cannot support this until we get
    // proper type traits.
    // TODO: Don't let memory types bind to AnyRegType.
    if (!ASTType(expectedType).isRegisterPassable(callLoc, shared))
      return emitDiagFor.argGenericMemType(expectedArgIdx, expectedType);

    // Handle case when there are no more provided positional operands.
    StringAttr argName = argListAttr.getName(expectedArgIdx);
    if (posOperandIdx == numPosOperands) {
      // If the argument is a varargs argument list or pack, then it can be
      // initialized with zero values no problem.
      if (signature.isPosVarArg(expectedArgIdx) ||
          signature.isPackVarArg(expectedArgIdx)) {
        // We consider an empty varargs list to be an implicit conversion,
        // so an exact signature match takes precedence.
        ++numImplicitConversions;
        continue;
      }

      // Check if the argument was passed as a keyword operand.
      if (std::optional<ASTExprAnd<AnyValue>> kwOperandOr =
              callOperands.findKwArg(argName)) {
        // If we found a keyword argument, we check it normally.
        auto [kind, ty] =
            checkAnOperand(*kwOperandOr, expectedConvention, expectedType);
        if (kind != kValidType) {
          return emitDiagFor.argTypeMismatch(kind, ty, *kwOperandOr,
                                             expectedArgIdx);
        }
        continue;
      }

      // We ensured earlier that that can be no missing positional arguments.
      assert(defaultHandler.getDefault(expectedArgIdx) &&
             "missing positional argument not caught by diagnostics");

      continue;
    }

    /// Check and process a single positional operand and advance the operand
    /// index.
    auto processPositionalOperand =
        [&](ASTType expectedType,
            ArgConvention conv) -> std::optional<InflightDiag> {
      ASTExprAnd<AnyValue> operand = posOperands[posOperandIdx];
      auto [kind, ty] = checkAnOperand(operand, conv, expectedType);
      if (kind != kValidType)
        return emitDiagFor.argTypeMismatch(kind, ty, operand, posOperandIdx);
      ++posOperandIdx;
      return std::nullopt;
    };

    // If we have a varargs argument, then it will eat the rest of the
    // positional arguments, but we have to check each of them.
    if (signature.isPosVarArg(expectedArgIdx)) {
      auto expectedVariadic = cast<VariadicType>(expectedType);
      auto varArgsEltType = expectedVariadic.getElementType();
      while (posOperandIdx != numPosOperands) {
        if (auto result = processPositionalOperand(
                varArgsEltType, expectedVariadic.getConvention()))
          return std::move(*result);
        passesVarArgArgument = true;
      }
      continue;
    }

    // If we have a pack type, it must have a known number of elements, and so
    // consume exactly that many positional operands.
    if (ASTType variadicPackType =
            signature.getIfVariadicPack(expectedArgIdx)) {
      auto actualArgConvention =
          signature.getPackVarArgConvention(expectedArgIdx);
      RefPackType packType = variadicPackType.getVariadicPackInfo();
      for (TypedAttr element : packType.getVariadicIfResolved().getValues()) {
        auto refType = packType.getElementRefTypeFor(ASTType(element).mlirType);
        if (auto result =
                processPositionalOperand(refType, actualArgConvention))
          return std::move(*result);
        passesVarArgArgument = true;
      }
      continue;
    }

    // Otherwise, we have an ordinary positional argument that is not varargs or
    // a pack. We ensured earlier that it is not also passed as a keyword
    // operand, so we process it as usual.
    assert(
        (argListAttr.getPassingKind(expectedArgIdx) == PassingKind::PosOnly ||
         (!argName.empty() && !callOperands.findKwArg(argName))) &&
        "redundant argument not caught by diagnostics");
    if (auto result =
            processPositionalOperand(expectedType, expectedConvention))
      return std::move(*result);
  }

  assert(posOperandIdx == numPosOperands &&
         "should handle argument mismatch above");

  // Otherwise we succeeded!
  return {newBindings,
          Payload{numImplicitConversions, numMismatchedConventions,
                  hasNonmaterializableConversion, passesVarArgArgument,
                  bindingFitness.hasVariadicParams}};
}
