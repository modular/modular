//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParamBindings.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/SharedState.h"
#include "MojoUtils.h"
#include "ParameterInference.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "Support/STLExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// If we're trying to call `foo.lork()`, like this:
///
///     fn callTraitMethodWithAliasArg[X: MyTrait](t: X, thing: MyStruct[X.T]):
///         t.lork(thing)
///
/// and lork happens to be a trait method with an alias, like:
///
///     trait MyTrait:
///         alias T: OtherTrait
///         fn lork(self, thing: MyStruct[T]): ...
///
/// Then we'll need to adjust our desired signature from:
///     fn lork(self, thing: MyStruct[T])
/// to:
///     fn lork(self, thing: MyStruct[get_vtable_entry(X, T)])
///
/// This function will do that conversion. If we aren't calling a trait method
/// with an alias, it'll return the given desiredSignature unmodified.
///
/// For more context, see
/// https://www.notion.so/modularai/verifyConformance-Arcana-13e1044d37bb80e88cb5c285a232784e?pvs=4#13e1044d37bb80bf8b42f3953af880f8
///
/// TODO(MOCO-1259): Support static methods with associated aliases
FnTypeGeneratorType LIT::substituteTraitAliasesIntoSignature(
    DeclResolver &declResolver, ASTDecl *traitDecl, FnOp candidateFunc,
    FnTypeGeneratorType desiredSignature, PValue selfPValue) {
  ParameterEvaluator traitAliasReplacer;
  for (auto &[name, decls] : traitDecl->getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      AliasDeclOp traitAlias = dyn_cast<LIT::AliasDeclOp>(*decl);
      if (!traitAlias)
        continue;
      StringAttr nameStringAttr = StringAttr::get(
          name.str(), StringType::get(candidateFunc->getContext()));
      TypedAttr aliasRef = ParamOperatorAttr::get(POC::GetVTableEntry,
                                                  {selfPValue, nameStringAttr},
                                                  traitAlias.getType());
      traitAliasReplacer.setParameterValue(traitAlias.getParamDecl(), aliasRef);
    }
  }
  return traitAliasReplacer.replace(desiredSignature);
}

//===----------------------------------------------------------------------===//
// ParamBindings
//===----------------------------------------------------------------------===//

ParamBindings::ParamBindings(ASTDecl &declScope)
    : declScope(declScope), shared(declScope.getShared()) {}

/// Replace our bindings with another set.  This can't be done with operator=
/// because we have
void ParamBindings::operator=(ParamBindings &&other) {
  parameters = std::move(other.parameters);
  defaultPosTypeParams = other.defaultPosTypeParams;
  defaultKwTypeParams = other.defaultKwTypeParams;
  ctadPogs = other.ctadPogs;
  numKwOnlyCtadParams = other.numKwOnlyCtadParams;
  numPosCtadParams = other.numPosCtadParams;
  numPreTypeChecked = other.numPreTypeChecked;
}

/// Create a (possibly partially unbound) set of bindings for the given type.
/// This can be used to initialize the binding set for methods. If the given
/// type is not a parametric user defined type, this returns empty bindings.
ParamBindings ParamBindings::getForDeclaredType(ASTDecl &declScope,
                                                ASTType type,
                                                const ExprNode *expr) {
  ParamBindings paramBindings(declScope);
  // TODO: this will not work with arbitrary parametric ancestors.
  // Default params need to come from the original declaration, instead of
  // TypeSignatureType, as the latter won't contain the full defaults list if
  // any have been bound already (when `type` is partially specified).
  ASTDecl *decl = type.getDecl(declScope.getShared());
  if (auto structDecl = dyn_cast_or_null<StructDeclOp>(decl)) {
    paramBindings.defaultPosTypeParams =
        structDecl.getSignature().getDefaultPosParams();
    paramBindings.defaultKwTypeParams =
        structDecl.getSignature().getDefaultKwOnlyParams();
    llvm::append_range(paramBindings.ctadPogs,
                       structDecl.getSignature().getParamListAttrs().getPogs());
    for (auto pog : paramBindings.ctadPogs) {
      if (pog.getPassingKind() == PassingKind::KwOnly)
        paramBindings.numKwOnlyCtadParams++;
      else
        paramBindings.numPosCtadParams++;
    }
  }

  // When binding a trait function, add the self type bindings.
  if (isa<TraitDeclOp>(decl)) {
    auto typeAttr = PValue(type).get();

    // The source value be something of trait type like Movable, or it may be
    // something of AnyTraitType type, like
    //   fn ex[Trait: MovableMetaType, T: Trait](argument: T):
    // where T is some type that is known to conform to Movable.  In the later
    // case we just know that the input type conforms to Movable, and we want to
    // look up members to bind in Movable, so bind the Trait type here.  If this
    // is a struct, or simple trait, keep it.
    if (auto paramType = dyn_cast<ParamType>(type.getMetaType())) {
      auto simpleTraitType =
          cast<AnyTraitType>(paramType.getParam().getType()).getTraitType();
      // Upcast from a parametric type of trait metatype value (e.g. "some
      // type that conforms to Movable) to the simple trait type (Movable)
      // so we can substitute the value into the signature.
      typeAttr =
          UpcastAttr::get(simpleTraitType, PValue(type),
                          VTableAttr::get(simpleTraitType.getContext(), {}));
    }
    paramBindings.addPrechecked(expr, typeAttr);
  } else if (isa<TraitType>(decl->getIfTypeValue())) {
    // If this is a trait composition, the method signature's self type won't
    // match directly (need to upcast the composition into the trait type that
    // declared the method). Add as _not_ prechecked.
    paramBindings.add(expr, PValue(type).get());
  }

  ArrayRef<TypedAttr> paramValues = type.getParamBindings();
  for (TypedAttr value : paramValues)
    paramBindings.addPrechecked(expr, value);
  return paramBindings;
}

void ParamBindings::addPrechecked(const ExprNode *expr,
                                  TypedAttr precheckedBinding) {
  assert(numPreTypeChecked == parameters.size() &&
         "Cannot add type prechecked after other bindings!");
  parameters.add({precheckedBinding, expr});
  ++numPreTypeChecked;
}

void ParamBindings::add(const ExprNode *expr, TypedAttr value) {
  parameters.add({value, expr});
}

void ParamBindings::add(const ExprNode *expr, PValue value, StringAttr name) {
  parameters.add(name, {value, expr});
}

//===----------------------------------------------------------------------===//
// verifyBindings
//===----------------------------------------------------------------------===//

/// Check a single binding and emit a parameter value if possible. If an
/// implicit conversion is required, the provided counter is incremented.
static PValue emitSingleParameterValue(ASTExprAnd<AnyValue> binding,
                                       ASTType expectedType,
                                       size_t &numImplicitConversions,
                                       ExprEmitter &emitter,
                                       ParameterEvaluator &evaluator) {

  PValue bindingVal = binding.ir.getIfPValue();
  assert(bindingVal && "Parameters are always PValue's");

  // Parameters can only be unpacked into a variadic.
  // FIXME: This results in a poor error message.
  if (isa<UnpackedAttr>(bindingVal.get()))
    return {};

  // Check the type matches what is expected, and perform an implicit
  // conversion if needed.
  expectedType = ASTType(evaluator.getReboundType(expectedType.mlirType));

  // We don't typecheck the '_' magic parameter, we propagate it.
  if (isa<UnboundAttr>(bindingVal.get()))
    return PValue(UnboundAttr::get(expectedType));

  // If the parameter already has the right type, then we're good.
  if (expectedType.isEqualCanon(bindingVal.getType()))
    return bindingVal;

  // If the parameter can be implicitly converted, do so.
  if (ExprEmitter::canImplicitlyConvertToType(
          {bindingVal, binding.expr}, expectedType, emitter.getDeclScope())) {
    numImplicitConversions += 2;
    return emitter.emitPValue(binding, EC_CallParamValue, expectedType);
  }
  return {};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindingsImpl(
    const CallOperands &origOperands, ArrayRef<Type> expectedParamTypes,
    PogListAttr paramListAttr, ParameterInferenceHookTy parameterInferenceHook,
    const DiagEmitter *diagEmitter, bool partial) const {
  assert(parameterInferenceHook && "expected a parameter inference hook");
  Fitness fitness{0, false};

  // Check to see if we have *_ or **_ and filter them from the parameter list.
  bool unpackedPos = false;
  bool unpackedKw = false;
  CallOperands operands;
  for (auto [idx, binding] : llvm::enumerate(origOperands.values)) {
    auto unpacked = dyn_cast<UnpackedAttr>(binding.ir.getIfPValue().get());
    // Check if the unpacked value is an UnboundAttr.
    if (!unpacked || !isa<UnboundAttr>(unpacked.getValue())) {
      operands.values.push_back(binding);
      continue;
    }
    if (unpacked.getKwOnly()) {
      unpackedKw = true;
      // Verify that **_ is the last keyword parameter.
      if (idx != origOperands.values.size() - 1) {
        if (diagEmitter)
          diagEmitter->emitUnpackedNotAtEnd(binding.expr, /*kw=*/true);
        return {{}, fitness};
      }
    } else {
      // Verify that *_ is the last positionally-passed parameter.
      if (idx != origOperands.values.size() - 1 &&
          !origOperands.values[idx + 1].keyword) {
        if (diagEmitter)
          diagEmitter->emitUnpackedNotAtEnd(binding.expr, /*kw=*/false);
        return {{}, fitness};
      }
      unpackedPos = true;
    }
  }

  // With that out of the way, we can now get onto normal type checking of
  // 'operands'.
  size_t numParams = expectedParamTypes.size();

  OperandValueList variadicKwOperands;
  auto [kwDiagRes, kwDiagNames] = operands.diagnoseKeywordOperands(
      paramListAttr, variadicKwOperands, /*allowMissingKwOnly=*/true);
  if (kwDiagRes != CallOperands::KwDiagResult::kValid) {
    switch (kwDiagRes) {
    case CallOperands::KwDiagResult::kMissingKwOnly:
      if (diagEmitter)
        diagEmitter->emitMissing(kwDiagNames, "keyword-only");
      break;
    case CallOperands::KwDiagResult::kOutOfOrderInferredKw:
      if (diagEmitter)
        diagEmitter->emitOutOfOrderInferredKw(kwDiagNames);
      break;
    case CallOperands::KwDiagResult::kPosOnlyPassedByKw:
      if (diagEmitter)
        diagEmitter->emitPosOnlyPassedByKw(kwDiagNames);
      break;
    case CallOperands::KwDiagResult::kUnknownKeywords:
      if (diagEmitter)
        diagEmitter->emitUnknownKeywords(kwDiagNames);
      break;
    default:
      llvm_unreachable("unknown KwDiagResult");
    }
    return {{}, fitness};
  }

  auto [posDiagRes, posDiagNames] =
      operands.diagnosePosOperands(paramListAttr, /*allowCountMismatch=*/true);
  if (posDiagRes == CallOperands::PosDiagResult::kByPosAndKw) {
    if (diagEmitter)
      diagEmitter->emitRedundantKeywords(posDiagNames);
    return {{}, fitness};
  }

  // Parameter inference and call emission rely on this function not failing
  // early due to missing or too many positional parameters.
  assert(posDiagRes == CallOperands::PosDiagResult::kValid &&
         "positional parameter operand check failed unexpectedly");

  /// We will attempt to find a binding for every expected parameter.
  SmallVector<TypedAttr> newBindings;
  newBindings.reserve(numParams);

  // Parameters defined at the beginning of the parameter list may be used by
  // the types of other parameters defined later in the list, e.g. in:
  //    [rank: Int, indices: StaticTuple[rank]]
  // the value provided to 'indices' should actually depend on the specified
  // value of 'rank'.  We use a ParameterEvaluator to keep track of the
  // mapping so far and remap types on demand.
  ParameterEvaluator evaluator;

  // This lambda installs the decl's value in the parameter evaluator and new
  // binding array.
  auto setParamValue = [&](TypedAttr value) {
    evaluator.addInputValue(value);
    newBindings.push_back(value);
  };

  // Use an expr emitter to perform implicit conversions within a parameter
  // context.
  ExprEmitter emitter(declScope, EC_ParameterList);

  // The next positional (or explicitly-specified inferred) binding index.
  size_t posBindingIdx = 0;
  size_t numBindings = operands.size();

  auto inferParameter = [&](Type requestedType) {
    PValue value = parameterInferenceHook(newBindings, evaluator);
    assert(!value || value.getType().mlirType == requestedType &&
                         "inferred a parameter value of wrong type");
    return value;
  };

  DefaultValueHandler defaultHandler(paramListAttr);
  auto fulfillValue = [&](Type requestedType, PassingKind kind) -> PValue {
    // If we have a method to infer parameter values, invoke it to see if we
    // can get an inferred value for the parameter.
    if (PValue value = inferParameter(requestedType))
      return value;

    // Unbind the parameters if those of this passing kind were unbound.
    if ((((kind == PassingKind::PosOnly || kind == PassingKind::PosOrKw) &&
          unpackedPos) ||
         ((kind == PassingKind::PosOrKw || kind == PassingKind::KwOnly) &&
          unpackedKw)) &&
        partial)
      return UnboundAttr::get(requestedType);

    // If the parameter decl is a variadic parameter list, and do not have
    // pack operands that could be used to infer those parameters, then we can
    // fulfill it with an empty list.  We know it must be the last parameter
    // decl. If this isn't actually a variadic type, then we simply reached
    // the end of the parameter list.
    size_t idx = newBindings.size();

    // If available, we use a default parameter value.
    // FIXME: Shouldn't this go into inference itself like empty variadic
    // binding is?
    if (TypedAttr defaultOr = defaultHandler.getDefault(idx)) {
      // Default parameter values may reference other parameter values, so we
      // need to evaluate these.
      return evaluator.getReboundAttribute(defaultOr);
    }

    // Determine if we can use a default parameter for CTAD
    if (ctadPogs.size() <= idx)
      return {};

    PassingKind passingKind = ctadPogs[idx].getPassingKind();
    ArrayRef<TypedAttr> defaults;
    unsigned numCtadParams;
    unsigned normalizedIdx;
    if (passingKind == PassingKind::KwOnly) {
      defaults = defaultKwTypeParams;
      numCtadParams = numKwOnlyCtadParams;
      normalizedIdx = idx - numPosCtadParams;
    } else {
      defaults = defaultPosTypeParams;
      numCtadParams = numPosCtadParams;
      normalizedIdx = idx;
    }

    size_t defaultStartIdx = numCtadParams - defaults.size();
    if (normalizedIdx < numCtadParams && normalizedIdx >= defaultStartIdx) {
      return evaluator.getReboundAttribute(
          defaults[normalizedIdx - defaultStartIdx]);
    }

    return {};
  };

  for (auto [idx, sigType, pog] :
       llvm::enumerate(expectedParamTypes, paramListAttr.getPogs())) {
    // This is the refined type expected by the signature.
    Type requestedType = evaluator.getReboundType(sigType);
    // This is the expected type of a value satisfying this parameter.
    ASTType expectedType = requestedType;
    // If this is a vararg parameter, infer using the element type.
    if (paramListAttr.isVariadic(idx))
      if (auto varType = dyn_cast<VariadicType>(expectedType))
        expectedType = ASTType(varType.getElementType());

    // Inferred params precede positional params, and if explicitly specified,
    // must be specified in order (skipping is allowed).
    if (pog.getPassingKind() == PassingKind::Inferred) {
      // Internally synthesized bindings may specify inferred parameters
      // positionally. Handle them here first.
      if (posBindingIdx < numBindings && !operands[posBindingIdx].keyword) {
        // First check if we are dealing with an UnboundAttr we have to deduce.
        ASTExprAnd<AnyValue> binding = operands[posBindingIdx];
        PValue bindingVal = binding.ir.getIfPValue();
        assert(bindingVal && "Parameters are always PValues");
        if (!partial && isa<UnboundAttr>(bindingVal.get())) {
          if (PValue value =
                  fulfillValue(requestedType, PassingKind::Inferred)) {
            setParamValue(value);
            ++posBindingIdx;
            continue;
          }
          // We tried but couldn't infer an unbound parameter, we must error.
          if (diagEmitter)
            diagEmitter->emitDeductionFailure(idx);
          return {{}, fitness};
        }

        // Otherwise if it's prechecked, consume directly.
        if (posBindingIdx < numPreTypeChecked) {
          setParamValue(bindingVal);
          ++posBindingIdx;
          continue;
        }
      }

      // Otherwise, they are user-provided inferred params, which must be
      // specified by keyword. If we see a non-keyword operand, the operand must
      // not be for this inferred parameter. Even if the operand is specified by
      // keyword, it may be for a later parameter (inferred or regular keyword)
      // instead of this one. In either case, it means this inferred param was
      // not explicitly specified. Use the inference hook for this inferred
      // param and continue.
      if (posBindingIdx >= numBindings || !operands[posBindingIdx].keyword ||
          operands[posBindingIdx].keyword != pog.getName()) {
        if (PValue value = inferParameter(requestedType)) {
          setParamValue(value);
          continue;
        }
        // If this context allows partial binding, leave the value as unbound.
        if (partial) {
          setParamValue(UnboundAttr::get(requestedType));
          continue;
        }
        // Otherwise, emit an inference failure.
        if (diagEmitter)
          diagEmitter->emitInferOnlyFailure(idx);
        return {{}, fitness};
      }

      // The param name matches this operand. Consume this operand.
      OperandValue &binding = operands[posBindingIdx];
      PValue pValue = emitSingleParameterValue(binding, expectedType,
                                               fitness.numImplicitConversions,
                                               emitter, evaluator);
      if (!pValue) {
        if (diagEmitter)
          diagEmitter->emitKwType(pog.getName(), binding, expectedType);
        return {{}, fitness};
      }
      setParamValue(pValue);
      ++posBindingIdx;
      continue;
    }

    // At this point, all inferred pogs have been processed.
    // Find the next positional operand.
    while (posBindingIdx < numBindings && operands[posBindingIdx].keyword)
      ++posBindingIdx;

    // Check to see if we ran out of bindings to provide to this param decl.
    // Implicit parameters are infer-only. They cannot be explicitly passed.
    PassingKind passingKind = pog.getPassingKind();
    StringAttr paramName = pog.getName();
    if (posBindingIdx == numBindings) {
      // We first check if we have a keyword parameter.
      if (const OperandValue *binding = operands.findKwArg(paramName)) {
        assert(passingKind != PassingKind::PosOnly);

        PValue pValue = emitSingleParameterValue(*binding, expectedType,
                                                 fitness.numImplicitConversions,
                                                 emitter, evaluator);
        if (!pValue) {
          if (diagEmitter)
            diagEmitter->emitKwType(paramName, *binding, expectedType);
          return {{}, fitness};
        }
        setParamValue(pValue);
        continue;
      }

      // If we couldn't find a keyword binding for this parameter, then we must
      // be able to infer it or otherwise provide a default value.
      if (PValue value = fulfillValue(requestedType, passingKind)) {
        setParamValue(value);
        continue;
      }

      // If this is a partial binding context, then we don't have a full binding
      // list. Allow parameters to be missing.
      if (partial) {
        setParamValue(UnboundAttr::get(requestedType));
        continue;
      }

      // Otherwise, if this is a parameter that we expected to be inferred, emit
      // an inference failure.
      if (passingKind == PassingKind::Implicit ||
          passingKind == PassingKind::Inferred) {
        if (diagEmitter)
          diagEmitter->emitInferOnlyFailure(idx);
        return {{}, fitness};
      }

      if (passingKind == PassingKind::KwOnly) {
        // If this is a missing keyword-only, we collect them. We put pretend
        // this is implicitly unbound, so we can error out in the end.
        setParamValue(UnboundAttr::get(requestedType));
        kwDiagNames.push_back(paramName);
        continue;
      }

      if (diagEmitter)
        diagEmitter->emitDeductionFailure(idx);
      return {{}, fitness};
    }

    // If we still have positional bindings left, first check if we are dealing
    // with an UnboundAttr we might have to deduce.
    ASTExprAnd<AnyValue> binding = operands[posBindingIdx];
    PValue bindingVal = binding.ir.getIfPValue();
    assert(bindingVal && "Parameters are always PValues");
    if (!partial && isa<UnboundAttr>(bindingVal.get())) {
      if (PValue value = fulfillValue(requestedType, passingKind)) {
        setParamValue(value);
        ++posBindingIdx;
        continue;
      }
      // We tried but couldn't infer an unbound parameter, we must error.
      if (diagEmitter)
        diagEmitter->emitDeductionFailure(idx);
      return {{}, fitness};
    }

    // If this value was already bound and checked, use it.
    /// FIXME: Remove this, why is this needed?
    if (posBindingIdx < numPreTypeChecked) {
      setParamValue(bindingVal);
      ++posBindingIdx;
      continue;
    }

    // Disallow implicit parameters to be explicitly specified. If we see one,
    // complain about too many parameters.
    if (passingKind == PassingKind::Implicit) {
      if (diagEmitter) {
        diagEmitter->emitParamCount(operands.getNumPositional(),
                                    passingKind == PassingKind::PosOnly);
      }
      return {{}, fitness};
    }

    // This lambda hides the diagnostic and error handling logic for checking a
    // single positional parameter binding.
    auto handlePosBinding = [&](size_t index, ASTExprAnd<AnyValue> binding,
                                ASTType expectedType) -> PValue {
      // If the parameter list expected a keyword only parameter, we have too
      // many positional parameters.
      if (passingKind == PassingKind::KwOnly) {
        if (diagEmitter)
          diagEmitter->emitParamCount(operands.getNumPositional(),
                                      /*posOnly=*/true);
        return {};
      }

      PValue pValue = emitSingleParameterValue(binding, expectedType,
                                               fitness.numImplicitConversions,
                                               emitter, evaluator);
      if (!pValue)
        if (diagEmitter)
          diagEmitter->emitPosType(index, binding, expectedType);
      return pValue;
    };

    // Scalar parameter values are installed directly.
    if (!paramListAttr.isVariadic(idx)) {
      PValue paramValue = handlePosBinding(idx, binding, expectedType);
      if (!paramValue)
        return {{}, fitness};
      setParamValue(paramValue);
      ++posBindingIdx;
      continue;
    }

    // If the parameter is a variadic list, it may consume many values, and they
    // all get packed up into a VariadicAttr.
    fitness.hasVariadicParams = true;

    // Unpacked variadics can be passed directly as a whole variadic parameter.
    if (auto unpacked =
            dyn_cast<UnpackedAttr>(binding.ir.getIfPValue().get())) {
      PValue paramValue = handlePosBinding(
          idx, {PValue(unpacked.getValue()), binding.expr}, requestedType);
      if (!paramValue)
        return {{}, fitness};
      setParamValue(paramValue);
      ++posBindingIdx;
      continue;
    }

    SmallVector<TypedAttr> elements;
    auto variadicType = cast<VariadicType>(requestedType);
    do {
      auto &binding = operands[posBindingIdx++];
      if (binding.keyword)
        continue;

      PValue pValue = handlePosBinding(idx, binding, expectedType);
      if (!pValue)
        return {{}, fitness};
      elements.emplace_back(pValue);
      // Passing `_` to a variadic is not allowed. Users should pass `*_` to
      // unbind a variadic parameter.
      if (isa<UnboundAttr>(elements.back())) {
        if (diagEmitter)
          diagEmitter->emitUnboundInVariadic(binding.expr);
        return {{}, fitness};
      }
    } while (posBindingIdx != numBindings);

    auto varType = VariadicType::get(evaluator.getReboundType(expectedType),
                                     variadicType.getConvention());
    setParamValue(VariadicAttr::get(elements, varType));
  }

  // Complain if we collected any missing keyword-only parameters.
  if (!kwDiagNames.empty()) {
    if (diagEmitter)
      diagEmitter->emitMissing(kwDiagNames, "keyword-only");
    return {{}, fitness};
  }

  // Check and complain if we have bindings that didn't get used.
  if (posBindingIdx != numBindings) {
    if (diagEmitter)
      diagEmitter->emitParamCount(operands.getNumPositional(),
                                  /*posOnly=*/false);
    return {{}, fitness};
  }

  return {ParameterExprArrayAttr::get(emitter.getContext(), newBindings),
          fitness};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(
    FnTypeGeneratorType sig, const DiagEmitter &diagEmitter,
    ParameterInferenceHookTy parameterInferenceHook) const {
  return verifyBindingsImpl(parameters, sig.getInputParamTypes(),
                            sig.getMetadata(), parameterInferenceHook,
                            &diagEmitter, /*partial=*/false);
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(FnTypeGeneratorType sig) const {
  return verifyBindings(sig.getInputParamTypes(), sig.getMetadata(),
                        /*partial=*/true);
}

ParameterExprArrayAttr ParamBindings::verifyBindings(ArrayRef<Type> paramTypes,
                                                     PogListAttr paramList,
                                                     bool partial) const {
  auto parameterInferenceHook = [&](ArrayRef<TypedAttr> bindingsSoFar,
                                    const ParameterEvaluator &evaluator) {
    // The inference diagnostics will be unused.
    ParameterInferenceDiagnostics inferenceDiags;
    ParameterInferenceState inference(declScope, getParameters(), bindingsSoFar,
                                      evaluator, inferenceDiags,
                                      /*allowImplicitConversions=*/true);

    inference.infer(paramTypes, paramList, /*hasArguments*/ partial);
    return PValue(inference.getInferredValue(bindingsSoFar.size()));
  };
  auto [bindings, _] = verifyBindingsImpl(parameters, paramTypes, paramList,
                                          parameterInferenceHook,
                                          /*diagEmitter=*/nullptr, partial);
  return bindings;
}

ParameterExprArrayAttr ParamBindings::verifyBindings(StructDeclOp structOp,
                                                     TypeSignatureType sig,
                                                     SMLoc exprLoc,
                                                     bool partial) const {
  auto [bindingValuesAttr, _, diag] =
      verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                     Twine("'") + structOp.getName() + "'", exprLoc,
                     structOp.getLoc(), partial);
  return bindingValuesAttr;
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(FnTypeGeneratorType sig, StringRef baseName,
                              SMLoc exprLoc,
                              std::optional<Location> opLoc) const {
  auto [newBindings, _, diag] =
      verifyBindings(sig.getInputParamTypes(), sig.getMetadata(),
                     opLoc ? Twine("'") + baseName + "'" : Twine(baseName),
                     exprLoc, opLoc, /*partial=*/true);
  return newBindings;
}

std::tuple<ParameterExprArrayAttr, ParamBindings::Fitness,
           std::optional<InflightDiag>>
ParamBindings::verifyBindings(ArrayRef<Type> expectedParamTypes,
                              PogListAttr paramListAttr, const Twine &baseName,
                              SMLoc exprLoc, std::optional<Location> opLoc,
                              bool partial) const {
  size_t maxAllowed = expectedParamTypes.size() -
                      countNumImplicitKinds(paramListAttr) -
                      countNumInferredKinds(paramListAttr);
  ParameterInferenceDiagnostics inferenceDiags;
  std::optional<InflightDiag> diag;
  DiagEmitter diagEmitter{
      /*emitParamCount=*/[&](size_t numActual, bool posOnly) {
        diag = shared.emitError(exprLoc, baseName);
        if (posOnly) {
          emitWrongArgOrParamCount(*diag, countNumPosOnly(paramListAttr),
                                   countNumPositional(paramListAttr), numActual,
                                   "positional parameter");
        } else {
          size_t minRequired = expectedParamTypes.size() -
                               paramListAttr.getDefaultPos().size() -
                               paramListAttr.getDefaultKwOnly().size();
          emitWrongArgOrParamCount(*diag, minRequired, maxAllowed, numActual,
                                   "parameter");
        }
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
        inferenceDiags.attach(paramListAttr, *diag, numActual);
      },
      /*emitPosType=*/
      [&](size_t index, ASTExprAnd<AnyValue> binding, ASTType expectedType) {
        PValue paramVal = binding.ir.getIfPValue();
        diag = shared.emitError(binding.expr->getLoc(), baseName)
               << " parameter #" << index << " has " << expectedType
               << " type, but value has type " << paramVal.getType()
               << binding.expr->getRange();
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitKwType=*/
      [&](StringAttr paramName, ASTExprAnd<AnyValue> binding,
          ASTType expectedType) {
        PValue paramVal = binding.ir.getIfPValue();
        diag = shared.emitError(binding.expr->getLoc(), baseName)
               << " parameter " << paramName << " has " << expectedType
               << " type, but value has type " << paramVal.getType()
               << binding.expr->getRange();
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnknownKeywords=*/
      [&](ArrayRef<StringAttr> unknownKeywords) {
        diag = shared.emitError(exprLoc);
        emitUnknownKeywords(*diag, unknownKeywords, "parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitRedundantKeywords=*/
      [&](ArrayRef<StringAttr> names) {
        diag = shared.emitError(exprLoc);
        emitByPosAndKw(*diag, names, "parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitPosOnlyPassedByKw=*/
      [&](ArrayRef<StringAttr> names) {
        diag = shared.emitError(exprLoc);
        emitPosOnlyPassedByKw(*diag, names, "parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitOutOfOrderInferredKw=*/
      [&](ArrayRef<StringAttr> names) {
        diag = shared.emitError(exprLoc);
        emitOutOfOrderInferredKw(*diag, names);
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitDeductionFailure=*/
      [&](size_t paramIdx) {
        assert(!partial && "parameter deduction failure in a context that "
                           "doesn't allow deduction");
        if (paramListAttr.getPassingKind(paramIdx) != PassingKind::Inferred) {
          diag = shared.emitError(exprLoc, baseName) << " missing required ";
          if (StringAttr name = paramListAttr.getName(paramIdx); !name.empty())
            *diag << "parameter " << name;
          else
            *diag << nameForPosOnly(paramIdx, "parameter");
        } else {
          diag = shared.emitError(exprLoc) << "failed to infer parameter ";
          printNameOrIdx(paramListAttr.getName(paramIdx), paramIdx, *diag);
          inferenceDiags.attach(paramListAttr, *diag);
        }
      },
      /*emitUnboundInVariadic=*/
      [&](const ExprNode *expr) {
        diag = shared.emitError(expr->getLoc());
        *diag << "unbound syntax (i.e. `_`) cannot be passed as a variadic "
                 "parameter";
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnpackedNotAtEnd=*/
      [&](const ExprNode *expr, bool kw) {
        diag = shared.emitError(expr->getLoc());
        *diag << "unbound pack `" << (kw ? "**_" : "*_")
              << "` must be the last " << (kw ? "keyword" : "positional")
              << " parameter" << expr->getRange();
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitInferOnlyFailure=*/
      [&](size_t paramIdx) {
        diag = shared.emitError(exprLoc) << "failed to infer parameter ";
        printNameOrIdx(paramListAttr.getName(paramIdx), paramIdx, *diag);
        inferenceDiags.attach(paramListAttr, *diag);
      },
      /*emitMissing=*/
      [&](ArrayRef<StringAttr> names, const Twine &kindStr) {
        diag = shared.emitError(exprLoc);
        emitMissing(*diag, names, kindStr + " parameter");
        if (opLoc)
          diag->attachNote(*opLoc) << baseName << " declared here";
      }};

  SyntheticNode errorLoc(exprLoc);
  auto parameterInferenceHook = [&](ArrayRef<TypedAttr> bindingsSoFar,
                                    const ParameterEvaluator &evaluator) {
    ParameterInferenceState inference(declScope, getParameters(), bindingsSoFar,
                                      evaluator, inferenceDiags,
                                      /*allowImplicitConversions=*/true);

    // Infer information from the current parameter list.
    inference.infer(expectedParamTypes, paramListAttr, partial);

    // See if we inferred information about the next value.
    if (auto result = inference.getInferredValue(bindingsSoFar.size()))
      return PValue(result);

    // If we succeeded inference but didn't get a value for this parameter, then
    // the parameter must not be present: complain.
    inferenceDiags.addFailure(bindingsSoFar.size(), errorLoc,
                              InferenceFailure::NotFoundFailure());
    return PValue();
  };
  auto [bindings, fitness] =
      verifyBindingsImpl(parameters, expectedParamTypes, paramListAttr,
                         parameterInferenceHook, &diagEmitter, partial);
  return {bindings, fitness, std::move(diag)};
}

TypedAttr ParamBindings::getBoundConstAttrFor(FnOp funcOp, StringRef baseName,
                                              const ExprNode *expr) const {
  FnTypeGeneratorType signature = funcOp.getFullSignature();

  // If this is a global function or struct reference, bind it directly.
  auto parentTrait = dyn_cast<TraitDeclOp>(funcOp->getParentOp());
  if (!parentTrait) {
    // If there are no parameters specified and if we allow unbound symbols,
    // just return the unbound symbol.
    if (empty())
      return funcOp.getBoundReference();

    // Check that the signature can be rebound with our set of bindings.
    ParameterExprArrayAttr newBindings =
        verifyBindings(signature, baseName, expr->getLoc(), funcOp.getLoc());
    if (!newBindings)
      return {};

    // Now that we checked the types match, form the binding.
    return funcOp.getBoundReference(newBindings);
  }

  // The first parameter to the fully bound signature will be the type (confined
  // to the current trait type) that ultimately expands to the concrete type
  // that conforms to the trait.
  assert(!parameters.values.empty());
  PValue selfExpr = parameters.values.front().ir.getIfPValue();
  assert(selfExpr && "parameters are always PValues");

  // When referencing a trait function, bind the reference using a parameter
  // expression instead of the direct reference. Also, drop the implicit trait
  // parameter.
  ParamBindings bindings = *this;
  SmallVector<TypedAttr> paramValues;
  paramValues.push_back(selfExpr);

  auto it = bindings.parameters.values.begin();
  bindings.parameters.values.erase(it, it + 1);
  for (Type type : signature.getInputParamTypes().drop_front())
    paramValues.push_back(UnboundAttr::get(type));

  ASTDecl &traitDecl = *selfExpr.getType().getDecl(shared);
  signature = substituteTraitAliasesIntoSignature(
      *shared.declResolver, &traitDecl, funcOp, signature, selfExpr);

  signature = signature.getSpecializedGenerator(paramValues, [&]() {
    return mlir::emitError(shared.translateLocation(expr->getLoc()))
           << "internal error: ";
  });
  assert(signature && "Error binding trait Self type");

  TypedAttr fnRef = ParamOperatorAttr::get(
      POC::GetVTableEntry,
      {selfExpr,
       StringAttr::get(baseName, StringType::get(funcOp.getContext()))},
      signature);

  if (bindings.empty())
    return fnRef;

  // Attempt to partially bind the parameters to the signature of the function.
  ParameterExprArrayAttr newBindings = bindings.verifyBindings(
      signature, baseName, expr->getLoc(), funcOp.getLoc());
  if (!newBindings)
    return {};

  return BindParamsAttr::get(fnRef, newBindings);
}

void ParamBindings::dump() const { llvm::errs() << parameters << "\n"; }
