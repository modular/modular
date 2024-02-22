//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/CallEmission.h"

#include "MojoUtils.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/OverloadFitness.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPOps.h"

#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

#include <limits>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

void CallOperands::dump() const { llvm::errs() << *this << '\n'; }

raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os,
                                      const CallOperands &value) {
  os << "CallOperands{ " << value.posOperands.size() << " pos args, "
     << value.getNumKwOperands() << " kw args";
  if (value.hasSelfOperand)
    os << " <HAS SELF OPERAND>";
  os << '\n';

  for (auto operand : value.posOperands)
    os << "  " << operand.ir << "\n";

  if (value.getNumKwOperands())
    os << "TODO: print KWArgs\n";

  return os << '}';
}

ParamBindings::ParamBindings(ExprEmitter &emitter)
    : ParamBindings(emitter.declScope, emitter.shared) {}

ParamBindings ParamBindings::getForDeclaredType(ASTDecl &declScope,
                                                SharedState &shared,
                                                ASTType type) {
  ParamBindings paramBindings(declScope, shared);
  ArrayRef<Type> params = type.getParameters();
  paramBindings.numCtadParams = params.size();
  paramBindings.defaultTypeParams = type.getDefaultPosParams();

  // When binding a trait function, add the self type bindings.
  if (auto trait = dyn_cast<TraitType>(type.getMetaTypeOrSelf())) {
    paramBindings.addPrechecked(
        TypeConstantAttr::get(trait, TypeType::get(trait.getContext())));
    paramBindings.addPrechecked(TypeConstantAttr::get(type, trait));
  }

  ArrayRef<TypedAttr> paramValues = type.getParamBindings();
  for (TypedAttr value : paramValues)
    paramBindings.addPrechecked(value);
  return paramBindings;
}

void ParamBindings::addPrechecked(TypedAttr precheckedBinding) {
  posBindings.push_back({nullptr, precheckedBinding, /*typeChecked=*/true});
}

void ParamBindings::addPrechecked(TypedAttr precheckedBinding,
                                  StringAttr name) {
  auto [_, addedNew] = kwBindings.try_emplace(
      name, Binding{nullptr, precheckedBinding, /*typeChecked=*/true});
  assert(addedNew && "duplicate keyword parameter");
}

void ParamBindings::add(const ExprNode *expr, TypedAttr value) {
  posBindings.push_back({expr, value, /*typeChecked=*/false});
}

void ParamBindings::add(const ExprNode *expr, TypedAttr value,
                        StringAttr name) {
  auto [_, addedNew] =
      kwBindings.try_emplace(name, Binding{expr, value, /*typeChecked=*/false});
  assert(addedNew && "duplicate keyword parameter");
}

//===----------------------------------------------------------------------===//
// ParamBindings Implementation
//===----------------------------------------------------------------------===//

/// Check a single binding and emit a parameter value if possible. If an
/// implicit conversion is required, the provided counter is incremented.
static PValue emitSingleParameterValue(ParamBindings::Binding binding,
                                       ASTType expectedType,
                                       size_t &numImplicitConversions,
                                       ExprEmitter &emitter,
                                       ParserParamEvaluator &evaluator) {
  assert(binding.expr &&
         "should always have an expr tree for unchecked bindings");

  // Check the type matches what is expected, and perform an implicit
  // conversion if needed.
  expectedType = ASTType(evaluator.getReboundType(expectedType.mlirType));

  // If the parameter already has the right type, then we're good.
  PValue bindingPVal(binding.getValue());
  if (expectedType.isEqualCanon(bindingPVal.getType()))
    return bindingPVal;

  // If the parameter can be implicitly converted, do so.
  if (emitter.canImplicitlyConvertToType({bindingPVal, binding.expr},
                                         expectedType)) {
    PValue argValue = emitter.emitPValue({bindingPVal, binding.expr},
                                         EC_CallParamValue, expectedType);
    if (!argValue)
      return {};
    ++numImplicitConversions;
    return argValue;
  }

  return {};
};

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(ArrayRef<Type> expectedParamTypes,
                              ArgParamListAttr paramListAttr,
                              ParameterInferenceHookTy parameterInferenceHook,
                              const DiagEmitter &diagEmitter,
                              bool allowPartiallyBound) const {
  ArrayRef<StringAttr> paramNames = paramListAttr.getNames();
  ArrayRef<PassingKind> paramPassingKinds = paramListAttr.getPassingKinds();
  DefaultValueHandler defaultHandler(paramListAttr);

  size_t numParams = expectedParamTypes.size();
  assert(paramNames.size() == numParams);

  // First, we separate the expected parameter names into positional-only and
  // keyword-passable (pos-or-keyword or kw-only), and ignore variadic names.
  SmallPtrSet<StringAttr, 4> kwPassableNames;
  SmallPtrSet<StringAttr, 4> posOnlyNames;
  for (auto [idx, paramName, passingKind] :
       llvm::enumerate(paramNames, paramPassingKinds)) {
    if (paramListAttr.isVariadic(idx) || paramListAttr.isPack(idx))
      continue; // Variadic/pack parameters cannot be specified by keyword.
    if (passingKind == PassingKind::PosOnly) {
      // Implicit parameters can be unnamed.
      if (!paramName.empty()) {
        auto [_, addedNew] = posOnlyNames.insert(paramName);
        assert(addedNew && "duplicate pos-only parameter name in declaration");
      }
      continue;
    } else if (passingKind == PassingKind::Implicit) {
      assert(paramName.empty());
      continue;
    }
    assert(!paramName.empty());
    auto [_, addedNew] = kwPassableNames.insert(paramName);
    assert(addedNew && "duplicate parameter name in declaration");
  }

  // Then we find all the keyword parameters with unknown names, or specifying
  // positional-only parameters; both of these will result in diagnostics.
  SmallPtrSet<StringAttr, 4> unknownKwParams;
  SmallPtrSet<StringAttr, 4> posOnlyPassedByKw;
  for (auto [name, operandVal] : kwBindings) {
    if (posOnlyNames.contains(name))
      posOnlyPassedByKw.insert(name);
    else if (!kwPassableNames.contains(name))
      unknownKwParams.insert(name);
  }

  auto setToVector = [](SmallPtrSet<StringAttr, 4> &names) {
    return llvm::map_to_vector(names,
                               [](StringAttr name) { return name.strref(); });
  };
  Fitness fitness{0, false};
  if (!unknownKwParams.empty()) {
    if (diagEmitter.emitUnknownKw)
      diagEmitter.emitUnknownKw(setToVector(unknownKwParams));
    return {{}, fitness};
  }
  if (!posOnlyPassedByKw.empty()) {
    if (diagEmitter.emitPosOnlyPassedByKw)
      diagEmitter.emitPosOnlyPassedByKw(setToVector(posOnlyPassedByKw));
    return {{}, fitness};
  }

  /// We will attempt to find a binding for every expected parameter.
  SmallVector<TypedAttr> newBindings;
  newBindings.reserve(numParams);

  // Parameters defined at the beginning of the parameter list may be used by
  // the types of other parameters defined later in the list, e.g. in:
  //    [rank: Int, indices: StaticTuple[rank]]
  // the value provided to 'indices' should actually depend on the specified
  // value of 'rank'.  We use a ParameterEvaluator to keep track of the mapping
  // so far and remap types on demand.
  ParserParamEvaluator evaluator(*shared.declResolver);

  // This lambda installs the decl's value in the parameter evaluator and new
  // binding array.
  auto setParamValue = [&](TypedAttr value) {
    evaluator.addInputValue(value);
    newBindings.push_back(value);
  };

  // First we identify all operands that are unpacked.
  llvm::BitVector unpackedMask(posBindings.size());
  for (auto [idx, binding] : llvm::enumerate(posBindings))
    if (isa<UnpackedType>(binding.getType()))
      unpackedMask.set(idx);

  SmallVector<Binding> unpackedPosBindingsStorage;
  ArrayRef<Binding> unpackedPosBindings(posBindings);
  if (unpackedMask.any()) {
    unpackedPosBindingsStorage.reserve(numParams);

    int unpackedUnboundIdx = -1;
    for (auto [idx, binding] : llvm::enumerate(posBindings)) {
      if (!unpackedMask[idx]) {
        unpackedPosBindingsStorage.push_back(binding);
        continue;
      }
      auto unpacked = cast<UnpackedAttr>(binding.value);
      TypedAttr packed = unpacked.getValue();

      // Unbound pack is special: it fills up all available positional slots, so
      // we must first unpack all others. Remember where it is.
      if (isa<UnboundAttr>(packed)) {
        assert(isa<DiscardType>(packed.getType()));
        if (unpackedUnboundIdx != -1) {
          if (diagEmitter.emitMultipleUnboundPack)
            diagEmitter.emitMultipleUnboundPack(binding);
          return {{}, fitness};
        }
        unpackedUnboundIdx = unpackedPosBindingsStorage.size();
        unpackedPosBindingsStorage.push_back(binding);
        continue;
      }

      // Unpacked variadic parameters are just flattened.
      if (auto varPacked = dyn_cast<VariadicAttr>(packed)) {
        for (TypedAttr val : varPacked.getValues()) {
          unpackedPosBindingsStorage.push_back(
              {binding.expr, val, binding.typeChecked});
        }
        continue;
      }

      // Otherwise we are not able to unpack.
      if (diagEmitter.emitUnpack)
        diagEmitter.emitUnpack(binding);
      return {{}, fitness};
    }

    // Check if we have too many parameters after unpacking
    bool hasParamVarArgs = !paramListAttr.getVariadicIndices().empty();
    size_t numPosPassable = countNumPositional(paramPassingKinds);
    size_t numUnpackedPositionals =
        unpackedPosBindingsStorage.size() - (unpackedUnboundIdx != -1);
    if (!hasParamVarArgs && numUnpackedPositionals > numPosPassable) {
      if (diagEmitter.emitParamCount)
        diagEmitter.emitParamCount(numUnpackedPositionals, /*posOnly=*/false);
      return {{}, fitness};
    }

    // Now we can handle the unpacked unbound attributes if needed.
    if (unpackedUnboundIdx != -1) {
      if (hasParamVarArgs) {
        if (diagEmitter.emitUnboundPackInVariadic) {
          diagEmitter.emitUnboundPackInVariadic(
              unpackedPosBindingsStorage[unpackedUnboundIdx]);
        }
        return {{}, fitness};
      }

      // If missing at least one positional parameter, we inject unbounds.
      if (numUnpackedPositionals < numPosPassable) {
        // We need to calculate how many UnboundAttrs we need to inject.
        auto it = unpackedPosBindingsStorage.begin() + unpackedUnboundIdx;
        int numUnbounds = numPosPassable - numUnpackedPositionals;

        auto unboundAttr =
            UnboundAttr::get(DiscardType::get(shared.getContext()));
        SmallVector<Binding> unbounds(numUnbounds,
                                      {it->expr, unboundAttr, it->typeChecked});
        llvm::replace(unpackedPosBindingsStorage, it, it + 1, unbounds);
      }
      assert(unpackedPosBindingsStorage.size() == numPosPassable);
    }

    unpackedPosBindings = unpackedPosBindingsStorage;
  }

  // Use an expr emitter to perform implicit conversions within a parameter
  // context.
  ExprEmitter emitter(shared, declScope, EC_ParameterList);

  size_t posBindingIdx = 0;
  size_t numPosBindings = unpackedPosBindings.size();
  for (auto [idx, type, paramName, passingKind] :
       llvm::enumerate(expectedParamTypes, paramNames, paramPassingKinds)) {
    // Check to see if we ran out of bindings to provide to this param decl.
    // Implicit parameters are infer-only. They cannot be explicitly passed.
    if (posBindingIdx == numPosBindings ||
        (parameterInferenceHook && passingKind == PassingKind::Implicit)) {
      // Determine what type we expect next.
      Type requestedType = evaluator.getReboundType(type);
      ASTType expectedType = requestedType;
      // If this is a vararg parameter, infer using the element type.
      if (paramListAttr.isVariadic(idx))
        if (auto varType = dyn_cast<VariadicType>(expectedType))
          expectedType = ASTType(varType.getElementType());

      // We first check if we have a keyword parameter.
      if (auto it = kwBindings.find(paramName); it != kwBindings.end()) {
        assert(passingKind != PassingKind::PosOnly);

        const Binding &binding = it->getSecond();

        // If this value was already bound and checked, use it.
        if (binding.typeChecked) {
          setParamValue(binding.value);
          continue;
        }

        PValue pValue = emitSingleParameterValue(binding, expectedType,
                                                 fitness.numImplicitConversions,
                                                 emitter, evaluator);
        if (!pValue) {
          if (diagEmitter.emitKwType)
            diagEmitter.emitKwType(paramName, binding, expectedType);
          return {{}, fitness};
        }
        setParamValue(pValue);
        continue;
      }

      // If we have a method to infer parameter values, invoke it to see if we
      // can get an inferred value for the parameter.
      if (parameterInferenceHook) {
        if (PValue pValue =
                parameterInferenceHook(idx, newBindings,
                                       /*defaultParam=*/{}, evaluator)) {
          assert(pValue.getType().mlirType == requestedType &&
                 "inferred a parameter value of wrong type");
          setParamValue(pValue);
          continue;
        }
        if (passingKind == PassingKind::Implicit) {
          diagEmitter.emitInferOnlyFailure(idx);
          return {{}, fitness};
        }
      }

      // If the parameter decl is a variadic parameter list, and do not have
      // pack operands that could be used to infer those parameters, then we can
      // fulfill it with an empty list.  We know it must be the last parameter
      // decl. If this isn't actually a variadic type, then we simply reached
      // the end of the parameter list.
      if (paramListAttr.isVariadic(idx)) {
        if (auto varType = dyn_cast<VariadicType>(type)) {
          setParamValue(VariadicAttr::get({}, varType));
          fitness.lastExpectedType = expectedType;
          continue;
        }
      }

      // If available, we use a default parameter value.
      if (TypedAttr defaultOr = defaultHandler.getDefault(idx)) {
        // Default parameter values may reference other parameter values, so we
        // need to evaluate these.
        expectedType = evaluator.getReboundType(expectedType);
        auto reboundAttr =
            cast<TypedAttr>(evaluator.getReboundAttribute(defaultOr));
        assert(expectedType.isEqualCanon(reboundAttr.getType()));

        setParamValue(reboundAttr);
        continue;
      }

      // Otherwise, we're simply missing bindings.
      fitness.lastExpectedType = expectedType;
      if (allowPartiallyBound) {
        setParamValue(UnboundAttr::get(expectedType));
        continue;
      }
      if (diagEmitter.emitParamCount) {
        diagEmitter.emitParamCount(numPosBindings,
                                   passingKind == PassingKind::PosOnly);
      }
      return {{}, fitness};
    }

    // If we still have positional bindings left, first check if we are dealing
    // with an UnboundAttr we might have to deduce.
    const Binding &binding = unpackedPosBindings[posBindingIdx];
    if (isa<UnboundAttr>(binding.value)) {
      if (parameterInferenceHook) {
        // Determine if we can use a default parameter for CTAD.
        TypedAttr defaultParam;
        size_t defaultStartIdx = numCtadParams - defaultTypeParams.size();
        if (idx < numCtadParams && idx >= defaultStartIdx) {
          defaultParam = cast<TypedAttr>(evaluator.getReboundAttribute(
              defaultTypeParams[idx - defaultStartIdx]));
        }

        Type requestedType = evaluator.getReboundType(type);
        if (PValue pValue = parameterInferenceHook(idx, newBindings,
                                                   defaultParam, evaluator)) {
          assert(pValue.getType().mlirType == requestedType &&
                 "inferred a parameter value of wrong type");
          setParamValue(pValue);
          ++posBindingIdx;
          continue;
        }

        // If this parameter is a variadic, allow binding an empty list if a
        // value is not provided and it will not be inferred from a pack vararg.
        if (paramListAttr.isVariadic(idx)) {
          if (auto varType = dyn_cast<VariadicType>(type)) {
            setParamValue(VariadicAttr::get({}, varType));
            ++posBindingIdx;
            fitness.lastExpectedType = varType.getElementType();
            continue;
          }
        }

        // We tried but couldn't infer an unbound parameter, we must error.
        if (diagEmitter.emitDeductionFailure)
          diagEmitter.emitDeductionFailure(idx);
        return {{}, fitness};
      }
    }

    // Helper to check and emit diagnostics for redundant keyword parameters.
    auto checkRedundantKwParam = [&, paramName = paramName,
                                  passingKind =
                                      passingKind]() -> LogicalResult {
      if (passingKind == PassingKind::PosOnly ||
          passingKind == PassingKind::Implicit)
        return success();
      assert(!paramName.empty());
      if (auto it = kwBindings.find(paramName); it == kwBindings.end())
        return success(); // Not redundant.
      if (diagEmitter.emitRedundantKw)
        diagEmitter.emitRedundantKw(posBindingIdx, paramName);
      return failure();
    };

    // If this value was already bound and checked, use it.
    if (binding.typeChecked) {
      if (failed(checkRedundantKwParam()))
        return {{}, fitness};
      setParamValue(binding.value);
      ++posBindingIdx;
      continue;
    }

    // This lambda hides the diagnostic and error handling logic for checking a
    // single positional parameter binding.
    auto handlePosBinding = [&](size_t index, const Binding &binding,
                                ASTType expectedType) -> PValue {
      PValue pValue = emitSingleParameterValue(binding, expectedType,
                                               fitness.numImplicitConversions,
                                               emitter, evaluator);
      if (!pValue)
        if (diagEmitter.emitPosType)
          diagEmitter.emitPosType(index, binding, expectedType);
      return pValue;
    };

    // Scalar parameter values are installed directly. Or, if we have a variadic
    // of the same type, we can use it as the value of the parameter directly.
    // FIXME: This allows passing a variadic `Ts` directly. Do we want a new
    // PValue classification for `*Ts`, which is required to pass this legally?
    if (!paramListAttr.isVariadic(idx) ||
        binding.getValue().getType() == type) {
      if (failed(checkRedundantKwParam()))
        return {{}, fitness};
      PValue paramValue =
          handlePosBinding(idx, binding, evaluator.getReboundType(type));
      if (!paramValue)
        return {{}, fitness};
      setParamValue(paramValue);
      ++posBindingIdx;
      continue;
    }

    // If the parameter is a variadic list, it may consume many values, and they
    // all get packed up into a VariadicAttr.
    fitness.hasVariadicParams = true;
    SmallVector<TypedAttr> elements;
    auto variadicType = cast<VariadicType>(type);
    Type expectedType = variadicType.getElementType();
    do {
      const Binding &binding = unpackedPosBindings[posBindingIdx++];
      PValue pValue = handlePosBinding(idx, binding, expectedType);
      if (!pValue)
        return {{}, fitness};
      elements.emplace_back(pValue);
    } while (posBindingIdx != numPosBindings);

    auto varType = VariadicType::get(evaluator.getReboundType(expectedType),
                                     variadicType.getConvention());
    setParamValue(VariadicAttr::get(elements, varType));
  }

  // Check and complain if we have bindings that didn't get used.
  if (posBindingIdx != numPosBindings) {
    if (diagEmitter.emitParamCount)
      diagEmitter.emitParamCount(numPosBindings, /*posOnly=*/false);
    return {{}, fitness};
  }

  return {ParameterExprArrayAttr::get(emitter.getContext(), newBindings),
          fitness};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(ArrayRef<Type> expectedParamTypes,
                              ArgParamListAttr paramListAttr,
                              const Twine &baseName, llvm::SMLoc exprLoc,
                              std::optional<Location> opLoc,
                              bool allowPartiallyBound) const {
  ArrayRef<PassingKind> paramPassingKinds = paramListAttr.getPassingKinds();
  size_t maxAllowed =
      expectedParamTypes.size() - countNumImplicitKinds(paramPassingKinds);
  DiagEmitter diagEmitter{
      /*emitParamCount=*/[&](size_t numActual, bool posOnly) {
        InflightDiag diag = shared.emitError(exprLoc, baseName);
        if (posOnly) {
          emitWrongArgOrParamCount(
              diag, /*minRequired=*/countNumPosOnly(paramPassingKinds),
              maxAllowed, numActual, "positional parameter");
        } else {
          size_t minRequired = expectedParamTypes.size() -
                               paramListAttr.getDefaultPos().size() -
                               paramListAttr.getDefaultKwOnly().size();
          emitWrongArgOrParamCount(diag, minRequired, maxAllowed, numActual,
                                   "parameter");
        }
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitPosType=*/
      [&](size_t index, const Binding &binding, ASTType expectedType) {
        auto diag = shared.emitError(binding.expr->getLoc(), baseName)
                    << " parameter #" << index << " has " << expectedType
                    << " type, but value has type "
                    << ASTType(binding.getValue().getType())
                    << binding.expr->getRange();
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitKwType=*/
      [&](StringAttr paramName, const Binding &binding, ASTType expectedType) {
        auto diag = shared.emitError(binding.expr->getLoc(), baseName)
                    << " parameter '" << paramName << "' has " << expectedType
                    << " type, but value has type "
                    << ASTType(binding.getValue().getType())
                    << binding.expr->getRange();
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnknownKw=*/
      [&](SmallVectorImpl<StringRef> &&unknownKeywords) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitUnknownKeywords(diag, std::move(unknownKeywords), "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitRedundantKw=*/
      [&](size_t paramIdx, StringAttr paramName) {
        InflightDiag diag = shared.emitError(exprLoc);
        diag << "parameter #" << paramIdx << " (" << paramName
             << ") passed both as positional and keyword operand";
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitPosOnlyPassedByKw=*/
      [&](SmallVectorImpl<StringRef> &&names) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitPosOnlyPassedByKw(diag, std::move(names), "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitDeductionFailure=*/
      [&](size_t paramIdx) {
        llvm_unreachable("parameter deduction failure in a context that "
                         "doesn't allow deduction");
      },
      /*emitUnboundPackInVariadic=*/
      [&](const Binding &binding) {
        InflightDiag diag = shared.emitError(binding.expr->getLoc());
        diag << "unbound pack syntax cannot be used where variadic parameters "
                "are expected";
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnpack=*/
      [&](const Binding &binding) {
        InflightDiag diag = shared.emitError(binding.expr->getLoc());
        diag << "cannot unpack non-literal variadic parameters";
      },
      /*emitMultipleUnboundPack=*/
      [&](const Binding &binding) {
        InflightDiag diag = shared.emitError(binding.expr->getLoc());
        diag << "multiple unbound pack symbols not allowed";
      },
      /*emitInferOnlyFailure=*/
      [&](size_t paramIdx) {
        llvm_unreachable("parameter deduction failure in a context that "
                         "doesn't allow deduction");
      }};

  return verifyBindings(expectedParamTypes, paramListAttr,
                        /*parameterInferenceHook=*/{}, diagEmitter,
                        allowPartiallyBound);
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(
    LITSignatureType sig, const DiagEmitter &diagEmitter,
    ParameterInferenceHookTy parameterInferenceHook) const {
  return verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                        parameterInferenceHook, diagEmitter,
                        /*allowPartiallyBound=*/false);
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(LITSignatureType sig) const {
  DiagEmitter diagEmitter{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                          nullptr, nullptr, nullptr, nullptr, nullptr};
  return verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                        /*parameterInferenceHook=*/{}, diagEmitter);
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(StructDeclOp structOp, TypeSignatureType sig,
                              llvm::SMLoc exprLoc,
                              bool allowPartiallyBound) const {
  auto [bindingValuesAttr, _] =
      verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                     Twine("'") + structOp.getName() + "'", exprLoc,
                     structOp.getLoc(), allowPartiallyBound);
  return bindingValuesAttr;
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(LITSignatureType sig, StringRef baseName,
                              llvm::SMLoc exprLoc,
                              std::optional<Location> opLoc) const {

  auto [newBindings, _] = verifyBindings(
      sig.getParamTypes(), sig.getParamListAttrs(),
      opLoc ? Twine("'") + baseName + "'" : Twine(baseName), exprLoc, opLoc);
  return newBindings;
}

//===----------------------------------------------------------------------===//
// OverloadSet Implementation
//===----------------------------------------------------------------------===//

OverloadSet::OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                         ParamBindings &&paramBindings, const ExprNode *expr,
                         CallSyntax syntax)
    : baseName(baseName), fnDecls(fnDecls.begin(), fnDecls.end()),
      paramBindings(std::move(paramBindings)), expr(expr), syntax(syntax) {}

/// Utility function to perform substitutions of the specified callable bindings
/// into the symbol for the given function declaration. It returns the resultant
/// SymbolConstantAttr or produces an error message and returns null.
static TypedAttr getBoundConstAttrFor(ASTType baseType, LIT::FuncOp funcOp,
                                      StringRef baseName,
                                      const ParamBindings &paramBindings,
                                      const ExprNode *expr) {
  // Try to dig out a trait base value.
  auto getIfTrait = [](ASTType type) -> ASTType {
    if (isa_and_nonnull<TraitType>(type.getMetaType()))
      return type;
    return {};
  };
  ASTType trait = getIfTrait(baseType);
  if (!trait) {
    // If there are no parameters specified and if we allow unbound symbols,
    // just return the unbound symbol.
    if (paramBindings.empty())
      return funcOp.getBoundReference();

    // Check that the signature can be rebound with our set of bindings.
    LITSignatureType signature = funcOp.getFullSignature();
    ParameterExprArrayAttr newBindings = paramBindings.verifyBindings(
        signature, baseName, expr->getLoc(), funcOp.getLoc());
    if (!newBindings)
      return {};

    // Now that we checked the types match, form the binding.
    return funcOp.getBoundReference(newBindings);
  }

  // When referencing at trait function, bind the reference using a parameter
  // expression instead of the direct reference. Drop the implicit trait
  // parameters.
  // FIXME(#25492): The implicit trait parameters probably need a rethink.
  LITSignatureType signature = funcOp.getFullSignature();
  ParamBindings bindings = paramBindings;
  assert(bindings.posBindings.size() >= 2);
  auto it = bindings.posBindings.begin();
  SmallVector<TypedAttr> paramValues({it->value, (it + 1)->value});
  bindings.posBindings.erase(it, it + 2);
  for (Type type : signature.getParamTypes().drop_front(2))
    paramValues.push_back(UnboundAttr::get(type));
  auto loc = paramBindings.shared.translateLocation(expr->getLoc());
  signature = signature.getSpecializedSignature(paramValues, loc);

  TypedAttr fnRef = ParamOperatorAttr::get(
      POC::GetTypeMethod,
      {PValue(trait),
       StringAttr::get(baseName, StringType::get(funcOp.getContext()))},
      signature);
  if (bindings.empty())
    return fnRef;

  ParameterExprArrayAttr newBindings = bindings.verifyBindings(
      signature, baseName, expr->getLoc(), funcOp.getLoc());
  if (!newBindings)
    return {};
  SmallVector<TypedAttr> operands{fnRef};
  llvm::append_range(operands, newBindings);
  return ParamOperatorAttr::get(POC::BindSignature, operands);
}

/// Resolve the callee into a single PValue callee.
static PValue getCallee(ASTType baseType, ArrayRef<ASTDecl *> fnDecls,
                        StringRef baseName, const ParamBindings &paramBindings,
                        const ExprNode *expr) {
  assert(fnDecls.size() == 1 && "expected a single resolved callee");
  auto funcOp = cast<LIT::FuncOp>(*fnDecls.front());
  return getBoundConstAttrFor(baseType, funcOp, baseName, paramBindings, expr);
}

/// Return if the given fitness is valid, and drop the diagnostics otherwise.
static bool isValid(OverloadFitness &eval) {
  if (eval.isValid())
    return true;
  eval.takeDiag().abandon();
  return false;
}

/// Assuming we have at least one valid candidate, filter the candidate list to
/// those with the best fitness. If there is more than one candidate with
/// maximal fitness, we filter for non-static methods.
///
/// To aid downstream diganostics, the function returns the fitness of the best
/// candidate. All diagnostics from erroneous candidates are dropped.
static const OverloadFitness *
selectBestCandidates(ArrayRef<ASTDecl *> fnDecls,
                     MutableArrayRef<OverloadFitness> evaluations,
                     SmallVectorImpl<ASTDecl *> &newFnDecls) {
  assert(newFnDecls.empty());
  bool areTheBestCandidatesStatic = true;

  // Find the first valid candidate.
  evaluations = evaluations.drop_until(isValid);
  const OverloadFitness *bestFitness = &evaluations.front();

  for (auto [candidate, eval] :
       llvm::zip(fnDecls.take_back(evaluations.size()), evaluations)) {
    // Ignore all subsequent failures and candidates that are definitely worse.
    if (!isValid(eval) || bestFitness->isBetter(eval))
      continue;

    // If we found a strictly better candidate, clear the list.
    if (eval.isBetter(*bestFitness)) {
      newFnDecls.clear();
      areTheBestCandidatesStatic = true;
    }

    // If the current best candidates are not static, we ignore new static
    // candidates.
    bool isStatic = cast<LIT::FuncOp>(*candidate).getIsStatic();
    if (!areTheBestCandidatesStatic && isStatic)
      continue;

    // If the current best candidates are static, and we just found a non-static
    // one, we clear the list.
    if (areTheBestCandidatesStatic && !isStatic) {
      newFnDecls.clear();
      areTheBestCandidatesStatic = false;
    }

    newFnDecls.push_back(candidate);
    bestFitness = &eval;
  }

  return bestFitness;
}

PValue OverloadSet::filterOverloadSet(const CallOperands &operands,
                                      bool allowImplicitConversions,
                                      bool emitDiagnosticOnFailure) const {
  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<OverloadFitness> evaluations;
  bool anyValid = false;
  for (ASTDecl *candidate : fnDecls) {
    auto func = cast<LIT::FuncOp>(*candidate);

    // If we are dealing with a static method, we check if the operands include
    // a self operand and remove it, otherwise the signature might not match.
    CallOperands callOperands(operands);
    if (func.getIsStatic() && operands.hasSelfOperand)
      callOperands.posOperands = callOperands.posOperands.drop_front();

    evaluations.push_back(OverloadFitness::evaluate(func.getFullSignature(),
                                                    *this, callOperands,
                                                    allowImplicitConversions));
    anyValid |= evaluations.back().isValid();
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (!anyValid) {
    if (emitDiagnosticOnFailure) {
      // If there is a single callee, emit a specific error about the call.
      if (fnDecls.size() == 1) {
        auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
        auto diag = getShared().emitError(expr->getLoc(), "invalid call to '")
                    << baseName << "': " << expr->getRange()
                    << evaluations[0].takeDiag();
        diag.attachNote(fnDecl.getLoc()) << "function declared here";
        return {};
      }

      // Otherwise emit an error, and a note for what is wrong with each
      // candidate.
      auto diag = getShared().emitError(expr->getLoc(),
                                        "no matching function in call to '")
                  << baseName << "': " << expr->getRange();
      for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
        auto fnDecl = cast<LIT::FuncOp>(*candidate);
        diag.attachNote(fnDecl->getLoc())
            << "candidate not viable: " << eval.takeDiag();
      }
      return {};
    }
    return {};
  }

  // Ok, we have at least one valid candidate, so filter for the best matches.
  SmallVector<ASTDecl *, 1> newFnDecls;
  const OverloadFitness *bestFitness =
      selectBestCandidates(fnDecls, evaluations, newFnDecls);

  // Notify the listener of the updated decl references for the call now that
  // invalid candidates have been filtered out.
  if (!newFnDecls.empty())
    getShared().notifyListenerOnRef(newFnDecls, baseName, expr, syntax);

  // If we found exactly one viable candidate then we succeed.
  if (newFnDecls.size() == 1) {
    // On success, wrap things up into one callee.
    ParamBindings newBindings(paramBindings.declScope, getShared());
    for (TypedAttr bind : bestFitness->getParamBindings())
      newBindings.addPrechecked(bind);
    return getCallee(baseType, newFnDecls, baseName, newBindings, expr);
  }

  // Otherwise, we have multiple viable candidates that are ambiguous because
  // they all require the same number of implicit conversions.
  if (emitDiagnosticOnFailure) {
    size_t minConversions = bestFitness->getNumImplicitConversions();
    auto diag = getShared().emitError(expr->getLoc(), "ambiguous call to '")
                << baseName << "', each candidate requires " << minConversions
                << " implicit conversion" << plural(minConversions)
                << ", disambiguate with an explicit cast" << expr->getRange();
    for (ASTDecl *candidate : newFnDecls)
      diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
          << "candidate declared here";
  }
  return {};
}

PValue
OverloadSet::filterOverloadSetForValueType(ASTType functionType,
                                           bool emitDiagnosticOnFailure) const {
  if (!emitDiagnosticOnFailure)
    return filterOverloadSetForValueType(functionType, /*emitError=*/nullptr);

  std::optional<InflightDiag> diag;
  return filterOverloadSetForValueType(
      functionType, [&](SMLoc loc) -> InflightDiag & {
        return diag.emplace(getShared().emitError(loc));
      });
}

PValue OverloadSet::filterOverloadSetForValueType(
    ASTType functionType, function_ref<InflightDiag &(SMLoc)> emitError) const {
  // If the target type is something weird then don't filter.  Let the error be
  // reported another way.
  if (!isa<SignatureType>(functionType.mlirType)) {
    if (emitError) {
      auto &diag = emitError(expr->getLoc())
                   << "cannot convert function to non-function type "
                   << functionType;
      for (ASTDecl *candidate : fnDecls)
        diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
            << "candidate declared here with type "
            << ASTType(cast<LIT::FuncOp>(*candidate).getFullSignature());
    }
    return {};
  }

  // TODO(#22771): This is using an exact match which is perhaps too specific of
  // a check. We could do some amount of parameter inference to support cases
  // like:
  //
  //    fn foo[Type: mlirtype]() -> Type
  //    var f : ()-> Int = foo
  //
  // We could also support generating a lambda for fancy implicit conversions
  // and subtyping some day.
  auto getBindingsForSignature =
      [&](LITSignatureType candidateType) -> ParameterExprArrayAttr {
    // Apply any bound parameters to the candidate's type since they will be
    // applied when a reference is made.
    // TODO(#22771): Parameter inference.
    auto [newBindings, _] = paramBindings.verifyBindings(candidateType);
    return newBindings;
  };

  auto isValidCandidate = [&](LITSignatureType candidateType) -> bool {
    // Apply any bound parameters to the candidate's type since they will be
    // applied when a reference is made.  We only do this if there are some
    // bindings present, because (unlike normal function calls) the result type
    // may have unbound parameters that we are trying to match, e.g. when in a
    // parameter expression context.
    if (!paramBindings.empty()) {
      auto newBindings = getBindingsForSignature(candidateType);
      if (!newBindings)
        return false; // If there is an error, return the problem.

      // If anything was bound, apply it to the signature so the expected
      // argument types are updated.
      if (!newBindings.empty())
        candidateType = candidateType.getSpecializedSignature(
            newBindings, getShared().translateLocation(expr->getLoc()));
    }

    return functionType.isEqualCanon(candidateType) ||
           canZeroCostConvert(getShared(), candidateType, functionType);
  };

  // Evaluate the fitness of each candidate in our overload set.
  SmallVector<ASTDecl *> validCandidates;
  for (ASTDecl *candidate : fnDecls) {
    LITSignatureType candidateType =
        cast<LIT::FuncOp>(*candidate).getFullSignature();
    if (isValidCandidate(candidateType))
      validCandidates.push_back(candidate);
  }

  // Notify the listener of the updated decl references for the call now that
  // invalid candidates have been filtered out.
  if (!validCandidates.empty())
    getShared().notifyListenerOnRef(validCandidates, baseName, expr, syntax);

  // If we have exactly one viable candidate, then we succeed.
  if (validCandidates.size() == 1) {
    if (paramBindings.empty())
      return getCallee(baseType, validCandidates, baseName, paramBindings,
                       expr);

    LITSignatureType candidateType =
        cast<LIT::FuncOp>(*fnDecls.front()).getFullSignature();

    ParamBindings newBindings(paramBindings.declScope, getShared());
    for (TypedAttr bind : getBindingsForSignature(candidateType))
      newBindings.addPrechecked(bind);
    return getCallee(baseType, validCandidates, baseName, newBindings, expr);
  }

  // If we aren't to emit a diagnostic, just return the failure.
  if (!emitError)
    return {};

  auto &diag = emitError(expr->getLoc());
  if (validCandidates.empty()) {
    diag << "no '" << baseName << "' candidates have type " << functionType
         << expr->getRange();
  } else {
    diag << "ambiguous use of '" << baseName << "' as type " << functionType
         << expr->getRange();
  }

  for (ASTDecl *candidate : fnDecls) {
    diag.attachNote(cast<LIT::FuncOp>(*candidate)->getLoc())
        << "candidate declared here with type "
        << ASTType(cast<LIT::FuncOp>(*candidate).getFullSignature());
  }
  return {};
}

/// Perform substitutions of the specified bindings into the symbol, returning
/// the resultant LITSymbolConstant attr or producing an error message and
/// returning null. This allows producing a reference to a parameterized
/// function without the parameters specified.  They can be bound later.
TypedAttr OverloadSet::getBoundConstantAttr() const {
  if (fnDecls.size() != 1) {
    assert(!fnDecls.empty() && "DirectCallable malformed");
    auto diag = getShared().emitError(
                    expr->getLoc(),
                    "cannot form a reference to overloaded declaration of '")
                << baseName << "'" << expr->getRange();
    for (ASTDecl *candidate : fnDecls) {
      auto funcOp = cast<LIT::FuncOp>(*candidate);
      diag.attachNote(funcOp.getLoc()) << "candidate declared here";
    }

    return {};
  }

  return getBoundConstAttrFor(baseType, cast<LIT::FuncOp>(*fnDecls[0]),
                              baseName, paramBindings, expr);
}

/// Get a OverloadSet for a lookup of a named method on the specified type.
/// If successful, this provides a non-null OverloadSet.
///
/// On failure, this returns a null OverloadSet and invokes errorHandler if
/// the problem hasn't already been diagnosed. This does not emit an error on
/// failure.
OverloadSet OverloadSet::lookup(ASTDecl &declScope, SharedState &shared,
                                ASTType type, StringRef methodName,
                                const ExprNode *expr, CallSyntax syntax,
                                function_ref<void()> errorHandler) {

  // If this is a previously-reported error, ignore and don't report an
  // additional error.
  if (type.isTypeCheckErrorType())
    return OverloadSet(declScope, shared, expr, syntax);

  SMLoc callLoc = expr->getLoc();

  // First perform a lookup to see if there are any candidates.
  auto lookupResult = shared.lookupAndResolveDecl(methodName, callLoc, type,
                                                  /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
  if (resultDecls.empty()) {
    if (!lookupResult.isErroneous() && errorHandler) // Already diagnosed?
      errorHandler();
    return OverloadSet(declScope, shared, expr, syntax);
  }

  // If we find a vardecl or any other thing, then fail because it cannot be
  // called.
  if (!isa<LIT::FuncOp>(*resultDecls[0]))
    return OverloadSet(declScope, shared, expr, syntax);

  OverloadSet result(methodName, resultDecls,
                     ParamBindings::getForDeclaredType(declScope, shared, type),
                     expr, syntax);
  result.baseType = type;
  return result;
}

/// Lookup of a named named method on the specified type, filtered to match a
/// concrete operand set. If successful, this provides a non-null PValue for a
/// single callee.
PValue OverloadSet::lookup(ASTDecl &declScope, SharedState &shared,
                           ASTType type, StringRef methodName,
                           const CallOperands &callOperands,
                           const ExprNode *callExpr, CallSyntax syntax,
                           function_ref<void()> errorHandler) {
  ASTType nmTarget = type.getNonmaterializableTarget(shared);
  bool shouldPrintError = bool(errorHandler);
  auto doLookup = [&](ASTType type, bool shouldPrintError) -> PValue {
    auto ovSet = OverloadSet::lookup(declScope, shared, type, methodName,
                                     callExpr, syntax, errorHandler);

    // If the core lookup failed, don't filter.
    if (ovSet.isNull())
      return {};

    // Filter the overload set with the actual operands list.  If this
    // fails, report an error (if we have an error handler) and reset to a
    // null state so the client can check this.
    return ovSet.filterOverloadSet(
        callOperands, /*allowImplicitConversions=*/true,
        /*emitDiagnosticOnFailure=*/shouldPrintError);
  };

  // If there is a nonmaterializableTarget, try using the original type first,
  // then falling back on the target.
  if (nmTarget) {
    PValue ret = doLookup(type, false);
    if (ret)
      return ret;
    type = nmTarget;
  }
  return doLookup(type, shouldPrintError);
}

/// Try to resolve the overload set to a single function candidate, using the
/// expected type if provided or using current bindings if an emitter is
/// provided.  This emits errors if 'emitter' is non-null, but does not if it
/// is null.
PValue OverloadSet::getDirectSymbol(ASTType expectedType) const {
  // Handle the case of a single candidate.
  if (fnDecls.size() == 1) {
    // This is an unbound function. Just return a reference.
    if (paramBindings.empty())
      return cast<LIT::FuncOp>(*fnDecls.front()).getBoundReference();

    // Bind the parameters.
    return getBoundConstantAttr();
  }

  // With an emitter and an expected type, the overload set can definitely be
  // resolved to a single candidate or not.
  if (expectedType) {
    return filterOverloadSetForValueType(expectedType,
                                         /*emitDiagnosticOnFailure=*/true);
  }
  // Otherwise, emit the "cannot form a reference to overloaded decl" error.
  return getBoundConstantAttr();
}

PValue OverloadSet::getIfPValue() const {
  // Overload sets with base values cannot be emitted as PValues since they
  // depend on a dynamic value.
  // TODO: A conversion can be emitted if the base value is a PValue.
  if (baseValue)
    return {};

  if (fnDecls.size() != 1)
    return {};

  return getBoundConstAttrFor(baseType, cast<LIT::FuncOp>(*fnDecls[0]),
                              baseName, paramBindings, expr);
}

/// Emit this as a RValue if it can be resolved, otherwise emit an ambiguity
/// error and return null.
CValue OverloadSet::emitAsCValue(ExprEmitter &emitter, ValueDest &dest) {
  // If we have an overload set with multiple possibilities, we'll fail to emit
  // this as a RValue.  Try to resolve it based on the destination's type.
  ASTType expectedType;
  if (fnDecls.size() > 1) {
    expectedType = dest.resolveImpliedType(expr->getLoc(),
                                           /*no implied type*/ Type(), emitter);
  }

  // We allow unbound symbols here which can be emitted as an PValue.  In the
  // case where we are partially applying, that will force the unbound symbol
  // into a SRValue which will catch symbols that are not fully bound.
  PValue directSymbolAttr = getDirectSymbol(expectedType);
  if (!directSymbolAttr)
    return {};

  // If we have no base value, then we are just a symbol, return it.
  if (!baseValue)
    return emitter.emitCResult(directSymbolAttr, expr, dest);

  auto loc = baseValue.expr->getLoc();

  // Otherwise, we have a base symbol for an instance method /and/ a self value
  // to apply to it.  Partially apply it to form a result closure.
  auto calleeSignature =
      cast<LITSignatureType>(directSymbolAttr.getType().mlirType);
  Type firstArgIRType = calleeSignature.getArguments()[0];
  ArgConvention selfConvention = calleeSignature.getArgConvention(0);
  Value firstArgValue;

  assert(!calleeSignature.isVarArg(0) && "Error: self shouldn't be varargs");

  switch (selfConvention) {
  case ArgConvention::ByRefResult:
  case ArgConvention::OwnedInMem:
  case ArgConvention::BorrowedInMem: {
    auto diag =
        emitter.emitError(
            loc, "TODO: partial application requires closure generation ")
        << baseValue.expr->getRange();
    if (auto cValue = baseValue.ir.getIfCValue())
      diag << cValue.getRValueType();
    return {};
  }

  case ArgConvention::ByRef:
  case ArgConvention::InitSelf: {
    ValueDest baseLVDest(dest.getContext());
    LValue baseLV = emitter.emitLValue(baseValue, baseLVDest);
    if (!baseLV)
      return {};

    // Using partial application over an lvalue isn't safe until we support an
    // ownership models with mutable borrows.
    emitter.emitError(loc, "TODO: partial application to mutable base isn't "
                           "supportable without a lifetime model")
        << baseValue.expr->getRange();
    return {};
  }
  case ArgConvention::BorrowedInReg:
  case ArgConvention::OwnedInReg:
    // Otherwise we can have either an lvalue or rvalue, but we need to convert
    // to an rvalue if we have an lvalue.
    firstArgValue = emitter.emitSRValue(baseValue, EC_CallArgValue);
    if (!firstArgValue)
      return {};

    // TODO: Partial application isn't handling ownership right at all, we
    // should probably disable it.
    break;
  case ArgConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }

  assert(firstArgIRType == firstArgValue.getType() &&
         "base types should always structurally line up");

  // TODO: Need to emit a closure instance that partially applies the 'self'
  // argument here.
  emitter.emitError(
      loc, "TODO: partial application of member methods is not yet supported");
  return {};
}

//===----------------------------------------------------------------------===//
// Call Emission Implementation
//===----------------------------------------------------------------------===//

/// Emit a function call to the specified callee with the specified operand
/// values.  This emits an error and returns null on failure.
CValue OverloadSet::emitCall(const CallOperands &callOperands, ValueDest &dest,
                             ExprEmitter &emitter) {
  if (isNull()) // Base was already diagnosed as an error.
    return {};

  // Used in some cases below, lifetime needs to exist for this whole method.
  SmallVector<ASTExprAnd<AnyValue>> posOperandsWithSelf;

  // If we have a bound self, add it to the operand list to simplify the logic
  // below.
  CallOperands operands = callOperands;
  if (baseValue) {
    ArrayRef<ASTExprAnd<AnyValue>> posOperands = operands.posOperands;
    posOperandsWithSelf.reserve(posOperands.size() + 1);
    posOperandsWithSelf.push_back(baseValue);
    posOperandsWithSelf.append(posOperands.begin(), posOperands.end());
    assert((syntax == CallSyntax::kMethodCall ||
            syntax == CallSyntax::kMethodCallSynthetic) &&
           "Unexpected syntax form");
    operands.posOperands = posOperandsWithSelf;
    operands.hasSelfOperand = true;
  }

  // Check the direct callees to see if they can be unambiguously resolved
  // with the bindings list and specified arguments.
  PValue callee = filterOverloadSet(operands,
                                    /*allowImplicitConversions=*/true,
                                    /*emitDiagnosticOnFailure=*/true);
  if (!callee)
    return {};
  return emitter.emitCallUnchecked(callee, operands, dest, expr);
}

CValue ExprEmitter::emitIndirectCall(CValue callee,
                                     const CallOperands &callOperands,
                                     ValueDest &dest,
                                     const ExprNode *callExpr) {
  auto calleeSig = dyn_cast<SignatureType>(callee.getRValueType());
  if (!calleeSig) {
    // If we are invoking something other than a SignatureType, try to invoke
    // its `__call__` method.
    SmallVector<ASTExprAnd<AnyValue>> posOperandsWithCallee;
    posOperandsWithCallee.push_back({callee, callExpr});
    llvm::append_range(posOperandsWithCallee, callOperands.posOperands);
    return emitNamedMethodCall(
        "__call__",
        CallOperands(posOperandsWithCallee, callOperands.kwOperands), dest,
        CallSyntax::kDirectCall, callExpr);
  }

  if (calleeSig.getNumResultParams()) {
    emitError(callExpr->getLoc(), "invalid indirect call: callee has ")
        << calleeSig.getNumResultParams() << " unbound result parameter"
        << plural(calleeSig.getNumResultParams()) << callExpr->getRange();
    dest.resetForError();
    return {};
  }

  // If we have a function pointer, resolve it to an RValue.
  RValue calleeRV = emitRValue({callee, callExpr}, EC_CallCalleeValue);
  if (!calleeRV) {
    dest.resetForError();
    return {};
  }

  // Check to see if we can apply these operands to the callee signature.
  OverloadSet bindings{"callee", /*fnDecls=*/{}, ParamBindings(*this), callExpr,
                       CallSyntax::kIndirectCall};
  auto fitness = OverloadFitness::evaluate(calleeSig, bindings, callOperands,
                                           /*allowImplicitConversions=*/true);
  if (!fitness.isValid()) {
    // If not, diagnose it with an error.
    emitError(callExpr->getLoc(), "invalid indirect call: ")
        << fitness.takeDiag();
    dest.resetForError();
    return {};
  }

  // If we have inferred parameters, bind them here. An indirect call with
  // inferred parameters must be a PValue.
  if (!fitness.getParamBindings().empty()) {
    SmallVector<TypedAttr> bindOperands;
    if (auto calleePVal = calleeRV.getIfPValue()) {
      bindOperands.push_back(calleePVal);
    } else {
      // The callee can be dynamic in cases where one of the parents had a
      // resolution error but we are inside the body of a closure. In this case
      // we want to silently error.
      for (ASTDecl *scope = &declScope; scope; scope = scope->getParentDecl()) {
        if (scope->hasReferenceError) {
          dest.resetForError();
          return {};
        }
      }
      llvm_unreachable("binding a dynamic callee?");
    }
    llvm::append_range(bindOperands, fitness.getParamBindings());
    calleeRV = PValue(ParamOperatorAttr::get(POC::BindSignature, bindOperands));
  }

  return emitCallUnchecked(calleeRV, callOperands, dest, callExpr);
}

CValue ExprEmitter::emitNamedMethodCall(StringRef methodName,
                                        const CallOperands &callOperands,
                                        ValueDest &dest, CallSyntax syntax,
                                        const ExprNode *callNode) {
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = callOperands.posOperands;
  assert(!posOperands.empty() &&
         "Cannot emit a method call without a receiver!");

  // Emit the first/self operand to a CValue so we can figure out which type to
  // lookup on.
  CValue selfVal = posOperands[0].ir.getIfCValue();
  SmallVector<ASTExprAnd<AnyValue>> updatedPosOperands;
  if (!selfVal) {
    selfVal = emitCValue(posOperands[0], EC_CallArgValue);
    if (!selfVal) {
      dest.resetForError();
      return {};
    }
    // We can't mutate posOperands because it's an ArrayRef.  If something
    // changed, recurse with a temporary buffer.
    updatedPosOperands.append(posOperands.begin(), posOperands.end());
    updatedPosOperands[0].ir = selfVal;
    posOperands = updatedPosOperands;
  }

  CallOperands operands(posOperands, callOperands.kwOperands);

  ASTType type = selfVal.getRValueType();

  auto emitNoMethodError = [&]() {
    auto diag = emitError(callNode->getLoc(), "")
                << type << " does not implement the '" << methodName
                << "' method";
    switch (syntax) {
    case CallSyntax::kMethodCallSynthetic:
    case CallSyntax::kMethodCall:
      [[fallthrough]];
    case CallSyntax::kOperator:
      diag << posOperands[0].expr->getRange();
      break;
    case CallSyntax::kReversedOperator:
      diag << posOperands[1].expr->getRange();
      break;
    default:
      break;
    }
  };

  PValue callee = {};
  if (ASTType nmTarget = type.getNonmaterializableTarget(shared)) {
    // If the type doesn't have the specified method, but it's
    // nonmaterializable, give it a second chance with the materialized type.
    // If the type doesn't have the specified method, emit an error.
    callee = OverloadSet::lookup(declScope, shared, type, methodName, operands,
                                 callNode, syntax);
    if (!callee) {
      ValueDest selfDest(EC_CallArgValue);
      CValue convertedSelf = emitConstructorCall(
          nmTarget, CallOperands({{selfVal, posOperands[0].expr}}), callNode,
          CallSyntax::kImplicitConvert, selfDest,
          /*allowImplicitConversion=*/true);
      if (!convertedSelf) {
        dest.resetForError();
        return {};
      }
      updatedPosOperands.clear();
      updatedPosOperands.append(posOperands.begin(), posOperands.end());
      updatedPosOperands[0].ir = convertedSelf;
      posOperands = updatedPosOperands;
      type = nmTarget;
    }
  }

  // If the type doesn't have the specified method, emit an error.
  if (!callee)
    callee = OverloadSet::lookup(declScope, shared, type, methodName, operands,
                                 callNode, syntax, emitNoMethodError);
  if (!callee) {
    dest.resetForError();
    return {};
  }

  return emitIndirectCall(callee, operands, dest, callNode);
}

CValue ExprEmitter::emitConstructorCall(ASTType type,
                                        const CallOperands &callOperands,
                                        const ExprNode *expr, CallSyntax syntax,
                                        ValueDest &dest,
                                        bool allowImplicitConversion) {
  // If the dest type is invalid, then an error has already been reported.
  if (type.isTypeCheckErrorType())
    return {};

  // Check to see if we can invoke an __init__ method to convert it.
  auto callee =
      OverloadSet::lookup(declScope, shared, type, "__init__", expr, syntax);
  return emitConstructorCall(type, callee, callOperands, expr, syntax, dest,
                             allowImplicitConversion);
}

CValue ExprEmitter::emitConstructorCall(ASTType type, const OverloadSet &callee,
                                        const CallOperands &callOperands,
                                        const ExprNode *expr, CallSyntax syntax,
                                        ValueDest &dest,
                                        bool allowImplicitConversion) {
  // Init for memory-only types get their self argument implicitly initialized
  // and passed in as the first argument.
  ArrayRef<ASTExprAnd<AnyValue>> origPosOperands = callOperands.posOperands;
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = origPosOperands;
  CallOperands operands = callOperands;
  bool isMemoryOnly = !type.isRegisterPassable(expr->getLoc(), shared);
  SmallVector<ASTExprAnd<AnyValue>> posOperandsWithSelf;
  auto argsAddSelf = [&]() {
    posOperandsWithSelf.clear();
    if (isMemoryOnly) {
      posOperandsWithSelf.reserve(posOperands.size() + 1);

      // Unfortunately, we can't just use 'type' or the dest LValue as the
      // buffer to initialize, because the concrete result type might need
      // parameters to be inferred, and those may depend on other value
      // arguments.  Handle this by setting up a placeholder with the type
      // we know so far, and use that to filter the overload set.
      auto attr = UnknownAttr::get(PointerType::get(type));
      posOperandsWithSelf.push_back({PValue(attr), expr});
      posOperandsWithSelf.append(posOperands.begin(), posOperands.end());
      operands.posOperands = posOperandsWithSelf;
      operands.hasSelfOperand = true;
    }
  };
  argsAddSelf();

  // Try to resolve the overload set to exactly one candidate, but don't emit an
  // error on failure (we typically want to customize the error).
  PValue calleeFn = callee.filterOverloadSet(operands, allowImplicitConversion,
                                             /*emitDiagnosticOnFailure=*/false);

  ASTType operandType;
  if (callOperands.posOperands.size() == 1 &&
      callOperands.posOperands[0].ir.getIfCValue()) {
    operandType = callOperands.posOperands[0].ir.getIfCValue().getRValueType();
  }

  CValue autoNonmaterializableConversion;
  SmallVector<ASTExprAnd<AnyValue>> autoConvertedArgs;
  if (!calleeFn) {
    // If we are converting from a nonmaterializable struct, always allow an
    // extra implicit conversion to the nonmaterializable target.  Then try
    // again to find a constructor.
    if (ASTType nonmaterializableTarget =
            operandType.getNonmaterializableTarget(shared)) {
      if (!nonmaterializableTarget.isEqualCanon(type)) {
        ValueDest autoDest(nonmaterializableTarget, EC_CallArgValue);
        autoNonmaterializableConversion = emitConstructorCall(
            nonmaterializableTarget, origPosOperands, origPosOperands[0].expr,
            syntax, autoDest, /*allowImplicitConversion=*/false);
        autoConvertedArgs.push_back(
            {autoNonmaterializableConversion, origPosOperands[0].expr});
        operands.posOperands = autoConvertedArgs;
        argsAddSelf();
        calleeFn = callee.filterOverloadSet(operands, allowImplicitConversion,
                                            /*emitDiagnosticOnFailure=*/false);
      }
    }
  }

  if (!calleeFn) {
    // If we failed to resolve the set, then try to emit a tailored error.  If
    // constructing from one value, then this is a type conversion (either
    // implicit or explicit).
    if (operandType) {
      auto diag = emitError(expr->getLoc());
      // Reject Int(x) where x is already an Int with an error + fixit.
      if (syntax == CallSyntax::kTypeCall && operandType.isEqualCanon(type) &&
          isa<CallNode>(expr)) {
        const CallNode &callNode = *cast<CallNode>(expr);
        // This removes the constructor call, but does not remove the parens
        // because we don't want to introduce precedence problems.
        diag << "cannot construct " << type
             << " with itself, you can remove the constructor call"
             << posOperands[0].expr->getRange()
             << FixIt::remove(callNode.callee->getRange());
        return {};
      }

      if (syntax != CallSyntax::kImplicitConvert) {
        diag << "cannot construct " << type << " from " << operandType
             << " value" << getContextMessage(dest.getContext())
             << expr->getRange();
        return {};
      }

      // Handle common type mismatches with tailored errors.
      bool isConvertingTypeValue = type.hasMetaType(operandType);
      if (dest.getContext() == EC_CallParamValue ||
          dest.getContext() == EC_CallArgValue) {
        diag << "cannot pass " << operandType
             << (isConvertingTypeValue ? " type" : "") << " value, "
             << ((dest.getContext() == EC_CallParamValue) ? "parameter"
                                                          : "argument")
             << " expected " << (isConvertingTypeValue ? "an instance of " : "")
             << type;
      } else {
        diag << "cannot implicitly convert " << operandType
             << (isConvertingTypeValue ? " type" : "") << " value to "
             << (isConvertingTypeValue ? "an instance of " : "") << type
             << getContextMessage(dest.getContext());
      }

      if (isConvertingTypeValue)
        diag << "; did you mean to instantiate " << operandType << "?";
      diag << expr->getRange();
      return {};
    }

    // If the type has no candidates, complain about that.
    if (callee.isNull()) {
      auto diag = emitError(expr->getLoc());
      if (!type.getDecl(shared)) {
        diag << "MLIR type " << type
             << " must be created with an MLIR operation, not constructor "
                "syntax";
      } else {
        diag << type << " does not implement any '__init__' methods";
      }
      diag << getContextMessage(dest.getContext()) << expr->getRange();
      return {};
    }

    // Otherwise, do it again to emit a generic overload set error.
    calleeFn = callee.filterOverloadSet(operands, allowImplicitConversion,
                                        /*emitDiagnosticOnFailure=*/true);
    assert(!calleeFn && "This should fail if it failed before");
    return {};
  }

  // If we successfully resolve the overload set, we know the call will succeed,
  // do it. Register-passable and parameter constructor calls do not require
  // result slot allocation.
  if (!isMemoryOnly)
    return emitCallUnchecked(calleeFn, operands, dest, expr);
  if (!builder) {
    operands = callOperands;
    return emitCallUnchecked(calleeFn, operands, dest, expr);
  }

  // We need to invoke memory-only constructors specially since the buffer is
  // exposed.
  auto calleeSig = cast<SignatureType>(calleeFn.getType().mlirType);
  auto firstArgRVType =
      ASTType(calleeSig.getArguments()[0]).getReferenceElementType();

  // For an initialization of a memory-only type, we need to replace the
  // destination buffer with the actual destination lvalue to use.
  MLValue destMLValue =
      dest.getMLValueForResult(expr->getLoc(), firstArgRVType, *this);
  posOperandsWithSelf[0].ir = destMLValue;
  if (!destMLValue)
    return {};

  // Emit the call, but not into 'dest', typically init will return None.
  ValueDest indirectDest(dest.getContext());
  CValue result = emitIndirectCall(calleeFn, operands, indirectDest, expr);
  if (!result)
    return {};

  // Now that we've emitted the result into the result buffer, emit a conversion
  // if the expected type and the actual type differ.  This can happen when the
  // ValueDest isn't the same as the result, e.g. "var x: MemFloat = MemInt()".
  return emitCResult(MRValue(destMLValue), expr, dest);
}
