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

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/OverloadFitness.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "OperandDiagnostics.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
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
  if (auto trait = dyn_cast_or_null<TraitType>(type.getMetaType()))
    paramBindings.addPrechecked(PValue(type).get());

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

void ParamBindings::replace(size_t idx, const ExprNode *expr, TypedAttr value) {
  posBindings[idx] = {expr, value, /*typeChecked=*/false};
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

  // We don't typecheck the '_' magic parameter, we propagate it.
  if (isa<UnboundAttr>(binding.value))
    return PValue(UnboundAttr::get(expectedType));

  // If the parameter already has the right type, then we're good.
  PValue bindingPVal(binding.getValue());
  if (expectedType.isEqualCanon(bindingPVal.getType()))
    return bindingPVal;

  // If the parameter can be implicitly converted, do so.
  if (emitter.canImplicitlyConvertToType({bindingPVal, binding.expr},
                                         expectedType)) {
    ++numImplicitConversions;
    return emitter.emitPValue({bindingPVal, binding.expr}, EC_CallParamValue,
                              expectedType);
  }
  return {};
};

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(ArrayRef<Type> expectedParamTypes,
                              PogListAttr paramListAttr,
                              ParameterInferenceHookTy parameterInferenceHook,
                              const DiagEmitter &diagEmitter,
                              Boundness boundness) const {
  size_t numParams = expectedParamTypes.size();

  // Handle *_ if it is present expanding posBindings into unpackedPosBindings.
  Fitness fitness{0, false};
  SmallVector<Binding> unpackedPosBindings;
  unpackedPosBindings.reserve(numParams);

  for (auto [idx, binding] : llvm::enumerate(posBindings)) {
    if (!isa<UnpackedAttr>(binding.value)) {
      unpackedPosBindings.push_back(binding);
      continue;
    }

    // UnpackedAttr is aka "*_": it fills up all remaining positional slots and
    // must be at the end of the parameter list.
    if (idx != posBindings.size() - 1) {
      if (diagEmitter.emitUnboundPackNotEnd)
        diagEmitter.emitUnboundPackNotEnd(binding);
      return {{}, fitness};
    }

    // *_ doesn't work with variadic parameter lists.
    // TODO: Why not? It should expand to a variadic on the enclosing context.
    if (paramListAttr.hasVariadic()) {
      if (diagEmitter.emitUnboundPackInVariadic)
        diagEmitter.emitUnboundPackInVariadic(binding);
      return {{}, fitness};
    }

    // Check if we have too many parameters after unpacking.
    size_t numPosPassable = countNumPositional(paramListAttr);
    size_t numUnpackedPositionals = unpackedPosBindings.size();
    if (unpackedPosBindings.size() > numPosPassable) {
      if (diagEmitter.emitTooManyPositional) {
        diagEmitter.emitTooManyPositional(numPosPassable,
                                          numUnpackedPositionals);
      }
      return {{}, fitness};
    }

    // If missing at least one positional parameter, we inject some number of
    // UnboundAttr's, just like the user wrote the right number of _'s.
    auto unboundAttr =
        UnboundAttr::get(UnresolvedType::get(shared.getContext()));
    Binding unboundBinding{binding.expr, unboundAttr, binding.typeChecked};

    // Calculate how many UnboundAttr's (_'s) we need to inject, and put them
    // where the *_ was found.
    ssize_t numUnbounds = numPosPassable - numUnpackedPositionals;
    unpackedPosBindings.append(numUnbounds, unboundBinding);
    assert(unpackedPosBindings.size() == numPosPassable);
  }

  // Create a view of the operands for ease of access.
  OperandContainer<Binding> operands(unpackedPosBindings, &kwBindings);

  KeywordOperandContainer<Binding> variadicKwOperands;
  bool allowMissingKwOnly =
      boundness == Boundness::Partial || parameterInferenceHook;
  auto [kwDiagRes, kwDiagNames] = diagnoseKeywordOperands(
      paramListAttr, variadicKwOperands, operands, allowMissingKwOnly);
  if (kwDiagRes != KwDiagResult::kValid) {
    switch (kwDiagRes) {
    case KwDiagResult::kMissingKwOnly:
      if (diagEmitter.emitMissing)
        diagEmitter.emitMissing(kwDiagNames, "keyword-only");
      break;
    case KwDiagResult::kPosOnlyPassedByKw:
      if (diagEmitter.emitPosOnlyPassedByKw)
        diagEmitter.emitPosOnlyPassedByKw(kwDiagNames);
      break;
    case KwDiagResult::kUnknownKeywords:
      if (diagEmitter.emitUnknownKeywords)
        diagEmitter.emitUnknownKeywords(kwDiagNames);
      break;
    default:
      llvm_unreachable("unknown KwDiagResult");
    }
    return {{}, fitness};
  }

  auto [posDiagRes, posDiagNames] =
      diagnosePosOperands(paramListAttr, operands, /*allowCountMismatch=*/true);
  if (posDiagRes == PosDiagResult::kByPosAndKw) {
    if (diagEmitter.emitRedundantKeywords)
      diagEmitter.emitRedundantKeywords(posDiagNames);
    return {{}, fitness};
  } else {
    // Parameter inference and call emission rely on this function not failing
    // early due to missing or too many positional parameters.
    assert(posDiagRes == PosDiagResult::kValid &&
           "positional parameter operand check failed unexpectedly");
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

  // Use an expr emitter to perform implicit conversions within a parameter
  // context.
  ExprEmitter emitter(shared, declScope, EC_ParameterList);

  size_t posBindingIdx = 0;
  size_t numPosBindings = operands.posOperands.size();

  DefaultValueHandler defaultHandler(paramListAttr);
  auto fulfillValue = [&](size_t idx, Type requestedType) -> PValue {
    // If we have a method to infer parameter values, invoke it to see if we
    // can get an inferred value for the parameter.
    if (parameterInferenceHook) {
      if (PValue value = parameterInferenceHook(idx, newBindings, evaluator)) {
        assert(value.getType().mlirType == requestedType &&
               "inferred a parameter value of wrong type");
        return value;
      }
    }

    // If the parameter decl is a variadic parameter list, and do not have
    // pack operands that could be used to infer those parameters, then we can
    // fulfill it with an empty list.  We know it must be the last parameter
    // decl. If this isn't actually a variadic type, then we simply reached
    // the end of the parameter list.
    if (paramListAttr.isVariadic(idx))
      if (auto varType = dyn_cast<VariadicType>(requestedType))
        return VariadicAttr::get({}, varType);

    // If available, we use a default parameter value.
    if (TypedAttr defaultOr = defaultHandler.getDefault(idx)) {
      // Default parameter values may reference other parameter values, so we
      // need to evaluate these.
      return evaluator.getReboundAttribute(defaultOr);
    }

    // Determine if we can use a default parameter for CTAD.
    size_t defaultStartIdx = numCtadParams - defaultTypeParams.size();
    if (idx < numCtadParams && idx >= defaultStartIdx) {
      return cast<TypedAttr>(evaluator.getReboundAttribute(
          defaultTypeParams[idx - defaultStartIdx]));
    }

    return {};
  };

  for (auto [idx, sigType, pogAttr] :
       llvm::enumerate(expectedParamTypes, paramListAttr.getPogs())) {
    // This is the refined type expected by the signature.
    Type requestedType = evaluator.getReboundType(sigType);
    // This is the expected type of a value satisfying this parameter.
    ASTType expectedType = requestedType;
    // If this is a vararg parameter, infer using the element type.
    if (paramListAttr.isVariadic(idx))
      if (auto varType = dyn_cast<VariadicType>(expectedType))
        expectedType = ASTType(varType.getElementType());

    // Check to see if we ran out of bindings to provide to this param decl.
    // Implicit parameters are infer-only. They cannot be explicitly passed.
    PassingKind passingKind = pogAttr.getPassingKind();
    StringAttr paramName = pogAttr.getName();
    if (posBindingIdx == numPosBindings ||
        (parameterInferenceHook && passingKind == PassingKind::Implicit)) {
      // We first check if we have a keyword parameter.
      if (std::optional<Binding> binding = operands.findKwArg(paramName)) {
        assert(passingKind != PassingKind::PosOnly);

        // If this value was already bound and checked, use it.
        if (binding->typeChecked) {
          setParamValue(binding->value);
          continue;
        }

        PValue pValue = emitSingleParameterValue(*binding, expectedType,
                                                 fitness.numImplicitConversions,
                                                 emitter, evaluator);
        if (!pValue) {
          if (diagEmitter.emitKwType)
            diagEmitter.emitKwType(paramName, *binding, expectedType);
          return {{}, fitness};
        }
        setParamValue(pValue);
        continue;
      }

      if (PValue value = fulfillValue(idx, requestedType)) {
        setParamValue(value);
        continue;
      }
      if (passingKind == PassingKind::Implicit) {
        if (diagEmitter.emitInferOnlyFailure)
          diagEmitter.emitInferOnlyFailure(idx);
        return {{}, fitness};
      }

      // Otherwise, we're simply missing bindings.
      if (boundness == Boundness::Partial) {
        setParamValue(UnboundAttr::get(requestedType));
        continue;
      }

      if (passingKind == PassingKind::KwOnly) {
        // If this is a missing keyword-only, we collect them. We put pretend
        // this is implicitly unbound, so we can error out in the end.
        setParamValue(UnboundAttr::get(requestedType));
        kwDiagNames.push_back(paramName);
        continue;
      }

      if (!fitness.lastExpectedType)
        fitness.lastExpectedType = expectedType;

      if (diagEmitter.emitParamCount) {
        diagEmitter.emitParamCount(numPosBindings,
                                   passingKind == PassingKind::PosOnly);
      }
      return {{}, fitness};
    }

    // If we still have positional bindings left, first check if we are dealing
    // with an UnboundAttr we might have to deduce.
    const Binding &binding = operands.posOperands[posBindingIdx];
    if (isa<UnboundAttr>(binding.value)) {
      if (boundness == Boundness::Full) {
        if (PValue value = fulfillValue(idx, requestedType)) {
          setParamValue(value);
          ++posBindingIdx;
          continue;
        }

        // We tried but couldn't infer an unbound parameter, we must error.
        if (diagEmitter.emitDeductionFailure)
          diagEmitter.emitDeductionFailure(idx);
        return {{}, fitness};
      }
    }

    // If this value was already bound and checked, use it.
    if (binding.typeChecked) {
      setParamValue(binding.value);
      ++posBindingIdx;
      continue;
    }

    // This lambda hides the diagnostic and error handling logic for checking a
    // single positional parameter binding.
    auto handlePosBinding =
        [&, &kwDiagNames = kwDiagNames](size_t index, const Binding &binding,
                                        ASTType expectedType) -> PValue {
      if (passingKind == PassingKind::KwOnly) {
        // If this is a keyword-only passed positionally, we remember it.
        kwDiagNames.push_back(paramName);
        return UnboundAttr::get(expectedType);
      }

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
        binding.getValue().getType() == requestedType) {
      PValue paramValue = handlePosBinding(idx, binding, requestedType);
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
    auto variadicType = cast<VariadicType>(requestedType);
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

  // Complain if we collected any missing keyword-only parameters.
  if (!kwDiagNames.empty()) {
    if (diagEmitter.emitMissing)
      diagEmitter.emitMissing(kwDiagNames, "keyword-only");
    return {{}, fitness};
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
                              PogListAttr paramListAttr, const Twine &baseName,
                              llvm::SMLoc exprLoc,
                              std::optional<Location> opLoc,
                              Boundness boundness) const {
  size_t maxAllowed =
      expectedParamTypes.size() - countNumImplicitKinds(paramListAttr);
  DiagEmitter diagEmitter{
      /*emitParamCount=*/[&](size_t numActual, bool posOnly) {
        InflightDiag diag = shared.emitError(exprLoc, baseName);
        if (posOnly) {
          emitWrongArgOrParamCount(
              diag, /*minRequired=*/countNumPosOnly(paramListAttr), maxAllowed,
              numActual, "positional parameter");
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
                    << " parameter " << paramName << " has " << expectedType
                    << " type, but value has type "
                    << ASTType(binding.getValue().getType())
                    << binding.expr->getRange();
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnknownKeywords=*/
      [&](ArrayRef<StringAttr> unknownKeywords) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitUnknownKeywords(diag, unknownKeywords, "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitRedundantKeywords=*/
      [&](ArrayRef<StringAttr> names) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitByPosAndKw(diag, names, "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitPosOnlyPassedByKw=*/
      [&](ArrayRef<StringAttr> names) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitPosOnlyPassedByKw(diag, names, "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitDeductionFailure=*/
      [&](size_t paramIdx) {
        assert(boundness == Boundness::Full &&
               "parameter deduction failure in a context that doesn't allow "
               "deduction");
        InflightDiag diag = shared.emitError(exprLoc, baseName)
                            << " missing required ";
        if (StringAttr name = paramListAttr.getName(paramIdx); !name.empty())
          diag << "parameter " << name;
        else
          diag << nameForPosOnly(paramIdx, "parameter");
      },
      /*emitUnboundPackInVariadic=*/
      [&](const Binding &binding) {
        InflightDiag diag = shared.emitError(binding.expr->getLoc());
        diag << "unbound pack syntax cannot be used where variadic parameters "
                "are expected";
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitUnboundPackNotEnd=*/
      [&](const Binding &binding) {
        InflightDiag diag = shared.emitError(binding.expr->getLoc());
        diag << "unbound pack must be at the end of the parameter list";
      },
      /*emitInferOnlyFailure=*/
      [&](size_t paramIdx) {
        llvm_unreachable("parameter deduction failure in a context that "
                         "doesn't allow deduction");
      },
      /*emitMissing=*/
      [&](ArrayRef<StringAttr> names, const Twine &kindStr) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitMissing(diag, names, kindStr + " parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      },
      /*emitTooManyPositional=*/
      [&](size_t numMaxAllowed, size_t numActual) {
        InflightDiag diag = shared.emitError(exprLoc);
        emitTooManyPositional(diag, numMaxAllowed, numActual, "parameter");
        if (opLoc)
          diag.attachNote(*opLoc) << baseName << " declared here";
      }};

  return verifyBindings(expectedParamTypes, paramListAttr,
                        /*parameterInferenceHook=*/{}, diagEmitter, boundness);
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(
    LITSignatureType sig, const DiagEmitter &diagEmitter,
    ParameterInferenceHookTy parameterInferenceHook) const {
  return verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                        parameterInferenceHook, diagEmitter, Boundness::Full);
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(LITSignatureType sig) const {
  DiagEmitter diagEmitter{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  return verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                        /*parameterInferenceHook=*/{}, diagEmitter);
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(StructDeclOp structOp, TypeSignatureType sig,
                              llvm::SMLoc exprLoc, Boundness boundness) const {
  auto [bindingValuesAttr, _] =
      verifyBindings(sig.getParamTypes(), sig.getParamListAttrs(),
                     Twine("'") + structOp.getName() + "'", exprLoc,
                     structOp.getLoc(), boundness);
  return bindingValuesAttr;
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(LITSignatureType sig, StringRef baseName,
                              llvm::SMLoc exprLoc,
                              std::optional<Location> opLoc) const {
  auto [newBindings, _] = verifyBindings(
      sig.getParamTypes(), sig.getParamListAttrs(),
      opLoc ? Twine("'") + baseName + "'" : Twine(baseName), exprLoc, opLoc,
      opLoc ? Boundness::Partial : Boundness::Explicit);
  return newBindings;
}

//===----------------------------------------------------------------------===//
// OverloadSet Implementation
//===----------------------------------------------------------------------===//

OverloadSet::OverloadSet(StringRef baseName, ArrayRef<ASTDecl *> fnDecls,
                         ParamBindings &&paramBindings, const ExprNode *expr,
                         CallSyntax syntax, bool erroneous)
    : baseName(baseName), fnDecls(fnDecls.begin(), fnDecls.end()),
      paramBindings(std::move(paramBindings)), expr(expr), syntax(syntax),
      erroneous(erroneous) {}

TypedAttr ParamBindings::getBoundConstAttrFor(LIT::FuncOp funcOp,
                                              StringRef baseName,
                                              const ExprNode *expr) const {
  LITSignatureType signature = funcOp.getFullSignature();

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
  assert(!posBindings.empty());
  PValue selfExpr = posBindings.front().value;

  // When referencing a trait function, bind the reference using a parameter
  // expression instead of the direct reference. Also, drop the implicit trait
  // parameter.
  ParamBindings bindings = *this;
  SmallVector<TypedAttr> paramValues;
  paramValues.push_back(selfExpr);

  auto it = bindings.posBindings.begin();
  bindings.posBindings.erase(it, it + 1);
  for (Type type : signature.getParamTypes().drop_front())
    paramValues.push_back(UnboundAttr::get(type));
  signature = signature.getSpecializedSignature(paramValues, [&]() {
    return mlir::emitError(shared.translateLocation(expr->getLoc()))
           << "internal error: ";
  });
  assert(signature && "Error binding trait Self type");

  TypedAttr fnRef = ParamOperatorAttr::get(
      POC::GetTypeMethod,
      {PValue(selfExpr),
       StringAttr::get(baseName, StringType::get(funcOp.getContext()))},
      signature);
  if (bindings.empty())
    return fnRef;

  ParameterExprArrayAttr newBindings = bindings.verifyBindings(
      signature, baseName, expr->getLoc(), funcOp.getLoc());
  if (!newBindings)
    return {};

  SmallVector<TypedAttr> bindSigOperands;
  bindSigOperands.push_back(fnRef);
  llvm::append_range(bindSigOperands, newBindings);
  return ParamOperatorAttr::get(POC::BindSignature, bindSigOperands);
}

/// Resolve the callee into a single PValue callee.
static PValue getCallee(ArrayRef<ASTDecl *> fnDecls, StringRef baseName,
                        const ParamBindings &paramBindings,
                        const ExprNode *expr) {
  assert(fnDecls.size() == 1 && "expected a single resolved callee");
  auto funcOp = cast<LIT::FuncOp>(*fnDecls.front());
  return paramBindings.getBoundConstAttrFor(funcOp, baseName, expr);
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
  SmallVector<ASTDecl *, 1> newFnDecls;
  return filterOverloadSet(operands, newFnDecls, allowImplicitConversions,
                           emitDiagnosticOnFailure);
}

enum class CallKind { kMethod, kFunction, kIndirect };

static CallKind getCallKind(CallSyntax syntax) {
  switch (syntax) {
  case CallSyntax::kDirectCall:      //< f()
  case CallSyntax::kTypeCall:        //< T()
  case CallSyntax::kImplicitConvert: //< Conversion in an argument context
    return CallKind::kFunction;
  case CallSyntax::kIndirectCall: //< expr()
    return CallKind::kIndirect;
  case CallSyntax::kMethodCall:       //< x.f()
  case CallSyntax::kOperator:         //< -x and x + y
  case CallSyntax::kReversedOperator: //< y + x
  case CallSyntax::kSubscript:        // v[1, 2]
  case CallSyntax::kAttribute:        // v.x
  case CallSyntax::kDestructor:       //< Destructor due to a value definition.
  case CallSyntax::kTupleGetItem:     //< Call to getitem in a tuple assignment.
  case CallSyntax::kMethodCallSynthetic:
    return CallKind::kMethod;
  }
  llvm_unreachable("invalid call syntax");
}

PValue OverloadSet::filterOverloadSet(const CallOperands &operands,
                                      SmallVectorImpl<ASTDecl *> &newFnDecls,
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

    evaluations.push_back(
        OverloadFitness::evaluate(func.getFullSignature(), candidate, *this,
                                  callOperands, allowImplicitConversions));
    anyValid |= evaluations.back().isValid();
  }

  // If all of the candidates are wrong, diagnose this as a failure.
  if (!anyValid) {
    if (emitDiagnosticOnFailure && !isErroneous()) {
      auto diag = getShared().emitError(expr->getLoc()) << expr->getRange();
      if (fnDecls.empty()) {
        diag << "invalid call to '" << baseName << "': no candidates found";
        return {};
      }

      if (fnDecls.size() == 1)
        diag << "invalid ";
      else {
        diag << "no matching ";
        diag << (getCallKind(syntax) == CallKind::kMethod ? "method"
                                                          : "function");
        diag << " in ";
      }

      switch (syntax) {
      default:
        diag << "call to '" << baseName << "'";
        break;
      case CallSyntax::kTypeCall:
        diag << "initialization";
        break;
      }

      // If there is a single callee, emit a specific error about the call.
      if (fnDecls.size() == 1) {
        auto fnDecl = cast<LIT::FuncOp>(*fnDecls[0]);
        diag << ": " << evaluations[0].takeDiag();
        diag.attachNote(fnDecl.getLoc()) << "function declared here";
        return {};
      }

      // Add a note for what is wrong with each candidate.
      for (auto [candidate, eval] : llvm::zip(fnDecls, evaluations)) {
        diag.attachNote(candidate->getLoc())
            << "candidate not viable: " << eval.takeDiag();
        auto func = cast<LIT::FuncOp>(candidate);
        if (func.getIsSynthetic()) {
          diag.attachNote(candidate->getLoc())
              << "generated function with type "
              << ASTType(func.getFullSignature());
        }
      }
      return {};
    }
    return {};
  }

  // Ok, we have at least one valid candidate, so filter for the best matches.
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
    return getCallee(newFnDecls, baseName, newBindings, expr);
  }

  // Otherwise, we have multiple viable candidates that are ambiguous because
  // they all require the same number of implicit conversions.
  if (emitDiagnosticOnFailure && !isErroneous()) {
    size_t minConversions = bestFitness->getNumImplicitConversions();
    auto diag = getShared().emitError(expr->getLoc(), "ambiguous call to '")
                << baseName << "', each candidate requires " << minConversions
                << " implicit conversion" << plural(minConversions)
                << ", disambiguate with an explicit cast" << expr->getRange();
    for (ASTDecl *candidate : newFnDecls) {
      auto func = cast<LIT::FuncOp>(candidate);
      InflightDiag &note = diag.attachNote(candidate->getLoc());
      if (func.getIsSynthetic()) {
        note << "candidate generated with type "
             << ASTType(func.getFullSignature());
      } else {
        note << "candidate declared here";
      }
    }
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
        diag.attachNote(candidate->getLoc())
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
           canConvertWithRebind(candidateType, functionType, getShared());
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
      return getCallee(validCandidates, baseName, paramBindings, expr);

    LITSignatureType candidateType =
        cast<LIT::FuncOp>(*fnDecls.front()).getFullSignature();

    ParamBindings newBindings(paramBindings.declScope, getShared());
    for (TypedAttr bind : getBindingsForSignature(candidateType))
      newBindings.addPrechecked(bind);
    return getCallee(validCandidates, baseName, newBindings, expr);
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
    diag.attachNote(candidate->getLoc())
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
      auto func = cast<LIT::FuncOp>(candidate);
      InflightDiag &note = diag.attachNote(candidate->getLoc());
      if (func.getIsSynthetic()) {
        note << "candidate generated with type "
             << ASTType(func.getFullSignature());
      } else {
        note << "candidate declared here";
      }
    }

    return {};
  }

  return paramBindings.getBoundConstAttrFor(cast<LIT::FuncOp>(*fnDecls[0]),
                                            baseName, expr);
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
    return OverloadSet(declScope, shared, expr, syntax, /*erroneous=*/true);

  SMLoc callLoc = expr->getLoc();

  // First perform a lookup to see if there are any candidates.
  auto lookupResult = shared.lookupAndResolveDecl(methodName, callLoc, type,
                                                  /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> resultDecls = lookupResult.getIfSuccess();
  if (resultDecls.empty()) {
    if (!lookupResult.isErroneous() && errorHandler) // Already diagnosed?
      errorHandler();
    return OverloadSet(declScope, shared, expr, syntax,
                       lookupResult.isErroneous());
  }

  // If we find a vardecl or any other thing, then fail because it cannot be
  // called.
  if (!isa<LIT::FuncOp>(*resultDecls[0]))
    return OverloadSet(declScope, shared, expr, syntax,
                       lookupResult.isErroneous());

  return OverloadSet(methodName, resultDecls,
                     ParamBindings::getForDeclaredType(declScope, shared, type),
                     expr, syntax, lookupResult.isErroneous());
}

/// Lookup of a named named method on the specified type, filtered to match a
/// concrete operand set. If successful, this provides a non-null PValue for a
/// single callee.
PValue OverloadSet::lookup(ASTDecl &declScope, SharedState &shared,
                           ASTType type, StringRef methodName,
                           const CallOperands &callOperands,
                           const ExprNode *callExpr, CallSyntax syntax,
                           function_ref<void()> lookupFailureErrorHandler,
                           bool shouldPrintOverloadErrors) {
  ASTType nmTarget = type.getNonmaterializableTarget(shared);
  auto doLookup = [&](ASTType type, bool shouldPrintError) -> PValue {
    auto ovSet =
        OverloadSet::lookup(declScope, shared, type, methodName, callExpr,
                            syntax, lookupFailureErrorHandler);

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
  return doLookup(type, shouldPrintOverloadErrors);
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

  return paramBindings.getBoundConstAttrFor(cast<LIT::FuncOp>(*fnDecls[0]),
                                            baseName, expr);
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

  assert(!calleeSignature.isAnyVarArg(0) && "Error: self shouldn't be varargs");

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
  auto fitness = OverloadFitness::evaluate(calleeSig, /*indirect*/ nullptr,
                                           bindings, callOperands,
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
        if (scope->isErroneous()) {
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

  // If the type doesn't have the specified method, emit an error.
  if (!callee)
    callee = OverloadSet::lookup(declScope, shared, type, methodName, operands,
                                 callNode, syntax, emitNoMethodError, true);
  if (!callee) {
    dest.resetForError();
    return {};
  }

  return emitIndirectCall(callee, operands, dest, callNode);
}

/// Emit a call to __new__ or __init__, returning an instance of the specified
/// type.  If `allowImplicitConversion` is true, the provided args are allowed
/// to implicitly convert to the expectations of the constructor signatures.
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
  shared.notifyListenerOnCall(callee.fnDecls, expr->getRangeEnd(),
                              callOperands);

  // Init gets a self argument passed in as the first argument by-ref.
  ArrayRef<ASTExprAnd<AnyValue>> origPosOperands = callOperands.posOperands;
  ArrayRef<ASTExprAnd<AnyValue>> posOperands = origPosOperands;
  CallOperands operands = callOperands;

  // As a special extension, register-only types are allowed to return their
  // self directly as a register value instead of taking a memory value in.
  // Check to see if the init members in the overload set are the kInitReg form.
  // TODO: Eliminate special register form.
  bool hasInitSelfArg = true;
  if (type.isRegisterPassable(expr->getLoc(), shared)) {
    for (auto fnDecl : callee.fnDecls) {
      if (cast<LIT::FuncOp>(*fnDecl).getSpecialFunctionKind() ==
          SpecialFunctionKind::kInitReg)
        hasInitSelfArg = false;
    }
  }

  SmallVector<ASTExprAnd<AnyValue>> posOperandsWithSelf;
  auto argsAddSelf = [&]() {
    posOperandsWithSelf.clear();
    if (hasInitSelfArg) {
      posOperandsWithSelf.reserve(posOperands.size() + 1);

      // Unfortunately, we can't just use 'type' or the dest LValue as the
      // buffer to initialize, because the concrete result type might need
      // parameters to be inferred, and those may depend on other value
      // arguments.  Handle this by setting up a placeholder with the type
      // we know so far, and use that to filter the overload set.
      auto attr = UnknownAttr::get(RefType::getImmortal(type, true));
      posOperandsWithSelf.push_back({PValue(attr), expr});
      posOperandsWithSelf.append(posOperands.begin(), posOperands.end());
      operands.posOperands = posOperandsWithSelf;
      operands.hasSelfOperand = true;
    }
  };
  argsAddSelf();

  // Try to resolve the overload set to exactly one candidate, but don't emit an
  // error on failure (we typically want to customize the error).
  SmallVector<ASTDecl *, 1> newFnDecls;
  PValue calleeFn =
      callee.filterOverloadSet(operands, newFnDecls, allowImplicitConversion,
                               /*emitDiagnosticOnFailure=*/false);

  ASTType operandType;
  if (callOperands.posOperands.size() == 1 &&
      callOperands.posOperands[0].ir.getIfCValue())
    operandType = callOperands.posOperands[0].ir.getIfCValue().getRValueType();

  CValue autoNonmaterializableConversion;
  SmallVector<ASTExprAnd<AnyValue>> autoConvertedArgs;
  if (!calleeFn && !callee.isErroneous()) {
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
        newFnDecls.clear();
        calleeFn = callee.filterOverloadSet(operands, newFnDecls,
                                            allowImplicitConversion,
                                            /*emitDiagnosticOnFailure=*/false);
      }
    }
  }

  if (!calleeFn) {
    dest.resetForError();

    // If we failed to resolve the set, then try to emit a tailored error.  If
    // constructing from one value, then this is a type conversion (either
    // implicit or explicit).
    if (operandType && newFnDecls.size() <= 1 && !callee.isErroneous()) {
      // Reject Int(x) where x is already an Int with an error + fixit.
      if (syntax == CallSyntax::kTypeCall && operandType.isEqualCanon(type) &&
          isa<CallNode>(expr)) {
        auto diag = emitError(expr->getLoc());
        const CallNode &callNode = *cast<CallNode>(expr);
        // This removes the constructor call, but does not remove the parens
        // because we don't want to introduce precedence problems.
        diag << "cannot construct " << type
             << " with itself, you can remove the constructor call"
             << posOperands[0].expr->getRange()
             << FixIt::remove(callNode.callee->getRange());
        return {};
      }

      // Diagnose implicit conversions with a custom message, unless this is
      // forming a Reference.
      // FIXME: Why are we duplicating this logic? Just let overload resolution
      // do its job.
      bool isReference = false;
      if (auto declRef = dyn_cast<DeclRefType>(type)) {
        auto symbolNestedRefs = declRef.getSymbol().getNestedReferences();
        if (!symbolNestedRefs.empty() &&
            symbolNestedRefs.back().getAttr().str() == "Reference")
          isReference = true;
      }

      if (syntax == CallSyntax::kImplicitConvert && !isReference) {
        // Handle common type mismatches with tailored errors.
        auto diag = emitError(expr->getLoc());

        // This is true if passing Int type to Int instead of Int() to Int.
        bool isConvertingTypeValue = type.getMetaType() == operandType;
        bool isImplConvert = dest.getContext() != EC_CallParamValue &&
                             dest.getContext() != EC_CallArgValue;
        diag << "cannot " << (isImplConvert ? "implicitly convert " : "pass ");

        if (isConvertingTypeValue)
          diag << type << " type as a";
        else
          diag << operandType;
        diag << " value" << (isImplConvert ? " to " : ", expected ");
        diag << (isConvertingTypeValue ? "an instance of " : "") << type
             << getContextMessage(dest.getContext());

        if (isConvertingTypeValue)
          diag << "; did you mean to instantiate " << type << "?";
        diag << expr->getRange();
        return {};
      }
    }

    // If the type has no candidates, complain about that.
    if (callee.isNull() && !callee.isErroneous()) {
      auto diag = emitError(expr->getLoc());
      if (!type.getDecl(shared)) {
        diag << "MLIR type " << type
             << " must be created with an MLIR operation, not constructor "
                "syntax";
      } else {
        // Emit helpful error message when user tried to call a module.
        if (auto refType = dyn_cast<ParamRefType>(type)) {
          if (auto moduleAttr = dyn_cast<LIT::ModuleAttr>(refType.getParam())) {
            auto metaType = cast<AnyStructType>(moduleAttr.getType());
            emitModuleCallSubscriptDiag(diag, metaType, "call", expr->getLoc(),
                                        shared);
            diag << expr->getRange();
            return {};
          }
        }

        // If the callee is not a module, emit generic message.
        diag << type << " does not implement any '__init__' methods";
      }
      diag << getContextMessage(dest.getContext()) << expr->getRange();
      return {};
    }

    // Otherwise, do it again to emit a generic overload set error.
    auto calleeFn = callee.filterOverloadSet(operands, allowImplicitConversion,
                                             /*emitDiagnosticOnFailure=*/true);
    assert(!calleeFn && "This should fail if it failed before");
    return {};
  }

  // If we successfully resolve the overload set, we know the call will succeed,
  // do it. Register-passable and parameter constructor calls do not require
  // result slot allocation.
  if (!hasInitSelfArg)
    return emitCallUnchecked(calleeFn, operands, dest, expr);
  if (!builder) {
    // If we are emitting into a PValue context, remove the 'inout self'
    // argument because it won't be used.
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
  if (!destMLValue) {
    dest.resetForError();
    return {};
  }

  // Emit the call, but not into 'dest', typically init will return None.
  ValueDest indirectDest(dest.getContext());
  CValue result = emitIndirectCall(calleeFn, operands, indirectDest, expr);
  if (!result) {
    dest.resetForError();
    return {};
  }

  // Now that we've emitted the result into the result buffer, emit a conversion
  // if the expected type and the actual type differ.  This can happen when the
  // ValueDest isn't the same as the result, e.g. "var x: MemFloat = MemInt()".
  return emitCResult(MRValue(destMLValue), expr, dest);
}
