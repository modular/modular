//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParamBindings.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "ParamInf.h"
#include "ParserEvaluationContext.h"
#include "Traits.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/Constraints.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/SharedState.h"
#include "Support/Compiler/OperationUtils.h"
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
///     fn lork(self, thing: MyStruct[get_witness(X, MyTrait, T)])
///
/// This function will do that conversion. If we aren't calling a trait method
/// with an alias, it'll return the given desiredSignature unmodified.
///
/// For more context, see
/// https://www.notion.so/modularai/verifyConformance-Arcana-13e1044d37bb80e88cb5c285a232784e?pvs=4#13e1044d37bb80bf8b42f3953af880f8
///
/// TODO(MOCO-1259): Support static methods with associated aliases
FnTypeGeneratorType LIT::substituteTraitAliasesIntoSignature(
    DeclResolver &declResolver, ASTDecl &traitDecl, FnOp candidateFunc,
    FnTypeGeneratorType desiredSignature, PValue selfPValue) {
  ParserParameterEvaluator traitAliasReplacer(declResolver.shared);
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      AliasDeclOp traitAlias =
          dyn_cast_or_null<LIT::AliasDeclOp>(decl->getIfOperation());
      if (!traitAlias)
        continue;
      StringAttr nameStringAttr =
          StringAttr::get(candidateFunc->getContext(), name.str());
      auto traitName = StringAttr::get(
          candidateFunc->getContext(),
          getFlattenedSymbolName(candidateFunc.getInheritedFrom().value_or(
              traitDecl.getSymbolRef())));
      TypedAttr aliasRef =
          declResolver.shared.getEvaluationContext().getGetWitnessAttr(
              selfPValue, traitName, nameStringAttr, traitAlias.getType());
      traitAliasReplacer.setDeclBinding(traitAlias.getParamDecl(), aliasRef);
    }
  }
  return traitAliasReplacer.replace(desiredSignature);
}

//===----------------------------------------------------------------------===//
// ParamBindings
//===----------------------------------------------------------------------===//

ParamBindings::ParamBindings(ASTDecl &declScope, const ExprNode *expr)
    : declScope(declScope), shared(declScope.getShared()), parameters(expr) {}

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

SMLoc ParamBindings::getExprLoc() const { return getExpr()->getLoc(); }

/// Create a (possibly partially unbound) set of bindings for the given type.
/// This can be used to initialize the binding set for methods. If the given
/// type is not a parametric user defined type, this returns empty bindings.
ParamBindings ParamBindings::getForDeclaredType(ASTDecl &declScope,
                                                ASTType type,
                                                const ExprNode *expr,
                                                Type optionalParentTraitType) {
  ParamBindings paramBindings(declScope, expr);
  // TODO: this will not work with arbitrary parametric ancestors.
  // Default params need to come from the original declaration, instead of
  // TypeSignatureType, as the latter won't contain the full defaults list if
  // any have been bound already (when `type` is partially specified).
  ASTDecl *decl = type.getDecl(declScope.getShared());
  if (decl) {
    if (auto structDecl =
            dyn_cast_or_null<StructDeclOp>(decl->getIfOperation())) {
      paramBindings.defaultPosTypeParams =
          structDecl.getSignature().getDefaultPosParams();
      paramBindings.defaultKwTypeParams =
          structDecl.getSignature().getDefaultKwOnlyParams();
      llvm::append_range(
          paramBindings.ctadPogs,
          structDecl.getSignature().getParamListAttrs().getPogs());
      for (auto pog : paramBindings.ctadPogs) {
        if (pog.getPassingKind() == PassingKind::KwOnly)
          paramBindings.numKwOnlyCtadParams++;
        else
          paramBindings.numPosCtadParams++;
      }
    }
  }

  // When binding a trait function, add the self type bindings.
  if (decl && isa_and_nonnull<TraitDeclOp>(decl->getIfOperation())) {
    auto typeAttr = PValue(type).get();

    // The source value be something of trait type like Movable, or it may be
    // something of AnyTraitType type, like
    //   fn ex[Trait: MovableMetaType, T: Trait](argument: T):
    // where T is some type that is known to conform to Movable.  In the later
    // case we just know that the input type conforms to Movable, and we want to
    // look up members to bind in Movable, so bind the Trait type here.  If this
    // is a struct, or simple trait, keep it.
    if (auto paramType = sugarDynCast<ParamType>(type.getMetaType())) {
      auto simpleTraitType =
          sugarCast<AnyTraitType>(paramType.getParam().getType())
              .getTraitType();
      // Upcast from a parametric type of trait metatype value (e.g. "some
      // type that conforms to Movable) to the simple trait type (Movable)
      // so we can substitute the value into the signature.
      typeAttr = UpcastAttr::get(simpleTraitType, PValue(type));
    }
    paramBindings.addPrechecked(expr, typeAttr);
  } else if (isa<TraitType>(decl->getIfTypeValue())) {
    if (optionalParentTraitType) {
      // If caller provided a parent trait type, we need to upcast the self.
      auto typeAttr = UpcastAttr::get(optionalParentTraitType, PValue(type));
      paramBindings.addPrechecked(expr, typeAttr);
    } else {
      // If this is a trait composition, the method signature's self type won't
      // match directly (need to upcast the composition into the trait type that
      // declared the method). Add as _not_ prechecked.
      paramBindings.add(expr, PValue(type).get());
    }
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

/// Helper function to emit diagnostics for unprovable constraints from a
/// Fitness result.
static void
emitUnprovableConstraintsFromFitness(const ParamBindings::Fitness &fitness,
                                     SharedState &shared, SMLoc exprLoc,
                                     ASTDecl *declIfKnown) {
  if (fitness.unprovableConstraints.empty())
    return;

  std::string baseName;
  if (declIfKnown)
    baseName = "'" + declIfKnown->getUserNameIfOperation()->str() + "'";
  else
    baseName = "parametric value";

  MojoInflightDiag diag = shared.emitError(exprLoc)
                          << "invalid bindings for " << baseName
                          << ": lacking evidence to prove correctness";
  if (declIfKnown)
    diag.attachNote(declIfKnown->getLoc())
        << "cannot prove constraint"
        << plural(fitness.unprovableConstraints.size());
  for (auto constraint : fitness.unprovableConstraints)
    LIT::emitConstraintInconclusive(shared.getDeclResolver(), diag, constraint);
}

/// Check a single binding and emit a parameter value if possible. If an
/// implicit conversion is required, the provided counter is incremented.
static PValue emitSingleParameterValue(ASTExprAnd<AnyValue> binding,
                                       ASTType expectedType, IREmitter &emitter,
                                       ParserParameterEvaluator &evaluator) {

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
    // Align sugar if necessary.
    return ParamOperatorAttr::getRebind(bindingVal, expectedType);

  // If the parameter can be implicitly converted, do so.
  if (IREmitter::canImplicitlyConvertToType(
          {bindingVal, binding.expr}, expectedType, emitter.getDeclScope())) {
    bindingVal = emitter.emitPValue(binding, EC_CallParamValue, expectedType);
    return bindingVal;
  }

  // Otherwise, we have an error.
  return {};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindingsImpl(
    const CallOperands &origOperands, ArrayRef<Type> expectedParamTypes,
    PogListAttr paramListAttr, ParamInfState &inference,
    ParamInfDiags &inferenceDiags,
    llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag,
    bool partial, ASTDecl *declIfDirect) const {

  Fitness fitness{{}};

  // Check to see if we have ... and remove it from the parameter list.
  bool hasEllipsis = false;
  CallOperands operands(origOperands.callExpr);
  for (auto [idx, binding] : llvm::enumerate(origOperands.values)) {
    if (isa<EllipsisAttr>(binding.ir.getIfPValue().get()))
      hasEllipsis = true;
    else
      operands.values.push_back(binding);
  }

  // With that out of the way, we can now get onto normal type checking of
  // 'operands'.
  size_t numParams = expectedParamTypes.size();

  OperandValueList variadicKwOperands;
  auto [kwDiagRes, kwDiagNames] = operands.diagnoseKeywordOperands(
      paramListAttr, variadicKwOperands, /*allowMissingKwOnly=*/true);
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
    return {{}, fitness};
  }

  auto [posDiagRes, posDiagNames] =
      operands.diagnosePosOperands(paramListAttr, /*allowCountMismatch=*/true);
  if (posDiagRes == CallOperands::PosDiagResult::kByPosAndKw) {
    emitByPosAndKw(getDiag({}), posDiagNames, "parameter");
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
  // value of 'rank'.  We use a ParserParameterEvaluator to keep track of the
  // mapping so far and remap types on demand.
  ParserParameterEvaluator evaluator(shared);

  // This lambda installs the decl's value in the parameter evaluator and new
  // binding array. Also verifies all constraints are satisfied.
  ArrayRef<PogMetadataAttr> pogs = paramListAttr.getPogs();
  auto setParamValueAndVerify = [&](TypedAttr value,
                                    Type requestedType) -> LogicalResult {
    size_t idx = newBindings.size();
    // The canonical types must match, now make sure sugar aligns.
    if (value.getType() != requestedType)
      value = ParamOperatorAttr::getRebind(value, requestedType);
    evaluator.appendIndexBinding(value);
    newBindings.push_back(value);

    // If this is a partial binding, we don't need to verify constraints. The
    // caller is expected to verify the full binding list later.
    if (partial && isa<UnboundAttr>(value))
      return success();

    ArrayRef<ConstraintAttr> constraints = pogs[idx].getConstraints();
    if (constraints.empty())
      return success();

    // Verify all constraints are satisfied, collecting unprovable constraints.
    ConstraintResult result = checkConstraints(
        declScope, paramListAttr, constraints, /*origConstraints=*/{}, getDiag,
        &fitness.unprovableConstraints, &evaluator);
    return success(result == ConstraintResult::Satisfied);
  };

  // Use an expr emitter to perform implicit conversions within a parameter
  // context.
  IREmitter emitter(declScope, EC_ParameterList);

  size_t numBindings = operands.size();

  auto inferParameter = [&](Type requestedType) {
    size_t paramIdx = newBindings.size();

    TypedAttr inferred = inference.getInferredValue(paramIdx);
    if (inferred) {
      assert(ASTType(inferred.getType()).isEqualCanon(requestedType) &&
             "inferred a parameter value of wrong type");
      return PValue(inferred);
    }

    // If we succeeded inference but didn't get a value for this parameter,
    // then the parameter must not be present: complain.
    inferenceDiags.addFailure(InferenceFailure::NotFoundFailure{paramIdx});
    return PValue();
  };

  DefaultValueHandler defaultHandler(paramListAttr);
  auto fulfillValue = [&](Type requestedType, PassingKind kind) -> PValue {
    // If we have a method to infer parameter values, invoke it to see if we
    // can get an inferred value for the parameter.
    if (PValue value = inferParameter(requestedType))
      return value;

    // Unbind the parameters if those of this passing kind were unbound.
    if (hasEllipsis && partial)
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

  auto emitInferenceFailure = [&](size_t paramIdx) {
    assert(!partial && "parameter deduction failure in a context that "
                       "doesn't allow deduction");
    MojoInflightDiag &diag = getDiag(getExprLoc());
    if (declIfDirect && isa<StructDeclOp>(declIfDirect->getIfOperation()))
      diag << "'" << *declIfDirect->getUserNameIfOperation() << "' ";

    {
      // The parameter name is scoped to 'declScope'.
      DeclResolver::DeclScopeChanger x(&declScope);
      diag << "failed to infer parameter "
           << ParamDeclRefAttr::get(paramListAttr.getName(paramIdx),
                                    expectedParamTypes[paramIdx]);
    }

    // If this is a method on a struct and we couldn't infer something from
    // its self parameters, complain about the struct.
    if (declIfDirect && isa<FnOp>(declIfDirect->getIfOperation())) {
      if (auto structOp = dyn_cast<StructDeclOp>(
              cast<FnOp>(declIfDirect->getIfOperation())->getParentOp())) {
        auto structSig = structOp.getSignature();
        if (paramIdx < structSig.getNumParams()) {
          diag << " of parent struct '" << structOp.getDeclName().getValue()
               << "'";
          inferenceDiags.addExplanation(diag);
          diag.attachNote(structOp.getLoc()) << " struct declared here";
          return;
        }
      }
    }
    inferenceDiags.addExplanation(diag);
  };

  auto emitTypeMismatch = [&](size_t index, ASTExprAnd<AnyValue> binding,
                              ASTType expectedType) {
    PValue paramVal = binding.ir.getIfPValue();
    DeclResolver::DeclScopeChanger x(&declScope);

    MojoInflightDiag &diag = getDiag({});
    if (declIfDirect) // Why only structs? Seems arbitrary, push higher?
      diag << "'" << *declIfDirect->getUserNameIfOperation() << "' ";
    diag << "parameter "
         << ParamDeclRefAttr::get(paramListAttr.getName(index),
                                  expectedParamTypes[index])
         << " has " << expectedType << " type, but value has type "
         << paramVal.getType() << binding.expr->getRange();
  };

  // The next positional (or explicitly-specified inferred) binding index.
  size_t posBindingIdx = 0;
  inference.inferFromParamList(partial);

  for (auto [idx, sigType, pog] : llvm::enumerate(expectedParamTypes, pogs)) {
    // This is the refined type expected by the signature.
    Type requestedType = evaluator.getReboundType(sigType);
    // This is the expected type of a value satisfying this parameter.
    ASTType expectedType = requestedType;
    // If this is a vararg parameter, infer using the element type.
    if (paramListAttr.isPosVarArg(idx))
      if (auto varType = sugarDynCast<VariadicType>(expectedType))
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
            if (failed(setParamValueAndVerify(value, requestedType)))
              return {{}, fitness};
            ++posBindingIdx;
            continue;
          }
          // We tried but couldn't infer an unbound parameter, we must error.
          emitInferenceFailure(idx);
          return {{}, fitness};
        }

        // Otherwise if it's prechecked, consume directly.
        if (posBindingIdx < numPreTypeChecked) {
          if (failed(setParamValueAndVerify(bindingVal, requestedType)))
            return {{}, fitness};
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
          if (failed(setParamValueAndVerify(value, requestedType)))
            return {{}, fitness};
          continue;
        }
        // If this context allows partial binding, leave the value as unbound.
        if (partial) {
          if (failed(setParamValueAndVerify(UnboundAttr::get(requestedType),
                                            requestedType)))
            return {{}, fitness};
          continue;
        }
        // Otherwise, emit an inference failure.
        emitInferenceFailure(idx);
        return {{}, fitness};
      }

      // The param name matches this operand. Consume this operand, the
      // parameter must have been installed in the ParamInfState, simply pull it
      // out. If there is no pValue, this is a type mismatch.
      //
      // TODO: is it always a type mismatch error? We probably should not guess
      // here and instead let `ParamInfState` determine the error kind.
      OperandValue &binding = operands[posBindingIdx];
      PValue pValue = inference.getInferredValue(newBindings.size());
      if (!pValue) {
        emitTypeMismatch(idx, binding, expectedType);
        return {{}, fitness};
      }
      if (failed(setParamValueAndVerify(pValue, requestedType)))
        return {{}, fitness};
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

        TypedAttr pValue = inference.getInferredValue(newBindings.size());
        // ParamInfState does not install `UnboundAttr`: we should not do it
        // here either now that evaluator support a sparse set of parameter
        // being bound.
        if (!pValue && sugarIsa<UnboundAttr>(binding->ir.getIfPValue().get()))
          pValue = UnboundAttr::get(expectedType);

        if (!pValue) {
          emitTypeMismatch(idx, *binding, expectedType);
          return {{}, fitness};
        }
        if (failed(setParamValueAndVerify(pValue, requestedType)))
          return {{}, fitness};
        continue;
      }

      // If we couldn't find a keyword binding for this parameter, then we must
      // be able to infer it or otherwise provide a default value.
      if (PValue value = fulfillValue(requestedType, passingKind)) {
        if (failed(setParamValueAndVerify(value, requestedType)))
          return {{}, fitness};
        continue;
      }

      // If this is a partial binding context, then we don't have a full binding
      // list. Allow parameters to be missing.
      if (partial) {
        if (failed(setParamValueAndVerify(UnboundAttr::get(requestedType),
                                          requestedType)))
          return {{}, fitness};
        continue;
      }

      if (passingKind == PassingKind::KwOnly) {
        // If this is a missing keyword-only, we collect them. We put pretend
        // this is implicitly unbound, so we can error out in the end.
        if (failed(setParamValueAndVerify(UnboundAttr::get(requestedType),
                                          requestedType)))
          return {{}, fitness};
        kwDiagNames.push_back(paramName);
        continue;
      }

      // Emit an inference failure.
      emitInferenceFailure(idx);
      return {{}, fitness};
    }

    // If we still have positional bindings left, first check if we are dealing
    // with an UnboundAttr we might have to deduce.
    ASTExprAnd<AnyValue> binding = operands[posBindingIdx];
    PValue bindingVal = binding.ir.getIfPValue();
    assert(bindingVal && "Parameters are always PValues");
    if (!partial && isa<UnboundAttr>(bindingVal.get())) {
      if (PValue value = fulfillValue(requestedType, passingKind)) {
        if (failed(setParamValueAndVerify(value, requestedType)))
          return {{}, fitness};
        ++posBindingIdx;
        continue;
      }
      // We tried but couldn't infer an unbound parameter, we must error.
      emitInferenceFailure(idx);
      return {{}, fitness};
    }

    // If this value was already bound and checked, use it.
    /// FIXME: Remove this, why is this needed?
    if (posBindingIdx < numPreTypeChecked) {
      if (failed(setParamValueAndVerify(bindingVal, requestedType)))
        return {{}, fitness};
      ++posBindingIdx;
      continue;
    }

    // Disallow implicit parameters to be explicitly specified. If we see one,
    // complain about too many parameters.
    if (passingKind == PassingKind::Implicit) {
      auto &diag = getDiag({});
      diag << "callee";
      emitWrongArgOrParamCount(diag, countNumPosOnly(paramListAttr),
                               countNumPositional(paramListAttr),
                               operands.getNumPositional(),
                               "positional parameter");
      return {{}, fitness};
    }

    // This lambda hides the diagnostic and error handling logic for checking a
    // single positional parameter binding.
    auto handlePosBinding = [&](size_t index, ASTExprAnd<AnyValue> binding,
                                ASTType expectedType) -> PValue {
      // If the parameter list expected a keyword only parameter, we have too
      // many positional parameters.
      if (passingKind == PassingKind::KwOnly) {
        auto &diag = getDiag({});
        diag << "callee";
        emitWrongArgOrParamCount(diag, countNumPosOnly(paramListAttr),
                                 countNumPositional(paramListAttr),
                                 operands.getNumPositional(),
                                 "positional parameter");
        return {};
      }

      PValue pValue =
          emitSingleParameterValue(binding, expectedType, emitter, evaluator);
      if (!pValue)
        emitTypeMismatch(index, binding, expectedType);
      return pValue;
    };

    // Scalar parameter values are installed directly.
    if (!paramListAttr.isPosVarArg(idx)) {
      PValue paramValue = handlePosBinding(idx, binding, expectedType);
      if (!paramValue)
        return {{}, fitness};
      if (failed(setParamValueAndVerify(paramValue, expectedType)))
        return {{}, fitness};
      ++posBindingIdx;
      continue;
    }

    // If the parameter is a variadic list, it may consume many values, and they
    // all get packed up into a VariadicAttr.
    // fitness.hasVariadicParams = true;

    // Unpacked variadics can be passed directly as a whole variadic parameter.
    if (auto unpacked =
            dyn_cast<UnpackedAttr>(binding.ir.getIfPValue().get())) {
      PValue paramValue = handlePosBinding(
          idx, {PValue(unpacked.getValue()), binding.expr}, requestedType);
      if (!paramValue)
        return {{}, fitness};
      if (failed(setParamValueAndVerify(paramValue, requestedType)))
        return {{}, fitness};
      ++posBindingIdx;
      continue;
    }

    SmallVector<TypedAttr> elements;
    do {
      auto &binding = operands[posBindingIdx++];
      if (binding.keyword)
        continue;

      PValue pValue = handlePosBinding(idx, binding, expectedType);
      if (!pValue)
        return {{}, fitness};

      // Realign sugar.
      if (pValue.getType().mlirType != expectedType)
        pValue = ParamOperatorAttr::getRebind(pValue, expectedType);

      elements.emplace_back(pValue);
      // Passing `_` to a variadic is not allowed. Users should pass `*_` to
      // unbind a variadic parameter.
      if (isa<UnboundAttr>(elements.back())) {
        auto &diag = getDiag(binding.expr->getLoc());
        diag << "unbound syntax (i.e. `_`) cannot be passed as a variadic "
                "parameter";
        return {{}, fitness};
      }
    } while (posBindingIdx != numBindings);

    auto varType = VariadicType::get(evaluator.getReboundType(expectedType));
    if (failed(setParamValueAndVerify(VariadicAttr::get(elements, varType),
                                      varType)))
      return {{}, fitness};
  }

  // Complain if we collected any missing keyword-only parameters.
  if (!kwDiagNames.empty()) {
    emitMissing(getDiag({}), kwDiagNames, "keyword-only parameter");
    return {{}, fitness};
  }

  // Check and complain if we have bindings that didn't get used.
  if (posBindingIdx != numBindings) {
    // Hide the implicit trait parameter from the diagnostic.
    size_t hidden = 0;
    if (declIfDirect) {
      if (auto fn = dyn_cast<FnOp>(declIfDirect->getIfOperation()))
        hidden = isa_and_nonnull<TraitDeclOp>(fn->getParentOp());
    }
    size_t numExpected = expectedParamTypes.size() - hidden -
                         countNumImplicitKinds(paramListAttr) -
                         countNumInferredKinds(paramListAttr);
    auto &diag = getDiag({});
    if (declIfDirect)
      diag << "'" << *declIfDirect->getUserNameIfOperation() << "'";
    else
      diag << "parametric value";
    emitWrongArgOrParamCount(diag, /*minRequired=*/numExpected,
                             /*maxAllowed=*/numExpected,
                             operands.getNumPositional() - hidden, "parameter");

    return {{}, fitness};
  }

  return {ParameterExprArrayAttr::get(emitter.getContext(), newBindings),
          fitness};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(
    LITGeneratorType sig,
    llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag,
    ParamInfState &inference, ParamInfDiags &inferenceDiags,
    ASTDecl *declIfKnown) const {
  return verifyBindingsImpl(parameters, sig.getInputParamTypes(),
                            sig.getMetadata(), inference, inferenceDiags,
                            getDiag,
                            /*partial=*/false, declIfKnown);
}

ParameterExprArrayAttr
ParamBindings::tryVerifyBindings(ArrayRef<Type> paramTypes,
                                 PogListAttr paramList, bool partial) const {
  // The inference diagnostics will be unused.
  ParamInfDiags inferenceDiags;
  ParamInfState inference(declScope, getParameters(), getNumPreCheckedParams(),
                          paramTypes, paramList, inferenceDiags,
                          /*allowImplicitConversions=*/true);
  std::optional<MojoInflightDiag> diag;
  auto [bindings, _] = verifyBindingsImpl(
      parameters, paramTypes, paramList, inference, inferenceDiags,
      [&](std::optional<SMLoc> loc) -> MojoInflightDiag & {
        // Ignore any errors.
        diag = shared.emitError(loc ? *loc : getExprLoc());
        diag->abandon();
        return *diag;
      },
      partial, /*declIfDirect=*/nullptr);
  return bindings;
}

ParameterExprArrayAttr
ParamBindings::verifyStructBindings(ASTDecl &structDecl, TypeSignatureType sig,
                                    bool partial) const {
  auto [bindingValuesAttr, fitness, diag] = verifyBindingsWithDiag(
      sig.getParamTypes(), sig.getParamListAttrs(), &structDecl, partial);

  if (diag) {
    diag->attachNote(structDecl.getLoc())
        << "'" << *structDecl.getUserNameIfOperation() << "' declared here";
    return {};
  }

  // Emit diagnostics for unprovable constraints if no other diagnostics were
  // emitted.
  if (!fitness.unprovableConstraints.empty() && !diag) {
    emitUnprovableConstraintsFromFitness(fitness, shared, getExprLoc(),
                                         &structDecl);
  }
  return bindingValuesAttr;
}

ParameterExprArrayAttr
ParamBindings::verifyBindings(LITGeneratorType sig,
                              ASTDecl *declIfKnown) const {
  auto [newBindings, fitness, diag] = verifyBindingsWithDiag(
      sig.getInputParamTypes(), sig.getMetadata(), declIfKnown,
      /*partial=*/true);

  if (declIfKnown && diag) {
    assert(isa<FnOp>(declIfKnown->getIfOperation()));
    diag->attachNote(declIfKnown->getLoc()) << "function declared here";
    return {};
  }

  // Emit diagnostics for unprovable constraints if no other diagnostics were
  // emitted.
  if (!fitness.unprovableConstraints.empty() && !diag) {
    emitUnprovableConstraintsFromFitness(fitness, shared, getExprLoc(),
                                         declIfKnown);
  }
  return newBindings;
}

std::tuple<ParameterExprArrayAttr, ParamBindings::Fitness,
           std::optional<MojoInflightDiag>>
ParamBindings::verifyBindingsWithDiag(ArrayRef<Type> expectedParamTypes,
                                      PogListAttr paramListAttr,
                                      ASTDecl *declIfKnown,
                                      bool partial) const {
  ParamInfDiags inferenceDiags;
  ParamInfState inference(declScope, getParameters(), getNumPreCheckedParams(),
                          expectedParamTypes, paramListAttr, inferenceDiags,
                          /*allowImplicitConversions=*/true);
  std::optional<MojoInflightDiag> diag;
  auto getDiags = [&](std::optional<SMLoc> loc) -> MojoInflightDiag & {
    diag = shared.emitError(loc ? *loc : getExprLoc());
    return *diag;
  };
  auto [bindings, fitness] = verifyBindingsImpl(
      parameters, expectedParamTypes, paramListAttr, inference, inferenceDiags,
      getDiags, partial, declIfKnown);
  return {bindings, fitness, std::move(diag)};
}

TypedAttr ParamBindings::getBoundConstAttrForFn(ASTDecl &fnDecl) const {
  auto funcOp = cast<FnOp>(fnDecl.getIfOperation());
  FnTypeGeneratorType signature = funcOp.getFullSignature();

  // If this is a global function or struct reference, bind it directly.
  auto parentTrait = dyn_cast<TraitDeclOp>(funcOp->getParentOp());
  if (!parentTrait) {
    // If there are no parameters specified and if we allow unbound symbols,
    // just return the unbound symbol.
    if (empty())
      return funcOp.getBoundReference(shared.getEvaluationContext());

    // Check that the signature can be rebound with our set of bindings.
    ParameterExprArrayAttr newBindings = verifyBindings(signature, &fnDecl);
    if (!newBindings)
      return {};

    // Now that we checked the types match, form the binding.
    return funcOp.getBoundReference(shared.getEvaluationContext(), newBindings);
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

  ASTDecl *traitDecl = selfExpr.getType().getDecl(shared);
  signature = substituteTraitAliasesIntoSignature(
      *shared.declResolver, *traitDecl, funcOp, signature, selfExpr);

  signature = signature.getSpecializedGenerator(
      paramValues, &shared.getEvaluationContext(), [&]() {
        return mlir::emitError(shared.translateLocation(getExprLoc()))
               << "internal error: ";
      });
  assert(signature && "Error binding trait Self type");

  auto traitName =
      StringAttr::get(funcOp.getContext(),
                      getFlattenedSymbolName(funcOp.getInheritedFrom().value_or(
                          traitDecl->getSymbolRef())));
  TypedAttr fnRef = shared.getEvaluationContext().getGetWitnessAttr(
      selfExpr, traitName, funcOp.getSymNameAttr(), signature);

  if (bindings.empty())
    return fnRef;

  // Attempt to partially bind the parameters to the signature of the function.
  ParameterExprArrayAttr newBindings =
      bindings.verifyBindings(signature, &fnDecl);
  if (!newBindings)
    return {};

  return BindParamsAttr::get(fnRef, newBindings,
                             &shared.getEvaluationContext());
}

void ParamBindings::dump() const { llvm::errs() << parameters << "\n"; }
