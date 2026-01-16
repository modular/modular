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
          declResolver.shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
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
    : declScope(declScope), shared(declScope.getShared()),
      parameters(CallSyntax::kParamBindings, expr) {}

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

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindingsImpl(const CallOperands &origOperands,
                                  ArrayRef<Type> expectedParamTypes,
                                  PogListAttr paramListAttr,
                                  ParamInf &inference, bool partial) const {

  if (failed(inference.finalizeWithUnbound()))
    return {{}, {}};

  // If succeeded, Simply return all the binding from the inference.
  return {ParameterExprArrayAttr::get(declScope.getContext(),
                                      inference.evaluator.getIndexBindings()),
          {}};
}

std::pair<ParameterExprArrayAttr, ParamBindings::Fitness>
ParamBindings::verifyBindings(LITGeneratorType sig, ParamInf &inference) const {
  return verifyBindingsImpl(parameters, sig.getInputParamTypes(),
                            sig.getMetadata(), inference,
                            /*partial=*/false);
}

ParameterExprArrayAttr
ParamBindings::tryVerifyBindings(ArrayRef<Type> paramTypes,
                                 PogListAttr paramList, bool partial) const {
  std::optional<MojoInflightDiag> diag;
  auto getDiag = [&](std::optional<SMLoc> loc) -> MojoInflightDiag & {
    // Ignore any errors.
    diag = shared.emitError(loc ? *loc : getExprLoc());
    diag->abandon();
    return *diag;
  };

  // The inference diagnostics will be unused.
  ParamInf inference(*this, paramTypes, paramList,
                     /*allowImplicitConversions=*/true, /*partial=*/partial,
                     getDiag,
                     /*declIfDirect=*/nullptr);

  if (failed(inference.inferFromParamList()))
    return nullptr;

  // Check to see if we have ... and remove it from the parameter list.
  bool hasEllipsis = llvm::any_of(parameters.values, [](OperandValue binding) {
    return isa<EllipsisAttr>(binding.ir.getIfPValue().get());
  });

  // FIXME: we also allow using `_` (UnboundAttr) in a very unprincipled
  // way: it can either be used to unbound a particular positional parameter or
  // as a placeholder (unknown value to be inferred). This creates a lot of
  // subtlety in the compiler, we should probably only allow `_` in partial
  // binding context too.
  // For the second use case (placeholder for unknown), we can, e.g., allow
  // users to type
  //
  // var c : ParamType[ , p1, , p2]
  // # instead of
  // var c : ParamType[_, p1, _, p2]

  // ParamInf should have already complained and returned.
  assert(!hasEllipsis || partial && "`...` used in full binding context");

  // If we had a variadic parameter that is unspecified, and no arguments to
  // infer it from, it must be because of an empty variadic list.
  //
  // FIXME: according to the specification, we should only do this when all
  // other parameter is bound.
  if (!hasEllipsis && failed(inference.inferFromDefaults()))
    return nullptr;

  auto [bindings, _] =
      verifyBindingsImpl(parameters, paramTypes, paramList, inference, partial);
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
  std::optional<MojoInflightDiag> diag;
  auto getDiags = [&](std::optional<SMLoc> loc) -> MojoInflightDiag & {
    diag = shared.emitError(loc ? *loc : getExprLoc());
    return *diag;
  };
  ParamInf inference(*this, expectedParamTypes, paramListAttr,
                     /*allowImplicitConversions=*/true, /*partial=*/partial,
                     getDiags, declIfKnown);

  if (failed(inference.inferFromParamList())) {
    return {nullptr, Fitness{std::move(inference.unprovableConstraints)},
            std::move(diag)};
  }

  // Check to see if we have ... and remove it from the parameter list.
  bool hasEllipsis = llvm::any_of(parameters.values, [](OperandValue binding) {
    return isa<EllipsisAttr>(binding.ir.getIfPValue().get());
  });

  // If we had a variadic parameter that is unspecified, and no arguments to
  // infer it from, it must be because of an empty variadic list.
  //
  // FIXME: according to the specification, we should only do this when all
  // other parameter is bound.
  if (!hasEllipsis && failed(inference.inferFromDefaults())) {
    return {nullptr, Fitness{std::move(inference.unprovableConstraints)},
            std::move(diag)};
  }

  auto [bindings, fitness] = verifyBindingsImpl(
      parameters, expectedParamTypes, paramListAttr, inference, partial);
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
  TypedAttr fnRef = shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
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
