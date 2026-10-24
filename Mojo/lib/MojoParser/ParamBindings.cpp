//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "ParamBindings.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "ParamInf.h"
#include "ParserEvaluationContext.h"
#include "Traits.h"

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/IRValues.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/STLExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

void LIT::emitUnprovableConstraintsFromFitness(
    ArrayRef<ConstraintAttr> unprovableConstraints, SharedState &shared,
    SMLoc exprLoc, ASTDecl *declIfKnown) {
  if (unprovableConstraints.empty())
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
        << "cannot prove constraint" << plural(unprovableConstraints.size());
  for (auto constraint : unprovableConstraints)
    LIT::emitConstraintInconclusive(shared.getDeclResolver(), diag, constraint);
}

//===----------------------------------------------------------------------===//
// ParamBindings
//===----------------------------------------------------------------------===//

ParamBindings::ParamBindings(ASTDecl &declScope, const ExprNode *expr)
    : declScope(declScope), shared(declScope.getShared()),
      parameters(CallSyntax::kParamBindings, expr, EC_InvalidContext) {}

/// Replace our bindings with another set.  This can't be done with operator=
/// because we have
void ParamBindings::operator=(ParamBindings &&other) {
  parameters = std::move(other.parameters);
  additionalConstraints = std::move(other.additionalConstraints);
  bindingKind = other.bindingKind;
}

SMLoc ParamBindings::getExprLoc() const { return getExpr()->getLoc(); }

void ParamBindings::add(const ExprNode *expr, AnyValue value, StringAttr name) {
  assert(!isa_and_nonnull<EllipsisAttr>(value.getIfPValue().get()) &&
         "ellipsis is not allowed as a binding value, set the BindingKind to "
         "kWithEllipsis instead");
  parameters.add(name, {value, expr},
                 name ? ArgUnpackStyle::kKeyword : ArgUnpackStyle::kPositional);
}

void ParamBindings::append(const ParamBindings &other) {
  llvm::append_range(parameters.values, other.parameters.values);
  llvm::append_range(additionalConstraints, other.additionalConstraints);
  bindingKind = other.bindingKind;
}

static bool isReferenceFromOwnFnBody(ASTDecl &fnDecl, ASTDecl &useScope) {
  auto *defDecl = dyn_cast<DefASTDecl>(&fnDecl);
  FnOp funcOp = defDecl ? defDecl->getOp() : nullptr;
  if (!funcOp || !isa_and_nonnull<FnOp>(funcOp->getParentOp()))
    return false;

  for (ASTDecl *scope = &useScope; scope; scope = scope->getParentDecl()) {
    if (scope == &fnDecl)
      return true;
  }
  return false;
}

/// Utility function to perform substitutions of the bindings into the symbol
/// for the given function declaration. It returns the resultant
/// SymbolConstantAttr or produces an error message and returns null.
TypedAttr LIT::getBoundConstAttrForFn(ASTDecl &fnDecl, SharedState &shared,
                                      const VerifiedParamBindings &verified,
                                      ASTDecl &useScope, SMLoc useLoc) {
  if (auto *defDecl = dyn_cast<DefASTDecl>(&fnDecl); defDecl) {
    FnOp funcOp = defDecl->getOp();
    if (isReferenceFromOwnFnBody(fnDecl, useScope)) {
      StringRef name = funcOp.getDeclName().getValue();
      auto diag = shared.emitError(
                      useLoc, "recursive references to nested functions are "
                              "not supported; define '")
                  << name << "' at file scope";
      diag.attachNote(fnDecl) << "nested function declared here";
      return {};
    }

    assert(!isa<TraitDeclOp>(funcOp->getParentOp()) &&
           "trait witness should always retrieved from a witness entry");
    return funcOp.getFuncLiteralGenerator(
        shared.getEvaluationContext(), verified.asAttr(),
        verified.getDischargedBodyConstraints());
  }

  auto *witnessEntry = dyn_cast<WitnessASTDecl>(&fnDecl);
  assert(witnessEntry);

  ArrayRef<TypedAttr> verifiedValues = verified.getValues();
  // Must at least have one `_Self` parameter.
  assert(!verifiedValues.empty());

  TypedAttr selfExpr = verifiedValues[0];
  auto signature =
      cast<FnTypeGeneratorType>(witnessEntry->getWitnessEntry().witnessType);

  SmallVector<TypedAttr> paramValues;
  paramValues.push_back(selfExpr);
  for (Type t : signature.getInputParamTypes().drop_front())
    paramValues.push_back(UnboundAttr::get(t));

  TraitSymbolAttr witnessFor = witnessEntry->traitSymbol;

  // Get the signature with only `_Self` bound.
  signature = signature.getSpecializedGenerator(paramValues,
                                                &shared.getEvaluationContext());

  TypedAttr fnRef = shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
      selfExpr, witnessFor, witnessEntry->getWitnessEntry().witnessName,
      signature);

  ArrayRef<TypedAttr> remainingParamValues = verifiedValues.drop_front();
  const llvm::BitVector &discharged = verified.getDischargedBodyConstraints();
  DenseBoolArrayAttr dischargedAttr =
      KGEN::getDenseBoolArrayAttr(fnRef.getContext(), discharged);
  return BindParamsAttr::get(fnRef.getContext(), fnRef, remainingParamValues,
                             dischargedAttr, &shared.getEvaluationContext());
}

TypedAttr LIT::getBoundConstAttrForFn(ASTDecl &fnDecl,
                                      const ParamBindings &unverified) {
  VerifiedParamBindings verifiedBindings;
  if (!unverified.empty()) {
    FnTypeGeneratorType signature = fnDecl.getDeclFullSignature();
    // Check that the signature can be rebound with our set of bindings.
    ParamInf inference(unverified, signature.getInputParamTypes(),
                       signature.getParamListAttrs(),
                       /*allowImplicitConversions=*/true, &fnDecl,
                       /*discardError=*/false);
    verifiedBindings = inference.inferForStruct();
    if (!verifiedBindings)
      return {};
  }
  return getBoundConstAttrForFn(fnDecl, unverified.shared, verifiedBindings,
                                unverified.declScope, unverified.getExprLoc());
}

void ParamBindings::dump() const { llvm::errs() << parameters << "\n"; }
