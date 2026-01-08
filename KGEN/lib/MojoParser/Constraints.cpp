//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/Constraints.h"
#include "MojoUtils.h"
#include "ParamBindings.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Emit a note explaining why a constraint is inconclusive. The incoming
/// constraint is expected to be the folded form with all input parameters
/// already substituted.
void LIT::emitConstraintInconclusive(DeclResolver &resolver,
                                     MojoInflightDiag &diag,
                                     ConstraintAttr constraint) {
  TypedAttr canonProp = getCanonicalAttr(constraint.getProposition());
  // Strip the outermost conversion from Bool to i1.
  if (auto structExtract = dyn_cast<LIT::StructExtractAttr>(canonProp))
    if (structExtract.getField() == "_mlir_value")
      canonProp = structExtract.getStructValue();

  // First point to the constraint declaration and explain what it folded into.
  diag.attachNote(constraint.getLoc())
      << "constraint declared here needs evidence for " << canonProp;

  // Walk the proposition to look for signs of inconclusiveness.
  canonProp.walk([&](ParamOperatorAttr op) {
    // If the constraint involves a function call, it must be inconclusive
    // because it calls a function that is not always_inline("builtin").
    if (op.getOpcode() == POC::Apply) {
      auto callee = op.getOperand(0);
      if (auto sym = dyn_cast<SymbolConstantAttr>(callee)) {
        ASTDecl *calleeDecl = resolver.getDeclForFuncSymbol(sym.getSymbol());
        diag.attachNote(calleeDecl->getLoc())
            << "cannot evaluate call to non-builtin function declared here";
      }
    }
    return WalkResult::skip();
  });
}

ConstraintResult LIT::checkConstraints(
    ASTDecl &declScope, PogListAttr paramListAttr,
    ArrayRef<ConstraintAttr> constraints,
    ArrayRef<ConstraintAttr> origConstraints,
    llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag,
    SmallVectorImpl<ConstraintAttr> *unprovableConstraints,
    ParameterEvaluator *evaluator) {
  if (constraints.empty())
    return ConstraintResult::Satisfied;

  SmallVector<ConstraintAttr> assumptions;
  declScope.getKnownAssumptionsIncludingParents(assumptions);
  SmallVector<TypedAttr> overallAssumptionOperands;
  for (ConstraintAttr assumption : assumptions) {
    TypedAttr prop = assumption.getProposition();
    TypedAttr canonProp = getCanonicalAttr(prop);
    overallAssumptionOperands.push_back(canonProp);
  }
  // A null overallAssumption means no contextual assumptions.
  TypedAttr overallAssumption;
  if (overallAssumptionOperands.size() == 1) {
    // Single assumption: no need to wrap in an AND.
    overallAssumption = overallAssumptionOperands.front();
  } else if (!overallAssumptionOperands.empty()) {
    overallAssumption =
        ParamOperatorAttr::get(POC::And, overallAssumptionOperands);
  }

  SmallVector<std::pair<size_t, ConstraintAttr>> failedConstraints;
  SmallVector<ConstraintAttr> localUnprovableConstraints;
  for (auto [idx, constraint] : llvm::enumerate(constraints)) {
    TypedAttr prop = constraint.getProposition();
    prop = getCanonicalAttr(prop);
    if (evaluator)
      prop = evaluator->getReboundAttribute(prop);

    // If the constraint evaluated to a constant, check its value directly.
    if (auto intValue = dyn_cast<IntegerAttr>(prop)) {
      if (intValue.getValue().isZero())
        failedConstraints.emplace_back(idx, constraint);
      continue;
    }

    // If there are contextual assumptions, and the constraint is implied by
    // them, skip it.
    if (overallAssumption &&
        ParamOperatorAttr::get(POC::And, {overallAssumption, prop}) ==
            overallAssumption)
      continue;

    // Unprovable constraint.
    localUnprovableConstraints.push_back(
        ConstraintAttr::get(prop, constraint.getLoc()));
  }

  if (!failedConstraints.empty()) {
    MojoInflightDiag &diag = getDiag({});
    diag << "violated constraint" << plural(failedConstraints.size());
    // Use constraints from the original signature since the ones in
    // `signature` have already been substituted with param bindings and
    // will have already been folded into `False`.
    IndexToDeclRefRemapper remapper(paramListAttr);
    for (auto [idx, constraint] : failedConstraints) {
      TypedAttr prop;
      if (!origConstraints.empty())
        prop = origConstraints[idx].getProposition();
      else
        prop = constraint.getProposition();

      diag.attachNote(constraint.getLoc())
          << "constraint declared here evaluated to False, expected "
          << remapper.replace(prop);
    }

    return ConstraintResult::Violated;
  }

  if (!localUnprovableConstraints.empty()) {
    // Populate the out-parameter if provided.
    if (unprovableConstraints)
      unprovableConstraints->append(localUnprovableConstraints.begin(),
                                    localUnprovableConstraints.end());
    return ConstraintResult::Unprovable;
  }

  return ConstraintResult::Satisfied;
}
