//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/Constraints.h"
#include "MojoUtils.h"
#include "ParamBindings.h"
#include "Traits.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/DeclSignaturePrinter.h"
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
  TypedAttr prop = constraint.getProposition();

  // First point to the constraint declaration and explain what it folded into.
  diag.attachNote(constraint.getLoc())
      << "constraint declared here needs evidence for " << prop;

  // Walk the proposition to look for signs of inconclusiveness.
  TypedAttr canonProp = getCanonicalAttr(prop);
  canonProp.walk([&](ParamOperatorAttr op) {
    // If the constraint involves a function call, it must be inconclusive
    // because it calls a function that is not always_inline("builtin").
    if (op.getOpcode() == POC::Apply) {
      auto callee = op.getOperand(0);
      if (auto sym = dyn_cast<SymbolConstantAttr>(callee)) {
        ASTDecl *calleeDecl = resolver.getDeclForFuncSymbol(sym.getSymbol());
        diag.attachNote(*calleeDecl)
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
    ParameterEvaluator *evaluator,
    ArrayRef<ConstraintAttr> additionalAssumptions) {
  if (constraints.empty())
    return ConstraintResult::Satisfied;

  SmallVector<ConstraintAttr> assumptions;
  declScope.getKnownAssumptionsIncludingParents(assumptions);
  // Add any additional assumptions passed in (e.g., conformance constraint).
  assumptions.append(additionalAssumptions.begin(),
                     additionalAssumptions.end());

  SmallVector<TypedAttr> overallAssumptionOperands;
  for (ConstraintAttr assumption : assumptions) {
    TypedAttr prop = assumption.getProposition();
    TypedAttr canonProp = getCanonicalAttr(prop);
    overallAssumptionOperands.push_back(canonProp);
  }
  // Combine contextual assumptions, or use trivial true when there are none so
  // constant constraints still resolve via inferConstraintRelation (handles
  // weakening rules, conjunction elimination, negation, etc.).
  TypedAttr overallAssumption;
  if (overallAssumptionOperands.size() == 1) {
    // Single assumption: no need to wrap in an AND.
    overallAssumption = overallAssumptionOperands.front();
  } else if (!overallAssumptionOperands.empty()) {
    overallAssumption =
        ParamOperatorAttr::get(POC::And, overallAssumptionOperands);
  } else {
    MLIRContext *ctx = constraints.front().getContext();
    overallAssumption = SIMDAttr::getScalarBool(ctx, true);
  }

  SmallVector<std::pair<size_t, ConstraintAttr>> failedConstraints;
  SmallVector<ConstraintAttr> localUnprovableConstraints;
  for (auto [idx, constraint] : llvm::enumerate(constraints)) {
    TypedAttr origProp = constraint.getProposition();
    if (evaluator)
      origProp = evaluator->getReboundAttribute(origProp);
    TypedAttr canonProp = getCanonicalAttr(origProp);

    // If assumptions imply the constraint, skip it; if they contradict it,
    // treat it as violated.
    switch (inferConstraintRelation(overallAssumption, canonProp)) {
    case ConstraintRelation::Implies:
      continue;
    case ConstraintRelation::Contradicts:
      failedConstraints.emplace_back(idx, constraint);
      continue;
    case ConstraintRelation::Unprovable:
      break;
    }

    // Unprovable constraint.
    localUnprovableConstraints.push_back(
        ConstraintAttr::get(origProp, constraint.getLoc()));
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
      diag.attachNote(constraint.getLoc(), prop)
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

/// Rewrite cond(a, b, a) patterns to and(a, b) for constraint propositions.
/// This handles patterns from "and"/"or" operators to make constraints
/// decomposable:
///   cond(a, b, a) -> and(a, b)  // from "a and b"
///   cond(a, a, b) -> or(a, b)   // from "a or b"
/// Also distributes struct field extraction through cond operations:
///   struct.extract(cond(a, b, c), field) ->
///     and/or(struct.extract(b, field), struct.extract(c, field))
/// Recursively processes nested cond operations to handle "a and b and c".
TypedAttr LIT::deShortCircuitCond(TypedAttr value) {
  assert(isScalarOf<KGENDType::kBool>(value.getType()) &&
         "expected scalar<bool>");

  // Check if this is a struct.extract on top of a cond.
  LIT::StructExtractAttr extractAttr;
  TypedAttr innerValue = getCanonicalAttr(value);
  if (auto extract = dyn_cast<LIT::StructExtractAttr>(innerValue)) {
    extractAttr = extract;
    innerValue = extract.getStructValue();
  }

  // Check if the inner value is a cond operation.
  auto paramOp = dyn_cast<ParamOperatorAttr>(innerValue);
  if (!paramOp || paramOp.getOpcode() != POC::Cond)
    return value; // Not a cond, return unchanged.

  ArrayRef<TypedAttr> operands = paramOp.getOperands();
  assert(operands.size() == 3 && "cond should have 3 operands");

  TypedAttr condOrig = operands[0];
  TypedAttr trueVal = operands[1];
  TypedAttr falseVal = operands[2];

  // If we had a field extraction, apply it to each operand.
  if (extractAttr) {
    trueVal = LIT::StructExtractAttr::get(trueVal, extractAttr.getField(),
                                          extractAttr.getType());
    falseVal = LIT::StructExtractAttr::get(falseVal, extractAttr.getField(),
                                           extractAttr.getType());
  }

  POC opcode;
  if (condOrig == falseVal) {
    opcode = POC::And;
  } else if (condOrig == trueVal) {
    opcode = POC::Or;
  } else {
    // Not a pattern we recognize. Return unchanged.
    return value;
  }

  // Recursively rewrite nested conds in all operands.
  trueVal = deShortCircuitCond(trueVal);
  falseVal = deShortCircuitCond(falseVal);

  TypedAttr logicalOp = ParamOperatorAttr::get(opcode, {trueVal, falseVal});
  // Use Preserved sugar to indicate this is an internal transformation for
  // preserving nested sugar.
  TypedAttr sugarOp = SugarAttr::getPreserved(value, logicalOp);
  return sugarOp;
}

bool LIT::attrConformsToTraitUnderAssumptions(
    TypedAttr actualAttr, TraitType expectedTrait, SharedState &shared,
    ArrayRef<ConstraintAttr> assumptions) {
  ASTType actualType(actualAttr.getType());
  if (actualType.checkConformance(expectedTrait, shared, {}) ==
      ConformanceResult::Yes) {
    return true;
  }

  // Keep original identity when possible: for type parameters this may be a
  // ParamDeclRefAttr, while canonicalized forms can lose that.
  TypedAttr typeExpr = extractParamDeclRef(actualAttr);
  if (!typeExpr)
    typeExpr = getCanonicalAttr(actualAttr);

  TraitType refinedBound =
      getTraitBoundFromAssumptions(typeExpr, shared, assumptions);
  if (!refinedBound)
    return false;

  return ASTType(refinedBound).checkConformance(expectedTrait, shared, {}) ==
         ConformanceResult::Yes;
}
