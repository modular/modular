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

static TypedAttr stripStructExtractFromBool(TypedAttr prop) {
  if (auto sugar = dyn_cast<SugarAttr>(prop)) {
    // If the prop is sugared, attempt to strip from both sides, but it's highly
    // likely only the sugared form will be un-canonicalized (and therefore
    // strippable). In that case, rebind the sugared form to have the same type
    // as the original input.
    TypedAttr sugared = stripStructExtractFromBool(sugar.getSugared());
    TypedAttr expanded = stripStructExtractFromBool(sugar.getExpanded());
    if (sugared.getType() != expanded.getType())
      sugared = ParamOperatorAttr::getRebind(sugared, expanded.getType());
    return SugarAttr::get(prop.getContext(), sugar.getKind(),
                          sugar.getMemberName(), sugared, expanded);
  }

  if (auto extract = dyn_cast<LIT::StructExtractAttr>(prop))
    if (extract.getField() == "_mlir_value")
      return extract.getStructValue();

  return prop;
}

bool LIT::constraintImplies(TypedAttr propA, TypedAttr propB) {
  // Canonicalize both to remove sugar and get structural forms.
  propA = getCanonicalAttr(propA);
  propB = getCanonicalAttr(propB);

  // Trivially true is implied by anything.
  if (auto intB = dyn_cast<IntegerAttr>(propB))
    if (intB.getValue().isOne())
      return true;
  // Direct equality: A implies A.
  if (propA == propB)
    return true;

  // Weakening rule: A implies (A OR B) for any B.
  // If propB is an OR and propA matches or implies any operand, we're done.
  if (auto paramOpB = dyn_cast<ParamOperatorAttr>(propB)) {
    if (paramOpB.getOpcode() == POC::Or) {
      for (Attribute operand : paramOpB.getOperands()) {
        if (constraintImplies(propA, cast<TypedAttr>(operand)))
          return true;
      }
    }
  }

  // Conjunction elimination: (A AND B) implies A, (A AND B) implies B.
  // If propA is an AND and any operand implies propB, we're done.
  if (auto paramOpA = dyn_cast<ParamOperatorAttr>(propA)) {
    if (paramOpA.getOpcode() == POC::And) {
      for (Attribute operand : paramOpA.getOperands()) {
        if (constraintImplies(cast<TypedAttr>(operand), propB))
          return true;
      }
    }
  }

  // Fallback: canonicalization trick - A implies B iff AND(A, B) == A.
  TypedAttr combined = ParamOperatorAttr::get(POC::And, {propA, propB});
  return combined == propA;
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
    prop = getCanonicalAttr(prop);

    // If the constraint evaluated to a constant, check its value directly.
    if (auto intValue = dyn_cast<IntegerAttr>(prop)) {
      if (intValue.getValue().isZero())
        failedConstraints.emplace_back(idx, constraint);
      continue;
    }

    // If there are contextual assumptions, and the constraint is implied by
    // them, skip it. Use constraintImplies for better implication checking
    // (handles weakening rules, conjunction elimination, etc.)
    if (overallAssumption && constraintImplies(overallAssumption, prop))
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

      prop = stripStructExtractFromBool(prop);
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
  TypedAttr sugarOp = SugarAttr::getAlias(value, logicalOp);
  return sugarOp;
}
