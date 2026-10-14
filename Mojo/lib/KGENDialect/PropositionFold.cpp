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
//
// Reduce a conjunction of already-canonical proposition leaves by inserting
// one clause at a time into a folded vector.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/PropositionFold.h"
#include "Mojo/KGENDialect/KGENAttrInterfaces.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"

using namespace M;
using namespace KGEN;

static ClauseInsertResult combine(ClauseInsertResult lhs,
                                  ClauseInsertResult rhs) {
  if (lhs == ClauseInsertResult::Contradiction ||
      rhs == ClauseInsertResult::Contradiction)
    return ClauseInsertResult::Contradiction;
  if (lhs == ClauseInsertResult::Added || rhs == ClauseInsertResult::Added)
    return ClauseInsertResult::Added;
  return ClauseInsertResult::Redundant;
}

static TypedAttr getNotOperand(TypedAttr prop) {
  auto xorOp = dyn_cast<ParamOperatorAttr>(prop);
  if (!xorOp || xorOp.getOpcode() != POC::Xor ||
      xorOp.getOperands().size() != 2)
    return {};

  for (auto [maybeInner, maybeTrue] :
       {std::pair{xorOp.getOperand(0), xorOp.getOperand(1)},
        std::pair{xorOp.getOperand(1), xorOp.getOperand(0)}}) {
    if (isTriviallyTrueProposition(maybeTrue))
      return maybeInner;
  }
  return {};
}

static bool sameTypeValue(TypedAttr lhs, TypedAttr rhs) {
  return isEqualCanon(stripIdentityWrappers(lhs), stripIdentityWrappers(rhs));
}

// Trait symbol lists never contain duplicates, which the size shortcut and the
// conformance phase's strict-subset reasoning both rely on.
static bool traitSubset(ArrayRef<TraitSymbolAttr> need,
                        ArrayRef<TraitSymbolAttr> have) {
  if (need.size() > have.size())
    return false;
  DenseSet<TraitSymbolAttr> haveSet(have.begin(), have.end());
  return llvm::set_is_subset(need, haveSet);
}

static void flattenAnd(TypedAttr prop, SmallVectorImpl<TypedAttr> &out) {
  if (auto op = sugarDynCast<ParamOperatorAttr>(prop);
      op && op.getOpcode() == POC::And) {
    llvm::append_range(out, op.getOperands());
    return;
  }
  out.push_back(prop);
}

static bool classContainsAll(ArrayRef<TypedAttr> eqClass,
                             ArrayRef<TypedAttr> need) {
  llvm::SmallDenseSet<Attribute, 8> members;
  for (TypedAttr member : eqClass)
    members.insert(stripIdentityWrappers(member));
  return llvm::all_of(need, [&](TypedAttr member) {
    return members.contains(stripIdentityWrappers(member));
  });
}

static bool classesOverlap(ArrayRef<TypedAttr> lhs, ArrayRef<TypedAttr> rhs) {
  llvm::SmallDenseSet<Attribute, 8> members;
  for (TypedAttr member : lhs)
    members.insert(stripIdentityWrappers(member));
  return llvm::any_of(rhs, [&](TypedAttr member) {
    return members.contains(stripIdentityWrappers(member));
  });
}

/// Whether `clauses` prove `goal`. Sound but incomplete: false means "not
/// proved", not "disproved". A stored OR never proves one of its disjuncts.
/// `clauses` never holds True or False, so there is no ex falso case; an AND
/// among them (only a NOT's inner) is in fold form, so one level of flattening
/// suffices.
static bool entailedBy(ArrayRef<TypedAttr> clauses, TypedAttr goal) {
  goal = SugarAttr::strip(goal);
  if (isTriviallyTrueProposition(goal))
    return true;
  if (isTriviallyFalseProposition(goal))
    return false;

  SmallVector<TypedAttr> flat;
  for (TypedAttr clause : clauses)
    flattenAnd(clause, flat);
  clauses = flat;

  if (auto op = sugarDynCast<ParamOperatorAttr>(goal)) {
    if (op.getOpcode() == POC::And)
      return llvm::all_of(op.getOperands(), [&](TypedAttr operand) {
        return entailedBy(clauses, operand);
      });
    if (op.getOpcode() == POC::Or)
      return llvm::any_of(op.getOperands(), [&](TypedAttr operand) {
        return entailedBy(clauses, operand);
      });
  }

  // Contrapositive: stored `not(s)` proves `not(g)` when `g` implies `s`.
  if (TypedAttr inner = getNotOperand(goal)) {
    return llvm::any_of(clauses, [&](TypedAttr clause) {
      TypedAttr storedInner = getNotOperand(clause);
      return storedInner && entailedBy({inner}, storedInner);
    });
  }

  if (std::optional<ArrayRef<TypedAttr>> goalClass = getIdentityClass(goal)) {
    return llvm::any_of(clauses, [&](TypedAttr clause) {
      std::optional<ArrayRef<TypedAttr>> stored = getIdentityClass(clause);
      return stored && classContainsAll(*stored, *goalClass);
    });
  }

  // The goal's traits may be split across several stored slots on the same
  // type value, so their symbols are unioned. A bound whose symbols cannot be
  // enumerated is proved only by an exact restatement.
  if (auto goalCT = dyn_cast<TypeConformsToTraitAttr>(goal)) {
    std::optional<ArrayRef<TraitSymbolAttr>> need = goalCT.getTraitSymbols();
    if (!need)
      return llvm::is_contained(clauses, goal);
    SmallVector<TraitSymbolAttr> have;
    for (TypedAttr clause : clauses) {
      auto storedCT = dyn_cast<TypeConformsToTraitAttr>(clause);
      if (!storedCT ||
          !sameTypeValue(storedCT.getTypeValue(), goalCT.getTypeValue()))
        continue;
      if (std::optional<ArrayRef<TraitSymbolAttr>> symbols =
              storedCT.getTraitSymbols())
        have.append(symbols->begin(), symbols->end());
    }
    return traitSubset(*need, have);
  }

  if (llvm::is_contained(clauses, goal))
    return true;

  // A folded positive fact makes a stored OR obsolete when it proves a
  // disjunct, which is handled on the OR-as-goal path above. A stored OR
  // does not prove a disjunct.
  return false;
}

/// Residual inner after dropping conjuncts already entailed by `facts`.
/// Returns null if the inner is fully proved (the NOT is then unsat).
/// `inner` must be in fold form, which the leaf constructors guarantee for any
/// canonical proposition.
static TypedAttr residualNotInner(TypedAttr inner, ArrayRef<TypedAttr> facts) {
  SmallVector<TypedAttr> remaining;
  SmallVector<TypedAttr> conjuncts;
  flattenAnd(inner, conjuncts);
  for (TypedAttr conjunct : conjuncts) {
    if (!entailedBy(facts, conjunct))
      remaining.push_back(conjunct);
  }
  if (remaining.empty())
    return {};
  // Re-interning an untouched fold-form inner rebuilds the same attribute.
  // This runs for every stored NOT on every insert.
  if (remaining.size() == conjuncts.size())
    return inner;
  return foldBoolConjunction(remaining, inner.getType());
}

static void eraseIndices(SmallVectorImpl<TypedAttr> &clauses,
                         ArrayRef<unsigned> indices) {
  SmallVector<unsigned> sorted(indices.begin(), indices.end());
  llvm::sort(sorted);
  sorted.erase(llvm::unique(sorted), sorted.end());
  for (unsigned index : llvm::reverse(sorted))
    clauses.erase(clauses.begin() + index);
}

/// Inserts one non-AND leaf. The identity, conformance, NOT and OR phases only
/// queue indices in `erase`, so each sees the same unmodified `clauses`; the
/// erasures are committed together, then stored NOTs are shrunk in place, and
/// the final check decides Redundant or Added. A Contradiction can return
/// after erasures or NOT rewrites, leaving `clauses` partially modified.
static ClauseInsertResult insertLeaf(SmallVectorImpl<TypedAttr> &clauses,
                                     TypedAttr leaf) {
  leaf = SugarAttr::strip(leaf);

  SmallVector<unsigned> erase;

  // Test incoming `leaf` for known patterns.
  if (std::optional<ArrayRef<TypedAttr>> incomingClass =
          getIdentityClass(leaf)) {
    // Incoming identity: merge overlapping classes into one n-ary identical.
    SmallVector<TypedAttr> members(incomingClass->begin(),
                                   incomingClass->end());
    for (auto [index, clause] : llvm::enumerate(clauses)) {
      std::optional<ArrayRef<TypedAttr>> stored = getIdentityClass(clause);
      if (!stored || !classesOverlap(*stored, *incomingClass))
        continue;
      members.append(stored->begin(), stored->end());
      erase.push_back(index);
    }
    TypedAttr merged = ParamIdenticalAttr::get(members);
    if (isTriviallyFalseProposition(merged))
      return ClauseInsertResult::Contradiction;
    if (isTriviallyTrueProposition(merged)) {
      eraseIndices(clauses, erase);
      return ClauseInsertResult::Redundant;
    }
    // A stored class equal to the merge stays, so restating a class ends as
    // Redundant at the final check rather than as a re-add.
    SmallVector<unsigned> absorbed;
    for (unsigned index : erase) {
      if (clauses[index] != merged)
        absorbed.push_back(index);
    }
    erase.swap(absorbed);
    leaf = merged;
  } else if (auto incomingCT = dyn_cast<TypeConformsToTraitAttr>(leaf)) {
    // Incoming conformance: drop stored bounds that the incoming bound
    // subsumes. A subset restatement is redundant. KGEN cannot intern a
    // TraitType union, so incomparable bounds on the same type value stay as
    // separate slots and `entailedBy` treats their symbols as a virtual union.
    // TODO: Move TraitType into KGEN so we can build a union of trait symbols.
    if (std::optional<ArrayRef<TraitSymbolAttr>> incomingSyms =
            incomingCT.getTraitSymbols()) {
      SmallVector<unsigned> weaker;
      for (auto [index, clause] : llvm::enumerate(clauses)) {
        auto storedCT = dyn_cast<TypeConformsToTraitAttr>(clause);
        if (!storedCT ||
            !sameTypeValue(storedCT.getTypeValue(), incomingCT.getTypeValue()))
          continue;
        std::optional<ArrayRef<TraitSymbolAttr>> storedSyms =
            storedCT.getTraitSymbols();
        if (!storedSyms)
          continue;
        if (traitSubset(*incomingSyms, *storedSyms))
          continue;
        if (traitSubset(*storedSyms, *incomingSyms))
          weaker.push_back(index);
      }
      // Stored slots that jointly prove the incoming bound (`T: A` and `T: B`
      // against `T: A & B`) stay, so the insert ends as Redundant.
      if (!weaker.empty() && !entailedBy(clauses, leaf))
        erase.append(weaker.begin(), weaker.end());
    }
  } else if (TypedAttr inner = getNotOperand(leaf)) {
    // Incoming NOT: discharge conjuncts already known, then drop stored NOTs
    // that the residual strictly subsumes.
    TypedAttr residual = residualNotInner(inner, clauses);
    if (!residual)
      return ClauseInsertResult::Contradiction;
    if (residual != inner)
      leaf = ParamOperatorAttr::getNot(residual);
    if (isTriviallyFalseProposition(leaf))
      return ClauseInsertResult::Contradiction;
    if (isTriviallyTrueProposition(leaf)) {
      eraseIndices(clauses, erase);
      return ClauseInsertResult::Redundant;
    }
    if (TypedAttr residualInner = getNotOperand(leaf)) {
      for (auto [index, clause] : llvm::enumerate(clauses)) {
        TypedAttr storedInner = getNotOperand(clause);
        if (!storedInner)
          continue;
        // A stored inner that implies the residual makes the incoming NOT the
        // stronger fact.
        if (entailedBy({storedInner}, residualInner) &&
            residualInner != storedInner)
          erase.push_back(index);
      }
    }
  } else if (auto incomingOr = sugarDynCast<ParamOperatorAttr>(leaf);
             incomingOr && incomingOr.getOpcode() == POC::Or) {
    // Incoming OR: redundant when a disjunct is already known.
    if (llvm::any_of(incomingOr.getOperands(), [&](TypedAttr operand) {
          return entailedBy(clauses, operand);
        })) {
      eraseIndices(clauses, erase);
      return ClauseInsertResult::Redundant;
    }
  }

  // A stored OR is obsolete when the incoming fact proves it.
  for (auto [index, clause] : llvm::enumerate(clauses)) {
    auto storedOr = sugarDynCast<ParamOperatorAttr>(clause);
    if (!storedOr || storedOr.getOpcode() != POC::Or)
      continue;
    if (entailedBy(ArrayRef<TypedAttr>(leaf), clause) ||
        llvm::any_of(storedOr.getOperands(), [&](TypedAttr operand) {
          return entailedBy(clauses, operand) ||
                 entailedBy(ArrayRef<TypedAttr>(leaf), operand);
        }))
      erase.push_back(index);
  }

  eraseIndices(clauses, erase);

  // A newly added positive (or a restated one) can shrink stored NOTs.
  SmallVector<unsigned> notErase;
  for (auto [index, clause] : llvm::enumerate(clauses)) {
    TypedAttr storedInner = getNotOperand(clause);
    if (!storedInner)
      continue;
    // Leave the NOT out of its own facts so it never discharges its own
    // conjuncts or justifies its own erasure.
    SmallVector<TypedAttr> facts(clauses.begin(), clauses.end());
    facts.erase(facts.begin() + index);
    facts.push_back(leaf);
    TypedAttr residual = residualNotInner(storedInner, facts);
    if (!residual)
      return ClauseInsertResult::Contradiction;
    if (residual == storedInner)
      continue;
    TypedAttr shrunk = ParamOperatorAttr::getNot(residual);
    if (isTriviallyFalseProposition(shrunk))
      return ClauseInsertResult::Contradiction;
    if (isTriviallyTrueProposition(shrunk) ||
        llvm::is_contained(clauses, shrunk) || entailedBy(facts, shrunk))
      notErase.push_back(index);
    else
      clauses[index] = shrunk;
  }
  eraseIndices(clauses, notErase);

  if (llvm::is_contained(clauses, leaf) || entailedBy(clauses, leaf))
    return ClauseInsertResult::Redundant;

  clauses.push_back(leaf);
  return ClauseInsertResult::Added;
}

ClauseInsertResult KGEN::insertClause(SmallVectorImpl<TypedAttr> &clauses,
                                      TypedAttr canonicalProp) {
  TypedAttr prop = SugarAttr::strip(canonicalProp);
  if (isTriviallyTrueProposition(prop))
    return ClauseInsertResult::Redundant;
  if (isTriviallyFalseProposition(prop))
    return ClauseInsertResult::Contradiction;

  if (auto op = sugarDynCast<ParamOperatorAttr>(prop);
      op && op.getOpcode() == POC::And) {
    ClauseInsertResult result = ClauseInsertResult::Redundant;
    for (TypedAttr operand : op.getOperands()) {
      result = combine(result, insertClause(clauses, operand));
      if (result == ClauseInsertResult::Contradiction)
        return ClauseInsertResult::Contradiction;
    }
    return result;
  }

  return insertLeaf(clauses, prop);
}

ClauseInsertResult KGEN::testClause(TypedAttr canonicalAssumption,
                                    TypedAttr canonicalGoal) {
  TypedAttr assumption = SugarAttr::strip(canonicalAssumption);
  if (isTriviallyFalseProposition(assumption))
    return ClauseInsertResult::Redundant;
  SmallVector<TypedAttr> clauses;
  if (!isTriviallyTrueProposition(assumption))
    flattenAnd(assumption, clauses);
  return insertClause(clauses, canonicalGoal);
}
