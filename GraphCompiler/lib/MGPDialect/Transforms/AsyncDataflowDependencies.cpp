//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Transforms/AsyncDataflowDependencies.h"

#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "MLRT/GraphRT/GRTDialect/GRTExtendsLifetimeInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTMayAliasInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/EquivalenceClasses.h"

using namespace M;
using namespace M::MGP;
using namespace std::placeholders;

#define DEBUG_TYPE "mgp-dataflow"

/// Calls the helper for each operand/result pair the op's alias or
/// lifetime-extension interface relates.
static bool considerMaybeSameAs(Operation *op, size_t operandIndex,
                                size_t resultIndex) {
  if (auto extends = dyn_cast<GRT::GRTExtendsLifetimeInterface>(op))
    return extends.extendsLifetime(operandIndex, resultIndex);
  else if (auto mayAlias = dyn_cast<GRT::GRTMayAliasInterface>(op))
    return mayAlias.mayAlias(operandIndex, resultIndex);
  else
    return false;
}

static bool considerMaybeSameAsForAnyOperand(Operation *op,
                                             size_t resultIndex) {
  for (size_t operandIndex = 0, numOperands = op->getNumOperands();
       operandIndex < numOperands; ++operandIndex) {
    if (considerMaybeSameAs(op, operandIndex, resultIndex))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

AsyncDataflowDependencies::AsyncDataflowDependencies(
    IndexSet<Value> *valueIndex, IndexSet<Operation *> *operationIndex,
    AnalysisLevel analysisLevel,
    llvm::unique_function<bool(Operation *)> exclusiveOpsFilter)
    : valueAnalysis(valueIndex), operationAnalysis(operationIndex),
      sameAsEquivalence(valueIndex->identityEquivalence()),
      maybeSameAsEquivalence(valueIndex->identityEquivalence()),
      alwaysReadyPredicate(valueIndex->emptySet()),
      exclusiveOpsFilter(std::move(exclusiveOpsFilter)) {
  setupSameAs();
  setupMaybeSameAs();
  setupAlwaysReady();
  setupImpliesReady();
  if (analysisLevel >= kDisjoint) {
    setupStructurallyDisjoint();
    setupRunsLater();
  }
  if (analysisLevel >= kExclusive) {
    setupExclusive();
  }
}

bool AsyncDataflowDependencies::sameAs(Value lhs, Value rhs) const {
  return valueAnalysis.index->areEquivalent(sameAsEquivalence, lhs, rhs);
}

bool AsyncDataflowDependencies::maybeSameAs(Value lhs, Value rhs) const {
  return valueAnalysis.index->areEquivalent(maybeSameAsEquivalence, lhs, rhs);
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::maybeSameAsSet(Value lhs) const {
  return valueAnalysis.index->getEquivalenceClass(maybeSameAsEquivalence, lhs);
}

bool AsyncDataflowDependencies::isAlwaysReady(Value value) const {
  return valueAnalysis.index->setContains(alwaysReadyPredicate, value);
}

const AsyncDataflowDependencies::ValueSet &
AsyncDataflowDependencies::getAlwaysReady() const {
  return alwaysReadyPredicate;
}

bool AsyncDataflowDependencies::impliesReady(Value lhs, Value rhs) const {
  return lhs == rhs ||
         valueAnalysis.containsEntry(impliesReadyRelation, lhs, rhs);
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::downwardsImpliesReadySet(Value rhs) const {
  return valueAnalysis.getEntriesAndId(inverseImpliesReadyRelation, rhs);
}

const AsyncDataflowDependencies::ValueSet &
AsyncDataflowDependencies::strictUpwardsImpliesReadySet(Value lhs) const {
  return valueAnalysis.getEntries(impliesReadyRelation, lhs);
}

void AsyncDataflowDependencies::simplify(ValueSet &values) const {
  // No need to wait for always-ready values, or for values implied ready by
  // another value in the set.
  values.intersectWithComplement(getAlwaysReady());

  for (auto value : valueAnalysis.index->values(values)) {
    values.intersectWithComplement(strictUpwardsImpliesReadySet(value));
  }
}

bool AsyncDataflowDependencies::areDisjoint(Operation *lhsOp,
                                            Operation *rhsOp) const {
  return operationAnalysis.containsEntry(runsLaterRelation, lhsOp, rhsOp) ||
         operationAnalysis.containsEntry(runsEarlierRelation, lhsOp, rhsOp) ||
         operationAnalysis.containsEntry(structurallyCannotOverlapRelation,
                                         lhsOp, rhsOp);
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::exclusiveSet(Value indicator) const {
  return valueAnalysis.getEntries(exclusiveRelation, indicator);
}

bool AsyncDataflowDependencies::isNonSelfOverlapping(Value value) const {
  Operation *op = value.getDefiningOp();
  if (!op) {
    // We can only talk about op results.
    return false;
  }

  if (isAtMostOnce(op)) {
    // Structurally impossible to execute op more than once.
    return true;
  }

  ValueSet dead = exclusiveSet(value);

  // The only way value could still be alive is if some alias of it is
  // fed back via some enclosing kLoop control flow op, and that alias
  // is not dead when value is defined.
  return llvm::all_of(allLoopBoundArgs(op), [&](Value arg) {
    return !maybeSameAs(value, arg) ||
           valueAnalysis.index->setContains(dead, arg);
  });
}

void AsyncDataflowDependencies::print(llvm::raw_ostream &os) {
  valueAnalysis.printRelation(os, "sameAs", sameAsRelation);
  valueAnalysis.printRelation(os, "maybeSameAs", maybeSameAsRelation);
  valueAnalysis.printEquivalence(os, "maybeSameAsEquivalence",
                                 maybeSameAsEquivalence);
  valueAnalysis.printPredicate(os, "isAlwaysReady", alwaysReadyPredicate);
  valueAnalysis.printRelation(os, "impliesReady", impliesReadyRelation);
  operationAnalysis.printRelation(os, "structurallyAtMostOne",
                                  structurallyAtMostOneRelation);
  operationAnalysis.printRelation(os, "structurallyCannotOverlapRelation",
                                  structurallyCannotOverlapRelation);
  operationAnalysis.printRelation(os, "runsLater", runsLaterRelation);
  operationAnalysis.printRelation(os, "runsEarlier", runsEarlierRelation);
  valueAnalysis.printRelation(os, "exclusive", exclusiveRelation);
}

//===----------------------------------------------------------------------===//
// Core relations
//===----------------------------------------------------------------------===//

// Rules are just the control flow rules.
void AsyncDataflowDependencies::flowSameAs(Operation *op,
                                           ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowSameAs, this, _1, _2));
  } else {
    valueAnalysis.flowDefault(op, domainMap);
  }
}

void AsyncDataflowDependencies::setupSameAs() {
  LLVM_DEBUG(llvm::dbgs() << "setupSameAs\n");
  ValueAnalysis::DomainMap domainMap(
      sameAsRelation, ValueAnalysis::kLubIsIntersection,
      valueAnalysis.index->emptySet(), valueAnalysis.index->fullSet());
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowSameAs, this, _1, _2));
  valueAnalysis.addRelationToEquivalence(sameAsEquivalence, sameAsRelation);
}

// Rules are:
//  - Given
//      ..., %lhs, ... = op (..., %rhs, ...)
//    where op has the GRTMayAliasInterface or GRTExtendsLifetimeInterface,
//    then maybeSameAs(%lhs, %rhs) if the interface method returns true for
//    those operands and result indexes. maybeSameAs does not distinguish
//    aliasing from lifetime extension.
//  - The control flow rules.
void AsyncDataflowDependencies::flowMaybeSameAs(Operation *op,
                                                ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowMaybeSameAs, this, _1, _2));
  } else if (isa<GRT::GRTExtendsLifetimeInterface>(op) ||
             isa<GRT::GRTMayAliasInterface>(op)) {
    // Lifetime extension is approximated as possible aliasing, giving the
    // related results and arguments the same worst-case lifetime.
    for (size_t resultIndex = 0, numResults = op->getNumResults();
         resultIndex < numResults; ++resultIndex) {
      ValueSet set = domainMap.bottom;
      for (size_t operandIndex = 0, numOperands = op->getNumOperands();
           operandIndex < numOperands; ++operandIndex) {
        if (considerMaybeSameAs(op, operandIndex, resultIndex))
          domainMap.leastUpperBound(
              set, valueAnalysis.getEntriesAndId(domainMap,
                                                 op->getOperand(operandIndex)));
      }
      valueAnalysis.setEntries(domainMap, op->getResult(resultIndex), set);
    }
  } else {
    valueAnalysis.flowDefault(op, domainMap);
  }
}

void AsyncDataflowDependencies::setupMaybeSameAs() {
  LLVM_DEBUG(llvm::dbgs() << "setupMaybeSameAs\n");
  ValueDomainMap domainMap(maybeSameAsRelation, ValueAnalysis::kLubIsUnion,
                           valueAnalysis.index->emptySet(),
                           valueAnalysis.index->fullSet());
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowMaybeSameAs, this, _1, _2));
  valueAnalysis.addRelationToEquivalence(maybeSameAsEquivalence,
                                         maybeSameAsRelation);
}

/// Implements the rules:
///  - Given
///      %x = grt.chain.create (%y1, ..., %yn)
///    where isAlwaysReady(%yi) for all i then isAlwaysReady(%x)
///  - The control flow rules.
void AsyncDataflowDependencies::flowAlwaysReady(Operation *op,
                                                ValueDomainMap &domainMap) {
  if (auto chainCreateOp = dyn_cast<GRT::GRTChainCreateOpInterface>(op)) {
    // This is equivalent to:
    //   if (isTrue(o) for all o in mergeOp.getOperands())
    //     setTrue(mergeOp.getResult())
    // but written to mirror how flowControlFlow works with the relation.
    ValueSet set = domainMap.bottom;
    for (Value operand : chainCreateOp.getInputChains()) {
      domainMap.leastUpperBound(
          set, valueAnalysis.getEntriesAndId(domainMap, operand));
    }
    valueAnalysis.setEntries(domainMap, chainCreateOp.getChainResult(), set);
  } else if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowAlwaysReady, this, _1, _2));
  } else {
    valueAnalysis.flowDefault(op, domainMap);
  }
}

void AsyncDataflowDependencies::setupAlwaysReady() {
  LLVM_DEBUG(llvm::dbgs() << "setupAlwaysReady\n");
  ValueRelation alwaysReadyRelation;
  ValueDomainMap domainMap(
      alwaysReadyRelation, ValueAnalysis::kLubIsIntersection,
      valueAnalysis.index->emptySet(),
      valueAnalysis.index->unitSet(valueAnalysis.index->distinguished));
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowAlwaysReady, this, _1, _2));
  valueAnalysis.addToPredicate(alwaysReadyPredicate, alwaysReadyRelation);
}

/// Implements the rules:
///  - Given
///      ..., %lhs, ... = op(..., %rhs, ...)
///    where %lhs is an 'indicating result' and %rhs is a 'gating operand'
///    then impliesReady(%lhs, %rhs)
///  - The control flow rules
///  - Given
///      ..., %lhs, ... = grt.if %rhs (...) ...
///    then impliesReady(%lhs, %rhs), since condition must have been ready
///    for grt.if to return.
///  - Given
///      ..., %lhs, ... = grt.repeat.i32 %rhs (...) ...
///    then impliesReady(%lhs, %rhs), since trip count must have been ready
///    for grt.repeat.i32 to return.
///  - Given
///      ..., %lhs, ... = grt.repeat.i32 ... {
///                         ...
///                         mef.output ..., %rhs, ...
///                       }
///    then impliesReady(%lhs, %rhs), since grt.repeat.i32 waits for all
///    loop body outputs to be ready after splicing it into the execution
///    graph.
///  - Given
///      ..., %lhs, ... = grt.while ... {
///                         ...
///                       } do {
///                         ...
///                         mef.output ..., %rhs, ...
///                      }
///    then impliesReady(%lhs, %rhs), since grt.while waits for all
///    loop body outputs to be ready after splicing it into the execution
///    graph.
void AsyncDataflowDependencies::flowImpliesReady(Operation *op,
                                                 ValueDomainMap &domainMap) {
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    valueAnalysis.flowControlFlow(
        cfOp, domainMap,
        std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
    // Union in the additional relation entries described above.
    for (Value lhs : cfOp->getResults()) {
      ValueSet set = valueAnalysis.getEntries(domainMap, lhs);
      // Condition/trip counts are strict.
      for (Value rhs : cfOp.getStrictExtraOperands()) {
        domainMap.greatestLowerBound(
            set, valueAnalysis.getEntriesAndId(domainMap, rhs));
      }
      if (cfOp.getFlavor() == GRT::kLoopLike) {
        // Loop body outputs are known to be ready.
        Block *lastBlock = &cfOp->getRegions().back().front();
        for (Value rhs : lastBlock->getTerminator()->getOperands()) {
          domainMap.greatestLowerBound(
              set, valueAnalysis.getEntriesAndId(domainMap, rhs));
        }
      }
      valueAnalysis.setEntries(domainMap, lhs, set);
    }
  } else {
    // Default 'gating' to 'indicating' dependency rule.
    ValueSet set = domainMap.top;
    for (Value rhs : getGatingOperands(op)) {
      domainMap.greatestLowerBound(
          set, valueAnalysis.getEntriesAndId(domainMap, rhs));
    }
    for (Value lhs : getIndicatingResults(op))
      valueAnalysis.setEntries(domainMap, lhs, set);
  }
}

void AsyncDataflowDependencies::setupImpliesReady() {
  LLVM_DEBUG(llvm::dbgs() << "setupImpliesReady\n");
  // Given:
  //    %a = if ... {
  //      %w = ...
  //      mef.output %x
  //    } else {
  //      %y = ...
  //      mef.output %z
  //    }
  // where impliesReady(%z, %y), we have two possibly interpretations of
  // impliesReady(%a, %y):
  //  - the relation can only hold if %y was actually created, thus it
  //    must be false in case the 'then' sub-block was taken.
  //  - the relation is allowed to hold 'vacuously' if %y was never created,
  //    thus can be true.
  //
  // We take the second (handleDisjointBranches) since it yields more
  // runsEarlier entries, and in turn more exclusive entries.
  ValueDomainMap domainMap(
      impliesReadyRelation, ValueAnalysis::kLubIsIntersection,
      valueAnalysis.index->emptySet(), valueAnalysis.index->fullSet(),
      /*handleDisjointBranches=*/true);
  valueAnalysis.flowModel(
      valueAnalysis.index->model, domainMap,
      std::bind(&AsyncDataflowDependencies::flowImpliesReady, this, _1, _2));
  inverseImpliesReadyRelation = valueAnalysis.inverse(impliesReadyRelation);
}

void AsyncDataflowDependencies::addCrossProduct(
    AsyncDataflowDependencies::OperationRelation &relation,
    const SmallVector<SmallVector<Operation *>> &opsPerSubBlock) {
  // For all pairs of distinct blocks...
  size_t n = opsPerSubBlock.size();
  for (size_t i = 0; i < n; ++i) {
    size_t numLhs = opsPerSubBlock[i].size();
    for (size_t j = 0; j < n; ++j) {
      if (i == j)
        continue;
      size_t numRhs = opsPerSubBlock[j].size();
      // For all pairs of ops from disjoint blocks...
      for (size_t k = 0; k < numLhs; ++k) {
        Operation *lhsOp = opsPerSubBlock[i][k];
        for (size_t l = 0; l < numRhs; ++l) {
          Operation *rhsOp = opsPerSubBlock[j][l];
          operationAnalysis.add(relation, lhsOp, rhsOp);
        }
      }
    }
  }
}

void AsyncDataflowDependencies::setupStructurallyDisjoint() {
  LLVM_DEBUG(llvm::dbgs() << "setupStructurallyDisjoint\n");

  /// The structurallyAtMostOneRelation is just for kBranches.
  structurallyAtMostOneRelation = operationAnalysis.emptyRelation();
  valueAnalysis.index->model->walk([&](GRT::GRTControlFlowInterface cfOp) {
    if (cfOp->getNumRegions() <= 1 || cfOp.getFlavor() != GRT::kBranchLike) {
      // Not a multi-armed kBranch.
      return;
    }

    if (!isAtMostOnce(cfOp)) {
      // kBranch could be executed more than once.
      return;
    }

    // Collect ops which can be executed at most once in each of the sub-blocks
    // (including nested sub-blocks).
    SmallVector<SmallVector<Operation *>> opsPerBranch;
    for (Region &region : cfOp->getRegions()) {
      SmallVector<Operation *> &allOps = opsPerBranch.emplace_back();
      region.walk([&](Operation *op) {
        if (getIndicatingResults(op).empty()) {
          // We are only interested in ops with indicating results since that's
          // the only way we have to track their execution.
          return;
        }
        if (!isAtMostOnce(op)) {
          // Sub-block containing op could be executed more than once.
          return;
        }
        allOps.emplace_back(op);
      });
    }

    addCrossProduct(structurallyAtMostOneRelation, opsPerBranch);
  });

  structurallyCannotOverlapRelation = operationAnalysis.emptyRelation();
  valueAnalysis.index->model->walk([&](GRT::GRTControlFlowInterface cfOp) {
    if (cfOp->getNumRegions() <= 1)
      // Not a multi-subblock control flow op.
      return;

    // Collect all the ops from each sub-block (including nested sub-blocks).
    // We don't care whether these ops can be executed multiple times due to
    // loops since even if so they cannot overlap in execution.
    SmallVector<SmallVector<Operation *>> allOpsPerSubBlock;
    for (Region &region : cfOp->getRegions()) {
      SmallVector<Operation *> &allOps = allOpsPerSubBlock.emplace_back();
      region.walk([&](Operation *op) {
        if (getIndicatingResults(op).empty()) {
          // We are only interested in ops with indicating results since that's
          // the only way we have to track their execution.
          return;
        }
        allOps.emplace_back(op);
      });
    }

    addCrossProduct(structurallyCannotOverlapRelation, allOpsPerSubBlock);
  });
}

AsyncDataflowDependencies::OperationSet
AsyncDataflowDependencies::runsLaterSet(Operation *earlierOp) const {
  OperationSet result = operationAnalysis.index->emptySet();
  // For each result of earlier op which indicates op has FINISHED...
  for (Value indicatingResult : getIndicatingResults(earlierOp)) {
    // Find all values which imply the indicatingResult is ready (including
    // the indicatingResult itself).
    ValueSet set = downwardsImpliesReadySet(indicatingResult);
    // Any use of a value in that set as a gating operand forces the
    // using operation to BEGIN only after earlierOp has FINISHED.
    for (Value user : valueAnalysis.index->values(set)) {
      for (OpOperand &opOperand : user.getUses()) {
        if (opOperand.getOwner()->getNumResults() == 0) {
          // Ignore ops with no results since we can't track them.
          continue;
        }
        if (isGatingOperand(&opOperand))
          operationAnalysis.index->addToSet(result, opOperand.getOwner());
      }
    }
  }
  return result;
}

void AsyncDataflowDependencies::setupRunsLater() {
  LLVM_DEBUG(llvm::dbgs() << "setupRunsLater\n");
  for (Operation *op : operationAnalysis.index->getAllVs()) {
    if (op == operationAnalysis.index->distinguished)
      continue;
    OperationSet set = runsLaterSet(op);
    runsLaterRelation.insert({op, set});
  }
  runsEarlierRelation = operationAnalysis.inverse(runsLaterRelation);
}

//===----------------------------------------------------------------------===//
// 'exclusiveSet'
//
// Defined from maybeSameAsRelation (possible aliasing), impliesReadyRelation
// (causal ordering of values), and runsEarlierRelation (causal ordering of
// ops, derived from impliesReady):
//
//   1. Given "%x = op ..."
//   2. Find all the ops "... = op %y" which must already be finished when op
//      begins (runsEarlier).
//   3. Find all the values %z which may be aliases of %y (maybeSameAs).
//   4. If all uses of %z are by ops in 2 then we know %x is exclusive to %z.
//   5. If impliesReady(%a, %z) we can say the same for %a (impliesReady).
//
// Note the use of two closures: 'upwards' for the earlier ops, and 'downwards'
// for the ready values.
//
// For SSA values inside kLoop ops, step 3 can additionally exploit two facts:
//   1. The SSA value bound by an op may be considered a 'fresh' instance,
//      distinct from previous values bound in earlier loop iterations.
//   2. Any in-flight loop has not yet allowed its loop results to escape
//      into the rest of the graph.
//
// So step 3 needs a 'program point' for the whole execution graph state, not
// just 'ready AsyncValue', to tell 'previous' from 'next' iterations.
//
// TODO(#15840): Alternatively define 'exclusiveSet' as its own dataflow
// analysis, as liveness traditionally is; it would likely still need
// something like getFullyConsumedWhenOpBegins.
//
//===----------------------------------------------------------------------===//

Operation *
AsyncDataflowDependencies::getEffectiveOwner(const OpOperand *opOperand) const {
  Operation *op = opOperand->getOwner();

  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    ArrayRef<OpOperand> inputOpOperands = cfOp.getInputOpOperands();
    if (contains_if(inputOpOperands, [&](const OpOperand &inputOpOperand) {
          return &inputOpOperand == opOperand;
        })) {
      // Inputs to control flow ops are just plumbing.
      return nullptr;
    }
  }

  if (isa<MEF::OutputOp>(op)) {
    if (auto modelOp = dyn_cast<ModelOp>(op->getParentOp())) {
      // The model op stands for the caller of the entry point.
      return modelOp;
    }

    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op->getParentOp())) {
      if (cfOp.getFlavor() == GRT::kLoopLike) {
        // The loop op stands for future iterations.
        return cfOp;
      }
    }

    // Operands to mef.output outside of loops and the top-level are just
    // plumbing.
    return nullptr;
  }

  return op;
}

bool AsyncDataflowDependencies::allUsesAreByOperationsInSet(
    const ValueSet &values, const OperationSet &ops) const {
  for (Value value : valueAnalysis.index->values(values)) {
    for (OpOperand &opOperand : value.getUses()) {
      Operation *owningOp = getEffectiveOwner(&opOperand);
      if (!owningOp) {
        // Ignore 'plumbing' uses.
        continue;
      }
      if (!isGatingOperand(&opOperand)) {
        // Unknown when the op is done with the operand, so the use is live.
        LLVM_DEBUG({
          valueAnalysis.index->printV(llvm::dbgs(), value);
          llvm::dbgs() << " not consumed since non-gating arg to ";
          operationAnalysis.index->printV(llvm::dbgs(), owningOp);
          llvm::dbgs() << "\n";
        });
        return false;
      }
      if (!operationAnalysis.index->setContains(ops, owningOp)) {
        // A relevant use outside the set keeps the operand live.
        LLVM_DEBUG({
          valueAnalysis.index->printV(llvm::dbgs(), value);
          llvm::dbgs() << " not consumed since used by unfinished op ";
          operationAnalysis.index->printV(llvm::dbgs(), owningOp);
          llvm::dbgs() << "\n";
        });
        return false;
      }
    }
  }
  return true;
}

AsyncDataflowDependencies::ValueEquivalence
AsyncDataflowDependencies::getEquivalenceAtOp(Operation *op) const {
  // 'Fresh' results cannot alias any operand.
  ValueSet freshResultValues = valueAnalysis.index->emptySet();
  if ((isa<GRT::GRTExtendsLifetimeInterface>(op) ||
       isa<GRT::GRTMayAliasInterface>(op))) {
    for (size_t resultIndex = 0, numResults = op->getNumResults();
         resultIndex < numResults; ++resultIndex) {
      if (!considerMaybeSameAsForAnyOperand(op, resultIndex))
        valueAnalysis.index->addToSet(freshResultValues,
                                      op->getResult(resultIndex));
    }
  } else {
    for (Value result : op->getResults()) {
      valueAnalysis.index->addToSet(freshResultValues, result);
    }
  }

  ValueSet loopArgs = valueAnalysis.index->emptySet();
  for (Value arg : allLoopBoundArgs(op))
    valueAnalysis.index->addToSet(loopArgs, arg);

  ValueSet loopResults = valueAnalysis.index->emptySet();
  for (Value res : allLoopResults(op))
    valueAnalysis.index->addToSet(loopResults, res);

  LLVM_DEBUG({
    llvm::dbgs() << "freshResultValues = ";
    valueAnalysis.index->printVs(
        llvm::dbgs(), valueAnalysis.index->setToVector(freshResultValues));
    llvm::dbgs() << "\n";
    llvm::dbgs() << "loopArgs = ";
    valueAnalysis.index->printVs(llvm::dbgs(),
                                 valueAnalysis.index->setToVector(loopArgs));
    llvm::dbgs() << "\n";
    llvm::dbgs() << "loopResults = ";
    valueAnalysis.index->printVs(llvm::dbgs(),
                                 valueAnalysis.index->setToVector(loopResults));
    llvm::dbgs() << "\n";
  });

  // Close maybeSameAs, ignoring entries (%lhs, ...) where %lhs is a loop
  // result (the in-flight loop has not let the alias escape), and (%lhs, %rhs)
  // where %lhs is a loop arg and %rhs a fresh result (not the previous
  // iteration's value).
  ValueEquivalence equivalence = valueAnalysis.index->identityEquivalence();
  for (const auto &[lhs, set] : maybeSameAsRelation) {
    if (valueAnalysis.index->setContains(loopResults, lhs))
      continue;
    for (auto rhs : valueAnalysis.index->values(set)) {
      if (valueAnalysis.index->setContains(loopArgs, lhs) &&
          valueAnalysis.index->setContains(freshResultValues, rhs))
        continue;
      valueAnalysis.index->setEquivalent(equivalence, lhs, rhs);
    }
  }

  return equivalence;
}

AsyncDataflowDependencies::ValueSet
AsyncDataflowDependencies::getFullyConsumedWhenOpBegins(Operation *op) const {
  LLVM_DEBUG({
    llvm::dbgs() << "getFullyConsumedWhenOpBegins(";
    operationAnalysis.index->printV(llvm::dbgs(), op);
    llvm::dbgs() << ")\n";
  });

  OperationSet earlierOps =
      operationAnalysis.getEntries(runsEarlierRelation, op);

  LLVM_DEBUG({
    llvm::dbgs() << "earlierOps = ";
    operationAnalysis.index->printVs(
        llvm::dbgs(), operationAnalysis.index->setToVector(earlierOps));
    llvm::dbgs() << "\n";
  });

  ValueSet candidates = valueAnalysis.index->emptySet();
  for (Operation *earlierOp : operationAnalysis.index->values(earlierOps)) {
    for (Value value : earlierOp->getOperands())
      valueAnalysis.index->addToSet(candidates, value);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "candidates = ";
    valueAnalysis.index->printVs(llvm::dbgs(),
                                 valueAnalysis.index->setToVector(candidates));
    llvm::dbgs() << "\n";
  });

  // It is the underlying AsyncValues that must have no more references, so
  // expand each candidate to its maybeSameAs class. Inside a kLoop sub-block,
  // what precedes op is a 'previous' iteration and what follows the 'next',
  // so entries from tying the loop knot can be ignored.
  std::unique_ptr<ValueEquivalence> equivalenceAtOp;
  const ValueEquivalence *equivalence;
  if (isAtMostOnce(op)) {
    equivalence = &maybeSameAsEquivalence;
  } else {
    equivalenceAtOp =
        std::make_unique<ValueEquivalence>(getEquivalenceAtOp(op));
    equivalence = equivalenceAtOp.get();
  }

  LLVM_DEBUG({
    valueAnalysis.printEquivalence(llvm::dbgs(), "equivalence", *equivalence);
  });

  // A class is dead only if ALL its values are consumed; expanding and
  // filtering candidates one at a time is unsound.
  ValueSet result = valueAnalysis.index->emptySet();
  for (auto candidate : candidates) {
    ValueSet maybeSameValues = valueAnalysis.index->emptySet();
    for (auto equivItr = equivalence->findLeader(candidate),
              endEquivItr = equivalence->member_end();
         equivItr != endEquivItr; ++equivItr)
      maybeSameValues.set(*equivItr);
    LLVM_DEBUG({
      llvm::dbgs() << "considering candidate ";
      valueAnalysis.index->printV(llvm::dbgs(),
                                  valueAnalysis.index->getV(candidate));
      llvm::dbgs() << " which is equivalent to ";
      valueAnalysis.index->printVs(
          llvm::dbgs(), valueAnalysis.index->setToVector(maybeSameValues));
      llvm::dbgs() << "\n";
    });
    if (allUsesAreByOperationsInSet(maybeSameValues, earlierOps)) {
      LLVM_DEBUG({
        llvm::dbgs() << "include because consumed ";
        valueAnalysis.index->printVs(
            llvm::dbgs(), valueAnalysis.index->setToVector(maybeSameValues));
        llvm::dbgs() << "\n";
      });
      result |= maybeSameValues;
    }
  }

  return result;
}

void AsyncDataflowDependencies::setupExclusive() {
  LLVM_DEBUG(llvm::dbgs() << "setupExclusive\n");
  exclusiveRelation = valueAnalysis.emptyRelation();

  for (Operation *op : operationAnalysis.index->getAllVs()) {
    if (op == operationAnalysis.index->distinguished)
      continue;

    if (!exclusiveOpsFilter(op)) {
      // This op is not of interest.
      continue;
    }

    if (op->getNumResults() == 0) {
      // No results to index the consumedSet by.
      continue;
    }

    // Values fully consumed (or never created) when op BEGINS.
    ValueSet consumedSet = getFullyConsumedWhenOpBegins(op);

    for (Value consumedValue : valueAnalysis.index->values(consumedSet)) {
      if (consumedValue.getDefiningOp() &&
          !exclusiveOpsFilter(consumedValue.getDefiningOp())) {
        // The consumed value is the result of an op which is not of interest.
        valueAnalysis.index->removeFromSet(consumedSet, consumedValue);
      }
    }

    // Any value which implies an indicating result of op is ready (including
    // the indicating result itself) can't overlap with any value in
    // consumedSet.
    for (Value indicatingResult : getIndicatingResults(op)) {
      auto downward = downwardsImpliesReadySet(indicatingResult);
      for (Value alsoReadyIndicatingResult :
           valueAnalysis.index->values(downward)) {
        if (alsoReadyIndicatingResult.getDefiningOp() &&
            !exclusiveOpsFilter(alsoReadyIndicatingResult.getDefiningOp())) {
          // Also-ready value is the result of an op which is not of interest.
          continue;
        }

        LLVM_DEBUG({
          llvm::dbgs() << "add because consumed ";
          valueAnalysis.index->printV(llvm::dbgs(), alsoReadyIndicatingResult);
          llvm::dbgs() << " |-> ";
          valueAnalysis.index->printVs(
              llvm::dbgs(), valueAnalysis.index->setToVector(consumedSet));
          llvm::dbgs() << "\n";
        });
        valueAnalysis.addAll(exclusiveRelation, alsoReadyIndicatingResult,
                             consumedSet);
      }
    }
  }

  // The above only works backwards from values bound by ops; close under
  // symmetry.
  valueAnalysis.unionInPlace(exclusiveRelation,
                             valueAnalysis.inverse(exclusiveRelation));

  // Results of distinct branch arms are never simultaneously alive. This needs
  // structurallyAtMostOneRelation, not structurallyCannotOverlapRelation: ops
  // that never overlap but run repeatedly can have overlapping results.
  //
  // Unlike the per-op loop above, downwardsImpliesReadySet is unsound here:
  //   %x = lhsOp ...
  //   %y = rhsOp ...
  //   (lhsOp, rhsOp) in structurallyAtMostOneRelation
  //   %z = otherOp ...
  //   impliesReady(%z, %x)
  // It is NOT the case that:
  //   (%y, %z) in exclusiveRelation.
  for (Operation *op : operationAnalysis.index->getAllVs()) {
    if (op == operationAnalysis.index->distinguished)
      continue;

    if (!exclusiveOpsFilter(op)) {
      // This op is not of interest.
      continue;
    }

    for (Operation *disjointOp : operationAnalysis.index->values(
             operationAnalysis.getEntries(structurallyAtMostOneRelation, op))) {
      if (!exclusiveOpsFilter(disjointOp)) {
        // The disjoint op is not of interest.
        continue;
      }

      for (Value opResult : op->getResults()) {
        for (Value disjointOpResult : disjointOp->getResults()) {
          LLVM_DEBUG({
            llvm::dbgs() << "add because disjoint ";
            valueAnalysis.index->printV(llvm::dbgs(), opResult);
            llvm::dbgs() << " |-> ";
            valueAnalysis.index->printV(llvm::dbgs(), disjointOpResult);
            llvm::dbgs() << "\n";
          });
          valueAnalysis.add(exclusiveRelation, opResult, disjointOpResult);
        }
      }
    }
  }
}
