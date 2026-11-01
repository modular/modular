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

#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"

#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MGPDialect/Transforms/IndexSet.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "GraphCompiler/MGPDialect/Transforms/RelationDataflowAnalysis.h"

using namespace std::placeholders;
using namespace mlir;

#define DEBUG_TYPE "exec-invariant-code-motion"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_EXECINVARIANTCODEMOTIONPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// Returns op's buffer result if op can be lifted into 'init', else null.
static Value getLiftableBufferOp(Operation *op) {
  if (auto liftableOp = dyn_cast<MGPLiftableBufferCreateOpInterface>(op))
    return liftableOp.getOutBuffer();
  return nullptr;
}

static llvm::SmallVector<Value> getLiftableIndexOutputs(Operation *op) {
  llvm::SmallVector<Value> res;
  if (auto kernelOp = dyn_cast<MOGG::AsyncRoutineOp>(op)) {
    for (Value result : op->getResults()) {
      if (result.getType().isIntOrIndex() ||
          MOSH::ShapeAttr::isDimType(result.getType()))
        res.push_back(result);
    }
  }
  return res;
}

static std::pair<TypedValue<MEF::ChainType>, uint64_t>
cacheIntValue(OpBuilder &builder, Location loc,
              TypedValue<MEF::ChainType> inChain,
              ContextCreateOp contextCreateOp, Value intValue) {
  intValue = dimensionToIndex(builder, loc, intValue);
  uint64_t intSlot = contextCreateOp.getNumSlots();
  contextCreateOp.setNumSlots(intSlot + 1);

  auto cacheOp = IntCacheOp::create(builder, loc, /*inChain=*/inChain,
                                    /*theContext=*/contextCreateOp,
                                    /*value=*/intValue,
                                    /*bIntSlot=*/intSlot);
  return {cacheOp.getOutChain(), intSlot};
}

static Value getCachedIntValue(OpBuilder &builder, Location loc,
                               TypedValue<ContextType> context,
                               TypedValue<MEF::ChainType> inChain,
                               uint64_t intSlot, Type resultType) {
  auto parentOp = builder.getBlock()->getParentOp();
  if (!isa<ModelOp>(parentOp))
    parentOp = parentOp->getParentOfType<MGP::ModelOp>();

  auto modelOp = cast<ModelOp>(parentOp);
  Value intSlotIndex = mlir::index::ConstantOp::create(builder, loc, intSlot);
  if (modelOp.getIsSubgraph()) {
    intSlotIndex = mlir::index::AddOp::create(builder, loc, intSlotIndex,
                                              modelOp.getSubgraphIndex());
  }

  Value intValue =
      IntGetCachedOp::create(builder, loc, builder.getIndexType(),
                             /*inChain=*/inChain, /*theContext=*/context,
                             /*bIntSlot=*/intSlotIndex);
  intValue = indexToDimension(builder, loc, intValue);
  return intValue;
}

/// Caches the integer and dimension results of a lifted op that 'execute'
/// still uses, and replaces those uses with values read back from the cache.
/// A compiled region that also returns a chain takes the chain path, so this
/// runs there too: otherwise its dims would move to 'init' with nothing in
/// 'execute' to read them back.
static void
cacheLiftedIndexResults(InitState &initState, ExecuteState &executeState,
                        Operation *liftedOp,
                        TypedValue<MEF::ChainType> cacheInputChain,
                        SmallVectorImpl<Value> &finalOutChains,
                        function_ref<bool(const OpOperand &)> isUnliftable) {
  for (Value oldIndexOut : getLiftableIndexOutputs(liftedOp)) {
    if (!llvm::any_of(oldIndexOut.getUses(), isUnliftable))
      continue;

    auto [cacheOutChain, currIndexSlot] =
        cacheIntValue(initState.builder, liftedOp->getLoc(), cacheInputChain,
                      initState.contextCreateOp, oldIndexOut);

    finalOutChains.push_back(cacheOutChain);

    auto newIndex = getCachedIntValue(
        executeState.builder, liftedOp->getLoc(), executeState.getContext(),
        executeState.getRootChain(), currIndexSlot, oldIndexOut.getType());

    oldIndexOut.replaceUsesWithIf(newIndex, isUnliftable);
    LLVM_DEBUG(llvm::dbgs()
               << "will lift shape function output and replace old index:\n"
               << oldIndexOut << "\nwith:\n"
               << newIndex << "\nvia cached:\n"
               << cacheOutChain << "\n");
  }
}

/// Returns true if op is liftable but trivial to compute. Such an op is not
/// worth lifting unless another 'init' op needs it: caching and retrieving
/// its result costs more than recomputing it on every 'execute'.
static bool isLiftableTrivialOp(Operation *op) {
  if (isMemoryEffectFree(op))
    return true;

  if (auto liftableOp = dyn_cast<MGPLiftableBufferCreateOpInterface>(op))
    return liftableOp.isTrivial();
  return false;
}

/// Returns op's chain result if op can be lifted into 'init', else null. The
/// chain must then flow into the 'init' output chain, which feeds the
/// 'execute' input chain.
static Value getLiftableChainOp(Operation *op) {
  if (auto createChainOp = dyn_cast<GRT::ChainCreateOp>(op))
    return createChainOp.getResult();
  if (auto regionOp = dyn_cast<MOGG::AsyncRoutineOp>(op)) {
    // TODO(GEX-4344): The 'liftable' interfaces assume an op returns at most
    // one value, which forces this special case for multi-result regions.
    // A chain plus dimension results can move to 'init': the chain is ordered
    // there, and each dimension is cached. Any other extra result has no cache
    // slot, so the region stays in 'execute'. The cache round-trip stores an
    // index and restores a dimension, so a plain integer result would change
    // type on the way back.
    if (regionOp->getNumResults() > 1) {
      if (!isa<MEF::ChainType>(regionOp->getResult(0).getType()))
        return nullptr;
      for (Value result : llvm::drop_begin(regionOp->getResults()))
        if (!MOSH::ShapeAttr::isDimType(result.getType()))
          return nullptr;
      return regionOp->getResult(0);
    }
  }
  if (auto createChainOp = dyn_cast<MGPLiftableChainCreateOpInterface>(op))
    return createChainOp.getOutChain();
  return nullptr;
}

/// Return true if the result of op can be conveyed from the 'init' block
/// to the 'execute' block via the !mgp.context cache or 'execute' block
/// arguments.
static bool canBeConveyed(Operation *op) {
  return getLiftableBufferOp(op) || getLiftableChainOp(op) ||
         !getLiftableIndexOutputs(op).empty();
}

/// Returns true if op's result cannot be cached but op may run in both 'init'
/// and 'execute', which lets downstream ops be lifted too.
static bool isRecomputableOp(Operation *op) { return isMemoryEffectFree(op); }

/// Returns true if the execution of op could be safely lifted or copied to
/// the 'init' block without causing unintended side effects.
static bool canBeLifted(Operation *op) {
  // NOTE(MAXPLAT-70): some external constants are externally mutable (e.g.
  // torch tensors might be mutated by someone outside our stack), so we cannot
  // lift them to the init phase.
  if (auto externalConstOp = dyn_cast<BufferConstantExternalOp>(op))
    if (externalConstOp.getDHasAlias())
      return false;
  return canBeConveyed(op) || isRecomputableOp(op);
}

// The analyses below reuse the 'dense' machinery (per Value/OpOperand) though
// a 'sparse' per-Operation analysis would do. Control flow ops and their
// contents are treated as unliftable, overriding some default dataflow
// results.
// TODO(#24751): Support lifting of and from within control flow.

//===----------------------------------------------------------------------===//
// DependsOnArgsAnalysis
//===----------------------------------------------------------------------===//

using OperationSet = IndexSet<Operation *>::Set;

/// A dense, forwards dataflow analysis to find ops which cannot be lifted
/// from the 'init' to the 'execute' block because:
///  - The op is already considered unliftable by an earlier analysis (if any).
///  - The op depends on an argument to the 'execute' block (other than the
///    !mef.chain and !mgp.context arguments).
///  - The op is a terminator.
///  - The op is, or is within, a control flow op.
///  - The op has side effects which must be observed on each call to the
///    'execute' block, and cannot instead be performed just once in the 'init'
///    block.
///  - The op depends on the result of an unliftable op.
///
/// We consider ops already in the 'init' block to be liftable so that
/// the 'init' block outputs don't taint the first two 'execute' block inputs
/// as being unliftable. This allows us to use the standard dataflow machinery.
///
/// During the analysis a value will be 'true' (ie mapped to a set containing
/// the distinguished value) if it is an 'execute' block arg or produced
/// by an unliftable op.
class DependsOnArgsAnalysis {
public:
  OperationSet unliftableOps;

  DependsOnArgsAnalysis(IndexSet<Value> *index,
                        IndexSet<Operation *> *operationIndex,
                        OperationSet unliftableOps)
      : unliftableOps(std::move(unliftableOps)), operationIndex(operationIndex),
        analysis(index), dependsOnArgsValues(index->emptySet()) {
    setup();
  }

  void print(llvm::raw_ostream &os) {
    analysis.printPredicate(os, "dependsOnArgsValues", dependsOnArgsValues);
    llvm::dbgs() << "unliftable ops: ";
    operationIndex->printVs(llvm::dbgs(),
                            operationIndex->setToVector(unliftableOps));
    llvm::dbgs() << "\n";
  }

private:
  using Analysis = ForwardsRelationDataflowAnalysis;
  using DomainMap = Analysis::DomainMap;

  void flowGeneric(Operation *op, DomainMap &domainMap) {
    bool isAlreadyUnliftable = operationIndex->setContains(unliftableOps, op);
    Analysis::Set set = domainMap.bottom;
    for (Value operand : op->getOperands()) {
      domainMap.leastUpperBound(set,
                                analysis.getEntriesAndId(domainMap, operand));
    }
    bool hasUnliftableOperands =
        set.test(0); // distinguished always has index 0
    bool isNotLiftable = !canBeLifted(op);
    bool inControlFlow = !isExactlyOnce(op);
    bool isUnliftable = isAlreadyUnliftable || hasUnliftableOperands ||
                        isNotLiftable || inControlFlow;
    if (isUnliftable) {
      LLVM_DEBUG({
        llvm::dbgs() << "DependsOnAnalysis: considering ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << " to be unliftable. isAlreadyUnliftable="
                     << isAlreadyUnliftable
                     << ", hasUnliftableOperands=" << hasUnliftableOperands
                     << ", isNotLiftable=" << isNotLiftable
                     << ", inControlFlow=" << inControlFlow << "\n";
      });
      operationIndex->addToSet(unliftableOps, op);
    }
    for (Value result : op->getResults()) {
      if (isUnliftable)
        analysis.setTrue(domainMap, result);
      else
        analysis.setDefault(domainMap, result);
    }
  }

  void flow(Operation *op, DomainMap &domainMap) {
    bool inInitBlock =
        op->getParentRegion()->isAncestor(&analysis.index->model.getInit());
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      analysis.flowControlFlow(
          cfOp, domainMap,
          std::bind(&DependsOnArgsAnalysis::flow, this, _1, _2));
      if (!inInitBlock) {
        // Control flow ops and their results are intrinsically unliftable.
        operationIndex->addToSet(unliftableOps, op);
        for (Value result : op->getResults())
          analysis.setTrue(domainMap, result);
      }
    } else if (isa<MEF::OutputOp>(op)) {
      if (!inInitBlock) {
        operationIndex->addToSet(unliftableOps, op);
      }
    } else if (inInitBlock) {
      // Ignore 'init' block ops.
      analysis.flowDefault(op, domainMap);
    } else {
      flowGeneric(op, domainMap);
    }
  }

  void setup() {
    Analysis::Relation dependsOnArgsRelation;
    DomainMap domainMap(dependsOnArgsRelation, Analysis::kLubIsUnion,
                        analysis.index->emptySet(),
                        analysis.index->unitSet(analysis.index->distinguished));

    // The extra 'execute' args (ie not the !mef.chain, !mgp.context, weights
    // registry, or devices) vacuously depend on 'execute' args.
    for (Value arg : analysis.index->model.getExecuteInputs())
      analysis.setTrue(domainMap, arg);

    analysis.flowModel(analysis.index->model, domainMap,
                       std::bind(&DependsOnArgsAnalysis::flow, this, _1, _2));
    analysis.addToPredicate(dependsOnArgsValues, dependsOnArgsRelation);
  }

  IndexSet<Operation *> *operationIndex;
  Analysis analysis;
  // Computed only for use by print for debugging.
  Analysis::Set dependsOnArgsValues;
};

//===----------------------------------------------------------------------===//
// SuppressTrivialAnalysis
//===----------------------------------------------------------------------===//

/// A dense, backwards dataflow analysis to find ops which cannot be lifted
/// from the 'init' to the 'execute' block because:
///  - the op is already considered unliftable by an earlier analysis.
///  - the op is trivial to compute and only used by unliftable ops, so
///    caching costs about as much as recomputing.
///  - the op's results are unused (DCE will remove it anyway).
///
/// During the analysis an operand will be 'true' (ie mapped to a set
/// containing the distinguished value) its using op is unliftable.
class SuppressTrivialAnalysis {
public:
  OperationSet unliftableOps;

  SuppressTrivialAnalysis(IndexSet<const OpOperand *> *index,
                          IndexSet<Operation *> *operationIndex,
                          OperationSet unliftableOps)
      : unliftableOps(std::move(unliftableOps)), operationIndex(operationIndex),
        analysis(index), suppressTrivialOperands(index->emptySet()) {
    setup();
  }

  void print(llvm::raw_ostream &os) {
    analysis.printPredicate(os, "suppressTrivialOperands",
                            suppressTrivialOperands);
    llvm::dbgs() << "unliftable ops: ";
    operationIndex->printVs(llvm::dbgs(),
                            operationIndex->setToVector(unliftableOps));
    llvm::dbgs() << "\n";
  }

private:
  using Analysis = BackwardsRelationDataflowAnalysis;
  using DomainMap = Analysis::DomainMap;

  void flowGeneric(Operation *op, DomainMap &domainMap) {
    bool isAlreadyUnliftable = operationIndex->setContains(unliftableOps, op);
    bool isTrivial = isLiftableTrivialOp(op);
    Analysis::Set set = domainMap.bottom;
    for (OpOperand &operand : op->getUses()) {
      domainMap.leastUpperBound(set,
                                analysis.getEntriesAndId(domainMap, &operand));
    }
    bool allUsesUnliftable = set.test(0); // distinguished always has index 0
    bool isUnliftable = isAlreadyUnliftable || (isTrivial && allUsesUnliftable);
    if (isUnliftable) {
      LLVM_DEBUG({
        llvm::dbgs() << "SuppressTrivialAnalysis: considering ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << " to be unliftable. isAlreadyUnliftable="
                     << isAlreadyUnliftable << ", isTrivial=" << isTrivial
                     << ", allUsesUnliftable=" << allUsesUnliftable << "\n";
      });
      operationIndex->addToSet(unliftableOps, op);
    }
    for (const OpOperand &opOperand : op->getOpOperands()) {
      if (isUnliftable)
        analysis.setTrue(domainMap, &opOperand);
      else
        analysis.setDefault(domainMap, &opOperand);
    }
  }

  void flow(Operation *op, DomainMap &domainMap) {
    bool inInitBlock =
        op->getParentRegion()->isAncestor(&analysis.index->model.getInit());
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      // Usual control flow rules.
      analysis.flowControlFlow(
          cfOp, domainMap,
          std::bind(&SuppressTrivialAnalysis::flow, this, _1, _2));
      if (!inInitBlock) {
        // Control flow ops are unliftable, so all their operands are too.
        for (const OpOperand &opOperand : op->getOpOperands())
          analysis.setTrue(domainMap, &opOperand);
      }
    } else if (isa<MEF::OutputOp>(op)) {
      // Ignore 'plumbing'.
    } else if (inInitBlock) {
      // Ignore the 'init' block ops.
      analysis.flowDefault(op, domainMap);
    } else {
      flowGeneric(op, domainMap);
    }
  }

  void setup() {
    Analysis::Relation suppressTrivialRelation;
    DomainMap domainMap(suppressTrivialRelation, Analysis::kLubIsIntersection,
                        analysis.index->emptySet(),
                        analysis.index->unitSet(analysis.index->distinguished));

    // All 'execute' block outputs can be considered to be unliftable.
    for (OpOperand &result : analysis.index->model.getExecuteBlock()
                                 ->getTerminator()
                                 ->getOpOperands())
      analysis.setTrue(domainMap, &result);

    analysis.flowModel(analysis.index->model, domainMap,
                       std::bind(&SuppressTrivialAnalysis::flow, this, _1, _2));
    analysis.addToPredicate(suppressTrivialOperands, suppressTrivialRelation);
  }

  IndexSet<Operation *> *operationIndex;
  Analysis analysis;
  // Computed only for use by print for debugging.
  Analysis::Set suppressTrivialOperands;
};

//===----------------------------------------------------------------------===//
// SuppressMutatedAnalysis
//===----------------------------------------------------------------------===//

/// A dense, backwards dataflow analysis to find ops which cannot be lifted
/// from the 'init' to the 'execute' block because:
///  - the op is already considered unliftable by an earlier analysis.
///  - at least one of the op's results is mutated by an op which cannot
///    be lifted. Since values held by the !mgp.context cache are read-only,
///    attempting to lift the op would result in a runtime error.
///
/// During the analysis an operand will be 'true' (ie mapped to a set
/// containing the distinguished value) if it is mutated by an unliftable op.
class SuppressMutatedAnalysis {
public:
  OperationSet unliftableOps;

  SuppressMutatedAnalysis(IndexSet<const OpOperand *> *index,
                          IndexSet<Operation *> *operationIndex,
                          OperationSet unliftableOps)
      : unliftableOps(std::move(unliftableOps)), operationIndex(operationIndex),
        analysis(index), suppressMutatedOperands(index->emptySet()) {
    setup();
  }

  void print(llvm::raw_ostream &os) {
    analysis.printPredicate(os, "suppressMutatedOperands",
                            suppressMutatedOperands);
    llvm::dbgs() << "unliftable ops: ";
    operationIndex->printVs(llvm::dbgs(),
                            operationIndex->setToVector(unliftableOps));
    llvm::dbgs() << "\n";
  }

private:
  using Analysis = BackwardsRelationDataflowAnalysis;
  using DomainMap = Analysis::DomainMap;

  void flowGeneric(Operation *op, DomainMap &domainMap) {
    bool isAlreadyUnliftable = operationIndex->setContains(unliftableOps, op);
    Analysis::Set set = domainMap.bottom;
    for (OpOperand &operand : op->getUses()) {
      domainMap.leastUpperBound(set,
                                analysis.getEntriesAndId(domainMap, &operand));
    }
    bool anyUsesMutatedByUnliftable =
        set.test(0); // distinguished olways has index 0
    bool isUnliftable = isAlreadyUnliftable || anyUsesMutatedByUnliftable;
    if (isUnliftable) {
      LLVM_DEBUG({
        llvm::dbgs() << "SuppressMutatedAnalysis: considering ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << " to be unliftable. isAlreadyUnliftable="
                     << isAlreadyUnliftable << ", anyUsesMutatedByUnliftable="
                     << anyUsesMutatedByUnliftable << "\n";
      });
      operationIndex->addToSet(unliftableOps, op);
    }
    // Only suppress upstream ops whose results this unliftable op *mutates*.
    for (OpOperand &opOperand : op->getOpOperands()) {
      if (isUnliftable && isMutated(opOperand))
        analysis.setTrue(domainMap, &opOperand);
      else
        analysis.setDefault(domainMap, &opOperand);
    }
  }

  void flow(Operation *op, DomainMap &domainMap) {
    bool inInitBlock =
        op->getParentRegion()->isAncestor(&analysis.index->model.getInit());
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      // Usual control flow rules.
      analysis.flowControlFlow(
          cfOp, domainMap,
          std::bind(&SuppressMutatedAnalysis::flow, this, _1, _2));
    } else if (isa<MEF::OutputOp>(op)) {
      // Ignore 'plumbing'.
    } else if (inInitBlock) {
      // Ignore the 'init' block ops.
      analysis.flowDefault(op, domainMap);
    } else {
      flowGeneric(op, domainMap);
    }
  }

  void setup() {
    Analysis::Relation suppressMutatedRelation;
    DomainMap domainMap(suppressMutatedRelation, Analysis::kLubIsUnion,
                        analysis.index->emptySet(),
                        analysis.index->unitSet(analysis.index->distinguished));

    // 'execute' block outputs are not mutated by the output, so stay 'false'.
    analysis.flowModel(analysis.index->model, domainMap,
                       std::bind(&SuppressMutatedAnalysis::flow, this, _1, _2));
    analysis.addToPredicate(suppressMutatedOperands, suppressMutatedRelation);
  }

  IndexSet<Operation *> *operationIndex;
  Analysis analysis;
  // Computed only for use by print for debugging.
  Analysis::Set suppressMutatedOperands;
};

//===----------------------------------------------------------------------===//
// NeedToCloneAnalysis
//===----------------------------------------------------------------------===//

/// A dense, backwards dataflow analysis to find ops which, though liftable,
/// must also be 'left behind' in the 'execute' block by cloning. The op must
/// be cloned if it is i) liftable, and ii) its results cannot be conveyed via
/// the !mgp.context or 'execute' block args, and iii) any:
///  - it provides operands to an unliftable op
///  - it provides operands to an op already considered needing to be cloned
///    by this analysis.
///
/// During the analysis an operand will be 'true' (ie mapped to a set
/// containing the distinguished value) if it is an operand to an unliftable
/// op or an op already recognized as needing to be cloned.
class NeedToCloneAnalysis {
public:
  OperationSet clonedOps;

  NeedToCloneAnalysis(IndexSet<const OpOperand *> *index,
                      IndexSet<Operation *> *operationIndex,
                      const OperationSet &unliftableOps)
      : clonedOps(operationIndex->emptySet()), operationIndex(operationIndex),
        unliftableOps(unliftableOps), analysis(index),
        needToCloneOperands(index->emptySet()) {
    setup();
  }

  void print(llvm::raw_ostream &os) {
    analysis.printPredicate(os, "needToCloneOperands", needToCloneOperands);
    llvm::dbgs() << "cloned ops: ";
    operationIndex->printVs(llvm::dbgs(),
                            operationIndex->setToVector(clonedOps));
    llvm::dbgs() << "\n";
  }

private:
  using Analysis = BackwardsRelationDataflowAnalysis;
  using DomainMap = Analysis::DomainMap;

  void flowGeneric(Operation *op, DomainMap &domainMap) {
    if (operationIndex->setContains(unliftableOps, op)) {
      // Ops which produce these operands may need to be cloned.
      for (OpOperand &opOperand : op->getOpOperands())
        analysis.setTrue(domainMap, &opOperand);
    } else if (canBeConveyed(op)) {
      // Results are recovered in 'execute', so operands constrain nothing
      // upstream.
      analysis.flowDefault(op, domainMap);
    } else {
      Analysis::Set set = domainMap.bottom;
      for (OpOperand &operand : op->getUses()) {
        domainMap.leastUpperBound(
            set, analysis.getEntriesAndId(domainMap, &operand));
      }
      bool neededByClonedOrUnliftable =
          set.test(0); // distinguished olways has index 0
      if (neededByClonedOrUnliftable) {
        LLVM_DEBUG({
          llvm::dbgs() << "NeedToCloneAnalysis: considering ";
          operationIndex->printV(llvm::dbgs(), op);
          llvm::dbgs() << " to require cloning. neededByClonedOrUnliftable="
                       << neededByClonedOrUnliftable << "\n";
        });
        operationIndex->addToSet(clonedOps, op);
      }
      for (OpOperand &opOperand : op->getOpOperands()) {
        if (neededByClonedOrUnliftable)
          analysis.setTrue(domainMap, &opOperand);
        else
          analysis.setDefault(domainMap, &opOperand);
      }
    }
  }

  void flow(Operation *op, DomainMap &domainMap) {
    bool inInitBlock =
        op->getParentRegion()->isAncestor(&analysis.index->model.getInit());
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      // Usual control flow rules.
      analysis.flowControlFlow(
          cfOp, domainMap, std::bind(&NeedToCloneAnalysis::flow, this, _1, _2));
      if (!inInitBlock) {
        // Control flow ops are unliftable, so all their operands are too.
        for (OpOperand &opOperand : op->getOpOperands())
          analysis.setTrue(domainMap, &opOperand);
      }
    } else if (isa<MEF::OutputOp>(op)) {
      // Ignore 'plumbing'.
    } else if (inInitBlock) {
      // Ignore 'init' block ops.
      analysis.flowDefault(op, domainMap);
    } else {
      flowGeneric(op, domainMap);
    }
  }

  void setup() {
    Analysis::Relation neededByExecuteRelation;
    DomainMap domainMap(neededByExecuteRelation, Analysis::kLubIsUnion,
                        analysis.index->emptySet(),
                        analysis.index->unitSet(analysis.index->distinguished));

    // All 'execute' block outputs are vacuously needed an unliftable op.
    for (OpOperand &result : analysis.index->model.getExecuteBlock()
                                 ->getTerminator()
                                 ->getOpOperands())
      analysis.setTrue(domainMap, &result);

    analysis.flowModel(analysis.index->model, domainMap,
                       std::bind(&NeedToCloneAnalysis::flow, this, _1, _2));
    analysis.addToPredicate(needToCloneOperands, neededByExecuteRelation);
  }

  IndexSet<Operation *> *operationIndex;
  const OperationSet &unliftableOps;
  Analysis analysis;
  // Computed only for use by print for debugging.
  Analysis::Set needToCloneOperands;
};

//===----------------------------------------------------------------------===//
// ExecInvariantCodeMotionPass
//===----------------------------------------------------------------------===//

class ExecInvariantCodeMotionPass
    : public impl::ExecInvariantCodeMotionPassBase<
          ExecInvariantCodeMotionPass> {
  using ExecInvariantCodeMotionPassBase::ExecInvariantCodeMotionPassBase;

  // Returns the unliftable 'execute' ops and the liftable ones that must be
  // cloned before lifting.
  std::pair<OperationSet, OperationSet> analyze(ModelOp modelOp) {
    OperationSet unliftableOps = operationIndex->emptySet();
    size_t origNumUnliftableOps;

    do {
      origNumUnliftableOps = unliftableOps.count();
      LLVM_DEBUG(llvm::dbgs() << "-------- analysis iteration --------\n");

      // Don't lift if depends on 'execute' block args.
      DependsOnArgsAnalysis dependsOnArgsAnalysis(valueIndex, operationIndex,
                                                  std::move(unliftableOps));
      LLVM_DEBUG(dependsOnArgsAnalysis.print(llvm::dbgs()));

      // Don't lift if trivial to recompute and only used by unliftable.
      SuppressTrivialAnalysis suppressTrivialAnalysis(
          operandIndex, operationIndex,
          std::move(dependsOnArgsAnalysis.unliftableOps));
      LLVM_DEBUG(suppressTrivialAnalysis.print(llvm::dbgs()));

      // Don't lift if mutated by an unliftable op.
      SuppressMutatedAnalysis suppressMutatedAnalysis(
          operandIndex, operationIndex,
          std::move(suppressTrivialAnalysis.unliftableOps));
      LLVM_DEBUG(suppressMutatedAnalysis.print(llvm::dbgs()));

      unliftableOps = std::move(suppressMutatedAnalysis.unliftableOps);
    } while (origNumUnliftableOps < unliftableOps.count());

    NeedToCloneAnalysis needToCloneAnalysis(operandIndex, operationIndex,
                                            unliftableOps);

    LLVM_DEBUG({
      llvm::dbgs() << "-------- analysis fixed point --------------\n";
      needToCloneAnalysis.print(llvm::dbgs());
      llvm::dbgs() << "--------------------------------------------\n";
    });

    return {std::move(unliftableOps), std::move(needToCloneAnalysis.clonedOps)};
  }

  void rewrite(const std::pair<OperationSet, OperationSet> &sets) {
    DenseSet<Operation *> unliftableOps;
    for (Operation *op : operationIndex->values(sets.first))
      unliftableOps.insert(op);
    DenseSet<Operation *> cloneOps;
    for (Operation *op : operationIndex->values(sets.second))
      cloneOps.insert(op);

    auto isOpOperandUnliftable = [&unliftableOps](const OpOperand &operand) {
      return unliftableOps.contains(operand.getOwner());
    };

    auto initOutputOp = cast<MEF::OutputOp>(*initState->init->rbegin());
    Value initOriginalChain = initState->getFinalChain();
    Location outputLoc = initOutputOp.getLoc();
    initState->builder.setInsertionPoint(initOutputOp);

    // A host-chain placeholder gating the caching of lifted buffers until all
    // side effects on them complete; replaced once lifting is done.
    auto placeholderBufferCacheInputChain =
        GRT::ChainCreateOp::create(initState->builder, outputLoc);

    // TODO(#24751): The following chain dependencies may be too aggressive,
    // relax via some analysis

    // Chains of side effects on lifted buffers; caching waits for ALL of them.
    SmallVector<Value> potentialBufferActionChains;

    // Cache-insertion chains, joined into the final 'init' output chain.
    SmallVector<Value> finalOutChains{initOriginalChain};

    // TODO(#24751): We only consider ops in the top-level 'execute' block.
    SmallVector<Operation *> allOps = getDataflowOrder(executeState->execute);

    LLVM_DEBUG(llvm::dbgs() << "first pass...\n");

    // Clone in a separate first pass so lifted results are renamed in the
    // right places.
    IRMapping cloneIRMapping;
    for (Operation *op : allOps) {
      if (!cloneOps.contains(op))
        continue;
      LLVM_DEBUG({
        llvm::dbgs() << "cloning ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << "\n";
      });
      ASSERT_STREAM(!unliftableOps.contains(op),
                    << "attempting to clone an op (" << *op
                    << ") which is unliftable");
      ASSERT_STREAM(isRecomputableOp(op), << "attempting to clone an op ("
                                          << *op
                                          << ") which is not recomputable");

      // For easier debugging only.
      executeState->builder.setInsertionPoint(op);

      Operation *clonedOp = executeState->builder.clone(*op, cloneIRMapping);
      // Clones stay in 'execute'.
      unliftableOps.insert(clonedOp);
      op->replaceUsesWithIf(clonedOp, isOpOperandUnliftable);
    }

    LLVM_DEBUG(llvm::dbgs() << "second pass...\n");

    for (Operation *op : allOps) {
      if (unliftableOps.contains(op))
        continue;
      LLVM_DEBUG({
        llvm::dbgs() << "moving ";
        operationIndex->printV(llvm::dbgs(), op);
        llvm::dbgs() << "\n";
      });

      // For easier debugging only.
      executeState->builder.setInsertionPoint(op);

      if (Value oldChain = getLiftableChainOp(op)) {
        // Uses staying in 'execute' switch to its root chain (CSE dedups);
        // other uses are lifted along with op.
        Value newChain = executeState->getRootChain();
        oldChain.replaceUsesWithIf(newChain, isOpOperandUnliftable);
        LLVM_DEBUG(llvm::dbgs() << "will lift chain op and replace old chain:\n"
                                << oldChain << "\nwith:\n"
                                << newChain << "\n");
        cacheLiftedIndexResults(*initState, *executeState, op,
                                placeholderBufferCacheInputChain,
                                finalOutChains, isOpOperandUnliftable);

      } else if (Value oldBuffer = getLiftableBufferOp(op)) {
        // Uses staying in 'execute' read the buffer back from the cache.
        if (llvm::any_of(op->getUses(), isOpOperandUnliftable)) {
          // Caching must wait for every writer. Rather than track writers per
          // buffer, *all* buffers wait for *all* writers.
          for (OpOperand &use : op->getUses()) {
            if (isMutated(use)) {
              // Fires when the parent is liftable but a "trivial" child is
              // not. Cloning the parent or moving the child would fix it, but
              // very few ops are trivial.
              ASSERT_STREAM(!isOpOperandUnliftable(use),
                            << "attempting to lift buffer (" << *op
                            << ") which will be further mutated "
                               "in the 'execute' block");
              for (Value res : use.getOwner()->getResults()) {
                if (isa<MEF::MEFChainTypeInterface>(res.getType()))
                  potentialBufferActionChains.push_back(res);
              }
            }
          }

          MGPBufferBuilder *bufferBuilder =
              TensorBufferDialectInterface::getBuilderForTypeStorage(
                  *builderContext, oldBuffer.getType());
          auto [bufferCacheOutChain, currBufferSlot] = bufferBuilder->cache(
              initState->builder, op->getLoc(), initState->getContext(),
              placeholderBufferCacheInputChain, initState->contextCreateOp,
              oldBuffer);

          finalOutChains.push_back(bufferCacheOutChain);

          auto newBuffer = bufferBuilder->getCached(
              executeState->builder, op->getLoc(), executeState->getContext(),
              executeState->getRootChain(), currBufferSlot,
              oldBuffer.getType());

          oldBuffer.replaceUsesWithIf(newBuffer, isOpOperandUnliftable);
          LLVM_DEBUG(llvm::dbgs()
                     << "will lift buffer op and replace old buffer:\n"
                     << oldBuffer << "\nwith:\n"
                     << newBuffer << "\nvia cached:\n"
                     << bufferCacheOutChain << "\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "will lift buffer op without caching\n");
        }

      } else if (!getLiftableIndexOutputs(op).empty()) {
        cacheLiftedIndexResults(*initState, *executeState, op,
                                placeholderBufferCacheInputChain,
                                finalOutChains, isOpOperandUnliftable);

      } else {
        ASSERT_STREAM(isRecomputableOp(op),
                      << "attempting to lift an op (" << *op
                      << ") which is neither a liftable chain or "
                         "buffer op, nor a recomputable op");

        // Note that the op may have been cloned by the first pass.
      }

      LLVM_DEBUG(llvm::dbgs() << "considering op to be recomputable:\n"
                              << *op << "\n");
      op->moveBefore(initOutputOp);
    }

    auto isInInitBlock = [this](OpOperand &operand) -> bool {
      return operand.getOwner()->getBlock() == initState->init;
    };

    // Lifted ops now use the 'init' equivalents of the 'execute' block args;
    // the chain is the 'init' output chain from before the updates above.
    executeState->getRootChain().replaceUsesWithIf(initOriginalChain,
                                                   isInInitBlock);

    executeState->getContext().replaceUsesWithIf(initState->contextCreateOp,
                                                 isInInitBlock);

    for (auto [execDeviceContext, initDeviceContext] :
         llvm::zip(executeState->getDevices(), initState->getDevices()))
      execDeviceContext.replaceUsesWithIf(initDeviceContext, isInInitBlock);

    executeState->getWeightsRegistry().replaceUsesWithIf(
        initState->getWeightsRegistry(), isInInitBlock);

    placeholderBufferCacheInputChain.replaceAllUsesWith(
        GRT::ChainCreateOp::create(initState->builder, outputLoc,
                                   potentialBufferActionChains)
            .getResult());
    placeholderBufferCacheInputChain.erase();

    auto mergedFinalOutChain = GRT::ChainCreateOp::create(
        initState->builder, outputLoc, finalOutChains);
    initState->updateFinalChain(mergedFinalOutChain);
  }

  void runOnOperation() override {
    assert(!initState && !executeState && !valueIndex && !operandIndex &&
           !operationIndex && !builderContext);
    ModelOp modelOp = getOperation();
    InitState theInitState(modelOp);
    initState = &theInitState;
    ExecuteState theExecuteState(modelOp.getContext(), modelOp);
    executeState = &theExecuteState;
    IndexSet<Value> theValueIndex = createValueIndexSet(modelOp);
    valueIndex = &theValueIndex;
    IndexSet<const OpOperand *> theOperandIndex =
        createOperandIndexSet(modelOp);
    operandIndex = &theOperandIndex;
    IndexSet<Operation *> theOperationIndex = createOperationIndexSet(modelOp);
    operationIndex = &theOperationIndex;

    BufferBuilderContext theBuilderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);
    builderContext = &theBuilderContext;

    LLVM_DEBUG({
      llvm::dbgs() << "-------- before ExecInvariantCodeMotionPass --------\n";
      valueIndex->printModel(llvm::dbgs());
      llvm::dbgs() << "----------------------------------------------------\n";
    });

    rewrite(analyze(modelOp));

    // Clear state because the instance might be reused
    builderContext = nullptr;
    operationIndex = nullptr;
    operandIndex = nullptr;
    valueIndex = nullptr;
    executeState = nullptr;
    initState = nullptr;
  }

  /// Rewrite helpers.
  InitState *initState = nullptr;
  ExecuteState *executeState = nullptr;
  IndexSet<Value> *valueIndex = nullptr;
  IndexSet<const OpOperand *> *operandIndex = nullptr;
  IndexSet<Operation *> *operationIndex = nullptr;

  /// All the type and buffer builders we'll need while working on the model.
  BufferBuilderContext *builderContext = nullptr;
};

} // namespace
} // namespace M::MGP
