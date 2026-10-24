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

#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFChainIndicatesFinishedInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFNonStrictInterface.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include <deque>

using namespace M;
using namespace M::MGP;

void M::MGP::accumFreeValues(mlir::SetVector<Value> &freeValues, Block *block,
                             Block *bindingBlock) {
  for (Operation &inOp : block->getOperations()) {
    for (Value inValue : inOp.getOperands()) {
      if (inValue.getParentRegion()->isProperAncestor(
              bindingBlock->getParent()))
        freeValues.insert(inValue);
      else
        ASSERT_STREAM(
            inValue.getParentRegion()->isAncestor(block->getParent()),
            << "result from operation"
            << Twine(inValue.getDefiningOp()->getName().getStringRef())
            << " used in operation " << Twine(inOp.getName().getStringRef())
            << " must be in the ancestor scope");
    }
    for (Region &subRegion : inOp.getRegions())
      accumFreeValues(freeValues, &subRegion.front(), bindingBlock);
  }
}

#ifdef MODULAR_DEBUG
/// Finds a cycle starting at op. Recursive, so only for diagnostics before a
/// hard failure.
static bool findCycle(SmallVectorImpl<Operation *> &stack, Operation *op) {
  if (llvm::is_contained(stack, op)) {
    llvm::dbgs() << "cycle is:\n";
    for (Operation *pathOp : stack) {
      llvm::dbgs() << "  ";
      pathOp->print(llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm());
      llvm::dbgs() << "\n";
    }
    return true;
  }
  stack.push_back(op);
  for (Value operand : op->getOperands()) {
    if (Operation *nextOp = operand.getDefiningOp()) {
      if (findCycle(stack, nextOp))
        return true;
    }
  }
  stack.pop_back();
  return false;
}
#endif

/// Returns true if currOp needs to be visited. Asserts on a cycle.
static bool shouldStartVisiting(DenseSet<Operation *> &visited,
                                Operation *currOp, Block *block) {
  if (currOp->getBlock() != block) {
    // Outer values were accounted for when visiting their own block.
    return false;
  }

  bool alreadyVisited = visited.contains(currOp);

#ifdef MODULAR_DEBUG
  if (alreadyVisited) {
    llvm::dbgs() << "model contains a dataflow cycle involving op:\n  ";
    currOp->print(llvm::dbgs(), mlir::OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";

    SmallVector<Operation *> stack;
    findCycle(stack, currOp);
    ModelOp modelOp = currOp->getParentOfType<ModelOp>();
    if (modelOp) {
      llvm::dbgs() << "within:\n";
      llvm::dbgs() << "----------------------------------------\n";
      modelOp->print(llvm::dbgs(),
                     mlir::OpPrintingFlags().printGenericOpForm());
      llvm::dbgs() << "\n----------------------------------------\n";
    }
  }
#endif

  return !alreadyVisited;
}

namespace {
/// Records a pending operand to visit.
struct Frame {
  Operation *frameOp;
  size_t nextOperandIndex;
  SmallVector<Value, 4> extraValues; // Free values of control flow ops.
  size_t nextExtraValueIndex;

  /// Returns the next operand of frameOp to visit, or null if done.
  Value next() {
    if (nextOperandIndex < frameOp->getNumOperands())
      return frameOp->getOperand(nextOperandIndex++);
    if (nextExtraValueIndex < extraValues.size())
      return extraValues[nextExtraValueIndex++];
    return {};
  }

  Frame(Operation *frameOp, size_t nextOperandIndex)
      : frameOp(frameOp), nextOperandIndex(nextOperandIndex),
        nextExtraValueIndex(0) {}
};
} // namespace

/// Pushes onto stack the frame necessary to visit currOp, if any.
static void startVisiting(DenseSet<Operation *> &visited,
                          SmallVector<Frame> &stack, Operation *currOp,
                          Block *block) {
  if (!shouldStartVisiting(visited, currOp, block))
    return;

  stack.emplace_back(currOp, 0);

  // Free values of nested regions are dependencies too.
  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(currOp)) {
    mlir::SetVector<Value> freeValues;
    for (Region &region : cfOp->getRegions())
      accumFreeValues(freeValues, &region.front(), &region.front());

    Frame &frame = stack.back();
    for (Value freeValue : freeValues) {
      if (freeValue.getParentBlock() == block)
        frame.extraValues.push_back(freeValue);
    }
  }
}

/// Records that currOp has been fully visited.
static void finishVisiting(DenseSet<Operation *> &visited, Operation *currOp) {
  [[maybe_unused]] auto [_, inserted] = visited.insert(currOp);
  ASSERT_STREAM(inserted, "operation already finished");
}

/// Iteratively visits inputs of op in DFS order, appending post-order visits
/// to dataflowOrder. Debug builds report cycles; release builds ignore
/// back-arcs.
static void visitInDataflowOrder(Operation *op,
                                 SmallVectorImpl<Operation *> &dataflowOrder,
                                 DenseSet<Operation *> &visited) {
  Block *block = op->getBlock();
  SmallVector<Frame> stack;

  startVisiting(visited, stack, op, block);

  while (!stack.empty()) {
    Frame &frame = stack.back();
    Value nextOperand = frame.next();
    if (!nextOperand) {
      finishVisiting(visited, frame.frameOp);
      dataflowOrder.push_back(frame.frameOp);
      stack.pop_back();
    } else if (Operation *inputOp = nextOperand.getDefiningOp()) {
      startVisiting(visited, stack, inputOp, block);
    }
  }
}

SmallVector<Operation *> M::MGP::getDataflowOrder(Block *block) {
  SmallVector<Operation *> dataflowOrder;
  // Ops whose visit has completed.
  DenseSet<Operation *> visited;
  for (Operation &op : block->getOperations())
    visitInDataflowOrder(&op, dataflowOrder, visited);
  return dataflowOrder;
}

bool M::MGP::isExactlyOnce(Operation *op) {
  op = op->getParentOp();
  while (op) {
    if (auto modelOp = dyn_cast<ModelOp>(op))
      return true;
    if (isa<GRT::GRTControlFlowInterface>(op))
      return false;
    op = op->getParentOp();
  }
  return true;
}

bool M::MGP::isAtMostOnce(Operation *op) {
  op = op->getParentOp();
  while (op) {
    if (auto modelOp = dyn_cast<ModelOp>(op))
      return true;
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
      if (cfOp.getFlavor() == GRT::kLoopLike)
        return false;
    }
    op = op->getParentOp();
  }
  return true;
}

SmallVector<Value> M::MGP::allLoopBoundArgs(Operation *op) {
  SmallVector<Value> result;
  while (op->getBlock()) {
    if (!isAtMostOnce(op)) {
      for (Value arg : op->getBlock()->getArguments())
        result.emplace_back(arg);
    }
    op = op->getParentOp();
  }
  return result;
}

SmallVector<Value> M::MGP::allLoopResults(Operation *op) {
  SmallVector<Value> result;
  while (op->getBlock()) {
    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op->getParentOp())) {
      if (cfOp.getFlavor() == GRT::kLoopLike) {
        for (Value res : cfOp->getResults())
          result.emplace_back(res);
      }
    }
    op = op->getParentOp();
  }
  return result;
}

SmallVector<Value> M::MGP::getIndicatingResults(Operation *op) {
  // A control flow op is finished when any of its results is ready.
  if (!isa<MEF::ChainIndicatesFinishedInterface>(op))
    // All results are indicating.
    return op->getResults();

  // Only chain results, if any, are indicating.
  SmallVector<Value> results;
  for (Value result : op->getResults()) {
    if (isa<MEF::MEFChainTypeInterface>(result.getType()))
      results.emplace_back(result);
  }
  return results;
}

SmallVector<Value> M::MGP::getGatingOperands(Operation *op) {
  SmallVector<Value> results;
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (isGatingOperand(&opOperand))
      results.emplace_back(opOperand.get());
  }
  return results;
}

bool M::MGP::isGatingOperand(const OpOperand *opOperand) {
  Operation *op = opOperand->getOwner();

  if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op)) {
    // Only the extra strict operands are gating for control flow ops.
    auto extraOperands = cfOp.getStrictExtraOperands();
    return llvm::is_contained(extraOperands, opOperand->get());
  }

  if (isa<MEF::OutputOp>(op))
    // Nothing gates a mef.output.
    return false;

  if (!isa<MEF::NonStrictInterface>(op))
    // All operands are gating for strict ops.
    return true;

  // Nothing gates a non-strict op; analyses tracking implied control-flow
  // dependencies must handle these specially.
  return false;
}

OpOperand *M::MGP::getUniqueMutator(Value value) {
  OpOperand *candidate = nullptr;
  for (OpOperand &use : value.getUses()) {
    if (isMutated(use) || isa<MGP::BufferSliceOp>(use.getOwner())) {
      if (candidate)
        return nullptr;
      candidate = &use;
    }
  }
  return candidate;
}

SmallVector<OpOperand *, 1> M::MGP::getInputChainOpOperands(Operation *op) {
  SmallVector<OpOperand *, 1> result;
  for (auto &operand : op->getOpOperands()) {
    if (isa<MEF::MEFChainTypeInterface>(operand.get().getType()))
      result.emplace_back(&operand);
  }
  return result;
}

void M::MGP::moveAllBefore(Operation *origTargetOp, Operation *origAnchorOp) {
  std::deque<std::pair<Operation *, Operation *>> worklist;
  worklist.emplace_back(origTargetOp, origAnchorOp);
  while (!worklist.empty()) {
    auto [targetOp, anchorOp] = worklist.front();
    worklist.pop_front();
    if (targetOp == nullptr || anchorOp == nullptr)
      continue;
    if (targetOp->getBlock() != anchorOp->getBlock())
      continue;
    if (targetOp->isBeforeInBlock(anchorOp))
      continue;
    targetOp->moveBefore(anchorOp);
    for (auto operand : targetOp->getOperands())
      worklist.emplace_front(operand.getDefiningOp(), targetOp);
  }
}

/// Adds chain to primitiveChains, looking through grt.chain.create ops.
void M::MGP::accumulatePrimitiveChains(ValueSetVector &primitiveChains,
                                       Value chain) {
  if (auto chainCreateOp =
          chain.getDefiningOp<GRT::GRTChainCreateOpInterface>()) {
    for (auto operand : chainCreateOp.getInputChains())
      accumulatePrimitiveChains(primitiveChains, operand);
  } else {
    primitiveChains.insert(chain);
  }
}

/// As waitForChains, but reuses existingUniqueChainCreateOp if non-null.
static Value waitForChainsReusing(
    OpBuilder &builder, mlir::Location loc, TypedValue<ContextType> context,
    MGPBufferBuilder *optBufferBuilder,
    GRT::GRTGatedOnChainOpInterface existingUniqueChainCreateOp,
    ArrayRef<Value> chains) {
  assert(!chains.empty());

  if (chains.size() == 1)
    return chains.front();

  if (existingUniqueChainCreateOp) {
    existingUniqueChainCreateOp.setInputChains(chains);
    return dyn_cast<GRT::GRTChainCreateOpInterface>(
               existingUniqueChainCreateOp.getOperation())
        .getChainResult();
  }
  return GRT::ChainCreateOp::create(builder, loc, chains);
}

Value M::MGP::waitForChains(OpBuilder &builder, mlir::Location loc,
                            TypedValue<ContextType> context,
                            MGPBufferBuilder *optBufferBuilder,
                            ArrayRef<Value> chains) {
  return waitForChainsReusing(builder, loc, context, optBufferBuilder,
                              /*existingUniqueChainCreateOp=*/{}, chains);
}

LogicalResult
M::MGP::alsoWaitForChains(OpBuilder &builder,
                          const BufferBuilderContext &builderContext,
                          Operation *op, ArrayRef<Value> additionalChains) {
  if (additionalChains.empty()) {
    // Nothing new to wait on.
    return success();
  }

  auto existingInChainOpOperands = getInputChainOpOperands(op);
  if (existingInChainOpOperands.empty()) {
    // Nothing on which to anchor the new chains.
    return failure();
  }

  TypedValue<ContextType> context =
      TensorBufferDialectInterface::getBuilderForOpExecution(builderContext, op)
          ->getContextForOp(op);

  for (OpOperand *inChainOpOperand : existingInChainOpOperands) {
    Value inChain = inChainOpOperand->get();

    // The builder for inChain, or null if it's a !mef.chain.
    MGPBufferBuilder *optChainBufferBuilder = nullptr;

    // Cheap syntactic checks avoid changes SimplifyChainsPass and CSE would
    // otherwise undo.
    ValueSetVector existingPrimitiveChains;
    accumulatePrimitiveChains(existingPrimitiveChains, inChain);
    ValueSetVector newPrimitiveChains;
    for (auto additionalChain : additionalChains)
      accumulatePrimitiveChains(newPrimitiveChains, additionalChain);

    if (llvm::set_is_subset(newPrimitiveChains, existingPrimitiveChains)) {
      // The op is already gated by the new chains.
      return success();
    }

    if (llvm::set_is_subset(existingPrimitiveChains, newPrimitiveChains) &&
        newPrimitiveChains.size() == 1) {
      // The new chain subsumes all the existing chains.
      Value newInChain = newPrimitiveChains.front();
      op->setOperand(inChainOpOperand->getOperandNumber(), newInChain);
      return success();
    }

    // Merge explicitly; SimplifyChainsPass and CSE clean up duplicate or
    // redundant merges. llvm::set_union does not work: SetVector::insert
    // returns a bool, not an (itr, bool) pair.
    for (Value newChain : newPrimitiveChains)
      existingPrimitiveChains.insert(newChain);

    // A chain.create used only here can be modified in place.
    GRT::GRTGatedOnChainOpInterface existingUniqueChainCreateOp = nullptr;
    if (inChain.hasOneUse()) {
      if (auto chainCreateOp =
              inChain.getDefiningOp<GRT::GRTGatedOnChainOpInterface>())
        existingUniqueChainCreateOp = chainCreateOp;
    }

    Value newInChain = waitForChainsReusing(
        builder, op->getLoc(), context, optChainBufferBuilder,
        existingUniqueChainCreateOp, existingPrimitiveChains.takeVector());

    op->setOperand(inChainOpOperand->getOperandNumber(), newInChain);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// InitState
//===----------------------------------------------------------------------===//

InitState::InitState(ModelOp model)
    : init(model.getInitBlock()), builder(model.getContext()), model(model) {
  init->walk([this](ContextCreateOp op) {
    contextCreateOp = op;
    return WalkResult::interrupt();
  });
  ASSERT_STREAM(contextCreateOp,
                << "model is missing mgp.context.create op in init block");
  builder.setInsertionPoint(&init->back());
}

size_t InitState::allocSlot() {
  size_t slot = contextCreateOp.getNumSlots();
  contextCreateOp.setNumSlots(slot + 1);
  return slot;
}

TypedValue<MEF::ChainType> InitState::getFinalChain() {
  return cast<TypedValue<MEF::ChainType>>(init->back().getOperand(0));
}

void InitState::updateFinalChain(Value newChain) {
  init->back().setOperand(0, newChain);
}

TypedValue<ContextType> InitState::getContext() {
  return contextCreateOp.getResult();
}

//===----------------------------------------------------------------------===//
// ExecuteState
//===----------------------------------------------------------------------===//

ExecuteState::ExecuteState(MLIRContext *context, ModelOp model)
    : execute(model.getExecuteBlock()), model(model), builder(context) {
  assert(isa<MEF::ChainType>(execute->getArgument(0).getType()));
  assert(isa<ContextType>(execute->getArgument(1).getType()));
}

/// The 'execute' block's first argument. It is ready whenever any other chain
/// is, so it can be ignored.
TypedValue<MEF::ChainType> ExecuteState::getRootChain() const {
  return cast<TypedValue<MEF::ChainType>>(execute->getArgument(0));
}

TypedValue<ContextType> ExecuteState::getContext() const {
  return cast<TypedValue<ContextType>>(execute->getArgument(1));
}

iterator_range<mlir::BlockArgument *> ExecuteState::getExtraArgs() {
  return model.getExecuteInputs();
}
