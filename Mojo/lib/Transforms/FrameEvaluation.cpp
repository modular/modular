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

#include "FrameEvaluation.h"
#include "FrameAnalysis.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "SuspendCrossings.h"
#include "llvm/ADT/SetVector.h"
#include <memory>

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace CO;
using llvm::BitVector;

namespace {

/// Liveness analysis in two passes over the reachable blocks: a postorder
/// sweep that applies each block's transfer function, then back-edge
/// propagation through the loops the CFG closes.
///
/// Blocks unreachable from the entry keep an empty live-in set.
BitVector liveAnalysisTwoPass(const Analysis &analysis, bool isHot) {
  unsigned numValues = analysis.allValues.size();
  DenseMap<VirtualBlock, BitVector> liveIn;
  for (auto &[head, info] : analysis.cfg)
    liveIn.try_emplace(head, numValues);

  for (auto [index, block] : llvm::enumerate(analysis.postorder)) {
    BitVector successorJoin(numValues);
    // A successor later in the postorder is reached by a back edge.
    for (VirtualBlock successor : analysis.cfg.at(block).successors)
      if (analysis.postorderIndex.at(successor) < index)
        successorJoin |= liveIn.at(successor);
    liveIn.at(block) = transferFunctionOperands(analysis, block, numValues)(
        std::move(successorJoin));
  }

  propagateBackEdges(analysis, liveIn);

  BitVector frameValues(numValues);
  for (VirtualBlock resumeBlock : analysis.resumeBlocks)
    frameValues |= liveIn.at(resumeBlock);
  if (!isHot)
    frameValues |= liveIn.at(analysis.entryBlock);
  // An unmarked allocation has no lifetime markers to bound it, so keep it in
  // the frame unconditionally.
  // TODO(stumpOS): frame only the allocations a framed value derives from,
  // instead of every unmarked allocation, once that precise walk gets its own
  // review.
  for (StackAllocationOp allocation : analysis.unmarkedAllocations)
    frameValues.set(analysis.valueToIndex.at(allocation->getResult(0)));
  return frameValues;
}

/// Answers the frame queries from liveness and the suspend crossings.
class LiveFrame final : public FrameInterface {
public:
  LiveFrame(Analysis analysis, BitVector frameValues,
            SuspendCrossings crossings, bool isHot)
      : analysis(std::move(analysis)), frameValues(std::move(frameValues)),
        crossings(std::move(crossings)), isHot(isHot) {}

  bool isTracked(Operation *op) override {
    return analysis.opToVirtualBlock.contains(op);
  }

  bool crossesSuspend(OpOperand &use) override {
    if (!isHot && isFunctionArgument(use.get()))
      return true;
    return crossings.crosses(definingVirtualBlock(use.get()),
                             lookupVirtualBlock(use.getOwner()));
  }

  bool suspendBetween(Operation *from, Operation *to) override {
    VirtualBlock fromBlock = lookupVirtualBlock(from);
    VirtualBlock toBlock = lookupVirtualBlock(to);
    if (!crossings.isReachableFrom(fromBlock, toBlock))
      return true;
    return crossings.crosses(fromBlock, toBlock);
  }

  bool isAfterSuspend(Operation *op) override {
    return crossings.isAfterSuspend(lookupVirtualBlock(op));
  }

  ArrayRef<VirtualBlock> firstStateVirtualBlocks() override {
    return firstStateBlocks;
  }

  const DenseSet<CO::SuspendOp> &firstSuspends() override {
    return firstSuspendOps;
  }

  void registerClone(Operation *clone, Operation *original,
                     Operation *user) override {
    analysis.opToVirtualBlock[clone] = lookupVirtualBlock(user);
    clones.insert(clone);
    clonedOriginals.push_back(original);
  }

  /// Updates the frame values for the clones.
  void finalize() {
    for (Operation *clone : clones) {
      Value result = clone->getResult(0);
      analysis.valueToIndex[result] = analysis.allValues.size();
      analysis.allValues.push_back(result);
    }
    frameValues.resize(analysis.allValues.size());
    for (Operation *original : clonedOriginals)
      frameValues.reset(analysis.valueToIndex.at(original->getResult(0)));
    for (Operation *clone : clones)
      for (OpOperand &use : clone->getOpOperands())
        if (crossesSuspend(use))
          frameValues.set(analysis.valueToIndex.at(use.get()));
  }
  /// Returns whether `value` is a tracked frame variable.
  bool isInFrame(Value value) const {
    auto index = analysis.valueToIndex.find(value);
    return index != analysis.valueToIndex.end() &&
           frameValues.test(index->second);
  }

  /// Whether the lowering erases `op` while rewriting the frame accesses.
  static bool erasedByLowering(Operation *op, const FrameData &frameData) {
    if (auto allocation = dyn_cast<StackAllocationOp>(op))
      return frameData.valueToIndexInFrame.contains(allocation.getResult());
    if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(op)) {
      if (op->getNumOperands() == 0)
        return true;
      return llvm::all_of(op->getOperands(), [&](Value operand) {
        return frameData.valueToIndexInFrame.contains(operand);
      });
    }
    return false;
  }

  /// Records the virtual blocks the entry reaches without passing a suspend on
  /// any path.
  void collectFirstState(const FrameData &frameData) {
    for (auto &[head, info] : analysis.cfg) {
      if (isa<SuspendOp>(info.lastOp) &&
          crossings.isReachableWithoutSuspend(head))
        firstSuspendOps.insert(cast<SuspendOp>(info.lastOp));
      if (!crossings.isReachable(head) || crossings.isAfterSuspend(head))
        continue;
      firstStateBlocks.push_back(erasedByLowering(head, frameData) ? info.lastOp
                                                                   : head);
    }
    llvm::stable_sort(firstStateBlocks, [&](Operation *a, Operation *b) {
      return analysis.postorderIndex.at(lookupVirtualBlock(a)) >
             analysis.postorderIndex.at(lookupVirtualBlock(b));
    });
  }

private:
  VirtualBlock lookupVirtualBlock(Operation *op) const {
    auto it = analysis.opToVirtualBlock.find(op);
    assert(it != analysis.opToVirtualBlock.end() &&
           "op was removed from the function after frame analysis");
    return it->second;
  }

  VirtualBlock definingVirtualBlock(Value value) const {
    if (Operation *definingOp = value.getDefiningOp())
      return lookupVirtualBlock(definingOp);
    return analysis.blockToVirtualBlock.at(
        cast<BlockArgument>(value).getOwner());
  }

  static bool isFunctionArgument(Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    return argument && isa<FuncOp>(argument.getOwner()->getParentOp());
  }

  Analysis analysis;
  BitVector frameValues;
  SuspendCrossings crossings;
  bool isHot;
  llvm::SetVector<Operation *> clones;
  SmallVector<Operation *> clonedOriginals;
  SmallVector<Operation *> firstStateBlocks;
  DenseSet<CO::SuspendOp> firstSuspendOps;
};
} // namespace

void M::KGEN::evaluateFrameByLiveness(
    FrameData &frameData, FuncOp originalFunction, Value errorValue,
    Value resultValue, FrameStateTransform rematerializeFrameValues,
    bool isHot) {
  auto &frameTypes = frameData.frameTypes;
  auto &valueToIndexInFrame = frameData.valueToIndexInFrame;
  auto &argsInFrame = frameData.argsInFrame;

  Analysis analysis = analyze(originalFunction);
  SuspendCrossings crossings(analysis);
  BitVector frameValues = liveAnalysisTwoPass(analysis, isHot);
  auto frame = std::make_unique<LiveFrame>(
      std::move(analysis), std::move(frameValues), std::move(crossings), isHot);
  rematerializeFrameValues(originalFunction, *frame);
  frame->finalize();
  auto isInFrame = [&](Value value) { return frame->isInFrame(value); };

  // Every value live across a suspend must be added to
  // the frame.
  auto addToFrame = [&](Value value, Type frameVariableType) {
    unsigned index = frameTypes.size();
    frameTypes.push_back(frameVariableType);
    valueToIndexInFrame.insert({value, index});
  };
  auto stackAllocationFrameType =
      [](StackAllocationOp stackAllocation) -> Type {
    int64_t count = cast<IntegerAttr>(stackAllocation.getCount()).getInt();
    if (count == 1) {
      return stackAllocation.getType().getElementType();
    } else {
      return POP::ArrayType::get(stackAllocation.getCount(),
                                 stackAllocation.getType().getElementType());
    }
  };
  // TODO(stumpOS): Build the frame by iterating the set bits of `frameValues`
  // instead of walking every use. That assigns slots in definition order
  // rather than first-use order, so the slot indices in the
  // lower-async-functions tests should be updated (or dropped, except for
  // arguments) in the same change.
  unsigned numFrameArguments = 0;

  originalFunction.walk([&](Operation *operation) {
    // Stack allocations without marked lifetimes are checked for frame
    // membership at their definition.
    if (auto stackAllocation = dyn_cast<StackAllocationOp>(operation)) {
      Value allocation = stackAllocation->getResult(0);
      if (!stackAllocation.getMarkedLifetimes() && isInFrame(allocation))
        addToFrame(allocation, stackAllocationFrameType(stackAllocation));
    }

    for (Value operand : operation->getOperands()) {
      // Results and Errors are stored in the header of the continuation, not in
      // the frame.
      if (operand == errorValue || operand == resultValue)
        continue;
      if (valueToIndexInFrame.contains(operand))
        continue;

      auto blockArgument = dyn_cast<BlockArgument>(operand);
      bool isFunctionArg =
          blockArgument &&
          blockArgument.getParentBlock()->getParentOp() == originalFunction;
      if (!isInFrame(operand) && !(isFunctionArg && !isHot))
        continue;

      if (isFunctionArg) {
        // Arguments are placed after every other value, below.
        valueToIndexInFrame.insert({operand, numFrameArguments++});
      } else if (auto stackAllocation =
                     operand.getDefiningOp<StackAllocationOp>()) {
        addToFrame(operand, stackAllocationFrameType(stackAllocation));
      } else {
        addToFrame(operand, operand.getType());
      }
    }
  });

  // Insert used arguments at end of frame, in first-use order. The walk
  // recorded each one's first-use position in place of its frame slot.
  unsigned offset = frameTypes.size();
  frameTypes.resize(offset + numFrameArguments);
  for (auto [index, functionArg] :
       llvm::enumerate(originalFunction.getRegion().front().getArguments())) {
    auto frameSlotMaybe = valueToIndexInFrame.find(functionArg);
    if (frameSlotMaybe == valueToIndexInFrame.end())
      continue;
    unsigned &frameSlot = frameSlotMaybe->second;
    frameSlot += offset;
    frameTypes[frameSlot] = functionArg.getType();
    argsInFrame.push_back(FrameData::ArgInFrame(index, frameSlot));
  }

  frame->collectFirstState(frameData);
  frameData.frame = std::move(frame);
}
