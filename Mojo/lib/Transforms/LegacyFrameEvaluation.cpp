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

#include "LegacyFrameEvaluation.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"

using namespace M;
using namespace KGEN;
using M::HLCF::ReturnOp;
using M::HLCF::UnreachableOp;
using namespace POP;
using namespace CO;

/// Given a value, determine the state of its defining op or block argument.
static int
getDefinitionStateForValue(const DenseMap<Operation *, int> &opToState,
                           Value operand, bool isHot) {
  Operation *definingOp = operand.getDefiningOp();
  // Initialize state to the entry state.
  int defState = isHot ? 0 : -1;
  if (!definingOp) {
    BlockArgument blockArgument = cast<BlockArgument>(operand);
    // We always store the function arguments in the frame because they
    // originate in the ramp function.
    Operation *parentOp = blockArgument.getOwner()->getParentOp();
    if (isa<FuncOp>(parentOp))
      return defState;

    Operation *firstOp = &*blockArgument.getOwner()->begin();
    defState = opToState.at(firstOp);
    // In a loop we want to compare state before entering loop to the use
    // inside the body.
    if (isa<HLCF::LoopOp>(parentOp))
      defState = opToState.at(parentOp);
    return defState;
  }
  return opToState.at(definingOp);
}

/// Update the ops in this virtual block.
static void updateVirtualBlock(DenseMap<Operation *, int> &opToState,
                               VirtualBlock virtualBlock, int newState) {
  Operation *op = virtualBlock;
  int state = newState;
  while (op) {
    if (isa<SuspendEndOp>(op))
      ++state;
    int &oldState = opToState[op];
    if (oldState < state)
      oldState = state;
    op = op->getNextNode();
    int &nextOldState = opToState[op];
    // respect control flow boundary
    if (op && isa<HLCF::ControlFlowNode>(op)) {
      if (nextOldState < state)
        nextOldState = state;
      break;
    }
  }
}

namespace {
using PathContainer = SmallVector<Operation *>;
struct PathInfo {
  PathInfo(VirtualBlock v) : virtualBlock(v), state(0) {}
  PathInfo(VirtualBlock v, int state, PathContainer const &parent, bool hitSus)
      : virtualBlock(v), path(parent), state(state), hitSus(hitSus) {}
  int existsAt(VirtualBlock virtualBlock) const;
  VirtualBlock virtualBlock;
  PathContainer path;
  int state;
  bool hitSus = false;
};
} // namespace

int PathInfo::existsAt(VirtualBlock virtualBlock) const {
  for (auto [i, node] : llvm::enumerate(path)) {
    if (node == virtualBlock)
      return i;
  }
  return -1;
}

void M::KGEN::evaluateOldFrame(FrameData &frameData, FuncOp originalFunction,
                               mlir::DominanceInfo &domInfo, Value errorValue,
                               Value resultValue, FrameStateTransform transform,
                               bool isHot) {
  auto &frameTypes = frameData.frameTypes;
  auto &valueToIndexInFrame = frameData.valueToIndexInFrame;
  auto &operationToIndexInFrame = frameData.operationToIndexInFrame;
  auto &opToState = frameData.opToState;
  auto &virtualBlocksFirstState = frameData.virtualBlocksFirstState;
  auto &argsInFrame = frameData.argsInFrame;
  auto &firstSuspends = frameData.firstSuspends;

  // Calculate Control Flow Graph.
  // We need to know the predecessors of each region so that
  // we don't process a region until all its predecessors have
  // been processed.
  DenseMap<VirtualBlock, SmallVector<VirtualBlock>> predecessors;
  {
    SmallVector<Region *> regions;
    DenseSet<Region *> visited;
    regions.push_back(&originalFunction.getBodyRegion());

    auto pushSuccessors =
        [&](SmallVector<HLCF::ControlFlowTarget> const &targets,
            Operation *controlFlowNode, Operation *controlFlowParent,
            Operation *predecessor) {
          // For the first op of the region of each target, add the control flow
          // node as a predecessor
          for (HLCF::ControlFlowTarget target : targets) {
            VirtualBlock successor;
            if (target.index.has_value()) {
              Region *succRegion =
                  &controlFlowParent->getRegion(target.index.value());
              successor = &*succRegion->front().begin();
              regions.push_back(succRegion);
            } else {
              successor = controlFlowParent->getNextNode();
            }
            predecessors[successor].push_back(controlFlowNode);
          }
        };

    // There are three types of ops that form virtual block boundaries within a
    // region: control flow nodes, control flow terminators, and coroutine
    // awaits.
    while (!regions.empty()) {
      Region *region = regions.back();
      regions.pop_back();
      if (visited.contains(region))
        continue;
      visited.insert(region);
      Operation *lastControlFlowNode = nullptr;
      CO::SuspendOp lastAwait = nullptr;
      for (Operation &op : region->front().getOperations()) {
        if (isa<ReturnOp, UnreachableOp>(op))
          continue;

        // add the control flow terminator as a predecessor to the first op of a
        // target block.
        if (auto controlFlowTerminator =
                dyn_cast<HLCF::ControlFlowTerminator>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowTerminatorOperands(
              controlFlowTerminator->getNumOperands(), Attribute());
          controlFlowTerminator.getBranchTargets(controlFlowTerminatorOperands,
                                                 targets);
          Operation *predecessor =
              lastControlFlowNode
                  ? lastControlFlowNode->getNextNode()
                  : &*controlFlowTerminator->getParentRegion()->front().begin();
          if (lastAwait) {
            if (domInfo.dominates(predecessor, lastAwait))
              predecessor = lastAwait->getNextNode();
          }
          pushSuccessors(targets, controlFlowTerminator,
                         getParentNode(controlFlowTerminator), predecessor);
        }
        if (auto controlFlowNode = dyn_cast<HLCF::ControlFlowNode>(op)) {
          lastControlFlowNode = controlFlowNode;
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowNodeOperands(
              controlFlowNode->getNumOperands(), Attribute());
          controlFlowNode.getEntryTargets(controlFlowNodeOperands, targets);
          pushSuccessors(targets, controlFlowNode, controlFlowNode,
                         &*controlFlowNode->getParentRegion()->front().begin());
        }
        if (auto suspend = dyn_cast<CO::SuspendOp>(op)) {
          Operation *next = suspend->getNextNode();
          if (!next)
            continue;
          lastAwait = suspend;
          predecessors[&*suspend.getBody().front().begin()].push_back(suspend);
          // Terminator is used because that will trigger updated state.
          predecessors[next].push_back(
              suspend.getBody().front().getTerminator());
          regions.push_back(&suspend.getBody());
        }
      }
    }
  }
  // Calculate the state of each op.
  {
    SmallVector<PathInfo> paths;
    VirtualBlock initial = &*originalFunction.getBodyRegion().front().begin();
    paths.push_back({initial});
    auto pushSuccessors =
        [&](SmallVector<HLCF::ControlFlowTarget> const &targets,
            Operation *controlFlowVirtualBlock, Operation *controlFlowParent,
            PathContainer &path, int state, bool hitSus) {
          for (HLCF::ControlFlowTarget target : targets) {
            if (target.index.has_value()) {
              auto o = &*controlFlowParent->getRegion(target.index.value())
                             .front()
                             .begin();
              paths.push_back({o, state, path, hitSus});
            } else {
              paths.push_back(
                  {controlFlowParent->getNextNode(), state, path, hitSus});
            }
          }
        };

    int j = 0;
    while (!paths.empty()) {
      if (j > 20000)
        llvm_unreachable("infinite loop");

      ++j;
      VirtualBlock virtualBlock = paths.back().virtualBlock;
      int indexOfMe = paths.back().existsAt(virtualBlock);
      PathContainer path = std::move(paths.back().path);
      int state = paths.back().state;
      bool hitSus = paths.back().hitSus;
      paths.pop_back();

      auto recordedStatePtr = opToState.find(virtualBlock);
      if (recordedStatePtr != opToState.end() &&
          state <= recordedStatePtr->getSecond())
        continue;

      for (auto pred : predecessors[virtualBlock]) {
        if (domInfo.dominates(virtualBlock, pred))
          continue;
        auto predPtr = opToState.find(pred);
        if (predPtr != opToState.end()) {
          if (predPtr->getSecond() > state)
            state = predPtr->getSecond();
        }
      }

      // We have reached a cycle. Terminate.
      if (indexOfMe > -1)
        continue;
      path.push_back(virtualBlock);

      // Iterate through each op in this virtual block to register its state.
      // The boundaries of a node are defined by awaits, control
      // flow nodes, and control flow terminators.
      Operation *current = virtualBlock;
      while (current) {
        Operation *op = current;
        current = op->getNextNode();
        if (auto susEnd = dyn_cast<SuspendEndOp>(op)) {
          Operation *next = susEnd->getParentOp()->getNextNode();
          ++state;
          if (next)
            paths.push_back({next, state, path, hitSus});
          opToState[op] = state;
          break;
        }
        opToState[op] = state;

        if (auto suspend = dyn_cast<CO::SuspendOp>(op)) {
          if (!hitSus)
            firstSuspends.insert(suspend);
          paths.push_back(
              {&*suspend.getBody().front().begin(), state, path, true});
          break;
        }
        if (isa<ReturnOp, UnreachableOp>(op))
          continue;
        if (auto controlFlowNode = dyn_cast<HLCF::ControlFlowNode>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowNodeOperands(
              controlFlowNode->getNumOperands(), Attribute());
          controlFlowNode.getEntryTargets(controlFlowNodeOperands, targets);
          pushSuccessors(targets, controlFlowNode, controlFlowNode, path, state,
                         hitSus);
          break;
        }
        if (auto controlFlowTerminator =
                dyn_cast<HLCF::ControlFlowTerminator>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowTerminatorOperands(
              controlFlowTerminator->getNumOperands(), Attribute());
          controlFlowTerminator.getBranchTargets(controlFlowTerminatorOperands,
                                                 targets);
          pushSuccessors(targets, controlFlowTerminator,
                         getParentNode(controlFlowTerminator), path, state,
                         hitSus);
          break;
        }
      }
    }
  }

  auto propagateChildSuspoints = [&]() -> bool {
    bool wasChange = false;
    for (auto [virtualBlock, preds] : predecessors) {
      auto stateMaybe = opToState.find(virtualBlock);
      if (stateMaybe == opToState.end()) {
        opToState[virtualBlock] = -1;
        continue;
      }

      int initialState = stateMaybe->second;
      int postState = initialState;
      int stateAtContinue = 0;
      int smallestPredState = initialState;
      for (Operation *pred : preds) {
        int predState = opToState[pred];
        if (domInfo.dominates(virtualBlock, pred)) {
          if (predState > stateAtContinue && predState > postState)
            stateAtContinue = predState;
          continue;
        }
        if (predState > postState)
          postState = predState;
        if (predState < smallestPredState)
          smallestPredState = predState;
      }
      wasChange = wasChange || (initialState != postState);
      if (initialState != postState)
        updateVirtualBlock(opToState, virtualBlock, postState);

      // Insert a new state at the parent because a child with a suspension
      // point branches to it.
      if (stateAtContinue > 0 && smallestPredState == initialState) {
        int newState = initialState + 1;
        for (auto &[_, state] : opToState) {
          if (state >= newState)
            ++state;
        }
        wasChange = true;
        updateVirtualBlock(opToState, virtualBlock, newState);
      }
    }
    return wasChange;
  };

  // FIXED POINT:
  // Update until for every path:
  // (1) if A -> B and A dominates B then state(A) >= state(B)
  // (2) if B -> A and A dominates B and state(A) < state(B), then if there is
  // a predecessor C of A unreachable from B then state(C) < state(A)
  // Condition (2) is achieved by
  // inserting a parent state. Note that intuitively this corresponds to a case
  // of a child cycle with a suspension point branching to a parent cycle.
  bool stateChanged = true;
  while (stateChanged)
    stateChanged = propagateChildSuspoints();

  transform(originalFunction, opToState);

  // Calculate the frame. Whenever there is a use whose def lives in another
  // state, it must be added to the frame. We need to store the location of the
  // value in the frame so that when we generate the resume function the frame
  // values can be extracted.
  auto addToFrame = [&](Type frameVariableType, Operation *definingOp) {
    unsigned index = frameTypes.size();
    frameTypes.push_back(frameVariableType);
    operationToIndexInFrame.insert({definingOp, index});
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
  SmallVector<Value> coldCoroTypes;
  originalFunction.walk([&](Operation *operation) {
    int useState = opToState[operation];
    if (StackAllocationOp stackAllocationOp =
            dyn_cast<StackAllocationOp>(operation)) {
      if (!stackAllocationOp.getMarkedLifetimes()) {
        Operation *terminator =
            stackAllocationOp->getParentRegion()->front().getTerminator();
        int endState = isa<SuspendEndOp>(terminator) ? opToState[terminator] - 1
                                                     : opToState[terminator];
        if (endState > useState)
          addToFrame(stackAllocationFrameType(stackAllocationOp),
                     stackAllocationOp);
      }
    }

    for (Value operand : operation->getOperands()) {
      // Results and Errors are stored in the header of the continuation, not in
      // the frame.
      if (operand == errorValue || operand == resultValue)
        continue;
      if (valueToIndexInFrame.contains(operand))
        continue;

      // Add to frame if the value was defined in a previous state.
      int defState = getDefinitionStateForValue(opToState, operand, isHot);
      if (defState != useState) {
        bool isArgument = false;
        if (auto blockArg = dyn_cast<BlockArgument>(operand))
          isArgument =
              blockArg.getParentBlock()->getParentOp() == originalFunction;

        unsigned index = isArgument ? coldCoroTypes.size() : frameTypes.size();
        if (Operation *definingOp = operand.getDefiningOp()) {
          if (auto stackAllocation = dyn_cast<StackAllocationOp>(definingOp)) {
            // Stack allocations without marked lifetimes are checked for frame
            // membership at definition site.
            if (stackAllocation.getMarkedLifetimes()) {
              addToFrame(stackAllocationFrameType(stackAllocation), definingOp);
            } else {
              valueToIndexInFrame.insert(
                  {operand, operationToIndexInFrame[definingOp]});
              continue;
            }
          } else {
            addToFrame(operand.getType(), definingOp);
          }
        } else {
          if (isArgument)
            coldCoroTypes.push_back(operand);
          else
            frameTypes.push_back(operand.getType());
        }
        valueToIndexInFrame.insert({operand, index});
      }
    }
  });

  // Insert used arguments at end of frame.
  unsigned offset = frameTypes.size();
  for (auto [index, functionArg] : llvm::enumerate(coldCoroTypes)) {
    frameTypes.push_back(functionArg.getType());
    unsigned newIndex = offset + index;
    valueToIndexInFrame[functionArg] = newIndex;
  }
  for (auto [index, functionArg] :
       llvm::enumerate(originalFunction.getRegion().front().getArguments())) {
    auto frameSlotMaybe = valueToIndexInFrame.find(functionArg);
    if (frameSlotMaybe == valueToIndexInFrame.end())
      continue;
    argsInFrame.push_back(FrameData::ArgInFrame(index, frameSlotMaybe->second));
  }

  // Remember the virtual ops in the first state for hot start resume
  // generation.
  for (auto &virtualBlockAndList : predecessors) {
    VirtualBlock virtualBlock = virtualBlockAndList.first;
    if (opToState[virtualBlock] == 0) {
      // Stack allocations used across frames will be deleted.
      if (isa<StackAllocationOp>(virtualBlock) &&
          valueToIndexInFrame.contains(virtualBlock->getResults().front()))
        virtualBlocksFirstState.push_back(virtualBlock->getNextNode());
      else
        virtualBlocksFirstState.push_back(virtualBlock);
    }
  }
  // If a dominates b then a should appear first.
  llvm::sort(
      virtualBlocksFirstState.begin(), virtualBlocksFirstState.end(),
      [&](Operation *a, Operation *b) { return domInfo.dominates(a, b); });
}
