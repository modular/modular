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

#include "FrameAnalysis.h"
#include "Mojo/CODialect/COOps.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

using namespace M;
using namespace KGEN;
using M::HLCF::ReturnOp;
using M::HLCF::UnreachableOp;
using namespace POP;
using namespace CO;
using llvm::BitVector;

namespace {

struct ControlFlowEdge {
  Operation *source;
  Operation *target;
};

/// Appends the outgoing control-flow edges of one op.
void addControlFlowEdges(Operation &op,
                         SmallVectorImpl<ControlFlowEdge> &edges) {
  if (isa<ReturnOp, UnreachableOp>(op))
    return;

  auto addTargets = [&](SmallVector<HLCF::ControlFlowTarget> const &targets,
                        Operation *controlFlowParent) {
    for (HLCF::ControlFlowTarget target : targets) {
      Operation *targetOp = nullptr;
      if (target.index.has_value()) {
        Region *targetRegion =
            &controlFlowParent->getRegion(target.index.value());
        // The block below is the target's entry, so it must be the whole
        // region.
        assert(targetRegion->hasOneBlock() &&
               "HLCF regions hold a single block");
        targetOp = &*targetRegion->front().begin();
      } else {
        targetOp = controlFlowParent->getNextNode();
      }
      // A branch with no target past the end of its region ends control flow.
      if (targetOp)
        edges.push_back({&op, targetOp});
    }
  };

  if (auto terminator = dyn_cast<HLCF::ControlFlowTerminator>(op)) {
    SmallVector<HLCF::ControlFlowTarget> targets;
    SmallVector<Attribute> operands(terminator->getNumOperands(), Attribute());
    terminator.getBranchTargets(operands, targets);
    addTargets(targets, getParentNode(terminator));
  }
  if (auto controlFlowNode = dyn_cast<HLCF::ControlFlowNode>(op)) {
    SmallVector<HLCF::ControlFlowTarget> targets;
    SmallVector<Attribute> operands(controlFlowNode->getNumOperands(),
                                    Attribute());
    controlFlowNode.getEntryTargets(operands, targets);
    addTargets(targets, controlFlowNode);
  }
  if (auto suspend = dyn_cast<CO::SuspendOp>(op)) {
    Operation *next = suspend->getNextNode();
    if (!next)
      return;
    edges.push_back({&op, &*suspend.getBody().front().begin()});
    // Terminator is used because that will trigger updated state.
    edges.push_back({suspend.getBody().front().getTerminator(), next});
  }
}

/// Runs an iterative DFS from the entry that records the reachable postorder in
/// `analysis` and, for every loop, its header and latches in `latches`.
void computePostorder(
    Analysis &analysis,
    llvm::MapVector<VirtualBlock, SmallVector<VirtualBlock>> &latches) {
  enum class State { New, OnStack, Done };
  DenseMap<VirtualBlock, State> state;
  auto stateOf = [&](VirtualBlock block) {
    auto it = state.find(block);
    return it == state.end() ? State::New : it->second;
  };

  SmallVector<std::pair<VirtualBlock, unsigned>> stack;
  stack.push_back({analysis.entryBlock, 0});
  state[analysis.entryBlock] = State::OnStack;
  while (!stack.empty()) {
    auto &[block, nextSuccessor] = stack.back();
    const SmallVector<VirtualBlock> &successors =
        analysis.cfg.at(block).successors;
    if (nextSuccessor == successors.size()) {
      state[block] = State::Done;
      analysis.postorder.push_back(block);
      stack.pop_back();
      continue;
    }
    VirtualBlock successor = successors[nextSuccessor++];
    switch (stateOf(successor)) {
    case State::OnStack:
      latches[successor].push_back(block);
      break;
    case State::New:
      state[successor] = State::OnStack;
      stack.push_back({successor, 0});
      break;
    case State::Done:
      break;
    }
  }

  for (auto [index, block] : llvm::enumerate(analysis.postorder))
    analysis.postorderIndex[block] = index;
}

bool endsVirtualBlock(Operation *op) {
  return isa<HLCF::ControlFlowNode, HLCF::ControlFlowTerminator, SuspendOp>(op);
}

} // namespace

M::KGEN::BlockTransferFunction M::KGEN::transferFunctionOperands(
    const Analysis &analysis, VirtualBlock currentBlock, unsigned numValues) {
  BlockTransferFunction transfer{
      /*usedButNotDefinedInMe=*/BitVector(numValues, false),
      /*invertDefinedByMe=*/BitVector(numValues, true)};
  const VirtualBlockInfo &info = analysis.cfg.at(currentBlock);
  auto operations = llvm::make_range(Block::iterator(currentBlock),
                                     std::next(Block::iterator(info.lastOp)));
  for (Operation &operation : llvm::reverse(operations)) {
    for (Value result : operation.getResults()) {
      unsigned index = analysis.valueToIndex.at(result);
      transfer.usedButNotDefinedInMe.reset(index);
      transfer.invertDefinedByMe.reset(index);
    }
    for (Value operand : operation.getOperands()) {
      unsigned index = analysis.valueToIndex.at(operand);
      transfer.usedButNotDefinedInMe.set(index);
      transfer.invertDefinedByMe.set(index);
    }
  }
  // Block arguments are considered definitions of this block.
  Block *block = currentBlock->getBlock();
  if (currentBlock == &block->front() && !isa<FuncOp>(block->getParentOp())) {
    for (BlockArgument argument : block->getArguments()) {
      unsigned index = analysis.valueToIndex.at(argument);
      transfer.usedButNotDefinedInMe.reset(index);
      transfer.invertDefinedByMe.reset(index);
    }
  }

  return transfer;
}

Analysis M::KGEN::analyze(FuncOp function) {
  Analysis analysis;
  auto addValue = [&](Value value) {
    analysis.valueToIndex[value] = analysis.allValues.size();
    analysis.allValues.push_back(value);
  };

  SmallVector<ControlFlowEdge> edges;
  auto &opToVirtualBlock = analysis.opToVirtualBlock;
  function->walk([&](Block *block) {
    assert(block->getParent()->hasOneBlock() &&
           "HLCF regions hold a single block");
    for (BlockArgument argument : block->getArguments())
      addValue(argument);
    VirtualBlock head = nullptr;
    for (Operation &op : *block) {
      for (Value result : op.getResults())
        addValue(result);
      auto stackAllocation = dyn_cast<StackAllocationOp>(op);
      if (stackAllocation && !stackAllocation.getMarkedLifetimes())
        analysis.unmarkedAllocations.push_back(stackAllocation);

      if (!head) {
        head = &op;
        analysis.cfg[head];
        analysis.blockToVirtualBlock.try_emplace(block, head);
        if (isa_and_nonnull<SuspendOp>(op.getPrevNode()))
          analysis.resumeBlocks.push_back(head);
      }
      analysis.cfg[head].lastOp = &op;
      opToVirtualBlock[&op] = head;
      addControlFlowEdges(op, edges);
      if (endsVirtualBlock(&op))
        head = nullptr;
    }
  });
  analysis.entryBlock =
      opToVirtualBlock.lookup(&function.getBodyRegion().front().front());

  for (ControlFlowEdge edge : edges) {
    VirtualBlock from = opToVirtualBlock.lookup(edge.source);
    VirtualBlock to = opToVirtualBlock.lookup(edge.target);
    assert(from && to && "virtual block edge endpoints must be in function");
    analysis.cfg[from].successors.push_back(to);
    analysis.cfg[to].predecessors.push_back(from);
  }
  // Order the CFG and collect latch/header pairs.
  llvm::MapVector<VirtualBlock, SmallVector<VirtualBlock>> latches;
  computePostorder(analysis, latches);
  analysis.loops.reserve(latches.size());
  for (auto &[header, headerLatches] : latches)
    analysis.loops.push_back({header, std::move(headerLatches)});
  llvm::sort(analysis.loops, [&](const Loop &a, const Loop &b) {
    return analysis.postorderIndex.at(a.header) >
           analysis.postorderIndex.at(b.header);
  });
  return analysis;
}

void M::KGEN::propagateBackEdges(const Analysis &analysis,
                                 DenseMap<VirtualBlock, BitVector> &liveIn) {
  for (const Loop &loop : analysis.loops) {
    const BitVector &headerLiveIn = liveIn.at(loop.header);
    DenseSet<VirtualBlock> visited;
    SmallVector<VirtualBlock> worklist(loop.latches.begin(),
                                       loop.latches.end());
    while (!worklist.empty()) {
      VirtualBlock block = worklist.pop_back_val();
      if (block == loop.header || !visited.insert(block).second)
        continue;
      BitVector &recorded = liveIn.at(block);
      BitVector updated = recorded;
      updated |= headerLiveIn;
      if (updated == recorded)
        continue;
      recorded = std::move(updated);
      for (VirtualBlock predecessor : analysis.cfg.at(block).predecessors)
        if (analysis.postorderIndex.contains(predecessor))
          worklist.push_back(predecessor);
    }
  }
}
