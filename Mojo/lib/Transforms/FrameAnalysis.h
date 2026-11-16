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

#ifndef KGEN_TRANSFORMS_FRAMEANALYSIS_H
#define KGEN_TRANSFORMS_FRAMEANALYSIS_H

#include "FrameData.h"
#include "Mojo/POPDialect/POPOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace M::KGEN {

using llvm::BitVector;
using POP::StackAllocationOp;

/// A block's transfer function in closed form:
///   liveIn(B) = usedButNotDefinedInMe(B) OR (liveOut(B) AND ~definedIn(B))
struct BlockTransferFunction {
  BitVector usedButNotDefinedInMe;
  BitVector invertDefinedByMe;

  BitVector operator()(BitVector liveOut) const {
    liveOut &= invertDefinedByMe;
    liveOut |= usedButNotDefinedInMe;
    return liveOut;
  }
};

/// One loop: the header its back edges target and the latches that end them.
struct Loop {
  VirtualBlock header;
  SmallVector<VirtualBlock> latches;
};

/// The span of a virtual block: from its key (the first op) to `lastOp`.
struct VirtualBlockInfo {
  Operation *lastOp = nullptr;
  SmallVector<VirtualBlock> successors;
  SmallVector<VirtualBlock> predecessors;
};

/// Indexes every value and splits the function into virtual blocks, giving the
/// control flow graph the liveness and suspend analyses query.
struct Analysis {
  // All the values in the async function.
  SmallVector<Value> allValues;
  // Given a value, find its index in the BitVector
  DenseMap<Value, unsigned> valueToIndex;
  // Given an op, find its virtual block.
  DenseMap<Operation *, VirtualBlock> opToVirtualBlock;
  // Given a block, find the virtual block that starts it.
  DenseMap<Block *, VirtualBlock> blockToVirtualBlock;
  // The control flow graph of the virtual blocks: each block's span and its
  // predecessors and successors.
  DenseMap<VirtualBlock, VirtualBlockInfo> cfg;
  // Values live at the entry point are stored in the frame for cold coroutines.
  VirtualBlock entryBlock = nullptr;
  /// The blocks starting right after each suspend. Values live into them must
  /// be stored in the frame.
  SmallVector<VirtualBlock> resumeBlocks;
  /// The stack allocations without marked lifetimes, in definition order.
  SmallVector<StackAllocationOp> unmarkedAllocations;
  /// The blocks reachable from the entry, in DFS postorder
  SmallVector<VirtualBlock> postorder;
  /// Position of each reachable block in `postorder`; membership marks
  /// reachability.
  DenseMap<VirtualBlock, unsigned> postorderIndex;
  /// The loops the CFG closes, outer to inner.
  SmallVector<Loop> loops;
};

/// Builds the CFG of `function` and the loops its back edges close.
Analysis analyze(FuncOp function);

/// Walks a block's ops once, so the sweep's transfers are bit operations
/// rather than op walks.
BlockTransferFunction transferFunctionOperands(const Analysis &analysis,
                                               VirtualBlock currentBlock,
                                               unsigned numValues);

/// Pushes each loop header's live-in through the loop body, so values live
/// around a back edge reach the blocks that cycle through it.
void propagateBackEdges(const Analysis &analysis,
                        DenseMap<VirtualBlock, llvm::BitVector> &liveIn);

} // namespace M::KGEN

#endif // KGEN_TRANSFORMS_FRAMEANALYSIS_H
