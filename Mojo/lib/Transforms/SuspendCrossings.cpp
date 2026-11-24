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

#include "SuspendCrossings.h"
#include "Mojo/CODialect/COOps.h"

using namespace M;
using namespace KGEN;
using namespace CO;
using llvm::BitVector;

M::KGEN::SuspendCrossings::SuspendCrossings(const Analysis &analysis) {
  for (auto &[head, info] : analysis.cfg)
    blockIndex.try_emplace(head, blockIndex.size());
  unsigned numBlocks = blockIndex.size();
  reachableWithoutSuspend.resize(numBlocks);

  // TODO(stumpOS): Replace with a sparser form, such as an interval or a
  // reachability label per block, when async functions routinely carry
  // thousands of virtual blocks.

  // Seed the worklist and initialize entries for each block.
  llvm::SetVector<VirtualBlock> worklist;
  for (auto &[head, index] : blockIndex) {
    reachableFrom.try_emplace(head, numBlocks).first->second.set(index);
    suspendFrom.try_emplace(head, numBlocks);
    worklist.insert(head);
  }

  // DFS through graph to mark reachability
  while (!worklist.empty()) {
    VirtualBlock currentBlock = worklist.pop_back_val();
    const VirtualBlockInfo &info = analysis.cfg.at(currentBlock);
    unsigned currentIndex = blockIndex.at(currentBlock);

    BitVector reachable(numBlocks);
    reachable.set(currentIndex);
    BitVector crossed(numBlocks);

    bool withoutSuspend = currentBlock == analysis.entryBlock;
    for (VirtualBlock predecessor : info.predecessors) {
      unsigned predecessorIndex = blockIndex.at(predecessor);

      const BitVector &predecessorReachable = reachableFrom.at(predecessor);
      reachable |= predecessorReachable;

      BitVector predecessorCrossed = suspendFrom.at(predecessor);
      Operation *predecessorEnd = analysis.cfg.at(predecessor).lastOp;
      if (isa<SuspendEndOp>(predecessorEnd))
        predecessorCrossed |= predecessorReachable;
      else
        predecessorCrossed.reset(predecessorIndex);

      crossed |= predecessorCrossed;
      withoutSuspend |= reachableWithoutSuspend.test(predecessorIndex) &&
                        !isa<SuspendOp, SuspendEndOp>(predecessorEnd);
    }

    bool changed = false;
    if (withoutSuspend && !reachableWithoutSuspend.test(currentIndex)) {
      reachableWithoutSuspend.set(currentIndex);
      changed = true;
    }
    BitVector &recordedReachable = reachableFrom.at(currentBlock);
    if (reachable != recordedReachable) {
      recordedReachable = std::move(reachable);
      changed = true;
    }
    BitVector &recordedCrossed = suspendFrom.at(currentBlock);
    if (crossed != recordedCrossed) {
      recordedCrossed = std::move(crossed);
      changed = true;
    }
    if (changed)
      for (VirtualBlock successor : info.successors)
        worklist.insert(successor);
  }
  entryIndex = blockIndex.at(analysis.entryBlock);
  // conservatively assume unreachable blocks have suspension point.
  for (auto &[head, reachable] : reachableFrom)
    if (!isReachable(head))
      suspendFrom.at(head).set();
}
