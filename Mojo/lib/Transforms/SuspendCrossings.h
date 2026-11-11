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

#ifndef KGEN_TRANSFORMS_SUSPENDCROSSINGS_H
#define KGEN_TRANSFORMS_SUSPENDCROSSINGS_H

#include "FrameAnalysis.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace M::KGEN {

using llvm::BitVector;

/// Answers whether a suspension point separates two virtual blocks, and which
/// blocks a path from the entry reaches without suspending.
class SuspendCrossings {
public:
  SuspendCrossings(const Analysis &analysis);

  bool isReachableFrom(VirtualBlock from, VirtualBlock to) const {
    return reachableFrom.at(to).test(blockIndex.at(from));
  }

  bool isAfterSuspend(VirtualBlock block) const {
    return suspendFrom.at(block).test(entryIndex);
  }

  bool isReachableWithoutSuspend(VirtualBlock block) const {
    return reachableWithoutSuspend.test(blockIndex.at(block));
  }

  bool isReachable(VirtualBlock block) const {
    return reachableFrom.at(block).test(entryIndex);
  }

  bool crosses(VirtualBlock definition, VirtualBlock use) const {
    if (definition == use)
      return false;
    return suspendFrom.at(use).test(blockIndex.at(definition));
  }

private:
  DenseMap<VirtualBlock, unsigned> blockIndex;
  /// `suspendFrom[dst][src]` is set iff a suspend lies on some path from `src`
  /// to `dst`
  DenseMap<VirtualBlock, BitVector> suspendFrom;
  /// `reachableFrom[dst][src]` is set if there exists a path from src to dst.
  DenseMap<VirtualBlock, BitVector> reachableFrom;
  /// Bit `i` is set iff there exists a path from entry to block `i`
  /// without hitting a suspension point.
  BitVector reachableWithoutSuspend;
  unsigned entryIndex = 0;
};

} // namespace M::KGEN

#endif // KGEN_TRANSFORMS_SUSPENDCROSSINGS_H
