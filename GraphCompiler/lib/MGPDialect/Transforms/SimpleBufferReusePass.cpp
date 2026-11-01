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

#include "GraphCompiler/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "mgp-simple-buffer-reuse"

using namespace mlir;

namespace M {
namespace MGP {

struct ReuseKey {
  mlir::TypedValue<mlir::IndexType> size;
  Attribute device;
  uint64_t alignment;

  bool operator==(const ReuseKey &other) const {
    return size == other.size && device == other.device &&
           alignment == other.alignment;
  }
};

} // namespace MGP
} // namespace M

namespace llvm {
template <>
struct DenseMapInfo<M::MGP::ReuseKey> {
  using KeyTy = M::MGP::ReuseKey;

  static unsigned getHashValue(const KeyTy &k) {
    return hash_combine(DenseMapInfo<mlir::Value>::getHashValue(k.size),
                        DenseMapInfo<Attribute>::getHashValue(k.device),
                        DenseMapInfo<uint64_t>::getHashValue(k.alignment));
  }

  static bool isEqual(const KeyTy &a, const KeyTy &b) { return a == b; }
};
} // namespace llvm

namespace M {
namespace MGP {

// Generate support code for SimpleBufferReusePass
#define GEN_PASS_DEF_SIMPLEBUFFERREUSEPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

class SimpleBufferReusePass
    : public impl::SimpleBufferReusePassBase<SimpleBufferReusePass> {
  using SimpleBufferReusePassBase::SimpleBufferReusePassBase;
  using LifetimeRange = std::pair<int, int>;

  struct AllocInfo {
    MGP::BufferAllocOp alloc;
    int firstUseIndex;
    int lastUseIndex;
    SmallVector<Operation *> uses;
  };

  // Reuse is safe when every current use runs after every candidate use.
  bool canSafelyReuseBuffer(ArrayRef<Operation *> currentUses,
                            ArrayRef<Operation *> candidateUses,
                            const DependencyTree &depTree) {
    if (currentUses.empty() || candidateUses.empty())
      return true;

    for (Operation *currentUse : currentUses) {
      for (Operation *candidateUse : candidateUses) {
        if (!depTree.executesAfter(currentUse, candidateUse)) {
          return false;
        }
      }
    }
    return true;
  }

public:
  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    auto &analysis = getAnalysis<BufferLifetimeAnalysisPass>();
    int reuseCount = 0;

    // Erasing allocs invalidates the analysis, but the next pass only queries
    // alloc uses, which stay valid.
    markAnalysesPreserved<BufferLifetimeAnalysisPass>();

    modelOp->walk<mlir::WalkOrder::PreOrder>([&](Block *block) {
      if (isa_and_present<MOGG::AsyncRoutineOp>(block->getParentOp()))
        return WalkResult::skip();

      DenseMap<ReuseKey, SmallVector<MGP::BufferAllocOp>> allocGroups;

      for (auto allocOp : block->getOps<MGP::BufferAllocOp>()) {
        ReuseKey key = {allocOp.getRuntimeSize(), allocOp.getDevice(),
                        allocOp.getRawAlign()};
        allocGroups[key].push_back(allocOp);
      }

      if (allocGroups.empty())
        return WalkResult::advance();

      auto &depTree = analysis.getDepTree(block);

      for (auto &[key, allocs] : allocGroups) {
        if (allocs.size() < 2)
          continue;

        SmallVector<std::pair<MGP::BufferAllocOp, SmallVector<Operation *>>>
            validAllocs;
        validAllocs.reserve(allocs.size());

        for (auto alloc : allocs) {
          ErrorOr<SmallVector<Operation *>> uses = getAllocUses(alloc);
          if (!uses.isError())
            validAllocs.emplace_back(alloc, std::move(*uses));
        }

        // Allocs with more uses go first.
        llvm::stable_sort(validAllocs, [](const auto &a, const auto &b) {
          return a.second.size() > b.second.size();
        });

        for (unsigned i = 0; i < validAllocs.size(); ++i) {
          auto &pair = validAllocs[i];
          MGP::BufferAllocOp currentAlloc = pair.first;
          auto &currentUses = pair.second;
          if (!currentAlloc)
            continue; // Already erased.

          for (unsigned j = 0; j < validAllocs.size(); ++j) {
            if (i == j)
              continue;
            auto &candidatePair = validAllocs[j];
            MGP::BufferAllocOp candidateAlloc = candidatePair.first;
            auto &candidateUses = candidatePair.second;
            if (!candidateAlloc)
              continue;

            // Buffers used by the same op cannot alias without a read/write
            // conflict.
            bool sharesOperation = false;
            for (Operation *op : currentUses) {
              if (llvm::is_contained(candidateUses, op)) {
                sharesOperation = true;
                break;
              }
            }
            if (sharesOperation)
              continue;

            if (canSafelyReuseBuffer(currentUses, candidateUses, depTree)) {
              currentAlloc->replaceAllUsesWith(candidateAlloc);
              currentAlloc->erase();
              validAllocs[i].first = nullptr;

              SmallVector<Operation *> newUses;
              newUses.reserve(candidateUses.size() + currentUses.size());
              newUses.append(candidateUses.begin(), candidateUses.end());
              newUses.append(currentUses.begin(), currentUses.end());
              validAllocs[j].second = std::move(newUses);

              ++reuseCount;
              break;
            }
          }
        }
      }

      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs()
               << "Reused " << reuseCount << " buffer allocations\n");
  }
};

} // namespace MGP
} // namespace M
