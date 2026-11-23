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

#include "GraphCompiler/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Support/AssertStream.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "mgp-dynamic-memory-planning"

using namespace mlir;

namespace M::MGP {

// Generate support code for DynamicMemoryPlanningPass
#define GEN_PASS_DEF_DYNAMICMEMORYPLANNINGPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

struct AllocInfo {
  MGP::BufferAllocOp alloc;
  Value runtimeSize;
  uint64_t alignment;
  SmallVector<Operation *> uses;
};

/// Returns true if `v` is computable from block arguments by arithmetic and
/// tensor_spec ops alone, i.e. an allocation size that is not data-dependent.
/// Results are cached per producing op in `trivial`.
bool isTriviallyComputable(Value v, DenseMap<Operation *, bool> &trivial) {
  Operation *producer = v.getDefiningOp();

  if (!producer)
    return true;

  if (auto it = trivial.find(producer); it != trivial.end())
    return it->second;

  auto cacheResult = [&](bool result) -> bool {
    trivial[producer] = result;
    return result;
  };

  bool isTrivialOp = producer->hasTrait<OpTrait::ConstantLike>() ||
                     isa<KGEN::POP::POPDialect>(*producer->getDialect()) ||
                     isa<mlir::index::IndexDialect>(*producer->getDialect()) ||
                     isa<TensorExtractTensorSpecOp, TensorSpecCreateOp,
                         BufferAllocOp, TensorSpecGetDimOp>(*producer);
  if (!isTrivialOp)
    return cacheResult(false);

  auto trivialOperands =
      llvm::all_of(producer->getOperands(), [&](Value operand) {
        return isTriviallyComputable(operand, trivial);
      });

  return cacheResult(trivialOperands);
}

/// Returns the blocks whose allocations are planned, each with the chain its
/// pool is allocated on: `execute`, then every control flow region block in
/// it. Regions are isolated from above, so each block's pool takes the block's
/// own chain argument; a block without one is not planned.
SmallVector<std::pair<Block *, Value>> getBlocksToPlan(Block *execute) {
  SmallVector<std::pair<Block *, Value>> blocks;
  auto addBlock = [&](Block *block) {
    auto chainArg = llvm::find_if(block->getArguments(), [](Value arg) {
      return isa<MLRT::MEF::ChainType>(arg.getType());
    });
    if (chainArg != block->args_end())
      blocks.emplace_back(block, *chainArg);
  };
  addBlock(execute);
  execute->walk([&](MLRT::GRT::GRTControlFlowInterface cfOp) {
    for (Region &region : cfOp->getRegions())
      for (Block &block : region)
        addBlock(&block);
  });
  return blocks;
}

} // end namespace

class DynamicMemoryPlanningPass
    : public impl::DynamicMemoryPlanningPassBase<DynamicMemoryPlanningPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();
    ExecuteState execState(modelOp->getContext(), modelOp);
    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);

    // A region block gets its own pools, so a region that runs every step
    // makes one allocation per device rather than one per buffer.
    SmallVector<BufferAllocOp> allocsToDelete;
    for (auto [block, chain] : getBlocksToPlan(execState.execute))
      planBlock(execState, builderContext, block, chain, allocsToDelete);

    for (BufferAllocOp alloc : allocsToDelete)
      alloc.erase();
  }

  /// Plans `block`'s allocations into one pool per device, allocated on
  /// `chain`.
  void planBlock(ExecuteState &execState, BufferBuilderContext &builderContext,
                 Block *block, Value chain,
                 SmallVectorImpl<BufferAllocOp> &allocsToDelete) {
    ModelOp modelOp = getOperation();

    auto getDeviceContext = [&modelOp](DeviceRefAttr dev,
                                       ArrayRef<BlockArgument> devices) {
      for (auto [deviceRefAttr, deviceContext] :
           llvm::zip_equal(modelOp.getDevices(), devices))
        if (deviceRefAttr == dev)
          return std::optional<Value>(deviceContext);
      return std::optional<Value>();
    };

    auto &analysis = getAnalysis<BufferLifetimeAnalysisPass>();
    OpBuilder &execBuilder = execState.builder;

    DependencyTree &tree = analysis.getDepTree(block);

    llvm::MapVector<DeviceRefAttr, SmallVector<AllocInfo>> deviceToAllocs;

    DenseMap<Operation *, bool> trivialVals;

    for (auto allocOp : block->getOps<MGP::BufferAllocOp>()) {
      // TODO(GEX-557): Plan CPU allocations too.
      if (allocOp.getDevice().getLabel() == kCPULabel)
        continue;

      ErrorOr<SmallVector<Operation *>> allocUses = getAllocUses(allocOp);

      if (allocUses.isError() || allocUses->empty())
        continue;

      if (!isTriviallyComputable(allocOp.getRuntimeSize(), trivialVals))
        continue;

      SmallVector<Operation *> uses;
      bool ineligibleAlloc = false;
      for (auto *use : allocUses.get()) {
        if (!tree.contains(use)) {
          ineligibleAlloc = true;
          break;
        }
        uses.push_back(use);
      }

      if (ineligibleAlloc)
        continue;

      SmallVector<AllocInfo> &allocs = deviceToAllocs[allocOp.getDevice()];
      allocs.push_back({allocOp, allocOp.getRuntimeSize(),
                        allocOp.getRawAlign(), std::move(uses)});
    }

    // At the block start, for readability.
    execBuilder.setInsertionPointToStart(block);

    for (auto &[device, allocs] : deviceToAllocs) {
      if (allocs.size() <= 1)
        continue;

      // Plan allocations needed sooner first.
      auto firstTopoIndex = [&](const AllocInfo &info) {
        unsigned minIdx = std::numeric_limits<unsigned>::max();
        for (Operation *use : info.uses)
          minIdx = std::min(minIdx, tree.getTopoIndex(use));
        return minIdx;
      };
      llvm::sort(allocs, [&](const auto &a, const auto &b) {
        return firstTopoIndex(a) < firstTopoIndex(b);
      });

      SmallVector<uint64_t> alignments;
      SmallVector<Value> runtimeSizes;

      // The pool is aligned to the LCM of all alignments.
      uint64_t totalAlignment = 1;
      for (const AllocInfo &a : allocs) {
        runtimeSizes.push_back(a.runtimeSize);
        alignments.push_back(a.alignment);
        totalAlignment = std::lcm(totalAlignment, a.alignment);
      }

      // canShare[i][j] = 1 iff all uses of i run after all uses of j, or vice
      // versa.
      size_t n = allocs.size();
      SmallVector<uint8_t> canShare(n * n, 0);
      for (size_t i : llvm::seq(n)) {
        for (size_t j : llvm::seq(i + 1, n)) {
          uint8_t sharable =
              (tree.executesAfter(allocs[i].uses, allocs[j].uses) ||
               tree.executesAfter(allocs[j].uses, allocs[i].uses))
                  ? 1
                  : 0;

          canShare[i * n + j] = sharable;
          canShare[j * n + i] = sharable;
        }
      }

      auto locations = llvm::map_to_vector(
          allocs, [](const AllocInfo &alloc) { return alloc.alloc->getLoc(); });

      auto loc = execBuilder.getFusedLoc(locations);
      MGPBufferBuilder &bufferBuilder = builderContext.getBufferBuilder(device);

      // A region is isolated from above, so its pool reuses the device
      // context its allocations take.
      std::optional<Value> deviceContext =
          block == execState.execute
              ? getDeviceContext(device, modelOp.getExecuteDevices())
              : std::optional<Value>(allocs.front().alloc.getDeviceContext());
      ASSERT_STREAM(deviceContext, "Could not find device context for device");

      auto bufferPlanOp = BufferPlanOp::create(execBuilder, loc, runtimeSizes,
                                               alignments, canShare,
                                               /*staticSizes=*/{});

      TypedValue<IndexType> poolSize = bufferPlanOp.getHighWatermark();

      Value alloc = bufferBuilder.bufferAlloc(
          execBuilder, loc, getContextForBlock(block), chain, poolSize,
          deviceContext.value(), totalAlignment);

      auto memPoolBuffer = cast<TypedValue<DeviceBufferType>>(alloc);

      // One bulk-slice op rather than N buffer.slice ops: sequence fusion can
      // duplicate each slice op independently (O(N) fan-out).
      SmallVector<Value> offsets(bufferPlanOp.getOffsets().begin(),
                                 bufferPlanOp.getOffsets().end());
      SmallVector<Value> sizes;
      for (const AllocInfo &a : allocs)
        sizes.push_back(a.runtimeSize);

      auto bulkSliceOp =
          BufferBulkSliceOp::create(execBuilder, loc, memPoolBuffer,
                                    deviceContext.value(), offsets, sizes);

      for (size_t i = 0; i < allocs.size(); ++i) {
        allocs[i].alloc->replaceAllUsesWith(
            ValueRange{bulkSliceOp.getOutBuffers()[i]});
        allocsToDelete.push_back(allocs[i].alloc);
      }
    }
  }
};
} // namespace M::MGP
