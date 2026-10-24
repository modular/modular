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

#include "GraphCompiler/MAPDialect/Support/InputFusion.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Support/DataflowUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEINPUT
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

bool isViewForeach(IterElementwiseOp op) {
  return op.getBody()
      .walk([&](IndexTransformOpInterface) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

bool producesThroughView(StoreOp store) {
  return dataflowGraphContains<WalkDirection::Producers>(
      store.getValue().getDefiningOp(),
      [](Operation *o) { return isa<IndexTransformOpInterface>(o); });
}

bool isPureViewForeach(IterElementwiseOp op) {
  return isViewForeach(op) &&
         !op.getBody()
              .walk([&](CallElementwiseOp) { return WalkResult::interrupt(); })
              .wasInterrupted();
}

// Above this many elements, re-reading a tensor operand is expensive.
constexpr int64_t kLargeTensorElementThreshold = 4096;

static bool isLargeTensorOperand(Type type) {
  auto tensorType = dyn_cast<MO::TensorType>(type);
  if (!tensorType)
    return false;
  std::optional<int64_t> numElements = tensorType.getNumElements();
  return !numElements.has_value() ||
         *numElements > kLargeTensorElementThreshold;
}

bool isCheapToDuplicate(IterElementwiseOp op) {
  int numLargeOperands = 0;
  for (Value input : op.getInputs())
    if (isLargeTensorOperand(input.getType()))
      if (++numLargeOperands > 1)
        return false;
  return true;
}

bool loadIndexCrossesReshape(LoadOp load) {
  return dataflowGraphContains<WalkDirection::Producers>(
      load.getIndex().getDefiningOp(),
      [](Operation *o) { return isa<ReshapeIndexOp>(o); });
}

bool anyLoadCrossesReshape(Value tensor) {
  return llvm::any_of(tensor.getUsers(), [](Operation *user) {
    auto load = dyn_cast<LoadOp>(user);
    return load && loadIndexCrossesReshape(load);
  });
}

namespace {

// TODO(GEX-4184): support multiple stores/outputs, which only arise via
// sibling fusion.
StoreOp producerStoreOf(IterElementwiseOp src) {
  return getUniqueStore(src.getBody().front());
}

// Guards against fusing across a chain-carrying kernel (e.g. mo.mutable.store)
// when doing so could change effective ordering relative to another consumer of
// the same chain.
bool checkMutableKernelFusible(Value srcInChain, Value srcOutChain,
                               Value dstInChain, Value dstOutChain) {
  if (srcInChain && srcOutChain && dstInChain && dstOutChain) {
    if (dstInChain != srcOutChain)
      return false;
    if (!srcOutChain.hasOneUse())
      return false;
  }
  if (dstInChain && !dstInChain.hasOneUse())
    return false;
  return true;
}

// The prologue capturing `dst`'s `argIdx`'th input and the capture's index, or
// null unless the input's only user is a prologue whose capture's only use is
// a load. Epilogue captures are reached through output fusion instead.
struct ClosureCapture {
  CapturingFusionClosure closure = nullptr;
  unsigned captureIdx = 0;
};

ClosureCapture findExistingPrologueCapture(IterOpaqueOp dst, unsigned argIdx) {
  BlockArgument dstArg = dst.getInputBlockArgs()[argIdx];
  if (!dstArg.hasOneUse())
    return {};
  auto closure = dyn_cast<CapturingFusionClosure>(*dstArg.getUsers().begin());
  if (!closure || !isa<IterPrologueOp>(closure.getOperation()))
    return {};
  auto captureIt = llvm::find(closure.getCaptures(), dstArg);
  if (captureIt == closure.getCaptures().end())
    return {};
  unsigned captureIdx =
      static_cast<unsigned>(captureIt - closure.getCaptures().begin());
  BlockArgument capArg = closure.getCaptureBlockArgs()[captureIdx];
  if (!capArg.hasOneUse())
    return {};
  // E.g. dynamic `map.index.slice` bounds have no load site.
  if (!isa<LoadOp>(*capArg.getUsers().begin()))
    return {};
  return {closure, captureIdx};
}

// An opaque consumer carries no device, so the result's tensor type stands in.
bool sameDevice(IterElementwiseOp src, Operation *dst) {
  auto srcDev = MO::getDeviceOrDefault(src);
  if (auto dstForeach = dyn_cast<IterElementwiseOp>(dst))
    return MO::getDeviceOrDefault(dstForeach) == srcDev;
  auto tensorTy = dyn_cast<MO::TensorType>(src.getResult(0).getType());
  return tensorTy && MO::getDeviceOrDefault(tensorTy) == srcDev;
}

// The first load of `src`'s result in `dst`, the site `shouldFuse` inspects.
LoadOp firstConsumerLoad(IterElementwiseOp dst, IterElementwiseOp src) {
  Block &body = dst.getBody().front();
  for (auto [i, input] : llvm::enumerate(dst.getInputs())) {
    if (input != src.getResult(0))
      continue;
    for (Operation *user : body.getArgument(i).getUsers())
      if (auto load = dyn_cast<LoadOp>(user))
        return load;
  }
  return nullptr;
}

// Folds the loads of `tensor` that read the same index in the same block into
// the first of them, as CSE would.
void mergeRedundantLoads(BlockArgument tensor, mlir::RewriterBase &rewriter) {
  DenseMap<std::pair<Block *, Value>, LoadOp> first;
  for (Operation *user : llvm::to_vector(tensor.getUsers())) {
    auto load = cast<LoadOp>(user);
    auto [it, inserted] =
        first.try_emplace({load->getBlock(), load.getIndex()}, load);
    if (inserted)
      continue;
    LoadOp kept = it->second;
    if (load->isBeforeInBlock(kept))
      std::swap(load, kept);
    rewriter.replaceAllUsesWith(load.getResult(), kept.getResult());
    rewriter.eraseOp(load);
    it->second = kept;
  }
}

// Splices `src` into `dst` and records it as a view or elementwise fusion.
void spliceIntoForeach(IterElementwiseOp dst, IterElementwiseOp src,
                       PatternRewriter &rewriter) {
  bool isView = producesThroughView(producerStoreOf(src));
  spliceProducer(dst, src, rewriter);

  if (isView)
    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionViewAttr>(
        src->getContext(), {src, dst}));
  else
    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionElementwiseAttr>(
        src->getContext(), {src, dst}));
  dst->setLoc(
      mlir::FusedLoc::get(src->getContext(), {src->getLoc(), dst->getLoc()}));
  dst.setHasReadEffect(src.getHasReadEffect() || dst.getHasReadEffect());
  dst.setHasWriteEffect(src.getHasWriteEffect() || dst.getHasWriteEffect());
  // `dst` keeps its own launch policy, as on the old path: a sync-launch
  // consumer such as a transpose (GEX-2569) must not turn async.
}

// Splices `src`'s body into `dst`'s existing prologue behind the captures
// `src` feeds.
void spliceIntoClosure(IterOpaqueOp dst, IterElementwiseOp src,
                       PatternRewriter &rewriter) {
  SmallVector<unsigned> fusionIndices;
  for (auto [i, input] : llvm::enumerate(dst.getInputs()))
    if (input == src.getResult(0))
      fusionIndices.push_back(i);
  assert(!fusionIndices.empty() && "src must be a current input of dst");

  unsigned numSrcInputs = src.getInputs().size();

  for (unsigned argIdx : llvm::reverse(fusionIndices)) {
    ClosureCapture capture = findExistingPrologueCapture(dst, argIdx);
    assert(capture.closure && "findEligibleProducer should have verified this");
    CapturingFusionClosure closure = capture.closure;
    // Captured before cloning inserts a second block ahead of it.
    Block &closureBody = closure.getBody().front();

    BlockArgument oldCap = closure.getCaptureBlockArgs()[capture.captureIdx];
    auto oldLoad = cast<LoadOp>(*oldCap.getUsers().begin());
    Value loadIndex = oldLoad.getIndex();

    rewriter.cloneRegionBefore(src.getBody(), closure.getBody(),
                               closure.getBody().begin());
    Block &srcClone = closure.getBody().front();

    auto storeOp = getUniqueStore(srcClone);
    Value fusedVal = storeOp.getValue();
    storeOp.erase();
    srcClone.eraseArgument(numSrcInputs);

    SmallVector<Value> fusedArgs;
    for (Value input : src.getInputs()) {
      unsigned dstInputIdx = dst.getInputs().size();
      dst.insertInputsAt(dstInputIdx, input);
      Value dstInputArg = dst.getInputBlockArgs()[dstInputIdx];
      closure.appendCaptureOperand(dstInputArg);
      fusedArgs.push_back(
          closureBody.addArgument(input.getType(), input.getLoc()));
    }
    fusedArgs.push_back(loadIndex);

    rewriter.inlineBlockBefore(&srcClone, oldLoad, fusedArgs);

    oldLoad.getResult().replaceAllUsesWith(fusedVal);
    oldLoad.erase();
    closure.eraseCaptureAt(capture.captureIdx);
    dst.eraseInputsAt(argIdx);

    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionPrologueAttr>(
        src->getContext(), {src, dst}, {argIdx}));
    dst->setLoc(
        mlir::FusedLoc::get(src->getContext(), {src->getLoc(), dst->getLoc()}));
  }

  // `dst` keeps its own launch policy, as on the old path: a sync-launch
  // consumer such as a transpose (GEX-2569) must not turn async.
}

// Whether `dstOp` has a site to read `src`'s result through AND `shouldFuse`
// accepts it. Without the latter, duplication would clone `src` into some users
// while another (e.g. an opaque, for the elementwise fuser) keeps it
// materialized -- pure recomputation.
bool hasFusionSite(IterElementwiseOp src, Operation *dstOp,
                   const InputFusionConfig &config) {
  if (src->getBlock() != dstOp->getBlock())
    return false;

  if (auto dst = dyn_cast<IterElementwiseOp>(dstOp)) {
    LoadOp load = firstConsumerLoad(dst, src);
    return load && config.shouldFuse(producerStoreOf(src), load);
  }

  auto dst = dyn_cast<IterOpaqueOp>(dstOp);
  if (!dst)
    return false;
  bool feedsInput = false;
  for (auto [i, in] : llvm::enumerate(dst.getInputs())) {
    if (in != src.getResult(0))
      continue;
    ClosureCapture capture = findExistingPrologueCapture(dst, i);
    if (!capture.closure)
      return false;
    BlockArgument capArg =
        capture.closure.getCaptureBlockArgs()[capture.captureIdx];
    if (capArg.use_empty())
      return false;
    auto load = dyn_cast<LoadOp>(*capArg.getUsers().begin());
    if (!load || !config.shouldFuse(producerStoreOf(src), load))
      return false;
    feedsInput = true;
  }
  return feedsInput;
}

// Duplication pays only by removing the materialization, so every user must
// absorb the clone; `mo.output`, or an opaque with no prologue on that input,
// keeps it in memory anyway.
bool duplicationAllowed(IterElementwiseOp src,
                        const InputFusionConfig &config) {
  if (src.getResult(0).hasOneUse())
    return true;
  for (Operation *user : src.getResult(0).getUsers())
    if (!hasFusionSite(src, user, config))
      return false;
  return config.canDuplicate(src);
}

// An eligible producer feeding `dstOp`, or null. A producer feeding several
// inputs must clear all of them, since the splice fuses them all.
IterElementwiseOp findEligibleProducer(Operation *dstOp,
                                       const InputFusionConfig &config) {
  if (auto dst = dyn_cast<IterElementwiseOp>(dstOp)) {
    Block &body = dst.getBody().front();
    DenseSet<Operation *> tried;
    // Fusing a producer moves it when `dst` is its only reader and copies it
    // otherwise. Take the single-use ones first: once `d = tanh(r)` is in,
    // `dst` is `r`'s only reader and `r` moves too. Taking `r` first copies it,
    // then copies it again inside `d`, doubling per level of such a chain.
    IterElementwiseOp duplicable;
    for (Value input : dst.getInputs()) {
      auto src = input.getDefiningOp<IterElementwiseOp>();
      if (!src || src.getNumResults() != 1 || !tried.insert(src).second)
        continue;
      // Fusion stays within a block: never across a control-flow boundary.
      if (src->getBlock() != dstOp->getBlock())
        continue;
      if (!sameDevice(src, dst) || !duplicationAllowed(src, config))
        continue;
      if (!checkMutableKernelFusible(src.getInputChain(), src.getOutputChain(),
                                     dst.getInputChain(), dst.getOutputChain()))
        continue;
      // Reshape barrier: a view cannot fuse into a load already past a reshape.
      if (llvm::any_of(llvm::enumerate(dst.getInputs()), [&](auto input) {
            return input.value() == src.getResult(0) &&
                   anyLoadCrossesReshape(body.getArgument(input.index()));
          }))
        continue;
      LoadOp load = firstConsumerLoad(dst, src);
      if (!load || !config.shouldFuse(producerStoreOf(src), load))
        continue;
      if (src.getResult(0).hasOneUse())
        return src;
      if (!duplicable)
        duplicable = src;
    }
    return duplicable;
  }

  auto dst = cast<IterOpaqueOp>(dstOp);
  DenseSet<Operation *> tried;
  // Highest operand index first, so erasing a fused input never shifts an
  // index still to be stamped into `operandIndices`.
  for (Value input : llvm::reverse(dst.getInputs())) {
    auto src = input.getDefiningOp<IterElementwiseOp>();
    if (!src || src.getNumResults() != 1 || !tried.insert(src).second)
      continue;
    if (src->getBlock() != dstOp->getBlock())
      continue;
    if (!sameDevice(src, dst) || !duplicationAllowed(src, config))
      continue;
    bool ok = true;
    LoadOp firstLoad = nullptr;
    for (auto [i, in] : llvm::enumerate(dst.getInputs())) {
      if (in != src.getResult(0))
        continue;
      ClosureCapture capture = findExistingPrologueCapture(dst, i);
      if (!capture.closure) {
        ok = false;
        break;
      }
      BlockArgument capArg =
          capture.closure.getCaptureBlockArgs()[capture.captureIdx];
      auto load = cast<LoadOp>(*capArg.getUsers().begin());
      if (loadIndexCrossesReshape(load)) {
        ok = false;
        break;
      }
      if (!firstLoad)
        firstLoad = load;
    }
    if (ok && firstLoad && config.shouldFuse(producerStoreOf(src), firstLoad))
      return src;
  }
  return nullptr;
}

} // namespace

void spliceProducer(IterOpInterface dst, IterElementwiseOp src,
                    mlir::RewriterBase &rewriter) {
  assert(isa<StructuredIterOpInterface>(dst.getOperation()) &&
         "an opaque kernel reads its inputs through closures, not loads");
  Region &dstRegion = dst->getRegion(0);
  Block &dstBody = dstRegion.front();
  unsigned numSrcInputs = src.getInputs().size();

  SmallVector<unsigned> fusionIndices;
  for (auto [i, input] : llvm::enumerate(dst.getInputs()))
    if (input == src.getResult(0))
      fusionIndices.push_back(i);
  assert(!fusionIndices.empty() && "src must be a current input of dst");

  for (unsigned argIdx : llvm::reverse(fusionIndices)) {
    dst.insertInputsAt(argIdx, src.getInputs());
    ValueRange newDstArgs = dstBody.getArguments().slice(argIdx, numSrcInputs);
    BlockArgument oldTensorArg = dstBody.getArgument(argIdx + numSrcInputs);

    // One load per index, so `src` is cloned once per value it computes; a
    // clone per load doubles the body at every fan-out level.
    mergeRedundantLoads(oldTensorArg, rewriter);
    for (Operation *user : llvm::to_vector(oldTensorArg.getUsers())) {
      auto load = cast<LoadOp>(user);
      rewriter.cloneRegionBefore(src.getBody(), dstRegion, dstRegion.begin());
      Block &srcClone = dstRegion.front();

      StoreOp storeOp = getUniqueStore(srcClone);
      Value fusedVal = storeOp.getValue();
      rewriter.eraseOp(storeOp);
      srcClone.eraseArgument(numSrcInputs);

      // Before `load`, so dst's own transform chain dominates.
      SmallVector<Value> fusedArgs(newDstArgs.begin(), newDstArgs.end());
      fusedArgs.push_back(load.getIndex());
      rewriter.inlineBlockBefore(&srcClone, load, fusedArgs);
      rewriter.replaceOp(load, fusedVal);
    }

    dst.eraseInputsAt(argIdx + numSrcInputs, 1);
  }
}

unsigned fuseInput(MO::GraphOp graphOp, const InputFusionConfig &config) {
  PatternRewriter rewriter(graphOp.getContext());

  // Kernels live in the graph body and in any nested control-flow regions (an
  // `mo.if`'s branches); a producer/consumer edge is always within one block.
  llvm::SmallSetVector<Block *, 8> blocks;
  graphOp.getBody().walk([&](Operation *op) {
    if (isa<IterElementwiseOp, IterOpaqueOp>(op))
      blocks.insert(op->getBlock());
  });

  // Consumers bottom-up, so each consumer's producer chain drains in one visit.
  MO::ParametricTopologyAnalysis topo(graphOp);
  DenseSet<Operation *> erased;
  unsigned numFused = 0;
  for (Block *block : blocks) {
    SmallVector<Operation *> ops;
    for (Operation &op : *block)
      ops.push_back(&op);
    topo.computeTopologicalSorting(ops);

    for (Operation *op : llvm::reverse(ops)) {
      if (erased.contains(op) || !isa<IterElementwiseOp, IterOpaqueOp>(op))
        continue;
      uint64_t fusedIntoConsumer = 0;
      while (IterElementwiseOp src = findEligibleProducer(op, config)) {
        // The cap bounds duplication; moving a sole-use producer adds no work.
        bool duplicates = !src.getResult(0).hasOneUse();
        if (duplicates && fusedIntoConsumer >= config.maxFusionSetSize)
          break;
        if (isa<IterElementwiseOp>(op))
          spliceIntoForeach(cast<IterElementwiseOp>(op), src, rewriter);
        else
          spliceIntoClosure(cast<IterOpaqueOp>(op), src, rewriter);
        if (src.getResult(0).use_empty()) {
          erased.insert(src);
          src->erase();
        }
        // Uniquing inputs re-exposes a shared producer as single-use, so this
        // drain fuses it once instead of duplicating or dropping it.
        cast<IterOpInterface>(op).dedupeInputs(rewriter);
        fusedIntoConsumer += duplicates;
        ++numFused;
      }
    }
  }
  return numFused;
}

namespace {

// Unconstrained input fusion (fuse every eligible edge). For testing.
struct FuseInputPass : public MAP::impl::FuseInputBase<FuseInputPass> {
  using Base::Base;

  void runOnOperation() override {
    InputFusionConfig config;
    config.shouldFuse = [](StoreOp, LoadOp) { return true; };
    config.canDuplicate = [](IterElementwiseOp) { return true; };
    fuseInput(getOperation(), config);
  }
};

} // namespace

} // namespace M::MAP
