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

#include "GraphCompiler/MAPDialect/Support/OutputFusion.h"

#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Support/DataflowUtils.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEOUTPUT
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

bool consumerReadsUntransformed(LoadOp load) {
  auto foreachOp = load->getParentOfType<IterElementwiseOp>();
  if (!foreachOp)
    return false;
  return load.getIndex() == foreachOp.getBody().front().getArguments().back();
}

bool consumerReadsThroughReshape(LoadOp load) {
  auto foreachOp = load->getParentOfType<IterElementwiseOp>();
  if (!foreachOp)
    return false;
  Value loopIdx = foreachOp.getBody().front().getArguments().back();
  auto reshape = load.getIndex().getDefiningOp<ReshapeIndexOp>();
  return reshape && reshape.getIndex() == loopIdx;
}

bool consumerIsPureReshape(LoadOp load) {
  if (!consumerReadsThroughReshape(load))
    return false;
  auto foreachOp = load->getParentOfType<IterElementwiseOp>();
  return llvm::all_of(
      getOutputStores(foreachOp.getBody().front()), [&](StoreOp store) {
        auto storedLoad = store.getValue().getDefiningOp<LoadOp>();
        return storedLoad && storedLoad.getTensor() == load.getTensor();
      });
}

namespace {

// One output-fusible edge: the consumer `src` reads `dst`'s `resultIdx` result
// (guarded by `epilogueOp`), either untransformed (`reshape` null) or through
// the single `reshape` of the loop index.
struct OutputFusionEdge {
  IterOpaqueOp dst;
  unsigned resultIdx;
  IterEpilogueOp epilogueOp;
  IterElementwiseOp src;
  ReshapeIndexOp reshape;
};

// Whether every use of `fusionArg` reads at the loop index.
bool readsAtLoopIndex(BlockArgument fusionArg, Value loopIdx) {
  for (Operation *user : fusionArg.getUsers()) {
    auto load = dyn_cast<LoadOp>(user);
    if (!load || load.getIndex() != loopIdx)
      return false;
  }
  return true;
}

// The single `map.index.reshape` of the loop index that every use of
// `fusionArg` loads through, or null if a use reads directly, through another
// transform, or through a different reshape.
ReshapeIndexOp reshapeOnFusedInput(Block &srcBody, BlockArgument fusionArg) {
  BlockArgument loopIdx = srcBody.getArguments().back();
  ReshapeIndexOp found = nullptr;
  for (Operation *user : fusionArg.getUsers()) {
    auto load = dyn_cast<LoadOp>(user);
    if (!load)
      return nullptr;
    auto reshape = load.getIndex().getDefiningOp<ReshapeIndexOp>();
    if (!reshape || reshape.getIndex() != loopIdx)
      return nullptr;
    if (found && found != reshape)
      return nullptr;
    found = reshape;
  }
  return found;
}

// The block arg of the single `src` input that reads `dstResult`.
BlockArgument fusedInputArg(IterElementwiseOp src, Value dstResult) {
  auto it = llvm::find(src.getInputs(), dstResult);
  return src.getBody().front().getArgument(it - src.getInputs().begin());
}

// The first load of `fusionArg` (all its uses are loads by the checks above).
LoadOp firstLoadOf(BlockArgument fusionArg) {
  for (Operation *user : fusionArg.getUsers())
    if (auto load = dyn_cast<LoadOp>(user))
      return load;
  return nullptr;
}

// The first output-fusible edge on `dst` that `config` accepts. Read-only.
std::optional<OutputFusionEdge>
findFusibleEdge(IterOpaqueOp dst, const OutputFusionConfig &config) {
  for (unsigned resultIdx = 0; resultIdx < dst.getNumNonChainResults();
       ++resultIdx) {
    Value res = dst->getResult(resultIdx);
    if (!res.hasOneUse())
      continue;
    IterEpilogueOp epilogueOp = dst.getEpilogueForResult(resultIdx);
    if (!epilogueOp)
      continue;
    Operation *user = res.getUses().begin()->getOwner();
    // Fusion stays within a block: never across a control-flow boundary.
    if (user->getBlock() != dst->getBlock())
      continue;
    auto src = dyn_cast<IterElementwiseOp>(user);
    if (!src)
      continue;
    // An epilogue already storing more than once serves several destinations
    // from one producer -- a consumer pair `FuseHorizontal` merged in, or a
    // result it kept for a reader of its own. Fusing rebuilds a single store,
    // so it cannot express that, whatever shape the consumer itself has.
    // TODO(GEX-4223): fuse into a multi-store epilogue by rebuilding only the
    // store the consumer reads.
    if (getOutputStores(epilogueOp.getBody().front()).size() != 1)
      continue;
    // A multi-output consumer needs a destination per output, which means
    // growing `dst`. Only all-destination-passing kernels qualify: a by-value
    // or chain result on either side carries ordering and yield constraints
    // this path does not model (`IterOpaqueOp::verify`).
    // TODO(GEX-4223): support multi-output consumers with by-value or chain
    // results.
    if (src.getNumResults() != 1) {
      if (llvm::any_of(src->getResultTypes(),
                       [](Type t) { return !isa<MO::TensorType>(t); }) ||
          llvm::any_of(dst->getResultTypes(),
                       [](Type t) { return !isa<MO::TensorType>(t); }) ||
          getOutputStores(src.getBody().front()).size() != src.getNumResults())
        continue;
    }
    if (llvm::count(src.getInputs(), res) != 1)
      continue;

    // A chain-returning consumer is `mo.mutable.store`, so the epilogue's
    // store is pointed at its buffer instead of `dst`'s own output.
    if (isa<MO::ChainType>(src.getResult(0).getType()) &&
        !isFusibleMutableStore(src, dst))
      continue;

    Block &srcBody = src.getBody().front();
    BlockArgument fusionArg = fusedInputArg(src, res);
    BlockArgument loopIdx = srcBody.getArguments().back();

    // Only a single reshape can be inverted onto the store (it preserves the
    // row-major offset).
    ReshapeIndexOp reshape = nullptr;
    if (!readsAtLoopIndex(fusionArg, loopIdx)) {
      reshape = reshapeOnFusedInput(srcBody, fusionArg);
      if (!reshape)
        continue;
    }

    // A `map.index.slice.select` in `src` moves into the epilogue whole (see
    // `fuseOutputOnce`), so it must hold at the epilogue's store index, which
    // a reshaped read would transform. The first store, rebuilt on `dst`'s own
    // result, must be a whole one.
    auto epilogueStore = getUniqueStore(epilogueOp.getBody().front());
    if (llvm::any_of(srcBody.getOps<SliceSelectOp>(),
                     [&](SliceSelectOp select) {
                       return reshape ||
                              failed(SliceSelectOp::verifySpecAt(
                                  select.getSpec(), epilogueStore.getIndex()));
                     }) ||
        getOutputStores(srcBody).front()->getBlock() != &srcBody)
      continue;

    // Reject if another operand transitively depends on `dst`: fusing would
    // create a cycle.
    if (llvm::any_of(src->getOperands(), [&](Value other) {
          return other != res && dataflowGraphDependsOn(other, dst);
        }))
      continue;

    if (config.shouldFuse(epilogueStore, firstLoadOf(fusionArg)))
      return OutputFusionEdge{dst, resultIdx, epilogueOp, src, reshape};
  }
  return std::nullopt;
}

} // namespace

IterOpaqueOp growOutputs(IterOpaqueOp dst, TypeRange extraTypes,
                         mlir::RewriterBase &rewriter) {
  if (extraTypes.empty())
    return dst;

  unsigned numDps = dst.getNumNonChainResults();
  SmallVector<Type> resultTypes(dst->getResultTypes());
  resultTypes.insert(resultTypes.begin() + numDps, extraTypes.begin(),
                     extraTypes.end());

  rewriter.setInsertionPoint(dst);
  auto newDst = IterOpaqueOp::create(
      rewriter, dst.getLoc(), resultTypes, dst.getInputs(), dst.getInputChain(),
      dst.getFusionInfoAttr(), dst.getHasReadEffect(), dst.getHasWriteEffect(),
      dst.getDispatch());
  // `device` and friends are discardable, so `create` does not carry them.
  newDst->setDiscardableAttrs(dst->getDiscardableAttrDictionary());
  newDst.getBody().takeBody(dst.getBody());

  Block &body = newDst.getBody().front();
  for (Type ty : extraTypes)
    body.addArgument(ty, dst.getLoc());

  for (auto [i, res] : llvm::enumerate(dst.getResults()))
    res.replaceAllUsesWith(
        newDst->getResult(i < numDps ? i : i + extraTypes.size()));
  rewriter.eraseOp(dst);
  return newDst;
}

namespace {

// Splices `src`'s per-element compute into `epilogueOp`. For a reshape read,
// the reshape is relocated onto the store index with in/out swapped, since the
// store direction is the inverse of the load.
IterOpaqueOp fuseOutputOnce(const OutputFusionEdge &edge,
                            PatternRewriter &rewriter) {
  IterOpaqueOp dst = edge.dst;
  IterEpilogueOp epilogueOp = edge.epilogueOp;
  IterElementwiseOp src = edge.src;
  ReshapeIndexOp reshape = edge.reshape;
  MLIRContext *ctx = dst.getContext();

  // `src`'s first output reuses the result `dst` already produces; the rest
  // need destinations of their own.
  SmallVector<Type> extraTypes(llvm::drop_begin(src->getResultTypes()));
  unsigned numDpsBefore = dst.getNumNonChainResults();
  dst = growOutputs(dst, extraTypes, rewriter);
  Value dstResult = dst->getResult(edge.resultIdx);
  Block &srcBody = src.getBody().front();
  Block &epilogueBody = epilogueOp.getBody().front();

  // May already reflect earlier fusions (e.g. a reshaped store index).
  auto epilogueStore = getUniqueStore(epilogueBody);
  Value val = epilogueStore.getValue();
  Value storeIdx = epilogueStore.getIndex();
  Value cap = epilogueStore.getTensor();
  Location storeLoc = epilogueStore.getLoc();

  Value storeTransform = storeIdx;
  if (reshape) {
    ViewSignatureAttr origSig = reshape.getSpec().getSignature();
    auto swappedSpec = ReshapeSpecAttr::get(
        ctx, ViewSignatureAttr::get(ctx, origSig.getOut(), origSig.getIn()));
    rewriter.setInsertionPoint(epilogueStore);
    storeTransform = ReshapeIndexOp::create(
        rewriter, storeLoc, IndexType::get(ctx, origSig.getOut().getRank()),
        swappedSpec, storeIdx);
  }

  auto fusionArgIt = llvm::find(src.getInputs(), dstResult);
  unsigned fusionArgIdx = fusionArgIt - src.getInputs().begin();
  BlockArgument fusionArg = srcBody.getArgument(fusionArgIdx);

  // `val` is the scalar `dst` produced for this element; it replaces every
  // load of the fused operand.
  for (Operation *user : llvm::to_vector(fusionArg.getUsers())) {
    auto load = cast<LoadOp>(user);
    load.getResult().replaceAllUsesWith(val);
    load.erase();
  }
  if (reshape)
    reshape.erase();

  // Every other input becomes a fresh epilogue capture, threaded through a new
  // shadowed input of `dst` (the epilogue is isolated from above).
  for (auto [i, input] : llvm::enumerate(src.getInputs())) {
    if (i == fusionArgIdx)
      continue;
    unsigned dstInputIdx = dst.getInputs().size();
    dst.insertInputsAt(dstInputIdx, input);
    Value dstInputArg = dst.getInputBlockArgs()[dstInputIdx];
    epilogueOp.getCapturesMutable().append(dstInputArg);
    BlockArgument capArg =
        epilogueBody.addArgument(input.getType(), input.getLoc());
    srcBody.getArgument(i).replaceAllUsesWith(capArg);
  }

  // The grown output block args are the extra stores' destinations. Read them
  // through the accessor: threading `src`'s inputs above shifted the args.
  SmallVector<Value> extraCaps;
  Block::BlockArgListType outArgs = dst.getOutputBlockArgs();
  for (unsigned i = 0; i < extraTypes.size(); ++i) {
    BlockArgument outArg = outArgs[numDpsBefore + i];
    epilogueOp.getCapturesMutable().append(outArg);
    extraCaps.push_back(
        epilogueBody.addArgument(outArg.getType(), outArg.getLoc()));
  }

  // The loop index is the body's last argument for both pointwise and
  // `mo.mutable.store` consumers; rewire it to the store index.
  SmallVector<StoreOp> srcStores = getOutputStores(srcBody);
  StoreOp srcStore = srcStores.front();
  Value finalVal = srcStore.getValue();
  // A store in a select keeps its place and its index, the select's argument;
  // only its destination moves to the new capture. The whole stores are
  // rebuilt at the store index below.
  SmallVector<std::pair<Value, Value>> extraStores;
  for (auto [store, extraCap] :
       llvm::zip_equal(llvm::drop_begin(srcStores), extraCaps)) {
    if (store->getBlock() != &srcBody)
      store.getTensorMutable().assign(extraCap);
    else
      extraStores.push_back({extraCap, store.getValue()});
  }
  // A `mo.mutable.store` consumer's buffer input is already an epilogue
  // capture; the store is rebuilt on it.
  Value storeTargetBuffer;
  if (isa<MO::ChainType>(src.getResult(0).getType()))
    storeTargetBuffer = srcStore.getTensor();
  for (StoreOp store : srcStores)
    if (store->getBlock() == &srcBody)
      store.erase();
  srcBody.getArguments().back().replaceAllUsesWith(storeTransform);
  srcBody.eraseArguments(0, srcBody.getNumArguments());

  // Retype the store-target path when the consumer changes the result's type.
  // For a reshape read the epilogue's RESULT keeps the producer's shape, which
  // the kernel (e.g. matmul) binds; the reshaped index reaches the destination.
  if (storeTargetBuffer) {
    cap = storeTargetBuffer;
    dst.setHasWriteEffect(true);
  }

  Type fusedTy = src.getResult(0).getType();
  if (!storeTargetBuffer && fusedTy != dstResult.getType()) {
    dstResult.setType(fusedTy);
    dst.getOutputBlockArgs()[edge.resultIdx].setType(fusedTy);
    cap.setType(fusedTy);
    if (!reshape)
      epilogueOp.getResult().setType(fusedTy);
  }

  epilogueStore.erase();
  rewriter.inlineBlockBefore(&srcBody, &epilogueBody, epilogueBody.end());
  rewriter.setInsertionPointToEnd(&epilogueBody);
  StoreOp::create(rewriter, storeLoc, cap, storeTransform, finalVal);
  for (auto [extraCap, extraVal] : extraStores)
    StoreOp::create(rewriter, storeLoc, extraCap, storeTransform, extraVal);

  // A `mo.mutable.store` consumer yields only an unused chain.
  if (!storeTargetBuffer) {
    SmallVector<Value> replacements{dstResult};
    for (unsigned i = 0; i < extraTypes.size(); ++i)
      replacements.push_back(dst->getResult(numDpsBefore + i));
    src->replaceAllUsesWith(replacements);
  }
  return dst;
}

} // namespace

bool isFusibleMutableStore(IterElementwiseOp consumer, Operation *dst) {
  if (consumer.getNumResults() != 1 ||
      !isa<MO::ChainType>(consumer.getResult(0).getType()))
    return false;
  // A fused kernel must not write a buffer it also reads.
  if (llvm::any_of(consumer->getOperands(), [&](Value operand) {
        return isa<MO::BufferType>(operand.getType()) &&
               llvm::is_contained(dst->getOperands(), operand);
      }))
    return false;
  // `dst` holds no chain, so the store's chains must carry no ordering: its
  // input chain is fresh and nothing uses its output chain.
  // TODO(GEX-3964): give `dst` a chain in/out the way
  // `FuseMutableLoads::fuseLoadBufferAndChains` does, and fuse either way.
  Value inputChain = consumer.getInputChain();
  return (!inputChain ||
          isa_and_nonnull<MO::ChainCreateOp>(inputChain.getDefiningOp())) &&
         consumer.getResult(0).use_empty();
}

unsigned fuseOutput(MO::GraphOp graphOp, const OutputFusionConfig &config) {
  PatternRewriter rewriter(graphOp.getContext());

  // Kernels live in the graph body and in any nested control-flow regions (an
  // `mo.if`'s branches); a producer/consumer edge is always within one block.
  llvm::SmallSetVector<Block *, 8> blocks;
  graphOp.getBody().walk(
      [&](IterOpaqueOp op) { blocks.insert(op->getBlock()); });

  // Producers before consumers. Every block is sorted before anything fuses:
  // fusion erases the consumer and can rebuild the producer (`growOutputs`),
  // while the analysis holds a raw pointer to any op a nested block reads.
  MO::ParametricTopologyAnalysis topo(graphOp);
  SmallVector<SmallVector<Operation *>> sortedBlocks;
  for (Block *block : blocks) {
    SmallVector<Operation *> &ops = sortedBlocks.emplace_back();
    for (Operation &op : *block)
      ops.push_back(&op);
    topo.computeTopologicalSorting(ops);
  }

  DenseSet<Operation *> erased;
  unsigned numFused = 0;
  for (ArrayRef<Operation *> ops : sortedBlocks) {
    for (Operation *op : ops) {
      if (erased.contains(op))
        continue;
      auto dst = dyn_cast<IterOpaqueOp>(op);
      if (!dst)
        continue;
      while (std::optional<OutputFusionEdge> edge =
                 findFusibleEdge(dst, config)) {
        dst = fuseOutputOnce(*edge, rewriter);
        dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionEpilogueAttr>(
            dst.getContext(), {dst.getOperation(), edge->src.getOperation()},
            {static_cast<uint32_t>(edge->resultIdx)}));
        dst->setLoc(mlir::FusedLoc::get(dst.getContext(),
                                        {edge->src->getLoc(), dst->getLoc()}));
        erased.insert(edge->src);
        edge->src->erase();
        IterOpInterface(dst).dedupeInputs(rewriter);
        ++numFused;
      }
    }
  }
  return numFused;
}

namespace {

// Unconstrained output fusion (fuse every eligible edge). For testing.
struct FuseOutputPass : public MAP::impl::FuseOutputBase<FuseOutputPass> {
  using Base::Base;

  void runOnOperation() override {
    OutputFusionConfig config;
    config.shouldFuse = [](StoreOp, LoadOp) { return true; };
    fuseOutput(getOperation(), config);
  }
};

} // namespace

} // namespace M::MAP
