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
//
// Fuses kernels into `map.iter.rowwise` reductions:
//
// - A producer only the rowwise reads is cloned at each load of its result.
// - A consumer over the per-row shape that reads a per-row result at its own
//   index becomes row-level ops; its result becomes a rowwise output, or, for
//   a `mo.mutable.store`, its store writes the buffer.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/InputFusion.h"
#include "GraphCompiler/MAPDialect/Support/OutputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEREDUCTIONS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

bool isPlainForeach(IterElementwiseOp op) {
  return op.getNumResults() == 1 && !op.getInputChain() && !op.getOutputChain();
}

bool sameDevice(IterElementwiseOp foreachOp, IterRowwiseOp rowwiseOp) {
  return MO::getDeviceOrDefault(
             cast<MO::MOHasDeviceInterface>(foreachOp.getOperation())) ==
         MO::getDeviceOrDefault(
             cast<MO::MOHasDeviceInterface>(rowwiseOp.getOperation()));
}

// Records `src` as fused into `dst`, in dataflow order: a producer before the
// rowwise, a consumer after it.
void stampFusion(IterRowwiseOp dst, IterElementwiseOp src, bool isProducer) {
  MLIRContext *ctx = dst->getContext();
  SmallVector<Operation *, 2> ops = {src, dst};
  if (!isProducer)
    std::swap(ops[0], ops[1]);
  dst.setFusionInfoAttr(
      MOGG::createFusionInfo<MOGG::FusionRowwiseAttr>(ctx, ops));
  dst->setLoc(FusedLoc::get(ctx, {ops[0]->getLoc(), ops[1]->getLoc()}));
  dst.setHasReadEffect(src.getHasReadEffect() || dst.getHasReadEffect());
  dst.setHasWriteEffect(src.getHasWriteEffect() || dst.getHasWriteEffect());
}

// TODO(GEX-3348): a producer other kernels also read stays materialized;
// duplicate it once shared reduction inputs are common.
bool canFuseProducer(IterRowwiseOp dst, IterElementwiseOp src) {
  if (!isPlainForeach(src) || src->getBlock() != dst->getBlock() ||
      !sameDevice(src, dst))
    return false;
  Value result = src.getResult(0);
  if (!llvm::all_of(result.getUsers(),
                    [&](Operation *user) { return user == dst; }))
    return false;
  // Reshape barrier, as in input fusion: a view cannot fuse into a load
  // already past a reshape.
  for (auto [input, arg] : llvm::zip(dst.getInputs(), dst.getInputBlockArgs()))
    if (input == result &&
        (!llvm::all_of(arg.getUsers(), llvm::IsaPred<LoadOp>) ||
         anyLoadCrossesReshape(arg)))
      return false;
  return true;
}

// Leaves `dst` with unique inputs, as `fuseConsumer` does.
void fuseProducer(IterRowwiseOp dst, IterElementwiseOp src,
                  RewriterBase &rewriter) {
  spliceProducer(dst, src, rewriter);
  stampFusion(dst, src, /*isProducer=*/true);
  IterOpInterface(dst).dedupeInputs(rewriter);
}

// The only store to output `outputIdx`, if it writes at the row index.
StoreOp perRowStore(IterRowwiseOp dst, unsigned outputIdx) {
  BlockArgument outputArg = dst.getOutputBlockArgs()[outputIdx];
  if (!outputArg.hasOneUse())
    return nullptr;
  auto store = dyn_cast<StoreOp>(*outputArg.getUsers().begin());
  if (!store || store->getParentOp() != dst ||
      store.getIndex() != dst.getRowArg())
    return nullptr;
  return store;
}

// Whether `shape` is `dst`'s shape with the reduced axis set to 1.
bool isPerRowShape(IterRowwiseOp dst, MOSH::ShapeAttr shape) {
  MOSH::ShapeAttr full = dst.getShape();
  int64_t axis = dst.getAxis();
  if (shape.getRank() != full.getRank() ||
      MOSH::ShapeAttr::getStaticDim(shape[axis]) != 1)
    return false;
  for (int64_t d = 0, e = full.getRank(); d < e; ++d)
    if (d != axis && shape[d] != full[d])
      return false;
  return true;
}

// Whether nothing between `dst` and `consumer` reads `dst`. Also rules out
// a cycle through another kernel.
bool canSinkTo(IterRowwiseOp dst, Operation *consumer) {
  for (Value result : dst.getResults())
    for (Operation *user : result.getUsers()) {
      Operation *top = dst->getBlock()->findAncestorOpInBlock(*user);
      if (top && top != consumer && top->isBeforeInBlock(consumer))
        return false;
    }
  return true;
}

// Whether one of `consumer`'s other inputs is defined after `dst`.
bool needsSink(IterRowwiseOp dst, IterElementwiseOp consumer) {
  return llvm::any_of(consumer.getInputs(), [&](Value input) {
    Operation *def = input.getDefiningOp();
    return def && def != dst && def->getBlock() == dst->getBlock() &&
           dst->isBeforeInBlock(def);
  });
}

bool canFuseConsumer(IterRowwiseOp dst, IterElementwiseOp consumer) {
  if (!(isPlainForeach(consumer) || isFusibleMutableStore(consumer, dst)) ||
      dst.getInputChain() || consumer->getBlock() != dst->getBlock() ||
      !sameDevice(consumer, dst) || !isPerRowShape(dst, consumer.getShape()))
    return false;

  Block &body = consumer.getBody().front();
  BlockArgument loopIdx = body.getArguments().back();
  for (auto [i, input] : llvm::enumerate(consumer.getInputs())) {
    auto result = dyn_cast<OpResult>(input);
    if (!result || result.getOwner() != dst)
      continue;
    if (!perRowStore(dst, result.getResultNumber()))
      return false;
    for (Operation *user : body.getArgument(i).getUsers()) {
      auto load = dyn_cast<LoadOp>(user);
      if (!load || load.getIndex() != loopIdx)
        return false;
    }
  }
  return !needsSink(dst, consumer) || canSinkTo(dst, consumer);
}

// Rebuilds `op` with the outputs in `kept` plus one per `extraTypes`. Dropped
// outputs must be unused.
IterRowwiseOp rebuildResults(IterRowwiseOp op, ArrayRef<unsigned> kept,
                             TypeRange extraTypes, RewriterBase &rewriter) {
  Value oldChain = op.getOutputChain();
  SmallVector<Type> types;
  for (unsigned idx : kept)
    types.push_back(op.getResult(idx).getType());
  llvm::append_range(types, extraTypes);
  if (oldChain)
    types.push_back(oldChain.getType());

  Block &body = op.getBody().front();
  unsigned numInputs = op.getInputs().size();
  unsigned numOutputs = op.getOutputBlockArgs().size();
  BitVector dropArgs(body.getNumArguments());
  for (unsigned idx = 0; idx < numOutputs; ++idx)
    if (!llvm::is_contained(kept, idx))
      dropArgs.set(numInputs + idx);
  body.eraseArguments(dropArgs);
  unsigned rowIdx = body.getNumArguments() - 1;
  for (Type type : extraTypes)
    body.insertArgument(rowIdx++, type, op.getLoc());

  rewriter.setInsertionPoint(op);
  auto newOp = IterRowwiseOp::create(
      rewriter, op.getLoc(), types, op->getOperands(), op.getProperties(),
      op->getDiscardableAttrDictionary().getValue());
  newOp.getBody().takeBody(op.getBody());
  for (auto [newIdx, oldIdx] : llvm::enumerate(kept))
    rewriter.replaceAllUsesWith(op.getResult(oldIdx), newOp.getResult(newIdx));
  if (oldChain)
    rewriter.replaceAllUsesWith(oldChain, newOp.getOutputChain());
  rewriter.eraseOp(op);
  return newOp;
}

IterRowwiseOp fuseConsumer(IterRowwiseOp dst, IterElementwiseOp consumer,
                           RewriterBase &rewriter) {
  if (needsSink(dst, consumer))
    dst->moveBefore(consumer);

  Block &consumerBody = consumer.getBody().front();
  IRMapping mapping;

  // Loads of a per-row result read the stored value instead.
  DenseMap<BlockArgument, Value> storedValues;
  SmallVector<unsigned> readResults;
  for (auto [i, input] : llvm::enumerate(consumer.getInputs())) {
    auto result = dyn_cast<OpResult>(input);
    BlockArgument arg = consumerBody.getArgument(i);
    if (result && result.getOwner() == dst) {
      storedValues[arg] = perRowStore(dst, result.getResultNumber()).getValue();
      readResults.push_back(result.getResultNumber());
      continue;
    }
    unsigned idx = dst.getInputs().size();
    dst.insertInputsAt(idx, input);
    mapping.map(arg, dst.getInputBlockArgs()[idx]);
  }

  // A `mo.mutable.store` stores into its buffer input, now one of `dst`'s, so
  // it adds no output; its chain has no use.
  Value consumerResult = consumer.getResult(0);
  bool storesToBuffer = isa<MO::ChainType>(consumerResult.getType());
  unsigned numOutputs = dst.getOutputBlockArgs().size();
  auto all = llvm::to_vector(llvm::seq<unsigned>(0, numOutputs));
  IterRowwiseOp newDst = rebuildResults(
      dst, all,
      storesToBuffer ? TypeRange{} : TypeRange{consumerResult.getType()},
      rewriter);
  mapping.map(consumerBody.getArguments().back(), newDst.getRowArg());
  if (!storesToBuffer)
    mapping.map(consumerBody.getArgument(consumer.getInputs().size()),
                newDst.getOutputBlockArgs().back());

  rewriter.setInsertionPointToEnd(&newDst.getBody().front());
  for (Operation &op : consumerBody) {
    if (auto load = dyn_cast<LoadOp>(op)) {
      auto arg = dyn_cast<BlockArgument>(load.getTensor());
      if (arg && storedValues.count(arg)) {
        mapping.map(load.getResult(), storedValues.lookup(arg));
        continue;
      }
    }
    rewriter.clone(op, mapping);
  }

  stampFusion(newDst, consumer, /*isProducer=*/false);
  if (storesToBuffer)
    newDst.setHasWriteEffect(true);
  else
    rewriter.replaceAllUsesWith(consumerResult, newDst.getResult(numOutputs));
  rewriter.eraseOp(consumer);
  IterOpInterface(newDst).dedupeInputs(rewriter);

  // Drop outputs only the consumer read.
  SmallVector<unsigned> kept;
  for (unsigned idx = 0, e = newDst.getOutputBlockArgs().size(); idx < e;
       ++idx) {
    if (!llvm::is_contained(readResults, idx) ||
        !newDst.getResult(idx).use_empty()) {
      kept.push_back(idx);
      continue;
    }
    rewriter.eraseOp(perRowStore(newDst, idx));
  }
  if (kept.size() == newDst.getOutputBlockArgs().size())
    return newDst;
  return rebuildResults(newDst, kept, TypeRange{}, rewriter);
}

IterElementwiseOp findProducer(IterRowwiseOp dst) {
  for (Value input : dst.getInputs())
    if (auto src = input.getDefiningOp<IterElementwiseOp>())
      if (canFuseProducer(dst, src))
        return src;
  return nullptr;
}

IterElementwiseOp findConsumer(IterRowwiseOp dst) {
  for (Value result : dst.getResults())
    for (Operation *user : result.getUsers())
      if (auto consumer = dyn_cast<IterElementwiseOp>(user))
        if (canFuseConsumer(dst, consumer))
          return consumer;
  return nullptr;
}

struct FuseReductionsPass
    : public MAP::impl::FuseReductionsBase<FuseReductionsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    IRRewriter rewriter(graphOp.getContext());
    SmallVector<IterRowwiseOp> rowwiseOps;
    graphOp.getBody().walk([&](IterRowwiseOp op) {
      rowwiseOps.push_back(op);
      return WalkResult::skip();
    });

    for (IterRowwiseOp dst : rowwiseOps) {
      bool changed = true;
      while (changed) {
        changed = false;
        while (IterElementwiseOp src = findProducer(dst)) {
          fuseProducer(dst, src, rewriter);
          rewriter.eraseOp(src);
          ++numKernelsFused;
          changed = true;
        }
        if (IterElementwiseOp consumer = findConsumer(dst)) {
          dst = fuseConsumer(dst, consumer, rewriter);
          ++numKernelsFused;
          changed = true;
        }
      }
    }
  }
};

} // namespace

} // namespace M::MAP
