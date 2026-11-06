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
// Fuses the kernels around a `map.iter.rowwise` into it, so a reduction and
// the elementwise work before and after it run as one kernel. Three fusers:
//
// A producer only the rowwise reads is cloned where the rowwise loads it:
//
//   %y = elementwise (%i) {                map.iter.rowwise inputs (%x) {
//     store %y[%i] = mul(%x[%i], %x[%i])     map.iter.reduce index (%i) {
//   }                                  =>      %v = mul(%x[%i], %x[%i])
//   map.iter.rowwise inputs (%y) {             ...
//     map.iter.reduce index (%i) {
//       %v = load %y[%i]
//       ...
//
// A consumer over the per-row shape that reads a per-row result at its own
// index becomes row-level ops. Its result becomes a rowwise output, or, for
// a `mo.mutable.store`, its store writes the buffer:
//
//   %s = map.iter.rowwise {                %r = map.iter.rowwise {
//     %acc = map.iter.reduce ...             %acc = map.iter.reduce ...
//     store %s[%row] = %acc        =>        %v = add(%acc, %b[%row])
//   }                                        store %r[%row] = %v
//   %r = elementwise (%i) {                }
//     store %r[%i] = add(%s[%i], %b[%i])
//   }
//
// A consumer over the rowwise shape that reads per-row results through a
// broadcast joins a new map phase. One that also reads the last map phase's
// result at its own element joins that phase instead, as an epilogue.
// `map-rlicm` then moves their row-invariant work to row level:
//
//   %s = map.iter.rowwise {                %y = map.iter.rowwise {
//     %acc = map.iter.reduce ...             %acc = map.iter.reduce ...
//     store %s[%row] = %acc        =>        map.iter.map index (%j) {
//   }                                          %v = mul(%x[%j], %acc)
//   %y = elementwise (%i) {                    store %y[%j] = %v
//     %b = load %s[bcast(%i)]                }
//     store %y[%i] = mul(%x[%i], %b)       }
//   }
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

// The map phase that ends `dst`'s body, if any.
IterMapOp lastMapPhase(IterRowwiseOp dst) {
  Block &body = dst.getBody().front();
  return body.empty() ? IterMapOp() : dyn_cast<IterMapOp>(body.back());
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
  // The Row API can't write per-row stores after a map phase: `Row.emit`
  // isn't phase-aware.
  if (!(isPlainForeach(consumer) || isFusibleMutableStore(consumer, dst)) ||
      lastMapPhase(dst) || dst.getInputChain() ||
      consumer->getBlock() != dst->getBlock() || !sameDevice(consumer, dst) ||
      !isPerRowShape(dst, consumer.getShape()))
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

// Whether a load whose index is loop dimensions `dims` reads the element at
// the loop index.
bool readsSameElement(ArrayRef<int64_t> dims, int64_t rank) {
  return static_cast<int64_t>(dims.size()) == rank &&
         llvm::all_of(llvm::enumerate(dims),
                      [](auto d) { return d.value() == int64_t(d.index()); });
}

// Whether a load of a per-row tensor whose index is loop dimensions `dims`
// reads the loop's row.
bool readsOwnRow(ArrayRef<int64_t> dims, int64_t rank, int64_t axis) {
  if (static_cast<int64_t>(dims.size()) != rank)
    return false;
  for (auto [k, d] : llvm::enumerate(dims))
    if (d != (int64_t(k) == axis ? kPinnedCoordinate : int64_t(k)))
      return false;
  return true;
}

// The only store to output `outputIdx`, if it writes the last map phase's
// element.
StoreOp perElementStore(IterRowwiseOp dst, unsigned outputIdx) {
  BlockArgument outputArg = dst.getOutputBlockArgs()[outputIdx];
  IterMapOp phase = lastMapPhase(dst);
  if (!phase || !outputArg.hasOneUse())
    return nullptr;
  auto store = dyn_cast<StoreOp>(*outputArg.getUsers().begin());
  if (!store || store->getParentOp() != phase ||
      store.getIndex() != phase.getIndexArg())
    return nullptr;
  return store;
}

// The value `dst` stores to output `outputIdx`, per row or per element.
Value storedValue(IterRowwiseOp dst, unsigned outputIdx) {
  if (StoreOp store = perRowStore(dst, outputIdx))
    return store.getValue();
  if (StoreOp store = perElementStore(dst, outputIdx))
    return store.getValue();
  return nullptr;
}

// The phase a full-row `consumer` joins: the last map phase when every row
// value it reads is defined before it, else a new one (null).
IterMapOp joinedPhase(IterRowwiseOp dst, IterElementwiseOp consumer) {
  IterMapOp phase = lastMapPhase(dst);
  if (!phase)
    return nullptr;
  for (Value input : consumer.getInputs()) {
    auto result = dyn_cast<OpResult>(input);
    if (!result || result.getOwner() != dst)
      continue;
    Operation *def = storedValue(dst, result.getResultNumber()).getDefiningOp();
    if (def && def->getBlock() == &dst.getBody().front() &&
        !def->isBeforeInBlock(phase))
      return nullptr;
  }
  return phase;
}

// Whether `value` reaches a rowwise kernel through kernels over `shape`.
bool feedsReduction(Value value, MOSH::ShapeAttr shape) {
  return llvm::any_of(value.getUsers(), [&](Operation *user) {
    if (isa<IterRowwiseOp>(user))
      return true;
    auto kernel = dyn_cast<IterElementwiseOp>(user);
    return kernel && kernel.getShape() == shape &&
           llvm::any_of(kernel->getResults(), [&](Value result) {
             return feedsReduction(result, shape);
           });
  });
}

bool canFuseFullRowConsumer(IterRowwiseOp dst, IterElementwiseOp consumer) {
  if (!isPlainForeach(consumer) || dst.getInputChain() ||
      consumer->getBlock() != dst->getBlock() || !sameDevice(consumer, dst) ||
      consumer.getShape() != dst.getShape())
    return false;

  int64_t rank = dst.getShape().getRank(), axis = dst.getAxis();
  Block &body = consumer.getBody().front();
  BlockArgument loopIdx = body.getArguments().back();
  bool readsElement = false;
  for (auto [i, input] : llvm::enumerate(consumer.getInputs())) {
    auto result = dyn_cast<OpResult>(input);
    if (!result || result.getOwner() != dst)
      continue;
    unsigned idx = result.getResultNumber();
    bool perRow = perRowStore(dst, idx);
    if (!perRow && !perElementStore(dst, idx))
      return false;
    readsElement |= !perRow;
    for (Operation *user : body.getArgument(i).getUsers()) {
      auto load = dyn_cast<LoadOp>(user);
      std::optional<SmallVector<int64_t>> dims =
          load ? traceLoopDims(load.getIndex(), loopIdx) : std::nullopt;
      if (!dims || !(perRow ? readsOwnRow(*dims, rank, axis)
                            : readsSameElement(*dims, rank)))
        return false;
    }
  }
  if (readsElement && !joinedPhase(dst, consumer))
    return false;

  // A later reduction takes the consumer as a producer instead, so its result
  // is never written.
  if (feedsReduction(consumer.getResult(0), dst.getShape()))
    return false;

  // A per-row output can't stay next to a map phase (`Row.emit` isn't
  // phase-aware), so fuse only when every per-row output goes away.
  for (unsigned idx = 0, e = dst.getOutputBlockArgs().size(); idx < e; ++idx)
    if (perRowStore(dst, idx) &&
        llvm::any_of(dst.getResult(idx).getUsers(),
                     [&](Operation *user) { return user != consumer; }))
      return false;
  return !needsSink(dst, consumer) || canSinkTo(dst, consumer);
}

IterRowwiseOp fuseFullRowConsumer(IterRowwiseOp dst, IterElementwiseOp consumer,
                                  RewriterBase &rewriter) {
  if (needsSink(dst, consumer))
    dst->moveBefore(consumer);
  bool joinsLastPhase = static_cast<bool>(joinedPhase(dst, consumer));

  Block &consumerBody = consumer.getBody().front();
  IRMapping mapping;
  DenseMap<BlockArgument, unsigned> readOutputs;
  for (auto [i, input] : llvm::enumerate(consumer.getInputs())) {
    auto result = dyn_cast<OpResult>(input);
    BlockArgument arg = consumerBody.getArgument(i);
    if (result && result.getOwner() == dst) {
      readOutputs[arg] = result.getResultNumber();
      continue;
    }
    unsigned idx = dst.getInputs().size();
    dst.insertInputsAt(idx, input);
    mapping.map(arg, dst.getInputBlockArgs()[idx]);
  }

  Value consumerResult = consumer.getResult(0);
  unsigned numOutputs = dst.getOutputBlockArgs().size();
  auto all = llvm::to_vector(llvm::seq<unsigned>(0, numOutputs));
  IterRowwiseOp newDst =
      rebuildResults(dst, all, TypeRange{consumerResult.getType()}, rewriter);
  mapping.map(consumerBody.getArgument(consumer.getInputs().size()),
              newDst.getOutputBlockArgs().back());

  IterMapOp phase = joinsLastPhase ? lastMapPhase(newDst) : IterMapOp();
  if (!phase) {
    rewriter.setInsertionPointToEnd(&newDst.getBody().front());
    phase = IterMapOp::create(rewriter, consumer.getLoc());
    Block *block = rewriter.createBlock(&phase.getBody());
    block->addArgument(newDst.getRowArg().getType(), consumer.getLoc());
  }
  mapping.map(consumerBody.getArguments().back(), phase.getIndexArg());

  // Loads of a rowwise result read the stored value instead.
  rewriter.setInsertionPointToEnd(&phase.getBody().front());
  for (Operation &op : consumerBody) {
    if (auto load = dyn_cast<LoadOp>(op)) {
      auto arg = dyn_cast<BlockArgument>(load.getTensor());
      if (arg && readOutputs.count(arg)) {
        mapping.map(load.getResult(),
                    storedValue(newDst, readOutputs.lookup(arg)));
        continue;
      }
    }
    rewriter.clone(op, mapping);
  }

  stampFusion(newDst, consumer, /*isProducer=*/false);
  rewriter.replaceAllUsesWith(consumerResult, newDst.getResult(numOutputs));
  rewriter.eraseOp(consumer);
  IterOpInterface(newDst).dedupeInputs(rewriter);

  // Drop the outputs nothing reads anymore, so no per-row store is left next
  // to the map phase.
  SmallVector<unsigned> kept;
  for (unsigned idx = 0, e = newDst.getOutputBlockArgs().size(); idx < e;
       ++idx) {
    StoreOp store = perRowStore(newDst, idx);
    if (!store)
      store = perElementStore(newDst, idx);
    if (!store || !newDst.getResult(idx).use_empty()) {
      kept.push_back(idx);
      continue;
    }
    rewriter.eraseOp(store);
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

// A consumer over the per-row shape `dst` can absorb as row-level ops.
IterElementwiseOp findConsumer(IterRowwiseOp dst) {
  for (Value result : dst.getResults())
    for (Operation *user : result.getUsers())
      if (auto consumer = dyn_cast<IterElementwiseOp>(user))
        if (canFuseConsumer(dst, consumer))
          return consumer;
  return nullptr;
}

// A consumer over the rowwise shape `dst` can absorb in a map phase.
IterElementwiseOp findFullRowConsumer(IterRowwiseOp dst) {
  for (Value result : dst.getResults())
    for (Operation *user : result.getUsers())
      if (auto consumer = dyn_cast<IterElementwiseOp>(user))
        if (canFuseFullRowConsumer(dst, consumer))
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
        if (IterElementwiseOp consumer = findFullRowConsumer(dst)) {
          dst = fuseFullRowConsumer(dst, consumer, rewriter);
          ++numKernelsFused;
          changed = true;
        }
      }
    }
  }
};

} // namespace

} // namespace M::MAP
