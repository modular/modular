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
// the elementwise work before and after it run as one kernel. Four fusers:
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
// A consumer a later reduction reads is left to it as a producer, so it is
// never written, unless something outside that reduction also reads it, as a
// graph output does. Then it is written anyway, so it joins the reduction
// before it:
//
//   %s = map.iter.rowwise {                %h = map.iter.rowwise {
//     %a = map.iter.reduce ...               %a = map.iter.reduce ...
//     store %s[%row] = %a          =>        map.iter.map index (%j) {
//   }                                          store %h[%j] = f(%x[%j], %a)
//   %h = elementwise (%i) {                  }
//     store %h[%i] = f(%x[%i], %s[...])    }
//   }                                      %y = map.iter.rowwise inputs (%h)
//   %y = map.iter.rowwise inputs (%h)      mo.output %y, %h
//   mo.output %y, %h
//
// A rowwise kernel whose per-row results only a later rowwise kernel over the
// same rows, with a map phase, reads at its own row runs at the start of that
// kernel's row:
//
//   %m = map.iter.rowwise {                %y = map.iter.rowwise {
//     %a = map.iter.reduce ...               %a = map.iter.reduce ...
//     store %m[%row] = %a          =>        %b = map.iter.reduce index (%i) {
//   }                                          ... sub(%x[%i], %a) ...
//   %y = map.iter.rowwise {                  }
//     %b = map.iter.reduce index (%i) {      map.iter.map index (%j) { ... }
//       ... sub(%x[%i], %m[bcast(%i)]) ... }
//     }
//     map.iter.map index (%j) { ... }
//   }
//
// When the earlier kernel has a map phase whose output the later kernel reads
// at the same element, the phase becomes a `map.iter.cache` the later phases
// read. An output something else reads is stored from the cache:
//
//   %h = map.iter.rowwise {                %y, %h2 = map.iter.rowwise {
//     %a = map.iter.reduce ...               %a = map.iter.reduce ...
//     map.iter.map index (%j) {     =>       %c = map.iter.cache index (%k) {
//       store %h[%j] = f(%a, %x[%j])           %v = f(%a, %x[%k])
//     }                                        store %h2[%k] = %v
//   }                                          map.yield %v
//   %y = map.iter.rowwise {                  }
//     %b = map.iter.reduce index (%i) {      %b = map.iter.reduce index (%i) {
//       ... %h[%i] ...                         ... map.iter.cache.load %c[%i]
//     }                                      }
//     map.iter.map index (%j) {              map.iter.map index (%j) {
//       ... %h[%j] ...                         ... map.iter.cache.load %c[%j]
//     }                                      }
//   }                                      }
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/InputFusion.h"
#include "GraphCompiler/MAPDialect/Support/OutputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "Support/AssertStream.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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

bool sameDevice(Operation *a, Operation *b) {
  return MO::getDeviceOrDefault(cast<MO::MOHasDeviceInterface>(a)) ==
         MO::getDeviceOrDefault(cast<MO::MOHasDeviceInterface>(b));
}

// Records `src` as fused into `dst`, in dataflow order: a producer before the
// rowwise, a consumer after it.
void stampFusion(IterRowwiseOp dst, IterOpInterface src, bool isProducer) {
  MLIRContext *ctx = dst->getContext();
  SmallVector<Operation *, 2> ops = {src.getOperation(), dst};
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
  for (auto [input, arg] : llvm::zip(dst.getInputs(), dst.getInputBlockArgs()))
    if (input == result &&
        (!llvm::all_of(arg.getUsers(), llvm::IsaPred<LoadOp>) ||
         crossesFusionBarrier(src, arg)))
      return false;
  // A concat select fuses only into a fold reading it at the fold's own
  // index: a map phase has no lowering for it, and a transformed index would
  // break the vector it picks its input for.
  //
  //   %c = concat(%a, %b); %s = reduce.add(%c)             -> fuses
  //   %c = concat(%a, %b); %y = add(%c, reduce.add(%c))    -> no: map phase
  //   %c = concat(%a, %b); %s = reduce.add(transpose(%c))  -> no: transformed
  auto atFoldIndex = [](Value index) {
    auto fold = dyn_cast<IterReduceOp>(index.getParentRegion()->getParentOp());
    return fold &&
           index ==
               cast<RowwisePhaseOpInterface>(fold.getOperation()).getIndexArg();
  };
  if (holdsConcatSelect(src))
    for (auto [input, arg] :
         llvm::zip(dst.getInputs(), dst.getInputBlockArgs()))
      if (input == result && !loadsOnlyAt(arg, atFoldIndex))
        return false;
  return true;
}

// Leaves `dst` with unique inputs, as `fuseConsumer` does.
IterRowwiseOp fuseProducer(IterRowwiseOp dst, IterElementwiseOp src,
                           RewriterBase &rewriter) {
  spliceProducer(dst, src, rewriter);
  stampFusion(dst, src, /*isProducer=*/true);
  IterOpInterface(dst).dedupeInputs(rewriter);
  rewriter.eraseOp(src);
  return dst;
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
  // A concat select runs only in a fold, never at the row index.
  if (!(isPlainForeach(consumer) || isFusibleMutableStore(consumer, dst)) ||
      holdsConcatSelect(consumer) || dst.getInputChain() ||
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
    if (!loadsOnlyAt(body.getArgument(i),
                     [&](Value index) { return index == loopIdx; }))
      return false;
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

  // Row-level ops go before the map phases, which stay last.
  Block &newBody = newDst.getBody().front();
  auto maps = newBody.getOps<IterMapOp>();
  if (maps.empty())
    rewriter.setInsertionPointToEnd(&newBody);
  else
    rewriter.setInsertionPoint(*maps.begin());
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

// Whether `value` is a per-row result of `reduction`, or is computed from one
// by kernels over its per-row shape, which fuse into it as row-level ops.
bool derivesFromRows(Value value, IterRowwiseOp reduction) {
  Operation *def = value.getDefiningOp();
  if (def == reduction.getOperation())
    return true;
  auto kernel = dyn_cast_or_null<IterElementwiseOp>(def);
  return kernel && isPerRowShape(reduction, kernel.getShape()) &&
         llvm::any_of(kernel->getOperands(), [&](Value input) {
           return derivesFromRows(input, reduction);
         });
}

// Whether `user` ends up in `reduction`'s kernel. It does when it is that
// kernel, or a kernel over its rows that:
//   - reads its per-row results (`derivesFromRows`): an elementwise kernel
//     joins it as a map phase, as softmax's division joins the sum, and a
//     rowwise kernel runs it at the start of its row;
//   - or has readers that all end up there.
// TODO(GEX-3346): this doesn't check that the reader reads its own row, as the
// fusers do. A reader through a transpose leaves the consumer as its own
// kernel, as before.
bool endsUpIn(Operation *user, IterRowwiseOp reduction) {
  if (user == reduction)
    return true;
  auto kernel = dyn_cast<StructuredIterOpInterface>(user);
  if (!kernel || kernel.getShape() != reduction.getShape())
    return false;
  if (llvm::any_of(kernel->getOperands(), [&](Value input) {
        return derivesFromRows(input, reduction);
      }))
    return true;
  return isa<IterElementwiseOp>(user) && !user->use_empty() &&
         llvm::all_of(user->getUsers(), [&](Operation *next) {
           return endsUpIn(next, reduction);
         });
}

// Whether a later reduction may take `consumer` as a producer, so its result
// is never written: it reaches a reduction through kernels over its rows, and
// its readers all end up in that reduction's kernel. A graph output or a
// hand-written kernel reads it from memory, so it is written anyway.
bool laterReductionTakes(IterElementwiseOp consumer) {
  SmallVector<IterRowwiseOp> reductions;
  SmallVector<Value> worklist = {consumer.getResult(0)};
  DenseSet<Operation *> seen;
  while (!worklist.empty())
    for (Operation *user : worklist.pop_back_val().getUsers()) {
      if (!seen.insert(user).second)
        continue;
      if (auto reduction = dyn_cast<IterRowwiseOp>(user))
        reductions.push_back(reduction);
      else if (auto kernel = dyn_cast<IterElementwiseOp>(user);
               kernel && kernel.getShape() == consumer.getShape())
        llvm::append_range(worklist, kernel->getResults());
    }
  return llvm::any_of(reductions, [&](IterRowwiseOp reduction) {
    return llvm::all_of(consumer->getUsers(), [&](Operation *user) {
      return endsUpIn(user, reduction);
    });
  });
}

bool canFuseFullRowConsumer(IterRowwiseOp dst, IterElementwiseOp consumer) {
  // A map phase has no lowering for a concat select.
  if (!isPlainForeach(consumer) || holdsConcatSelect(consumer) ||
      dst.getInputChain() || consumer->getBlock() != dst->getBlock() ||
      !sameDevice(consumer, dst) || consumer.getShape() != dst.getShape())
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

  if (laterReductionTakes(consumer))
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

  // Drop the outputs nothing reads anymore.
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

// Whether `consumer` reads its input `arg`, a per-row tensor, only at its own
// row: at its row index, or through a broadcast in one of its phases.
bool readsOwnRowOnly(IterRowwiseOp consumer, BlockArgument arg) {
  int64_t rank = consumer.getShape().getRank(), axis = consumer.getAxis();
  return llvm::all_of(arg.getUsers(), [&](Operation *user) {
    auto load = dyn_cast<LoadOp>(user);
    if (!load)
      return false;
    if (load.getIndex() == consumer.getRowArg())
      return true;
    auto phase = load->getParentOfType<RowwisePhaseOpInterface>();
    std::optional<SmallVector<int64_t>> dims =
        phase ? traceLoopDims(load.getIndex(), phase.getIndexArg())
              : std::nullopt;
    return dims && readsOwnRow(*dims, rank, axis);
  });
}

// Whether a later rowwise kernel can read its input `arg` from a cache: it
// reads it only in its folds and map phases, at their own element, and those
// phases read no other cache.
bool readsOwnElementOnly(BlockArgument arg) {
  return llvm::all_of(arg.getUsers(), [](Operation *user) {
    auto load = dyn_cast<LoadOp>(user);
    auto phase = load ? dyn_cast<RowwisePhaseOpInterface>(load->getParentOp())
                      : RowwisePhaseOpInterface();
    return phase && !isa<IterCacheOp>(*phase) && !phase.getReadCache() &&
           load.getIndex() == phase.getIndexArg();
  });
}

// Whether `consumer` reads `dst`'s per-row outputs only at its own row, and
// at most one of its per-element outputs, only from its folds and map phases
// at their own element. Sets `cached` to that one.
bool readsOwnRowOrElement(IterRowwiseOp dst, IterRowwiseOp consumer,
                          std::optional<unsigned> &cached) {
  for (auto [input, arg] :
       llvm::zip(consumer.getInputs(), consumer.getInputBlockArgs())) {
    auto result = dyn_cast<OpResult>(input);
    if (!result || result.getOwner() != dst)
      continue;
    unsigned idx = result.getResultNumber();
    if (perRowStore(dst, idx)) {
      if (!readsOwnRowOnly(consumer, arg))
        return false;
      continue;
    }
    if (!perElementStore(dst, idx) || (cached && *cached != idx) ||
        !readsOwnElementOnly(arg))
      return false;
    cached = idx;
  }
  return true;
}

// Whether `dst` can run at the start of `consumer`'s row, its map phase, if
// any, becoming a cache. `consumer` alone reads `dst`'s per-row outputs, at
// its own row, and reads one of the map phase's outputs at the same element;
// other readers of those read `consumer`'s copy. `consumer` writes every
// element: a row that writes one value per row launches as one phase, which
// holds one fold. A `dst` that writes a buffer stays.
bool canFuseRowwiseConsumer(IterRowwiseOp dst, IterRowwiseOp consumer) {
  if (consumer == dst || consumer->getBlock() != dst->getBlock() ||
      !sameDevice(consumer, dst) || dst.getInputChain() ||
      dst.getOutputChain() || dst.getHasWriteEffect() ||
      consumer.getInputChain() || consumer.getShape() != dst.getShape() ||
      consumer.getAxis() != dst.getAxis() || !lastMapPhase(consumer))
    return false;
  IterMapOp phase = lastMapPhase(dst);
  if (llvm::count_if(dst.getBody().front(), llvm::IsaPred<IterMapOp>) !=
          (phase ? 1 : 0) ||
      (phase && cast<RowwisePhaseOpInterface>(*phase).getReadCache()))
    return false;
  for (auto [idx, result] : llvm::enumerate(dst.getResults())) {
    bool othersRead = llvm::any_of(
        result.getUsers(), [&](Operation *user) { return user != consumer; });
    if (perRowStore(dst, idx) ? othersRead : !perElementStore(dst, idx))
      return false;
  }
  std::optional<unsigned> cached;
  return readsOwnRowOrElement(dst, consumer, cached) &&
         static_cast<bool>(phase) == cached.has_value() &&
         canSinkTo(dst, consumer);
}

// Clones `phase` into a `map.iter.cache` that yields `cachedValue`. A store
// to an output `mapping` doesn't know, which only the consumer read, goes.
IterCacheOp cacheFromPhase(IterMapOp phase, Value cachedValue,
                           IRMapping &mapping, RewriterBase &rewriter) {
  Location loc = phase.getLoc();
  auto cache = IterCacheOp::create(rewriter, loc, cachedValue.getType());
  BlockArgument index = cast<RowwisePhaseOpInterface>(*phase).getIndexArg();
  Block *block =
      rewriter.createBlock(&cache.getBody(), {}, {index.getType()}, {loc});
  mapping.map(index, block->getArgument(0));
  for (Operation &op : phase.getBody().front()) {
    auto store = dyn_cast<StoreOp>(op);
    if (!store || mapping.contains(store.getTensor()))
      rewriter.clone(op, mapping);
  }
  YieldOp::create(rewriter, loc, mapping.lookup(cachedValue));
  rewriter.setInsertionPointAfter(cache);
  return cache;
}

// Moves `dst`'s row to the start of `consumer`'s, which then reads the values
// `dst` stored instead of loading them: a per-row value directly, the map
// phase's output from the cache the phase becomes. Outputs other kernels also
// read become `consumer`'s.
IterRowwiseOp fuseRowwiseConsumer(IterRowwiseOp dst, IterRowwiseOp consumer,
                                  RewriterBase &rewriter) {
  std::optional<unsigned> cachedIdx;
  bool reads = readsOwnRowOrElement(dst, consumer, cachedIdx);
  ASSERT_STREAM(reads, << "checked by canFuseRowwiseConsumer");
  SmallVector<unsigned> keptOutputs;
  SmallVector<Type> keptTypes;
  for (auto [idx, result] : llvm::enumerate(dst.getResults()))
    if (perElementStore(dst, idx) &&
        llvm::any_of(result.getUsers(),
                     [&](Operation *user) { return user != consumer; })) {
      keptOutputs.push_back(idx);
      keptTypes.push_back(result.getType());
    }
  unsigned numOutputs = consumer.getOutputBlockArgs().size();
  IterRowwiseOp newConsumer =
      keptTypes.empty()
          ? consumer
          : rebuildResults(consumer,
                           llvm::to_vector(llvm::seq<unsigned>(0, numOutputs)),
                           keptTypes, rewriter);

  IRMapping mapping;
  for (auto [input, arg] :
       llvm::zip(dst.getInputs(), dst.getInputBlockArgs())) {
    unsigned idx = newConsumer.getInputs().size();
    newConsumer.insertInputsAt(idx, input);
    mapping.map(arg, newConsumer.getInputBlockArgs()[idx]);
  }
  mapping.map(dst.getRowArg(), newConsumer.getRowArg());
  for (auto [i, idx] : llvm::enumerate(keptOutputs))
    mapping.map(dst.getOutputBlockArgs()[idx],
                newConsumer.getOutputBlockArgs()[numOutputs + i]);

  rewriter.setInsertionPointToStart(&newConsumer.getBody().front());
  IterMapOp phase = lastMapPhase(dst);
  IterCacheOp cache;
  SmallVector<Value> stored(dst.getNumResults());
  for (Operation &op : dst.getBody().front()) {
    if (auto store = dyn_cast<StoreOp>(op)) {
      auto arg = cast<BlockArgument>(store.getTensor());
      stored[arg.getArgNumber() - dst.getInputs().size()] =
          mapping.lookupOrDefault(store.getValue());
    } else if (&op == phase.getOperation()) {
      cache = cacheFromPhase(phase, perElementStore(dst, *cachedIdx).getValue(),
                             mapping, rewriter);
    } else {
      rewriter.clone(op, mapping);
    }
  }

  // Loads of `dst`'s outputs read the stored row values, or the cache.
  for (int64_t idx = newConsumer.getInputs().size() - 1; idx >= 0; --idx) {
    auto result = dyn_cast<OpResult>(newConsumer.getInputs()[idx]);
    if (!result || result.getOwner() != dst)
      continue;
    BlockArgument arg = newConsumer.getInputBlockArgs()[idx];
    for (Operation *user : llvm::make_early_inc_range(arg.getUsers())) {
      auto load = cast<LoadOp>(user);
      if (cachedIdx != result.getResultNumber()) {
        rewriter.replaceOp(load, stored[result.getResultNumber()]);
        continue;
      }
      rewriter.setInsertionPoint(load);
      rewriter.replaceOp(
          load, IterCacheLoadOp::create(rewriter, load.getLoc(), load.getType(),
                                        cache.getResult(), load.getIndex()));
    }
    rewriter.modifyOpInPlace(newConsumer,
                             [&] { newConsumer.eraseInputsAt(idx, 1); });
  }

  for (auto [i, idx] : llvm::enumerate(keptOutputs))
    rewriter.replaceAllUsesWith(dst.getResult(idx),
                                newConsumer.getResult(numOutputs + i));
  stampFusion(newConsumer, dst, /*isProducer=*/true);
  rewriter.eraseOp(dst);
  IterOpInterface(newConsumer).dedupeInputs(rewriter);
  return newConsumer;
}

// A rowwise kernel `dst` can run at the start of.
IterRowwiseOp findRowwiseConsumer(IterRowwiseOp dst) {
  for (Value result : dst.getResults())
    for (Operation *user : result.getUsers())
      if (auto consumer = dyn_cast<IterRowwiseOp>(user))
        if (canFuseRowwiseConsumer(dst, consumer))
          return consumer;
  return nullptr;
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

// Drops each erased kernel from the worklist: a merge moves on to a later
// entry, which a consumer fusion may then replace.
struct WorklistListener : RewriterBase::Listener {
  explicit WorklistListener(SetVector<Operation *> &worklist)
      : worklist(worklist) {}
  void notifyOperationErased(Operation *op) override { worklist.remove(op); }
  SetVector<Operation *> &worklist;
};

struct FuseReductionsPass
    : public MAP::impl::FuseReductionsBase<FuseReductionsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    SetVector<Operation *> worklist;
    graphOp.getBody().walk([&](IterRowwiseOp op) {
      worklist.insert(op);
      return WalkResult::skip();
    });
    WorklistListener listener(worklist);
    IRRewriter rewriter(graphOp.getContext(), &listener);

    while (!worklist.empty()) {
      auto dst = cast<IterRowwiseOp>(worklist.front());
      worklist.remove(dst);
      auto fuseWith = [&](auto find, auto fuse) {
        auto other = find(dst);
        if (!other)
          return false;
        dst = fuse(dst, other, rewriter);
        ++numKernelsFused;
        return true;
      };
      bool changed = true;
      while (changed) {
        changed = false;
        while (fuseWith(findProducer, fuseProducer))
          changed = true;
        changed |= fuseWith(findConsumer, fuseConsumer);
        changed |= fuseWith(findFullRowConsumer, fuseFullRowConsumer);
        changed |= fuseWith(findRowwiseConsumer, fuseRowwiseConsumer);
      }
    }
  }
};

} // namespace

} // namespace M::MAP
