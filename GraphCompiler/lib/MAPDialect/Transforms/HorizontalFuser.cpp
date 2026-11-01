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
// Merges two sibling foreaches over the same index space that share an input
// into one multi-output kernel, so the shared operand is loaded once. Unlike
// the other fusers this pairs siblings rather than a producer with a consumer.
// It also moves a consumer into its producer's epilogue when the producer's
// result has other readers, which output fusion leaves alone; one that reads a
// strided window of the result runs there under a `map.index.slice.select`.
// An elementwise producer in that position merges with its consumer too,
// keeping its store and handing the consumer the value it computed.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/OutputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Support/DataflowUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <tuple>

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEHORIZONTAL
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// The pair is rebuilt at `b`, so anything between the two that reads `a`'s
// results would end up ahead of its definition. A graph region allows that and
// the dependency walks already rule out a cycle, so this only keeps kernels in
// def-before-use order, conservatively. A reader inside a region (an `mo.if`
// branch) counts against the op that encloses it here, which is what actually
// moves relative to `b`.
bool resultsUnusedBefore(IterElementwiseOp a, IterElementwiseOp b) {
  Block *block = b->getBlock();
  return llvm::none_of(a->getUsers(), [&](Operation *user) {
    Operation *inBlock = block->findAncestorOpInBlock(*user);
    return inBlock && inBlock->isBeforeInBlock(b);
  });
}

// Whether `op`'s body loads from its input `in`.
bool loadsInput(IterElementwiseOp op, Value in) {
  Block &body = op.getBody().front();
  for (auto [i, operand] : llvm::enumerate(op.getInputs()))
    if (operand == in &&
        llvm::any_of(body.getArgument(i).getUsers(),
                     [](Operation *user) { return isa<LoadOp>(user); }))
      return true;
  return false;
}

// The kernels `fusible` can pair: same shape and device, both loading
// `loadedOperand`.
struct FusionCandidateGroup {
  Attribute shape;
  Attribute device;
  Value loadedOperand;

  std::tuple<Attribute, Attribute, Value> key() const {
    return {shape, device, loadedOperand};
  }
  bool operator==(const FusionCandidateGroup &other) const {
    return key() == other.key();
  }
};

// A kernel belongs to one group per operand it loads.
SmallVector<FusionCandidateGroup>
fusionCandidateGroupsOf(IterElementwiseOp op) {
  SmallVector<FusionCandidateGroup> groups;
  for (Value in : op.getInputs()) {
    FusionCandidateGroup group{op.getShapeAttr(), MO::getDeviceOrDefault(op),
                               in};
    if (loadsInput(op, in) && !llvm::is_contained(groups, group))
      groups.push_back(group);
  }
  return groups;
}

bool fusible(IterElementwiseOp a, IterElementwiseOp b) {
  // 1) Check that both kernels have the same shape and device.
  if (a.getShapeAttr() != b.getShapeAttr())
    return false;
  if (MO::getDeviceOrDefault(a) != MO::getDeviceOrDefault(b))
    return false;

  // 2) Skip kernels with side effects (not supported for now).
  // A chain or a memory effect carries an ordering this pass doesn't model.
  if (a.getChain() || b.getChain() || a.getHasReadEffect() ||
      a.getHasWriteEffect() || b.getHasReadEffect() || b.getHasWriteEffect())
    return false;

  // A view chain becomes a zero-copy view after fusion, which a multi-result
  // kernel can't, so merging it would turn a free view into a copy.
  if (isRaisableViewChain(a) || isRaisableViewChain(b))
    return false;

  // 3) Only pair kernels that both load a shared operand.
  // Loading it once is the whole gain. An operand only read as slice bounds
  // (starts/stops/steps, shared by every window slice) saves nothing and would
  // steal the slice from the producer it could fuse into. Most pairs fail here,
  // so this runs before the dependency walks.
  if (!llvm::any_of(a.getInputs(), [&](Value in) {
        return llvm::is_contained(b.getInputs(), in) && loadsInput(a, in) &&
               loadsInput(b, in);
      }))
    return false;

  // 4) Check that no dependency between the 2 kernels makes the fused kernel
  // invalid.
  // The merged kernel takes `b`'s place, so nothing in between may read `a`'s
  // results, and neither kernel may depend on the other, even indirectly, or it
  // would consume its own result.
  if (!resultsUnusedBefore(a, b))
    return false;
  return !dataflowGraphDependsOn(b, a) && !dataflowGraphDependsOn(a, b);
}

// Whether a slice leaves a strided window of its input that no view can
// express -- the only kind worth a select in the producer's kernel. A
// contiguous slice (every step 1, and at most one sliced dim ahead of the
// first non-unit one) is an offset view, better left to view fusion than paid
// for with a second store.
bool isStridedWindow(SliceSpecAttr spec) {
  MOSH::ShapeAttr in = spec.getSignature().getIn();
  MOSH::ShapeAttr out = spec.getSignature().getOut();
  ArrayRef<int64_t> steps = spec.getSteps().asArrayRef();
  if (llvm::any_of(steps, [](int64_t step) { return step != 1; }))
    return true;
  bool nonUnitDimSeen = false;
  for (unsigned d = 0, e = in.getRank(); d < e; ++d) {
    if (in.getValues()[d] != out.getValues()[d] && nonUnitDimSeen)
      return true;
    if (in.getIntDims()[d] != 1)
      nonUnitDimSeen = true;
  }
  return false;
}

// The slice through which a consumer reads the fused operand, when a select
// at `selectIdx` can invert it: one load, through one `map.index.slice` of the
// loop index, in a body with one store at the loop index; a strided window no
// view expresses; and within the select's rules at that index. The body then
// moves into the select, where the loop index is the slice's output index --
// the consumer's own -- so elementwise work between the load and the store
// keeps its meaning there. Null otherwise.
// TODO(slariau): also take loads of the consumer's other inputs at the loop
// index, which the select keeps in bounds; today they decline the merge.
SliceIndexOp matchSelectableSlice(BlockArgument fusionArg,
                                  BlockArgument loopIdx,
                                  ArrayRef<StoreOp> stores, Value selectIdx) {
  if (stores.size() != 1 || !fusionArg.hasOneUse())
    return nullptr;
  StoreOp store = stores.front();
  auto load = dyn_cast<LoadOp>(*fusionArg.getUsers().begin());
  if (!load)
    return nullptr;
  auto slice = load.getIndex().getDefiningOp<SliceIndexOp>();
  if (!slice || slice.getIndex() != loopIdx || !slice->hasOneUse())
    return nullptr;
  if (store.getIndex() != loopIdx)
    return nullptr;
  if (llvm::any_of(loopIdx.getUsers(), [&](Operation *user) {
        return user != slice.getOperation() && user != store.getOperation() &&
               !isa<CallElementwiseOp>(user);
      }))
    return nullptr;
  if (!isStridedWindow(slice.getSpec()) ||
      failed(SliceSelectOp::verifySpecAt(slice.getSpec(), selectIdx)))
    return nullptr;
  return slice;
}

// The producer's own output store and each consumer of that result write one
// element of the same index space, so they are siblings -- but one of them
// sits inside the producer, where `fuse` cannot reach it. So the merge goes
// the other way: the consumer moves into the producer's epilogue and stores
// beside the store already there, which is the one the producer still owes the
// readers fusion cannot absorb. Output fusion will not do this, since it takes
// a result with a single reader.
bool fuseEscapingReader(IterOpaqueOp &dst) {
  for (unsigned idx = 0, e = dst.getNumNonChainResults(); idx < e; ++idx) {
    Value res = dst->getResult(idx);
    if (!isa<MO::TensorType>(res.getType()))
      continue;

    IterElementwiseOp src;
    unsigned numConsumers = 0;
    bool escapes = false;
    for (OpOperand &use : res.getUses()) {
      auto consumer = dyn_cast<IterElementwiseOp>(use.getOwner());
      if (consumer && use.getOwner()->getBlock() == dst->getBlock()) {
        ++numConsumers;
        src = consumer;
        continue;
      }
      escapes = true;
    }
    // Without an escaping reader this is a plain producer/consumer edge that
    // output fusion already takes. More than one consumer waits for `fuse` to
    // merge them with each other first, since moving one in would strand the
    // other behind a producer it can no longer reach.
    if (!escapes || numConsumers != 1)
      continue;
    if (src.getChain() || src.getHasReadEffect() || src.getHasWriteEffect())
      continue;
    if (llvm::count(src.getInputs(), res) != 1)
      continue;
    if (llvm::any_of(src->getResultTypes(),
                     [](Type t) { return !isa<MO::TensorType>(t); }) ||
        llvm::any_of(dst->getResultTypes(),
                     [](Type t) { return !isa<MO::TensorType>(t); }))
      continue;

    IterEpilogueOp epilogueOp = dst.getEpilogueForResult(idx);
    if (!epilogueOp)
      continue;
    Block &epilogueBody = epilogueOp.getBody().front();
    SmallVector<StoreOp> epilogueStores = getOutputStores(epilogueBody);
    Block &srcBody = src.getBody().front();
    SmallVector<StoreOp> srcStores = getOutputStores(srcBody);
    if (epilogueStores.size() != 1 || srcStores.size() != src.getNumResults())
      continue;
    // A select in the consumer moves into the epilogue whole, so it must hold
    // at the epilogue's store index.
    if (llvm::any_of(
            srcBody.getOps<SliceSelectOp>(), [&](SliceSelectOp select) {
              return failed(SliceSelectOp::verifySpecAt(
                  select.getSpec(), epilogueStores.front().getIndex()));
            }))
      continue;

    // The consumer must read the fused operand at its own loop index -- the
    // epilogue hands over one element, with no way to honor a transform --
    // or through a strided slice of it, which a select inverts: the consumer
    // then runs only where the slice selects the element, at its own index.
    unsigned fusionArgIdx =
        llvm::find(src.getInputs(), res) - src.getInputs().begin();
    BlockArgument fusionArg = srcBody.getArgument(fusionArgIdx);
    BlockArgument srcLoopIdx = srcBody.getArguments().back();
    SliceIndexOp slice;
    SliceSpecAttr sliceSpec;
    if (llvm::any_of(fusionArg.getUsers(), [&](Operation *user) {
          auto load = dyn_cast<LoadOp>(user);
          return !load || load.getIndex() != srcLoopIdx;
        })) {
      StoreOp epilogueStore = epilogueStores.front();
      if (!isa<MO::TensorType>(epilogueStore.getTensor().getType()))
        continue;
      // The select would sit at the epilogue store's index, so the merge
      // checks the select's rules there. They reject a transformed index,
      // such as a fused reshape's, which can carry one stored vector across
      // rows of the slice's input.
      slice = matchSelectableSlice(fusionArg, srcLoopIdx, srcStores,
                                   epilogueStore.getIndex());
      if (!slice)
        continue;
      sliceSpec = slice.getSpec();
    }
    // Another operand reaching back through `dst` would make the merge a cycle.
    if (llvm::any_of(src->getOperands(), [&](Value other) {
          return other != res && dataflowGraphDependsOn(other, dst);
        }))
      continue;

    IRRewriter rewriter(dst.getContext());
    unsigned numDpsBefore = dst.getNumNonChainResults();
    dst = growOutputs(dst, src->getResultTypes(), rewriter);
    epilogueOp = dst.getEpilogueForResult(idx);
    Block &body = epilogueOp.getBody().front();

    StoreOp epilogueStore = getOutputStores(body).front();
    Value val = epilogueStore.getValue();
    Value storeIdx = epilogueStore.getIndex();
    Location loc = epilogueStore.getLoc();

    // The fused operand's block arg is a tensor; `val` is already the scalar
    // `dst` produced for this element, so every load of it becomes `val`.
    for (Operation *user : llvm::to_vector(fusionArg.getUsers())) {
      auto load = cast<LoadOp>(user);
      load.getResult().replaceAllUsesWith(val);
      load.erase();
    }
    // The select takes over the slice, spec and all.
    if (slice)
      slice.erase();

    // Every other input the body still reads becomes a capture, threaded
    // through a new input of `dst` -- the epilogue is isolated from above. A
    // slice's bounds tensors are no longer read, since the select carries them
    // statically.
    for (auto [i, input] : llvm::enumerate(src.getInputs())) {
      if (i == fusionArgIdx || srcBody.getArgument(i).use_empty())
        continue;
      unsigned dstInputIdx = dst.getInputs().size();
      dst.insertInputsAt(dstInputIdx, input);
      epilogueOp.getCapturesMutable().append(
          dst.getInputBlockArgs()[dstInputIdx]);
      srcBody.getArgument(i).replaceAllUsesWith(
          body.addArgument(input.getType(), input.getLoc()));
    }

    // The grown output block args are the consumer's destinations.
    SmallVector<Value> caps;
    Block::BlockArgListType outArgs = dst.getOutputBlockArgs();
    for (unsigned i = 0, n = src->getNumResults(); i < n; ++i) {
      BlockArgument outArg = outArgs[numDpsBefore + i];
      epilogueOp.getCapturesMutable().append(outArg);
      caps.push_back(body.addArgument(outArg.getType(), outArg.getLoc()));
    }

    // A store in a select keeps its place and its index, the select's
    // argument; only its destination moves to the new capture. The whole
    // stores are rebuilt at the consumer's index below.
    SmallVector<std::pair<Value, Value>> wholeStores;
    for (auto [store, cap] : llvm::zip_equal(getOutputStores(srcBody), caps)) {
      if (store->getBlock() != &srcBody) {
        store.getTensorMutable().assign(cap);
        continue;
      }
      wholeStores.push_back({cap, store.getValue()});
      store.erase();
    }

    // The producer's own store stays: it is what the escaping reader is owed.
    // Through a slice, the consumer's body moves into a select, which runs it
    // at the slice's index and only where the slice selects the element.
    Block *consumerBody = &body;
    Value consumerIdx = storeIdx;
    if (sliceSpec) {
      rewriter.setInsertionPointToEnd(&body);
      auto select = SliceSelectOp::create(rewriter, loc, sliceSpec, storeIdx);
      consumerBody =
          rewriter.createBlock(&select.getBody(), select.getBody().end(),
                               {storeIdx.getType()}, {loc});
      consumerIdx = consumerBody->getArgument(0);
    }
    srcLoopIdx.replaceAllUsesWith(consumerIdx);
    srcBody.eraseArguments(0, srcBody.getNumArguments());
    rewriter.inlineBlockBefore(&srcBody, consumerBody, consumerBody->end());
    rewriter.setInsertionPointToEnd(consumerBody);
    for (auto [cap, storedVal] : wholeStores)
      StoreOp::create(rewriter, loc, cap, consumerIdx, storedVal);

    SmallVector<Value> replacements;
    for (unsigned i = 0, n = src->getNumResults(); i < n; ++i)
      replacements.push_back(dst->getResult(numDpsBefore + i));
    src->replaceAllUsesWith(replacements);

    dst.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionEpilogueAttr>(
        dst.getContext(), {dst.getOperation(), src.getOperation()},
        {static_cast<uint32_t>(idx)}));
    dst->setLoc(
        mlir::FusedLoc::get(dst.getContext(), {src->getLoc(), dst->getLoc()}));
    src->erase();
    IterOpInterface(dst).dedupeInputs(rewriter);
    return true;
  }
  return false;
}

// The block argument `op`'s body reads its input `in` through. Body args are
// [inputs..., outputs..., index], as MOToMAP builds them.
BlockArgument inputArgFor(IterElementwiseOp op, Value in) {
  return op.getBody().front().getArgument(llvm::find(op.getInputs(), in) -
                                          op.getInputs().begin());
}

// `forwarded`, when set, is one of `a`'s results that `b` reads:
//
//   %x = mo.add(%p, %q)    // a
//   %y = mo.relu(%x)       // b, forwarded = %x
//   mo.output %x, %y       // %x is also read elsewhere
//
// The merged kernel takes no input for `%x`: `b`'s loads of it become the
// value `a` stores to it, and `a`'s store stays for the other readers.
//
// `slice`, when set, is how `b` reads `%x`, `%y = mo.relu(%x[:, 1::2])`: `b`'s
// body then runs in a `map.index.slice.select` at `a`'s index, where the slice
// selects the element, and stores at the slice's index.
//
// Returns the merged kernel, created where `b` was.
IterElementwiseOp fuse(IterElementwiseOp a, IterElementwiseOp b,
                       Value forwarded = {}, SliceIndexOp slice = {}) {
  assert((!slice || forwarded) && "a slice is how `b` reads `forwarded`");
  // Merged inputs: `a`'s, then the ones only `b` carries. A shared operand
  // keeps `a`'s slot, which is what collapses the duplicate load.
  SmallVector<Value> inputs(a.getInputs());
  SmallVector<unsigned> slotForBInput;
  for (auto [i, in] : llvm::enumerate(b.getInputs())) {
    // No slot for the forwarded result, whose loads are replaced below, nor
    // for the slice's bounds, which the select carries statically.
    BlockArgument arg = b.getBody().front().getArgument(i);
    if (in == forwarded || (slice && !arg.use_empty() &&
                            llvm::all_of(arg.getUsers(), [&](Operation *user) {
                              return user == slice.getOperation();
                            }))) {
      slotForBInput.push_back(0);
      continue;
    }
    auto *it = llvm::find(inputs, in);
    slotForBInput.push_back(std::distance(inputs.begin(), it));
    if (it == inputs.end())
      inputs.push_back(in);
  }

  SmallVector<Type> resultTypes(a->getResultTypes());
  llvm::append_range(resultTypes, b->getResultTypes());

  MLIRContext *ctx = a.getContext();
  Location loc = mlir::FusedLoc::get(ctx, {a.getLoc(), b.getLoc()});
  OpBuilder builder(b);
  // Deferred if either member is, as input fusion combines the launch policy.
  auto fused = IterElementwiseOp::create(
      builder, loc, resultTypes, a.getShapeAttr(), inputs,
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::createFusionInfo<MOGG::FusionHorizontalAttr>(
          ctx, {a.getOperation(), b.getOperation()}),
      /*hasReadEffect=*/false, /*hasWriteEffect=*/false,
      /*dispatch=*/
      a.getDispatch() == MOGG::DispatchKind::Deferred ||
              b.getDispatch() == MOGG::DispatchKind::Deferred
          ? MOGG::DispatchKind::Deferred
          : MOGG::DispatchKind::Immediate);

  // `device` and friends are discardable, so `create` does not carry them.
  // The pair shares its device; on any other clash `a`'s attribute wins.
  NamedAttrList attrs(b->getDiscardableAttrDictionary());
  for (NamedAttribute attr : a->getDiscardableAttrs())
    attrs.set(attr.getName(), attr.getValue());
  fused->setDiscardableAttrs(attrs.getDictionary(ctx));

  Block &aBody = a.getBody().front();
  Block *body = &fused.getBody().emplaceBlock();
  for (Value in : inputs)
    body->addArgument(in.getType(), loc);
  for (Type ty : resultTypes)
    body->addArgument(ty, loc);
  body->addArgument(aBody.getArguments().back().getType(), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument index = body->getArguments().back();
  unsigned outputBase = inputs.size();

  IRMapping map;
  BlockArgument forwardedArg = forwarded ? inputArgFor(b, forwarded) : nullptr;
  Value forwardedValue;
  auto cloneBody = [&](IterElementwiseOp src, ArrayRef<unsigned> inputSlots,
                       unsigned firstOutput, OpBuilder &builder,
                       Value srcIndex) {
    // Body args are [inputs..., outputs..., index], as MOToMAP builds them.
    Block &srcBody = src.getBody().front();
    unsigned numInputs = src.getInputs().size();
    for (unsigned i = 0; i < numInputs; ++i)
      map.map(srcBody.getArgument(i), body->getArgument(inputSlots[i]));
    for (unsigned i = 0, e = src->getNumResults(); i < e; ++i)
      map.map(srcBody.getArgument(numInputs + i),
              body->getArgument(outputBase + firstOutput + i));
    map.map(srcBody.getArguments().back(), srcIndex);
    for (Operation &op : srcBody) {
      // A load of the forwarded result is the value `a` just stored.
      auto load = dyn_cast<LoadOp>(op);
      if (load && forwardedArg && load.getTensor() == forwardedArg) {
        map.map(load.getResult(), forwardedValue);
        continue;
      }
      if (slice && &op == slice.getOperation())
        continue;
      builder.clone(op, map);
    }
  };

  SmallVector<unsigned> aSlots(
      llvm::to_vector(llvm::seq<unsigned>(0, a.getInputs().size())));
  cloneBody(a, aSlots, 0, bodyBuilder, index);
  if (forwarded) {
    // `a` stores the result once, at its loop index (see
    // `fuseEscapingConsumer`); `b` loads it only at its own.
    unsigned resultIdx = cast<OpResult>(forwarded).getResultNumber();
    auto store =
        cast<StoreOp>(*aBody.getArgument(a.getInputs().size() + resultIdx)
                           .getUsers()
                           .begin());
    forwardedValue = map.lookup(store.getValue());
  }
  if (slice) {
    auto select =
        SliceSelectOp::create(bodyBuilder, loc, slice.getSpec(), index);
    Block *selectBody = bodyBuilder.createBlock(
        &select.getBody(), select.getBody().end(), {index.getType()}, {loc});
    OpBuilder selectBuilder = OpBuilder::atBlockEnd(selectBody);
    cloneBody(b, slotForBInput, a->getNumResults(), selectBuilder,
              selectBody->getArgument(0));
  } else {
    cloneBody(b, slotForBInput, a->getNumResults(), bodyBuilder, index);
  }

  unsigned split = a->getNumResults();
  a.replaceAllUsesWith(fused.getResults().take_front(split));
  b.replaceAllUsesWith(fused.getResults().drop_front(split));
  b.erase();
  a.erase();
  return fused;
}

// Whether `opaque` reads `value` only through prologues, where prologue
// fusion can recompute its producer instead of loading it.
bool readsOnlyThroughPrologue(IterOpaqueOp opaque, Value value) {
  for (auto [in, arg] :
       llvm::zip(opaque.getInputs(), opaque.getInputBlockArgs()))
    if (in == value &&
        !(arg.hasOneUse() && isa<IterPrologueOp>(*arg.getUsers().begin())))
      return false;
  return true;
}

// The elementwise counterpart of `fuseEscapingReader(IterOpaqueOp &)`:
// `dstKernel`'s result has one foreach consumer, `srcKernel`, and other
// readers too, so input fusion leaves the pair alone. They merge into one
// kernel that stores the result for the other readers and hands `srcKernel`
// the value just computed instead of a reload.
bool fuseEscapingConsumer(IterElementwiseOp dstKernel) {
  if (dstKernel.getChain() || dstKernel.getHasReadEffect() ||
      dstKernel.getHasWriteEffect() || isRaisableViewChain(dstKernel))
    return false;
  Block &dstBody = dstKernel.getBody().front();
  for (auto [k, result] : llvm::enumerate(dstKernel->getResults())) {
    // Returns the one foreach that reads `result`, or null. The merge pays
    // only if `result` stays in memory anyway, because something else reads
    // it:
    //
    //   %x = mo.add(%p, %q)
    //   %y = mo.relu(%x)          // the consumer
    //   mo.output %x, %y          // %x stays stored: merge, relu stops
    //                             // reloading it
    //
    // Not when the other reader is an opaque kernel's prologue:
    //
    //   %b = mo.static.broadcast_to(%p)
    //   %x = mo.add(%b, %q)
    //   %y = mo.relu(%x)
    //   %g = mo.gather(%x, %i)    // later fusion recomputes the add in the
    //                             // relu and in the gather's prologue, so
    //                             // %x is never stored
    //
    // Merging there would store %x where main doesn't, so we skip it, even
    // though for some producers (a plain add) that recompute never happens.
    // Several foreach readers return null too: they merge with each other
    // first.
    // TODO: this guesses whether later fusion would leave `dstKernel`'s
    // result materialized, which is brittle. Fusion should choose between
    // this merge and prologue fusion by comparing what each costs, rather
    // than by predicting what the other pass will do.
    auto getAndCheckConsumerKernel = [&]() -> IterElementwiseOp {
      IterElementwiseOp consumerKernel;
      bool needsStore = false;
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        bool sameBlock = user->getBlock() == dstKernel->getBlock();
        if (auto consumer = dyn_cast<IterElementwiseOp>(user);
            consumer && sameBlock) {
          if (consumerKernel)
            return nullptr;
          consumerKernel = consumer;
        } else if (auto opaque = dyn_cast<IterOpaqueOp>(user);
                   opaque && sameBlock) {
          needsStore |= !readsOnlyThroughPrologue(opaque, result);
        } else {
          needsStore = true;
        }
      }
      return needsStore ? consumerKernel : nullptr;
    };
    IterElementwiseOp srcKernel = getAndCheckConsumerKernel();
    if (!srcKernel)
      continue;

    // Both kernels run on the same device, and `srcKernel` has no chain or
    // memory effect and isn't a view chain.
    if (MO::getDeviceOrDefault(dstKernel) !=
            MO::getDeviceOrDefault(srcKernel) ||
        srcKernel.getChain() || srcKernel.getHasReadEffect() ||
        srcKernel.getHasWriteEffect() || isRaisableViewChain(srcKernel))
      continue;
    // `srcKernel` takes exactly one input from `dstKernel`, this result, so
    // there is one value to forward.
    if (llvm::count_if(srcKernel.getInputs(), [&](Value in) {
          return in.getDefiningOp() == dstKernel.getOperation();
        }) != 1)
      continue;
    // It reads that input only as `x[i]` at its own loop index, over the same
    // indices: the element iteration i of the merged kernel holds. Or through
    // one strided slice, `x[slice(i)]`, which a select inverts at
    // `dstKernel`'s index.
    Block &srcBody = srcKernel.getBody().front();
    BlockArgument srcIdx = srcBody.getArguments().back();
    BlockArgument fusionArg = inputArgFor(srcKernel, result);
    bool readsAtOwnIndex =
        llvm::all_of(fusionArg.getUsers(), [&](Operation *user) {
          auto load = dyn_cast<LoadOp>(user);
          return load && load->getBlock() == &srcBody &&
                 load.getIndex() == srcIdx;
        });
    SliceIndexOp slice;
    if (!readsAtOwnIndex) {
      slice = matchSelectableSlice(fusionArg, srcIdx, getOutputStores(srcBody),
                                   dstBody.getArguments().back());
      if (!slice)
        continue;
    } else if (dstKernel.getShapeAttr() != srcKernel.getShapeAttr()) {
      continue;
    }
    // `dstKernel` writes the result once, `x[i] = v` at its own loop index;
    // `v` is the value forwarded.
    BlockArgument out = dstBody.getArgument(dstKernel.getInputs().size() + k);
    auto store = out.hasOneUse() ? dyn_cast<StoreOp>(*out.getUsers().begin())
                                 : StoreOp();
    if (!store || store->getBlock() != &dstBody ||
        store.getIndex() != dstBody.getArguments().back())
      continue;

    // The merged kernel takes `srcKernel`'s place, as a sibling merge's does,
    // so nothing in between may read `dstKernel`, and no other operand of
    // `srcKernel` may depend on it.
    if (!resultsUnusedBefore(dstKernel, srcKernel) ||
        llvm::any_of(srcKernel->getOperands(), [&](Value other) {
          return other != result && dataflowGraphDependsOn(other, dstKernel);
        }))
      continue;
    fuse(dstKernel, srcKernel, result, slice);
    return true;
  }
  return false;
}

struct FuseHorizontalPass
    : public MAP::impl::FuseHorizontalBase<FuseHorizontalPass> {
  using Base::Base;
  FuseHorizontalPass(const FuseHorizontalPass &other) : Base(other) {}

  void runOnOperation() override {
    // Fusion stays within a block, so each block runs to its own fixpoint, as
    // in `fuseOutput`. Blocks are not fully independent: a select merge in a
    // branch drops the slice's bounds operands, which can remove the capture
    // an outer pair's cycle check went through, so what fuses can depend on
    // the order blocks are visited in. Every order gives valid IR. Unlike in
    // `fuseOutput`, a block is not sorted first: the greedy choices follow
    // block order.
    getOperation().getBody().walk([&](Block *block) { fuseBlock(*block); });
  }

  void fuseBlock(Block &block) {
    // Pairing can leave a producer one consumer for the merge, and a merge can
    // expose a new pair, so alternate until none of them fuses anything.
    while (true) {
      bool pairedSiblings = fuseSiblings(block);
      bool mergedReaders = fuseEscapingReaders(block);
      bool mergedConsumers = fuseEscapingConsumers(block);
      if (!pairedSiblings && !mergedReaders && !mergedConsumers)
        break;
    }
  }

  // Returns whether any pair was fused.
  bool fuseSiblings(Block &block) {
    // Kernels with the same shape and device that load the same operand form
    // a candidate group, and a kernel joins one group per operand it loads.
    // The groups are built once. One sweep then visits the kernels in block
    // order and fuses each with the first fusible later member of its groups.
    // A fused pair empties `a`'s slot and puts the merged kernel in `b`'s,
    // where `fuse` creates it, so slots stay in block order and the sweep
    // reaches the merged kernel again, to try it against the members after
    // it. Only the pair and the users of its results need new groups;
    // rebuilding every kernel's groups after each pair would rescan the whole
    // block once per fusion.
    //
    // For example, K0..K3 all load %x, so group %x = {K0, K1, K2, K3}, and
    // every pair is fusible:
    //
    //   slot     0     1      2       3
    //   start    K0    K1     K2      K3
    //   i = 0    -     K01    K2      K3       fuse(K0, K1) into slot 1
    //   i = 1    -     -      K012    K3       fuse(K01, K2) into slot 2
    //   i = 2    -     -      -       K0123    fuse(K012, K3) into slot 3
    //
    // The sweep goes kernel by kernel, not group by group: kernels of other
    // groups between these slots are visited as the sweep reaches them, and
    // groups are patched after every fusion, not rebuilt once a group is
    // done. The sweep never revisits earlier kernels: a merged kernel reads
    // everything both members read, so every pair rejected before it is still
    // rejected.
    SmallVector<IterElementwiseOp> kernels =
        llvm::to_vector(block.getOps<IterElementwiseOp>());
    DenseMap<Operation *, unsigned> slotOf;

    // A pair that shares no group fails `fusible`, so a kernel is tried only
    // against the later kernels of its own groups. They are still visited in
    // block order, which the greedy choice depends on.
    SmallVector<SmallVector<FusionCandidateGroup>> groupsOf(kernels.size());
    DenseMap<std::tuple<Attribute, Attribute, Value>, SmallVector<unsigned>>
        members;
    auto addGroups = [&](unsigned slot) {
      groupsOf[slot] = fusionCandidateGroupsOf(kernels[slot]);
      for (const FusionCandidateGroup &group : groupsOf[slot])
        members[group.key()].push_back(slot);
    };
    auto dropGroups = [&](unsigned slot) {
      for (const FusionCandidateGroup &group : groupsOf[slot])
        llvm::erase(members[group.key()], slot);
      groupsOf[slot].clear();
    };
    for (auto [slot, kernel] : llvm::enumerate(kernels)) {
      slotOf[kernel] = slot;
      addGroups(slot);
    }

    bool fusedAny = false;
    for (unsigned i = 0, e = kernels.size(); i < e; ++i) {
      if (!kernels[i])
        continue;
      SmallVector<unsigned> candidates;
      for (const FusionCandidateGroup &group : groupsOf[i])
        for (unsigned j : members.at(group.key()))
          if (j > i)
            candidates.push_back(j);
      llvm::sort(candidates);
      candidates.erase(llvm::unique(candidates), candidates.end());
      for (unsigned j : candidates) {
        IterElementwiseOp a = kernels[i];
        IterElementwiseOp b = kernels[j];
        if (!fusible(a, b))
          continue;
        // Users of the pair's results load the merged kernel's results after
        // the fusion, which changes their groups.
        llvm::SetVector<unsigned> users;
        for (IterElementwiseOp member : {a, b})
          for (Operation *user : member->getUsers())
            if (auto it = slotOf.find(user); it != slotOf.end())
              users.insert(it->second);
        users.remove(i);
        users.remove(j);
        for (unsigned slot : users)
          dropGroups(slot);
        dropGroups(i);
        dropGroups(j);
        slotOf.erase(a);
        slotOf.erase(b);

        IterElementwiseOp merged = fuse(a, b);
        kernels[i] = nullptr;
        kernels[j] = merged;
        slotOf[merged] = j;
        addGroups(j);
        for (unsigned slot : users)
          addGroups(slot);
        ++numKernelsFused;
        fusedAny = true;
        break;
      }
    }
    return fusedAny;
  }

  // Returns whether any consumer was merged.
  bool fuseEscapingReaders(Block &block) {
    bool mergedAny = false;
    // Merging a consumer into a producer's epilogue rebuilds the producer, so
    // the scan restarts after each one.
    bool merged = true;
    while (merged) {
      merged = false;
      auto opaques = llvm::to_vector(block.getOps<IterOpaqueOp>());
      for (IterOpaqueOp dst : opaques) {
        if (fuseEscapingReader(dst)) {
          ++numKernelsFused;
          merged = true;
          mergedAny = true;
          break;
        }
      }
    }
    return mergedAny;
  }

  // Returns whether any consumer was merged.
  bool fuseEscapingConsumers(Block &block) {
    bool mergedAny = false;
    // A merge replaces both kernels, so the scan restarts after each one.
    bool merged = true;
    while (merged) {
      merged = false;
      for (IterElementwiseOp dst :
           llvm::to_vector(block.getOps<IterElementwiseOp>())) {
        if (fuseEscapingConsumer(dst)) {
          ++numKernelsFused;
          merged = true;
          mergedAny = true;
          break;
        }
      }
    }
    return mergedAny;
  }

  Statistic numKernelsFused{this, "num-kernels-fused",
                            "Number of kernels fused"};
};

} // namespace

} // namespace M::MAP
