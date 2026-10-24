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
// Hoists chains of index-transform ops (ops implementing
// MAP::IndexTransformOpInterface, e.g. map.index.slice) out of any
// CapturingFusionClosure (map.iter.elementwise.body, map.iter.prologue,
// map.iter.epilogue) into a chain of "stride" ops (e.g.
// map.index.slice.stride) in the enclosing map.iter.opaque, so the transform
// runs once per kernel invocation instead of once per loop index.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_LOOPINVARIANTVIEWMOTION
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Walks backward from `index` through a chain of index-transform ops,
// closest-to-load first, stopping at the first value that isn't one
// (typically the closure's own index block argument).
SmallVector<IndexTransformOpInterface> collectIndexTransformChain(Value index) {
  SmallVector<IndexTransformOpInterface> chain;
  Value current = index;
  while (Operation *def = current.getDefiningOp()) {
    auto transform = dyn_cast<IndexTransformOpInterface>(def);
    if (!transform)
      break;
    chain.push_back(transform);
    current = transform.getIndex();
  }
  return chain;
}

// Whether `op`'s result reaches a `map.store` index, directly or through
// further index transforms. Such an op maps an INDEX, not a tensor, so it
// cannot become a `.stride` view and stays in the closure. A load-side view
// composed on a store-index transform still hoists (e.g. a broadcast on a
// reshaped store index hoists; the reshape stays). The walk stays in index
// space, so a store consuming a loaded VALUE does not count.
bool transitivelyFeedsStore(IndexTransformOpInterface op) {
  return walkDataflowGraph<WalkDirection::Consumers>(
             op.getOperation(),
             [](Operation *user) -> WalkResult {
               if (isa<StoreOp>(user))
                 return WalkResult::interrupt();
               if (isa<IndexTransformOpInterface>(user))
                 return WalkResult::advance();
               return WalkResult::skip();
             })
      .wasInterrupted();
}

// Length of the leading run of `chain` that does not feed a store. Feeding a
// store is monotonic toward the loop index, so the hoistable ops are a prefix.
size_t hoistablePrefixLen(ArrayRef<IndexTransformOpInterface> chain) {
  size_t n = 0;
  while (n < chain.size() && !transitivelyFeedsStore(chain[n]))
    ++n;
  return n;
}

// Hoists `chain` (feeding `loadOp`) into stride ops before `closure`, mapping
// capture block arguments through `outerValues` since the hoisted ops land
// outside the closure. Rewrites `loadOp` to read the new capture at the chain's
// outermost index. `DedupeCapturingClosureCaptures` folds duplicate captures
// later.
void hoistChain(LoadOp loadOp, ArrayRef<IndexTransformOpInterface> chain,
                OpBuilder &builder, CapturingFusionClosure closure,
                const IRMapping &outerValues) {
  Value outerTensor = outerValues.lookupOrDefault(loadOp.getTensor());

  Value tensor = outerTensor;
  for (IndexTransformOpInterface transform : chain)
    tensor = transform.createStridedView(builder, tensor, outerValues);
  BlockArgument newCaptureArg = closure.addCapture(tensor);

  IndexTransformOpInterface outermost = chain.back();
  loadOp.getTensorMutable().assign(newCaptureArg);
  loadOp.getIndexMutable().assign(outermost.getIndex());
}

// A fused elementwise producer shares its consumer's transformed index like a
// fused load; rewire it the same way, with nothing to hoist.
void rewireElementwiseIndices(Region &body) {
  // Only operands change, so rewiring in-walk is safe.
  body.walk([&](CallElementwiseOp call) {
    SmallVector<IndexTransformOpInterface> chain =
        collectIndexTransformChain(call.getIndex());
    size_t prefix = hoistablePrefixLen(chain);
    if (prefix == 0)
      return;
    // The loop index, or the result of the first store-index transform.
    call.getIndexMutable().assign(chain[prefix - 1].getIndex());
  });
}

LogicalResult loopInvariantViewMotionInClosure(CapturingFusionClosure closure) {
  Region &body = closure.getBody();
  OpBuilder builder(closure.getOperation());
  builder.setInsertionPoint(closure.getOperation());

  IRMapping outerValues;
  Block::BlockArgListType captureArgs = closure.getCaptureBlockArgs();
  ValueRange captures = closure.getCaptures();
  for (auto [arg, capture] : llvm::zip_equal(captureArgs, captures))
    outerValues.map(arg, capture);

  SmallVector<LoadOp> loads;
  body.walk([&](LoadOp op) { loads.push_back(op); });

  for (LoadOp loadOp : loads) {
    SmallVector<IndexTransformOpInterface> chain =
        collectIndexTransformChain(loadOp.getIndex());
    size_t prefix = hoistablePrefixLen(chain);
    if (prefix == 0)
      continue;
    chain.truncate(prefix);
    hoistChain(loadOp, chain, builder, closure, outerValues);
  }

  rewireElementwiseIndices(body);

  // Erase closest-to-load first so each outer hop is unused when reached.
  SmallVector<IndexTransformOpInterface> transforms;
  body.walk([&](IndexTransformOpInterface op) { transforms.push_back(op); });
  for (IndexTransformOpInterface transform : llvm::reverse(transforms))
    if (transform->use_empty())
      transform->erase();

  return success();
}

struct LoopInvariantViewMotionPass
    : public MAP::impl::LoopInvariantViewMotionBase<
          LoopInvariantViewMotionPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();

    // Walk, since a kernel can sit inside control flow (e.g. mo.while).
    WalkResult liftResult = graphOp.walk([&](CapturingFusionClosure closure) {
      if (failed(loopInvariantViewMotionInClosure(closure)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (liftResult.wasInterrupted())
      return signalPassFailure();

    // Invariant: only store-feeding index-transform ops remain.
    WalkResult invariant = graphOp.walk([&](IndexTransformOpInterface op) {
      if (transitivelyFeedsStore(op))
        return WalkResult::advance();
      op.emitError(
          "LoopInvariantViewMotion: an index-transform op survived hoisting "
          "-- this is a bug in the pass, not in the input IR");
      return WalkResult::interrupt();
    });
    if (invariant.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MAP
