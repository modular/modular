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
// Raises unfused slice/reshape `map.iter.elementwise` view kernels back to
// MO/MOGG ops that lower without a copy.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_RAISEBACKCONTIGUOUSVIEWS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

std::optional<int64_t> matchContiguousSliceDim(SliceIndexOp slice);

// Returns the index-transform chain of `op`'s body if that body is a pure view:
// a run of `map.index.slice`/`map.index.reshape` ops threading the loop index
// into a single load of the sole data input, feeding the store -- i.e. exactly
//   [t_1, ..., t_n, load, store]
// where `t_1` consumes the loop index, each `t_{i+1}` consumes `t_i`'s result
// (single-use), and `load` reads `t_n`'s result from input-0's block arg. The
// returned ops are in body order. Empty if the body is anything else, e.g. a
// compute kernel or a broadcast/transpose view. A chain arises when `ViewFuser`
// fuses adjacent views into one kernel.
SmallVector<Operation *> matchViewChain(IterElementwiseOp op) {
  SmallVector<Operation *> empty;
  if (op.getNumResults() != 1)
    return empty;
  Block &body = op.getBody().front();
  auto &ops = body.getOperations();

  SmallVector<Operation *> transforms;
  auto it = ops.begin();
  for (; it != ops.end() && isa<SliceIndexOp, ReshapeIndexOp>(&*it); ++it)
    transforms.push_back(&*it);
  if (transforms.empty())
    return empty;

  auto load = it != ops.end() ? dyn_cast<LoadOp>(&*it++) : nullptr;
  auto store = it != ops.end() ? dyn_cast<StoreOp>(&*it++) : nullptr;
  if (!load || !store || it != ops.end())
    return empty;

  // Each transform takes its index as operand 0.
  Value cursor = store.getIndex();
  for (Operation *t : transforms) {
    if (t->getOperand(0) != cursor || !t->getResult(0).hasOneUse())
      return empty;
    cursor = t->getResult(0);
  }
  if (load.getIndex() != cursor)
    return empty;
  if (load.getTensor() != body.getArgument(0))
    return empty;

  // Reshape is always raise-able; every slice must be contiguous.
  for (Operation *t : transforms)
    if (auto slice = dyn_cast<SliceIndexOp>(t))
      if (!matchContiguousSliceDim(slice))
        return empty;
  return transforms;
}

// The sliced dimension is the first input dim of size != 1, or the input rank
// if there is none.
int64_t findSliceDim(ArrayRef<int64_t> inDims) {
  for (auto [i, dim] : llvm::enumerate(inDims))
    if (dim != 1)
      return i;
  return inDims.size();
}

// Returns the sliced dimension `k` if `slice` is a contiguous slice: every
// dimension but `k` keeps its input's full extent, and `k`'s start/stop/step
// are static with step 1. A slice with any symbolic dim takes the general
// (already zero-copy) MAPToMOGG path instead.
std::optional<int64_t> matchContiguousSliceDim(SliceIndexOp slice) {
  MAP::SliceSpecAttr spec = slice.getSpec();
  SmallVector<int64_t> inDims = spec.getSignature().getIn().getIntDims();
  SmallVector<int64_t> outDims = spec.getSignature().getOut().getIntDims();
  if (llvm::any_of(inDims, ShapedType::isDynamic) ||
      llvm::any_of(outDims, ShapedType::isDynamic))
    return std::nullopt;

  int64_t k = findSliceDim(inDims);
  for (auto [i, inDim] : llvm::enumerate(inDims))
    if (static_cast<int64_t>(i) != k && inDim != outDims[i])
      return std::nullopt;

  ArrayRef<int64_t> starts = spec.getStarts().asArrayRef();
  ArrayRef<int64_t> stops = spec.getStops().asArrayRef();
  ArrayRef<int64_t> steps = spec.getSteps().asArrayRef();
  if (k == static_cast<int64_t>(inDims.size()) ||
      ShapedType::isDynamic(starts[k]) || ShapedType::isDynamic(stops[k]) ||
      steps[k] != 1)
    return std::nullopt;

  return k;
}

// Raises a `matchViewChain` kernel to standalone zero-copy ops: each slice
// becomes `mogg.slice` and each reshape `mo.static.reshape`.
//
// The body threads the loop index (output space) through the transforms into
// the input load, so the transforms run output->input; the data ops are their
// reverse (input->output). E.g. body `slice -> reshape -> load(in)` raises to
// `reshape(in) -> slice(...)` on the buffer.
bool raiseViewChain(IterElementwiseOp op, unsigned &numSlicesRaised,
                    unsigned &numReshapesRaised) {
  SmallVector<Operation *> chain = matchViewChain(op);
  if (chain.empty())
    return false;

  OpBuilder builder(op);
  Value cur = op.getInputs()[0];
  unsigned slicesHere = 0, reshapesHere = 0;
  for (Operation *t : llvm::reverse(chain)) {
    auto curType = cast<MO::TensorType>(cur.getType());
    // The last data op (first body transform) yields the kernel's result type.
    bool isLast = (t == chain.front());
    if (auto reshape = dyn_cast<ReshapeIndexOp>(t)) {
      Type outType = isLast ? op.getResultTypes()[0]
                            : Type(curType.cloneWith(
                                  reshape.getSpec().getSignature().getOut()));
      cur = MO::Static::ReshapeOp::create(builder, op.getLoc(),
                                          TypeRange{outType}, ValueRange{cur})
                ->getResult(0);
      ++reshapesHere;
    } else {
      auto slice = cast<SliceIndexOp>(t);
      MAP::SliceSpecAttr spec = slice.getSpec();
      int64_t k = *matchContiguousSliceDim(slice);
      ArrayRef<int64_t> starts = spec.getStarts().asArrayRef();
      ArrayRef<int64_t> stops = spec.getStops().asArrayRef();
      // The slice's start operand is a host tensor; reuse its device for the
      // scalar bound constants.
      auto hostDevice =
          cast<MO::TensorType>(slice->getOperand(1).getType()).getDeviceRef();
      Value startK =
          MO::ConstantOp::create(builder, op.getLoc(), starts[k], hostDevice);
      Value stopK =
          MO::ConstantOp::create(builder, op.getLoc(), stops[k], hostDevice);
      Type outType =
          isLast ? op.getResultTypes()[0]
                 : Type(curType.cloneWith(spec.getSignature().getOut()));
      cur = MOGG::SliceOp::create(builder, op.getLoc(), outType, cur, startK,
                                  stopK)
                ->getResult(0);
      ++slicesHere;
    }
  }
  op->replaceAllUsesWith(ValueRange{cur});
  op->erase();
  numSlicesRaised += slicesHere;
  numReshapesRaised += reshapesHere;
  return true;
}

struct RaiseBackContiguousViewsPass
    : public MAP::impl::RaiseBackContiguousViewsBase<
          RaiseBackContiguousViewsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();
    // Walk, since a view kernel can sit inside control flow (e.g.
    // mo.parallel).
    unsigned slices = 0, reshapes = 0;
    SmallVector<IterElementwiseOp> foreachOps;
    graphOp.walk([&](IterElementwiseOp op) { foreachOps.push_back(op); });
    for (IterElementwiseOp op : foreachOps)
      raiseViewChain(op, slices, reshapes);
    numSlicesRaised = slices;
    numReshapesRaised = reshapes;
  }
};

} // namespace

bool isRaisableViewChain(IterElementwiseOp op) {
  return !matchViewChain(op).empty();
}

} // namespace M::MAP
