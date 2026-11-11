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
// Fuses each `mo.static.reshape` and contiguous `mogg.slice` into the
// `map.iter.opaque` kernels that read it, as a zero-copy stride view in the
// kernel's body.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_FUSECONTIGUOUSVIEWS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// The `map.index.slice.stride` spec of `slice`, or null when its shapes are
// parametric or its bounds aren't constants. `mogg.slice` cuts the dimension
// `findContiguousSliceDim` names and keeps the others whole.
SliceSpecAttr getSliceSpec(MOGG::SliceOp slice) {
  auto inTy = cast<MO::TensorType>(slice.getInput().getType());
  auto outTy = cast<MO::TensorType>(slice.getResult().getType());
  if (!inTy.getShape().isStatic() || !outTy.getShape().isStatic())
    return {};
  std::optional<int64_t> start =
      MO::tryGetConstantInt(cast<MO::TensorValue>(slice.getStart()));
  std::optional<int64_t> stop =
      MO::tryGetConstantInt(cast<MO::TensorValue>(slice.getStop()));
  if (!start || !stop)
    return {};

  SmallVector<int64_t> inDims = inTy.getShape().getIntDims();
  int64_t rank = inDims.size();
  int64_t k = findContiguousSliceDim(inDims);
  SmallVector<int64_t> starts(rank, 0), stops(inDims), steps(rank, 1);
  starts[k] = *start;
  stops[k] = *stop;

  MLIRContext *ctx = slice.getContext();
  return SliceSpecAttr::get(
      ctx, ViewSignatureAttr::get(ctx, inTy.getShape(), outTy.getShape()),
      DenseI64ArrayAttr::get(ctx, starts), DenseI64ArrayAttr::get(ctx, stops),
      DenseI64ArrayAttr::get(ctx, steps));
}

// Rewrites each input of `kernel` that a reshape or contiguous slice produces
// to read the view's input through a stride view at the top of the body. The
// view keeps the view op's result type, so the call operands and closure
// captures that read the old input stay well typed. The old input and the view
// op are left to the canonicalizer.
unsigned fuseViewsInto(IterOpaqueOp kernel) {
  MLIRContext *ctx = kernel.getContext();
  Block &body = kernel.getBody().front();

  llvm::DenseMap<Operation *, Value> views;
  unsigned numFused = 0;
  // A view's input can be a view too (e.g. a reshape of a slice), so the
  // inputs this appends are visited as well.
  for (unsigned idx = 0; idx < kernel.getInputs().size(); ++idx) {
    Operation *viewOp = kernel.getInputs()[idx].getDefiningOp();
    if (!isa_and_nonnull<MO::Static::ReshapeOp, MOGG::SliceOp>(viewOp))
      continue;

    Value &view = views[viewOp];
    if (!view) {
      // Each view goes at the top of the body, ahead of any view built over
      // it.
      OpBuilder builder = OpBuilder::atBlockBegin(&body);
      unsigned first = kernel.getInputs().size();
      Value input = viewOp->getOperand(0);
      auto viewTy = cast<MO::TensorType>(viewOp->getResult(0).getType());
      if (auto reshape = dyn_cast<MO::Static::ReshapeOp>(viewOp)) {
        kernel.insertInputsAt(first, input);
        auto spec = ReshapeSpecAttr::get(
            ctx, ViewSignatureAttr::get(
                     ctx, cast<MO::TensorType>(input.getType()).getShape(),
                     viewTy.getShape()));
        view = ReshapeStrideOp::create(builder, reshape.getLoc(), viewTy, spec,
                                       kernel.getInputBlockArgs()[first]);
      } else {
        auto slice = cast<MOGG::SliceOp>(viewOp);
        SliceSpecAttr spec = getSliceSpec(slice);
        if (!spec)
          continue;
        // The bounds live on the host, like the slice's own start and stop.
        OpBuilder outer(slice);
        auto host =
            cast<MO::TensorType>(slice.getStart().getType()).getDeviceRef();
        SmallVector<Value> operands{input};
        for (DenseI64ArrayAttr bound :
             {spec.getStarts(), spec.getStops(), spec.getSteps()})
          operands.push_back(MO::ConstantOp::create(outer, slice.getLoc(),
                                                    bound.asArrayRef(), host));
        kernel.insertInputsAt(first, operands);
        Block::BlockArgListType args =
            kernel.getInputBlockArgs().slice(first, operands.size());
        view = SliceStrideOp::create(builder, slice.getLoc(), viewTy, spec,
                                     args[0], args[1], args[2], args[3]);
      }
    }
    kernel.getInputBlockArgs()[idx].replaceAllUsesWith(view);
    ++numFused;
  }
  return numFused;
}

struct FuseContiguousViewsPass
    : public MAP::impl::FuseContiguousViewsBase<FuseContiguousViewsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Walk, since a kernel can sit inside control flow (e.g. mo.while).
    SmallVector<IterOpaqueOp> kernels;
    getOperation().walk([&](IterOpaqueOp op) { kernels.push_back(op); });

    unsigned fused = 0;
    for (IterOpaqueOp kernel : kernels)
      fused += fuseViewsInto(kernel);
    numViewsFused = fused;
  }
};

} // namespace

} // namespace M::MAP
