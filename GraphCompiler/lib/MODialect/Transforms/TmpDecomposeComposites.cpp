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
// Replaces each composite that no kernel implements on its device with the
// ops it is made of, computed as its hand-written kernel computes them, so the
// graph compiler fuses them into a kernel of its own. A composite a target
// implements keeps running that kernel:
//
//   %y = mo.composite.reduce.rms_norm(%x, %w)     %xa = mo.cast(%x)
//                                           =>    %s = mo.reduce.add(%xa * %xa)
//                                                 %inv = mo.rsqrt(%s / N + eps)
//                                                 %y = %xa * %inv * %w
//
// Temporary, until no pattern depends on a composite (rms_norm + rope, for
// example).
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "Support/MDialect/MAttrs.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace M::MO {

#define GEN_PASS_DEF_TMPDECOMPOSECOMPOSITES
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;
using namespace mlir;

namespace {

// The dtype a float reduction accumulates in, as Mojo's `get_accum_type`:
// f32 for a narrower float.
DType accumDType(DType dtype) {
  DType f32 = DType::f32;
  return dtype.isFloat() && dtype.getValue() < f32.getValue() ? f32 : dtype;
}

TensorValue castTo(RewriterBase &rewriter, Location loc, TensorValue value,
                   DType dtype) {
  if (value.getType().getDtype() == dtype)
    return value;
  return CastOp::create(rewriter, loc, value, dtype);
}

TensorValue broadcastTo(RewriterBase &rewriter, Location loc, TensorValue value,
                        MOSH::ShapeAttr shape) {
  if (value.getType().getShape() == shape)
    return value;
  return Static::BroadcastToOp::create(rewriter, loc,
                                       value.getType().cloneWith(shape), value);
}

// The rank-0 f32 constant `value`, in `dtype`, on `like`'s device.
TensorValue scalar(RewriterBase &rewriter, Location loc, ElementsAttr value,
                   DType dtype, MO::TensorType like) {
  TensorValue f32 = ConstantOp::create(
      rewriter, loc,
      MO::TensorType::getScalarType(rewriter.getF32Type(), like.getDeviceRef(),
                                    like.getMetadata()),
      value);
  return castTo(rewriter, loc, f32, dtype);
}

// The extent of `type`'s last dim as a rank-0 `dtype` tensor on its device: a
// constant when static, else read from the shape on the host and copied over.
TensorValue lastDimExtent(RewriterBase &rewriter, Location loc,
                          MO::TensorType type, DType dtype) {
  DeviceRefAttr device = type.getDeviceRef();
  int64_t extent = type.getIntDim(type.getRank() - 1);
  if (!ShapedType::isDynamic(extent))
    return castTo(
        rewriter, loc,
        ConstantOp::create(rewriter, loc, static_cast<float>(extent), device),
        dtype);

  // A tensor with no device stays without one.
  DeviceRefAttr host =
      device ? getHostDevice(rewriter.getContext()) : DeviceRefAttr();
  Value index = MOSH::ParamToValueOp::create(rewriter, loc,
                                             type.getDim(type.getRank() - 1));
  TensorValue value =
      castTo(rewriter, loc,
             IndexToTensorOp::create(
                 rewriter, loc,
                 MO::TensorType::get(ArrayRef<int64_t>{},
                                     IntegerType::get(rewriter.getContext(), 64,
                                                      IntegerType::Unsigned),
                                     host, type.getMetadata()),
                 index),
             dtype);
  if (device == host)
    return value;
  return TransferOp::create(rewriter, loc, value, device,
                            ChainCreateOp::create(rewriter, loc, ValueRange{}),
                            /*alwaysElideSameDeviceCopy=*/false)
      .getResult();
}

// TODO(GEX-3346): a target that wants a composite whole should register a
// custom op for it instead, and this check goes away with the pass.
bool hasKernel(const MOGG::MojoRegistry &registry, Operation *op) {
  StringRef name = op->getName().getStringRef();
  return registry.containsKernel(name) &&
         !registry
              .getResolvedKernel(name,
                                 MOGG::MojoRegistry::getDeviceRefForOp(op))
              .isError();
}

// As the hand-written rms_norm computes it, in `accum` = get_accum_type[dtype]:
//
//   xa  = cast(x, accum)
//   inv = rsqrt(reduce.add(xa * xa) / N + epsilon)                 per row
//   multiply_before_cast: cast(xa * inv * (cast(w, accum) + offset), dtype)
//   otherwise:            cast(xa * inv, dtype) * (w + cast(offset, dtype))
struct DecomposeRMSNorm final
    : public OpRewritePattern<Composite::ReduceRMSNormOp> {
  DecomposeRMSNorm(MLIRContext *ctx, const MOGG::MojoRegistry &registry)
      : OpRewritePattern(ctx), registry(registry) {}

  LogicalResult matchAndRewrite(Composite::ReduceRMSNormOp op,
                                PatternRewriter &rewriter) const override {
    if (hasKernel(registry, op))
      return rewriter.notifyMatchFailure(op, "a kernel implements it");
    Location loc = op.getLoc();
    TensorValue x = op.getInput();
    MO::TensorType type = x.getType();
    DType dtype = type.getDtype();
    DType accum = accumDType(dtype);

    TensorValue xa = castTo(rewriter, loc, x, accum);
    TensorValue sum = ReduceAddOp::create(
        rewriter, loc, MulOp::create(rewriter, loc, xa, xa), /*axis=*/-1);
    MOSH::ShapeAttr rowShape = sum.getType().getShape();
    TensorValue meanSquare = DivOp::create(
        rewriter, loc, sum,
        broadcastTo(rewriter, loc, lastDimExtent(rewriter, loc, type, accum),
                    rowShape));
    TensorValue variance = AddOp::create(
        rewriter, loc, meanSquare,
        broadcastTo(rewriter, loc,
                    scalar(rewriter, loc, op.getEpsilon(), accum, type),
                    rowShape));
    TensorValue inv =
        RsqrtOp::create(rewriter, loc, variance.getType(), variance);
    TensorValue normed = MulOp::create(
        rewriter, loc, xa, broadcastTo(rewriter, loc, inv, type.getShape()));

    TensorValue weight = op.getWeight();
    MOSH::ShapeAttr weightShape = weight.getType().getShape();
    DType weightDType = op.getMultiplyBeforeCast() ? accum : dtype;
    TensorValue gamma =
        AddOp::create(rewriter, loc, castTo(rewriter, loc, weight, weightDType),
                      broadcastTo(rewriter, loc,
                                  scalar(rewriter, loc, op.getWeightOffset(),
                                         weightDType, type),
                                  weightShape));
    gamma = broadcastTo(rewriter, loc, gamma, type.getShape());
    TensorValue result =
        op.getMultiplyBeforeCast()
            ? castTo(rewriter, loc, MulOp::create(rewriter, loc, normed, gamma),
                     dtype)
            : TensorValue(MulOp::create(
                  rewriter, loc, castTo(rewriter, loc, normed, dtype), gamma));
    rewriter.replaceOp(op, result);
    return mlir::success();
  }

  const MOGG::MojoRegistry &registry;
};

struct TmpDecomposeCompositesPass final
    : public MO::impl::TmpDecomposeCompositesBase<TmpDecomposeCompositesPass> {
  using Base::Base;

  void runOnOperation() override {
    MOGG::MojoRegistry registry(
        getOperation()->getParentOfType<mlir::ModuleOp>());
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeRMSNorm>(&getContext(), registry);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
