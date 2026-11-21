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
//   %y = mo.composite.reduce.softmax(%x)          %xa = mo.cast(%x)
//                                           =>    %m = mo.reduce.max(%xa)
//                                                 %e = mo.exp(%xa - %m)
//                                                 %s = mo.reduce.add(%e)
//                                                 %y = %e * (1 / %s)
//
//   %q, %s = mo.composite                         %n = <rms_norm in f32>
//     .rms_norm_fused_quantize_              =>   %m = mo.reduce.max(abs(%n))
//     dynamic_scaled_fp8(%x, %w, %ub)             %s = min(%m, %ub) / fp8_max
//                                                 %q = cast(clamp(%n / %s))
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

// The same, from a float.
TensorValue scalar(RewriterBase &rewriter, Location loc, float value,
                   DType dtype, MO::TensorType like) {
  return scalar(
      rewriter, loc,
      cast<ElementsAttr>(FloatArrayElementsAttr::get(
          RankedTensorType::get({}, rewriter.getF32Type()), APFloat(value))),
      dtype, like);
}

// `value` on `device`: a constant is built there again, anything else is
// copied over. A null `device` keeps `value` where it is.
TensorValue onDevice(RewriterBase &rewriter, Location loc, TensorValue value,
                     DeviceRefAttr device) {
  if (!device || value.getType().getDeviceRef() == device)
    return value;
  if (auto constant = value.getDefiningOp<ConstantOp>())
    return ConstantOp::create(rewriter, loc, value.getType().cloneWith(device),
                              constant.getValue());
  return TransferOp::create(rewriter, loc, value, device,
                            ChainCreateOp::create(rewriter, loc, ValueRange{}),
                            /*alwaysElideSameDeviceCopy=*/false)
      .getResult();
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
  return onDevice(rewriter, loc, value, device);
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

// xa * rsqrt(reduce.add(xa * xa) / N + epsilon), the row normalized in
// `xa`'s dtype, before the weight.
TensorValue normalizeRows(RewriterBase &rewriter, Location loc, TensorValue xa,
                          ElementsAttr epsilon) {
  MO::TensorType type = xa.getType();
  DType accum = type.getDtype();
  TensorValue sum = ReduceAddOp::create(
      rewriter, loc, MulOp::create(rewriter, loc, xa, xa), /*axis=*/-1);
  MOSH::ShapeAttr rowShape = sum.getType().getShape();
  TensorValue meanSquare = DivOp::create(
      rewriter, loc, sum,
      broadcastTo(rewriter, loc, lastDimExtent(rewriter, loc, type, accum),
                  rowShape));
  TensorValue variance = AddOp::create(
      rewriter, loc, meanSquare,
      broadcastTo(rewriter, loc, scalar(rewriter, loc, epsilon, accum, type),
                  rowShape));
  TensorValue inv =
      RsqrtOp::create(rewriter, loc, variance.getType(), variance);
  return MulOp::create(rewriter, loc, xa,
                       broadcastTo(rewriter, loc, inv, type.getShape()));
}

// (cast(w, dtype) + cast(offset, dtype)), broadcast over `like`'s rows.
TensorValue offsetWeight(RewriterBase &rewriter, Location loc,
                         TensorValue weight, ElementsAttr offset, DType dtype,
                         MO::TensorType like) {
  MOSH::ShapeAttr weightShape = weight.getType().getShape();
  TensorValue gamma = AddOp::create(
      rewriter, loc, castTo(rewriter, loc, weight, dtype),
      broadcastTo(rewriter, loc, scalar(rewriter, loc, offset, dtype, like),
                  weightShape));
  return broadcastTo(rewriter, loc, gamma, like.getShape());
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

    TensorValue normed = normalizeRows(
        rewriter, loc, castTo(rewriter, loc, x, accum), op.getEpsilon());
    DType weightDType = op.getMultiplyBeforeCast() ? accum : dtype;
    TensorValue gamma = offsetWeight(rewriter, loc, op.getWeight(),
                                     op.getWeightOffset(), weightDType, type);
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

// As the hand-written softmax computes it, in `accum` = get_accum_type[dtype]:
//
//   xa  = cast(x, accum)
//   e   = exp(xa - reduce.max(xa))
//   y   = cast(e * (1 / reduce.add(e)), dtype)     the divide once per row
struct DecomposeSoftmax final
    : public OpRewritePattern<Composite::ReduceSoftmaxOp> {
  DecomposeSoftmax(MLIRContext *ctx, const MOGG::MojoRegistry &registry)
      : OpRewritePattern(ctx), registry(registry) {}

  LogicalResult matchAndRewrite(Composite::ReduceSoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    if (hasKernel(registry, op))
      return rewriter.notifyMatchFailure(op, "a kernel implements it");
    Location loc = op.getLoc();
    TensorValue x = op.getInput();
    MO::TensorType type = x.getType();
    DType dtype = type.getDtype();
    DType accum = accumDType(dtype);
    int64_t axis = op.getAxisAttr().getInt();

    TensorValue xa = castTo(rewriter, loc, x, accum);
    TensorValue max = ReduceMaxOp::create(rewriter, loc, xa, axis);
    TensorValue shifted = SubOp::create(
        rewriter, loc, xa, broadcastTo(rewriter, loc, max, type.getShape()));
    TensorValue e = ExpOp::create(rewriter, loc, shifted.getType(), shifted);
    TensorValue sum = ReduceAddOp::create(rewriter, loc, e, axis);
    TensorValue one = castTo(
        rewriter, loc,
        ConstantOp::create(rewriter, loc, 1.0f, type.getDeviceRef()), accum);
    TensorValue recip = DivOp::create(
        rewriter, loc,
        broadcastTo(rewriter, loc, one, sum.getType().getShape()), sum);
    TensorValue result = castTo(
        rewriter, loc,
        MulOp::create(rewriter, loc, e,
                      broadcastTo(rewriter, loc, recip, type.getShape())),
        dtype);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }

  const MOGG::MojoRegistry &registry;
};

// The FP8 quantize of `xa`, one scale per row, as the hand-written kernels
// compute it, in `xa`'s dtype `accum` and the scales dtype `sdt`:
//
//   scale = min(cast(reduce.max(abs(xa)), sdt), cast(ub, sdt)) / fp8_max
//   inv   = 1 / cast(scale, accum), or 0 where that is not finite
//   q     = cast(clamp(xa * inv, -fp8_max, fp8_max), out)
//
// Returns `q` and the `[..., 1]` scales.
std::pair<TensorValue, TensorValue>
quantizePerRow(RewriterBase &rewriter, Location loc, TensorValue xa,
               TensorValue scaleUb, MO::TensorType outType, DType scalesDType) {
  MO::TensorType type = xa.getType();
  DType accum = type.getDtype();
  float fp8Max =
      APFloat::getLargest(
          cast<FloatType>(outType.getElementType()).getFloatSemantics())
          .convertToFloat();

  TensorValue max = ReduceMaxOp::create(
      rewriter, loc, AbsOp::create(rewriter, loc, type, xa), /*axis=*/-1);
  MOSH::ShapeAttr rowShape = max.getType().getShape();
  TensorValue ub = castTo(rewriter, loc,
                          onDevice(rewriter, loc, scaleUb, type.getDeviceRef()),
                          scalesDType);
  TensorValue scale = DivOp::create(
      rewriter, loc,
      MinOp::create(rewriter, loc, castTo(rewriter, loc, max, scalesDType),
                    broadcastTo(rewriter, loc, ub, rowShape)),
      broadcastTo(rewriter, loc,
                  scalar(rewriter, loc, fp8Max, scalesDType, type), rowShape));

  TensorValue inv = DivOp::create(
      rewriter, loc,
      broadcastTo(rewriter, loc, scalar(rewriter, loc, 1.0f, accum, type),
                  rowShape),
      castTo(rewriter, loc, scale, accum));
  MO::TensorType maskType = inv.getType().cloneWith(DType::kBool);
  TensorValue notFinite =
      OrOp::create(rewriter, loc, IsInfOp::create(rewriter, loc, maskType, inv),
                   IsNaNOp::create(rewriter, loc, maskType, inv));
  inv = SelectOp::create(rewriter, loc, inv.getType(), notFinite,
                         broadcastTo(rewriter, loc,
                                     scalar(rewriter, loc, 0.0f, accum, type),
                                     rowShape),
                         inv);

  TensorValue scaled = MulOp::create(
      rewriter, loc, xa, broadcastTo(rewriter, loc, inv, type.getShape()));
  TensorValue clamped = MinOp::create(
      rewriter, loc,
      MaxOp::create(rewriter, loc, scaled,
                    broadcastTo(rewriter, loc,
                                scalar(rewriter, loc, -fp8Max, accum, type),
                                type.getShape())),
      broadcastTo(rewriter, loc, scalar(rewriter, loc, fp8Max, accum, type),
                  type.getShape()));
  return {castTo(rewriter, loc, clamped, outType.getDtype()), scale};
}

// rms_norm (multiply_before_cast) then a per-token FP8 quantize, as the
// hand-written fused kernel computes them: the normalized row stays in
// `accum` = get_accum_type[dtype] between the two.
//
//   xa = cast(x, accum)
//   n  = xa * inv * (cast(w, accum) + offset)                as rms_norm
//   q, scale = quantizePerRow(n)
//
// The scales are the per-row scale, reshaped to the op's scales type.
struct DecomposeRMSNormFusedQuantize final
    : public OpRewritePattern<
          Composite::RMSNormFusedQuantizeDynamicScaledFP8Op> {
  DecomposeRMSNormFusedQuantize(MLIRContext *ctx,
                                const MOGG::MojoRegistry &registry)
      : OpRewritePattern(ctx), registry(registry) {}

  LogicalResult
  matchAndRewrite(Composite::RMSNormFusedQuantizeDynamicScaledFP8Op op,
                  PatternRewriter &rewriter) const override {
    if (hasKernel(registry, op))
      return rewriter.notifyMatchFailure(op, "a kernel implements it");
    Location loc = op.getLoc();
    TensorValue x = cast<TensorValue>(op.getInput());
    MO::TensorType type = x.getType();
    DType accum = accumDType(type.getDtype());

    TensorValue normed = MulOp::create(
        rewriter, loc,
        normalizeRows(rewriter, loc, castTo(rewriter, loc, x, accum),
                      op.getEpsilon()),
        offsetWeight(rewriter, loc, cast<TensorValue>(op.getWeight()),
                     op.getWeightOffset(), accum, type));
    auto scalesType = cast<MO::TensorType>(op.getScale().getType());
    auto [output, scale] = quantizePerRow(
        rewriter, loc, normed, cast<TensorValue>(op.getScaleUb()),
        cast<MO::TensorType>(op.getOutput().getType()), scalesType.getDtype());
    rewriter.replaceOp(op, {output, Static::ReshapeOp::create(
                                        rewriter, loc, scalesType, scale)});
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
    patterns
        .add<DecomposeRMSNorm, DecomposeRMSNormFusedQuantize, DecomposeSoftmax>(
            &getContext(), registry);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
