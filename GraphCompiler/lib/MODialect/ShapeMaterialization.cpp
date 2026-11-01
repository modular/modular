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
// Implementations of `Staticization` interface for various `mo` ops.
//
// TODO(GEX-4093): `getDataDependentInputIndices` is hand-maintained. Once a
// shape function takes only trait-bound tensor args, derive it from the
// declared bounds: a bare `Tensor` bound cannot read element values, so such
// inputs are not data-dependent.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/Support/Dims.h"
#include "Support/AssertStream.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> BatchMatmulOp::getDataDependentInputIndices() {
  return {};
}

ShapeMaterializeResult
BatchMatmulOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputA = inputAttrs[0];
  TypedAttr inputB = inputAttrs[1];

  if (inputAttrs.size() > 2) {
    [[maybe_unused]] auto transposeBAsBool =
        llvm::dyn_cast_or_null<mlir::BoolAttr>(inputAttrs[2]);
    ASSERT_STREAM(!transposeBAsBool || !transposeBAsBool.getValue(),
                  "Do not support staticization of transposeB.");
  }

  return materializeBatchMatmulShape(graphOp, loc, inputA, inputB);
}

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> BroadcastShapeOp::getDataDependentInputIndices() {
  return {};
}

ShapeMaterializeResult
BroadcastShapeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr lhsShape = inputAttrs[0];
  TypedAttr rhsShape = inputAttrs[1];
  return materializeBroadcastShapeShape(graphOp, loc, lhsShape, rhsShape);
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> BroadcastToOp::getDataDependentInputIndices() {
  return {1};
}

ShapeMaterializeResult
BroadcastToOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr targetShape = inputAttrs[1];
  return materializeBroadcastToShape(graphOp, loc, inputShape, targetShape);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ConcatOp::getDataDependentInputIndices() { return {}; }

ShapeMaterializeResult
ConcatOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();

  return materializeConcatShape(graphOp, loc, getAxisAttr(), inputAttrs);
}

//===----------------------------------------------------------------------===//
//

//===----------------------------------------------------------------------===//

SmallVector<int32_t> ConvOp::getDataDependentInputIndices() {
  return {2, 3, 4, 5};
}

ShapeMaterializeResult
ConvOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr filterShape = inputAttrs[1];
  TypedAttr strides = inputAttrs[2];
  TypedAttr dilations = inputAttrs[3];
  TypedAttr paddings = inputAttrs[4];
  TypedAttr numGroup = inputAttrs[5];
  StringAttr inputLayout = getInputLayoutAttr();
  LayoutAttr filterLayout = getFilter().getType().getLayoutAttr();
  return materializeConvShape(graphOp, loc, inputShape, filterShape, strides,
                              dilations, paddings, numGroup, inputLayout,
                              filterLayout);
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ConvTransposeOp::getDataDependentInputIndices() {
  return {2, 3, 4, 5};
}

ShapeMaterializeResult
ConvTransposeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr filterShape = inputAttrs[1];
  TypedAttr strides = inputAttrs[2];
  TypedAttr dilations = inputAttrs[3];
  TypedAttr paddings = inputAttrs[4];
  TypedAttr outPaddings = inputAttrs[5];
  StringAttr inputLayout = getInputLayoutAttr();
  LayoutAttr filterLayout = getFilter().getType().getLayoutAttr();
  return materializeConvTransposeShape(graphOp, loc, inputShape, filterShape,
                                       strides, dilations, paddings,
                                       outPaddings, inputLayout, filterLayout);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// `axis` is a compile-time `index` attribute, so no operand value is needed.
SmallVector<int32_t> GatherOp::getDataDependentInputIndices() { return {}; }

ShapeMaterializeResult
GatherOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr indicesShape = inputAttrs[1];
  return materializeGatherShape(graphOp, loc, inputShape, indicesShape,
                                getAxisAttr());
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

// `axis` is a compile-time `index` attribute, so no operand value is needed.
SmallVector<int32_t> ScatterOp::getDataDependentInputIndices() { return {}; }

ShapeMaterializeResult
ScatterOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr updatesShape = inputAttrs[1];
  TypedAttr indicesShape = inputAttrs[2];
  return materializeScatterShape(graphOp, loc, inputShape, updatesShape,
                                 indicesShape, getAxisAttr());
}

//===----------------------------------------------------------------------===//
// GatherNdOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> GatherNdOp::getDataDependentInputIndices() { return {}; }

ShapeMaterializeResult
GatherNdOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr indicesShape = inputAttrs[1];

  auto batchDims = getBatchDimsAttr().getInt();
  return materializeGatherNdShape(graphOp, loc, inputShape, indicesShape,
                                  batchDims);
}

//===----------------------------------------------------------------------===//
// Pad*Op
//===----------------------------------------------------------------------===//

SmallVector<int32_t> PadConstantOp::getDataDependentInputIndices() {
  return {1};
}
SmallVector<int32_t> PadReflectOp::getDataDependentInputIndices() {
  return {1};
}
SmallVector<int32_t> PadRepeatOp::getDataDependentInputIndices() { return {1}; }

static ShapeMaterializeResult
commonPadMaterializeShapeDefs(Operation *op, ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr paddings = inputAttrs[1];

  return materializePadShape(graphOp, loc, inputShape, paddings);
}

ShapeMaterializeResult
PadConstantOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPadMaterializeShapeDefs(getOperation(), inputAttrs);
}
ShapeMaterializeResult
PadReflectOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPadMaterializeShapeDefs(getOperation(), inputAttrs);
}
ShapeMaterializeResult
PadRepeatOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPadMaterializeShapeDefs(getOperation(), inputAttrs);
}

//===----------------------------------------------------------------------===//
// *PoolOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> MaxPoolOp::getDataDependentInputIndices() {
  return {1, 2, 3, 4};
}
SmallVector<int32_t> AvgPoolOp::getDataDependentInputIndices() {
  return {1, 2, 3, 4};
}
SmallVector<int32_t> MaxPoolCeilOp::getDataDependentInputIndices() {
  return {1, 2, 3, 4};
}
SmallVector<int32_t> AvgPoolCeilOp::getDataDependentInputIndices() {
  return {1, 2, 3, 4};
}

static ShapeMaterializeResult
commonPoolMaterializeShapeDefs(Operation *op, ArrayRef<TypedAttr> inputAttrs,
                               bool ceilMode) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr filter = inputAttrs[1];
  TypedAttr strides = inputAttrs[2];
  TypedAttr dilations = inputAttrs[3];
  TypedAttr paddings = inputAttrs[4];

  return materializePoolShape(graphOp, loc, inputShape, filter, strides,
                              dilations, paddings, ceilMode);
}

ShapeMaterializeResult
MaxPoolOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPoolMaterializeShapeDefs(getOperation(), inputAttrs, false);
}
ShapeMaterializeResult
AvgPoolOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPoolMaterializeShapeDefs(getOperation(), inputAttrs, false);
}
ShapeMaterializeResult
MaxPoolCeilOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPoolMaterializeShapeDefs(getOperation(), inputAttrs, true);
}
ShapeMaterializeResult
AvgPoolCeilOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return commonPoolMaterializeShapeDefs(getOperation(), inputAttrs, true);
}

//===----------------------------------------------------------------------===//
// RandomNormalOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> RandomNormalOp::getDataDependentInputIndices() {
  return {0};
}

ShapeMaterializeResult
RandomNormalOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  GenericAdaptor<ArrayRef<TypedAttr>> adaptor(inputAttrs);
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  return materializeTargetShapeLike(graphOp, loc, adaptor.getShape());
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> RandomUniformOp::getDataDependentInputIndices() {
  return {0};
}

ShapeMaterializeResult
RandomUniformOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  GenericAdaptor<ArrayRef<TypedAttr>> adaptor(inputAttrs);
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  return materializeTargetShapeLike(graphOp, loc, adaptor.getShape());
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> RangeOp::getDataDependentInputIndices() {
  return {0, 1, 2};
}

ShapeMaterializeResult
RangeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr startAttr = inputAttrs[0];
  TypedAttr stopAttr = inputAttrs[1];
  TypedAttr stepAttr = inputAttrs[2];
  return materializeRangeShape(graphOp, loc, startAttr, stopAttr, stepAttr);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ReshapeOp::getDataDependentInputIndices() { return {1}; }

ShapeMaterializeResult
ReshapeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr targetShape = inputAttrs[1];
  return materializeReshapeShape(graphOp, loc, inputShape, targetShape);
}

//===----------------------------------------------------------------------===//
// Resize*Op
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ResizeLinearOp::getDataDependentInputIndices() {
  return {1};
}
SmallVector<int32_t> ResizeNearestOp::getDataDependentInputIndices() {
  return {1};
}
SmallVector<int32_t> ResizeBicubicOp::getDataDependentInputIndices() {
  return {1};
}

ShapeMaterializeResult
ResizeLinearOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  return materializeTargetShapeLike(graphOp, loc, inputAttrs[1]);
}
ShapeMaterializeResult
ResizeNearestOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  return materializeTargetShapeLike(graphOp, loc, inputAttrs[1]);
}
ShapeMaterializeResult
ResizeBicubicOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  return materializeTargetShapeLike(graphOp, loc, inputAttrs[1]);
}

//===----------------------------------------------------------------------===//
// ROIAlignOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> RoiAlignOp::getDataDependentInputIndices() {
  return {2, 3};
}

ShapeMaterializeResult
RoiAlignOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr roisShape = inputAttrs[1];
  TypedAttr outHeight = inputAttrs[2];
  TypedAttr outWidth = inputAttrs[3];
  return materializeRoiAlignShape(graphOp, loc, inputShape, roisShape,
                                  outHeight, outWidth);
}

//===----------------------------------------------------------------------===//
// ScatterNdOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> ScatterNdOp::getDataDependentInputIndices() { return {}; }
SmallVector<int32_t> ScatterNdAddOp::getDataDependentInputIndices() {
  return {};
}
SmallVector<int32_t> ScatterNdMaxOp::getDataDependentInputIndices() {
  return {};
}
SmallVector<int32_t> ScatterNdMulOp::getDataDependentInputIndices() {
  return {};
}
SmallVector<int32_t> ScatterNdMinOp::getDataDependentInputIndices() {
  return {};
}

ShapeMaterializeResult
materializeCommonScatterNdShapeDefs(Operation *op,
                                    ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr updatesShape = inputAttrs[1];
  TypedAttr indicesShape = inputAttrs[2];
  return materializeScatterNdShape(graphOp, loc, inputShape, updatesShape,
                                   indicesShape);
}

ShapeMaterializeResult
ScatterNdOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return materializeCommonScatterNdShapeDefs(getOperation(), inputAttrs);
}

ShapeMaterializeResult
ScatterNdAddOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return materializeCommonScatterNdShapeDefs(getOperation(), inputAttrs);
}

ShapeMaterializeResult
ScatterNdMaxOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return materializeCommonScatterNdShapeDefs(getOperation(), inputAttrs);
}

ShapeMaterializeResult
ScatterNdMulOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return materializeCommonScatterNdShapeDefs(getOperation(), inputAttrs);
}

ShapeMaterializeResult
ScatterNdMinOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return materializeCommonScatterNdShapeDefs(getOperation(), inputAttrs);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> SliceOp::getDataDependentInputIndices() {
  return {1, 2, 3};
}

ShapeMaterializeResult
SliceOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr input = inputAttrs[0];
  TypedAttr starts = inputAttrs[1];
  TypedAttr stops = inputAttrs[2];
  TypedAttr steps = inputAttrs[3];
  return materializeSliceShape(graphOp, loc, input, starts, stops, steps);
}

//===----------------------------------------------------------------------===//
// SqueezeShapeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> SqueezeShapeOp::getDataDependentInputIndices() {
  return {};
}

ShapeMaterializeResult
SqueezeShapeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr indicesShape = inputAttrs[1];
  return materializeSqueezeShapeShape(graphOp, loc, inputShape, indicesShape);
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

// `axis` is a compile-time attribute; only `splitSizes` (operand 1) is
// data-dependent.
SmallVector<int32_t> SplitOp::getDataDependentInputIndices() { return {1}; }

ShapeMaterializeResult
SplitOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr splitSizes = inputAttrs[1];
  return materializeSplitOutputShapes(graphOp, loc, inputShape, splitSizes,
                                      getAxisAttr());
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> TileOp::getDataDependentInputIndices() { return {1}; }

ShapeMaterializeResult
TileOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr repeats = inputAttrs[1];
  return materializeTileShape(graphOp, loc, inputShape, repeats);
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> TopKOp::getDataDependentInputIndices() { return {1, 2}; }
SmallVector<int32_t> BottomKOp::getDataDependentInputIndices() {
  return {1, 2};
}

ShapeMaterializeResult
materializeCommonTopKShapeDefs(Operation *op, ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr k = inputAttrs[1];
  TypedAttr axis = inputAttrs[2];
  return materializeTopKShape(graphOp, loc, inputShape, k, axis);
}

ShapeMaterializeResult
TopKOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return materializeCommonTopKShapeDefs(getOperation(), inputAttrs);
}
ShapeMaterializeResult
BottomKOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  return materializeCommonTopKShapeDefs(getOperation(), inputAttrs);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> TransposeOp::getDataDependentInputIndices() { return {1}; }

ShapeMaterializeResult
TransposeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr permutations = inputAttrs[1];
  return materializeTransposeShape(graphOp, loc, inputShape, permutations);
}

//===----------------------------------------------------------------------===//
// UnsqueezeShapeOp
//===----------------------------------------------------------------------===//

SmallVector<int32_t> UnsqueezeShapeOp::getDataDependentInputIndices() {
  return {};
}

ShapeMaterializeResult
UnsqueezeShapeOp::materializeShapeDefs(ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = getOperation()->getParentOfType<GraphOp>();
  Location loc = getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr indicesShape = inputAttrs[1];
  return materializeUnsqueezeShapeShape(graphOp, loc, inputShape, indicesShape);
}

} // namespace M::MO
