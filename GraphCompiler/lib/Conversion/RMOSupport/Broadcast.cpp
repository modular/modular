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

#include "GraphCompiler/Conversion/RMOSupport/Broadcast.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/RMODialect/RMOOps.h"
#include "Support/AssertStream.h"

using M::MO::ShapeAttr;
using M::MO::ShapeType;
using M::MO::TensorType;
using M::MO::TensorValue;

namespace M::RMO {

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from, TensorType reshapeType,
                                  TensorValue to) {
  Type elemType = builder.getIntegerType(64, /*isSigned=*/true);

  ASSERT_STREAM(
      reshapeType.hasRank(),
      << "reshapeType needs to be a statically ranked tensor type but it is "
      << reshapeType);
  auto shapeType = TensorType::get({(int64_t)reshapeType.getRank()}, elemType,
                                   reshapeType.getDeviceRef());

  auto valueAttr = builder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int64_t>(reshapeType.getIntDims()), mlir::IntegerType::Signed);

  auto reshapeShapeValue = MO::ConstantOp::create(
      builder, loc, shapeType, cast<ElementsAttr>(valueAttr));

  auto reshapedValue = RMO::MOReshapeOp::create(builder, loc, reshapeType, from,
                                                reshapeShapeValue);

  return broadcastTo(builder, loc, reshapedValue, to);
}

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from,
                                  ArrayRef<int64_t> reshapeShape,
                                  TensorValue to) {
  TensorType type = from.getType();
  TensorType reshapeType =
      TensorType::get(reshapeShape, type.getElementType(), type.getDeviceRef());
  return reshapeAndBroadcastTo(builder, loc, from, reshapeType, to);
}

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from, TensorType reshapeType,
                                  TensorValue reshapeShape, TensorValue to) {
  auto reshapedValue =
      RMO::MOReshapeOp::create(builder, loc, reshapeType, from, reshapeShape);
  return broadcastTo(builder, loc, reshapedValue, to);
}

TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                  TensorValue from,
                                  ArrayRef<int64_t> reshapeDims,
                                  TensorValue reshapeShape, TensorValue to) {
  auto reshapedValue = RMO::MOReshapeOp::create(
      builder, loc,
      MO::TensorType::get(reshapeDims, from.getType().getElementType(),
                          from.getType().getDeviceRef()),
      from, reshapeShape);
  return broadcastTo(builder, loc, reshapedValue, to);
}

/// Makes tensor types before and after broadcast comparable.
[[maybe_unused]] static TensorType unifyShapeType(TensorType type) {
  if (!type.hasRank())
    return type;
  SmallVector<TypedAttr> result{type.getDims()};
  ShapeType shapeTy = type.getShape().getType();
  auto shapeAttr = ShapeAttr::get(type.getContext(), result, shapeTy);
  return type.cloneWith(shapeAttr);
}

/// Broadcast `x` and `y` to a common dynamic shape given by `bcastCommonShape`.
/// All inputs must have a static rank.
static std::pair<TensorValue, TensorValue>
broadcastToDynamicShape(OpBuilder &builder, TensorValue x, TensorValue y,
                        ShapeAttr bcastCommonShape) {
  TensorType xType = x.getType();
  assert(xType.hasRank());

  TensorType yType = y.getType();
  assert(yType.hasRank());

  DeviceRefAttr deviceRef = xType.getDeviceRef();
  if (!deviceRef)
    deviceRef = yType.getDeviceRef();
  else if (DeviceRefAttr yDeviceRef = yType.getDeviceRef())
    assert(deviceRef == yDeviceRef);

  // We propagate device refs explicitly to keep shapes on the same device.
  auto intType = builder.getIntegerType(64, /*isSigned=*/true);

  auto xShapeType =
      TensorType::get({(int64_t)xType.getRank()}, intType, deviceRef);
  auto xShape = RMO::MOShapeOfOp::create(builder, x.getLoc(), xShapeType, x);

  auto yShapeType =
      TensorType::get({(int64_t)yType.getRank()}, intType, deviceRef);
  auto yShape = RMO::MOShapeOfOp::create(builder, y.getLoc(), yShapeType, y);

  auto bcastShapeType = TensorType::get({(int64_t)bcastCommonShape.getRank()},
                                        intType, deviceRef);
  auto bcastShape =
      RMO::MOBroadcastShapeOp::create(builder, x.getLoc(), bcastShapeType,
                                      xShape.getShape(), yShape.getShape());

  auto bcastXOp = RMO::MOBroadcastToOp::create(
      builder, x.getLoc(), x.getType().cloneWith(bcastCommonShape), x,
      bcastShape);
  auto bcastYOp = RMO::MOBroadcastToOp::create(
      builder, y.getLoc(), y.getType().cloneWith(bcastCommonShape), y,
      bcastShape);
  return {bcastXOp, bcastYOp};
}

/// Broadcast `x`, `y` and `z` to a common dynamic shape given by
/// `bcastCommonShape`.
/// All inputs must have a static rank.
static std::tuple<TensorValue, TensorValue, TensorValue>
broadcastToDynamicShape(OpBuilder &builder, TensorValue x, TensorValue y,
                        TensorValue z, ShapeAttr xyCommonShape,
                        ShapeAttr bcastCommonShape) {
  TensorType xType = x.getType();
  assert(xType.hasRank());
  TensorType yType = y.getType();
  assert(yType.hasRank());
  TensorType zType = z.getType();
  assert(zType.hasRank());

  DeviceRefAttr deviceRef = xType.getDeviceRef();
  assert(deviceRef == yType.getDeviceRef());
  assert(deviceRef == zType.getDeviceRef());

  auto intType = builder.getIntegerType(64, /*isSigned=*/true);
  auto xyShape =
      TensorType::get({(int64_t)xyCommonShape.getRank()}, intType, deviceRef);
  auto shape = TensorType::get({(int64_t)bcastCommonShape.getRank()}, intType,
                               deviceRef);

  auto xShape = RMO::MOShapeOfOp::create(
      builder, x.getLoc(),
      TensorType::get({(int64_t)xType.getRank()}, intType, deviceRef), x);

  auto yShape = RMO::MOShapeOfOp::create(
      builder, y.getLoc(),
      TensorType::get({(int64_t)yType.getRank()}, intType, deviceRef), y);

  auto zShape = RMO::MOShapeOfOp::create(
      builder, z.getLoc(),
      TensorType::get({(int64_t)zType.getRank()}, intType, deviceRef), z);

  auto bcastXYShape = RMO::MOBroadcastShapeOp::create(
      builder, x.getLoc(), xyShape, xShape.getShape(), yShape.getShape());
  auto bcastShape = RMO::MOBroadcastShapeOp::create(
      builder, x.getLoc(), shape, bcastXYShape.getShape(), zShape.getShape());

  auto bcastXOp = RMO::MOBroadcastToOp::create(
      builder, x.getLoc(), x.getType().cloneWith(bcastCommonShape), x,
      bcastShape);
  auto bcastYOp = RMO::MOBroadcastToOp::create(
      builder, y.getLoc(), y.getType().cloneWith(bcastCommonShape), y,
      bcastShape);
  auto bcastZOp = RMO::MOBroadcastToOp::create(
      builder, z.getLoc(), z.getType().cloneWith(bcastCommonShape), z,
      bcastShape);
  return {bcastXOp, bcastYOp, bcastZOp};
}

TensorValue broadcastToStaticShape(OpBuilder &builder, Location loc,
                                   TensorValue x, ShapeAttr s) {
  auto constOp = MO::ConstantOp::create(builder, loc, s.getIntDims(),
                                        x.getType().getDeviceRef());
  return RMO::MOBroadcastToOp::create(builder, loc, x.getType().cloneWith(s), x,
                                      constOp);
}

std::pair<TensorValue, TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, TensorValue x,
                          TensorValue y) {
  TensorType xShape = x.getType();
  TensorType yShape = y.getType();
  if ((xShape == yShape) && (xShape.hasStaticShape()))
    return {x, y};

  ASSERT_STREAM(xShape.hasRank() && yShape.hasRank(),
                << "broadcasting does not support dynamic rank");

  ErrorOr<ShapeAttr> errorOrBcastCommonShape =
      ShapeAttr::broadcastShapes(xShape.getShape(), yShape.getShape());

  ASSERT_STREAM(!errorOrBcastCommonShape.isError(),
                << "the shapes are incompatible: "
                << errorOrBcastCommonShape.takeError());

  ShapeAttr bcastCommonShape = *errorOrBcastCommonShape;

  if (!bcastCommonShape.isStatic())
    return broadcastToDynamicShape(builder, x, y, bcastCommonShape);

  // Static broadcast, can potentially elide some broadcasts.
  if (xShape.getShape() != bcastCommonShape)
    x = broadcastToStaticShape(builder, loc, x, bcastCommonShape);
  if (yShape.getShape() != bcastCommonShape)
    y = broadcastToStaticShape(builder, loc, y, bcastCommonShape);
  return {x, y};
}

std::tuple<TensorValue, TensorValue, TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, TensorValue x,
                          TensorValue y, TensorValue z) {
  TensorType xShape = x.getType();
  TensorType yShape = y.getType();
  TensorType zShape = z.getType();
  if ((xShape == yShape) && (yShape == zShape) && (xShape.hasStaticShape()))
    return {x, y, z};

  ASSERT_STREAM(xShape.hasRank() && yShape.hasRank() && zShape.hasRank(),
                << "broadcasting does not support dynamic rank");

  ErrorOr<ShapeAttr> errorOrXYBcastCommonShape =
      ShapeAttr::broadcastShapes(xShape.getShape(), yShape.getShape());
  ASSERT_STREAM(!errorOrXYBcastCommonShape.isError(),
                << "the shapes are incompatible: "
                << errorOrXYBcastCommonShape.takeError());
  ShapeAttr xyCommonShape = *errorOrXYBcastCommonShape;

  ErrorOr<ShapeAttr> errorOrBcastCommonShape =
      ShapeAttr::broadcastShapes(*errorOrXYBcastCommonShape, zShape.getShape());

  ASSERT_STREAM(!errorOrBcastCommonShape.isError(),
                << "the shapes are incompatible: "
                << errorOrBcastCommonShape.takeError());

  ShapeAttr bcastCommonShape = *errorOrBcastCommonShape;

  if (!bcastCommonShape.isStatic())
    return broadcastToDynamicShape(builder, x, y, z, xyCommonShape,
                                   bcastCommonShape);

  // Static broadcast, can potentially elide some broadcasts.
  if (xShape.getShape() != bcastCommonShape)
    x = broadcastToStaticShape(builder, loc, x, bcastCommonShape);
  if (yShape.getShape() != bcastCommonShape)
    y = broadcastToStaticShape(builder, loc, y, bcastCommonShape);
  if (zShape.getShape() != bcastCommonShape)
    z = broadcastToStaticShape(builder, loc, z, bcastCommonShape);
  return {x, y, z};
}

TensorValue broadcastTo(OpBuilder &builder, Location loc, TensorValue x,
                        TensorValue y) {
  auto [newX, newY] = reifyImplicitBroadcasting(builder, loc, x, y);

  ASSERT_STREAM(
      newY.getType() == unifyShapeType(y.getType()),
      << "don't use this helper unless you can guarantee x broadcasts to y");

  return newX;
}

TensorValue broadcastFloatTo(OpBuilder &builder, Location loc,
                             const APFloat &scalar, TensorValue toVal) {
  auto toTy = cast<TensorType>(toVal.getType());

  auto graphOp = cast<MO::GraphOp>(toVal.getParentBlock()->getParentOp());
  auto toShape = RMO::MOShapeOfOp::create(builder, loc, toVal, graphOp);

  auto scalarConst = MO::ConstantOp::create(
      builder, loc, scalar, toTy.getElementType(), toTy.getDeviceRef());
  return RMO::MOBroadcastToOp::create(builder, loc, toTy, scalarConst, toShape);
}

SmallVector<MO::TensorValue>
broadcastToCompatibleShape(SmallVector<MO::TensorValue> &values,
                           OpBuilder &rewriter) {
  Operation *op = values[0].getDefiningOp();
  auto graphOp = op->getParentOfType<MO::GraphOp>();
  MO::TensorValue commonShape = RMO::MOShapeOfOp::create(
      rewriter, op->getLoc(), ValueRange{values[0]}, graphOp);

  SmallVector<MO::TensorValue> answer(values.size());
  for (size_t i = 1; i < values.size(); ++i) {
    MO::TensorValue curShape = RMO::MOShapeOfOp::create(
        rewriter, op->getLoc(), ValueRange{values[i]}, graphOp);
    commonShape = RMO::MOBroadcastShapeOp::create(
        rewriter, op->getLoc(), ValueRange{commonShape, curShape}, graphOp);
  }
  for (size_t i = 0; i < values.size(); ++i) {
    MO::TensorValue cur = values[i];
    cur = RMO::MOBroadcastToOp::create(rewriter, op->getLoc(),
                                       ValueRange{cur, commonShape}, graphOp);
    answer[i] = cur;
  }
  return answer;
}

} // namespace M::RMO
