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
// Utilities for materializing shape functions as parametric expressions, used
// by GraphAPI and Graph Compiler shape inference.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "Support/AssertStream.h"

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H

namespace M::MO {
class Var;

/// Builds parameter expressions, recording each non-trivial one as a
/// `ParamBinding` in `createdOps`. `graphOp` generates parameter names.
class ParamExprBuilder {
public:
  ParamExprBuilder(GraphOp graphOp, Location loc,
                   SmallVectorImpl<ParamBinding> *createdOps)
      : graphOp(graphOp), loc(loc), createdOps(createdOps) {}

  /// `Var` holds a pointer to its builder.
  ParamExprBuilder(const ParamExprBuilder &other) = delete;
  ParamExprBuilder(ParamExprBuilder &&other) = delete;

  Var createVar(TypedAttr attr);

  /// Wraps the constant `n` of DimType.
  Var createVar(int64_t n);

  /// Builds `<poc(operands...)>`. If it doesn't fold to a trivial attribute
  /// (parameter reference or constant), tracks a binding for
  /// `kgen.param.declare declPrefix... = <poc(operands...)>` and returns a
  /// reference to it.
  TypedAttr createKgenExpr(KGEN::POC poc, ArrayRef<TypedAttr> operands,
                           StringRef declPrefix = "expr");

  /// Returns `attr` if trivial (parameter reference or constant); otherwise
  /// tracks a binding `kgen.param.declare declPrefix... = attr` and returns a
  /// reference to it.
  TypedAttr wrapNonTrivialAttrWithDecl(TypedAttr attr,
                                       StringRef declPrefix = "expr");

  /// Tracks `kgen.param.declare ... = <...assert(condVar, errorMsg)...>`.
  void createAssert(const Var &condVar, StringRef errorMsg);

  /// Tracks `kgen.param.declare ... = <cond(condV, thenV, elseV)>`.
  Var createCond(const Var &condV, const Var &thenV, const Var &elseV);

  /// Tracks `kgen.param.declare ... = <max(lhs, rhs)>`.
  Var createMax(const Var &lhs, const Var &rhs);

  /// Tracks `kgen.param.declare ... = <min(lhs, rhs)>`.
  Var createMin(const Var &lhs, const Var &rhs);

  template <class... V, typename = std::enable_if_t<(true && ... &&
                                                     std::is_same_v<V, Var>)>>
  ShapeAttr getShapeAttr(const V &...dim) {
    return getShapeAttr({dim.getAttr()...});
  }

  ShapeAttr getShapeAttr(ArrayRef<TypedAttr> dimAttrs);

private:
  GraphOp graphOp;
  Location loc;
  SmallVectorImpl<ParamBinding> *createdOps;
};

/// Wraps a parameter attribute to build expressions via operator overloading.
///
/// Example:
///   ParamExprBuilder b(...);
///   Var x = b.createVar(xAttr);
///   Var y = (x * x) / 2;
///   TypedAttr yAttr = y.getAttr();
class Var {
private:
  // Either constant IntegerAttr or DeclRef, to minimize expression nesting and
  // make debugging possible (otherwise expressions will fold by construction).
  TypedAttr attr;
  ParamExprBuilder *builder;

  Var(TypedAttr attr, ParamExprBuilder *builder)
      : attr(attr), builder(builder) {
    ASSERT_STREAM(
        builder, << "Var should be constructed with non-null builder");
  }

  friend class ParamExprBuilder;

public:
  Var() = delete;
  Var(const Var &other) = default;
  Var &operator=(const Var &other) = default;
  Var(Var &&other) = default;
  Var &operator=(Var &&other) = default;

  //===--------------------------------------------------------------------===//
  // Arithmetic
  //===--------------------------------------------------------------------===//

  Var operator+(TypedAttr otherAttr) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Add, attr, otherAttr),
        "var_add");
    return Var(resDeclRef, builder);
  }

  Var operator+(const Var &other) const { return *this + other.attr; }

  Var &operator+=(TypedAttr otherAttr) {
    *this = *this + otherAttr;
    return *this;
  }

  Var &operator+=(const Var &other) { return *this += other.attr; }

  Var operator+(int64_t other) const {
    return *this + builder->createVar(other);
  }

  Var operator-(const Var &other) const { return *this + (other * -1); }

  Var operator-=(const Var &other) {
    *this = *this - other;
    return *this;
  }

  Var operator-(int64_t other) const {
    return *this - builder->createVar(other);
  }

  Var operator-() const { return *this * -1; }

  Var operator*(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap, attr, other.attr),
        "var_mul");
    return Var(resDeclRef, builder);
  }

  Var operator*(int64_t other) const {
    return *this * builder->createVar(other);
  }

  Var operator/(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Div, attr, other.attr),
        "var_div");
    return Var(resDeclRef, builder);
  }

  Var operator/(int64_t other) const {
    return *this / builder->createVar(other);
  }

  Var operator%(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Mod, attr, other.attr),
        "var_mod");
    return Var(resDeclRef, builder);
  }

  Var operator%(int64_t other) const {
    return *this % builder->createVar(other);
  }

  //===--------------------------------------------------------------------===//
  // Comparison
  //===--------------------------------------------------------------------===//

  Var operator!=(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::getNE(attr, other.attr), "var_ne");
    return Var(resDeclRef, builder);
  }

  Var operator==(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::EQ, {attr, other.attr}),
        "var_eq");
    return Var(resDeclRef, builder);
  }

  Var operator<(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::LT, attr, other.attr),
        "var_lt");
    return Var(resDeclRef, builder);
  }

  Var operator<=(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::LE, attr, other.attr),
        "var_le");
    return Var(resDeclRef, builder);
  }

  Var operator>(const Var &other) const { return other < *this; }

  Var operator>=(const Var &other) const { return other <= *this; }

  Var operator!=(int64_t other) const {
    return *this != builder->createVar(other);
  }

  Var operator==(int64_t other) const {
    return *this == builder->createVar(other);
  }

  Var operator<(int64_t other) const {
    return *this < builder->createVar(other);
  }

  Var operator<=(int64_t other) const {
    return *this <= builder->createVar(other);
  }

  Var operator>(int64_t other) const {
    return *this > builder->createVar(other);
  }

  Var operator>=(int64_t other) const {
    return *this >= builder->createVar(other);
  }

  //===--------------------------------------------------------------------===//
  // Logical
  //===--------------------------------------------------------------------===//

  Var operator&&(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::And, attr, other.attr),
        "var_and");
    return Var(resDeclRef, builder);
  }

  Var operator||(const Var &other) const {
    TypedAttr resDeclRef = builder->wrapNonTrivialAttrWithDecl(
        KGEN::ParamOperatorAttr::get(KGEN::POC::Or, attr, other.attr),
        "var_or");
    return Var(resDeclRef, builder);
  }

  TypedAttr getAttr() const { return attr; }
  ParamExprBuilder &getBuilder() const { return *builder; }
};

inline std::optional<int64_t> tryGetConstantInt(TypedAttr attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().getSExtValue();
  return MOSH::ShapeAttr::getStaticDim(attr);
}

inline std::optional<double> tryGetConstantFloat(TypedAttr attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValueAsDouble();
  return std::nullopt;
}

inline bool isStaticDimAttr(TypedAttr attr) {
  return tryGetConstantInt(attr).has_value();
}

std::optional<SmallVector<int64_t>> tryGetConstantIntVector(TypedAttr attr);

// Rank-extends matmul input shapes with static 1 dims so they match after
// broadcasting.
std::pair<SmallVector<TypedAttr>, SmallVector<TypedAttr>>
expandMatmulInputsForBroadcasting(ShapeAttr lhsShape, ShapeAttr rhsShape);

//----------------------------------------------------------------------------//
//------------------------Shape functions begin-------------------------------//

// `GraphOp` generates parameter names and `Location` is used for created
// ops. Each `TypedAttr` input is one of:
//   - `IntegerAttr` (a constant integer)
//   - `KGEN::ParamDeclRefAttr` (a parameter)
//   - `KGEN::ParamOperatorAttr` (a parameter expression)
//   - `MOSH::ShapeAttr` (a list of the aforementioned integers)

/// Materialize the shape logic for `MO::BatchMatmulOp`.
ShapeMaterializeResult materializeBatchMatmulShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr lhsShapeAttr,
                                                   TypedAttr rhsShapeAttr);

/// Materialize the shape logic for `RMO::MatmulOp`, which broadcasts only
/// static `1` dimensions.
ShapeMaterializeResult materializeRmoMatmulShape(GraphOp graphOp, Location loc,
                                                 TypedAttr lhsShapeAttr,
                                                 TypedAttr rhsShapeAttr);

/// Materialize the shape logic for RMO binary elementwise ops.
ShapeMaterializeResult materializeRmoBinaryElementwiseShape(GraphOp graphOp,
                                                            Location loc,
                                                            ShapeAttr lhsShape,
                                                            ShapeAttr rhsShape);

/// Materialize the shape logic for RMO select ops.
ShapeMaterializeResult materializeRmoSelectShape(GraphOp graphOp, Location loc,
                                                 ShapeAttr condShape,
                                                 ShapeAttr xShape,
                                                 ShapeAttr yShape);

/// Materialize the shape logic for RMO slice ops.
/// The caller checks:
/// - step != 0
/// - step > 0 => start < stop
/// - step < 0 => stop < start
ShapeMaterializeResult
materializeRmoSliceShape(GraphOp graphOp, Location loc, ShapeAttr inputShape,
                         ShapeAttr starts, ShapeAttr stops,
                         const SmallVector<int64_t> &steps);

/// Materialize the shape logic for `MO::BroadcastShapeOp`.
ShapeMaterializeResult materializeBroadcastShapeShape(GraphOp graphOp,
                                                      Location loc,
                                                      TypedAttr lhsShapeAttr,
                                                      TypedAttr rhsShapeAttr);

/// Materialize the shape logic for `MO::BroadcastToOp`.
ShapeMaterializeResult materializeBroadcastToShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr inputShapeAttr,
                                                   TypedAttr targetShapeAttr);

/// Materialize the shape logic for `MO::ConcatOp`.
ShapeMaterializeResult
materializeConcatShape(GraphOp graphOp, Location loc, TypedAttr axisAttr,
                       ArrayRef<TypedAttr> inputShapeAttrs);

/// Materialize the shape logic for `MO::ConvOp`.
ShapeMaterializeResult
materializeConvShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterShapeAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     TypedAttr numGroupsAttr, StringAttr inputLayoutAttr,
                     LayoutAttr filterLayoutAttr);

/// Materialize the shape logic for `MO::ConvTransposeOp`.
ShapeMaterializeResult materializeConvTransposeShape(
    GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
    TypedAttr filterShapeAttr, TypedAttr stridesAttr, TypedAttr dilationsAttr,
    TypedAttr paddingsAttr, TypedAttr outputPaddingsAttr,
    StringAttr inputLayoutAttr, LayoutAttr filterLayoutAttr);

/// Materialize the shape logic for `MO::GatherOp`.
ShapeMaterializeResult materializeGatherShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShapeAttr,
                                              TypedAttr indicesShapeAttr,
                                              TypedAttr axisAttr);

/// Materialize the shape logic for `MO::GatherNdOp`.
ShapeMaterializeResult materializeGatherNdShape(GraphOp graphOp, Location loc,
                                                TypedAttr inputShapeAttr,
                                                TypedAttr indicesShapeAttr,
                                                int64_t batchDims);

/// Materialize the shape logic for `MO::PadLike` ops.
ShapeMaterializeResult materializePadShape(GraphOp graphOp, Location loc,
                                           TypedAttr inputShapeAttr,
                                           TypedAttr paddingsAttr);

/// Materialize the shape logic for Pooling ops (e.g., `MO::AvgPoolOp`,
/// `MO::MaxPoolCeilOp`).
ShapeMaterializeResult
materializePoolShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     bool ceilMode);

/// Materialize the shape logic for `MO::RangeOp`.
ShapeMaterializeResult materializeRangeShape(GraphOp graphOp, Location loc,
                                             TypedAttr startAttr,
                                             TypedAttr stopAttr,
                                             TypedAttr stepAttr);

/// Materialize the shape logic for reduction ops like `MO::ReduceAddOp`.
ShapeMaterializeResult materializeReduceShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShape,
                                              TypedAttr axis);

/// Materialize the shape logic for `MO::ReshapeOp`.
///
/// Returns `targetShapeAttr` as the output shape and accumulates an assertion
/// that `inputShapeAttr` and `targetShapeAttr` have the same number of
/// elements.
ShapeMaterializeResult materializeReshapeShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr targetShapeAttr);

/// Materialize the shape logic for `MO::RoiAlignOp`.
ShapeMaterializeResult materializeRoiAlignShape(GraphOp graphOp, Location loc,
                                                TypedAttr inputShapeAttr,
                                                TypedAttr roisShapeAttr,
                                                TypedAttr outHeightAttr,
                                                TypedAttr outWidthAttr);

/// Materialize the shape logic for `MO::ScatterOp`.
ShapeMaterializeResult materializeScatterShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr updatesShapeAttr,
                                               TypedAttr indicesShapeAttr,
                                               TypedAttr axisAttr);

/// Materialize the shape logic for `MO::ScatterNd`-family ops.
ShapeMaterializeResult materializeScatterNdShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr updatesShapeAttr,
                                                 TypedAttr indicesShapeAttr);

/// Materialize the shape logic for `MO::SliceOp`.
ShapeMaterializeResult materializeSliceShape(GraphOp graphOp, Location loc,
                                             TypedAttr inputShapeAttr,
                                             TypedAttr startsAttr,
                                             TypedAttr stopsAttr,
                                             TypedAttr stepsAttr);

/// Materialize the shape logic for `MO::SplitOp`.
ShapeMaterializeResult materializeSplitOutputShapes(GraphOp graphOp,
                                                    Location loc,
                                                    TypedAttr inputShapeAttr,
                                                    TypedAttr splitSizesAttr,
                                                    TypedAttr axisAttr);

/// Materialize shape logic for ops whose output shape is `targetShapeAttr`
/// (e.g. `mo.resize_*`).
ShapeMaterializeResult materializeTargetShapeLike(GraphOp graphOp, Location loc,
                                                  TypedAttr targetShapeAttr);

/// Materialize the shape logic for `MO::SqueezeShapeOp`.
ShapeMaterializeResult materializeSqueezeShapeShape(GraphOp graphOp,
                                                    Location loc,
                                                    TypedAttr inputShapeAttr,
                                                    TypedAttr indicesShapeAttr);

/// Materialize the shape logic for `MO::TileOp`.
ShapeMaterializeResult materializeTileShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShapeAttr,
                                            TypedAttr repeatsAttr);

/// Materialize the shape logic for `MO::TopKOp` and `MO::BottomKOp`
ShapeMaterializeResult materializeTopKShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShape, TypedAttr k,
                                            TypedAttr axis);

/// Materialize the shape logic for `MO::TransposeOp`.
ShapeMaterializeResult materializeTransposeShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr permutationsAttr);

/// Materialize the shape logic for `MO::UnsqueezeShapeOp`.
ShapeMaterializeResult
materializeUnsqueezeShapeShape(GraphOp graphOp, Location loc,
                               TypedAttr inputShapeAttr,
                               TypedAttr indicesShapeAttr);

//------------------------Shape functions end---------------------------------//
//----------------------------------------------------------------------------//

/// If `attr` is a `mo.assert false`, return the error message associated
/// with it, else return null.
StringAttr tryGetAssertFalseErrMsg(TypedAttr attr);

/// Substitutes and folds `bindings` in a single pass and returns the simplified
/// `targetShapes`. Fails with the assert's message on an assert that doesn't
/// fold to `true`.
ErrorOr<SmallVector<ShapeAttr>> propagateShape(ArrayRef<ParamBinding> bindings,
                                               ArrayRef<ShapeAttr> targetShapes,
                                               bool validateAsserts = true);

// TODO move the following into a new interface, which drops `kgenParams`
// (backward compatibility for the fallback shape function) and `loc` (the op's
// own location).
using MaterializeShapeFuncTy = std::function<ShapeMaterializeResult(
    ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams, GraphOp graphOp,
    Location loc)>;

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_SHAPE_FUNCTIONS_H
