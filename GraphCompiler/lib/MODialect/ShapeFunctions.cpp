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

#include "GraphCompiler/MODialect/ShapeFunctions.h"

#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <optional>
#include <utility>

namespace M::MO {

KGEN::ParamDeclareOp ParamBinding::toOp(Location loc, OpBuilder b) const {
  return KGEN::ParamDeclareOp::create(b, loc, decl, value);
}

Var ParamExprBuilder::createVar(TypedAttr attr) {
  // No need to create a new binding if `attr` is simple.
  TypedAttr varAttr = wrapNonTrivialAttrWithDecl(attr, "var");
  return Var(varAttr, this);
}

Var ParamExprBuilder::createVar(int64_t n) {
  TypedAttr indexValue = ShapeAttr::getDim(graphOp.getContext(), n);
  return createVar(indexValue);
}

TypedAttr ParamExprBuilder::createKgenExpr(KGEN::POC poc,
                                           ArrayRef<TypedAttr> operands,
                                           StringRef declPrefix) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(poc, operands);
  return wrapNonTrivialAttrWithDecl(exprAttr, declPrefix);
}

TypedAttr ParamExprBuilder::wrapNonTrivialAttrWithDecl(TypedAttr attr,
                                                       StringRef declPrefix) {
  if (isa<IntegerAttr, KGEN::SIMDAttr, KGEN::ParamDeclRefAttr>(attr))
    return attr;

  Type resTy = attr.getType();
  KGEN::ParamDeclAttr declAttr = graphOp.getNewParamOfType(declPrefix, resTy);
  auto declRefAttr = KGEN::ParamDeclRefAttr::get(declAttr);
  createdOps->push_back(ParamBinding(declAttr, attr));
  return declRefAttr;
}

Var ParamExprBuilder::createCond(const Var &condV, const Var &thenV,
                                 const Var &elseV) {
  TypedAttr staticCond = condV.getAttr();
  // POC::Cond expect a scalar<bool> cond, cast to it if necessary.
  if (!KGEN::isScalarOf<KGEN::KGENDType::kBool>(staticCond.getType())) {
    if (!isa<KGEN::SIMDType>(staticCond.getType()))
      staticCond = KGEN::CastFromBuiltinAttr::get(staticCond);
    MLIRContext *ctx = staticCond.getContext();
    staticCond = KGEN::POP::CastAttr::get(
        ctx, staticCond, KGEN::SIMDType::get(ctx, 1, KGEN::KGENDType::kBool));
  }
  auto exprAttr = KGEN::ParamOperatorAttr::get(
      KGEN::POC::Cond, {staticCond, thenV.getAttr(), elseV.getAttr()});
  return createVar(wrapNonTrivialAttrWithDecl(exprAttr, "cond"));
}

Var ParamExprBuilder::createMax(const Var &lhs, const Var &rhs) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::Max, lhs.getAttr(),
                                               rhs.getAttr());
  return createVar(wrapNonTrivialAttrWithDecl(exprAttr, "max"));
}

Var ParamExprBuilder::createMin(const Var &lhs, const Var &rhs) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::Min, lhs.getAttr(),
                                               rhs.getAttr());
  return createVar(wrapNonTrivialAttrWithDecl(exprAttr, "min"));
}

ShapeAttr ParamExprBuilder::getShapeAttr(ArrayRef<TypedAttr> dimAttrs) {
  return ShapeAttr::get(dimAttrs, ShapeType::get(graphOp.getContext()));
}

void ParamExprBuilder::createAssert(const Var &condVar, StringRef errorMsg) {
  TypedAttr condAttr = condVar.getAttr();

  if (KGEN::isTriviallyTrueProposition(condAttr))
    return;

  MLIRContext *ctx = condAttr.getContext();
  Type chainType = MO::ChainType::get(ctx);

  auto signature = FuncTypeGeneratorType::get(
      /*inputParamTypes=*/{},
      FunctionType::get(ctx, {condAttr.getType()}, chainType));
  NamedAttribute msgAttr(StringAttr::get(ctx, "message"),
                         StringAttr::get(errorMsg, KGEN::StringType::get(ctx)));
  auto dictAttr = DictionaryAttr::get(ctx, {msgAttr});
  auto opNameAttr = StringAttr::get(ctx, MO::AssertOp::getOperationName());
  SmallVector<TypedAttr> applyOperands{
      KGEN::MLIROpAttr::get(opNameAttr, dictAttr, signature), condAttr};

  (void)createKgenExpr(KGEN::POC::Apply, applyOperands, "assert");
}

std::optional<SmallVector<int64_t>> tryGetConstantIntVector(TypedAttr attr) {
  auto shapeAttr = dyn_cast<ShapeAttr>(attr);
  if (!shapeAttr)
    return std::nullopt;

  SmallVector<int64_t> result;
  for (TypedAttr maybeIntAttr : shapeAttr.getValues()) {
    if (auto intOr = tryGetConstantInt(maybeIntAttr))
      result.push_back(intOr.value());
    else
      return std::nullopt;
  }
  return result;
}

/// Compute the output spatial dimension for a sliding window op (e.g.,
/// convolution or pooling).
static Var getSlidingWindowOutDim(const Var &inputDim, const Var &filterDim,
                                  const Var &dilation, const Var &stride,
                                  const Var &padding, bool ceilMode = false) {
  Var numerator =
      inputDim + padding - (dilation * (filterDim - 1) + 1) + stride;
  if (ceilMode)
    return (numerator + stride - 1) / stride;
  return numerator / stride;
}

// Compute the size of a range (assuming validity of input values).
static Var computeRangeSize(const Var &start, const Var &stop,
                            const Var &step) {
  // The offset must round away from zero for either sign, e.g. range(0, 3, 2)
  // and range(0, -3, -2).
  ParamExprBuilder &b = start.getBuilder();
  Var diff = stop - start;
  Var offset = b.createCond(step > 0, step - 1, step + 1);
  Var size = (diff + offset) / step;
  return size;
}

// Error for dims that are neither equal nor broadcastable, for RMO ops.
static std::string
shapeBroadcastingErrorMessage(StringRef lhsName, ShapeAttr lhsShapeAttr,
                              size_t lhsAxis, StringRef rhsName,
                              ShapeAttr rhsShapeAttr, size_t rhsAxis) {
  std::string errorMsg;
  llvm::raw_string_ostream os(errorMsg);
  StreamAsmPrinter printer(os);
  printer << "Input " << lhsName << " (shape ";
  MOSH::printShapeTypedAttr(printer, lhsShapeAttr);
  printer << ") dimension at axis " << lhsAxis << " (value ";
  MOSH::printDimensionValue(printer, lhsShapeAttr[lhsAxis]);
  printer << ") and input " << rhsName << " (shape ";
  MOSH::printShapeTypedAttr(printer, rhsShapeAttr);
  printer << ") dimension at axis " << rhsAxis << " (value ";
  MOSH::printDimensionValue(printer, rhsShapeAttr[rhsAxis]);
  printer << ") are neither equivalent nor broadcastable. Insert an "
             "explicit rebind if these dims will be equivalent at runtime.";
  return errorMsg;
}

static std::string matmulInnerDimErrorMessage(ShapeAttr lhsShapeAttr,
                                              TypedAttr lhsDim,
                                              ShapeAttr rhsShapeAttr,
                                              TypedAttr rhsDim) {
  std::string errorMsg;
  llvm::raw_string_ostream os(errorMsg);
  StreamAsmPrinter printer(os);
  printer << "Matrix multiplication input lhs (shape ";
  MOSH::printShapeTypedAttr(printer, lhsShapeAttr);
  printer << ") dimension at axis -1 (value ";
  MOSH::printDimensionValue(printer, lhsDim);
  printer << ") must match input rhs (shape ";
  MOSH::printShapeTypedAttr(printer, rhsShapeAttr);
  printer << ") dimension at axis ";

  if (rhsShapeAttr.getRank() == 1)
    printer << "-1";
  else
    printer << "-2";

  printer << " (value ";
  MOSH::printDimensionValue(printer, rhsDim);
  printer << ")";
  return errorMsg;
}

// Expands matmul inputs to a common full rank, padding with static 1s.
std::pair<SmallVector<TypedAttr>, SmallVector<TypedAttr>>
expandMatmulInputsForBroadcasting(ShapeAttr lhsShape, ShapeAttr rhsShape) {
  size_t lhsRank = lhsShape.getRank();
  size_t rhsRank = rhsShape.getRank();

  size_t fullRank = std::max(size_t(2), std::max(lhsRank, rhsRank));

  auto intDim1 = ShapeAttr::getDim(lhsShape.getContext(), 1ll);
  SmallVector<TypedAttr> lhsFullShape(fullRank, intDim1);
  SmallVector<TypedAttr> rhsFullShape(fullRank, intDim1);
  if (lhsRank == 1) {
    // `D` -> `1xD`
    lhsFullShape[fullRank - 1] = lhsShape[0];
  } else {
    size_t offset = fullRank - lhsRank;
    for (size_t i = 0; i < lhsRank; ++i)
      lhsFullShape[i + offset] = lhsShape[i];
  }

  if (rhsRank == 1) {
    // `D` -> `Dx1`
    rhsFullShape[fullRank - 2] = rhsShape[0];
  } else {
    size_t offset = fullRank - rhsRank;
    for (size_t i = 0; i < rhsRank; ++i)
      rhsFullShape[i + offset] = rhsShape[i];
  }

  return {lhsFullShape, rhsFullShape};
}

//----------------------------------------------------------------------------//
//------------------------Shape functions begin-------------------------------//

// Common arguments: `GraphOp` names new parameters, `Location` locates new
// ops, and each `TypedAttr` is a parametric representation of an op input.

/// Materialize the shape logic for `MO::BatchMatmulOp`.
ShapeMaterializeResult materializeBatchMatmulShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr lhsShapeAttr,
                                                   TypedAttr rhsShapeAttr) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  ShapeAttr lhsShape = dyn_cast<ShapeAttr>(lhsShapeAttr);
  ShapeAttr rhsShape = dyn_cast<ShapeAttr>(rhsShapeAttr);
  if (!lhsShape || !rhsShape)
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = lhsShape.getRank();
  if (rhsShape.getRank() != inputRank || inputRank <= 2) {
    return ShapeMaterializeResult::error(
        "[batch_matmul] input ranks must match and exceed 2");
  }

  SmallVector<TypedAttr> resDims;
  for (size_t batchDim = 0; batchDim < inputRank - 2; ++batchDim) {
    Var lhsDim = b.createVar(lhsShape[batchDim]);
    Var rhsDim = b.createVar(rhsShape[batchDim]);
    b.createAssert(lhsDim == rhsDim,
                   "[batch_matmul] inputs batch dimensions must match");
    resDims.push_back(lhsDim.getAttr());
  }

  Var lhsInnerDim = b.createVar(lhsShape[inputRank - 1]);
  Var rhsInnerDim = b.createVar(rhsShape[inputRank - 2]);
  b.createAssert(lhsInnerDim == rhsInnerDim,
                 "[batch_matmul] inputs inner dimensions must match");

  resDims.push_back(lhsShape[inputRank - 2]);
  resDims.push_back(rhsShape[inputRank - 1]);
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeRmoMatmulShape(GraphOp graphOp, Location loc,
                                                 TypedAttr lhsShapeAttr,
                                                 TypedAttr rhsShapeAttr) {
  mlir::MLIRContext *ctx = lhsShapeAttr.getContext();
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  ShapeAttr lhsShape = dyn_cast<ShapeAttr>(lhsShapeAttr);
  ShapeAttr rhsShape = dyn_cast<ShapeAttr>(rhsShapeAttr);
  if (!lhsShape || !rhsShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Vectors act as `1xD` / `Dx1` matrices whose `1` is squeezed from the
  // output. Dimensions outside the innermost 2 are broadcast.
  size_t lhsRank = lhsShape.getRank();
  size_t rhsRank = rhsShape.getRank();
  if (lhsRank == 0 || rhsRank == 0)
    return ShapeMaterializeResult::error("Scalar inputs are not supported");

  // The rank the underlying mo.matmul/mo.batch_matmul executes with.
  size_t fullRank = std::max(size_t(2), std::max(lhsRank, rhsRank));

  auto intDim1 = ShapeAttr::getDim(ctx, 1ll);
  auto [lhsFullShape, rhsFullShape] =
      expandMatmulInputsForBroadcasting(lhsShape, rhsShape);

  TypedAttr lhsInnerDim = lhsFullShape[fullRank - 1];
  TypedAttr rhsInnerDim = rhsFullShape[fullRank - 2];
  std::string errorMsg =
      matmulInnerDimErrorMessage(lhsShape, lhsInnerDim, rhsShape, rhsInnerDim);
  b.createAssert(b.createVar(lhsInnerDim) == b.createVar(rhsInnerDim),
                 errorMsg);

  SmallVector<TypedAttr> resDims;
  resDims.reserve(fullRank);
  size_t lhsRankOffset = fullRank - lhsRank;
  size_t rhsRankOffset = fullRank - rhsRank;
  for (size_t batchDim = 0; batchDim < fullRank - 2; ++batchDim) {
    if (lhsFullShape[batchDim] == intDim1) {
      resDims.push_back(rhsFullShape[batchDim]);
    } else if (rhsFullShape[batchDim] == intDim1) {
      resDims.push_back(lhsFullShape[batchDim]);
    } else {
      TypedAttr lhsDim = lhsFullShape[batchDim];
      TypedAttr rhsDim = rhsFullShape[batchDim];

      std::string errorMsg = shapeBroadcastingErrorMessage(
          "lhs", lhsShape, batchDim - lhsRankOffset, "rhs", rhsShape,
          batchDim - rhsRankOffset);
      b.createAssert(b.createVar(lhsDim) == b.createVar(rhsDim), errorMsg);
      resDims.push_back(lhsDim);
    }
  }

  if (lhsRank != 1)
    resDims.push_back(lhsFullShape[fullRank - 2]);
  if (rhsRank != 1)
    resDims.push_back(rhsFullShape[fullRank - 1]);

  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult
materializeRmoBinaryElementwiseShape(GraphOp graphOp, Location loc,
                                     ShapeAttr lhsShape, ShapeAttr rhsShape) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  auto intDim1 = ShapeAttr::getDim(graphOp.getContext(), 1ll);
  size_t maxRank = std::max(lhsShape.getRank(), rhsShape.getRank());

  auto getFromEnd = [](ShapeAttr shape, size_t i) -> TypedAttr {
    if (i >= shape.getRank())
      return nullptr;
    return shape[shape.getRank() - i - 1];
  };

  // Check that all inputs are equivalent or broadcastable while building the
  // output shape.
  SmallVector<TypedAttr> outputDims{maxRank, intDim1};
  for (size_t i = 0; i < maxRank; ++i) {
    TypedAttr lhsDim = getFromEnd(lhsShape, i);
    TypedAttr rhsDim = getFromEnd(rhsShape, i);
    size_t outputIndex = maxRank - i - 1;
    if (lhsDim && rhsDim) {
      if (lhsDim == intDim1) {
        outputDims[outputIndex] = rhsDim;
      } else if (rhsDim == intDim1) {
        outputDims[outputIndex] = lhsDim;
      } else {
        b.createAssert(b.createVar(lhsDim) == b.createVar(rhsDim),
                       shapeBroadcastingErrorMessage(
                           "lhs", lhsShape, lhsShape.getRank() - i - 1, "rhs",
                           rhsShape, rhsShape.getRank() - i - 1));
        outputDims[outputIndex] = lhsDim;
      }
    } else if (lhsDim) {
      outputDims[outputIndex] = lhsDim;
    } else {
      outputDims[outputIndex] = rhsDim;
    }
  }

  auto outputShape =
      ShapeAttr::get(outputDims, ShapeType::get(graphOp.getContext()));
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult materializeRmoSelectShape(GraphOp graphOp, Location loc,
                                                 ShapeAttr condShape,
                                                 ShapeAttr xShape,
                                                 ShapeAttr yShape) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  auto intDim1 = ShapeAttr::getDim(graphOp.getContext(), 1ll);
  size_t maxRank = std::max(condShape.getRank(),
                            std::max(xShape.getRank(), yShape.getRank()));

  auto getFromEndOrOne = [intDim1](ShapeAttr shape, size_t i) -> TypedAttr {
    if (i >= shape.getRank())
      return intDim1;
    return shape[shape.getRank() - i - 1];
  };

  // Check that all inputs are equivalent or broadcastable while building the
  // output shape.
  SmallVector<TypedAttr> outputDims{maxRank, intDim1};
  for (size_t i = 0; i < maxRank; ++i) {
    TypedAttr condDim = getFromEndOrOne(condShape, i);
    TypedAttr xDim = getFromEndOrOne(xShape, i);
    TypedAttr yDim = getFromEndOrOne(yShape, i);
    size_t outputIndex = maxRank - i - 1;

    SmallVector<std::tuple<StringRef, ShapeAttr, TypedAttr>, 3> nonOneDims{};
    if (condDim != intDim1)
      nonOneDims.push_back({"condition", condShape, condDim});
    if (xDim != intDim1)
      nonOneDims.push_back({"x", xShape, xDim});
    if (yDim != intDim1)
      nonOneDims.push_back({"y", yShape, yDim});

    for (size_t j = 1; j < nonOneDims.size(); ++j) {
      auto [expectedName, expectedShape, expectedDim] = nonOneDims[0];
      auto [actualName, actualShape, actualDim] = nonOneDims[j];
      b.createAssert(b.createVar(expectedDim) == b.createVar(actualDim),
                     shapeBroadcastingErrorMessage(
                         expectedName, expectedShape,
                         expectedShape.getRank() - i - 1, actualName,
                         actualShape, actualShape.getRank() - i - 1));
    }

    if (!nonOneDims.empty()) {
      auto [_name, _shape, dim] = nonOneDims[0];
      outputDims[outputIndex] = dim;
    }
  }

  ShapeAttr outputShape = b.getShapeAttr(outputDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult
materializeRmoSliceShape(GraphOp graphOp, Location loc, ShapeAttr inputShape,
                         ShapeAttr starts, ShapeAttr stops,
                         const SmallVector<int64_t> &steps) {
  SmallVector<MO::ParamBinding> paramBindings;
  MO::ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Constraints not verifiable at staging time are deferred to the lowering.
  SmallVector<TypedAttr> outDims;
  for (size_t axis = 0; axis < inputShape.getRank(); ++axis) {
    Var inputDim = b.createVar(inputShape[axis]);
    int64_t step = steps[axis];
    Var start = b.createVar(starts[axis]);
    Var stop = b.createVar(stops[axis]);

    // The caller verifies inputs, so this is an assert.
    ASSERT_STREAM(step != 0,
                  << llvm::formatv("rmo.slice step should be != 0 but got {0}",
                                   step));

    // Normalize to [0, inputDim).
    auto maybeStartConst = tryGetConstantInt(start.getAttr());
    auto maybeStopConst = tryGetConstantInt(stop.getAttr());
    start = maybeStartConst && *maybeStartConst < 0
                ? b.createVar(*maybeStartConst) + inputDim
                : start;
    stop = maybeStopConst && *maybeStopConst < 0
               ? b.createVar(*maybeStopConst) + inputDim
               : stop;
    // Offset for computing ceil((stop - start) / step).
    int64_t stepOffset = step > 0 ? step - 1 : step + 1;

    // Start/stop not statically < 0 are assumed positive (asserted in the RMO
    // to MO lowering). This keeps conds out of the result shape while handling
    // the common slice-by-dim case.

    // Overflow is checked before bounds so it is reported first.
    maybeStartConst = tryGetConstantInt(start.getAttr());
    maybeStopConst = tryGetConstantInt(stop.getAttr());
    if (maybeStartConst && maybeStopConst) {
      int64_t startConst = *maybeStartConst;
      int64_t stopConst = *maybeStopConst;
      int64_t diff;
      int64_t dim;
      if (llvm::SubOverflow(stopConst, startConst, diff) ||
          llvm::AddOverflow(diff, stepOffset, dim)) {
        return ShapeMaterializeResult::error(llvm::formatv(
            "rmo.slice index computation overflow: doesn't fit in an int64 "
            "start {0}, stop {1} for step {2}",
            startConst, stopConst, step));
      }
    }

    auto maybeInputDimConst = tryGetConstantInt(inputDim.getAttr());
    if (maybeInputDimConst) {
      int64_t inputDimConst = *maybeInputDimConst;

      if (maybeStartConst) {
        int64_t startConst = *maybeStartConst;
        if (startConst < 0 || startConst > inputDimConst) {
          return ShapeMaterializeResult::error(llvm::formatv(
              "rmo.slice start index {0} out of range for dimension size {1}",
              startConst, inputDimConst));
        }
      }

      if (maybeStopConst) {
        int64_t stopConst = *maybeStopConst;
        if (stopConst < 0 || stopConst > inputDimConst) {
          return ShapeMaterializeResult::error(llvm::formatv(
              "rmo.slice stop index {0} out of range for dimension size {1}",
              stopConst, inputDimConst));
        }
      }
    }

    // (stop - start) must have the sign of step, or outDim is wrong.
    maybeStartConst = tryGetConstantInt(start.getAttr());
    maybeStopConst = tryGetConstantInt(stop.getAttr());
    if (maybeStartConst && maybeStopConst) {
      int64_t startConst = *maybeStartConst;
      int64_t stopConst = *maybeStopConst;
      if (!((step > 0 && (startConst <= stopConst)) ||
            (step < 0 && (startConst >= stopConst))))
        return ShapeMaterializeResult::error(llvm::formatv(
            "rmo.slice start and stop should be increasing for positive step "
            "and decreasing for negative step, but got "
            "start {0}, stop {1} for step {2}",
            startConst, stopConst, step));
    }

    // Invariants that should hold at this point:
    // - step != 0
    // - step > 0 => start < stop
    // TODO(AIPIPE-109): step < 0 is disallowed currently, but almost works.
    // - step < 0 => stop < start
    // - start, stop both in [0, inputDim)
    // Symbolic start/stop may only be checked at runtime.

    // ceil((stop - start) / step).
    Var outDim = ((stop - start) + stepOffset) / step;
    outDims.push_back(outDim.getAttr());
  }

  ShapeAttr outputShape = b.getShapeAttr(outDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult materializeBroadcastShapeShape(GraphOp graphOp,
                                                      Location loc,
                                                      TypedAttr lhsShapeAttr,
                                                      TypedAttr rhsShapeAttr) {
  ShapeAttr lhsShape = dyn_cast<ShapeAttr>(lhsShapeAttr);
  ShapeAttr rhsShape = dyn_cast<ShapeAttr>(rhsShapeAttr);
  if (!lhsShape || !rhsShape)
    return ShapeMaterializeResult::nonMaterializable();

  if (lhsShape.getRank() != 1 || rhsShape.getRank() != 1) {
    return ShapeMaterializeResult::error(
        "[broadcast_shape] inputs must be rank 1 shape tensors");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);
  Var lhsDim = b.createVar(lhsShape[0]);
  Var rhsDim = b.createVar(rhsShape[0]);
  ShapeAttr outShape = b.getShapeAttr(b.createMax(lhsDim, rhsDim));
  return ShapeMaterializeResult::success(std::move(paramBindings), outShape);
}

/// Materialize the shape logic for `MO::BroadcastToOp`.
ShapeMaterializeResult materializeBroadcastToShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr inputShapeAttr,
                                                   TypedAttr targetShapeAttr) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr targetShape = dyn_cast<ShapeAttr>(targetShapeAttr);
  if (!inputShape || !targetShape)
    return ShapeMaterializeResult::nonMaterializable();

  int inputRank = inputShape.getRank();
  int outputRank = targetShape.getRank();
  if (inputRank > outputRank) {
    return ShapeMaterializeResult::error("[broadcast_to] input ranks must less "
                                         "than or equal to the output rank");
  }

  for (int i : llvm::seq(inputRank)) {
    int inputAxis = inputRank - i - 1;
    int outputAxis = outputRank - i - 1;
    Var inputDim = b.createVar(inputShape[inputAxis]);
    Var outputDim = b.createVar(targetShape[outputAxis]);

    auto message =
        llvm::formatv(
            "[broadcast_to] input dimension at index {0} ({1}) must be either "
            "1 or equal to "
            "corresponding output dimension at index {2} ({3})",
            inputAxis, KGEN::getParamAsString(inputShape[inputAxis]),
            outputAxis, KGEN::getParamAsString(outputDim.getAttr()))
            .str();

    b.createAssert(inputDim == 1 || inputDim == outputDim, message);
  }
  return ShapeMaterializeResult::success(std::move(paramBindings), targetShape);
}

/// Materialize the shape logic for `MO::ConcatOp`.
ShapeMaterializeResult
materializeConcatShape(GraphOp graphOp, Location loc, TypedAttr axisAttr,
                       ArrayRef<TypedAttr> inputShapeAttrs) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  std::optional<int64_t> axisOr = tryGetConstantInt(axisAttr);
  if (!axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t axis = axisOr.value();

  SmallVector<ShapeAttr> inputShapes;
  for (TypedAttr inputShapeAttr : inputShapeAttrs) {
    if (auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr)) {
      inputShapes.push_back(inputShape);
    } else {
      return ShapeMaterializeResult::nonMaterializable();
    }
  }

  // The verifier ensures at least 1 input and matching ranks.
  ShapeAttr firstInputShape = inputShapes.front();
  auto restInputShapes = ArrayRef(inputShapes).drop_front();
  int64_t inputRank = firstInputShape.getRank();

  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= axis) {
    return ShapeMaterializeResult::error(
        "[concat] normalized axis must be within range [0, input_rank)");
  }

  SmallVector<TypedAttr> resDims(firstInputShape.getValues());
  // Sum the axis dims in one variadic `add`, as `LowerConcat` does, not via
  // `Var`. Accumulating mints a parameter per partial sum, and KGEN orders
  // commutative operands by parameter name, so the term order would depend on
  // the parameter counter. Re-running inference for verification, with a
  // larger counter, would then see a reordered sum as a type mismatch.
  SmallVector<TypedAttr> axisDims{firstInputShape[axis]};
  for (ShapeAttr inputShape : restInputShapes) {
    for (auto [idx, firstInputDimAttr, inputDimAttr] :
         llvm::enumerate(firstInputShape.getValues(), inputShape.getValues())) {
      if (static_cast<int64_t>(idx) != axis) {
        Var firstInputDim = b.createVar(firstInputDimAttr);
        Var inputDim = b.createVar(inputDimAttr);
        b.createAssert(
            firstInputDim == inputDim,
            "[concat] input shapes must match except at concat axis");
      }
    }
    axisDims.push_back(inputShape[axis]);
  }
  resDims[axis] = KGEN::ParamOperatorAttr::get(KGEN::POC::Add, axisDims);
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult
materializeConvShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterShapeAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     TypedAttr numGroupsAttr, StringAttr inputLayoutAttr,
                     LayoutAttr filterLayoutAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filterShape = dyn_cast<ShapeAttr>(filterShapeAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  // `numGroupsAttr` is a scalar, not a shape.
  if (!inputShape || !filterShape || !strides || !dilations || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  size_t inputRank = inputShape.getRank();
  if (inputRank < 3) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (input_rank >= 3)");
  }
  if (inputRank != filterShape.getRank()) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (input_rank == filter_rank)");
  }
  if (strides.getSize() != inputRank - 2 ||
      dilations.getSize() != inputRank - 2) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (len(strides) == len(dilations) == input_rank "
        "- 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (len(paddings) == 2 * (input rank - 2))");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Default layouts: input/output [batch, ...spatial..., channels], filter
  // [...spatial..., filter_channels, output_channels].
  bool isNCHW = inputLayoutAttr == "NCHW";
  bool isFCRS = filterLayoutAttr && (filterLayoutAttr.getFormat() == "FCRS" ||
                                     filterLayoutAttr.getFormat() == "FCQRS");
  size_t filterChannelIdx = isFCRS ? 1 : inputRank - 2;
  size_t outputChannelIdx = isFCRS ? 0 : inputRank - 1;
  size_t inputChannelIdx = isNCHW ? 1 : inputRank - 1;
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputChannelIdx]);
  Var filterChannels = b.createVar(filterShape[filterChannelIdx]);
  Var outputChannels = b.createVar(filterShape[outputChannelIdx]);
  if (!ShapeAttr::isDim(numGroupsAttr))
    numGroupsAttr = KGEN::CastFromBuiltinAttr::get(numGroupsAttr);
  Var numGroups = b.createVar(numGroupsAttr);

  b.createAssert(inputChannels == (numGroups * filterChannels),
                 "[convolution] requires (input_channels == num_groups * "
                 "filter_channels)");
  b.createAssert(
      (outputChannels % numGroups) == 0,
      "[convolution] output_channels must be divisible by num_groups");

  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());
  for (size_t axis = 0; axis < inputRank - 2; ++axis) {
    Var inputSpatialDim = b.createVar(inputShape[isNCHW ? axis + 2 : axis + 1]);
    Var filterSpatialDim = isFCRS ? b.createVar(filterShape[axis + 2])
                                  : b.createVar(filterShape[axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);

    Var outputSpatialDim =
        getSlidingWindowOutDim(inputSpatialDim, filterSpatialDim, dilation,
                               stride, prePadding + postPadding);
    b.createAssert(outputSpatialDim > 0,
                   "[convolution] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  resDims.push_back(outputChannels.getAttr());
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeConvTransposeShape(
    GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
    TypedAttr filterShapeAttr, TypedAttr stridesAttr, TypedAttr dilationsAttr,
    TypedAttr paddingsAttr, TypedAttr outputPaddingsAttr,
    StringAttr inputLayoutAttr, LayoutAttr filterLayoutAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filterShape = dyn_cast<ShapeAttr>(filterShapeAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  auto outPaddings = dyn_cast<ShapeAttr>(outputPaddingsAttr);
  if (!inputShape || !filterShape || !strides || !dilations || !paddings ||
      !outPaddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  size_t inputRank = inputShape.getRank();
  if (inputRank != 4 && inputRank != 5) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (input_rank == 4 or 5)");
  }
  if (inputRank != filterShape.getRank()) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (input_rank == kernel_rank)");
  }
  if (strides.getSize() != inputRank - 2 ||
      dilations.getSize() != inputRank - 2) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (len(strides) == len(dilations) == "
        "input_rank - 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (len(paddings) == 2 * (input rank - 2))");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Default layouts as for conv, but being conv's inverse w.r.t. the input,
  // the output is [batch, ...spatial..., filter_channels].
  bool isNCHW = inputLayoutAttr == "NCHW";
  bool isCFRS = filterLayoutAttr && filterLayoutAttr.getFormat() == "CFRS";
  size_t filterChannelIdx = isCFRS ? 0 : inputRank - 1;
  size_t outputChannelIdx = isCFRS ? 1 : inputRank - 2;
  size_t inputChannelIdx = isNCHW ? 1 : inputRank - 1;
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputChannelIdx]);
  Var filterChannels = b.createVar(filterShape[filterChannelIdx]);
  Var outputChannels = b.createVar(filterShape[outputChannelIdx]);

  b.createAssert(inputChannels == filterChannels,
                 "[conv_transpose] requires input channel to match filter "
                 "channel in kernel");

  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());

  if (isNCHW)
    resDims.push_back(outputChannels.getAttr());

  auto computeOutDim = [](const Var &inputDim, const Var &filterDim,
                          const Var &dilation, const Var &stride,
                          const Var &prePad, const Var &postPad,
                          const Var &outPad) -> Var {
    return stride * (inputDim - 1) + outPad + ((filterDim - 1) * dilation + 1) -
           prePad - postPad;
  };

  for (size_t axis = 0; axis < inputRank - 2; ++axis) {
    Var inputSpatialDim = b.createVar(inputShape[isNCHW ? axis + 2 : axis + 1]);
    Var filterSpatialDim = b.createVar(filterShape[isCFRS ? axis + 2 : axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePad = b.createVar(paddings[2 * axis]);
    Var postPad = b.createVar(paddings[2 * axis + 1]);
    Var outPad = b.createVar(outPaddings[axis]);

    Var outputSpatialDim =
        computeOutDim(inputSpatialDim, filterSpatialDim, dilation, stride,
                      prePad, postPad, outPad);
    b.createAssert(outputSpatialDim > 0,
                   "[conv_transpose] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  if (!isNCHW)
    resDims.push_back(outputChannels.getAttr());

  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializePadShape(GraphOp graphOp, Location loc,
                                           TypedAttr inputShapeAttr,
                                           TypedAttr paddingsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  if (!inputShape || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  size_t inputRank = inputShape.getRank();
  if (2 * inputRank != paddings.getSize()) {
    return ShapeMaterializeResult::error(
        "[pad] paddings shape must be (2 * input_rank)");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; ++axis) {
    Var inputDim = b.createVar(inputShape[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);
    Var outputDim = inputDim + prePadding + postPadding;
    resDims.push_back(outputDim.getAttr());
  }

  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult
materializePoolShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     bool ceilMode) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filter = dyn_cast<ShapeAttr>(filterAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  if (!inputShape || !filter || !strides || !dilations || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  size_t inputRank = inputShape.getRank();
  // Arbitrary ranks would be trivial; kept matching the Mojo shape function.
  if (inputRank != 4) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (input_rank == 4)");
  }

  size_t filterSize = filter.getSize();
  if (filterSize != inputRank - 2 ||
      !llvm::all_equal({filterSize, dilations.getSize(), strides.getSize()})) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (len(filter) == len(strides) == len(dilations) == "
        "input_rank - 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (len(paddings) == 2 * (input rank - 2))");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // Input/output layout is [batch, ...spatial..., channels].
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputRank - 1]);

  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());
  for (size_t axis = 0; axis < inputRank - 2; ++axis) {
    Var inputSpatialDim = b.createVar(inputShape[axis + 1]);
    Var filterSpatialDim = b.createVar(filter[axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);

    Var outputSpatialDim =
        getSlidingWindowOutDim(inputSpatialDim, filterSpatialDim, dilation,
                               stride, prePadding + postPadding, ceilMode);
    b.createAssert(outputSpatialDim > 0,
                   "[pooling] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  resDims.push_back(inputChannels.getAttr());
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeGatherShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShapeAttr,
                                              TypedAttr indicesShapeAttr,
                                              TypedAttr axisAttr) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  auto axisOr = tryGetConstantInt(axisAttr);
  if (!inputShape || !indicesShape || !axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = inputShape.getRank();
  int64_t axis = axisOr.value();
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= static_cast<size_t>(axis)) {
    return ShapeMaterializeResult::error(
        "[gather] normalized axis must be within range [0, input_rank)");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  // output_shape := input_shape[:axis] + indices_shape + input_shape[axis+1:]
  SmallVector<TypedAttr> resDims;
  for (int64_t idx = 0; idx < axis; ++idx)
    resDims.push_back(inputShape[idx]);
  llvm::append_range(resDims, indicesShape.getValues());
  for (size_t idx = axis + 1; idx < inputRank; ++idx)
    resDims.push_back(inputShape[idx]);

  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeGatherNdShape(GraphOp graphOp, Location loc,
                                                TypedAttr inputShapeAttr,
                                                TypedAttr indicesShapeAttr,
                                                int64_t batchDims) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = inputShape.getRank();
  size_t indicesRank = indicesShape.getRank();
  if (inputRank < 1 || indicesRank < 1) {
    return ShapeMaterializeResult::error(
        "[gather_nd] input_rank and indices_rank must be >= 1");
  }

  auto idxSizeOr = tryGetConstantInt(indicesShape[indicesRank - 1]);
  if (!idxSizeOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t idxSize = idxSizeOr.value();

  if (idxSize < 1 || inputRank - batchDims < static_cast<size_t>(idxSize)) {
    return ShapeMaterializeResult::error(
        "[gather_nd] index size must be within range [1, input_rank - "
        "batch_dims]");
  }
  if (batchDims >= static_cast<int64_t>(indicesRank)) {
    return ShapeMaterializeResult::error(
        "[gather_nd] requires (batch_dims < indices_rank)");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  //   output_shape = input_shape[:batch_dims]
  //                  + indices_shape[batch_dims:-1]
  //                  + input_shape[batch_dims + indices_shape[-1]:]
  SmallVector<TypedAttr> resDims;

  for (int64_t idx = 0; idx < batchDims; ++idx)
    resDims.push_back(inputShape[idx]);
  llvm::append_range(resDims,
                     indicesShape.getValues().slice(batchDims).drop_back());
  for (size_t idx = batchDims + idxSize; idx < inputRank; ++idx)
    resDims.push_back(inputShape[idx]);

  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

// Materialize the shape logic for ops whose shape is the *value* of an input.
ShapeMaterializeResult materializeTargetShapeLike(GraphOp graphOp, Location loc,
                                                  TypedAttr targetShapeAttr) {
  auto asShapeAttr = dyn_cast<ShapeAttr>(targetShapeAttr);
  if (!asShapeAttr)
    return ShapeMaterializeResult::nonMaterializable();

  return ShapeMaterializeResult::success({}, asShapeAttr);
}

/// Materialize the shape logic for `MO::RangeOp`.
ShapeMaterializeResult materializeRangeShape(GraphOp graphOp, Location loc,
                                             TypedAttr startAttr,
                                             TypedAttr stopAttr,
                                             TypedAttr stepAttr) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  Var start = b.createVar(startAttr);
  Var stop = b.createVar(stopAttr);
  Var step = b.createVar(stepAttr);

  b.createAssert(step != 0, "[range] step must be non-zero");

  b.createAssert(start <= stop || step < 0,
                 "[range] requires (start <= stop) for positive step");
  b.createAssert(stop <= start || step > 0,
                 "[range] requires (stop <= start) for negative step");

  Var outDim = computeRangeSize(start, stop, step);
  ShapeAttr outputShape = b.getShapeAttr(outDim);
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult materializeReduceShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShape,
                                              TypedAttr axis) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  auto inputShapeAttr = dyn_cast<ShapeAttr>(inputShape);
  // A reduction without an axis changes no dimension.
  if (!axis && inputShapeAttr)
    return ShapeMaterializeResult::success(std::move(paramBindings),
                                           inputShapeAttr);
  auto axisOr = tryGetConstantInt(axis);

  if (!axisOr.has_value() || !inputShapeAttr)
    return ShapeMaterializeResult::nonMaterializable();

  int64_t rank = inputShapeAttr.getRank();
  int64_t normalizedAxis = (*axisOr < 0) ? (*axisOr + rank) : *axisOr;

  if (normalizedAxis < 0 || normalizedAxis >= rank)
    return ShapeMaterializeResult::error(
        "[reduction] normalized axis must be within range [0, input_rank)");

  SmallVector<TypedAttr> finalShape(inputShapeAttr.getValues());
  finalShape[normalizedAxis] = b.createVar(1).getAttr();
  return ShapeMaterializeResult::success(std::move(paramBindings),
                                         b.getShapeAttr(finalShape));
}

ShapeMaterializeResult materializeScatterShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr updatesShapeAttr,
                                               TypedAttr indicesShapeAttr,
                                               TypedAttr axisAttr) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr updatesShape = dyn_cast<ShapeAttr>(updatesShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  auto axisOr = tryGetConstantInt(axisAttr);
  if (!inputShape || !updatesShape, !indicesShape || !axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = inputShape.getRank();
  if (!llvm::all_equal(
          {inputRank, updatesShape.getRank(), indicesShape.getRank()})) {
    return ShapeMaterializeResult::error(
        "[scatter] rank of inputs must be equal");
  }

  int64_t axis = axisOr.value();
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= static_cast<size_t>(axis)) {
    return ShapeMaterializeResult::error(
        "[scatter] normalized axis must be within range [0, input_rank)");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  for (size_t i = 0; i < inputRank; ++i) {
    Var inputDim = b.createVar(inputShape[i]);
    Var updatesDim = b.createVar(updatesShape[i]);
    Var indicesDim = b.createVar(indicesShape[i]);
    b.createAssert(indicesDim == updatesDim,
                   "[scatter] indices and updates must have the same shape");
    b.createAssert(indicesDim <= inputDim,
                   "[scatter] indices shape cannot be bigger than input shape");
  }

  return ShapeMaterializeResult::success(std::move(paramBindings), inputShape);
}

ShapeMaterializeResult materializeScatterNdShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr updatesShapeAttr,
                                                 TypedAttr indicesShapeAttr) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr updatesShape = dyn_cast<ShapeAttr>(updatesShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !updatesShape, !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  int64_t inputRank = inputShape.getRank();
  int64_t updatesRank = updatesShape.getRank();
  int64_t indicesRank = indicesShape.getRank();
  if (indicesRank < 1) {
    return ShapeMaterializeResult::error(
        "[scatter_nd] indices cannot be a scalar");
  }

  auto numSlicedDimsOr = tryGetConstantInt(indicesShape[indicesRank - 1]);
  if (!numSlicedDimsOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t numSlicedDims = numSlicedDimsOr.value();

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  if (numSlicedDims > inputRank) {
    return ShapeMaterializeResult::error(
        "[scatter_nd] cannot slice more dimensions than what input has");
  }
  if (updatesRank != indicesRank - 1 + inputRank - numSlicedDims) {
    return ShapeMaterializeResult::error(
        "[scatter_nd] requires (updates_rank == indices_rank - 1 + input_rank "
        "- num_sliced_dims)");
  }

  for (int64_t i = 0; i < indicesRank - 1; ++i) {
    Var indicesDim = b.createVar(indicesShape[i]);
    Var updatesDim = b.createVar(updatesShape[i]);
    b.createAssert(
        indicesDim == updatesDim,
        "[scatter_nd] batch dimensions of indices and updates don't match");
  }
  for (int64_t i = 0; i < inputRank - numSlicedDims; ++i) {
    Var inputDim = b.createVar(inputShape[i + numSlicedDims]);
    Var updatesDim = b.createVar(updatesShape[i + indicesRank - 1]);
    b.createAssert(
        inputDim == updatesDim,
        "[scatter_nd] updated dimensions of input and updates don't match");
  }

  return ShapeMaterializeResult::success(std::move(paramBindings), inputShape);
}

/// Materialize the shape logic for `MO::ReshapeOp`.
///
/// Returns `targetShapeAttr` as the output shape and accumulates an assertion
/// that `inputShapeAttr` and `targetShapeAttr` have the same number of
/// elements.
ShapeMaterializeResult materializeReshapeShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr targetShapeAttr) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr targetShape = dyn_cast<ShapeAttr>(targetShapeAttr);
  if (!inputShape || !targetShape)
    return ShapeMaterializeResult::nonMaterializable();

  if (llvm::any_of(targetShape.getValues(), [](TypedAttr dim) {
        return ShapeAttr::getStaticDim(dim) == -1;
      }))
    return ShapeMaterializeResult::error(
        "[reshape] -1 size unsupported in target shape");

  Var inputNumElems = b.createVar(inputShape.getSymbolicNumElements());
  Var outputNumElems = b.createVar(targetShape.getSymbolicNumElements());
  b.createAssert(
      inputNumElems == outputNumElems,
      llvm::formatv(
          "[reshape] input and output number of elements must match: "
          "{0} != {1}\n"
          "If you are confident this is correct, consider a rebind to assert "
          "that the reshape is valid.\n"
          "Ex: ```\n"
          "    n, m = x.shape\n"
          "    x = x.rebind([(n // 4) * 4, m])\n"
          "    x.reshape([n // 4, 4, m])\n"
          "```",
          inputShape, targetShape)
          .str());
  return ShapeMaterializeResult::success(std::move(paramBindings), targetShape);
}

ShapeMaterializeResult materializeRoiAlignShape(GraphOp graphOp, Location loc,
                                                TypedAttr inputShapeAttr,
                                                TypedAttr roisShapeAttr,
                                                TypedAttr outHeightAttr,
                                                TypedAttr outWidthAttr) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr roisShape = dyn_cast<ShapeAttr>(roisShapeAttr);
  if (!inputShape || !roisShape)
    return ShapeMaterializeResult::nonMaterializable();

  if (inputShape.getRank() != 4) {
    return ShapeMaterializeResult::error(
        "[roi_align] input shape must have rank 4");
  }
  if (roisShape.getRank() != 2) {
    return ShapeMaterializeResult::error(
        "[roi_align] rois shape must have rank 2");
  }
  if (!MOSH::ShapeAttr::isDim(outHeightAttr)) {
    return ShapeMaterializeResult::error(
        "[roi_align] output height must be a single integer");
  }
  if (!MOSH::ShapeAttr::isDim(outHeightAttr)) {
    return ShapeMaterializeResult::error(
        "[roi_align] output width must be a single integer");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  Var outHeight = b.createVar(outHeightAttr);
  Var outWidth = b.createVar(outWidthAttr);
  b.createAssert(outHeight >= 0,
                 "[roi_align] output height must be non-negative");
  b.createAssert(outWidth >= 0,
                 "[roi_align] output width must be non-negative");

  ShapeAttr outShape = b.getShapeAttr(
      {roisShape[0], outHeightAttr, outWidthAttr, inputShape[3]});
  return ShapeMaterializeResult::success(std::move(paramBindings), outShape);
}

/// Materialize the shape logic for `MO::SliceOp`.
ShapeMaterializeResult materializeSliceShape(GraphOp graphOp, Location loc,
                                             TypedAttr inputShapeAttr,
                                             TypedAttr startsAttr,
                                             TypedAttr stopsAttr,
                                             TypedAttr stepsAttr) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr starts = dyn_cast<ShapeAttr>(startsAttr);
  ShapeAttr stops = dyn_cast<ShapeAttr>(stopsAttr);
  ShapeAttr steps = dyn_cast<ShapeAttr>(stepsAttr);
  if (!inputShape || !starts || !stops || !steps)
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = inputShape.getRank();
  if (starts.getSize() != inputRank ||
      !llvm::all_equal({starts.getSize(), stops.getSize(), steps.getSize()})) {
    return ShapeMaterializeResult::error(
        "Input rank must match the number of elements in starts/stops/steps "
        "bounds");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  SmallVector<TypedAttr> outDims;
  for (size_t axis = 0; axis < inputRank; ++axis) {
    Var inDim = b.createVar(inputShape[axis]);
    Var start = b.createVar(starts[axis]);
    Var stop = b.createVar(stops[axis]);
    Var step = b.createVar(steps[axis]);

    b.createAssert(step != 0, "[slice] step must be non-zero");

    Var idxMin = b.createCond(step > 0, b.createVar(0), b.createVar(-1));
    Var idxMax = b.createCond(step > 0, inDim, inDim - 1);

    start = b.createCond(start < 0, start + inDim, start);
    start = b.createMax(idxMin, b.createMin(idxMax, start));
    stop = b.createCond(stop < 0, stop + inDim, stop);
    stop = b.createMax(idxMin, b.createMin(idxMax, stop));

    // Make sure start and stop don't cross each other.
    b.createAssert(step < 0 || start <= stop,
                   "[slice] normalized stop cannot be smaller than start for "
                   "positive step");
    b.createAssert(step > 0 || start >= stop,
                   "[slice] normalized start cannot be smaller than stop for "
                   "negative step");

    Var outDim = computeRangeSize(start, stop, step);
    outDims.push_back(outDim.getAttr());
  }

  ShapeAttr outputShape = b.getShapeAttr(outDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), outputShape);
}

ShapeMaterializeResult materializeSplitOutputShapes(GraphOp graphOp,
                                                    Location loc,
                                                    TypedAttr inputShapeAttr,
                                                    TypedAttr splitSizesAttr,
                                                    TypedAttr axisAttr) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr splitSizes = dyn_cast<ShapeAttr>(splitSizesAttr);
  if (!inputShape || !splitSizes)
    return ShapeMaterializeResult::nonMaterializable();

  std::optional<int64_t> axisOr = tryGetConstantInt(axisAttr);
  if (!axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t axis = axisOr.value();

  size_t inputRank = inputShape.getRank();
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= static_cast<size_t>(axis)) {
    return ShapeMaterializeResult::error(
        "[split] normalized axis must be within range [0, rank)");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  Var splitSizesSum = b.createVar(0);
  for (TypedAttr splitSize : splitSizes.getValues())
    splitSizesSum = splitSizesSum + b.createVar(splitSize);

  b.createAssert(
      splitSizesSum == b.createVar(inputShape[axis]),
      "[split] sum of split sizes must match input dimension at split axis");

  SmallVector<ShapeAttr> outShapes;
  for (size_t i = 0; i < splitSizes.getRank(); ++i) {
    SmallVector<TypedAttr> outDims(inputShape.getValues());
    outDims[axis] = splitSizes[i];
    ShapeAttr outputShape = b.getShapeAttr(outDims);
    outShapes.push_back(outputShape);
  }
  return ShapeMaterializeResult::success(std::move(paramBindings),
                                         std::move(outShapes));
}

ShapeMaterializeResult
materializeSqueezeShapeShape(GraphOp graphOp, Location loc,
                             TypedAttr inputShapeAttr,
                             TypedAttr indicesShapeAttr) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  if (inputShape.getRank() != 1 || indicesShape.getRank() != 1) {
    return ShapeMaterializeResult::error(
        "[squeeze_shape] inputs must be rank 1 shape tensors");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);
  Var inputNumElems = b.createVar(inputShape[0]);
  Var indicesNumElems = b.createVar(indicesShape[0]);
  b.createAssert(
      inputNumElems >= indicesNumElems,
      "[squeeze_shape] cannot remove more dimensions than there exists");

  Var outputNumElems = inputNumElems - indicesNumElems;
  ShapeAttr outShape = b.getShapeAttr(outputNumElems);
  return ShapeMaterializeResult::success(std::move(paramBindings), outShape);
}

ShapeMaterializeResult materializeTileShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShapeAttr,
                                            TypedAttr repeatsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto repeats = dyn_cast<ShapeAttr>(repeatsAttr);
  if (!inputShape || !repeats)
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = inputShape.getRank();
  if (inputRank != repeats.getSize()) {
    return ShapeMaterializeResult::error(
        "[tile] requires (len(repeats) == input_rank)");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; ++axis) {
    Var inputDim = b.createVar(inputShape[axis]);
    Var repeat = b.createVar(repeats[axis]);
    Var outputDim = inputDim * repeat;
    resDims.push_back(outputDim.getAttr());
  }

  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeTransposeShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr permutationsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  if (!inputShape)
    return ShapeMaterializeResult::nonMaterializable();

  auto permutations = dyn_cast<ShapeAttr>(permutationsAttr);
  size_t inputRank = inputShape.getRank();
  if (inputRank != permutations.getSize()) {
    return ShapeMaterializeResult::error(
        "[transpose] permutation size must match input rank");
  }

  SmallVector<int64_t> permInts;
  for (TypedAttr permAttr : permutations.getValues()) {
    if (auto permInt = tryGetConstantInt(permAttr)) {
      permInts.push_back(permInt.value());
    } else {
      return ShapeMaterializeResult::nonMaterializable();
    }
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; ++axis) {
    int64_t perm = permInts[axis];
    if (perm < 0 || inputRank <= static_cast<size_t>(perm)) {
      return ShapeMaterializeResult::error(
          "[transpose] each permutation must be within range [0, rank)");
    }
    TypedAttr outputDim = inputShape[perm];
    resDims.push_back(outputDim);
  }

  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(paramBindings), resShape);
}

ShapeMaterializeResult materializeTopKShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShape, TypedAttr k,
                                            TypedAttr axis) {
  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);

  auto axisOr = tryGetConstantInt(axis);
  auto inputShapeAttr = dyn_cast<ShapeAttr>(inputShape);

  if (!axisOr.has_value() || !inputShapeAttr)
    return ShapeMaterializeResult::nonMaterializable();

  int64_t rank = inputShapeAttr.getRank();
  if (*axisOr < 0 || *axisOr >= rank)
    return ShapeMaterializeResult::error(
        "[top/bottom-k] axis must be within [0, rank)");

  auto kVar = b.createVar(k);
  auto dimVar = b.createVar(inputShapeAttr.getValues()[*axisOr]);
  b.createAssert(kVar >= 0 && kVar <= dimVar,
                 "[top/bottom-k] k must be between [0, dim_size].");

  SmallVector<TypedAttr> finalShape(inputShapeAttr.getValues());
  finalShape[*axisOr] = k;

  ShapeAttr shape = b.getShapeAttr(finalShape);
  return ShapeMaterializeResult::success(std::move(paramBindings),
                                         SmallVector<ShapeAttr>{shape, shape});
}

ShapeMaterializeResult
materializeUnsqueezeShapeShape(GraphOp graphOp, Location loc,
                               TypedAttr inputShapeAttr,
                               TypedAttr indicesShapeAttr) {
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  if (inputShape.getRank() != 1 || indicesShape.getRank() != 1) {
    return ShapeMaterializeResult::error(
        "[squeeze_shape] inputs must be rank 1 shape tensors");
  }

  SmallVector<ParamBinding> paramBindings;
  ParamExprBuilder b(graphOp, loc, &paramBindings);
  Var inputNumElems = b.createVar(inputShape[0]);
  Var indicesNumElems = b.createVar(indicesShape[0]);

  ShapeAttr outShape = b.getShapeAttr(inputNumElems + indicesNumElems);
  return ShapeMaterializeResult::success(std::move(paramBindings), outShape);
}

//------------------------Shape functions end---------------------------------//
//----------------------------------------------------------------------------//

/// Returns `(condAttr, errorMsgAttr)` if `attr` encodes a `mo.assert`.
static std::optional<std::pair<TypedAttr, StringAttr>>
tryGetAssertCondAndMsg(TypedAttr attr) {
  auto paramOperatorAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
  if (!paramOperatorAttr)
    return std::nullopt;

  auto mlirOpAttr = dyn_cast<KGEN::MLIROpAttr>(paramOperatorAttr.getOperand(0));
  if (!mlirOpAttr || mlirOpAttr.getName() != MO::AssertOp::getOperationName())
    return std::nullopt;

  TypedAttr cond = paramOperatorAttr.getOperand(1);
  auto msgAttr = mlirOpAttr.getAttrs().getAs<StringAttr>("message");
  return std::make_pair(cond, msgAttr);
}

StringAttr tryGetAssertFalseErrMsg(TypedAttr attr) {
  if (auto condAndMsg = tryGetAssertCondAndMsg(attr)) {
    TypedAttr cond = condAndMsg.value().first;
    StringAttr errMsg = condAndMsg.value().second;

    if (KGEN::isTriviallyFalseProposition(cond))
      return errMsg;
  }
  return {};
}

/// Implementation of `propagateShape` which doesn't erase `bindings`.
ErrorOr<SmallVector<ShapeAttr>> propagateShape(ArrayRef<ParamBinding> bindings,
                                               ArrayRef<ShapeAttr> targetShapes,
                                               bool validateAsserts) {

  KGEN::ParameterEvaluator evaluator;
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](KGEN::ParamDeclRefAttr declRef) -> Attribute {
    return evaluator.getReboundAttribute(declRef);
  });

  // Process individual ops and propagate by simple substitution.
  for (const ParamBinding &binding : bindings) {
    KGEN::ParamDeclAttr lhsAttr = binding.decl;
    TypedAttr rhsAttr = binding.value;

    rhsAttr = cast<TypedAttr>(replacer.replace(rhsAttr));

    if (auto condAndMsg = tryGetAssertCondAndMsg(rhsAttr)) {
      TypedAttr cond = condAndMsg.value().first;
      StringAttr errMsg = condAndMsg.value().second;

      if (validateAsserts) {
        if (!KGEN::isTriviallyTrueProposition(cond))
          return Error(errMsg.str());
      } else if (KGEN::isTriviallyFalseProposition(cond)) {
        return Error(errMsg.str());
      }
    }

    evaluator.setDeclBinding(lhsAttr, rhsAttr);
  }

  SmallVector<ShapeAttr> resultShapes;
  for (ShapeAttr targetShape : targetShapes) {
    auto resultShape = cast<ShapeAttr>(replacer.replace(targetShape));
    resultShapes.push_back(resultShape);
  }
  return resultShapes;
}

} // namespace M::MO
