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
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/ML/RangeUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

namespace M::MO {

namespace {

template <typename OpType>
struct StaticizeAdaptor
    : public OpType::template GenericAdaptor<ArrayRef<TypedAttr>> {
  using Base = typename OpType::template GenericAdaptor<ArrayRef<TypedAttr>>;
  using Base::Base;
};

/// Return whether the given attribute is a constant integer 1.
bool isConstantOne(TypedAttr attr) {
  if (auto v = MOSH::ShapeAttr::getStaticDim(attr))
    return *v == 1;
  return false;
}

/// Return whether the input tensor type has shape `[1]`.
bool is1dTensorWithOneElem(TensorType tensorTy) {
  return tensorTy.hasRank() && tensorTy.getRank() == 1 &&
         isConstantOne(tensorTy.getShape()[0]);
}

/// Return whether the input tensor type has shape `[]`.
bool isScalar(TensorType tensorTy) {
  return tensorTy.hasRank() && tensorTy.getRank() == 0;
}

/// Staticization implementation for `mo.reshape` and `mo.static.reshape`.
static TypedAttr tryStaticizeReshapeLikeOp(TensorType inputTy,
                                           TensorType outputTy,
                                           TypedAttr inputAttr,
                                           ParamExprBuilder &b) {
  auto inputShape = inputTy.getShape();
  auto outputShape = outputTy.getShape();
  if (!inputShape || !outputShape)
    return TypedAttr{};

  if (is1dTensorWithOneElem(inputTy) && isScalar(outputTy)) {
    if (auto inputElems = dyn_cast<ShapeAttr>(inputAttr))
      return inputElems[0];

  } else if (isScalar(inputTy) && is1dTensorWithOneElem(outputTy)) {
    return b.getShapeAttr({inputAttr});
  }

  return TypedAttr{};
}

/// Staticization implementation for `mo.broadcast_to` and
/// `mo.static.broadcast_to`.
TypedAttr tryStaticizeBroadcastLikeOp(TensorType inputTy, TensorType resultTy,
                                      TypedAttr staticizedInput,
                                      ParamExprBuilder &b) {
  if (!resultTy.hasRank() || resultTy.getRank() != 1)
    return TypedAttr{};
  auto resultDimIntAttr =
      MOSH::ShapeAttr::getStaticDim(resultTy.getShape().getValues().front());
  if (!resultDimIntAttr)
    return TypedAttr{};
  int64_t resultDimInt = resultDimIntAttr.value();

  if (!inputTy.hasRank())
    return TypedAttr{};

  TypedAttr inputScalarAttr;
  if (isScalar(inputTy)) {
    inputScalarAttr = staticizedInput;

  } else if (is1dTensorWithOneElem(inputTy)) {
    inputScalarAttr = cast<ShapeAttr>(staticizedInput)[0];
  }

  // Only support constant input for now.
  std::optional<int64_t> maybeInt = tryGetConstantInt(inputScalarAttr);
  if (!maybeInt.has_value())
    return TypedAttr{};

  SmallVector<TypedAttr> broadcastedConst(resultDimInt, inputScalarAttr);
  return b.getShapeAttr(broadcastedConst);
}

/// Staticization implementation for elementwise ops (e.g., `mo.add`). `func`
/// maps scalar inputs to the scalar output, applied per index:
///   inputs[0] := ShapeAttr([N0, N1])
///   inputs[1] := ShapeAttr([1, 2])
/// yields
///   ShapeAttr([func({N0, 1}), func({N1, 2}))
template <typename OpTy>
static ErrorOr<TypedAttr> tryStaticizeElemWiseOp(
    TensorType outTy, ArrayRef<TypedAttr> inputs, ParamExprBuilder &b,
    llvm::function_ref<TypedAttr(StaticizeAdaptor<OpTy> adaptor)> func) {
  if (!outTy.hasRank())
    return TypedAttr{};

  if (outTy.getRank() == 0)
    return func(inputs);

  SmallVector<ShapeAttr> inputShapeAttrs;
  for (auto [i, input] : llvm::enumerate(inputs)) {
    auto inputShapeAttr = dyn_cast<ShapeAttr>(input);
    if (!inputShapeAttr)
      return TypedAttr{};

    if (!inputShapeAttrs.empty() &&
        inputShapeAttrs.front().getSize() != inputShapeAttr.getSize()) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "Cannot staticize binary ops whose input have different number of "
            "elements -- first input: "
         << inputShapeAttrs.front() << ", " << i
         << "-th input: " << inputShapeAttr;
      return Error(msg);
    }
    inputShapeAttrs.push_back(inputShapeAttr);
  }

  SmallVector<TypedAttr> outVals;
  size_t inputNumElems = inputShapeAttrs.front().getSize();
  for (size_t i = 0; i < inputNumElems; ++i) {
    SmallVector<TypedAttr> ithIndexScalarInputs;
    for (ShapeAttr inputShapeAttr : inputShapeAttrs)
      ithIndexScalarInputs.push_back(inputShapeAttr[i]);

    StaticizeAdaptor<OpTy> adaptor(ithIndexScalarInputs);
    outVals.push_back(func(adaptor));
  }
  return b.getShapeAttr(outVals);
}

} // namespace

ErrorOr<TypedAttr> BroadcastShapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                  ParamExprBuilder &b) {
  StaticizeAdaptor<BroadcastShapeOp> adaptor(inputs);

  auto lhsShapeAttr = dyn_cast<ShapeAttr>(adaptor.getInputX());
  auto rhsShapeAttr = dyn_cast<ShapeAttr>(adaptor.getInputY());
  if (lhsShapeAttr && rhsShapeAttr) {

    if (lhsShapeAttr.getRank() < rhsShapeAttr.getRank())
      std::swap(lhsShapeAttr, rhsShapeAttr);

    SmallVector<TypedAttr> resDims;
    size_t rankDiff = lhsShapeAttr.getRank() - rhsShapeAttr.getRank();
    for (size_t axis = 0; axis < lhsShapeAttr.getRank(); ++axis) {
      if (axis < rankDiff) {
        resDims.push_back(lhsShapeAttr[axis]);

      } else {
        Var dim1 = b.createVar(lhsShapeAttr[axis]);
        Var dim2 = b.createVar(rhsShapeAttr[axis - rankDiff]);
        b.createAssert(dim1 == 1 || dim2 == 1 || dim1 == dim2,
                       "[broadcast_shape] dimensions are not broadcastable");

        Var newResDim = b.createCond(dim1 == 1, dim2, dim1);
        resDims.push_back(newResDim.getAttr());
      }
    }
    auto res = MOSH::ShapeAttr::get(resDims, lhsShapeAttr.getType());
    return res;
  }

  return TypedAttr{};
}

ErrorOr<TypedAttr>
Static::BroadcastToOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                    ParamExprBuilder &b) {
  StaticizeAdaptor<Static::BroadcastToOp> adaptor(inputs);
  TensorType inputTy = getInput().getType();
  TensorType resultTy = getType();
  return tryStaticizeBroadcastLikeOp(inputTy, resultTy, adaptor.getInput(), b);
}

ErrorOr<TypedAttr> BroadcastToOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                               ParamExprBuilder &b) {
  StaticizeAdaptor<BroadcastToOp> adaptor(inputs);
  TensorType inputTy = getInput().getType();
  TensorType resultTy = getType();
  return tryStaticizeBroadcastLikeOp(inputTy, resultTy, adaptor.getInput(), b);
}

ErrorOr<TypedAttr> AddOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<AddOp>(getType(), inputs, b, [&](auto adaptor) {
    return b.createKgenExpr(KGEN::POC::Add,
                            {adaptor.getInputX(), adaptor.getInputY()});
  });
}

ErrorOr<TypedAttr> CastOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                        ParamExprBuilder &b) {
  // Only allow int to int cast for now.
  DType sourceDtype = getInput().getType().getDtype();
  DType targetDtype = getType().getDtype();
  if (sourceDtype.isInt() && targetDtype.isInt())
    return inputs[0];

  return TypedAttr{};
}

ErrorOr<TypedAttr> ConcatOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  // We only staticize over 1D tensors for now, so axis must be 0.
  if (getAxisAttr().getInt() != 0)
    return TypedAttr{};

  SmallVector<TypedAttr> outVals;
  for (TypedAttr input : inputs) {
    if (auto inputShapeAttr = dyn_cast<ShapeAttr>(input)) {
      llvm::append_range(outVals, inputShapeAttr.getValues());

    } else {
      return TypedAttr{};
    }
  }

  return b.getShapeAttr(outVals);
}

ErrorOr<TypedAttr> GatherOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  StaticizeAdaptor<GatherOp> adaptor(inputs);

  // We only staticize over 1D tensors for now, so axis must be 0.
  if (getAxisAttr().getInt() != 0)
    return TypedAttr{};

  auto inputElems = dyn_cast<ShapeAttr>(adaptor.getInput());
  if (!inputElems)
    return TypedAttr{};
  int64_t numInputElems = inputElems.getSize();

  SmallVector<int64_t> indices;
  bool isIndicesScalar = false;
  if (auto indicesOr = tryGetConstantIntVector(adaptor.getIndices())) {
    indices = indicesOr.value();

  } else if (auto idxOr = tryGetConstantInt(adaptor.getIndices())) {
    indices.push_back(idxOr.value());
    isIndicesScalar = true;

  } else {
    return TypedAttr{};
  }

  SmallVector<TypedAttr> outElems;
  for (int64_t idx : indices) {
    int64_t normalizedIdx = normalizeAxis(idx, numInputElems);
    if (normalizedIdx < 0 || numInputElems <= normalizedIdx) {
      std::stringstream ss;
      ss << "[gather] index out of bound: " << idx
         << " for input size: " << numInputElems;
      return Error(ss.str());
    }
    outElems.push_back(inputElems[normalizedIdx]);
  }

  if (isIndicesScalar)
    return outElems.front();
  return b.getShapeAttr(outElems);
}

ErrorOr<TypedAttr> DivOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<DivOp>(getType(), inputs, b, [&](auto adaptor) {
    return b.createKgenExpr(KGEN::POC::Div, adaptor.getOperands());
  });
}

ErrorOr<TypedAttr> EqualOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                         ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<EqualOp>(
      getType(), inputs, b, [&](auto adaptor) {
        return b.createKgenExpr(KGEN::POC::EQ, adaptor.getOperands());
      });
}

ErrorOr<TypedAttr> GreaterOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                           ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<GreaterOp>(
      getType(), inputs, b, [&](auto adaptor) {
        return b.createKgenExpr(KGEN::POC::LE,
                                {adaptor.getInputY(), adaptor.getInputX()});
      });
}

ErrorOr<TypedAttr> GreaterEqualOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<GreaterEqualOp>(
      getType(), inputs, b, [&](auto adaptor) {
        return b.createKgenExpr(KGEN::POC::LT,
                                {adaptor.getInputY(), adaptor.getInputX()});
      });
}

ErrorOr<TypedAttr> IndexToTensorOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                 ParamExprBuilder &b) {
  return inputs[0];
}

ErrorOr<TypedAttr> NegativeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                            ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<NegativeOp>(
      getType(), inputs, b, [&](auto adaptor) {
        Var input = b.createVar(adaptor.getInput());
        Var res = input * -1;
        return res.getAttr();
      });
}

ErrorOr<TypedAttr> NotEqualOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                            ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<NotEqualOp>(
      getType(), inputs, b, [&](auto adaptor) {
        auto eq = b.createKgenExpr(KGEN::POC::EQ, adaptor.getOperands());
        auto neq = KGEN::ParamOperatorAttr::getNot(eq);
        return b.wrapNonTrivialAttrWithDecl(neq, "not_equal");
      });
}

ErrorOr<TypedAttr> MaxOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<MaxOp>(getType(), inputs, b, [&](auto adaptor) {
    return b.createKgenExpr(KGEN::POC::Max, adaptor.getOperands());
  });
}

ErrorOr<TypedAttr> MinOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<MinOp>(getType(), inputs, b, [&](auto adaptor) {
    return b.createKgenExpr(KGEN::POC::Min, adaptor.getOperands());
  });
}

ErrorOr<TypedAttr> MulOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<MulOp>(getType(), inputs, b, [&](auto adaptor) {
    return b.createKgenExpr(KGEN::POC::MulNoWrap, adaptor.getOperands());
  });
}

ErrorOr<TypedAttr> NotOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<NotOp>(getType(), inputs, b, [&](auto adaptor) {
    auto attr = KGEN::ParamOperatorAttr::getNot(adaptor.getInput());
    return b.wrapNonTrivialAttrWithDecl(attr, "not");
  });
}

ErrorOr<TypedAttr> RangeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                         ParamExprBuilder &b) {
  StaticizeAdaptor<RangeOp> adaptor(inputs);

  std::optional<int64_t> startOr = tryGetConstantInt(adaptor.getStart());
  std::optional<int64_t> limitOr = tryGetConstantInt(adaptor.getLimit());
  std::optional<int64_t> stepOr = tryGetConstantInt(adaptor.getStep());
  if (!startOr.has_value() || !limitOr.has_value() || !stepOr.has_value())
    return TypedAttr{};
  int64_t start = startOr.value();
  int64_t limit = limitOr.value();
  int64_t step = stepOr.value();

  ErrorOr<int64_t> numOutElemsOr = getRangeNumElements(start, limit, step);
  if (numOutElemsOr.isError())
    return numOutElemsOr.takeError();
  int64_t numOutElems = numOutElemsOr.get();

  SmallVector<TypedAttr> outElems;
  int64_t elem = start;
  for (int64_t i = 0; i < numOutElems; ++i) {
    outElems.push_back(b.createVar(elem).getAttr());
    elem += step;
  }
  auto resAttr = b.getShapeAttr(outElems);
  return resAttr;
}

ErrorOr<TypedAttr> RebindOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  return StaticizeAdaptor<RebindOp>(inputs).getInput();
}

ErrorOr<TypedAttr> TransferOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                            ParamExprBuilder &b) {
  return StaticizeAdaptor<TransferOp>(inputs).getInput();
}

ErrorOr<TypedAttr> ReduceMulOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                             ParamExprBuilder &b) {
  StaticizeAdaptor<ReduceMulOp> adaptor(inputs);

  // We only staticize over 1D tensors for now, so axis must be 0.
  if (getAxisAttr().getInt() != 0)
    return TypedAttr{};

  auto inputElems = dyn_cast<ShapeAttr>(adaptor.getInput());
  if (!inputElems)
    return TypedAttr{};

  TypedAttr productAttr =
      b.createKgenExpr(KGEN::POC::MulNoWrap, inputElems.getValues());
  return b.getShapeAttr({productAttr});
}

ErrorOr<TypedAttr> ReshapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                           ParamExprBuilder &b) {
  StaticizeAdaptor<ReshapeOp> adaptor(inputs);
  return tryStaticizeReshapeLikeOp(getInput().getType(), getType(),
                                   adaptor.getInput(), b);
}

ErrorOr<TypedAttr> SelectOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                          ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<SelectOp>(
      getType(), inputs, b, [&](auto adaptor) {
        Var condVar = b.createVar(adaptor.getCond());
        Var thenVar = b.createVar(adaptor.getX());
        Var elseVar = b.createVar(adaptor.getY());
        Var res = b.createCond(condVar, thenVar, elseVar);
        return res.getAttr();
      });
}

ErrorOr<TypedAttr> SliceOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                         ParamExprBuilder &b) {
  StaticizeAdaptor<SliceOp> adaptor(inputs);

  auto inputElems = dyn_cast<ShapeAttr>(adaptor.getInput());
  if (!inputElems)
    return TypedAttr{};

  std::optional<SmallVector<int64_t>> startOr =
      tryGetConstantIntVector(adaptor.getStart());
  std::optional<SmallVector<int64_t>> stopOr =
      tryGetConstantIntVector(adaptor.getStop());
  std::optional<SmallVector<int64_t>> stepOr =
      tryGetConstantIntVector(adaptor.getStep());
  if (!(startOr.has_value() && startOr->size() == 1 && stopOr.has_value() &&
        stopOr->size() == 1 && stepOr.has_value() && stepOr->size() == 1))
    return TypedAttr{};

  // Only support step = 1 for now.
  if (stepOr->front() != 1)
    return TypedAttr{};

  int64_t inputSize = inputElems.getSize();
  int64_t normalizedStart = clampAxis(startOr->front(), inputSize);
  int64_t normalizedStop = clampAxis(stopOr->front(), inputSize);
  if (normalizedStop < normalizedStart) {
    std::stringstream ss;
    ss << "[slice] clamped stop (" << normalizedStop
       << ") cannot be smaller than clamped start (" << normalizedStart << ")";
    return Error(ss.str());
  }

  ArrayRef<TypedAttr> slicedInputElems = inputElems.getValues().slice(
      normalizedStart, normalizedStop - normalizedStart);
  auto resultAttr = b.getShapeAttr(slicedInputElems);
  return resultAttr;
}

ErrorOr<TypedAttr> SqueezeShapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                ParamExprBuilder &b) {
  StaticizeAdaptor<SqueezeShapeOp> adaptor(inputs);

  auto inputShapeAttr = dyn_cast<ShapeAttr>(adaptor.getInputShape());
  auto squeezeAxesOr = tryGetConstantIntVector(adaptor.getRemoveIndices());
  if (!inputShapeAttr || !squeezeAxesOr.has_value())
    return TypedAttr{};

  SmallVector<int64_t> &squeezeAxes = squeezeAxesOr.value();
  DenseSet<int64_t> normalizedSqueezeAxisSet;
  int64_t inputShapeLen = inputShapeAttr.getValues().size();
  for (auto &squeezeAxis : squeezeAxes) {
    if (squeezeAxis < -inputShapeLen || squeezeAxis >= inputShapeLen)
      return Error("Squeeze axis not in [-in_rank, in_rank).");

    if (squeezeAxis < 0)
      squeezeAxis += inputShapeLen;
    normalizedSqueezeAxisSet.insert(squeezeAxis);
  }

  SmallVector<TypedAttr> finalShape;
  for (auto [axis, dimAttr] : llvm::enumerate(inputShapeAttr.getValues())) {
    if (normalizedSqueezeAxisSet.contains(axis)) {
      b.createAssert(b.createVar(dimAttr) == 1,
                     "Attempting to unsqueeze a dimension that is not 1.");
      continue;
    }
    finalShape.push_back(dimAttr);
  }
  auto squeezedShapeAttr = b.getShapeAttr(finalShape);

  return squeezedShapeAttr;
}

ErrorOr<TypedAttr> Static::ReshapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                   ParamExprBuilder &b) {
  StaticizeAdaptor<Static::ReshapeOp> adaptor(inputs);
  return tryStaticizeReshapeLikeOp(getInput().getType(), getType(),
                                   adaptor.getInput(), b);
}

ErrorOr<TypedAttr> SubOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                       ParamExprBuilder &b) {
  return tryStaticizeElemWiseOp<SubOp>(getType(), inputs, b, [&](auto adaptor) {
    Var lhs = b.createVar(adaptor.getInputX());
    Var rhs = b.createVar(adaptor.getInputY());
    Var res = lhs - rhs;
    return res.getAttr();
  });
}

ErrorOr<TypedAttr> UnsqueezeShapeOp::tryStaticize(ArrayRef<TypedAttr> inputs,
                                                  ParamExprBuilder &b) {
  StaticizeAdaptor<UnsqueezeShapeOp> adaptor(inputs);

  auto inputShapeAttr = dyn_cast<ShapeAttr>(adaptor.getInputShape());
  auto unsqueezeAxesOr = tryGetConstantIntVector(adaptor.getPaddingIndices());
  if (!inputShapeAttr || !unsqueezeAxesOr.has_value())
    return TypedAttr{};

  const SmallVector<int64_t> &unsqueezeAxes = unsqueezeAxesOr.value();
  DenseSet<int64_t> unsqueezeAxesSet;
  int64_t outputShapeLen =
      inputShapeAttr.getValues().size() + unsqueezeAxes.size();
  for (auto unsqueezeAxis : unsqueezeAxes) {
    if (unsqueezeAxis < -outputShapeLen || unsqueezeAxis >= outputShapeLen)
      return Error("Unsqueeze axis not in [-out_rank, out_rank).");

    if (unsqueezeAxis < 0)
      unsqueezeAxis += outputShapeLen;
    unsqueezeAxesSet.insert(unsqueezeAxis);
  }

  SmallVector<TypedAttr> unsqueezedDims;
  TypedAttr intOneAttr = b.createVar(1).getAttr();
  for (int64_t idx = 0, nextDimAttrIdx = 0; idx < outputShapeLen; ++idx) {
    if (unsqueezeAxesSet.contains(idx))
      unsqueezedDims.push_back(intOneAttr);
    else
      unsqueezedDims.push_back(inputShapeAttr[nextDimAttrIdx++]);
  }
  auto unsqueezedShapeAttr =
      ShapeAttr::get(unsqueezedDims, inputShapeAttr.getType());

  return unsqueezedShapeAttr;
}

} // namespace M::MO
