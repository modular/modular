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
// This file implements the RMO dialect operations that are analogue with MO.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GraphCompiler/RMODialect/RMOOps.h"
#include "GraphCompiler/Support/Dims.h"
#include "Support/AssertStream.h"
#include "Support/ML/RangeUtils.h"
#include <optional>

using namespace M;
using namespace RMO;

using MO::TensorType;
using MO::TensorValue;

//===----------------------------------------------------------------------===//
// RMODialect op support
//===----------------------------------------------------------------------===//

/// Create a new attribute representing a static integer dimension.
static TypedAttr createStaticDimensionType(MO::GraphOp op, int64_t dimSize) {
  return ShapeAttr::getDim(op.getContext(), dimSize);
}

/// Emits a diagnostic and returns false if no explicit result type was given.
/// Output symbolic dims are user-declared per the Graph API contract, so the
/// compiler must not invent `RMO_D*` symbols. On false, callers return early,
/// leaving `moResultTypes` empty so verification surfaces the error as a
/// Python exception rather than an abort.
static bool requireRmoResults(MO::GraphOp graphOp,
                              std::optional<ValueRange> rmoResults,
                              llvm::StringRef opMnemonic) {
  if (rmoResults.has_value() && !rmoResults->empty())
    return true;
  graphOp->emitOpError()
      << "rmo.mo." << opMnemonic
      << " requires an explicit result type at construction time";
  return false;
}

/// Returns the dtype of tensor `values[index]`, or `fallbackDType` if `values`
/// is absent.
static DType getDType(std::optional<ValueRange> values, size_t index,
                      DType fallbackDType) {
  if (!values.has_value())
    return fallbackDType;

  return cast<TensorType>(values.value()[index].getType()).getDtype();
}

TensorType RMO::refineShapeWithDType(TensorType ty1, TensorType ty2,
                                     DType dtype) {
  // TODO(AIPIPE-312): Assert matching devices once the graph API propagates
  // them.
  ASSERT_STREAM(ty1.getRank() == ty2.getRank(),
                << " tried to refine types with different ranks");
  SmallVector<TypedAttr> newDims;
  auto refineDim = [&ty1, &ty2](size_t index) {
    if (ty1.isIntegerDim(index) && ty2.isIntegerDim(index))
      ASSERT_STREAM(
          ty1.getIntDim(index) == ty2.getIntDim(index),
          << " trying to refine two tensors whose integer dimensions don't "
             "match.");

    // Prefer integers over anything else
    if (ty1.isIntegerDim(index))
      return ty1.getDim(index);
    if (ty2.isIntegerDim(index))
      return ty2.getDim(index);

    // Otherwise both are parameters, just favor the first
    return ty1.getDim(index);
  };
  for (size_t i = 0, e = ty1.getRank(); i < e; ++i)
    newDims.push_back(refineDim(i));

  return TensorType::get(ty1.getContext(), newDims, dtype,
                         /*deviceRef=*/ty1.getDeviceRef(), ty1.getMetadata());
}

// Keeps ty1's dtype.
TensorType RMO::refineShape(TensorType ty1, TensorType ty2) {
  return refineShapeWithDType(ty1, ty2, ty1.getDtype());
}

//===----------------------------------------------------------------------===//
// RMODialect MO analogue ops
//===----------------------------------------------------------------------===//

void MOShapeOfOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // Match the declared result dtype when known.
  DType outDType = getDType(rmoResults, 0, DType::si64);
  TensorValue input = cast<TensorValue>(operands.getInput());

  int64_t rank = input.getType().getRank();
  TensorType outType = TensorType::get(
      context, SmallVector<TypedAttr>{createStaticDimensionType(graphOp, rank)},
      outDType, input.getType().getDeviceRef());

  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moResultTypes.push_back(outType);
  moOutputParamDecls.push_back({});
}

void MOBroadcastShapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // Inputs are rank-1 shape tensors (MO_ShapeLikeIfRanked); pass through.
  moOperandTypes.push_back(cast<TensorType>(operands.getInputX().getType()));
  moOperandTypes.push_back(cast<TensorType>(operands.getInputY().getType()));
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "broadcast_shape"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOReshapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `new_shape` is rank-1 by typed constraint.
  moOperandTypes.push_back(cast<TensorType>(operands.getInput().getType()));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(cast<TensorType>(operands.getNewShape().getType()));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "reshape"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOBroadcastToOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `new_shape` is rank-1 by typed constraint.
  moOperandTypes.push_back(cast<TensorType>(operands.getInput().getType()));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(cast<TensorType>(operands.getNewShape().getType()));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "broadcast_to"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOCastOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // Without a declared result this is a no-op cast; the `build` below is the
  // useful inferred-type builder.
  TensorValue input = cast<TensorValue>(operands.getInput());
  DType outputDType = getDType(rmoResults, 0, input.getType().getDtype());

  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  moResultTypes.push_back(input.getType().cloneWith(outputDType));
  moOutputParamDecls.push_back({});
}

void MOCastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, DType dtype) {
  build(odsBuilder, odsState, input.getType().cloneWith(dtype), input);
}

static void inferConvMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue filter, TensorValue strides, TensorValue dilations,
    TensorValue paddings, TensorValue outputPaddings, TensorValue numGroups,
    bool isTranspose, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // input / filter / strides / dilations / paddings / output_paddings pass
  // through (rank-1 window operands are enforced by MO_Rank1IndexTensor).
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(filter.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(strides.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(dilations.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(paddings.getType());
  moInputParamDecls.push_back({});

  if (isTranspose) {
    moOperandTypes.push_back(outputPaddings.getType());
    moInputParamDecls.push_back({});
  } else {
    // num_groups must be a scalar.
    moOperandTypes.push_back(
        numGroups.getType().cloneWith(SmallVector<TypedAttr>{}));
    moInputParamDecls.push_back({});
  }

  if (!requireRmoResults(graphOp, rmoResults,
                         isTranspose ? "conv_transpose" : "conv"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOConvOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  inferConvMOTypesImpl(context, graphOp, cast<TensorValue>(operands.getInput()),
                       cast<TensorValue>(operands.getFilter()),
                       cast<TensorValue>(operands.getStrides()),
                       cast<TensorValue>(operands.getDilations()),
                       cast<TensorValue>(operands.getPaddings()), nullptr,
                       cast<TensorValue>(operands.getNumGroups()), false,
                       rmoResults, moOperandTypes, moResultTypes,
                       moInputParamDecls, moOutputParamDecls);
}

void MOConvTransposeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  inferConvMOTypesImpl(context, graphOp, cast<TensorValue>(operands.getInput()),
                       cast<TensorValue>(operands.getFilter()),
                       cast<TensorValue>(operands.getStrides()),
                       cast<TensorValue>(operands.getDilations()),
                       cast<TensorValue>(operands.getPaddings()),
                       cast<TensorValue>(operands.getOutputPaddings()), nullptr,
                       true, rmoResults, moOperandTypes, moResultTypes,
                       moInputParamDecls, moOutputParamDecls);
}

static void inferMOTypesPoolImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue filterShape, TensorValue strides, TensorValue dilations,
    TensorValue paddings, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // input / filter_shape / strides / dilations / paddings pass through
  // (rank-1 window operands are enforced by MO_Rank1IndexTensor).
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(filterShape.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(strides.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(dilations.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(paddings.getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "pool"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

#define DEFINE_POOL_INFER_MO_TYPES(ClassName)                                  \
  void ClassName::inferMOTypesImpl(                                            \
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,             \
      std::optional<ValueRange> rmoResults,                                    \
      SmallVectorImpl<Type> &moOperandTypes,                                   \
      SmallVectorImpl<Type> &moResultTypes,                                    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) { \
    inferMOTypesPoolImpl(                                                      \
        context, graphOp, cast<TensorValue>(operands.getInput()),              \
        cast<TensorValue>(operands.getFilterShape()),                          \
        cast<TensorValue>(operands.getStrides()),                              \
        cast<TensorValue>(operands.getDilations()),                            \
        cast<TensorValue>(operands.getPaddings()), rmoResults, moOperandTypes, \
        moResultTypes, moInputParamDecls, moOutputParamDecls);                 \
  }

DEFINE_POOL_INFER_MO_TYPES(MOAvgPoolOp)
DEFINE_POOL_INFER_MO_TYPES(MOAvgPoolCeilOp)
DEFINE_POOL_INFER_MO_TYPES(MOMaxPoolOp)
DEFINE_POOL_INFER_MO_TYPES(MOMaxPoolCeilOp)

#undef DEFINE_POOL_INFER_MO_TYPES

void MOSelectOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  MO::TensorType condType = cast<MO::TensorType>(operands.getCond().getType());
  MO::TensorType xType = cast<MO::TensorType>(operands.getX().getType());
  MO::TensorType yType = cast<MO::TensorType>(operands.getY().getType());

  MO::TensorType refinedShape = refineShapeWithDType(
      refineShape(xType, yType), condType, xType.getDtype());
  moOperandTypes.push_back(refinedShape.cloneWith(DType::kBool));
  moOperandTypes.push_back(refinedShape);
  moOperandTypes.push_back(refinedShape);
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});

  moResultTypes.push_back(refinedShape);
  moOutputParamDecls.push_back({});
}

static void inferMOTypesSqueezeShapeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue inputShape,
    TensorValue indices, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // Inputs are rank-1 shape tensors by typed constraint; pass through.
  moOperandTypes.push_back(inputShape.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(indices.getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "squeeze_shape"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOSqueezeShapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesSqueezeShapeImpl(
      context, graphOp, cast<TensorValue>(operands.getInputShape()),
      cast<TensorValue>(operands.getRemoveIndices()), rmoResults,
      moOperandTypes, moResultTypes, moInputParamDecls, moOutputParamDecls);
}
