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

#include "GraphCompiler/MODialect/Patterns/Algebraic.h"

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "Support/ML/Permutation.h"

using namespace M;
using namespace MO;
using namespace GraphCompiler::Matchers;

namespace {

template <typename T>
static inline ErrorOr<MO::ConstantOp>
createScalarConstantOpOfType(TensorType type, T value,
                             DeviceAwareRewriter &rewriter, Location loc) {
  auto createConstantOp = [&](auto v) {
    return rewriter.create<MO::ConstantOp>(loc, v, type.getDeviceRef());
  };
  return type.getDtype()
      .dispatch<ErrorOr<MO::ConstantOp>>()
      .template when<M::DType::si32>(
          [&]() { return createConstantOp(int32_t(value)); })
      .template when<M::DType::si64>(
          [&]() { return createConstantOp(int64_t(value)); })
      .template when<M::DType::f32>(
          [&]() { return createConstantOp(float(value)); })
      .template when<M::DType::f64>(
          [&]() { return createConstantOp(double(value)); })
      .template when<M::DType::kBool>(
          [&]() { return createConstantOp(bool(value)); })
      .otherwise([&]() { return Error("Unsupported dtype"); });
}

template <typename T>
static inline ErrorOr<Value>
createBroadcastedScalarConstantOpOfType(TensorType bcastType, T value,
                                        DeviceAwareRewriter &rewriter,
                                        Location loc) {
  auto cstErrOr = createScalarConstantOpOfType(bcastType, value, rewriter, loc);
  if (cstErrOr.isError())
    return Error(cstErrOr.getError());

  return rewriter.create<MO::Static::BroadcastToOp>(loc, bcastType, *cstErrOr);
}

template <typename OpType, typename T, typename Func>
static LogicalResult
replaceOpResultWithBcastOfScalarIf(OpType op, T value,
                                   DeviceAwareRewriter &rewriter, Func &&func) {
  if (!func())
    return failure();

  auto bcastErrOr = createBroadcastedScalarConstantOpOfType(
      op.getType(), value, rewriter, op.getLoc());
  if (bcastErrOr.isError())
    return rewriter.notifyMatchFailure(op, bcastErrOr.getError());

  rewriter.replaceAllUsesWith(op, *bcastErrOr);

  return success();
}

/// Rewrites a batch matmul whose rhs is broadcast on the batch axis into a 2D
/// matmul, reshaping lhs to flatten the batch axis and the result to restore
/// it. This enables more matmul optimizations.
///
/// lhs : tensor<[4, 3, 2]>
/// rhs : tensor<[2, 3]>
///
///   bcast = broadcast(rhs) : ... -> tensor<[4, 2, 3]>
///   res   = batch_matmul(lhs, rhs) : ... -> tensor<[4, 3, 3]>
/// ===>
///   flat_lhs = reshape(lhs) : ... -> tensor<[12, 2]>
///   res_2d   = matmul(flat_lhs, rhs) : ... -> tensor<[12, 3]>
///   res      = reshape(res_2d) : ... -> tensor<[4, 3, 3]>
///
/// A transpose on `rhs` is kept. Aggressive transpose sinking puts it *after*
/// the static broadcast:
///
///   bcast = broadcast(rhs) : ... -> tensor<[4, 3, 2]>
///   transpose = transpose(rhs) : ... -> tensor<[4, 2, 3]>
///   res   = batch_matmul(lhs, transpose) : ... -> tensor<[4, 3, 3]>
/// ===>
///   transpose = transpose(rhs) : ... -> tensor<[2, 3]>
///   flat_lhs = reshape(lhs) : ... -> tensor<[12, 2]>
///   res_2d   = matmul(flat_lhs, transpose) : ... -> tensor<[12, 3]>
///   res      = reshape(res_2d) : ... -> tensor<[4, 3, 3]>
struct ConvertBatchMatmulToMatmulPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, BatchMatmulOp> {
  ConvertBatchMatmulToMatmulPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 4) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::BatchMatmulOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input, bcastedTensor;
    Operation *broadcast = nullptr;
    Operation *transposeOpCapture = nullptr;
    auto weightBcastP = m_Op<Static::BroadcastToOp>(
        &broadcast, m_MOMatrix(m_Any(&bcastedTensor)));
    auto weightMaybeTransposed = m_Optional(
        m_Op<TransposeOp>(&transposeOpCapture, weightBcastP, m_Any()),
        weightBcastP);

    if (!m_Op<BatchMatmulOp>(m_Any(&input),
                             m_Optional(m_Op<RebindOp>(rewriter.pushOp(),
                                                       weightMaybeTransposed),
                                        weightMaybeTransposed))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    TransposeOp transposeOp;
    // The transpose may only permute the inner-most 2 (non-batch) dims.
    if (transposeOpCapture) {
      transposeOp = cast<TransposeOp>(transposeOpCapture);
      std::optional<SmallVector<int64_t>> maybeInts =
          tryGetConstantIntVector(transposeOp.getPerm());

      if (!maybeInts.has_value())
        return rewriter.notifyMatchFailure(
            transposeOp, "Transpose does not have static perm.");

      SmallVector<int64_t> &transposePerm = maybeInts.value();
      int64_t permSize = static_cast<int64_t>(transposePerm.size());
      for (int64_t i = 0; i < permSize; ++i) {
        if (i < permSize - 2 && transposePerm[i] != i)
          return rewriter.notifyMatchFailure(
              transposeOp, "Transpose affects more than matmul dims.");

        if (i >= permSize - 2 && transposePerm[i] == i)
          return rewriter.notifyMatchFailure(
              transposeOp, "Transpose is a identity permutation.");
      }
    }

    // weight refers to the RHS input of the batched_matmul.
    Value weight = transposeOp ? transposeOp.getResult() : bcastedTensor;

    if (!input || !weight)
      return rewriter.notifyMatchFailure(op, "null input value");
    auto inputTy = cast<MO::TensorType>(input.getType());
    auto bcastedTensorTy = cast<MO::TensorType>(bcastedTensor.getType());
    auto weightTy = cast<MO::TensorType>(weight.getType());
    auto broadcastTy = cast<MO::TensorType>(broadcast->getResult(0).getType());
    auto bmmTy = cast<MO::TensorType>(op->getResult(0).getType());
    if (!bmmTy.hasRank()) {
      return rewriter.notifyMatchFailure(
          op, "ops aren't parameterized, can't rewrite");
    }
    ArrayRef<TypedAttr> outDims =
        broadcastTy.getDims().drop_front(broadcastTy.getRank() - 2);
    ArrayRef<TypedAttr> inDims = bcastedTensorTy.getDims();

    if (!inDims.equals(outDims)) {
      return rewriter.notifyMatchFailure(op, "Broadcast not on batch axis");
    }

    if (!rewriter.addOps({broadcast})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    ArrayRef<TypedAttr> oldShape = inputTy.getDims();
    ArrayRef<TypedAttr> kVec = oldShape.take_back();
    TypedAttr batch = KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap,
                                                   oldShape.drop_back(1));
    SmallVector<TypedAttr> newShape{batch, kVec[0]};

    auto shapeTy = rewriter.getType<ShapeType>();
    MO::TensorType reshapeTy =
        inputTy.cloneWith(ShapeAttr::get(newShape, shapeTy));
    Value reshape =
        rewriter.create<Static::ReshapeOp>(op->getLoc(), reshapeTy, input)
            .getResult();

    ArrayRef<TypedAttr> weightShape = weightTy.getDims();
    MO::TensorType matmulTy =
        bmmTy.cloneWith(ShapeAttr::get({batch, weightShape.back()}, shapeTy));

    Value matmul;
    if (transposeOp) {
      auto perm =
          rewriter.createHostConstant(op->getLoc(), SmallVector<int64_t>{1, 0});
      TensorType bcastedTensorTy = cast<TensorType>(bcastedTensor.getType());
      ArrayRef<TypedAttr> oldbcastedTensorShape = bcastedTensorTy.getDims();
      SmallVector<TypedAttr> newTensorShape = {oldbcastedTensorShape[1],
                                               oldbcastedTensorShape[0]};
      auto newWeight = rewriter.create<TransposeOp>(
          op->getLoc(), bcastedTensorTy.cloneWith(newTensorShape),
          bcastedTensor, perm);
      matmul =
          rewriter.create<MatmulOp>(op->getLoc(), matmulTy, reshape, newWeight)
              .getResult();
    } else {
      matmul =
          rewriter.create<MatmulOp>(op->getLoc(), matmulTy, reshape, weight)
              .getResult();
    }

    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, bmmTy, matmul);

    return mlir::success();
  }
};

/// Converts a batch matmul with batch size one to a matmul. A broadcast that
/// cannot be sunk becomes a reshape.
///
///   lhs : tensor<[1, 1, 3, 2]>
///   rhs : tensor<[1, 1, 2, 3]>
///
///   res   = batch_matmul(lhs, rhs) : ... -> tensor<[1, 1, 2, 2]>
/// ===>
///   reshape_lhs = reshape(lhs) : ... -> tensor<[3, 2]>
///   reshape_rhs = reshape(rhs) : ... -> tensor<[2, 3]>
///   matmul      = matmul(reshape_lhs, reshape_rhs) : ... -> tensor<[2, 2]>
///   res         = broadcast_to(matmul) : ... -> tensor<[1, 1, 2, 2]>
struct ConvertBatchMatmulBatchSizeOneToMatmulPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, BatchMatmulOp> {
  ConvertBatchMatmulBatchSizeOneToMatmulPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 5) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::BatchMatmulOp op,
                             DeviceAwareRewriter &rewriter) const override {

    TensorType aType = op.getTensorA().getType();
    TensorType bType = op.getTensorB().getType();
    TensorType cType = op.getResult().getType();

    if (!aType.hasRank())
      return rewriter.notifyMatchFailure(
          op, "rank must be known in order to match");

    auto rank = aType.getRank();
    // Verifier ensures rank >= 3, so requiredOneDims >= 1
    auto requiredOneDims = rank - 2;

    for (size_t i = 0, e = (requiredOneDims); i < e; ++i)
      if ((aType.getIntDim(i) != 1) || (bType.getIntDim(i) != 1))
        return rewriter.notifyMatchFailure(op, "failed to match pattern");

    auto resTy = op.getResult().getType();

    SmallVector<TypedAttr> aShapeDims;
    SmallVector<TypedAttr> bShapeDims;
    SmallVector<TypedAttr> resShapeDims;
    for (auto i = requiredOneDims; i < rank; ++i) {
      aShapeDims.push_back(aType.getDim(i));
      bShapeDims.push_back(bType.getDim(i));
      resShapeDims.push_back(resTy.getDim(i));
    }

    auto newAType = aType.cloneWith(aShapeDims);
    auto newBType = bType.cloneWith(bShapeDims);
    auto newResType = cType.cloneWith(resShapeDims);

    auto newATensor = rewriter.create<Static::ReshapeOp>(op.getLoc(), newAType,
                                                         op.getTensorA());
    auto newBTensor = rewriter.create<Static::ReshapeOp>(op.getLoc(), newBType,
                                                         op.getTensorB());

    auto matmul = rewriter.create<MatmulOp>(op.getLoc(), newResType, newATensor,
                                            newBTensor);
    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(op, resTy, matmul);
    return success();
  }
};

/// Rewrites a `mo.conv` with a 1x1 filter, unit strides and dilations, zero
/// paddings, and num_groups=1 into a matmul between static reshapes.
///
/// input  : tensor<[N, H, W, Ci]>
/// filter : tensor<[1, 1, Ci, Co]>
///
///   res = conv(input, filter)[
///     strides = {1, ...}, dilations = {1, ...}, paddings = {0, ...},
///     num_groups = 1] : Sh
/// ===>
///   input'  = reshape(input)  : tensor<[N, H, W, Ci]> -> tensor<[N*H*W, Ci]>
///   filter' = reshape(filter) : tensor<[1, 1, Ci, Co]> -> tensor<[Ci, Co]>
///   matmul  = matmul(input', filter')
///   res     = reshape(matmul) : ... -> Sh
struct ConvToMatmulPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, ConvOp> {
  ConvToMatmulPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ConvOp op,
                             DeviceAwareRewriter &rewriter) const override {

    TensorValue input = op.getInput();
    TensorValue filter = op.getFilter();
    TensorType opTy = op.getType();
    Location loc = op.getLoc();

    TensorType inputTy = input.getType();
    TensorType filterTy = filter.getType();
    if (!inputTy.hasRank() || inputTy.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "input rank must be 4");

    // Detect filter layout: FCRS has spatial dims at [2,3], RSCF at [0,1].
    LayoutAttr filterLayout = filterTy.getLayoutAttr();
    bool isFCRS = filterLayout && (filterLayout.getFormat() == "FCRS" ||
                                   filterLayout.getFormat() == "FCQRS");

    ArrayRef<TypedAttr> inputDims = inputTy.getShape().getValues();
    ArrayRef<TypedAttr> filterDims = filterTy.getShape().getValues();
    TypedAttr intOneAttr = rewriter.getDimAttr(1);
    size_t spatialStart = isFCRS ? 2 : 0;
    if (filterDims[spatialStart] != intOneAttr ||
        filterDims[spatialStart + 1] != intOneAttr)
      return rewriter.notifyMatchFailure(op, "filter size must be 1x1");

    // Dilations are irrelevant for a 1x1 filter.
    auto areConstant = [](TensorValue val, int64_t check) {
      std::optional<SmallVector<int64_t>> maybeInts =
          tryGetConstantIntVector(val);
      return maybeInts.has_value() &&
             llvm::all_of(maybeInts.value(),
                          [&check](int64_t n) { return n == check; });
    };
    auto isConstant = [](TensorValue val, int64_t check) {
      std::optional<int64_t> maybeInt = tryGetConstantInt(val);
      return maybeInt && (maybeInt == check);
    };
    if (!areConstant(op.getStrides(), 1))
      return rewriter.notifyMatchFailure(op, "strides must be all 1s");
    if (!areConstant(op.getPaddings(), 0))
      return rewriter.notifyMatchFailure(op, "paddings must be all 0s");
    if (!isConstant(op.getNumGroups(), 1))
      return rewriter.notifyMatchFailure(op, "num_groups must be 1");

    // Both layouts produce a [Ci,Co] = [K,N] weight.
    auto shapeTy = rewriter.getType<MOSH::ShapeType>();
    // TODO provide a better API for symbolic dimension manipulations
    auto matmulDimM = KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap,
                                                   inputDims.slice(0, 3));
    TypedAttr matmulDimK = inputDims.back(); // Ci from NHWC input
    TypedAttr matmulDimN = isFCRS ? filterDims[0] : filterDims.back(); // Co

    auto matmulInputShape =
        MOSH::ShapeAttr::get({matmulDimM, matmulDimK}, shapeTy);
    auto matmulShape = MOSH::ShapeAttr::get({matmulDimM, matmulDimN}, shapeTy);
    auto matmulInputTy = inputTy.cloneWith(matmulInputShape);
    auto matmulTy = opTy.cloneWith(matmulShape);
    auto matmulInput =
        rewriter.create<MO::Static::ReshapeOp>(loc, matmulInputTy, input);

    auto matmulWeightShape =
        MOSH::ShapeAttr::get({matmulDimK, matmulDimN}, shapeTy);
    auto matmulWeightTy = filterTy.cloneWith(matmulWeightShape);
    Value matmulWeight;
    if (isFCRS) {
      // FCRS [Co,Ci,1,1] → reshape [Co,Ci] → transpose to [Ci,Co]. Not
      // transpose_b=true: MatmulOp::canonicalize rederives it from the weight
      // layout attribute.
      auto reshapeShape =
          MOSH::ShapeAttr::get({matmulDimN, matmulDimK}, shapeTy);
      auto reshapeTy = filterTy.cloneWith(reshapeShape);
      auto reshaped =
          rewriter.create<MO::Static::ReshapeOp>(loc, reshapeTy, filter);
      auto perm = rewriter.createHostConstant(loc, SmallVector<int64_t>{1, 0});
      matmulWeight =
          rewriter.create<MO::TransposeOp>(loc, matmulWeightTy, reshaped, perm);
    } else {
      // RSCF [1,1,Ci,Co] → reshape [Ci,Co]
      matmulWeight =
          rewriter.create<MO::Static::ReshapeOp>(loc, matmulWeightTy, filter);
    }

    auto matmul =
        rewriter.create<MO::MatmulOp>(loc, matmulTy, matmulInput, matmulWeight);
    rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(op, opTy, matmul);
    return mlir::success();
  }
};

/// Flattens single-use nested `mo.concat`s on the same axis into one.
///
/// t0 : tensor<[1, ...]>
/// t1 : tensor<[2, ...]>
/// t2 : tensor<[3, ...]>
///
///   res0 = concat[axis_0](t0, t1)
///   res1 = concat[axis_0](res0, t2)
/// ===>
///   res1 = concat[axis_0](t0, t1, t2)
struct FlattenNestedConcatPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, ConcatOp> {
  FlattenNestedConcatPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ConcatOp op,
                             DeviceAwareRewriter &rewriter) const override {

    int64_t axis = op.getAxisAttr().getInt();

    bool foundSingleUserNestedConcat = false;
    SmallVector<Value> flattenedInputs;
    for (Value input : op.getInputs()) {
      auto nestedConcatOp = input.getDefiningOp<ConcatOp>();
      if (nestedConcatOp && nestedConcatOp->hasOneUse() &&
          nestedConcatOp.getAxisAttr().getInt() == axis) {
        rewriter.pushOp(nestedConcatOp);
        llvm::append_range(flattenedInputs, nestedConcatOp.getInputs());
        foundSingleUserNestedConcat = true;
      } else {
        flattenedInputs.push_back(input);
      }
    }

    if (!foundSingleUserNestedConcat) {
      return rewriter.notifyMatchFailure(
          op, "failed to find single-user nested concat");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // Types match: shape inference verified them statically or dropped
    // runtime asserts.
    rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(), op.getAxisAttr(),
                                          flattenedInputs);
    return mlir::success();
  }
};

struct BroadcastReshapePattern
    : public DeviceAwareRewritePattern<OpRewritePattern, Static::ReshapeOp> {
  BroadcastReshapePattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(Static::ReshapeOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input;
    if (!m_Op<Static::ReshapeOp>(
             m_Op<Static::BroadcastToOp>(rewriter.pushOp(), m_Any(&input)))
             .match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input)
      return rewriter.notifyMatchFailure(op, "null input value");

    TensorType inType = llvm::cast<TensorType>(input.getType());

    if (!inType.hasRank())
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    for (size_t i = 0, e = inType.getRank(); i < e; ++i)
      if (inType.getIntDim(i) != 1)
        return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorType scalarType =
        inType.cloneWith(SmallVector<int64_t>{}, inType.getElementType());
    auto squash =
        rewriter.create<Static::ReshapeOp>(op.getLoc(), scalarType, input);

    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(op, op.getType(),
                                                       squash);
    return mlir::success();
  }
};

// clang-format off
/// Merges matmuls that share activations to increase throughput.
/// BEFORE:
/// matmul(A, B1) -> C1
/// matmul(A, B2) -> C2
/// matmul(A, B3) -> C3
/// AFTER:
/// matmul(A, concat(B1, B2, B3)) -> C -> split -> C1
///                                            |-> C2
///                                            |-> C3
// clang-format on
template <typename MatmulOpType>
struct MergeParallelMatmulPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, MatmulOpType> {
  MergeParallelMatmulPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern<OpRewritePattern, MatmulOpType>(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MatmulOpType op,
                             DeviceAwareRewriter &rewriter) const override {
    Value activations = op.getInputA();
    SmallVector<MatmulOpType> matmulUsers;
    for (Operation *user : activations.getUsers()) {
      if (auto matmulOp = dyn_cast<MatmulOpType>(user)) {
        rewriter.pushOp(matmulOp);
        matmulUsers.push_back(matmulOp);
      }
    }

    if (matmulUsers.size() == 1)
      return rewriter.notifyMatchFailure(op, "No parallel matmuls");

    if (llvm::all_of(matmulUsers, [](MatmulOpType matmul) {
          return llvm::all_of(matmul->getUsers(), [](Operation *user) {
            return isa<OutputOp>(user);
          });
        })) {
      // Outputs would force the split to materialize, which is slower.
      return rewriter.notifyMatchFailure(
          op, "Matmuls are all feeding into output ops");
    }

    auto oldMatmulType = cast<TensorType>(op.getType());
    TypedAttr expectedNumberOfRows =
        oldMatmulType.getDim(oldMatmulType.getRank() - 2);
    if (!llvm::all_of(matmulUsers, [&](MatmulOpType matmulOp) {
          auto resultType = matmulOp.getType();

          if (!resultType.hasRank())
            return false;

          if (resultType.getDim(0) != expectedNumberOfRows)
            return false;

          if (mlir::ShapedType::isDynamic(resultType.getIntDim(1)))
            return false;

          if (!MOGG::dependsOnNoAliasConstantLikesOnly(
                  matmulOp.getInputB().getDefiningOp()))
            return false;

          // TODO(GEX-1978): Skipped to unblock GEX-1968, as it can forbid
          // `ForceMatmulTransposeBForGPU`. Sink the transpose after the concat
          // instead to enable both patterns:
          //
          // mo.transpose
          // mo.matmul
          // mo.transpose
          // mo.matmul
          // =>
          // mo.concat
          // mo.transpose
          // mo.matmul
          //
          // rather than the current:
          //
          // mo.transpose
          // mo.transpose
          // mo.concat
          // mo.matmul
          if (isa<MO::TransposeOp>(matmulOp.getInputB().getDefiningOp()))
            return false;

          return true;
        })) {
      return rewriter.notifyMatchFailure(
          op, "Parallel matmuls must have the same number of rows, a static "
              "number of columns and constant weights");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // The last user is the earliest in the block.
    auto firstMatmulIndex = matmulUsers.size() - 1;
    Location loc = matmulUsers[firstMatmulIndex]->getLoc();
    rewriter.setInsertionPoint(matmulUsers[firstMatmulIndex]);

    SmallVector<Value> weightsToConcatenate;
    SmallVector<Type> matmulTypes;
    SmallVector<int64_t> columnsInEachMatmul;
    int64_t totalColumns = 0;
    for (MatmulOpType matmulOp : matmulUsers) {
      weightsToConcatenate.push_back(matmulOp.getInputB());
      TensorType matmulType = matmulOp.getType();
      matmulTypes.push_back(matmulType);

      size_t matmulRank = matmulType.getRank();
      columnsInEachMatmul.push_back(matmulType.getIntDim(matmulRank - 1));
      totalColumns += columnsInEachMatmul.back();
    }

    auto concatType =
        cast<TensorType>(op.getInputB().getType())
            .cloneWithNewDim(oldMatmulType.getRank() - 1, totalColumns);

    int64_t innerDimIndex = oldMatmulType.getRank() - 1;
    Value concatenatedWeights =
        rewriter
            .create<ConcatOp>(loc, concatType,
                              rewriter.getIndexAttr(innerDimIndex),
                              weightsToConcatenate)
            .getResult();

    auto newMatmulType = oldMatmulType.cloneWithNewDim(
        oldMatmulType.getRank() - 1, totalColumns);
    Value newMatmul = rewriter
                          .create<MatmulOpType>(loc, newMatmulType, activations,
                                                concatenatedWeights)
                          .getResult();

    Value splitSizes = rewriter.createHostConstant(loc, columnsInEachMatmul);
    auto splitOp =
        rewriter.create<SplitOp>(loc, matmulTypes, newMatmul, splitSizes,
                                 llvm::APInt(64, innerDimIndex));

    for (uint64_t idx = 0; idx < matmulUsers.size(); ++idx) {
      rewriter.replaceAllUsesWith(matmulUsers[idx].getResult(),
                                  splitOp.getResult(idx));
      // Erase to avoid re-doing the same optimization on the other matmuls.
      rewriter.eraseOp(matmulUsers[idx]);
    }

    return mlir::success();
  }
};

// add(x, 0) -> x
struct ArithmeticFoldingAddWithZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::AddOp> {
  ArithmeticFoldingAddWithZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::AddOp addOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_BinaryCommutativeOp<MO::AddOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 0))
             .match(addOp)) {
      return failure();
    }
    rewriter.replaceAllUsesWith(addOp, x);

    return success();
  }
};

// sub(x, 0) -> x
struct ArithmeticFoldingSubWithZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::SubOp> {
  ArithmeticFoldingSubWithZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::SubOp subOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::SubOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 0))
             .match(subOp)) {
      return failure();
    }
    rewriter.replaceAllUsesWith(subOp, x);

    return success();
  }
};

// sub(x, x) -> bcast(0)
struct ArithmeticFoldingSubWithSame
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::SubOp> {
  ArithmeticFoldingSubWithSame(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::SubOp subOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(subOp, 0, rewriter, [&]() {
      return subOp.getInputX() == subOp.getInputY();
    });

    return success();
  }
};

// mul(x, 1) -> x
struct ArithmeticFoldingMulWithOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  ArithmeticFoldingMulWithOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_BinaryCommutativeOp<MO::MulOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1))
             .match(mulOp)) {
      return failure();
    }
    rewriter.replaceAllUsesWith(mulOp, x);

    return success();
  }
};

// mul(x, 0) -> bcast(0)
struct ArithmeticFoldingMulWithZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  ArithmeticFoldingMulWithZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(mulOp, 0, rewriter, [&]() {
      return m_BinaryCommutativeOp<MO::MulOp>(
                 m_Any(), m_SplatOrBroadcastedConstantOfAnyType<
                              MO::Static::BroadcastToOp>(0))
          .match(mulOp);
    });

    return success();
  }
};

// div(0, x) -> bcast(0)
struct ArithmeticFoldingZeroDiv
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  ArithmeticFoldingZeroDiv(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(divOp, 0, rewriter, [&]() {
      return m_Op<MO::DivOp>(m_SplatOrBroadcastedConstantOfAnyType<
                                 MO::Static::BroadcastToOp>(0),
                             m_Any())
          .match(divOp);
    });
  }
};

// div(x, 1) -> x
struct ArithmeticFoldingDivByOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  ArithmeticFoldingDivByOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::DivOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1))
             .match(divOp)) {
      return failure();
    }
    rewriter.replaceAllUsesWith(divOp, x);

    return success();
  }
};

// pow(x, 1) -> x
struct ArithmeticFoldingPowerByOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ArithmeticFoldingPowerByOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::PowOp>(m_Any(&x), m_SplatConstantOfAnyType(1)).match(powOp))
      return failure();

    rewriter.replaceAllUsesWith(powOp, x);

    return success();
  }
};

// pow(x, 0) -> bcast(1)
struct ArithmeticFoldingPowByZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ArithmeticFoldingPowByZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(powOp, 1, rewriter, [&]() {
      return m_Op<MO::PowOp>(m_Any(), m_SplatOrBroadcastedConstantOfAnyType<
                                          MO::Static::BroadcastToOp>(0))
          .match(powOp);
    });
  }
};

// pow(0, y) -> bcast(0)
struct ArithmeticFoldingPowerOfZero
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ArithmeticFoldingPowerOfZero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(powOp, 0, rewriter, [&]() {
      return m_Op<MO::PowOp>(m_SplatOrBroadcastedConstantOfAnyType<
                                 MO::Static::BroadcastToOp>(0),
                             m_Any())
          .match(powOp);
    });
  }
};

// pow(1, y) -> bcast(1)
struct ArithmeticFoldingPowerOfOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ArithmeticFoldingPowerOfOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(powOp, 1, rewriter, [&]() {
      return m_Op<MO::PowOp>(m_SplatOrBroadcastedConstantOfAnyType<
                                 MO::Static::BroadcastToOp>(1),
                             m_Any())
          .match(powOp);
    });

    return success();
  }
};

// max(x, x) -> x
struct ArithmeticFoldingMaxSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MaxOp> {
  ArithmeticFoldingMaxSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MaxOp maxOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (maxOp.getInputX() != maxOp.getInputY())
      return failure();

    rewriter.replaceAllUsesWith(maxOp, maxOp.getInputX());

    return success();
  }
};

// min(x, x) -> x
struct ArithmeticFoldingMinSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MinOp> {
  ArithmeticFoldingMinSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MinOp minOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (minOp.getInputX() != minOp.getInputY())
      return failure();

    rewriter.replaceAllUsesWith(minOp, minOp.getInputX());

    return success();
  }
};

// min(x, max(x, y)) -> x
struct ArithmeticFoldingMinOfMax
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MinOp> {
  ArithmeticFoldingMinOfMax(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MinOp minOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_BinaryCommutativeOp<MO::MinOp>(
             m_Any(&x), m_BinaryCommutativeOp<MO::MaxOp>(m_Any(&y), m_Any(&z)))
             .match(minOp)) {
      return failure();
    }

    if (x != z && x != y)
      return failure();

    rewriter.replaceAllUsesWith(minOp, x);

    return success();
  }
};

// max(x, min(x, y)) -> x
struct ArithmeticFoldingMaxOfMin
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MaxOp> {
  ArithmeticFoldingMaxOfMin(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MaxOp maxOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_BinaryCommutativeOp<MO::MaxOp>(
             m_Any(&x), m_BinaryCommutativeOp<MO::MinOp>(m_Any(&y), m_Any(&z)))
             .match(maxOp)) {
      return failure();
    }

    if (x != z && x != y)
      return failure();

    rewriter.replaceAllUsesWith(maxOp, x);

    return success();
  }
};

// mod(x, x) -> bcast(0)
struct ArithmeticFoldingModSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::ModOp> {
  ArithmeticFoldingModSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::ModOp modOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(modOp, 0, rewriter, [&]() {
      return modOp.getInputX() == modOp.getInputY() &&
             !m_Optional(
                  m_SplatConstantOfAnyType(0),
                  m_Op<MO::Static::BroadcastToOp>(m_SplatConstantOfAnyType(0)))
                  .match(modOp.getInputX());
    });
  }
};

// mod(x, 1) -> bcast(0)
struct ArithmeticFoldingModByOne
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::ModOp> {
  ArithmeticFoldingModByOne(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::ModOp modOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(modOp, 0, rewriter, [&]() {
      return m_Op<MO::ModOp>(
                 m_Any(), m_Optional(m_Optional(m_SplatConstant(int64_t(1)),
                                                m_SplatConstant(int32_t(1))),
                                     m_Op<MO::Static::BroadcastToOp>(m_Optional(
                                         m_SplatConstant(int64_t(1)),
                                         m_SplatConstant(int32_t(1))))))
          .match(modOp);
    });
  }
};

// >=(x, x) -> bcast(true)
struct ArithmeticFoldingGreaterOrEqualThanSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::GreaterEqualOp> {
  ArithmeticFoldingGreaterOrEqualThanSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::GreaterEqualOp geOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(geOp, true, rewriter, [&]() {
      return geOp.getInputX() == geOp.getInputY();
    });

    return success();
  }
};

// >(x, x) -> bcast(false)
struct ArithmeticFoldingGreaterThanSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::GreaterOp> {
  ArithmeticFoldingGreaterThanSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::GreaterOp gtOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(gtOp, false, rewriter, [&]() {
      return gtOp.getInputX() == gtOp.getInputY();
    });
  }
};

// !=(x, x) -> bcast(false)
struct ArithmeticFoldingNotEqualSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::NotEqualOp> {
  ArithmeticFoldingNotEqualSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::NotEqualOp neOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(neOp, false, rewriter, [&]() {
      return neOp.getInputX() == neOp.getInputY();
    });

    return success();
  }
};

// ==(x, x) -> bcast(true)
struct ArithmeticFoldingEqualSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::EqualOp> {
  ArithmeticFoldingEqualSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::EqualOp eqOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(eqOp, true, rewriter, [&]() {
      return eqOp.getInputX() == eqOp.getInputY();
    });
  }
};

// and(x, x) -> x
struct ArithmeticFoldingAndSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::AndOp> {
  ArithmeticFoldingAndSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::AndOp andOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (andOp.getInputX() != andOp.getInputY())
      return failure();

    rewriter.replaceAllUsesWith(andOp, andOp.getInputX());

    return success();
  }
};

// or(x, x) -> x
struct ArithmeticFoldingOrSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::OrOp> {
  ArithmeticFoldingOrSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::OrOp orOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (orOp.getInputX() != orOp.getInputY())
      return failure();

    rewriter.replaceAllUsesWith(orOp, orOp.getInputX());

    return success();
  }
};

// xor(x, x) -> bcast(false)
struct ArithmeticFoldingXorSameOperand
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::XorOp> {
  ArithmeticFoldingXorSameOperand(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::XorOp xorOp,
                             DeviceAwareRewriter &rewriter) const override {
    return replaceOpResultWithBcastOfScalarIf(xorOp, false, rewriter, [&]() {
      return xorOp.getInputX() == xorOp.getInputY();
    });
  }
};

// pow(pow(x, y), z) = pow(x, y * z)
// This pattern can be repeatedly applied for arbitrarily long chains.
struct MathematicalCanonicalizationChainsOfPowToSinglePow
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  MathematicalCanonicalizationChainsOfPowToSinglePow(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_Op<MO::PowOp>(m_Op<MO::PowOp>(m_Any(&x), m_Any(&y)), m_Any(&z))
             .match(powOp)) {
      return failure();
    }

    if (y.getType() != z.getType())
      return failure();

    Location loc = powOp->getLoc();
    auto add = rewriter.create<MO::MulOp>(loc, y.getType(), y, z);
    auto newPow = rewriter.create<MO::PowOp>(loc, powOp.getType(), x, add);
    rewriter.replaceAllUsesWith(powOp, newPow);

    return success();
  }
};

// rsqrt(x) -> div(bcast(1), sqrt(x))
struct ForwardMathematicalCanonicalizationRsqrtToDivSqrt
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::RsqrtOp> {
  ForwardMathematicalCanonicalizationRsqrtToDivSqrt(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::RsqrtOp rsqrtOp,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = rsqrtOp->getLoc();
    TensorType tensorType = rsqrtOp.getType();
    auto bcast1ErrOr =
        createBroadcastedScalarConstantOpOfType(tensorType, 1, rewriter, loc);
    if (bcast1ErrOr.isError())
      return rewriter.notifyMatchFailure(rsqrtOp, bcast1ErrOr.getError());
    auto sqrt =
        rewriter.create<MO::SqrtOp>(loc, tensorType, rsqrtOp.getInput());
    auto div = rewriter.create<MO::DivOp>(loc, tensorType, *bcast1ErrOr, sqrt);
    rewriter.replaceAllUsesWith(rsqrtOp, div);

    return success();
  }
};

// div(bcast(1), sqrt(x)) -> rsqrt(x)
struct BackwardMathematicalCanonicalizationDivSqrtToRsqrt
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  BackwardMathematicalCanonicalizationDivSqrtToRsqrt(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::DivOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1),
             m_Op<MO::SqrtOp>(m_Any(&x)))
             .match(divOp)) {
      return failure();
    }
    auto rsqrt =
        rewriter.create<MO::RsqrtOp>(divOp->getLoc(), divOp.getType(), x);
    rewriter.replaceAllUsesWith(divOp, rsqrt);

    return success();
  }
};

// Builds an activation op of type `OpT` that replaces `root`, reusing `root`'s
// location and result type.
template <typename OpT>
static OpT createActivationOp(DeviceAwareRewriter &rewriter,
                              mlir::Operation *root, Value input) {
  return rewriter.create<OpT>(
      root->getLoc(), llvm::cast<TensorType>(root->getResult(0).getType()),
      input);
}

// relu(x) -> max(x, bcast(0))
struct ForwardMathematicalCanonicalizationReluToMax
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::ReluOp> {
  ForwardMathematicalCanonicalizationReluToMax(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::ReluOp reluOp,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = reluOp->getLoc();
    TensorType tensorType = reluOp.getType();

    auto bcast0ErrOr =
        createBroadcastedScalarConstantOpOfType(tensorType, 0, rewriter, loc);
    if (bcast0ErrOr.isError())
      return rewriter.notifyMatchFailure(reluOp, bcast0ErrOr.getError());
    auto maxOp = rewriter.create<MO::MaxOp>(loc, tensorType, reluOp.getInput(),
                                            *bcast0ErrOr);
    rewriter.replaceAllUsesWith(reluOp, maxOp);

    return success();
  }
};

// max(x, bcast(0)) -> mo.relu(x)
struct BackwardMathematicalCanonicalizationMaxToRelu
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MaxOp> {
  BackwardMathematicalCanonicalizationMaxToRelu(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MaxOp maxOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_BinaryCommutativeOp<MO::MaxOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 0),
             m_Any(&x))
             .match(maxOp)) {
      return failure();
    }
    auto relu = createActivationOp<MO::ReluOp>(rewriter, maxOp, x);
    rewriter.replaceAllUsesWith(maxOp, relu);

    return success();
  }
};

// div(1, add(1, exp(-x))) -> mo.sigmoid(x)
struct BackwardMathematicalCanonicalizationExpToSigmoid
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  BackwardMathematicalCanonicalizationExpToSigmoid(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::DivOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1),
             m_BinaryCommutativeOp<MO::AddOp>(
                 m_SplatOrBroadcastedConstantOfAnyType<
                     MO::Static::BroadcastToOp>(1),
                 m_Op<MO::ExpOp>(m_Op<MO::NegativeOp>(m_Any(&x)))))
             .match(divOp)) {
      return failure();
    }
    auto sigmoid = createActivationOp<MO::SigmoidOp>(rewriter, divOp, x);
    rewriter.replaceAllUsesWith(divOp, sigmoid);

    return success();
  }
};

// mul(x, mo.sigmoid(x)) -> mo.silu(x)
struct BackwardMathematicalCanonicalizationSigmoidMulToSilu
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  BackwardMathematicalCanonicalizationSigmoidMulToSilu(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    // Returns true if `act` is `mo.sigmoid(in)`.
    auto isSigmoidOf = [](Value act, Value in) {
      auto sigmoidOp = act.getDefiningOp<MO::SigmoidOp>();
      return sigmoidOp && sigmoidOp.getInput() == in;
    };

    Value lhs = mulOp.getInputX();
    Value rhs = mulOp.getInputY();
    Value x;
    if (isSigmoidOf(lhs, rhs))
      x = rhs;
    else if (isSigmoidOf(rhs, lhs))
      x = lhs;
    else
      return failure();

    auto silu = createActivationOp<MO::SiluOp>(rewriter, mulOp, x);
    rewriter.replaceAllUsesWith(mulOp, silu);

    return success();
  }
};

// mul(x, mo.sigmoid(mul(1.702, x))) -> mo.gelu_quick(x)
struct BackwardMathematicalCanonicalizationSigmoidMulToGeluQuick
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  BackwardMathematicalCanonicalizationSigmoidMulToGeluQuick(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 3) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    // Returns true if `act` is `mo.sigmoid(mul(1.702, in))`.
    auto isQuickSigmoidOf = [](Value act, Value in) {
      auto sigmoidOp = act.getDefiningOp<MO::SigmoidOp>();
      if (!sigmoidOp)
        return false;
      Value inner;
      if (!m_BinaryCommutativeOp<MO::MulOp>(
               m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                   1.702),
               m_Any(&inner))
               .match(sigmoidOp.getInput()))
        return false;
      return inner == in;
    };

    Value lhs = mulOp.getInputX();
    Value rhs = mulOp.getInputY();
    Value x;
    if (isQuickSigmoidOf(lhs, rhs))
      x = rhs;
    else if (isQuickSigmoidOf(rhs, lhs))
      x = lhs;
    else
      return failure();

    auto gelu = createActivationOp<MO::GeluQuickOp>(rewriter, mulOp, x);
    rewriter.replaceAllUsesWith(mulOp, gelu);

    return success();
  }
};

// mul(mul(0.5, x), add(1, erf(x / sqrt(2)))) -> mo.gelu(x)
struct BackwardMathematicalCanonicalizationErfToGelu
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  BackwardMathematicalCanonicalizationErfToGelu(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    static constexpr double kSqrt2 = 1.4142135623730951;
    static constexpr double kInvSqrt2 = 0.7071067811865476;

    // The erf argument is `x / sqrt(2)`. With fast-math enabled the core
    // canonicalizations rewrite the division into a multiplication by the
    // reciprocal, so accept both forms (and the reciprocal whether or not it
    // has been constant-folded yet).
    auto matchScaledInput = [](Value arg, Value *base) -> bool {
      if (matchPattern(arg,
                       m_Op<MO::DivOp>(m_Any(base),
                                       m_SplatOrBroadcastedConstantOfAnyType<
                                           MO::Static::BroadcastToOp>(kSqrt2))))
        return true;
      if (m_BinaryCommutativeOp<MO::MulOp>(
              m_Any(base),
              m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                  kInvSqrt2))
              .match(arg))
        return true;
      return m_BinaryCommutativeOp<MO::MulOp>(
                 m_Any(base), m_Op<MO::Static::BroadcastToOp>(m_Op<MO::DivOp>(
                                  m_SplatOrBroadcastedConstantOfAnyType<
                                      MO::Static::BroadcastToOp>(1),
                                  m_SplatOrBroadcastedConstantOfAnyType<
                                      MO::Static::BroadcastToOp>(kSqrt2))))
          .match(arg);
    };

    Value xHalf;
    Value erfArg;
    auto halfX = m_BinaryCommutativeOp<MO::MulOp>(
        m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(0.5),
        m_Any(&xHalf));
    auto onePlusErf = m_BinaryCommutativeOp<MO::AddOp>(
        m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(1),
        m_Op<MO::ErfOp>(m_Any(&erfArg)));
    if (!m_BinaryCommutativeOp<MO::MulOp>(halfX, onePlusErf).match(mulOp))
      return failure();

    Value xErf;
    if (!matchScaledInput(erfArg, &xErf) || xErf != xHalf)
      return failure();

    auto gelu = createActivationOp<MO::GeluOp>(rewriter, mulOp, xHalf);
    rewriter.replaceAllUsesWith(mulOp, gelu);

    return success();
  }
};

// mul(mul(0.5, x),
//     add(1, tanh(mul(0.7978845608028654, add(x, mul(0.044715, x^3))))))
//   -> mo.gelu_tanh(x)
struct BackwardMathematicalCanonicalizationTanhToGeluTanh
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  BackwardMathematicalCanonicalizationTanhToGeluTanh(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    // Returns true if `v` is `pow(x, 3)` or the equivalent `x * x * x` (the
    // forward canonicalization decomposes the cube into multiplications).
    auto isSquareOf = [](Value v, Value x) {
      auto sqOp = v.getDefiningOp<MO::MulOp>();
      return sqOp && sqOp.getInputX() == x && sqOp.getInputY() == x;
    };
    auto isCubeOf = [&](Value v, Value x) -> bool {
      Value base;
      if (matchPattern(v, m_Op<MO::PowOp>(m_Any(&base),
                                          m_SplatOrBroadcastedConstantOfAnyType<
                                              MO::Static::BroadcastToOp>(3))))
        return base == x;
      auto cubeOp = v.getDefiningOp<MO::MulOp>();
      if (!cubeOp)
        return false;
      return (isSquareOf(cubeOp.getInputX(), x) && cubeOp.getInputY() == x) ||
             (isSquareOf(cubeOp.getInputY(), x) && cubeOp.getInputX() == x);
    };

    Value xHalf;
    Value xPoly;
    Value cube;
    auto halfX = m_BinaryCommutativeOp<MO::MulOp>(
        m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(0.5),
        m_Any(&xHalf));
    auto poly = m_BinaryCommutativeOp<MO::AddOp>(
        m_Any(&xPoly),
        m_BinaryCommutativeOp<MO::MulOp>(
            m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                0.044715),
            m_Any(&cube)));
    auto onePlusTanh = m_BinaryCommutativeOp<MO::AddOp>(
        m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(1),
        m_Op<MO::TanhOp>(m_BinaryCommutativeOp<MO::MulOp>(
            m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                0.7978845608028654),
            poly)));
    if (!m_BinaryCommutativeOp<MO::MulOp>(halfX, onePlusTanh).match(mulOp))
      return failure();
    if (xHalf != xPoly || !isCubeOf(cube, xHalf))
      return failure();

    auto gelu = createActivationOp<MO::GeluTanhOp>(rewriter, mulOp, xHalf);
    rewriter.replaceAllUsesWith(mulOp, gelu);

    return success();
  }
};

// mul(x, bcast(2)) = add(x, x)
struct ForwardMathematicalCanonicalizationMulByTwoToAdd
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  ForwardMathematicalCanonicalizationMulByTwoToAdd(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_BinaryCommutativeOp<MO::MulOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 2),
             m_Any(&x))
             .match(mulOp)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<MO::AddOp>(mulOp, mulOp.getType(), x, x);

    return success();
  }
};

// add(x, x) = mul(x, bcast(2))
struct BackwardMathematicalCanonicalizationMulByTwoToAdd
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::AddOp> {
  BackwardMathematicalCanonicalizationMulByTwoToAdd(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::AddOp addOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (addOp.getInputX() != addOp.getInputY()) {
      return failure();
    }

    auto tensorType = addOp.getType();
    auto loc = addOp->getLoc();
    auto bcast2ErrOr =
        createBroadcastedScalarConstantOpOfType(tensorType, 2, rewriter, loc);
    if (bcast2ErrOr.isError())
      return rewriter.notifyMatchFailure(addOp, bcast2ErrOr.getError());

    rewriter.replaceOpWithNewOp<MO::MulOp>(addOp, tensorType, addOp.getInputX(),
                                           *bcast2ErrOr);

    return success();
  }
};

// pow(x, 2) = mul(x, x)
struct ForwardMathematicalCanonicalizationPowByTwoToMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ForwardMathematicalCanonicalizationPowByTwoToMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::PowOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 2))
             .match(powOp)) {
      return failure();
    }
    auto square =
        rewriter.create<MO::MulOp>(powOp->getLoc(), powOp.getType(), x, x);
    rewriter.replaceAllUsesWith(powOp, square);

    return success();
  }
};

// pow(x, 3) = mul(mul(x, x), x)
struct ForwardMathematicalCanonicalizationPowByThreeToMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ForwardMathematicalCanonicalizationPowByThreeToMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::PowOp>(
             m_Any(&x),
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 3))
             .match(powOp)) {
      return failure();
    }
    auto square =
        rewriter.create<MO::MulOp>(powOp->getLoc(), powOp.getType(), x, x);
    auto cube =
        rewriter.create<MO::MulOp>(powOp->getLoc(), powOp.getType(), square, x);
    rewriter.replaceAllUsesWith(powOp, cube);

    return success();
  }
};

// pow(x, -{1, 2}) = div(1, pow(x, {1, 2})
struct ForwardMathematicalCanonicalizationPowByNegativeOneOrTwoToDivAndPow
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  ForwardMathematicalCanonicalizationPowByNegativeOneOrTwoToDivAndPow(
      MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    Value one;
    Value two;
    if (!m_Op<MO::PowOp>(m_Any(&x),
                         m_Optional(m_SplatOrBroadcastedConstantOfAnyType<
                                        MO::Static::BroadcastToOp>(&one, -1),
                                    m_SplatOrBroadcastedConstantOfAnyType<
                                        MO::Static::BroadcastToOp>(&two, -2)))
             .match(powOp)) {
      return failure();
    }

    auto tensorType = powOp.getType();
    auto loc = powOp->getLoc();

    auto bcastErrOr = createBroadcastedScalarConstantOpOfType(
        tensorType, one ? 1 : 2, rewriter, loc);
    if (bcastErrOr.isError())
      return rewriter.notifyMatchFailure(powOp, bcastErrOr.getError());

    auto bcast1ErrOr =
        createBroadcastedScalarConstantOpOfType(tensorType, 1, rewriter, loc);
    if (bcast1ErrOr.isError())
      return rewriter.notifyMatchFailure(powOp, bcast1ErrOr.getError());

    rewriter.modifyOpInPlace(powOp, [&]() { powOp.setRhsInput(*bcastErrOr); });
    rewriter.setInsertionPointAfter(powOp);
    auto div = rewriter.create<MO::DivOp>(loc, tensorType, *bcast1ErrOr, powOp);
    rewriter.replaceAllUsesExcept(powOp, div, div);

    return success();
  }
};

// div(1, pow(x, {1, 2}) = pow(x, -{1, 2})
struct BackwardMathematicalCanonicalizationDivAndPowToPowByNegativeOneOrTwo
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  BackwardMathematicalCanonicalizationDivAndPowToPowByNegativeOneOrTwo(
      MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    Value one;
    Value two;
    if (!m_Op<MO::DivOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 1),
             m_Op<MO::PowOp>(
                 m_Any(&x),
                 m_Optional(m_SplatOrBroadcastedConstantOfAnyType<
                                MO::Static::BroadcastToOp>(&one, 1),
                            m_SplatOrBroadcastedConstantOfAnyType<
                                MO::Static::BroadcastToOp>(&two, 2))))
             .match(divOp)) {
      return failure();
    }

    assert(one || two);

    auto tensorType = divOp.getType();
    auto loc = divOp->getLoc();

    auto bcastErrOr = createBroadcastedScalarConstantOpOfType(
        tensorType, one ? -1 : -2, rewriter, loc);
    if (bcastErrOr.isError())
      return rewriter.notifyMatchFailure(divOp, bcastErrOr.getError());

    auto pow = rewriter.create<MO::PowOp>(loc, tensorType, x, *bcastErrOr);
    rewriter.replaceAllUsesWith(divOp, pow);

    return success();
  }
};

// sub(0, x) = negative(x)
struct MathematicalCanonicalizationZeroSubToNegative
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::SubOp> {
  MathematicalCanonicalizationZeroSubToNegative(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::SubOp subOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x;
    if (!m_Op<MO::SubOp>(
             m_SplatOrBroadcastedConstantOfAnyType<MO::Static::BroadcastToOp>(
                 0),
             m_Any(&x))
             .match(subOp)) {
      return failure();
    }
    auto neg =
        rewriter.create<MO::NegativeOp>(subOp->getLoc(), subOp.getType(), x);
    rewriter.replaceAllUsesWith(subOp, neg);

    return success();
  }
};

// add(x, negative(y)) = sub(x, y)
struct MathematicalSimplificationAddWithNegative
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::AddOp> {
  MathematicalSimplificationAddWithNegative(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::AddOp addOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y;
    if (!m_BinaryCommutativeOp<MO::AddOp>(m_Any(&x),
                                          m_Op<MO::NegativeOp>(m_Any(&y)))
             .match(addOp)) {
      return failure();
    }

    auto sub =
        rewriter.create<MO::SubOp>(addOp->getLoc(), addOp.getType(), x, y);
    rewriter.replaceAllUsesWith(addOp, sub);

    return success();
  }
};

// sub(x, negative(y)) = add(x, y)
struct MathematicalSimplificationSubWithNegative
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::SubOp> {
  MathematicalSimplificationSubWithNegative(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::SubOp subOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y;
    if (!m_Op<MO::SubOp>(m_Any(&x), m_Op<MO::NegativeOp>(m_Any(&y)))
             .match(subOp)) {
      return failure();
    }

    auto add =
        rewriter.create<MO::AddOp>(subOp->getLoc(), subOp.getType(), x, y);
    rewriter.replaceAllUsesWith(subOp, add);

    return success();
  }
};

// div(x, div(y, z)) = div(mul(x, z), y)
struct MathematicalSimplificationDivByDivToDivOfMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  MathematicalSimplificationDivByDivToDivOfMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_Op<MO::DivOp>(m_Any(&x), m_Op<MO::DivOp>(m_Any(&y), m_Any(&z)))
             .match(divOp)) {
      return failure();
    }

    auto mul =
        rewriter.create<MO::MulOp>(divOp->getLoc(), divOp.getType(), x, z);
    rewriter.modifyOpInPlace(divOp, [&]() {
      divOp.setLhsInput(mul);
      divOp.setRhsInput(y);
    });

    return success();
  }
};

// div(div(x, y), z) = div(x, mul(y, z))
struct MathematicalSimplificationDivOfDivToDivByMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  MathematicalSimplificationDivOfDivToDivByMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    Value x, y, z;
    if (!m_Op<MO::DivOp>(m_Op<MO::DivOp>(m_Any(&x), m_Any(&y)), m_Any(&z))
             .match(divOp)) {
      return failure();
    }

    auto mul =
        rewriter.create<MO::MulOp>(divOp->getLoc(), divOp.getType(), y, z);
    rewriter.modifyOpInPlace(divOp, [&]() {
      divOp.setLhsInput(x);
      divOp.setRhsInput(mul);
    });

    return success();
  }
};

static std::optional<int64_t> getMinBroadcastFactor(TensorType inputType,
                                                    TensorType bcastType) {
  if (!inputType.hasRank() || !bcastType.hasRank())
    return std::nullopt;

  ArrayRef<TypedAttr> inDims = inputType.getDims();
  ArrayRef<TypedAttr> outDims = bcastType.getDims();
  if (inDims.size() > outDims.size())
    return std::nullopt;

  size_t rankDiff = outDims.size() - inDims.size();
  int64_t factor = 1;
  for (auto [i, outDim] : llvm::enumerate(outDims)) {
    bool isBroadcastDim =
        i < rankDiff || tryGetConstantInt(inDims[i - rankDiff]) == 1;
    if (!isBroadcastDim)
      continue;
    if (auto outDimValue = tryGetConstantInt(outDim))
      factor *= *outDimValue;
  }
  return factor;
}

// div(x, bcast(CST_or_small)) = mul(x, bcast(div(bcast(1), CST_or_small)))
struct MathematicalSimplificationDivBySmallTensorToMul
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::DivOp> {
  MathematicalSimplificationDivBySmallTensorToMul(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::DivOp divOp,
                             DeviceAwareRewriter &rewriter) const override {
    auto tensorType = divOp.getType();
    Location loc = divOp->getLoc();
    if (!tensorType.getDtype().isFloat())
      return rewriter.notifyMatchFailure(loc, "input type isn't float");

    auto x = divOp.getInputX();
    auto y = divOp.getInputY();

    // This is to avoid looping.
    bool xConstantFoldable =
        dataflowGraphFoldsToConstants<MO::ConstantOp, ChainCreateOp>(
            x.getDefiningOp());
    if (xConstantFoldable)
      return failure();

    Value origY = y;
    if (auto bcast =
            dyn_cast_or_null<MO::Static::BroadcastToOp>(origY.getDefiningOp()))
      origY = bcast.getInput();

    auto origYType = llvm::cast<TensorType>(origY.getType());

    static constexpr int64_t kMulByDivReciprocalThroughputEstimate = 8;
    // Add 1 because it still takes time to do a multiplication.
    static constexpr int64_t kThreshold =
        kMulByDivReciprocalThroughputEstimate + 1;
    // This is checking if the broadcasted value is a constant or if it was
    // broadcasted by a factor of at least `kThreshold`.
    bool yConstantFoldable =
        dataflowGraphFoldsToConstants<MO::ConstantOp, ChainCreateOp>(
            y.getDefiningOp());
    if (!yConstantFoldable) {
      std::optional<int64_t> factor =
          getMinBroadcastFactor(origYType, tensorType);
      if (!factor || *factor < kThreshold)
        return failure();
    }

    auto bcast1ErrOr =
        createBroadcastedScalarConstantOpOfType(origYType, 1, rewriter, loc);
    if (bcast1ErrOr.isError())
      return rewriter.notifyMatchFailure(divOp, bcast1ErrOr.getError());

    auto div = rewriter.create<MO::DivOp>(loc, origYType, *bcast1ErrOr, origY);
    auto bcastDiv =
        rewriter.create<MO::Static::BroadcastToOp>(loc, tensorType, div);
    auto mul = rewriter.create<MO::MulOp>(loc, tensorType, x, bcastDiv);

    rewriter.replaceAllUsesWith(divOp, mul);

    return success();
  }
};

// mul(x, div(1, y)) = div(x, y)
// where y is not a constant or a broadcasted value.
// The inverse of the previous pattern, except for constants and broadcasts.
struct MathematicalSimplificationMulOfDivToDiv
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MulOp> {
  MathematicalSimplificationMulOfDivToDiv(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::MulOp mulOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (!mulOp.getInputY().getDefiningOp())
      return failure();

    Value x, y;
    if (!m_Op<MulOp>(m_Any(&x),
                     m_Op<DivOp>(m_SplatOrBroadcastedConstantOfAnyType<
                                     MO::Static::BroadcastToOp>(1),
                                 m_Any(&y)))
             .match(mulOp)) {
      return failure();
    }

    if (isa_and_nonnull<MO::Static::BroadcastToOp>(y.getDefiningOp()) ||
        dataflowGraphFoldsToConstants<MO::ConstantOp, ChainCreateOp>(
            y.getDefiningOp())) {
      return failure();
    }

    auto div =
        rewriter.create<MO::DivOp>(mulOp->getLoc(), mulOp.getType(), x, y);
    rewriter.replaceAllUsesWith(mulOp, div);

    return success();
  }
};

// pow(x, bcast(CST_float)) = pow(x, bcast(int(CST_float)))
// where CST_float == float(int(CST_float))
struct MathematicalSimplificationPowByFloatToPowByInt
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::PowOp> {
  MathematicalSimplificationPowByFloatToPowByInt(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}
  LogicalResult
  deviceAwareMatchAndRewrite(MO::PowOp powOp,
                             DeviceAwareRewriter &rewriter) const override {

    auto bcast = dyn_cast_or_null<MO::Static::BroadcastToOp>(
        powOp.getInputY().getDefiningOp());
    if (!bcast)
      return failure();

    auto cst =
        dyn_cast_or_null<MO::ConstantOp>(bcast.getInput().getDefiningOp());
    if (!cst)
      return failure();

    auto dtype = cst.getType().getDtype();
    if (!dtype.isFloat())
      return failure();

    std::optional<double> value = [&]() -> std::optional<double> {
      if (auto val = tryGetConstantFloat(cst))
        return *val;
      if (auto val = tryGetConstantDouble(cst))
        return *val;

      return {};
    }();

    if (!value)
      return rewriter.notifyMatchFailure(
          powOp, "unable to get the floating point exponent");

    double exponent;
    if (std::modf(*value, &exponent))
      return rewriter.notifyMatchFailure(
          powOp, "the exponent contains a fractional part");

    // Prefer 32 bits when the exponent fits.
    int bitWidth =
        std::abs(exponent) < std::numeric_limits<int32_t>::max() ? 32 : 64;

    auto bcastIntType = TensorType::get(
        powOp.getType().getShape(), rewriter.getIntegerType(bitWidth, true),
        powOp.getType().getDeviceRef(), powOp.getType().getMetadata());
    auto newCst = createBroadcastedScalarConstantOpOfType(
        bcastIntType, int64_t(exponent), rewriter, powOp.getLoc());
    if (newCst.isError())
      return rewriter.notifyMatchFailure(powOp, newCst.getError());

    rewriter.modifyOpInPlace(powOp, [&]() { powOp.setRhsInput(*newCst); });

    return success();
  }
};

/// ===>
///   res = conv(tmp, weight, strides, dilations, pads[2:-2] + paddings)
struct PadConvPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, ConvOp> {
  PadConvPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ConvOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input, filter, strides, dilations, pads, pads2, groups;
    if (!m_Op<ConvOp>(m_Op<PadConstantOp>(rewriter.pushOp(), m_Any(&input),
                                          m_Any(&pads), m_ScalarConstant(0.0)),
                      m_Any(&filter), m_Any(&strides), m_Any(&dilations),
                      m_Any(&pads2), m_Any(&groups))
             .match(op)) {
      return rewriter.notifyMatchFailure(op, "failed to match pattern");
    }

    if (!input || !filter || !strides || !dilations || !pads || !pads2 ||
        !groups) {
      return rewriter.notifyMatchFailure(op, "null input value");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    std::optional<SmallVector<int64_t>> maybePads =
        tryGetConstantIntVector(cast<TensorValue>(pads));

    if (!maybePads.has_value())
      return rewriter.notifyMatchFailure(op, "can't rewrite dynamic padding");

    auto constPads = maybePads.value();
    if (constPads[0] != 0 || constPads[1] != 0 || constPads.end()[-2] != 0 ||
        constPads.end()[-1] != 0) {
      return rewriter.notifyMatchFailure(
          op, "is padding more than the image dimensions");
    }

    auto loc = op.getLoc();
    Value newPads = rewriter.createHostConstant(
        loc, SmallVector<int64_t>{constPads.begin() + 2, constPads.end() - 2});
    // pads2 is sometimes si32.
    pads2 = rewriter.create<CastOp>(
        loc, cast<TensorType>(pads2.getType()).cloneWith(M::DType::si64),
        pads2);

    newPads = rewriter.create<AddOp>(loc, newPads, pads2);

    op.getInputMutable().assign(input);
    op.getPaddingsMutable().assign(newPads);
    return mlir::success();
  }
};

/// Rewrites gathers that together amount to a split into that split.
///
/// input : tensor<[N, ...]>
///
///   res0 = gather(input, /*indices=*/0, /*axis=*/0)
///   res1 = gather(input, /*indices=*/1, /*axis=*/0)
///   ...
///   resN = gather(input, /*indices=*/N, /*axis=*/0)
/// ===>
///   res  = split(input, /*splitSizes=*/[1, 1, ..., 1], /*axis=*/0)
///   res0 = res#0
///   res1 = res#1
///   ...
///   resN = res#N
struct GathersToSplitPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, GatherOp> {
  GathersToSplitPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(GatherOp op,
                             DeviceAwareRewriter &rewriter) const override {

    TensorValue input = op.getInput();
    TensorType inputTy = input.getType();
    if (!inputTy.hasRank())
      return rewriter.notifyMatchFailure(op, "unranked input");

    int64_t inputAxis0Dim = inputTy.getIntDim(0);
    if (inputAxis0Dim <= 0) {
      return rewriter.notifyMatchFailure(
          op, "input axis 0 isn't static and positive");
    }

    SmallDenseMap<size_t, GatherOp> idxToGather;
    MO::GatherOp dominantGatherOp; // one that domindates all other gather ops
    for (Operation *user : input.getUsers()) {
      auto userGatherOp = dyn_cast<GatherOp>(user);
      if (!userGatherOp)
        continue;

      if (userGatherOp.getAxisAttr().getInt() != 0)
        continue;

      std::optional<SmallVector<int64_t>> maybeIndices =
          tryGetConstantIntVector(userGatherOp.getIndices());
      if (!maybeIndices.has_value() || maybeIndices.value().size() != 1)
        continue;

      int64_t idx = maybeIndices.value().front();
      if (idx < 0 || idx >= inputAxis0Dim || idxToGather.contains(idx))
        continue;

      if (!dominantGatherOp)
        dominantGatherOp = userGatherOp;

      // TODO account for control-flow
      if (userGatherOp->getBlock() != dominantGatherOp->getBlock())
        continue;

      if (userGatherOp->isBeforeInBlock(dominantGatherOp))
        dominantGatherOp = userGatherOp;

      rewriter.pushOp(userGatherOp);
      idxToGather.try_emplace(idx, userGatherOp);
    }

    // The gathers must cover all of axis 0.
    if (idxToGather.size() != inputAxis0Dim) {
      return rewriter.notifyMatchFailure(
          op, "not enough qualified gather users for input");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    SmallVector<Type> resultTypes;
    for (int64_t idx = 0; idx < inputAxis0Dim; ++idx)
      resultTypes.push_back(idxToGather[idx].getType());

    // Insert before the dominant gather: the split must dominate all users, and
    // the constants stay nearby for debuggability.
    Location loc = dominantGatherOp.getLoc();
    rewriter.setInsertionPoint(dominantGatherOp);
    Value splitSizes = rewriter.createHostConstant(
        loc, SmallVector<int64_t>(inputAxis0Dim, 1));
    auto splitOp = rewriter.create<SplitOp>(loc, resultTypes, input, splitSizes,
                                            llvm::APInt(64, 0));

    for (auto [idx, gatherOp] : idxToGather)
      rewriter.replaceOp(gatherOp, splitOp.getResult(idx));

    return success();
  }
};

/// Returns `ty` with its dimensions permuted by `perm`.
static TensorType permuteDims(TensorType ty, ArrayRef<int64_t> perm) {
  SmallVector<TypedAttr> permutedDims =
      permute(ty.getShape().getValues(), perm);
  auto shapeTy = ShapeType::get(ty.getContext());
  return ty.cloneWith(ShapeAttr::get(permutedDims, shapeTy));
}

/// Rewrites a nonzero-axis split whose results are each squeezed (by reshape)
/// into a transpose, an axis-0 split, and the reshapes. An axis-0 split lowers
/// in place, and the transpose kernel is much faster than the split kernel.
///
/// input : tensor<[..., N, ...]>
///   ress = split(input, /*splitSizes=*/[1, 1, ..., 1], /*axis=*/...)
///   res0 = reshape(ress#0) : tensor<[..., 1, ...]> -> tensor<[...]>
///   res1 = reshape(ress#1) : tensor<[..., 1, ...]> -> tensor<[...]>
///   ...
///   resN = reshape(ress#1) : tensor<[..., 1, ...]> -> tensor<[...]>
/// ===>
///   inputT = transpose(input) : ... -> tensor<[N, ...]>
///   ress   = split(inputT, /*splitSizes=*/[1, 1, ..., 1], /*axis=*/0)
///   res0   = reshape(ress#0) : tensor<[1, ...]> -> tensor<[...]>
///   res1   = reshape(ress#1) : tensor<[1, ...]> -> tensor<[...]>
///   ...
///   resN   = reshape(ress#1) : tensor<[1, ...]> -> tensor<[...]>
struct SplitAndSqueezesPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SplitOp> {
  // TODO(#13349) Generalize or remove this pattern
  SplitAndSqueezesPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  // Whether `op` is a `mo.static.reshape` (optionally after a rebind) that
  // squeezes `squeezeAxis`.
  static bool isSqueezingReshape(DeviceAwareRewriter &rewriter, Operation *op,
                                 ArrayRef<TypedAttr> inferredInDims,
                                 size_t squeezeAxis) {
    rewriter.pushOp(op);

    // TODO(#13348) remove this check once we have shape inference for SplitOp
    if (auto rebindOp = dyn_cast<RebindOp>(op)) {
      if (!rebindOp->hasOneUse())
        return false;
      op = *rebindOp->user_begin();
      rewriter.pushOp(op);
    }

    auto reshapeOp = dyn_cast<Static::ReshapeOp>(op);
    if (!reshapeOp)
      return false;

    TensorType inTy = reshapeOp.getInput().getType();
    TensorType outTy = reshapeOp.getResult().getType();
    if (!inTy.hasRank() || !outTy.hasRank())
      return false;

    // TODO(#13348) simplify all these once we have shape inference for SplitOp
    size_t inRank = inTy.getRank();
    if (inRank != outTy.getRank() + 1 || inRank <= squeezeAxis)
      return false;

    auto intOneAttr = rewriter.getDimAttr(1);
    SmallVector<TypedAttr> expectedInDims(outTy.getShape().getValues());
    expectedInDims.insert(expectedInDims.begin() + squeezeAxis, intOneAttr);
    ArrayRef<TypedAttr> inDims = inTy.getShape().getValues();
    for (auto [inDim, inferredInDim, expectedInDim] :
         llvm::zip(inDims, inferredInDims, expectedInDims)) {
      TypedAttr resolvedInDim =
          isStaticDimAttr(inferredInDim) ? inferredInDim : inDim;
      if (resolvedInDim != expectedInDim)
        return false;
    }

    return true;
  }

  LogicalResult
  deviceAwareMatchAndRewrite(SplitOp op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue input = op.getInput();
    TensorType inputTy = input.getType();
    if (!inputTy.hasRank())
      return rewriter.notifyMatchFailure(op, "unranked input");
    int64_t inputRank = inputTy.getRank();

    int64_t splitAxis = op.getAxisAttr().getInt();
    splitAxis = (splitAxis < 0) ? inputRank + splitAxis : splitAxis;
    if (splitAxis <= 0) {
      return rewriter.notifyMatchFailure(
          op, "normalized split axis isn't positive");
    }

    int64_t inputSplitAxisDim = inputTy.getIntDim(splitAxis);
    if (mlir::ShapedType::isDynamic(inputSplitAxisDim)) {
      return rewriter.notifyMatchFailure(
          op, "input split axis dimension isn't static");
    }

    TensorValue splitSizes = op.getSplitSizes();
    std::optional<SmallVector<int64_t>> maybeSplitSizes =
        tryGetConstantIntVector(splitSizes);
    if (maybeSplitSizes != SmallVector<int64_t>(inputSplitAxisDim, 1)) {
      return rewriter.notifyMatchFailure(
          op, "split sizes isn't all ones of inputSplitAxisDim");
    }

    // Manual shape inference: `mo.split` has no shape function.
    ArrayRef<TypedAttr> inputDims = inputTy.getShape().getValues();
    SmallVector<TypedAttr> inferredResDims(inputDims);
    inferredResDims[splitAxis] = rewriter.getDimAttr(1);

    for (Value res : op.getResults()) {
      if (!res.hasOneUse())
        return rewriter.notifyMatchFailure(op, "a result has multiple users");

      if (!isSqueezingReshape(rewriter, *res.user_begin(), inferredResDims,
                              splitAxis)) {
        return rewriter.notifyMatchFailure(
            op, "a result isn't consumed by a reshape that squeezes");
      }
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    Location loc = op.getLoc();
    SmallVector<int64_t> permVec(inputRank);
    std::iota(permVec.begin() + 1, permVec.begin() + splitAxis + 1, 0);
    std::iota(permVec.begin() + splitAxis + 1, permVec.end(), splitAxis + 1);
    permVec[0] = splitAxis;
    Value perm = rewriter.createHostConstant(loc, permVec);

    TensorType transposedInputTy = permuteDims(inputTy, permVec);
    auto transposeOp =
        rewriter.create<TransposeOp>(loc, transposedInputTy, input, perm);

    SmallVector<Type> newSplitResTypes;
    for (Type oldResTy : op.getResultTypes()) {
      auto oldResTensorTy = cast<TensorType>(oldResTy);
      TensorType newResTy = permuteDims(oldResTensorTy, permVec);
      newSplitResTypes.push_back(newResTy);
    }
    auto newSplitOp = rewriter.replaceOpWithNewOp<SplitOp>(
        op, newSplitResTypes, transposeOp, splitSizes, llvm::APInt(64, 0),
        op.getOutputParamDecls());

    // Retype rebinds for the permuted results; their squeezing-reshape users
    // stay compatible.
    // TODO(#13348) remove once we have shape inference for SplitOp
    for (Value res : newSplitOp.getResults()) {
      ASSERT_STREAM(res.hasOneUse(), << "guards should've enforced this");
      if (auto rebindOp = dyn_cast<RebindOp>(*res.user_begin())) {
        TensorType oldResTy = rebindOp.getResult().getType();
        TensorType newResTy = permuteDims(oldResTy, permVec);
        rebindOp.getResult().setType(newResTy);
      }
    }

    return success();
  }
};

} // namespace

namespace M::MO {

void populateAlgebraicPatterns(mlir::RewritePatternSet &patterns,
                               bool enableFastMath, bool enableUndefinedMath) {
  MLIRContext *context = patterns.getContext();

  patterns
      .add<ArithmeticFoldingAddWithZero, ArithmeticFoldingSubWithZero,
           ArithmeticFoldingMulWithOne, ArithmeticFoldingDivByOne,
           ArithmeticFoldingPowerByOne, ArithmeticFoldingPowerOfOne,
           ArithmeticFoldingMaxSameOperand, ArithmeticFoldingMinSameOperand,
           ArithmeticFoldingEqualSameOperand, ArithmeticFoldingAndSameOperand,
           ArithmeticFoldingOrSameOperand, ArithmeticFoldingXorSameOperand,
           MathematicalCanonicalizationChainsOfPowToSinglePow,
           MathematicalCanonicalizationZeroSubToNegative,
           MathematicalSimplificationAddWithNegative,
           MathematicalSimplificationSubWithNegative>(context);

  patterns.add<BroadcastReshapePattern>(context);

  patterns.add<ConvertBatchMatmulToMatmulPattern,
               ConvertBatchMatmulBatchSizeOneToMatmulPattern,
               ConvToMatmulPattern, MergeParallelMatmulPattern<MatmulOp>,
               MergeParallelMatmulPattern<BatchMatmulOp>>(context);

  patterns.add<FlattenNestedConcatPattern, GathersToSplitPattern,
               SplitAndSqueezesPattern, PadConvPattern>(context);

  if (enableFastMath) {
    patterns.add<ArithmeticFoldingSubWithSame, ArithmeticFoldingMulWithZero,
                 ArithmeticFoldingPowByZero, ArithmeticFoldingMinOfMax,
                 ArithmeticFoldingMaxOfMin,
                 ArithmeticFoldingGreaterOrEqualThanSameOperand,
                 ArithmeticFoldingGreaterThanSameOperand,
                 ArithmeticFoldingNotEqualSameOperand,
                 MathematicalSimplificationDivByDivToDivOfMul,
                 MathematicalSimplificationDivOfDivToDivByMul,
                 MathematicalSimplificationDivBySmallTensorToMul,
                 MathematicalSimplificationMulOfDivToDiv,
                 MathematicalSimplificationPowByFloatToPowByInt>(context);
  }

  if (enableUndefinedMath) {
    patterns.add<ArithmeticFoldingZeroDiv, ArithmeticFoldingPowerOfZero,
                 ArithmeticFoldingModByOne, ArithmeticFoldingModSameOperand>(
        context);
  }
}

void populateForwardCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns
      .add<ForwardMathematicalCanonicalizationRsqrtToDivSqrt,
           ForwardMathematicalCanonicalizationReluToMax,
           ForwardMathematicalCanonicalizationMulByTwoToAdd,
           ForwardMathematicalCanonicalizationPowByTwoToMul,
           ForwardMathematicalCanonicalizationPowByThreeToMul,
           ForwardMathematicalCanonicalizationPowByNegativeOneOrTwoToDivAndPow>(
          context);
}

void populateBackwardCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<
      BackwardMathematicalCanonicalizationDivSqrtToRsqrt,
      BackwardMathematicalCanonicalizationMaxToRelu,
      BackwardMathematicalCanonicalizationExpToSigmoid,
      BackwardMathematicalCanonicalizationSigmoidMulToSilu,
      BackwardMathematicalCanonicalizationSigmoidMulToGeluQuick,
      BackwardMathematicalCanonicalizationErfToGelu,
      BackwardMathematicalCanonicalizationTanhToGeluTanh,
      BackwardMathematicalCanonicalizationMulByTwoToAdd,
      BackwardMathematicalCanonicalizationDivAndPowToPowByNegativeOneOrTwo>(
      context);
}

} // namespace M::MO
