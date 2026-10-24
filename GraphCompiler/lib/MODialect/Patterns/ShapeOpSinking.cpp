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

#include "GraphCompiler/MODialect/Patterns/ShapeOpSinking.h"

#include "GraphCompiler/MODialect/Analysis/StaticReshapeDimensionsMappingAnalysis.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/SplitSlicePatterns.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/AssertStream.h"
#include "Support/ML/Permutation.h"

#include <numeric>

using namespace M;
using namespace MO;
using namespace GraphCompiler::Matchers;

namespace {

/// Construct a new TensorType based on `ty` with the shape dimensions
/// permuted by `perm`.
static TensorType permuteDims(TensorType ty, ArrayRef<int64_t> perm) {
  SmallVector<TypedAttr> permutedDims =
      permute(ty.getShape().getValues(), perm);
  auto shapeTy = ShapeType::get(ty.getContext());
  return ty.cloneWith(ShapeAttr::get(permutedDims, shapeTy));
}

static bool isStaticConcat(ConcatOp op) {
  auto inputType = op.getType();
  if (!inputType.hasRank())
    return false;

  uint64_t axis = normalizeAxis(op.getAxisAttr().getInt(), inputType.getRank());
  if (axis >= inputType.getRank())
    return false;

  if (!isStaticDimAttr(inputType.getDim(axis)))
    return false;

  return true;
}

template <typename... OpTypes>
static bool allOperandsOfThisTypeHaveOneUse(Operation::operand_range operands) {
  return llvm::all_of(operands, [&](Value operand) {
    auto op = operand.getDefiningOp();
    if (!op || !isa<OpTypes...>(op))
      return false;

    return llvm::range_size(operand.getUsers()) == 1;
  });
}

// Returns the squeeze axis if the reshape from `inputType` to `outputType`
// only drops one size-1 dim, std::nullopt otherwise.
//
// e.g.
// inputType = tensor<[D1, 2, 1, D3]>
// outputType = tensor<[D1, 2, D3]>
// res = optional{2}
//
// inputType = tensor<[D1, 2, 1, D4]>
// outputType = tensor<[D1, 2, D3]>
// res = nullopt
static std::optional<int64_t> findSqueezeAxis(MO::TensorType inputType,
                                              MO::TensorType outputType) {

  auto inputDims = inputType.getDims();
  auto outputDims = outputType.getDims();

  if (inputDims.size() != outputDims.size() + 1)
    return std::nullopt;

  auto zipIter = llvm::zip_longest(inputDims, outputDims);

  auto it = llvm::find_if(zipIter, [](const auto &ioDims) {
    return std::get<0>(ioDims) != std::get<1>(ioDims);
  });

  auto [inputDim, outputDim] = *it;
  std::optional<int64_t> inputDimInt = tryGetConstantInt(*inputDim);

  if (!inputDimInt)
    return std::nullopt;

  if (*inputDimInt != 1)
    return std::nullopt;

  auto offset = std::distance(zipIter.begin(), it);

  if (!std::equal(inputDims.begin(), inputDims.begin() + offset,
                  outputDims.begin()))
    return std::nullopt;

  if (!std::equal(inputDims.begin() + offset + 1, inputDims.end(),
                  outputDims.begin() + offset))
    return std::nullopt;

  return offset;
}

// Returns true if the split operates on the last dimension and that the
// dimension is statically known.
static bool isStaticSplit(SplitOp op) {
  auto inputType = llvm::cast<TensorType>(op.getInput().getType());
  if (!inputType.hasRank())
    return false;

  int64_t axis = normalizeAxis(op.getAxisAttr().getInt(), inputType.getRank());
  if (!isStaticDimAttr(inputType.getDim(axis)))
    return false;

  return true;
}

template <typename... OpTypes>
static bool allResultsAreUniquelyUsedBy(Operation *op) {
  return llvm::all_of(op->getResults(), [](Value result) {
    auto users = result.getUsers();
    if (llvm::range_size(users) != 1)
      return false;

    return isa<OpTypes...>(*users.begin());
  });
}

/// Maps a split/slice-like axis through a static reshape, or returns
/// std::nullopt. The mapping is valid when:
/// - After skipping trailing 1-dimensions, a non-1, non-dynamic dim is found.
/// - That dimension is not mixed with left-neighbor dimensions in the reshape.
static std::optional<int64_t> mapAxisThroughStaticReshape(
    const StaticReshapeDimensionsMappingAnalysis &analysis,
    TensorType beforeType, int64_t axis) {
  // Skip trailing 1-dims (they are ignored by the analysis).
  int64_t mappedAxis = axis;
  while (mappedAxis < static_cast<int64_t>(beforeType.getRank()) &&
         beforeType.getIntDim(mappedAxis) == 1) {
    ++mappedAxis;
  }

  // No non-1 dimension found, or it is dynamic — unsupported.
  if (mappedAxis == static_cast<int64_t>(beforeType.getRank()) ||
      !isStaticDimAttr(beforeType.getDim(mappedAxis))) {
    return std::nullopt;
  }

  // Reject if the axis is mixed with left-neighbor dimensions in the reshape.
  int64_t newAxis = analysis.forwardMapping()[mappedAxis].front();
  bool couldMix = analysis.backwardMapping()[newAxis].size() > 1;
  bool mixedButAxisIsFirst =
      analysis.backwardMapping()[newAxis].front() == mappedAxis;
  if (couldMix && !mixedButAxisIsFirst)
    return std::nullopt;

  return newAxis;
}

/// A pattern that detects `mo.arg_nonzero` ops preceded by a flattening reshape
/// and a rank-increasing broadcast, and rewrites them so that the broadcast
/// happens after the `mo.arg_nonzero` op.
///
/// This decreases the workload on `mo.arg_nonzero` and improves the compilers
/// ability to constant fold before and fuse after the arg_nonzero op.
///
/// Example:
///
/// input : tensor<[2, 3]>
///
///   bcast = broadcast(input) : ... -> tensor<[5, 2, 3]>
///   flat  = reshape(bcast)   : ... -> tensor<[30]>
///   res   = arg_nonzero(flat)      : ...
/// ===>
///   flat        = reshape(input)   : ... -> tensor<[6]>
///   arg_nonzero = arg_nonzero(flat)      : ...
///   res         = arg_nonzero + broadcast(range(...)))
struct CommuteBroadcastWithFlatennedArgNonzero
    : DeviceAwareRewritePattern<OpRewritePattern, ArgNonzeroOp> {
  CommuteBroadcastWithFlatennedArgNonzero(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 4) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ArgNonzeroOp argNonzeroOp,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = argNonzeroOp.getLoc();

    TensorValue inputVal = argNonzeroOp.getInput();
    TensorType inputTy = inputVal.getType();
    if (!inputTy.hasRank() || inputTy.getRank() != 1)
      return rewriter.notifyMatchFailure(loc, "input is not rank-1");

    auto reshapeOp = inputVal.getDefiningOp<Static::ReshapeOp>();
    if (!reshapeOp)
      return rewriter.notifyMatchFailure(loc, "not preceded by static reshape");

    TensorValue reshapeInput = reshapeOp.getInput();
    TensorType reshapeInputTy = reshapeInput.getType();
    if (!reshapeInputTy.hasRank()) {
      return rewriter.notifyMatchFailure(
          loc, "preceding reshape has dynamic input rank");
    }

    auto bcastOp = reshapeInput.getDefiningOp<Static::BroadcastToOp>();
    if (!bcastOp)
      return rewriter.notifyMatchFailure(loc, "not preceded by static bcast");

    TensorValue bcastInput = bcastOp.getInput();
    TensorType bcastInputTy = bcastInput.getType();
    if (!bcastInputTy.hasRank()) {
      return rewriter.notifyMatchFailure(
          loc, "preceding static bcast has dynamic input rank");
    }

    // A rank-increasing broadcast is the exit condition.
    int64_t rankDiff =
        (int64_t)reshapeInputTy.getRank() - (int64_t)bcastInputTy.getRank();
    if (rankDiff <= 0) {
      return rewriter.notifyMatchFailure(
          loc, "preceding static bcast does not increase rank");
    }

    if (!rewriter.addOps({reshapeOp, bcastOp})) {
      return rewriter.notifyMatchFailure(
          loc, "matched ops have inconsistent devices");
    }

    // flattened = bcastInput.flatten(); FS is the flat size.
    GraphOp graphOp = argNonzeroOp->getParentOfType<GraphOp>();
    KGEN::ParamDeclAttr fsDecl = graphOp.getNewDimParam("FS");
    ArrayRef<TypedAttr> reshapeInputDims = reshapeInputTy.getDims();
    ArrayRef<TypedAttr> trailingDims = reshapeInputDims.drop_front(rankDiff);
    auto fsAttr =
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap, trailingDims);
    rewriter.create<KGEN::ParamDeclareOp>(loc, fsDecl, fsAttr);

    auto flattenedTy = bcastInputTy.cloneWith(
        SmallVector<TypedAttr>{KGEN::ParamDeclRefAttr::get(fsDecl)});
    auto flattened =
        rewriter.create<Static::ReshapeOp>(loc, flattenedTy, bcastInput);

    // argNonzero_res = np.expand_dims(np.argNonzero(flattened)[0], axis=1),
    // with W the number of unique elements.
    KGEN::ParamDeclAttr wDecl = graphOp.getNewDimParam("W");
    auto wRef = KGEN::ParamDeclRefAttr::get(wDecl);
    IntegerType si64Ty = rewriter.getIntegerType(64, /*isSigned=*/true);
    SmallVector<TypedAttr> argNonzeroDims{wRef, rewriter.getDimAttr(1)};
    DeviceRefAttr deviceRef = bcastInputTy.getDeviceRef();
    auto argNonzeroTy = TensorType::get(argNonzeroDims, si64Ty, deviceRef);
    auto newArgNonzeroOp = rewriter.create<ArgNonzeroOp>(
        loc, argNonzeroTy, flattened, ParamDeclArrayAttr::get(wDecl));

    // range = np.arange(0, FS*BS, FS), with BS the batch size.
    auto start = rewriter.createHostConstant(loc, (int64_t)0);

    ArrayRef<TypedAttr> batchDims = reshapeInputDims.take_front(rankDiff);
    auto bsAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap, batchDims);

    SmallVector<TypedAttr> limFactors{bsAttr, fsAttr};
    auto limAttr =
        KGEN::ParamOperatorAttr::get(KGEN::POC::MulNoWrap, limFactors);
    auto lim = rewriter.create<MOSH::ParamToValueOp>(loc, limAttr);
    auto limT = rewriter.create<IndexToTensorOp>(loc, start.getType(), lim);

    auto step = rewriter.create<MOSH::ParamToValueOp>(loc, fsAttr);
    auto stepT = rewriter.create<IndexToTensorOp>(loc, start.getType(), step);

    KGEN::ParamDeclAttr bsDecl = graphOp.getNewDimParam("BS");
    rewriter.create<KGEN::ParamDeclareOp>(loc, bsDecl, bsAttr);
    auto bsRef = KGEN::ParamDeclRefAttr::get(bsDecl);
    auto rangeTy =
        TensorType::get(SmallVector<TypedAttr>{bsRef}, si64Ty, deviceRef);
    auto range = rewriter.create<RangeOp>(loc, rangeTy, start, limT, stepT);

    // rangeExpanded = np.expand_dims(range, axis=1)
    auto expandedTy =
        TensorType::get(SmallVector<TypedAttr>{bsRef, rewriter.getDimAttr(1)},
                        si64Ty, deviceRef);
    auto rangeExpanded =
        rewriter.create<Static::ReshapeOp>(loc, expandedTy, range);

    // indexes = np.broadcast_to(argNonzero_res.T, [BS, W])
    //         + np.broadcast_to(rangeExpanded, [BS, W])
    auto perm = rewriter.createHostConstant(loc, SmallVector<int64_t>{1, 0});
    auto transposeTy = TensorType::get(
        SmallVector<TypedAttr>{argNonzeroDims[1], argNonzeroDims[0]}, si64Ty,
        deviceRef);
    auto wTranspose =
        rewriter.create<TransposeOp>(loc, transposeTy, newArgNonzeroOp, perm);

    auto bcastResTy =
        TensorType::get(SmallVector<TypedAttr>{bsRef, wRef}, si64Ty, deviceRef);
    auto indexes = rewriter.create<AddOp>(
        loc,
        rewriter.create<Static::BroadcastToOp>(loc, bcastResTy, wTranspose),
        rewriter.create<Static::BroadcastToOp>(loc, bcastResTy, rangeExpanded));

    // Don't drop the original argNonzero's parameter declaration.
    ArrayRef<ParamDeclAttr> oldDecls = argNonzeroOp.getOutputParamDecls();
    ASSERT_STREAM(oldDecls.size() == 1,
                  << "mo.argNonzero declares more than one new param");
    rewriter.modifyOpInPlace(argNonzeroOp,
                             [&]() { argNonzeroOp.setOutputParamDecls({}); });
    auto numResAttr = KGEN::ParamOperatorAttr::get(
        KGEN::POC::MulNoWrap, SmallVector<TypedAttr>{bsAttr, wRef});
    rewriter.create<KGEN::ParamDeclareOp>(loc, oldDecls[0], numResAttr);

    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(
        argNonzeroOp, argNonzeroOp.getType(), indexes);
    return success();
  }
};

/// Moves a transpose after the unary op that consumes it.
///
///  input     perm       input
///    \        /           |
///    transpose     ==>  unary(*)    perm
///        |                 \       /
///      unary               transpose(*)
///
/// NOTE: (*) means result type or operands of the op were updated
struct SinkTransposeThroughUnaryOpPattern
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseUnary> {
  SinkTransposeThroughUnaryOpPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseUnary op,
                             DeviceAwareRewriter &rewriter) const override {
    Value perm;
    Operation *opaqueTransposeOp = nullptr;
    auto unaryP = m_Op<ElementWiseUnary>(
        m_Op<TransposeOp>(&opaqueTransposeOp, m_Any(), m_Any(&perm)));
    if (!unaryP.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!perm)
      return rewriter.notifyMatchFailure(op, "null permutation value");

    // Sinking a multi-use transpose would recompute it for the other users.
    Value transpose = opaqueTransposeOp->getResult(0);
    if (!transpose.hasOneUse())
      return rewriter.notifyMatchFailure(op, "transpose op has multiple uses");

    if (!rewriter.addOps({opaqueTransposeOp})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorType finalOutputType = op.getResult().getType();
    auto transposeOp = cast<TransposeOp>(opaqueTransposeOp);
    TensorValue input = transposeOp.getInput();
    TensorValue opRes = op.getResult();
    rewriter.replaceAllUsesWith(op->getResults(), transposeOp.getResult());
    rewriter.modifyOpInPlace(op, [&]() {
      op.setInput(input);
      DType unaryResultDType = opRes.getType().getDtype();
      opRes.setType(input.getType().cloneWith(unaryResultDType));
    });

    rewriter.modifyOpInPlace(transposeOp, [&]() {
      transposeOp.getInputMutable().assign(opRes);
      transposeOp->moveAfter(op);
      transposeOp.getResult().setType(finalOutputType);
    });
    return mlir::success();
  }
};

/// Replaces matching transposes on both binary op inputs with one transpose
/// after the binary op.
///
/// Example:
///
/// input1     perm   input2     perm
///    \        /        \        /
///    transpose1        transpose2
///           \           /
///               binary
///
/// =======>
///
///  input1   input2
///     \      /
///      binary(*)    perm1 (or perm2, they are the same)
///          \        /
///          transpose
///
/// NOTE (*) means result type or operands of the op were updated
struct SinkTransposeThroughBinaryOpPattern
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  SinkTransposeThroughBinaryOpPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue input1, input2, perm1, perm2;
    auto transpose1P =
        m_Op<TransposeOp>(rewriter.pushOp(), m_Any(&input1), m_Any(&perm1));
    auto transpose2P =
        m_Op<TransposeOp>(rewriter.pushOp(), m_Any(&input2), m_Any(&perm2));
    auto binaryP = m_Op<ElementWiseBinary>(transpose1P, transpose2P);
    if (!binaryP.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input1 || !input2 || !perm1 || !perm2)
      return rewriter.notifyMatchFailure(op, "null input or permutation value");

    // Sinking a multi-use transpose would keep the original for other users.
    if (!op.getLhsInput().hasOneUse())
      return rewriter.notifyMatchFailure(op, "lhs transpose has multiple uses");

    if (!op.getRhsInput().hasOneUse())
      return rewriter.notifyMatchFailure(op, "rhs transpose has multiple uses");

    if (perm1 != perm2)
      return rewriter.notifyMatchFailure(op, "permutations are different");

    TensorType input1Ty = input1.getType();
    TensorType input2Ty = input2.getType();
    if (input1Ty.getShape() != input2Ty.getShape())
      return rewriter.notifyMatchFailure(op, "input shapes are different");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorValue opRes = op.getResult();
    TensorType oldOpResTy = opRes.getType();
    rewriter.modifyOpInPlace(op, [&]() {
      op.setLhsInput(input1);
      op.setRhsInput(input2);

      opRes.setType(input1Ty.cloneWith(oldOpResTy.getDtype()));
    });

    rewriter.setInsertionPointAfter(op);
    auto finalTransposeOp =
        rewriter.create<TransposeOp>(op->getLoc(), oldOpResTy, opRes, perm1);
    rewriter.replaceAllUsesExcept(opRes, finalTransposeOp, finalTransposeOp);

    return mlir::success();
  }
};

/// Moves a transpose after the broadcast that consumes it.
///
///  input     perm       input
///    \        /           |
///    transpose     ==>  broadcast(*)    perm
///        |                 \           /
///    broadcast              transpose(*)
///
/// NOTE (*) means result type or operands of the op were updated
struct SinkTransposeThroughStaticBroadcastToOpPattern
    : public DeviceAwareRewritePattern<OpRewritePattern,
                                       Static::BroadcastToOp> {
  SinkTransposeThroughStaticBroadcastToOpPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  // Inverse of `permute(attrs, perm)`.
  static SmallVector<TypedAttr> unpermute(ArrayRef<TypedAttr> attrs,
                                          ArrayRef<int64_t> perm) {
    // e.g., if perm == [2, 0, 1], inversePerm == [1, 2  0]
    SmallVector<TypedAttr> result(perm.size());
    for (auto [idx, permIdx] : llvm::enumerate(perm))
      result[permIdx] = attrs[idx];
    return result;
  }

  LogicalResult
  deviceAwareMatchAndRewrite(Static::BroadcastToOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Value input, perm;
    Operation *opaqueTransposeOp = nullptr;
    auto broadcastP = m_Op<Static::BroadcastToOp>(
        m_Op<TransposeOp>(&opaqueTransposeOp, m_Any(&input), m_Any(&perm)));
    if (!broadcastP.match(op))
      return rewriter.notifyMatchFailure(op, "failed to match pattern");

    if (!input || !perm)
      return rewriter.notifyMatchFailure(op, "null input or permutation value");

    // Sinking a multi-use transpose would recompute it for the other users.
    auto transposeOp = cast<TransposeOp>(opaqueTransposeOp);
    if (!transposeOp->hasOneUse())
      return rewriter.notifyMatchFailure(op, "transpose op has multiple uses");

    MO::TensorType oldBroadcastResTy = op.getType();
    MO::TensorType transposeResTy = transposeOp.getType();
    if (!oldBroadcastResTy.hasRank())
      return rewriter.notifyMatchFailure(op, "broadcast shape is unranked");
    if (!transposeResTy.hasRank())
      return rewriter.notifyMatchFailure(op, "transpose shape is unranked");

    auto maybePermVec = tryGetConstantIntVector(cast<TensorValue>(perm));
    if (!maybePermVec.has_value())
      return rewriter.notifyMatchFailure(op, "permutation isn't constant");

    if (!rewriter.addOps({opaqueTransposeOp})) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    SmallVector<int64_t> permVec = maybePermVec.value();
    // Pad the permutation for the dims a rank-increasing broadcast adds, e.g.
    //   input shape      = [1, 3, 2]
    //   permutation      = [0, 2, 1]
    //   transposed shape = [1, 2, 3]
    //   bcast shape      = [9, 1, 2, 3]
    // then after sinking the transpose, we must have
    //   input shape      = [1, 3, 2]
    //   bcast shape      = [9, 1, 3, 2]
    //   permutation      = [0, 1, 3, 2]
    //   transposed shape = [9, 1, 2, 3]
    if (oldBroadcastResTy.getRank() > transposeResTy.getRank()) {
      size_t extraDims = oldBroadcastResTy.getRank() - transposeResTy.getRank();
      for (int64_t &axis : permVec)
        axis += extraDims;
      auto paddedAxes =
          llvm::iota_range<int64_t>(0, extraDims, /*Inclusive=*/false);
      permVec.insert(permVec.begin(), paddedAxes.begin(), paddedAxes.end());
      auto newPerm = rewriter.createHostConstant(op.getLoc(), permVec);

      rewriter.modifyOpInPlace(
          transposeOp, [&]() { transposeOp.getPermMutable().assign(newPerm); });
    }
    SmallVector<TypedAttr> newBroadcastDims =
        unpermute(oldBroadcastResTy.getDims(), permVec);

    auto shapeTy = rewriter.getType<ShapeType>();
    MO::TensorType newBroadcastResTy =
        oldBroadcastResTy.cloneWith(ShapeAttr::get(newBroadcastDims, shapeTy));

    rewriter.replaceAllUsesWith(op, transposeOp.getResult());
    rewriter.modifyOpInPlace(op, [&]() {
      op.getInputMutable().assign(input);
      op.getResult().setType(newBroadcastResTy);
    });

    rewriter.modifyOpInPlace(transposeOp, [&]() {
      transposeOp.getInputMutable().assign(op.getResult());
      transposeOp->moveAfter(op);
      transposeOp.getResult().setType(oldBroadcastResTy);
    });

    return mlir::success();
  }
};

/// Swims a concat through a ReshapeOpType (usually a static reshape or a
/// rebind op).
/// CONDITIONS:
/// - Concat axis is statically known.
/// - Concat dimension is statically known.
/// - All concat providers are only used by the concat.
/// - Concat must be uniform (same size).
/// - All concat providers must be `ReshapeOpType`.
/// - The reshape must not mix dynamic and static dimensions.
/// - The concat axis must not be mixed with left dimensions after the reshape.
/// - [Unchecked] Row major-like memory layout.
/// BEFORE:
/// -> reshape -> concat ->
/// -> reshape ----| |
/// -> reshape ------|
/// AFTER:
/// -> concat -> reshape
template <typename ReshapeOpType>
struct SwimConcatThroughReshapeLikeOpsPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, ConcatOp> {
  SwimConcatThroughReshapeLikeOpsPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ConcatOp concatOp,
                             DeviceAwareRewriter &rewriter) const override {
    if (!isStaticConcat(concatOp) ||
        !allOperandsOfThisTypeHaveOneUse<ReshapeOpType>(concatOp.getInputs())) {
      return rewriter.notifyMatchFailure(
          concatOp, "Unsupported concat. Must be static and providing reshapes "
                    "must have no other user");
    }

    rewriter.pushOp(concatOp);
    SmallVector<ReshapeOpType> reshapeOps;
    SmallVector<Value> reshapeOperands;
    for (Value operand : concatOp.getInputs()) {
      auto op = operand.getDefiningOp();
      auto reshapeLikeOp = dyn_cast<ReshapeOpType>(op);
      if (!reshapeLikeOp.getType().hasRank()) {
        return rewriter.notifyMatchFailure(concatOp,
                                           "Reshape inputs must be ranked");
      }
      reshapeOps.push_back(reshapeLikeOp);
      reshapeOperands.push_back(reshapeLikeOp.getInput());
      rewriter.pushOp(op);
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          concatOp, "matched ops have inconsistent devices");
    }

    auto firstBeforeType =
        cast<TensorType>(reshapeOps.front().getInput().getType());
    for (auto reshapeLikeOp : ArrayRef(reshapeOps).slice(1)) {
      auto beforeType = cast<TensorType>(reshapeLikeOp.getInput().getType());
      if (beforeType != firstBeforeType) {
        return rewriter.notifyMatchFailure(concatOp,
                                           "Only supports uniform concat");
      }
    }

    auto firstAfterType = cast<TensorType>(reshapeOps.front().getType());
    const StaticReshapeDimensionsMappingAnalysis
        staticReshapeDimensionsMappingAnalysis(firstAfterType.getDims(),
                                               firstBeforeType.getDims());
    if (!staticReshapeDimensionsMappingAnalysis) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "Unsupported reshape operand");
    }

    auto concatInputType = firstAfterType;
    const int64_t concatAxis = normalizeAxis(concatOp.getAxisAttr().getInt(),
                                             concatInputType.getRank());

    uint64_t mappedConcatAxis = concatAxis;
    // Skip to the first non-1 dim (dynamic included); the analysis ignores 1s.
    while (mappedConcatAxis < concatInputType.getRank() &&
           concatInputType.getIntDim(mappedConcatAxis) == 1) {
      ++mappedConcatAxis;
    }

    // A missing or dynamic non-1 dim can't be concatenated in the new shape.
    const bool couldNotFindAnotherAxis =
        mappedConcatAxis == concatInputType.getRank();
    if (couldNotFindAnotherAxis ||
        !isStaticDimAttr(concatInputType.getDim(mappedConcatAxis))) {
      return rewriter.notifyMatchFailure(
          concatOp,
          "Trailing ones or trailing dynamic dimensions are unsupported");
    }

    // Mixing means several original dims contribute to one reshaped dim.
    // Assuming row-major layout, a concat is only valid if the concat axis is
    // the leftmost of the mixed dims.
    const bool couldMix =
        staticReshapeDimensionsMappingAnalysis
            .backwardMapping()[staticReshapeDimensionsMappingAnalysis
                                   .forwardMapping()[mappedConcatAxis]
                                   .front()]
            .size() > 1;
    const bool mixedAndConcatAxisIsNotFirst =
        staticReshapeDimensionsMappingAnalysis
            .backwardMapping()[staticReshapeDimensionsMappingAnalysis
                                   .forwardMapping()[mappedConcatAxis]
                                   .front()]
            .front() != static_cast<int64_t>(mappedConcatAxis);
    if (couldMix && mixedAndConcatAxisIsNotFirst) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "Reshape is mixing static dimensions");
    }

    // The new concat axis is the left most axis in the shape after the reshape.
    const int64_t newConcatAxis = staticReshapeDimensionsMappingAnalysis
                                      .forwardMapping()[mappedConcatAxis]
                                      .front();

    Location loc = concatOp->getLoc();
    rewriter.setInsertionPoint(concatOp);

    SmallVector<TypedAttr> newConcatOutputShape(firstBeforeType.getDims());
    newConcatOutputShape[newConcatAxis] = rewriter.getDimAttr(
        *tryGetConstantInt(newConcatOutputShape[newConcatAxis]) *
        reshapeOps.size());
    auto newConcatOutputType = concatInputType.cloneWith(newConcatOutputShape);
    auto newConcatOp = rewriter.create<ConcatOp>(
        loc, newConcatOutputType,
        rewriter.getIndexAttr(newConcatAxis).getValue(), reshapeOperands,
        concatOp.getOutputParamDecls());

    auto newReshapeOp = rewriter.create<ReshapeOpType>(
        loc, concatOp.getType(), newConcatOp->getResult(0));

    rewriter.replaceAllUsesWith(concatOp, newReshapeOp);
    rewriter.eraseOp(concatOp);
    for (auto reshapeOp : reshapeOps)
      rewriter.eraseOp(reshapeOp);

    return success();
  }
};

/// Moves a binary op whose operands are a reshape and a broadcast before the
/// reshape, for better fusion opportunities.
///
/// ┌─────┐         ┌────────┐               ┌─────┐  ┌────────┐
/// │input│         │elemwise│               │input│  │elemwise│
/// └──┬──┘         └───┬────┘               └──┬──┘  └───┬────┘
///    │                │                       │         │
///    │                │                       │         │
/// reshape          broadcast                  │     broadcast*
///    │                │                       │         │
///    │                │         ───────►      └─binOp*──┘
///    └──── binOp ─────┘                         (add)
///          (add)                                  │
///            │                                    │
///            │                                  reshape*
///         ┌──┴───┐                                │
///         │output│                             ┌──┴───┐
///         └──────┘                             │output│
///                                              └──────┘
/// Only valid if `elemwise` has the least rank `r` among `elemwise`, `input`
/// and `reshape`, and the last `r` dims of `broadcast`, `input` and `reshape`
/// match.
template <typename ReshapeOpType>
struct SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue inputTensor, elemwiseTensor;
    auto reshapeP = m_Op<ReshapeOpType>(rewriter.pushOp(), m_Any(&inputTensor));
    auto broadcastP =
        m_Op<Static::BroadcastToOp>(rewriter.pushOp(), m_Any(&elemwiseTensor));
    auto elemwiseOpP1 =
        m_Op<ElementWiseBinary>(rewriter.pushOp(), reshapeP, broadcastP);
    auto elemwiseOpP2 =
        m_Op<ElementWiseBinary>(rewriter.pushOp(), broadcastP, reshapeP);

    bool reshapedWasLeftOriginally;
    TensorValue reshapedTensor, broadcastTensor;
    if (elemwiseOpP1.match(op)) {
      reshapedTensor = op.getLhsInput();
      broadcastTensor = op.getRhsInput();
      reshapedWasLeftOriginally = true;
    } else if (elemwiseOpP2.match(op)) {
      broadcastTensor = op.getLhsInput();
      reshapedTensor = op.getRhsInput();
      reshapedWasLeftOriginally = false;
    } else {
      return rewriter.notifyMatchFailure(op, "Failed to match the pattern.");
    }
    if (!inputTensor || !elemwiseTensor || !reshapedTensor || !broadcastTensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    TensorType inputTensorTy = inputTensor.getType();

    // Ensure the smallest rank tensor is `elemwise`
    TensorType reshapedTensorTy = reshapedTensor.getType();
    TensorType elemwiseTensorTy = elemwiseTensor.getType();
    TensorType broadcastTensorTy = broadcastTensor.getType();
    if (!reshapedTensorTy.hasRank() || !elemwiseTensorTy.hasRank() ||
        !inputTensorTy.hasRank() || !broadcastTensorTy.hasRank())
      return rewriter.notifyMatchFailure(
          op, "Cannot prove correctness in case of dynamic rank.");
    size_t smallestRank =
        std::min(reshapedTensorTy.getRank(), elemwiseTensorTy.getRank());
    smallestRank = std::min(smallestRank, inputTensorTy.getRank());
    smallestRank = std::min(smallestRank, broadcastTensorTy.getRank());

    if (smallestRank != elemwiseTensorTy.getRank())
      return rewriter.notifyMatchFailure(
          op,
          "Original shape of broadcasted operand does not have smallest rank.");

    // Check trailing dims of `input`, `reshape` and `broadcast` match
    for (size_t i = 0; i < smallestRank; ++i) {
      TypedAttr reshapedTensorDim =
          reshapedTensorTy.getDim(reshapedTensorTy.getRank() - i - 1);
      TypedAttr broadcastTensorDim =
          broadcastTensorTy.getDim(broadcastTensorTy.getRank() - i - 1);
      TypedAttr inputTensorDim =
          inputTensorTy.getDim(inputTensorTy.getRank() - i - 1);

      if (reshapedTensorDim != broadcastTensorDim ||
          broadcastTensorDim != inputTensorDim)
        return rewriter.notifyMatchFailure(
            op, "Trailing dimensions of the reshape, input, and elemwise "
                "tensor do not match.");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    // Moving the broadcast up preserves results: slices across the trailing
    // dims share elemwise operands, and the reshape keeps the trailing dims.
    DType broadcastDType = elemwiseTensor.getType().getDtype();
    TensorValue newBroadcast = rewriter.create<Static::BroadcastToOp>(
        op->getLoc(), inputTensor.getType().cloneWith(broadcastDType),
        elemwiseTensor);
    TensorType oldType = op.getResult().getType();

    auto newOp = cast<ElementWiseBinary>(rewriter.clone(*op));

    // make sure to preserve ordering for ops like mo.div
    if (reshapedWasLeftOriginally) {
      newOp.setLhsInput(inputTensor);
      newOp.setRhsInput(newBroadcast);
    } else {
      newOp.setLhsInput(newBroadcast);
      newOp.setRhsInput(inputTensor);
    }
    // The reshaped operand may not have the result's dtype (`mo.pow`'s
    // exponent).
    newOp.getResult().setType(
        inputTensor.getType().cloneWith(oldType.getDtype()));

    TensorValue output = rewriter.create<ReshapeOpType>(op->getLoc(), oldType,
                                                        newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Drops reshapes-like ops through Elementwise Unary Ops.
///                    ┌─────┐     ┌─────┐
///                    │input│     │input│
///                    └──┬──┘     └──┬──┘
///                       │           │
///                    reshape     unaryOp*
///                       │         (erf)
///                       │    ──►    │
///                    unaryOp        │
///                     (erf)      reshape*
///                       │           │
///                    ┌──┴───┐    ┌──┴───┐
///                    │output│    │output│
///                    └──────┘    └──────┘
template <typename ReshapeOpType>
struct SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseUnary> {
  SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseUnary op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue inputTensor;
    auto reshapeP = m_Op<ReshapeOpType>(rewriter.pushOp(), m_Any(&inputTensor));
    auto elemwiseOpP = m_Op<ElementWiseUnary>(reshapeP);

    if (!elemwiseOpP.match(op))
      return rewriter.notifyMatchFailure(op, "Pattern not matched.");

    if (!inputTensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    auto newOp = cast<ElementWiseUnary>(rewriter.clone(*op));

    // We cannot assume output will be the same dtype as input (e.g. mo.cast)
    DType outputDType = newOp.getResult().getType().getDtype();
    newOp.setInput(inputTensor);
    newOp.getResult().setType(inputTensor.getType().cloneWith(outputDType));
    TensorValue output = rewriter.create<ReshapeOpType>(
        op->getLoc(), op.getResult().getType(), newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Drops reshape ops through SoftmaxOp as long as the reshape doesn't affect
/// the last axis (the compute axis of softmax).
///                    ┌─────┐     ┌─────┐
///                    │input│     │input│
///                    └──┬──┘     └──┬──┘
///                       │           │
///                    reshape        │
///                       │           │
///                    softmax ──► softmax
///                       │           │
///                       │        reshape
///                       │           │
///                    ┌──┴───┐    ┌──┴───┐
///                    │output│    │output│
///                    └──────┘    └──────┘
struct SinkStaticReshapeOpThroughSoftmaxOp
    : public DeviceAwareRewritePattern<OpRewritePattern, SoftmaxOp> {
  SinkStaticReshapeOpThroughSoftmaxOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SoftmaxOp op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue inputTensor;
    auto reshapeP =
        m_Op<Static::ReshapeOp>(rewriter.pushOp(), m_Any(&inputTensor));
    auto softmaxOpP = m_Op<SoftmaxOp>(reshapeP);

    if (!softmaxOpP.match(op))
      return rewriter.notifyMatchFailure(op, "Pattern not matched.");

    if (!inputTensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    TensorType inputTy = inputTensor.getType();
    TensorType reshapedTy = op.getInput().getType();

    if (!inputTy.hasRank() || !reshapedTy.hasRank())
      return rewriter.notifyMatchFailure(op, "Shape must be ranked");

    int64_t axisVal = op.getAxisAttr().getInt();

    // Only support sinking when the final axis is used.
    if (axisVal != clampAxis(-1, (int64_t)reshapedTy.getRank()))
      return rewriter.notifyMatchFailure(op, "Can't sink non-last axis");

    // Ensure the shape of the last axis is not changed
    if (inputTy.getRank() > 0 && (reshapedTy.getRank() == 0 ||
                                  inputTy.getShape().getValues().back() !=
                                      reshapedTy.getShape().getValues().back()))
      return rewriter.notifyMatchFailure(op, "Last axis is modified");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    auto newOp = cast<SoftmaxOp>(rewriter.clone(*op));

    // We cannot assume output will be the same dtype as input (e.g. mo.cast)
    DType outputDType = newOp.getResult().getType().getDtype();
    newOp.getInputMutable().set(inputTensor);
    newOp.getResult().setType(inputTensor.getType().cloneWith(outputDType));

    // Ensure the softmax axis still points to the last axis.
    newOp.setAxisAttr(
        mlir::Builder(newOp.getContext())
            .getIndexAttr(clampAxis(-1, (int64_t)inputTy.getRank())));

    TensorValue output = rewriter.create<Static::ReshapeOp>(
        op->getLoc(), op.getResult().getType(), newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Sinks operands which are reshapes through binary operators if
/// it can be proven `input1` and `input2` are the same shape
/// ┌──────┐     ┌──────┐    ┌──────┐   ┌──────┐
/// │input1│     │input2│    │input1│   │input2│
/// └──┬───┘     └──┬───┘    └──┬───┘   └───┬──┘
///    │            │           │           │
/// Reshape      Reshape        └───binOp───┘
///    │            │               (add)
///    │            │    ─────►       │
///    └────binOp───┘                 │
///         (add)                  Reshape
///           │                       │
///       ┌───┴──┐                 ┌──┴───┐
///       │output│                 │output│
///       └──────┘                 └──────┘
template <typename ReshapeOpType>
struct SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue input1Tensor, input2Tensor;
    auto reshapeP1 =
        m_Op<ReshapeOpType>(rewriter.pushOp(), m_Any(&input1Tensor));
    auto reshapeP2 =
        m_Op<ReshapeOpType>(rewriter.pushOp(), m_Any(&input2Tensor));
    auto elemwiseOpP = m_Op<ElementWiseBinary>(reshapeP1, reshapeP2);

    if (!elemwiseOpP.match(op))
      return rewriter.notifyMatchFailure(op, "Pattern not matched.");

    if (!input1Tensor || !input2Tensor)
      return rewriter.notifyMatchFailure(op, "Null values found in pattern.");

    TensorType input1TensorTy = input1Tensor.getType();
    TensorType input2TensorTy = input2Tensor.getType();

    if (input1TensorTy.getShape() != input2TensorTy.getShape())
      return rewriter.notifyMatchFailure(op, "Inputs shapes must be equal");

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    TensorType finalOutputType = op.getResult().getType();
    auto newOp = cast<ElementWiseBinary>(rewriter.clone(*op));
    newOp.setLhsInput(input1Tensor);
    newOp.setRhsInput(input2Tensor);
    newOp.getResult().setType(
        input1TensorTy.cloneWith(finalOutputType.getDtype()));
    TensorValue output = rewriter.create<ReshapeOpType>(
        op->getLoc(), finalOutputType, newOp.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), output);
    return mlir::success();
  }
};

/// Moves a single-axis squeeze-like reshape (see findSqueezeAxis) after the
/// transpose that consumes it.
///
///   input : tensor<[D1, 1, D2]>
///   reshaped : tensor<[D1, D2]>
///   transposed : tensor<[D2, D1]>
///
///   reshaped = reshape(input)
///   transposed = transpose(reshaped)
///
/// ========>
///
///   input : tensor<[D1, 1, D2]>
///   transposed : tensor<[D2, 1, D1]>
///   reshaped : tensor<[D2, D1]>
///
///   transposed = transpose(input)
///   reshaped = reshape(transposed)
///
/// Motivated by https://github.com/modularml/modular/issues/27800.
struct SinkSqueezeLikeReshapeThroughTranspose
    : public DeviceAwareRewritePattern<OpRewritePattern, TransposeOp> {

  SinkSqueezeLikeReshapeThroughTranspose(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(TransposeOp op,
                             DeviceAwareRewriter &rewriter) const override {

    MO::TensorValue transposeInputType = op.getInput();

    auto transposeDefiningOp = transposeInputType.getDefiningOp();

    if (!transposeDefiningOp)
      return rewriter.notifyMatchFailure(
          op, "Input to transpose was a block argument");

    auto reshapeOp = dyn_cast<Static::ReshapeOp>(transposeDefiningOp);

    if (!reshapeOp)
      return rewriter.notifyMatchFailure(
          op, "Did not find corresponding reshape op");

    if (!op.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(op,
                                         "Transpose op has more than one use");

    MO::TensorType inputType = reshapeOp.getInput().getType();
    MO::TensorType outputType = reshapeOp.getResult().getType();

    if (!outputType.hasRank() || !inputType.hasRank())
      return rewriter.notifyMatchFailure(op,
                                         "Input/output ranks not fully known");

    auto maybeSqueezeIdx = findSqueezeAxis(inputType, outputType);

    if (!maybeSqueezeIdx)
      return rewriter.notifyMatchFailure(
          op, "Reshape does not correspond to a squeeze-like operation");

    int64_t squeezeIdx = *maybeSqueezeIdx;

    auto maybePermVec = tryGetConstantIntVector(op.getPerm());
    if (!maybePermVec.has_value())
      return rewriter.notifyMatchFailure(
          op, "Transpose permutation is not produced by a constant op");

    SmallVector<int64_t> permVec(maybePermVec.value());
    for (int64_t i = 0, size = permVec.size(); i < size; ++i) {
      if (permVec[i] >= squeezeIdx)
        permVec[i] = permVec[i] + 1;
    }
    permVec.insert(permVec.begin() + squeezeIdx, squeezeIdx);

    Location loc = op->getLoc();

    auto constantOp = rewriter.create<MO::ConstantOp>(loc, permVec);
    auto permuted = permuteDims(inputType, permVec);

    auto newTransposeOp = rewriter.create<MO::TransposeOp>(
        loc, permuted, reshapeOp.getInput(), constantOp);

    rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(
        op, op.getResult().getType(), newTransposeOp);

    return success();
  }
};

// A reshape that only pads leading 1-dims before a broadcast is redundant,
// since broadcasting pads dims on the left.
//
// %input: !mo.tensor<[A, B, C], f32>
// %reshape = mo.static.reshape(%input): !mo.tensor<[A, B, C], f32>
//    -> !mo.tensor<[1, 1, A, B, C], f32>
// %bcast = mo.static.broadcast_to(%reshape): !mo.tensor<[1, 1, A, B, C], f32>
//    -> !mo.tensor<[10, 10, A, B, C], f32>
//
// ----->
// %bcast = mo.static.broadcast_to(%input): !mo.tensor<[A, B, C], f32>
//    -> !mo.tensor<[10, 10, A, B, C], f32>
struct RemoveExtraneousPaddingReshapeWithBroadcast
    : public DeviceAwareRewritePattern<OpRewritePattern,
                                       Static::BroadcastToOp> {

  RemoveExtraneousPaddingReshapeWithBroadcast(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(Static::BroadcastToOp op,
                             DeviceAwareRewriter &rewriter) const override {

    auto reshapeOp = op.getInput().getDefiningOp<Static::ReshapeOp>();
    if (!reshapeOp)
      return rewriter.notifyMatchFailure(
          op, "operation is not preceded by static reshape.");

    TensorType reshapeInputTy = reshapeOp.getInput().getType();
    TensorType reshapeOutputTy = reshapeOp.getType();

    if (!reshapeInputTy.hasRank() || !reshapeOutputTy.hasRank())
      return rewriter.notifyMatchFailure(
          reshapeOp, "Reshape is not statically ranked fully.");

    int64_t inputRank = reshapeInputTy.getRank();
    int64_t outputRank = reshapeOutputTy.getRank();
    if (inputRank >= outputRank)
      return rewriter.notifyMatchFailure(
          reshapeOp, "Reshape is not padding dims, output rank <= input rank.");

    // All dims except the padded ones must match.
    for (int64_t dimOutput = 0; dimOutput < outputRank; ++dimOutput) {
      // If negative, means not part of original dimensions
      int64_t dimInput = dimOutput - outputRank + inputRank;

      if (dimInput < 0) {
        if (!reshapeOutputTy.isIntegerDim(dimOutput)) {
          return rewriter.notifyMatchFailure(
              reshapeOp, "Cannot prove padding dimension is 1.");
        }
        if (reshapeOutputTy.getIntDim(dimOutput) != 1) {
          return rewriter.notifyMatchFailure(reshapeOp,
                                             "Padding dimension is not 1.");
        }
      } else if (reshapeInputTy.getDim(dimInput) !=
                 reshapeOutputTy.getDim(dimOutput)) {
        return rewriter.notifyMatchFailure(
            reshapeOp, "Reshape is not only padding dimensions.");
      }
    }

    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(op, op.getType(),
                                                       reshapeOp.getInput());
    return success();
  }
};

/// A pattern that detects binary operator where one operand is the result
/// of either a transpose or static reshape and the other is ultimately
/// the result of operations on solely constant values.
///
/// ┌─────┐         ┌────────┐               ┌─────┐  ┌────────┐
/// │input│         │constant│               │input│  │constant│
/// └──┬──┘         └───┬────┘               └──┬──┘  └───┬────┘
///    │                │                       │         │
///    │                │                       │       .....
///    │                │                       │         │
///  ViewOp           .....                     │     ViewOp*
///    │                │                       │         │
///    │                │         ───────►      └─binOp*──┘
///    └──── binOp ─────┘                           │
///            │                                    │
///            │                                  ViewOp*
///         ┌──┴───┐                                │
///         │output│                             ┌──┴───┐
///         └──────┘                             │output│
///                                              └──────┘
///
template <typename ViewOp>
struct SwimViewOpThroughBinaryOpWithConstantSubtree
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElementWiseBinary> {
  SwimViewOpThroughBinaryOpWithConstantSubtree(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::ElementWiseBinary op,
                             DeviceAwareRewriter &rewriter) const override {

    TensorValue lhs = op.getLhsInput();
    TensorValue rhs = op.getRhsInput();
    TensorValue result = op.getResult();

    auto lhsProducer = lhs.getDefiningOp();
    auto rhsProducer = rhs.getDefiningOp();

    if (!lhsProducer || !rhsProducer) {
      return rewriter.notifyMatchFailure(
          op, "One of the inputs is a block argument");
    }

    if (!result.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          op, "Only applies to binop outputs with one use");
    }

    Operation *maybeViewOp{nullptr};
    Operation *maybeConstantSubtreeRootOp{nullptr};

    Input constantSubtreeSide;

    if (isa<ViewOp>(lhsProducer) && !isa<ViewOp>(rhsProducer)) {
      maybeViewOp = lhsProducer;
      maybeConstantSubtreeRootOp = rhsProducer;
      constantSubtreeSide = Input::RHS;
    } else if (isa<ViewOp>(rhsProducer) && !isa<ViewOp>(lhsProducer)) {
      maybeViewOp = rhsProducer;
      maybeConstantSubtreeRootOp = lhsProducer;
      constantSubtreeSide = Input::LHS;
    } else if (isa<ViewOp>(lhsProducer) && isa<ViewOp>(rhsProducer)) {
      constantSubtreeSide = Input::UNK;
    } else {
      return rewriter.notifyMatchFailure(
          op, "Failed to match, no appropriate view op was found");
    }

    if (constantSubtreeSide != Input::UNK &&
        !dataflowGraphFoldsToConstants<MO::ConstantOp, ChainCreateOp>(
            maybeConstantSubtreeRootOp)) {
      return rewriter.notifyMatchFailure(
          op, "Failed to match, the view op input is not a constant subtree");
    }

    // Not applying the pattern to constant foldable subgraphs because there is
    // no gain and it can lead to cycles with other patterns that sink reshapes
    // through elementwise operations.
    bool lhsConstantFoldable =
        dataflowGraphFoldsToConstants<ConstantOp, ChainCreateOp>(lhsProducer);
    bool rhsConstantFoldable =
        dataflowGraphFoldsToConstants<ConstantOp, ChainCreateOp>(rhsProducer);
    if (lhsConstantFoldable && rhsConstantFoldable) {
      return rewriter.notifyMatchFailure(
          op, "Not applying the pattern to a constant foldable subgraph");
    }

    if (constantSubtreeSide == Input::UNK) {
      if (lhsConstantFoldable) {
        constantSubtreeSide = Input::LHS;
        maybeViewOp = rhsProducer;
        maybeConstantSubtreeRootOp = lhsProducer;
      } else if (rhsConstantFoldable) {
        constantSubtreeSide = Input::RHS;
        maybeViewOp = lhsProducer;
        maybeConstantSubtreeRootOp = rhsProducer;
      } else {
        return rewriter.notifyMatchFailure(
            op,
            "Failed to match, no constant subtree to the op could be found");
      }
    }

    ViewOp viewOp = dyn_cast<ViewOp>(maybeViewOp);

    TensorValue cso =
        constantSubtreeSide == Input::LHS ? op.getLhsInput() : op.getRhsInput();

    auto constantSubtreeViewOp =
        materializeConstantSubtreeViewOp(rewriter, op, viewOp, cso);

    if (constantSubtreeViewOp.isError()) {
      return rewriter.notifyMatchFailure(op, constantSubtreeViewOp.getError());
    }

    switch (constantSubtreeSide) {
    case Input::LHS:
      op.setLhsInput(*constantSubtreeViewOp);
      op.setRhsInput(viewOp.getInput());
      break;
    case Input::RHS:
      op.setLhsInput(viewOp.getInput());
      op.setRhsInput(*constantSubtreeViewOp);
      break;
    default:
      llvm_unreachable("Should not reach here");
    }
    op.getResult().setType(
        viewOp.getInput().getType().cloneWith(result.getType().getDtype()));

    rewriter.setInsertionPointAfter(op);
    auto binOpViewOp = materializeBinOpViewOp(rewriter, op, viewOp);
    rewriter.replaceAllUsesExcept(result, binOpViewOp.getResult(), binOpViewOp);

    return mlir::success();
  }

  /// Create the view ops that replace `viewOp`: one on top of the constant
  /// subtree and one on top of the binary op.
  virtual ErrorOr<ViewOp>
  materializeConstantSubtreeViewOp(DeviceAwareRewriter &rewriter,
                                   ElementWiseBinary binOp, ViewOp viewOp,
                                   TensorValue constantSubtreeOutput) const = 0;

  virtual ViewOp materializeBinOpViewOp(DeviceAwareRewriter &rewriter,
                                        ElementWiseBinary binOp,
                                        ViewOp viewOp) const = 0;

private:
  enum class Input {
    LHS,
    RHS,
    UNK,
  };
};

/// Swims transposes through binary operators when the other operand is
/// derived from constants. See SwimViewOpThroughBinaryOpWithConstantSubtree.
struct SwimTransposeThroughBinaryOpWithConstantSubtree
    : public SwimViewOpThroughBinaryOpWithConstantSubtree<TransposeOp> {
  SwimTransposeThroughBinaryOpWithConstantSubtree(MLIRContext *ctx)
      : SwimViewOpThroughBinaryOpWithConstantSubtree<TransposeOp>(ctx) {}

  ErrorOr<TransposeOp> materializeConstantSubtreeViewOp(
      DeviceAwareRewriter &rewriter, ElementWiseBinary binOp,
      TransposeOp viewOp,
      TensorValue constantSubtreeOutput) const override final {

    auto maybePermVec = tryGetConstantIntVector(viewOp.getPerm());

    if (!maybePermVec) {
      return {"Transpose permutation tensor was not a constant input"};
    }

    // The constant subtree gets the inverse permutation.
    const auto &permVec = maybePermVec.value();
    SmallVector<int64_t> invertedPerm(permVec.size());
    for (auto [idx, val] : llvm::enumerate(permVec)) {
      invertedPerm[val] = static_cast<int64_t>(idx);
    }

    auto invertedPermProducer =
        rewriter.createHostConstant(binOp->getLoc(), invertedPerm);

    auto constantSubtreeTranspose = rewriter.create<TransposeOp>(
        binOp->getLoc(), viewOp.getInput().getType(), constantSubtreeOutput,
        invertedPermProducer);

    return constantSubtreeTranspose;
  }

  TransposeOp materializeBinOpViewOp(DeviceAwareRewriter &rewriter,
                                     ElementWiseBinary binOp,
                                     TransposeOp viewOp) const override final {
    auto out = rewriter.create<TransposeOp>(
        binOp->getLoc(), viewOp.getResult().getType(), binOp.getResult(),
        viewOp.getPerm());
    return out;
  }
};

/// Swims static reshapes through binary operators when the other operand is
/// derived from constants. See SwimViewOpThroughBinaryOpWithConstantSubtree.
struct SwimReshapeThroughBinaryOpWithConstantSubtree
    : public SwimViewOpThroughBinaryOpWithConstantSubtree<Static::ReshapeOp> {
  SwimReshapeThroughBinaryOpWithConstantSubtree(MLIRContext *ctx)
      : SwimViewOpThroughBinaryOpWithConstantSubtree<Static::ReshapeOp>(ctx) {}

  ErrorOr<Static::ReshapeOp> materializeConstantSubtreeViewOp(
      DeviceAwareRewriter &rewriter, ElementWiseBinary binOp,
      Static::ReshapeOp viewOp,
      TensorValue constantSubtreeOutput) const override final {

    return rewriter.create<Static::ReshapeOp>(
        binOp->getLoc(), viewOp.getInput().getType(), constantSubtreeOutput);
  }

  Static::ReshapeOp
  materializeBinOpViewOp(DeviceAwareRewriter &rewriter, ElementWiseBinary binOp,
                         Static::ReshapeOp viewOp) const override final {

    return rewriter.create<Static::ReshapeOp>(
        binOp->getLoc(), viewOp.getResult().getType(), binOp.getResult());
  }
};

/// Moves a split after identical elementwise operations.
/// CONDITIONS:
/// - Split on the last axis.
/// - Last axis of the split is statically known.
/// - Split must be uniform (same size).
/// - Elementwise must be the same.
/// - Elementwise operand use must be the same.
/// - Elementwise are fed with a constant or a constant and a broadcast.
/// - The innermost dimension must be unchanged after the broadcast.
/// - All broadcasts must be the same.
/// BEFORE:
/// -> split -> elemwise ->
///          -> elemwise ->
///          -> elemwise ->
/// AFTER:
/// -> elemwise -> split ->
struct SinkSplitThroughElementwisePattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SplitOp> {
  SinkSplitThroughElementwisePattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SplitOp op,
                             DeviceAwareRewriter &rewriter) const override {
    if (!isStaticSplit(op) ||
        !allResultsAreUniquelyUsedBy<ElementWiseUnary, ElementWiseBinary>(op)) {
      return rewriter.notifyMatchFailure(
          op, "Split op must be statically ranked and have one use and one "
              "user per result");
    }

    auto firstResultType =
        llvm::cast<TensorType>(op.getResults().front().getType());
    int64_t splitAxis =
        normalizeAxis(op.getAxisAttr().getInt(), firstResultType.getRank());

    if (splitAxis != static_cast<int64_t>(firstResultType.getRank() - 1))
      return rewriter.notifyMatchFailure(op, "Split must be on last axis");

    SmallVector<Value> elementwiseValues;
    for (auto result : op.getResults()) {
      if (result.getType() != firstResultType) {
        return rewriter.notifyMatchFailure(
            op, "Split results don't have the same size");
      }

      Operation *user = result.getUses().begin()->getOwner();
      rewriter.pushOp(user);
      elementwiseValues.push_back(user->getResult(0));
    }

    auto &firstUse = *op->getResult(0).getUses().begin();
    uint64_t firstUseIndex = firstUse.getOperandNumber();
    Operation *firstOwner = firstUse.getOwner();

    SmallVector<Value> constantsToConcatenate;
    Static::BroadcastToOp broadcastOp;
    for (auto result : op.getResults()) {
      auto &use = *result.getUses().begin();

      uint64_t index = use.getOperandNumber();
      if (index != firstUseIndex) {
        return rewriter.notifyMatchFailure(
            op, "Split result use is not at the same operand index");
      }

      Operation *owner = use.getOwner();
      if (owner->getName() != firstOwner->getName()) {
        return rewriter.notifyMatchFailure(
            op, "Split results users are not all the same operation");
      }

      if (isa<ElementWiseBinary>(owner)) {
        uint64_t otherIndex = 1 - index;
        auto firstOtherOperandOp =
            firstOwner->getOperand(otherIndex).getDefiningOp();
        auto ownerOtherOperandOp =
            owner->getOperand(otherIndex).getDefiningOp();
        if (firstOtherOperandOp->getName() != ownerOtherOperandOp->getName()) {
          return rewriter.notifyMatchFailure(
              op, "Operations that feed the elementwise don't match");
        }

        if (!isa<ConstantOp, Static::BroadcastToOp>(ownerOtherOperandOp)) {
          return rewriter.notifyMatchFailure(
              op, "Operations that feed the elementwise must be constants or "
                  "static broadcasts");
        }

        auto constantOp = dyn_cast<ConstantOp>(ownerOtherOperandOp);
        if ((broadcastOp =
                 dyn_cast<Static::BroadcastToOp>(ownerOtherOperandOp))) {
          constantOp = dyn_cast<ConstantOp>(
              ownerOtherOperandOp->getOperand(0).getDefiningOp());
          if (!constantOp) {
            return rewriter.notifyMatchFailure(
                op, "Input to the static broadcast must be a constant");
          }
        }
        assert(constantOp);

        constantsToConcatenate.push_back(constantOp.getResult());
        auto constantOpType = llvm::cast<TensorType>(constantOp.getType());
        if (constantOpType.getRank() == 0 ||
            constantOpType.getDims().back() !=
                firstResultType.getDims().back()) {
          return rewriter.notifyMatchFailure(
              op, "Unsupported broadcasting that would duplicate data");
        }
      }
    }

    if (!llvm::all_of(constantsToConcatenate, [&](auto constantOp) {
          return constantOp.getType() ==
                 constantsToConcatenate.front().getType();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Constants are being broadcasted differently");
    }

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto inputType = llvm::cast<TensorType>(op.getInput().getType());
    uint64_t lastDimSize =
        *MOSH::ShapeAttr::getStaticDim(inputType.getDims().back());

    SmallVector<Value, 2> elementwiseOperands;
    if (!constantsToConcatenate.empty()) {
      TensorType constantType =
          llvm::cast<TensorType>(constantsToConcatenate.front().getType());
      auto concatType =
          constantType.cloneWithNewDim(constantType.getRank() - 1, lastDimSize);
      Value concatenatedConstants =
          rewriter
              .create<ConcatOp>(
                  loc, concatType,
                  rewriter.getIndexAttr(int64_t(constantType.getRank() - 1)),
                  constantsToConcatenate)
              .getResult();

      Value broadcast = concatenatedConstants;
      if (broadcastOp) {
        auto broadcastType =
            broadcastOp.getType().cloneWithNewDim(splitAxis, lastDimSize);
        broadcast = rewriter
                        .create<Static::BroadcastToOp>(loc, broadcastType,
                                                       concatenatedConstants)
                        .getResult();
      }

      elementwiseOperands.resize(2);
      elementwiseOperands[1 - firstUseIndex] = broadcast;
      elementwiseOperands[firstUseIndex] = op.getInput();
    } else {
      elementwiseOperands.push_back(op.getInput());
    }

    auto elementwiseType =
        llvm::cast<TensorType>(firstOwner->getResultTypes().front())
            .cloneWithNewDim(splitAxis, lastDimSize);
    // Carry over attributes; otherwise required inherent attributes are lost.
    OperationState elementwiseOpState(loc, firstOwner->getName(),
                                      elementwiseOperands, elementwiseType,
                                      firstOwner->getAttrs());
    auto newElementwise = rewriter.create(elementwiseOpState)->getResult(0);

    SmallVector<Type> newSplitTypes;
    for (auto val : elementwiseValues)
      newSplitTypes.push_back(val.getType());
    SmallVector<Value> newSplitOpOperands = op.getOperands();
    newSplitOpOperands[0] = newElementwise;
    OperationState splitOpState(loc, op->getName(), newSplitOpOperands,
                                newSplitTypes, op->getAttrs(),
                                op->getSuccessors());
    auto newSplit = rewriter.create(splitOpState);

    SmallVector<Operation *> elemwiseOps(op->getUsers());
    SmallVector<Value> splitResults(newSplit->getResults());
    for (uint64_t i = 0; i < elemwiseOps.size(); ++i) {
      rewriter.replaceAllUsesWith(elemwiseOps[i]->getResult(0),
                                  splitResults[i]);
      rewriter.eraseOp(elemwiseOps[i]);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

/// Sinks a split through a ReshapeOpType (usually a static reshape or a
/// rebind op).
/// CONDITIONS:
/// - Split axis is statically known.
/// - Split dimension is statically known.
/// - All split results are used at least once.
/// - Split must be uniform (same size).
/// - All split users must be `ReshapeOpType`.
/// - The reshape must not mix dynamic and static dimensions.
/// - The split axis must not be mixed with left dimensions after the reshape.
/// - [Unchecked] Row major-like memory layout.
/// BEFORE:
/// -> split -> reshape ->
///          -> reshape ->
///          -> reshape ->
/// AFTER:
/// -> reshape -> split ->
///                     ->
///                     ->
template <typename ReshapeOpType>
struct SinkSplitThroughStaticReshapeLikeOpsPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SplitOp> {
  SinkSplitThroughStaticReshapeLikeOpsPattern(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SplitOp op,
                             DeviceAwareRewriter &rewriter) const override {
    if (!isStaticSplit(op) || !allResultsAreUniquelyUsedBy<ReshapeOpType>(op)) {
      return rewriter.notifyMatchFailure(
          op, "Split op must be statically ranked, operate on last statically "
              "known axis and have one use and one user per result");
    }

    auto resultTypes = op->getResultTypes();
    if (!llvm::all_of(resultTypes, [&](Type resultType) {
          return resultType == resultTypes.front();
        })) {
      return rewriter.notifyMatchFailure(op, "Split is not uniform");
    }

    auto users = op->getUsers();
    if (!llvm::all_of(users, [&](Operation *user) {
          auto reshapeOp = llvm::dyn_cast<ReshapeOpType>(user);
          auto type = reshapeOp.getType();

          return type.getRank() &&
                 type == cast<ReshapeOpType>(*users.begin()).getType();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Users must be identical non-scalar reshapes");
    }

    for (Operation *user : users)
      rewriter.pushOp(user);

    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    auto splitType = llvm::cast<TensorType>(op.getType(0));
    auto shapeBeforeReshape = splitType.getDims();
    auto shapeAfterReshape =
        llvm::cast<TensorType>(users.begin()->getResults().front().getType())

            .getDims();
    const StaticReshapeDimensionsMappingAnalysis
        staticReshapeDimensionsMappingAnalysis(shapeBeforeReshape,
                                               shapeAfterReshape);
    if (!staticReshapeDimensionsMappingAnalysis)
      return rewriter.notifyMatchFailure(op, "Unsupported reshape");

    int64_t splitAxis =
        normalizeAxis(op.getAxisAttr().getInt(), splitType.getRank());

    std::optional<int64_t> newSplitAxisOpt = mapAxisThroughStaticReshape(
        staticReshapeDimensionsMappingAnalysis, splitType, splitAxis);
    if (!newSplitAxisOpt) {
      return rewriter.notifyMatchFailure(
          op, "Cannot map split axis through reshape");
    }
    const int64_t newSplitAxis = *newSplitAxisOpt;

    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);

    auto firstReshapeType = cast<ReshapeOpType>(*users.begin()).getType();
    SmallVector<TypedAttr> newReshapeShape(firstReshapeType.getDims());
    newReshapeShape[newSplitAxis] =
        rewriter.getDimAttr(*tryGetConstantInt(newReshapeShape[newSplitAxis]) *
                            op->getNumResults());
    auto newReshapeType = firstReshapeType.cloneWith(newReshapeShape);
    auto newReshape =
        rewriter.create<ReshapeOpType>(loc, newReshapeType, op.getInput())
            .getResult();

    SmallVector<Type> newSplitTypes(op.getNumResults(), firstReshapeType);
    SmallVector<int64_t> newSplitSizes(
        op.getNumResults(), firstReshapeType.getIntDim(newSplitAxis));
    SmallVector<Value> newSplitOpOperands = op.getOperands();
    newSplitOpOperands[0] = newReshape;
    newSplitOpOperands[1] = rewriter.createHostConstant(loc, newSplitSizes);
    // `axis` is an attribute, not an operand.
    SmallVector<NamedAttribute> newSplitAttrs(op->getAttrs());
    auto newSplit = rewriter.create<SplitOp>(
        loc, newSplitTypes, newSplitOpOperands[0], newSplitOpOperands[1],
        llvm::APInt(64, newSplitAxis));
    newSplit->setAttr(op.getOutputParamDeclsAttrName(),
                      op.getOutputParamDeclsAttr());

    SmallVector<Operation *> reshapeOps(users);
    SmallVector<Value> splitResults(newSplit->getResults());
    for (uint64_t i = 0; i < reshapeOps.size(); ++i) {
      rewriter.replaceAllUsesWith(reshapeOps[i]->getResult(0), splitResults[i]);
      rewriter.eraseOp(reshapeOps[i]);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

/// Swims a reshape through a slice operation.
///
/// CONDITIONS:
/// - Slice result is used exactly once by a ReshapeOpType.
/// - Slice operates on exactly one dimension (one dim differs between
///   input and output).
/// - The slice dimension is statically known in both input and output.
/// - start/stop/step are constants.
/// - The reshape must not mix dynamic and static dimensions.
/// - The slice axis must not be mixed with left dimensions after the reshape.
/// - [Unchecked] Row major-like memory layout.
/// When the slice axis expands into multiple dimensions in the reshape
/// (expansion case), additional constraints apply:
/// - step must be 1.
/// - start must be aligned to the contraction factor (product of the inner
///   expanded dims).
/// - stop must be aligned to the contraction factor or past the input size.
/// - The input dimension must be divisible by the contraction factor.
///
/// CASE 1 — 1-to-1 mapping (the sliced axis maps to a single axis after
/// reshape, contraction factor = 1):
///
///   Input [10, 4]                    Input [10, 4]
///       │                                │
///       ▼                                ▼
///   ┌───────┐                        ┌─────────┐
///   │ slice │ axis=0                 │ reshape │
///   │ [2:6:2]                        └────┬────┘
///   └───┬───┘                             │ [10, 2, 2]
///       │ [2, 4]           ─────►         ▼
///       ▼                            ┌───────┐
///   ┌─────────┐                      │ slice │ axis=0
///   │ reshape │                      │ [2:6:2]
///   └────┬────┘                      └───┬───┘
///        │ [2, 2, 2]                     │ [2, 2, 2]
///        ▼                               ▼
///
/// CASE 2 — Expansion case (the sliced axis expands into multiple axes after
/// reshape, contraction factor > 1). Axis 0 of [12, 4] expands into [N, 2]
/// (factor = 2), so start/stop are divided by 2:
///
///   Input [12, 4]                    Input [12, 4]
///       │                                │
///       ▼                                ▼
///   ┌───────┐                        ┌─────────┐
///   │ slice │ axis=0                 │ reshape │
///   │ [4:8:1]                        └────┬────┘
///   └───┬───┘                             │ [6, 2, 4]
///       │ [4, 4]           ─────►         ▼
///       ▼                            ┌───────┐
///   ┌─────────┐                      │ slice │ axis=0
///   │ reshape │                      │ [2:4:1]
///   └────┬────┘                      └───┬───┘
///        │ [2, 2, 4]                     │ [2, 2, 4]
///        ▼                               ▼
template <typename ReshapeOpType>
struct SwimStaticReshapeLikeOpsThroughSliceOp
    : public DeviceAwareRewritePattern<OpRewritePattern, SliceOp> {
  SwimStaticReshapeLikeOpsThroughSliceOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SliceOp op,
                             DeviceAwareRewriter &rewriter) const override {
    auto result = op.getResult();
    auto users = result.getUsers();
    if (llvm::range_size(users) != 1 || !isa<ReshapeOpType>(*users.begin())) {
      return rewriter.notifyMatchFailure(
          op, "Slice result must have exactly one user of ReshapeOpType");
    }

    auto reshapeOp = cast<ReshapeOpType>(*users.begin());

    rewriter.pushOp(reshapeOp);
    if (!rewriter.addOps()) {
      return rewriter.notifyMatchFailure(
          op, "matched ops have inconsistent devices");
    }

    auto inputType = llvm::cast<TensorType>(op.getInput().getType());
    auto sliceType = llvm::cast<TensorType>(op.getResult().getType());
    auto reshapeType = reshapeOp.getType();

    if (!inputType.hasRank() || !sliceType.hasRank() || !reshapeType.hasRank())
      return rewriter.notifyMatchFailure(op, "Types must be ranked");

    int64_t sliceAxis = -1;
    int64_t differentDimensions = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(inputType.getRank()); ++i) {
      if (inputType.getDim(i) != sliceType.getDim(i)) {
        sliceAxis = i;
        ++differentDimensions;
      }
    }

    if (differentDimensions != 1) {
      return rewriter.notifyMatchFailure(
          op, "Slice must operate on exactly one dimension");
    }

    if (!isStaticDimAttr(inputType.getDim(sliceAxis)) ||
        !isStaticDimAttr(sliceType.getDim(sliceAxis))) {
      return rewriter.notifyMatchFailure(
          op, "Slice dimension must be statically known");
    }

    // The reshape analysis skips 1-dims, so a size-1 slice axis would remap to
    // another axis while the rebuild still uses sliceAxis, producing a reshape
    // whose numel does not match the input.
    if (sliceType.getIntDim(sliceAxis) == 1) {
      return rewriter.notifyMatchFailure(
          op, "Slice that reduces an axis to size 1 is not supported");
    }

    auto startValues = tryGetConstantIntVector(op.getStart());
    auto stepValues = tryGetConstantIntVector(op.getStep());
    if (!startValues || !stepValues) {
      return rewriter.notifyMatchFailure(op, "start/step must be constants");
    }
    int64_t stopAtAxis;
    if (auto stopValues = tryGetConstantIntVector(op.getStop())) {
      stopAtAxis = (*stopValues)[sliceAxis];
    } else if ((*stepValues)[sliceAxis] == 1) {
      stopAtAxis = (*startValues)[sliceAxis] + sliceType.getIntDim(sliceAxis);
    } else {
      return rewriter.notifyMatchFailure(
          op, "stop must be constant unless the slice axis has unit step");
    }

    const StaticReshapeDimensionsMappingAnalysis
        staticReshapeDimensionsMappingAnalysis(sliceType.getDims(),
                                               reshapeType.getDims());
    if (!staticReshapeDimensionsMappingAnalysis)
      return rewriter.notifyMatchFailure(op, "Unsupported reshape");

    std::optional<int64_t> newSliceAxisOpt = mapAxisThroughStaticReshape(
        staticReshapeDimensionsMappingAnalysis, sliceType, sliceAxis);
    if (!newSliceAxisOpt) {
      return rewriter.notifyMatchFailure(
          op, "Cannot map slice axis through reshape");
    }
    int64_t newSliceAxis = *newSliceAxisOpt;

    // Reject contraction (e.g. [K, M] → [K*M]): rebuilding the reshape type
    // would need the other contracted dims, not just newSliceAxis.
    // TODO: Improve this for the contraction case.
    if (staticReshapeDimensionsMappingAnalysis.backwardMapping()[newSliceAxis]
            .size() > 1) {
      return rewriter.notifyMatchFailure(
          op, "Slice axis participates in a dimension contraction");
    }

    // Contraction factor: product of the inner dims the slice axis expands
    // into (1 for a 1-to-1 mapping).
    uint64_t mappedSliceAxis = sliceAxis;
    while (mappedSliceAxis < sliceType.getRank() &&
           sliceType.getIntDim(mappedSliceAxis) == 1) {
      ++mappedSliceAxis;
    }
    const auto &expandedDims = staticReshapeDimensionsMappingAnalysis
                                   .forwardMapping()[mappedSliceAxis];
    int64_t contractionFactor = 1;
    for (size_t i = 1; i < expandedDims.size(); ++i)
      contractionFactor *= reshapeType.getIntDim(expandedDims[i]);

    // Slice boundaries must fall on whole rows of the inner dims.
    if (contractionFactor > 1) {
      if ((*stepValues)[sliceAxis] != 1) {
        return rewriter.notifyMatchFailure(op,
                                           "Expansion case requires step == 1");
      }
      if ((*startValues)[sliceAxis] % contractionFactor != 0) {
        return rewriter.notifyMatchFailure(
            op, "start is not aligned to contraction factor");
      }
      if (inputType.getIntDim(sliceAxis) % contractionFactor != 0) {
        return rewriter.notifyMatchFailure(
            op, "Input dimension is not divisible by contraction factor");
      }
      int64_t stop = stopAtAxis;
      int64_t inputDimSize = inputType.getIntDim(sliceAxis);
      if (stop % contractionFactor != 0 && stop < inputDimSize) {
        return rewriter.notifyMatchFailure(
            op, "stop is not aligned to contraction factor");
      }
    }

    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);

    SmallVector<TypedAttr> newReshapeShape(reshapeType.getDims());
    newReshapeShape[newSliceAxis] =
        rewriter.getDimAttr(inputType.getIntDim(sliceAxis) / contractionFactor);
    auto newReshapeType = reshapeType.cloneWith(newReshapeShape);
    auto newReshape =
        rewriter.create<ReshapeOpType>(loc, newReshapeType, op.getInput())
            .getResult();

    // Dividing by the contraction factor addresses the outer expanded dim.
    int64_t newRank = reshapeType.getRank();
    SmallVector<int64_t> newStart(newRank, 0);
    SmallVector<int64_t> newStop(newRank, std::numeric_limits<int64_t>::max());
    SmallVector<int64_t> newStep(newRank, 1);
    newStart[newSliceAxis] = (*startValues)[sliceAxis] / contractionFactor;
    newStop[newSliceAxis] = stopAtAxis / contractionFactor;
    newStep[newSliceAxis] = (*stepValues)[sliceAxis];

    Value newStartVal = rewriter.createHostConstant(loc, newStart);
    Value newStopVal = rewriter.createHostConstant(loc, newStop);
    Value newStepVal = rewriter.createHostConstant(loc, newStep);

    auto newSlice = rewriter.create<SliceOp>(
        loc, reshapeType, newReshape, newStartVal, newStopVal, newStepVal);

    rewriter.replaceAllUsesWith(reshapeOp.getResult(), newSlice.getResult());
    rewriter.eraseOp(reshapeOp);
    rewriter.eraseOp(op);

    return success();
  }
};

/// Attempts to sink broadcasts through reduction style operators.
///
///   Input                    Input      Axis
///     │                         │         │
///     ▼                         │         │
/// ┌─────────┐                   ▼         ▼
/// │Broadcast│   Axis           ┌───────────┐
/// └────┬────┘    │             │ReductionOp│
///      │         │             └─────┬─────┘
///      ▼         ▼    ─────►         │
///     ┌───────────┐                  ▼
///     │ReductionOp│             ┌─────────┐
///     └─────┬─────┘             │Broadcast│
///           │                   └────┬────┘
///           ▼                        │
///                                    ▼
///
/// This holds when the reduction axis is not modified by the broadcast, since
/// broadcast is a "view"/"reindexing" of the original buffer. For example if:
///    %T1 : !mo.tensor<[A, B, C, 1], f32>
///    %T2 : mo.static.broadcast_to(%A):
///         !mo.tensor<[A, B, C, 1], f32> -> !mo.tensor<[P1, A, B, C, P2], f32>
///
/// Then T2[p1, a, b, c, p2] == T1[a, b, c, 0] for all valid variables!
///
/// Therefore if we are reduce-add across dimension 'A':
///
/// T2[p1, 0, b, c, p2] = \sum_i T2[p1, i, b, c, p2]
///                     = \sum_i T1[i, b, c, 0]
///
/// Which is a reduction across dimension 'A' in the original tensor, with
/// a broadcast with the same initial output shape!
///
/// When the broadcast modifies the reduction axis, the values along it are
/// identical and the reduction becomes a simpler op before the BroadcastTo:
/// - For reduce.add (reduce.mul), the input must be multiplied (use pow) by the
/// size of the axis.
/// - For min / max / mean, the input isn't modified (identity).
/// - For argmin / argmax, the input is transformed into a tensor of 0s.
struct SinkBroadcastsThroughReduction
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern, Reduction> {
  SinkBroadcastsThroughReduction(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(Reduction op,
                             DeviceAwareRewriter &rewriter) const override {
    TensorValue input = op.getInput();
    Location loc = op->getLoc();
    MLIRContext *ctx = op->getContext();

    auto bcastOp = input.getDefiningOp<Static::BroadcastToOp>();
    if (!bcastOp)
      return rewriter.notifyMatchFailure(loc, "Not preceded by broadcast.");

    auto bcastOpInput = bcastOp.getInput();
    if (!bcastOpInput.getType().hasRank() || !bcastOp.getType().hasRank())
      return rewriter.notifyMatchFailure(
          loc, "BCast to be statically ranked to work.");

    std::optional<int64_t> axisOr = op.getReductionAxis();
    if (!axisOr)
      return rewriter.notifyMatchFailure(loc, "reduction has no axis");
    int64_t reductionAxis = *axisOr;

    // Map the reduction axis onto the non-broadcasted input by counting the
    // broadcast-added dims before it. E.g. for [A, B, 1] --> [1, D, A, B, 1]
    // reducing along 'A', that count is 2 (`1` and `D`).
    auto bcastShapeAttr = bcastOp.getType().getShape();
    ErrorOr<SmallVector<int64_t>> maybeBcastedAxis =
        ShapeAttr::getBroadcastedAxes(bcastOp.getInput().getType().getShape(),
                                      bcastShapeAttr);
    if (maybeBcastedAxis.isError())
      return rewriter.notifyMatchFailure(loc, maybeBcastedAxis.getError());

    SmallVector<int64_t> &bcastedAxis = maybeBcastedAxis.get();
    size_t numBeforeAxis = 0;
    for (int64_t bcastedDimIndex : bcastedAxis) {
      if (bcastedDimIndex == reductionAxis) {
        auto maybeNewInput =
            TypeSwitch<Operation *, FailureOr<TensorValue>>(op)
                .Case<ReduceMinOp, ReduceMaxOp>([&](auto op) {
                  // For min / max, the reduction is removed.
                  return bcastOpInput;
                })
                .Case<ReduceAddOp, ReduceMulOp>([&](auto op) {
                  // For reduce.add, the input is multiplied by the size of the
                  // broadcast axis. (For reduce.mul, we use mo.pow instead).
                  TypedAttr dimAttr = bcastShapeAttr[reductionAxis];
                  auto dimIndex =
                      rewriter.create<MOSH::ParamToValueOp>(loc, dimAttr);
                  auto dimT = rewriter.create<IndexToTensorOp>(
                      loc,
                      TensorType::get(
                          ArrayRef<int64_t>({}),
                          IntegerType::get(ctx, 64, IntegerType::Unsigned),
                          input.getType().getDeviceRef(),
                          input.getType().getMetadata()),
                      dimIndex);
                  auto dimCast = rewriter.create<CastOp>(
                      loc, dimT, bcastOpInput.getType().getDtype());
                  auto dimBroadcast = rewriter.create<Static::BroadcastToOp>(
                      loc,
                      dimCast.getType().cloneWith(
                          bcastOpInput.getType().getShape()),
                      dimCast);
                  TensorValue scaledInput =
                      isa<ReduceAddOp>(op)
                          ? rewriter
                                .create<MulOp>(loc, bcastOpInput, dimBroadcast)
                                .getResult()
                          : rewriter
                                .create<PowOp>(loc, bcastOpInput.getType(),
                                               bcastOpInput, dimBroadcast)
                                .getResult();
                  return scaledInput;
                })
                .Case<ArgMinOp, ArgMaxOp>([&](auto op) {
                  // For arg_min / arg_max, the output becomes a tensor of 0s.
                  auto v0 = rewriter.create<MO::ConstantOp>(
                      loc, 0, input.getType().getDeviceRef());
                  TensorValue v0Cast = rewriter.create<CastOp>(
                      loc, v0, op.getResult().getType().getDtype());
                  return v0Cast;
                })
                .Default([&](auto op) {
                  return rewriter.notifyMatchFailure(
                      loc, "Cannot optimize if reducing along axes modified by "
                           "broadcast: Unsupported reduce operation");
                });
        if (failed(maybeNewInput))
          return failure();

        rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(
            op, op.getResult().getType(), *maybeNewInput);
        return success();
      }
      if (bcastedDimIndex > reductionAxis)
        break;
      ++numBeforeAxis;
    }

    // Reducing along a non-broadcasted axis: reduce first, broadcast after.
    int64_t newAxis = reductionAxis - numBeforeAxis;
    auto clonedReduction = cast<MO::Reduction>(*rewriter.clone(*op));
    clonedReduction.getInputMutable().assign(bcastOpInput);
    // An `axis` attribute moves; a shape reduction reads the new result type.
    clonedReduction.setAxisAttribute(newAxis);

    TensorType newTy = bcastOpInput.getType()
                           .cloneWithNewDim(newAxis, 1)
                           .cloneWith(op.getResult().getType().getDtype());
    clonedReduction.getResult().setType(newTy);

    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(
        op, op.getResult().getType(), clonedReduction.getResult());

    return success();
  }
};

template <typename ElemwiseInterfaceT>
struct SinkBroadcastsThroughElemwise
    : public DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                       ElemwiseInterfaceT> {
  SinkBroadcastsThroughElemwise(MLIRContext *ctx)
      : DeviceAwareRewritePattern<OpInterfaceRewritePattern,
                                  ElemwiseInterfaceT>(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(ElemwiseInterfaceT op,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<TypedAttr> shapeParams;
    for (Value arg : op->getOperands()) {
      auto bcastOp = arg.getDefiningOp<Static::BroadcastToOp>();
      if (!bcastOp)
        return rewriter.notifyMatchFailure(loc, "not preceded by broadcast");
      shapeParams.emplace_back(bcastOp.getInput().getType().getShape());
    }

    TypedAttr commonBcastShape =
        ShapeAttr::getCommonBroadcastShape(shapeParams);

    if (!commonBcastShape)
      return rewriter.notifyMatchFailure(
          loc, "No common shape can be proven for broadcasts.");

    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          loc, "Assume only one result from elemwise op.");

    TensorType oldTy = cast<TensorType>(op->getResult(0).getType());
    auto commonBcastShapeAttr = cast<MOSH::ShapeAttr>(commonBcastShape);
    if (commonBcastShapeAttr == oldTy.getShape())
      return rewriter.notifyMatchFailure(
          loc, "Already at the mode reduced broadcast form.");

    TensorValue resVal = cast<TensorValue>(op->getResult(0));
    Operation *clonedOp = rewriter.clone(*op);
    TensorValue clonedResVal = cast<TensorValue>(clonedOp->getResult(0));

    rewriter.setInsertionPoint(clonedOp);
    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      auto oldBcastOp =
          op->getOperand(i).template getDefiningOp<Static::BroadcastToOp>();
      TensorType targetType =
          oldBcastOp.getType().cloneWith(commonBcastShapeAttr);

      if (oldBcastOp.getType() != targetType) {
        TensorValue newOperand = rewriter.create<Static::BroadcastToOp>(
            oldBcastOp->getLoc(), targetType, oldBcastOp.getInput());
        clonedOp->setOperand(i, newOperand);
      }
    }
    clonedResVal.setType(resVal.getType().cloneWith(commonBcastShapeAttr));

    // To type check, insert a broadcast back to the original shape
    rewriter.setInsertionPointAfter(clonedOp);
    auto newResVal = rewriter.create<Static::BroadcastToOp>(
        op->getLoc(), resVal.getType(), clonedResVal);
    rewriter.replaceAllUsesWith(resVal, newResVal);

    return success();
  }
};

struct SinkBroadcastsThroughLinalgBandPartOp
    : public DeviceAwareRewritePattern<OpRewritePattern, LinAlg::BandPartOp> {
  SinkBroadcastsThroughLinalgBandPartOp(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(LinAlg::BandPartOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MO::TensorType resTy = op.getType();
    if (!resTy.hasRank())
      return rewriter.notifyMatchFailure(loc, "not statically ranked");

    MO::TensorValue inputVal = op.getInput();
    auto bcastOp = inputVal.getDefiningOp<MO::Static::BroadcastToOp>();
    if (!bcastOp)
      return rewriter.notifyMatchFailure(loc, "not preceded by broadcast");

    MO::TensorValue bcastInput = bcastOp.getInput();
    MO::TensorType bcastInputTy = bcastInput.getType();
    if (!bcastInputTy.hasRank())
      return rewriter.notifyMatchFailure(
          loc, "preceding broadcast has unranked input");

    MO::TensorType inputTy = inputVal.getType();
    SmallVector<TypedAttr> newBcastResDims;
    if (bcastInputTy.getRank() > 2) {
      ArrayRef<TypedAttr> inBatchDims = bcastInputTy.getDims().drop_back(2);
      ArrayRef<TypedAttr> outBatchDims = inputTy.getDims().drop_back(2);
      if (inBatchDims.equals(outBatchDims))
        return rewriter.notifyMatchFailure(
            loc, "preceding broadcast with batch dims canonical");
      newBcastResDims.append(inBatchDims.begin(), inBatchDims.end());
    } else {
      // From below rank 2, it's enough to broadcast to rank 2.
      if (inputTy.getRank() == 2)
        return rewriter.notifyMatchFailure(
            loc, "preceding broadcast without batch dims canonical");
    }

    // The new broadcast skips the batch dims; if it doesn't broadcast the
    // trailing two dims either, the broadcast canonicalizer erases it.
    ArrayRef<TypedAttr> matDims = inputTy.getDims().take_back(2);
    newBcastResDims.append(matDims.begin(), matDims.end());

    auto shapeTy = rewriter.getType<ShapeType>();
    MO::TensorType newBcastResTy =
        inputTy.cloneWith(ShapeAttr::get(newBcastResDims, shapeTy));
    auto newBcastOp = rewriter.create<MO::Static::BroadcastToOp>(
        loc, newBcastResTy, bcastInput);

    SmallVector<Value> newOperands(op.getOperands());
    newOperands[0] = newBcastOp;
    auto newBandPartOp = rewriter.create<MO::LinAlg::BandPartOp>(
        loc, newBcastResTy, newOperands, op.getProperties(),
        llvm::to_vector(op.getOperation()->getDiscardableAttrs()));

    rewriter.replaceOpWithNewOp<MO::Static::BroadcastToOp>(op, resTy,
                                                           newBandPartOp);
    return success();
  }
};

} // namespace

namespace M::MO {

void populateShapeOpSinkingPatterns(mlir::RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<SinkTransposeThroughUnaryOpPattern,
               SinkTransposeThroughBinaryOpPattern,
               SinkTransposeThroughStaticBroadcastToOpPattern>(context);

  patterns.add<
      SinkStaticReshapeOpThroughSoftmaxOp,
      SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp<
          Static::ReshapeOp>,
      SinkStaticReshapeLikeOpThroughBroadcastedElementwiseUnaryOp<RebindOp>,
      SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp<
          Static::ReshapeOp>,
      SinkStaticReshapeLikeOpThroughBroadcastedElementwiseBinaryOp<RebindOp>,
      SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp<Static::ReshapeOp>,
      SinkSameStaticReshapeLikeOpsThroughElementwiseBinaryOp<RebindOp>,
      SinkSqueezeLikeReshapeThroughTranspose,
      RemoveExtraneousPaddingReshapeWithBroadcast,
      SwimReshapeThroughBinaryOpWithConstantSubtree,
      SwimTransposeThroughBinaryOpWithConstantSubtree>(context);

  patterns.add<SinkBroadcastsThroughReduction,
               CommuteBroadcastWithFlatennedArgNonzero,
               SinkBroadcastsThroughElemwise<ElementWiseBinary>,
               SinkBroadcastsThroughElemwise<ElementWiseUnary>,
               SinkBroadcastsThroughLinalgBandPartOp>(context);

  patterns.add<SinkSplitThroughStaticReshapeLikeOpsPattern<Static::ReshapeOp>,
               SinkSplitThroughStaticReshapeLikeOpsPattern<RebindOp>,
               SinkSplitThroughElementwisePattern>(context);

  patterns.add<SwimStaticReshapeLikeOpsThroughSliceOp<Static::ReshapeOp>,
               SwimStaticReshapeLikeOpsThroughSliceOp<RebindOp>>(context);

  patterns.add<SwimConcatThroughReshapeLikeOpsPattern<Static::ReshapeOp>,
               SwimConcatThroughReshapeLikeOpsPattern<RebindOp>>(context);
}

} // namespace M::MO
