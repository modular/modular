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
// Patterns converting between adjacent slice ops and a single split op, shared
// by SymbolicOptimization.cpp and PatternFusion.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGSUPPORT_SPLITSLICEPATTERNS_H
#define GENERICML_GRAPHCOMPILER_MOGGSUPPORT_SPLITSLICEPATTERNS_H

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/Support/Dims.h"

#include <limits>

namespace M::MO {

/// Replaces slice ops that together partition one axis of a common input with
/// a single split op.
template <bool AllowDynamicStop = false>
struct SliceToSplitPattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SliceOp> {
  SliceToSplitPattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SliceOp op,
                             DeviceAwareRewriter &rewriter) const override {
    SmallVector<SliceOp> sliceOps;
    Value input = op.getInput();
    for (auto user : input.getUsers()) {
      if (auto sliceOp = dyn_cast<SliceOp>(user))
        sliceOps.push_back(sliceOp);
    }

    if (sliceOps.size() == 1)
      return rewriter.notifyMatchFailure(op, "Only one slice op");

    if (!llvm::all_of(sliceOps, [](SliceOp sliceOp) {
          bool startStepConstant =
              sliceOp.getStart().getDefiningOp<ConstantOp>() &&
              sliceOp.getStep().getDefiningOp<ConstantOp>();
          if constexpr (AllowDynamicStop)
            return startStepConstant;
          else
            return startStepConstant &&
                   sliceOp.getStop().getDefiningOp<ConstantOp>();
        })) {
      return rewriter.notifyMatchFailure(
          op, "Start and step operands must be constants");
    }

    auto inputType = llvm::cast<TensorType>(input.getType());
    int64_t sliceDimension = -1;
    const auto sliceHasStaticRankKnownShapeAndSlicesOneDimension =
        [&](SliceOp sliceOp) {
          auto type = sliceOp.getResult().getType();
          if (!type.hasRank())
            return false;

          int64_t differentDimensions = 0;
          int64_t i = 0;
          for (auto [before, after] :
               llvm::zip(inputType.getDims(), type.getDims())) {
            if (before != after) {
              sliceDimension = i;
              ++differentDimensions;
            }
            ++i;
          }

          return differentDimensions == 1;
        };

    if (!llvm::all_of(sliceOps,
                      sliceHasStaticRankKnownShapeAndSlicesOneDimension)) {
      return rewriter.notifyMatchFailure(
          op, "All slices must have a static rank, a known shape and slice "
              "through only one dimension");
    }

    if (llvm::any_of(sliceOps, [&](SliceOp sliceOp) {
          return sliceOp.getType().getIntDim(sliceDimension) ==
                 ShapedType::kDynamic;
        })) {
      return rewriter.notifyMatchFailure(
          op, "Sliced dimension must be statically known for all slices");
    }

    int64_t nElements = 0;
    for (SliceOp sliceOp : sliceOps)
      nElements += sliceOp.getType().getIntDim(sliceDimension);

    if (nElements != inputType.getIntDim(sliceDimension)) {
      return rewriter.notifyMatchFailure(
          op, "Union of slices must cover all of the input");
    }

    const auto getSliceValue = [&](Value value) {
      return SmallVector<APInt>(
                 cast<IntArrayElementsAttr>(
                     value.getDefiningOp<ConstantOp>().getValue())
                     .getValues())[sliceDimension]
          .getSExtValue();
    };

    // Split results are consecutive chunks, so order slices by their real
    // start. `mo.slice` bounds are numpy-style (-8 on a 32-wide axis starts at
    // 24); `clampAxis` matches shape inference's normalization.
    const int64_t sliceDimSize = inputType.getIntDim(sliceDimension);
    const auto sliceStart = [&](SliceOp sliceOp) {
      return clampAxis(getSliceValue(sliceOp.getStart()), sliceDimSize);
    };
    // Use the result type's clamped extent; a stop bound may be negative or
    // `int64` max.
    const auto sliceSize = [&](SliceOp sliceOp) {
      return sliceOp.getType().getIntDim(sliceDimension);
    };

    // Strided slices can cover an axis while interleaving rather than
    // partitioning it (two step-2 slices of a 32-wide axis).
    if (llvm::any_of(sliceOps, [&](SliceOp sliceOp) {
          return getSliceValue(sliceOp.getStep()) != 1;
        })) {
      return rewriter.notifyMatchFailure(op, "Sliced dimension step must be 1");
    }

    // Under `AllowDynamicStop` the extent comes from the declared result type,
    // not shape inference, so it can run past the axis.
    if (llvm::any_of(sliceOps, [&](SliceOp sliceOp) {
          return sliceStart(sliceOp) + sliceSize(sliceOp) > sliceDimSize;
        })) {
      return rewriter.notifyMatchFailure(
          op, "Slice falls outside the sliced dimension");
    }

    std::sort(sliceOps.begin(), sliceOps.end(), [&](SliceOp lhs, SliceOp rhs) {
      return sliceStart(lhs) < sliceStart(rhs);
    });

    for (uint64_t i = 1; i < sliceOps.size(); ++i) {
      if (sliceStart(sliceOps[i]) <
          sliceStart(sliceOps[i - 1]) + sliceSize(sliceOps[i - 1])) {
        return rewriter.notifyMatchFailure(op, "Slices are overlapping");
      }
    }

    SmallVector<int64_t> splitSizes;
    for (SliceOp sliceOp : sliceOps)
      splitSizes.push_back(sliceSize(sliceOp));

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);

    SmallVector<Type> splitTypes;
    for (SliceOp sliceOp : sliceOps)
      splitTypes.push_back(sliceOp.getType());
    auto splitSizesOp = rewriter.createHostConstant(loc, splitSizes);

    SmallVector<Value> originalSliceValues;
    for (SliceOp sliceOp : sliceOps)
      originalSliceValues.push_back(sliceOp.getResult());

    auto splitOp = rewriter.create<SplitOp>(loc, splitTypes, input,
                                            splitSizesOp.getResult(),
                                            llvm::APInt(64, sliceDimension));

    rewriter.replaceAllUsesWith(originalSliceValues, splitOp.getResults());
    for (SliceOp sliceOp : sliceOps)
      rewriter.eraseOp(sliceOp);

    return success();
  }
};

/// Replaces a split op with one slice op per result. `SkipAxisZero` avoids
/// clashing with MGP optimizations on axis 0.
template <bool SkipAxisZero = false>
struct SplitToSlicePattern
    : public DeviceAwareRewritePattern<OpRewritePattern, SplitOp> {
  SplitToSlicePattern(MLIRContext *ctx) : DeviceAwareRewritePattern(ctx, 2) {}

  LogicalResult
  deviceAwareMatchAndRewrite(SplitOp op,
                             DeviceAwareRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value in = op.getInput();

    if (!op.getOutputParamDecls().empty())
      return rewriter.notifyMatchFailure(op,
                                         "Op needs a shape function first.");

    TensorType inTy = cast<TensorType>(in.getType());
    if (!inTy.hasRank())
      return rewriter.notifyMatchFailure(op, "Pattern needs known input shape");
    size_t rank = inTy.getRank();

    // Output users won't be fused in.
    for (Value out : op.getResults()) {
      for (Operation *user : out.getUsers()) {
        if (isa<OutputOp>(user))
          return rewriter.notifyMatchFailure(op, "Split used in output.");
      }
    }

    if (llvm::any_of(op.getResultTypes(), [&](Type type) {
          auto tensor = cast<TensorType>(type);
          return !tensor.hasRank() || tensor.getRank() != rank;
        }))
      return rewriter.notifyMatchFailure(op, "Output shape is unknown.");

    int64_t axis = op.getAxisAttr().getInt();

    if constexpr (SkipAxisZero) {
      if (axis == 0)
        return rewriter.notifyMatchFailure(
            op, "Avoid clashing with MGP optimization when axis == 0");
    }

    axis = M::normalizeAxis(axis, rank);

    std::optional<SmallVector<int64_t>> maybeSplits =
        tryGetConstantIntVector(op.getSplitSizes());
    if (!maybeSplits.has_value())
      return rewriter.notifyMatchFailure(op, "Splits are not constant");
    SmallVector<int64_t> splitSizes = *maybeSplits;

    SmallVector<int64_t> step(rank, 1);
    DeviceRefAttr host = getHostDevice(op);
    Value stepVal = rewriter.create<ConstantOp>(loc, step, host);

    int64_t startVal = 0, splitIndex = 0;

    for (Value result : op.getResults()) {
      SmallVector<int64_t> start, stop;

      for (int64_t i = 0; i < static_cast<int64_t>(rank); ++i) {
        if (axis == i) {
          start.push_back(startVal);
          stop.push_back(startVal + splitSizes[splitIndex]);
          startVal += splitSizes[splitIndex];
          ++splitIndex;
        } else {
          start.push_back(0);
          stop.push_back(std::numeric_limits<int64_t>::max());
        }
      }
      Value startTensor = rewriter.create<ConstantOp>(loc, start, host);
      Value stopTensor = rewriter.create<ConstantOp>(loc, stop, host);

      Value split = rewriter.create<SliceOp>(loc, result.getType(), in,
                                             startTensor, stopTensor, stepVal);
      rewriter.replaceAllUsesWith(result, split);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MOGGSUPPORT_SPLITSLICEPATTERNS_H
