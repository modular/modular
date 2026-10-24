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
// This file implements canonicalization and folding methods for
// MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/Permutation.h"
#include "Support/ML/RangeUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {
bool isOne(Value val) {
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp)
    return false;

  // Look through views.
  if (isa<RebindOp, Static::BroadcastToOp, SliceOp, TransposeOp>(definingOp))
    return isOne(definingOp->getOperand(0));

  if (auto asCst = dyn_cast<ConstantOp>(definingOp)) {
    TensorValue out = asCst.getResult();
    size_t rank = asCst.getType().getRank();
    Type elemTy = out.getType().getElementType();
    if (elemTy.isInteger(1)) {
      if (auto maybeBlob = tryGetConstantOfRank<bool>(out, rank)) {
        return llvm::all_of(maybeBlob.value(), [](bool b) { return b; });
      }
    } else if (elemTy.isInteger()) {
      if (auto maybeBlob = tryGetConstantOfRank<int64_t>(out, rank)) {
        return llvm::all_of(maybeBlob.value(),
                            [](int64_t v) { return v == 1; });
      }
    } else if (elemTy.isIntOrFloat()) {
      if (auto maybeBlob = tryGetConstantOfRank<float>(out, rank)) {
        return llvm::all_of(maybeBlob.value(),
                            [](float v) { return v == 1.0f; });
      }
    }
  }

  return false;
}

/// If input rank is known, rewrites `op`'s `axis` into [0, rank).
/// NOTE this should be used as the first canonicalization, so other logic in
/// canonicalization will be able to utilize it.
LogicalResult
canonicalizeStaticNegativeIndicesInPlace(Reduction op, int operandIndex,
                                         PatternRewriter &rewriter) {
  // `operandIndex` is unused: `axis` is an attribute, not an operand.
  (void)operandIndex;
  TensorType inputTy = op.getInput().getType();
  if (!inputTy.hasRank())
    return failure();

  IntegerAttr axisAttr = op.getAxisAttribute();
  if (!axisAttr)
    return failure();
  int64_t axis = axisAttr.getInt();
  if (axis >= 0)
    return failure();

  int64_t newAxis = axis + inputTy.getRank();
  rewriter.modifyOpInPlace(op, [&] { op.setAxisAttribute(newAxis); });
  return success();
}

template <typename SliceTy, typename AdaptorTy>
OpFoldResult foldSliceHelper(SliceTy slice, AdaptorTy adaptor) {
  auto steps = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getStep());
  if (!steps)
    return {};

  TypedValue<TensorType> input = slice.getInput();
  TensorType inputTy = input.getType();

  if (slice.getResult().getType() != inputTy)
    return {};

  // A step of -1 can also preserve the shape, so same shape is only an
  // identity when every step is 1.
  auto stepsData = tryGetConstantBlob<int64_t>(
      cast<ElementsAttr>(steps), slice.getStep().getType().getDtype());
  if (!stepsData.has_value())
    return {};

  for (int64_t step : *stepsData) {
    if (step != 1)
      return {};
  }
  return input;
}

ArrayElementsAttr foldReshape(ArrayRef<int64_t> outIntDims,
                              ArrayElementsAttr inputArr) {
  auto outTy = RankedTensorType::get(outIntDims, inputArr.getElementType());
  return ArrayElementsAttr::get(inputArr.getRawData(), outTy);
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                         Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Find the closest IsolatedFromAbove parent region for moving an operation to.
/// Taken from `mlir/Transforms/Utils/FoldUtils.cpp`.
Region *getInsertionRegion(Block *insertionBlock) {
  while (Region *region = insertionBlock->getParent()) {
    auto *parentOp = region->getParentOp();
    if (parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>() ||
        !parentOp->getBlock())
      return region;
    insertionBlock = parentOp->getBlock();
  }
  llvm_unreachable("expected valid insertion region");
}

LogicalResult materializeShapeInference(RebindOp op,
                                        PatternRewriter &rewriter) {
  ArrayRef<KGEN::ParamDeclAttr> paramDecls = op.getOutputParamDecls();
  if (paramDecls.empty())
    return failure();
  // TODO lift into a method on MO::TensorType?
  Location loc = op.getLoc();
  TypedAttr inputShapeAttr = op.getInput().getType().getShape();
  if (paramDecls.size() == 1) {
    KGEN::ParamDeclAttr newDecl = paramDecls.front();
    if (isa<ShapeType>(newDecl.getType())) {
      KGEN::ParamDeclareOp::create(rewriter, loc, newDecl, inputShapeAttr);
      rewriter.modifyOpInPlace(op, [&] { op.setOutputParamDecls({}); });
      return success();
    }
  }
  ShapeAttr outputDimAttrs = op.getResult().getType().getShape();
  bool allDeclsAreDimAttrs = llvm::all_of(paramDecls, [](ParamDeclAttr decl) {
    return ShapeAttr::isDimType(decl.getType());
  });
  if (allDeclsAreDimAttrs && outputDimAttrs) {
    llvm::SmallDenseMap<StringAttr, KGEN::ParamDeclAttr> nameToParamDecl;
    for (KGEN::ParamDeclAttr paramDecl : paramDecls)
      nameToParamDecl.try_emplace(paramDecl.getName(), paramDecl);
    if (auto inputShapeParamRef =
            dyn_cast<KGEN::ParamDeclRefAttr>(inputShapeAttr)) {
      // Example:
      //   rebind<() -> D1> : Sh0 -> [D0, D1, 44]
      // --->
      //   kgen.param.declare D1 = <apply("mosh.get_dim", Sh0, 1)>
      //   rebind : Sh0 -> [D0, D1, 44]
      for (auto [axis, outputDimAttr] :
           llvm::enumerate(outputDimAttrs.getValues())) {
        KGEN::ParamDeclAttr paramDecl;
        auto outputDimParamRef =
            dyn_cast<KGEN::ParamDeclRefAttr>(outputDimAttr);
        if (outputDimParamRef &&
            (paramDecl = nameToParamDecl.lookup(outputDimParamRef.getName()))) {
          TypedAttr mlirOpCallAttr =
              MOSH::GetDimOp::emitStaticCall(inputShapeParamRef, axis);
          KGEN::ParamDeclareOp::create(rewriter, loc, paramDecl,
                                       mlirOpCallAttr);
        }
      }
      rewriter.modifyOpInPlace(op, [&] { op.setOutputParamDecls({}); });
      return success();
    }
    if (auto inputDimAttrs = dyn_cast<ShapeAttr>(inputShapeAttr)) {
      // Example:
      //   rebind<() -> D4> : [D0, D1, D2] -> [D3, D4, 42]
      // --->
      //   kgen.param.declare D4 = <D1>
      //   rebind : [D0, D1, D2] -> [D0, D4, 42]
      llvm::DenseSet<KGEN::ParamDeclAttr> unknownParamDecls;
      // Avoid redeclaring multiple parameters. Example:
      //  rebind<() -> C> : [A, B] -> [C, C]
      // --->
      //   kgen.param.declare C = <A>
      //   rebind : [A, B] -> [C, C]
      llvm::SmallSetVector<KGEN::ParamDeclAttr, 4> declaredParams;
      for (auto [inputDimAttr, outputDimAttr] :
           llvm::zip(inputDimAttrs.getValues(), outputDimAttrs.getValues())) {
        if (auto outputDimParamRef =
                dyn_cast<KGEN::ParamDeclRefAttr>(outputDimAttr)) {
          if (KGEN::ParamDeclAttr paramDecl =
                  nameToParamDecl.lookup(outputDimParamRef.getName())) {
            if (declaredParams.insert(paramDecl)) {
              KGEN::ParamDeclareOp::create(rewriter, loc, paramDecl,
                                           inputDimAttr);
            }
          }
        }
      }
      if (unknownParamDecls.size() == paramDecls.size())
        return failure();
      rewriter.modifyOpInPlace(op, [&] {
        op.setOutputParamDecls(SmallVector<KGEN::ParamDeclAttr>(
            unknownParamDecls.begin(), unknownParamDecls.end()));
      });
      return success();
    }
  }
  // TODO: support materialization for other rebinds
  return failure();
}

} // namespace

namespace M::MO {

//===----------------------------------------------------------------------===//
// AllReduceSumOp
//===----------------------------------------------------------------------===//
LogicalResult AllReduceSumOp::fold(FoldAdaptor adaptor,
                                   SmallVectorImpl<OpFoldResult> &results) {
  if (getInputs().size() == 1 && getOutputs().size() == 1 &&
      getInputs()[0].getType() == getOutputs()[0].getType()) {
    results.push_back(getInputs()[0]);
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// BundledAllReduceSumOp
//===----------------------------------------------------------------------===//
LogicalResult
BundledAllReduceSumOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  if (getInputs().size() == 1 && getSignalBuffers().size() == 1 &&
      getInputs()[0].getType() == getOutput().getType()) {
    results.push_back(getInputs()[0]);
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// BundledBroadcastOp
//===----------------------------------------------------------------------===//
LogicalResult BundledBroadcastOp::fold(FoldAdaptor adaptor,
                                       SmallVectorImpl<OpFoldResult> &results) {
  if (getSignalBuffers().size() == 1 &&
      getInput().getType() == getOutput().getType()) {
    results.push_back(getInput());
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// BundledAllGatherOp
//===----------------------------------------------------------------------===//
LogicalResult BundledAllGatherOp::fold(FoldAdaptor adaptor,
                                       SmallVectorImpl<OpFoldResult> &results) {
  if (getInputs().size() == 1 && getSignalBuffers().size() == 1 &&
      getInputs()[0].getType() == getOutputs()[0].getType()) {
    results.push_back(getInputs()[0]);
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// BundledExpandOp
//===----------------------------------------------------------------------===//
LogicalResult BundledExpandOp::fold(FoldAdaptor adaptor,
                                    SmallVectorImpl<OpFoldResult> &results) {
  // In a single-launch parent parallel op, the expand is an identity.
  if (getNumResults() == 1 && getResult(0).getType() == getInput().getType()) {
    results.push_back(getInput());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ReduceScatterSumOp
//===----------------------------------------------------------------------===//
LogicalResult ReduceScatterSumOp::fold(FoldAdaptor adaptor,
                                       SmallVectorImpl<OpFoldResult> &results) {
  if (getInputs().size() == 1 && getOutputs().size() == 1 &&
      getInputs()[0].getType() == getOutputs()[0].getType()) {
    results.push_back(getInputs()[0]);
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//
LogicalResult AllGatherOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {
  // The verifier ensures that with 1 input, there is exactly 1 output.
  if (getInputs().size() == 1 &&
      getInputs()[0].getType() == getOutputs()[0].getType()) {
    results.push_back(getInputs()[0]);
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// DistributedScatterOp
//===----------------------------------------------------------------------===//
LogicalResult
DistributedScatterOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  if (getInputs().size() == 1 && getOutputs().size() == 1 &&
      getInputs()[0].getType() == getOutputs()[0].getType()) {
    results.push_back(getInputs()[0]);
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// DistributedBroadcastOp
//===----------------------------------------------------------------------===//
LogicalResult
DistributedBroadcastOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<OpFoldResult> &results) {
  if (getOutputs().size() == 1 &&
      getInput().getType() == getOutputs()[0].getType()) {
    results.push_back(getInput());
    results.push_back(getInChain());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//
OpFoldResult AssertOp::fold(FoldAdaptor adaptor) {
  // A false condition is left in place: a pass can report it, or it becomes a
  // guaranteed runtime failure.
  TypedAttr condAttr = dyn_cast_if_present<TypedAttr>(adaptor.getCond());
  if (!condAttr)
    return {};

  if (KGEN::isTriviallyTrueProposition(condAttr))
    return ChainAttr::get(ChainType::get(getContext()));

  return {};
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//
LogicalResult ArgMaxOp::canonicalize(ArgMaxOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// ArgMinOp
//===----------------------------------------------------------------------===//
LogicalResult ArgMinOp::canonicalize(ArgMinOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//
OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  TensorType resTy = getResult().getType();
  TensorType inTy = getInput().getType();
  if (inTy == resTy)
    return getInput();

  // TODO(#10811): support other types and do this using a MOJO kernel? Waiting
  // on #23197
  Attribute inputAttr = adaptor.getInput();
  if (!inputAttr)
    return {};

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  auto resElemTy = rankedResTy.getElementType();
  if (auto intElemsAttr =
          dyn_cast_if_present<IntArrayElementsAttr>(inputAttr)) {
    if (isa<IntegerType>(resElemTy)) {
      // This is inefficient, but simple.
      SmallVector<APInt> res = llvm::map_to_vector(
          intElemsAttr.getValues(), [&](const llvm::APInt &dim) {
            if (inTy.getDtype().isSInt())
              return APInt(64, dim.getSExtValue(), /*isSigned=*/true);
            return APInt(64, dim.getZExtValue(), /*isSigned=*/false);
          });

      return IntArrayElementsAttr::get(rankedResTy, res);
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//
LogicalResult BroadcastToOp::canonicalize(BroadcastToOp op,
                                          PatternRewriter &rewriter) {
  TensorValue inputVal = op.getInput();

  if (op.getOutputParamDecls().empty()) {
    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(op, op.getType(),
                                                       inputVal);
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//
OpFoldResult ChainCreateOp::fold(FoldAdaptor adaptor) {
  if (getInputs().empty())
    return nullptr;

  // Forward the input if its just a single chain value.
  if (getNumOperands() == 1 && isa<ChainType>(getOperand(0).getType()))
    return getOperand(0);

  llvm::SmallSetVector<Value, 4> keptValues;

  for (Value val : getInputs()) {
    auto chainDefOp = val.getDefiningOp<MOSH::ParamToValueOp>();

    if (chainDefOp && isa<MO::ChainAttr>(chainDefOp.getValue()))
      continue;

    keptValues.insert(val);
  }

  // All inputs were constant chains:
  //   mosh.param.to_value: !mo.chain = <#mo.chain>
  if (keptValues.empty())
    return getInputs().front();

  if (keptValues.size() == getInputs().size())
    return nullptr;

  getOperation()->setOperands(keptValues.getArrayRef());

  // De-duplication may have left a single chain operand.
  if (getNumOperands() == 1 && isa<ChainType>(getOperand(0).getType()))
    return getOperand(0);

  return nullptr;
}

namespace {
struct FlattenChainCreate final : public OpRewritePattern<ChainCreateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ChainCreateOp op,
                                PatternRewriter &rewriter) const override {
    if (llvm::none_of(op.getOperands(), [](Value val) {
          return val.getDefiningOp<ChainCreateOp>();
        }))
      return failure();

    llvm::SmallSetVector<Value, 4> operands;
    for (Value operand : op.getOperands()) {
      if (auto innerChainCreate = operand.getDefiningOp<ChainCreateOp>()) {
        // Ignore empty chain.create operands
        if (innerChainCreate.getNumOperands() == 0)
          continue;

        // Unfold single-use chain creates into the consumer
        if (innerChainCreate.getResult().getNumUses() == 1) {
          operands.insert_range(innerChainCreate.getOperands());
          continue;
        }
      }

      operands.insert(operand);
    }

    llvm::SmallSetVector<Value, 4> oldOperands(llvm::from_range,
                                               op.getOperands());
    if (operands == oldOperands)
      return failure();

    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(operands.getArrayRef()); });
    return success();
  }
};
} // namespace

void ChainCreateOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<FlattenChainCreate>(ctx);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//
OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  TensorType outType = getResult().getType();

  int64_t axis = getAxisAttr().getInt();
  if (outType.hasRank())
    axis = normalizeAxis(axis, outType.getRank());

  SplitOp split = nullptr;

  // If the concat is the sole product of a split along the same axis we can
  // fold and just use the split input.
  size_t operandIndex = 0;
  for (Value operand : getInputs()) {
    auto definingOp = operand.getDefiningOp<MO::SplitOp>();
    if (!definingOp) {
      split = nullptr;
      break;
    }

    // All operands must have the same defining op.
    if (!split) {
      split = definingOp;
    } else if (split != definingOp) {
      split = nullptr;
      break;
    }

    // Split must have same axis.
    int64_t splitAxis = definingOp.getAxisAttr().getInt();
    if (outType.hasRank())
      splitAxis = normalizeAxis(splitAxis, outType.getRank());
    if (splitAxis != axis) {
      split = nullptr;
      break;
    }

    if (cast<OpResult>(operand).getResultNumber() != operandIndex) {
      split = nullptr;
      break;
    }

    // Each split operand must be in the same order as they appear here.
    ++operandIndex;
  }

  if (split && split.getInput().getType() == outType &&
      split.getNumResults() == getInputs().size())
    return split.getInput();

  // Rank 1 only, so axis and strides don't matter.
  if (!outType.hasRank() || outType.getRank() != 1 || !outType.hasStaticShape())
    return {};

  SmallVector<APInt> concatTensor;

  DType outDtype = outType.getDtype();
  for (Attribute attr : adaptor.getInputs()) {
    auto asIntAttr = dyn_cast_if_present<IntArrayElementsAttr>(attr);
    if (!asIntAttr)
      return {};

    auto maybeValue =
        tryGetConstantBlob<int64_t>(cast<ElementsAttr>(asIntAttr), outDtype);
    if (!maybeValue)
      return {};

    for (size_t idx = 0, e = maybeValue->size(); idx < e; ++idx)
      concatTensor.push_back(APInt(64, (*maybeValue)[idx], true));
  }

  auto rankedResTy =
      RankedTensorType::get(outType.getIntDims(), outType.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, concatTensor);
}

LogicalResult ConcatOp::canonicalize(ConcatOp op, PatternRewriter &rewriter) {
  Operation::operand_range inputs = op.getInputs();
  SmallVector<Value> nonEmptyInputs;
  if (!op.getOutputParamDecls().empty())
    return failure();
  for (Value arg : inputs) {
    TensorType argTy = cast<TensorType>(arg.getType());
    auto isNonZero = [](int64_t dim) { return dim != 0; };
    if (!argTy.hasRank() || llvm::all_of(argTy.getIntDims(), isNonZero))
      nonEmptyInputs.emplace_back(arg);
  }
  // TODO: implement this
  if (nonEmptyInputs.empty())
    return failure();
  if (nonEmptyInputs.size() == 1) {
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(),
                                          nonEmptyInputs.front());
    return success();
  }
  if (nonEmptyInputs.size() != inputs.size()) {
    op.getInputsMutable().assign(nonEmptyInputs);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

//===----------------------------------------------------------------------===//
// ConstantExternalOp
//===----------------------------------------------------------------------===//
LogicalResult ConstantExternalOp::canonicalize(ConstantExternalOp op,
                                               PatternRewriter &rewriter) {
  // mo.constant.external cannot be ConstantLike, so the
  // GreedyPatternRewriteDriver won't hoist it for us.
  Region *insertRegion = getInsertionRegion(op->getBlock());
  Block *insertBlock = &insertRegion->front();
  if (insertBlock == op->getBlock())
    return failure();
  rewriter.moveOpBefore(op, insertBlock, insertBlock->begin());
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantScalarOp
//===----------------------------------------------------------------------===//
OpFoldResult ConstantScalarOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//
LogicalResult ConvOp::canonicalize(ConvOp op, PatternRewriter &rewriter) {
  StringRef filterLayoutStr = "";
  LayoutAttr filterLayout = op.getFilter().getType().getLayoutAttr();
  if (filterLayout)
    filterLayoutStr = filterLayout.getFormat();
  if (op.getFilterLayout() == filterLayoutStr)
    return failure();
  rewriter.modifyOpInPlace(op, [&]() { op.setFilterLayout(filterLayoutStr); });
  return success();
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//
LogicalResult ConvTransposeOp::canonicalize(ConvTransposeOp op,
                                            PatternRewriter &rewriter) {
  StringRef filterLayoutStr = "";
  LayoutAttr filterLayout = op.getFilter().getType().getLayoutAttr();
  if (filterLayout)
    filterLayoutStr = filterLayout.getFormat();
  if (op.getFilterLayout() == filterLayoutStr)
    return failure();
  rewriter.modifyOpInPlace(op, [&]() { op.setFilterLayout(filterLayoutStr); });
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

static ArrayElementsAttr getEmptyArray(Type elementType) {
  return ArrayElementsAttr::get({}, RankedTensorType::get({0}, elementType));
}

OpFoldResult GatherOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  if (!getOutputParamDecls().empty())
    return {};

  auto inputElems = dyn_cast_if_present<ElementsAttr>(adaptor.getInput());
  auto idxElems = dyn_cast_if_present<ElementsAttr>(adaptor.getIndices());

  // The verifier should ensure that indices is empty, but we don't check it.
  if (inputElems && inputElems.empty())
    return getEmptyArray(inputElems.getElementType());

  if (idxElems && idxElems.empty())
    return getEmptyArray(getResult().getType().getElementType());

  if (!inputElems || !idxElems)
    return {};

  TensorType resTy = getResult().getType();
  TensorType inTy = getInput().getType();

  if (!resTy.hasRank() || !inTy.hasRank())
    return {};

  if (resTy.getRank() != inTy.getRank() || inTy.getRank() != 1 ||
      !resTy.hasStaticShape() || !inTy.hasStaticShape())
    return {};

  DType dataTy = getInput().getType().getDtype();
  DType indicesTy = getIndices().getType().getDtype();

  // Ints only for now. Rank 1 means the axis is implied.
  auto dataAttr = dyn_cast<IntArrayElementsAttr>(inputElems);
  auto iAttr = dyn_cast<IntArrayElementsAttr>(idxElems);
  if (!dataAttr || !iAttr)
    return {};

  auto maybeData =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(dataAttr), dataTy);
  auto maybeInd =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(iAttr), indicesTy);
  if (!maybeData.has_value() || !maybeInd.has_value())
    return {};

  ArrayRef<int64_t> data = *maybeData;
  ArrayRef<int64_t> indices = *maybeInd;

  SmallVector<APInt> gatheredValues(indices.size());
  for (size_t idx = 0, e = indices.size(); idx < e; ++idx) {
    int64_t index = indices[idx];
    if (index < 0 || static_cast<size_t>(index) > data.size())
      return {};

    int64_t val = data[index];
    gatheredValues[idx] = APInt(64, val, /*isSigned=*/true);
  }

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, gatheredValues);
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

namespace {

/// Removes duplicate guard operands:
///   %r0, %r1, %r2, %r3 = mo.guard[%ch](%a, %b, %a, %b)
/// Into:
///   %f0, %f1 = mo.guard[%ch](%a, %b)
struct DedupeGuardOperands : public OpRewritePattern<GuardOp> {
  using OpRewritePattern<GuardOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GuardOp guard,
                                PatternRewriter &rewriter) const override {
    // Map each unique operand to the index of its first occurrence.
    llvm::MapVector<Value, size_t> uniqueOperands;
    for (auto [index, operand] : llvm::enumerate(guard.getInputs()))
      uniqueOperands.try_emplace(operand, index);
    if (uniqueOperands.size() == guard.getInputs().size())
      return rewriter.notifyMatchFailure(guard, "no duplicated operands");
    auto chain = guard.getChain();
    auto newOperands = llvm::to_vector(llvm::make_first_range(uniqueOperands));
    auto newGuard =
        GuardOp::create(rewriter, guard.getLoc(), chain, newOperands);
    SmallVector<Value> newResults;
    newResults.reserve(guard.getNumResults());
    for (Value operand : guard.getInputs()) {
      size_t index = uniqueOperands[operand];
      newResults.push_back(newGuard.getResults()[index]);
    }
    rewriter.replaceOp(guard, newResults);
    return success();
  }
};

} // namespace

LogicalResult GuardOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  auto chainDefOp = getChain().getDefiningOp<MOSH::ParamToValueOp>();
  if (!chainDefOp)
    return failure();
  bool isTrivialChain = isa<MO::ChainAttr>(chainDefOp.getValue());
  if (!isTrivialChain)
    return failure();
  results.insert(results.begin(), getInputs().begin(), getInputs().end());
  return success();
}

void GuardOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<DedupeGuardOperands>(ctx);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//
namespace {
struct ConstantFoldIf : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  static std::optional<bool> getConstantCondition(IfOp op) {
    if (auto tensor = dyn_cast<TensorValue>(op.getCond()))
      if (auto value = tryGetConstantBool(tensor))
        return value;

    BoolAttr attr;
    if (matchPattern(op.getCond(), m_Constant(&attr)))
      return attr.getValue();

    return {};
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    auto condValue = getConstantCondition(op);
    if (!condValue)
      return rewriter.notifyMatchFailure(op, "non-constant condition");

    Region &region = *condValue ? op.getThenRegion() : op.getElseRegion();
    auto yieldOp = cast<YieldOp>(region.front().getTerminator());

    // Capture output params and yield params before inlining.
    Location loc = op.getLoc();
    SmallVector<KGEN::ParamDeclAttr> outputParams(op.getOutputParamDecls());
    SmallVector<TypedAttr> yieldParams(yieldOp.getParameters());

    replaceOpWithRegion(rewriter, op, region);

    // Alias output params to the inlined yield params (M = <A>, N = <B>) so
    // uses of M, N outside the IfOp still resolve.
    for (auto [outputParam, yieldParam] :
         llvm::zip_equal(outputParams, yieldParams)) {
      KGEN::ParamDeclareOp::create(rewriter, loc, outputParam, yieldParam);
    }

    return success();
  }
};

struct RemoveUnusedResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  static void transferBody(Block *source, Block *dest,
                           ArrayRef<OpResult> usedResults,
                           PatternRewriter &rewriter) {
    rewriter.mergeBlocks(source, dest);
    auto yieldOp = cast<YieldOp>(dest->getTerminator());

    auto usedOperands = llvm::map_to_vector(usedResults, [&](OpResult res) {
      return yieldOp.getOperand(res.getResultNumber());
    });

    rewriter.modifyOpInPlace(yieldOp,
                             [&]() { yieldOp->setOperands(usedOperands); });
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    auto usedResults = llvm::filter_to_vector(
        op.getResults(), [](OpResult res) { return !res.use_empty(); });

    if (usedResults.size() == op.getNumResults())
      return failure();

    auto newTypes = llvm::map_to_vector(
        usedResults, [](OpResult res) { return res.getType(); });

    auto newOp = IfOp::create(rewriter, op.getLoc(), newTypes, op.getCond());
    rewriter.createBlock(&newOp.getThenRegion());
    rewriter.createBlock(&newOp.getElseRegion());

    // An IfOp with results always has both a then and an else region.
    transferBody(op.getBody(0), newOp.getBody(0), usedResults, rewriter);
    transferBody(op.getBody(1), newOp.getBody(1), usedResults, rewriter);

    SmallVector<Value> repResults(op.getNumResults());
    for (const auto &en : llvm::enumerate(usedResults))
      repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
    rewriter.replaceOp(op, repResults);
    return success();
  }
};

/// Replace uses of the condition value with a true constant in the then branch
/// and false in the else branch.
struct ConditionPropagation : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  static Value createBoolOfType(PatternRewriter &rewriter, Location loc,
                                Type type, bool value) {
    if (auto tensorTy = dyn_cast<TensorType>(type))
      return ConstantOp::create(rewriter, loc, value, tensorTy.getDeviceRef());

    assert(isa<ScalarType>(type) && "must be a tensor or scalar type");
    return ConstantScalarOp::create(rewriter, loc, rewriter.getBoolAttr(value));
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    // Created lazily, at most once each.
    Value constantTrue = nullptr;
    Value constantFalse = nullptr;

    for (OpOperand &use : llvm::make_early_inc_range(op.getCond().getUses())) {
      auto useRegion = use.getOwner()->getParentRegion();

      if (op.getThenRegion().isAncestor(useRegion)) {
        changed = true;

        if (!constantTrue)
          constantTrue = createBoolOfType(rewriter, op.getLoc(),
                                          op.getCond().getType(), true);

        rewriter.modifyOpInPlace(use.getOwner(),
                                 [&]() { use.set(constantTrue); });
      } else if (op.getElseRegion().isAncestor(useRegion)) {
        changed = true;

        if (!constantFalse)
          constantFalse = createBoolOfType(rewriter, op.getLoc(),
                                           op.getCond().getType(), false);

        rewriter.modifyOpInPlace(use.getOwner(),
                                 [&]() { use.set(constantFalse); });
      }
    }

    return success(changed);
  }
};

} // end namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<ConstantFoldIf, ConditionPropagation, RemoveUnusedResults>(ctx);
}

//===----------------------------------------------------------------------===//
// IndexToTensorOp
//===----------------------------------------------------------------------===//
OpFoldResult IndexToTensorOp::fold(FoldAdaptor adaptor) {
  auto inputAttr = dyn_cast_if_present<KGEN::SIMDAttr>(adaptor.getInput());
  if (!inputAttr)
    return {};

  SmallVector<APInt> res{inputAttr.getValues().front().getIntVal()};
  auto rankedResTy = RankedTensorType::get({}, getType().getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
}

LogicalResult IndexToTensorOp::canonicalize(IndexToTensorOp op,
                                            PatternRewriter &rewriter) {
  auto paramToValueOp = op.getInput().getDefiningOp<MOSH::ParamToValueOp>();
  if (!paramToValueOp)
    return failure();

  std::optional<int64_t> inputInt =
      tryGetConstantInt(paramToValueOp.getValue());
  if (!inputInt)
    return failure();

  SmallVector<APInt> res{APInt(64, *inputInt, /*isSigned=*/true)};
  auto rankedResTy = RankedTensorType::get({}, op.getType().getElementType());
  auto elementsAttr =
      cast<ElementsAttr>(IntArrayElementsAttr::get(rankedResTy, res));
  rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), elementsAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// LayoutTransformOp
//===----------------------------------------------------------------------===//
LogicalResult LayoutTransformOp::canonicalize(LayoutTransformOp op,
                                              PatternRewriter &rewriter) {
  LayoutAttr inputLayout = op.getInput().getType().getLayoutAttr();
  LayoutAttr outputLayout = op.getResult().getType().getLayoutAttr();
  if (inputLayout == outputLayout) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
LogicalResult MatmulOp::canonicalize(MatmulOp op, PatternRewriter &rewriter) {
  bool transposeB = false;
  bool packedB = false;

  LayoutAttr weightsLayout = op.getInputB().getType().getLayoutAttr();
  if (weightsLayout) {
    transposeB = weightsLayout.getFormat() == "NK";
    packedB = weightsLayout.getFormat() == "KNkni";
  }

  if (transposeB == op.getTransposeB() && packedB == op.getPackedB())
    return failure();

  rewriter.modifyOpInPlace(op, [&]() {
    op.setTransposeB(transposeB);
    op.setPackedB(packedB);
  });

  return success();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//
LogicalResult MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) {
  if (isOne(op.getInputX())) {
    rewriter.replaceOp(op, op.getInputY());
    return success();
  } else if (isOne(op.getInputY())) {
    rewriter.replaceOp(op, op.getInputX());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MutableLoadOp
//===----------------------------------------------------------------------===//

// Collapses two sequential loads of the same buffer into the upstream one.
LogicalResult MutableLoadOp::canonicalize(MutableLoadOp loadOp,
                                          mlir::PatternRewriter &rewriter) {
  auto inChain = loadOp.getInChain();
  auto upstreamLoadOp =
      dyn_cast_if_present<MutableLoadOp>(inChain.getDefiningOp());

  if (!upstreamLoadOp) {
    return rewriter.notifyMatchFailure(
        loadOp.getLoc(), "Upstream op (on chain) is not a mo.mutable.load op.");
  }

  if (upstreamLoadOp.getInBuffer() != loadOp.getInBuffer()) {
    return rewriter.notifyMatchFailure(
        loadOp.getLoc(),
        "Upstream mo.mutable.load op does not operate on same buffer.");
  }

  if (upstreamLoadOp->getBlock() != loadOp->getBlock()) {
    return rewriter.notifyMatchFailure(
        loadOp.getLoc(),
        "Upstream mo.mutable.load op is in a different block.");
  }

  rewriter.replaceAllUsesWith(loadOp.getOutChain(),
                              upstreamLoadOp.getOutChain());
  rewriter.replaceAllUsesWith(loadOp.getOutTensor(),
                              upstreamLoadOp.getOutTensor());

  // Canonicalization DCE erases loadOp.
  return success();
}

//===----------------------------------------------------------------------===//
// TensorBundleOp
//===----------------------------------------------------------------------===//

/// Fold `bundle(unbundle(%X)#0, ..., unbundle(%X)#N-1)` back to `%X`, the
/// inverse of the `unbundle(bundle(...))` fold, so downstream passes see SSA
/// equality across a transient unbundle/re-bundle.
OpFoldResult TensorBundleOp::fold(FoldAdaptor adaptor) {
  auto inputs = getInputs();
  if (inputs.empty())
    return {};

  auto firstOperand = dyn_cast<OpResult>(inputs.front());
  if (!firstOperand)
    return {};
  auto unbundleOp = dyn_cast<TensorUnbundleOp>(firstOperand.getOwner());
  if (!unbundleOp)
    return {};

  // Inputs must be all N results of one unbundle, in order.
  if (unbundleOp.getNumResults() != inputs.size())
    return {};
  for (auto [i, input] : llvm::enumerate(inputs)) {
    auto result = dyn_cast<OpResult>(input);
    if (!result || result.getOwner() != unbundleOp.getOperation() ||
        result.getResultNumber() != i)
      return {};
  }

  // Implied by the above; checked defensively.
  if (unbundleOp.getInput().getType() != getResult().getType())
    return {};

  return unbundleOp.getInput();
}

//===----------------------------------------------------------------------===//
// TensorUnbundleOp
//===----------------------------------------------------------------------===//

LogicalResult TensorUnbundleOp::fold(FoldAdaptor adaptor,
                                     SmallVectorImpl<OpFoldResult> &results) {
  if (auto bundleOp = getInput().getDefiningOp<TensorBundleOp>()) {
    results.append(bundleOp.getInputs().begin(), bundleOp.getInputs().end());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//
LogicalResult ParallelOp::fold(FoldAdaptor adaptor,
                               SmallVectorImpl<OpFoldResult> &results) {
  // Erasing the parallel would drop body work that uses the buffers or chain.
  Operation *thisOp = getOperation();
  auto hasUserInBody = [thisOp](Value v) {
    for (Operation *user : v.getUsers()) {
      if (user == thisOp)
        continue;
      for (Operation *p = user; p; p = p->getParentOp()) {
        if (p == thisOp)
          return true;
      }
    }
    return false;
  };
  for (Value buf : getBuffers())
    if (hasUserInBody(buf))
      return failure();
  if (hasChain() && hasUserInBody(getInChain()))
    return failure();

  Block &body = getBodyRegion().front();

  // No-op iff the body is only a yield forwarding its block arguments.
  if (!body.without_terminator().empty())
    return failure();

  auto yieldOp = cast<YieldOp>(body.getTerminator());
  for (size_t yieldIdx = 0; yieldIdx < yieldOp.getNumOperands(); ++yieldIdx) {
    if (yieldIdx >= body.getNumArguments() ||
        yieldOp.getOperand(yieldIdx) != body.getArgument(yieldIdx))
      return failure();
  }

  results = llvm::to_vector_of<OpFoldResult>(getInputs());
  if (hasChain())
    results.push_back(getInChain());
  return success();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//
OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?

  auto startAttr =
      dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getStart());
  auto limitAttr =
      dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getLimit());
  auto stepAttr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getStep());
  if (!startAttr || !limitAttr || !stepAttr)
    return {};

  int64_t start = getIntBlob(startAttr).front();
  int64_t limit = getIntBlob(limitAttr).front();
  int64_t step = getIntBlob(stepAttr).front();

  auto numResElementsOr = getRangeNumElements(start, limit, step);
  if (numResElementsOr.isError())
    return {};
  int64_t numResElements = *numResElementsOr;
  SmallVector<APInt> res((size_t)numResElements);
  for (size_t idx = 0; idx < (size_t)numResElements; ++idx) {
    res[idx] = APInt(64, start, /*isSigned=*/true);
    start += step;
  }

  auto attrTy =
      RankedTensorType::get({numResElements}, startAttr.getElementType());
  auto foldedAttr = IntArrayElementsAttr::get(attrTy, res);
  TensorType outTy = getType();
  if (outTy.hasStaticShape())
    return foldedAttr;

  Location loc = getLoc();
  OpBuilder builder(*this);
  auto resTy = outTy.cloneWith({numResElements}, outTy.getElementType());
  auto constantOp =
      ConstantOp::create(builder, loc, resTy, cast<ElementsAttr>(foldedAttr));
  auto rebindOp =
      RebindOp::create(builder, loc, outTy, constantOp, getOutputParamDecls());
  return rebindOp.getResult();
}

//===----------------------------------------------------------------------===//
// ReduceMulOp
//===----------------------------------------------------------------------===//
LogicalResult ReduceMulOp::canonicalize(ReduceMulOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//
LogicalResult RebindOp::canonicalize(RebindOp op,
                                     PatternRewriter &origRewriter) {
  DeviceAwareRewriter rewriter(op, origRewriter);
  if (op.getInput().getType() == op.getType()) {
    assert(op.getOutputParamDecls().empty());
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  LogicalResult paramsResolved = materializeShapeInference(op, origRewriter);
  // Likely already processed, or doesn't help shape inference.
  if (op.getInput().getDefiningOp<RebindOp>())
    return success(succeeded(paramsResolved));
  return success(succeeded(paramsResolved));
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto inputArr = dyn_cast_if_present<ArrayElementsAttr>(adaptor.getInput());
  auto shapeArr =
      dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getNewShape());
  auto outputShapeAttr = getType().getShape();
  if (!inputArr || !shapeArr || !outputShapeAttr)
    return {};

  int64_t shapeNumElems = 1;
  for (llvm::APInt dim : shapeArr.getValues())
    shapeNumElems *= dim.getSExtValue();

  int64_t outputNumElems = 1;
  SmallVector<int64_t> outIntDims = outputShapeAttr.getIntDims();
  for (int64_t intDim : outIntDims) {
    if (mlir::ShapedType::isDynamic(intDim))
      return {};
    outputNumElems *= intDim;
  }

  if (shapeNumElems == outputNumElems)
    return foldReshape(outIntDims, inputArr);

  return {};
}

LogicalResult ReshapeOp::canonicalize(ReshapeOp op, PatternRewriter &rewriter) {
  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();

  TensorType resTy = op.getType();
  if (!inputTy.hasRank() || !resTy.hasRank())
    return failure();

  // TODO(#11269) Emit runtime checks.
  // [] --> [?]
  if (inputTy.getRank() == 0 && resTy.getRank() == 1) {
    auto inferredResTy =
        resTy.cloneWith(SmallVector<int64_t>({1}), resTy.getElementType());
    auto newReshapeOp = Static::ReshapeOp::create(rewriter, op.getLoc(),
                                                  inferredResTy, inputVal);
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newReshapeOp,
                                          op.getOutputParamDecls());
    return success();
  }

  // [?] --> []
  if (inputTy.getRank() == 1 && resTy.getRank() == 0) {
    auto inferredResTy =
        resTy.cloneWith(SmallVector<int64_t>({}), resTy.getElementType());
    auto newReshapeOp = Static::ReshapeOp::create(rewriter, op.getLoc(),
                                                  inferredResTy, inputVal);
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newReshapeOp,
                                          op.getOutputParamDecls());
    return success();
  }

  if (!op.getOutputParamDecls().empty())
    return failure();

  rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, resTy, inputVal);
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeOf
//===----------------------------------------------------------------------===//
LogicalResult ShapeOfOp::canonicalize(ShapeOfOp op, PatternRewriter &rewriter) {
  TensorType inputTy = op.getInput().getType();
  TensorType resTy = op.getResult().getType();
  if (!inputTy.hasRank())
    return failure();

  auto moshVal =
      MOSH::ParamToValueOp::create(rewriter, op.getLoc(), inputTy.getShape());
  rewriter.replaceOpWithNewOp<MO::Shape::ToTensorOp>(op, resTy, moshVal);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape::FromTensorOp
//===----------------------------------------------------------------------===//
LogicalResult Shape::FromTensorOp::canonicalize(Shape::FromTensorOp op,
                                                PatternRewriter &rewriter) {
  // mo.shape.from_tensor(mo.shape.to_tensor(x)) == x
  auto input = op.getInput();
  if (auto toTensor = input.getDefiningOp<Shape::ToTensorOp>()) {
    if (toTensor.getInput().getType() == op.getType()) {
      rewriter.replaceOp(op, toTensor.getInput());
      return success();
    }
  }

  // fold mo.shape.from_tensor(<constant>)
  IntArrayElementsAttr newShape;
  if (mlir::matchPattern(input, mlir::m_Constant(&newShape))) {
    auto shape = ShapeAttr::get(newShape, op.getType());
    rewriter.replaceOpWithNewOp<MOSH::ParamToValueOp>(op, shape);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Shape::ToTensorOp
//===----------------------------------------------------------------------===//
LogicalResult Shape::ToTensorOp::canonicalize(Shape::ToTensorOp op,
                                              PatternRewriter &rewriter) {
  auto input = op.getInput();
  if (auto paramToValueOp = input.getDefiningOp<MOSH::ParamToValueOp>()) {
    auto inputShapeAttr = dyn_cast<MOSH::ShapeAttr>(paramToValueOp.getValue());
    if (!inputShapeAttr)
      return failure();

    TensorType resTy = op.getType();
    if (!resTy.hasStaticShape()) {
      TensorType newTy = resTy.cloneWith(ShapeAttr::get(
          {(int64_t)inputShapeAttr.getRank()}, rewriter.getType<ShapeType>()));
      Location loc = op.getLoc();
      auto newOp =
          Shape::ToTensorOp::create(rewriter, loc, newTy, paramToValueOp);
      rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newOp);
      return success();
    }

    if (inputShapeAttr.isConstant()) {
      auto outTy =
          RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
      SmallVector<APInt> res =
          llvm::map_to_vector(inputShapeAttr.getIntDims(), [](int64_t dim) {
            return APInt(64, dim, /*isSigned=*/true);
          });
      auto elementsAttr =
          cast<ElementsAttr>(IntArrayElementsAttr::get(outTy, res));
      rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), elementsAttr);
      return success();
    }
  }

  // mo.shape.to_tensor(mo.shape.from_tensor(x)) => x
  if (auto fromTensor = input.getDefiningOp<Shape::FromTensorOp>()) {
    if (fromTensor.getInput().getType() == op.getType()) {
      rewriter.replaceOp(op, fromTensor.getInput());
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

namespace {
/// Moves ops that read only constants and weights out of a side-stream body,
/// so exec-invariant code motion can lift them to init.
struct HoistConstantDerivedOps : public OpRewritePattern<SequenceOp> {
  using OpRewritePattern<SequenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SequenceOp seq,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    Block &body = seq.getBodyRegion().front();
    for (Operation &op :
         llvm::make_early_inc_range(body.without_terminator())) {
      // An op with regions may read block arguments from inside them, which
      // the producer walk does not see.
      if (op.getNumRegions() == 0 &&
          dataflowGraphFoldsToConstants<ConstantLike, ChainCreateOp>(&op)) {
        rewriter.moveOpBefore(&op, seq);
        changed = true;
      }
    }
    return success(changed);
  }
};
} // namespace

void SequenceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {
  patterns.add<HoistConstantDerivedOps>(ctx);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//
OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  return foldSliceHelper(*this, adaptor);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//
OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  // TODO refactor parts of this with other folders

  if (getX() == getY())
    return getX();

  auto condAttr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getCond());
  auto xAttr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getX());
  auto yAttr = dyn_cast_if_present<IntArrayElementsAttr>(adaptor.getY());
  if (!condAttr)
    return {};

  DType condDType = getCond().getType().getDtype();

  // With x or y unknown, a uniform condition still selects one of them.
  if (!xAttr || !yAttr) {
    auto maybeCond =
        tryGetConstantBlob<int64_t>(cast<ElementsAttr>(condAttr), condDType);
    if (!maybeCond.has_value())
      return {};

    ArrayRef<int64_t> condArr = *maybeCond;

    int64_t value = condArr[0];
    for (size_t idx = 1, e = condAttr.size(); idx < e; ++idx) {
      if (value != condArr[idx])
        return {};
    }
    return value ? getX() : getY();
  }

  TensorType resTy = getType();
  DType resDType = resTy.getDtype();
  auto maybeCond =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(condAttr), condDType);
  auto maybeX =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(xAttr), resDType);
  auto maybeY =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(yAttr), resDType);
  if (!maybeX.has_value() || !maybeY.has_value())
    return {};

  ArrayRef<int64_t> condArr = *maybeCond;
  ArrayRef<int64_t> xArr = *maybeX;
  ArrayRef<int64_t> yArr = *maybeY;
  SmallVector<APInt> res(condAttr.size());
  for (size_t idx = 0, e = condAttr.size(); idx < e; ++idx) {
    int64_t resVal = condArr[idx] ? xArr[idx] : yArr[idx];
    res[idx] = APInt(64, resVal, /*isSigned=*/true);
  }

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//
LogicalResult SplitOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  if (getNumResults() == 1 && getInput().getType() == getResult(0).getType()) {
    results.push_back(getInput());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SqueezeShapeOp
//===----------------------------------------------------------------------===//
LogicalResult SqueezeShapeOp::canonicalize(SqueezeShapeOp op,
                                           PatternRewriter &rewriter) {
  // Normalize negative indices and make result in sorted order
  TensorValue indicesValue = op.getRemoveIndices();
  TensorType inputShapeType = op.getInputShape().getType();
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(indicesValue);
  if (!maybeIndices || !inputShapeType.getNumElements().has_value())
    return failure();

  SmallVector<int64_t> &indices = maybeIndices.value();
  int64_t inputRank = inputShapeType.getNumElements().value();
  bool changedIndices = false;
  for (int64_t &index : indices) {
    // Leave an out-of-range index for the verifier to reject; normalizing it
    // would land on a valid axis after enough rewrites.
    if (index < -inputRank || index >= inputRank)
      return failure();
    if (index < 0) {
      index += inputRank;
      changedIndices = true;
    }
  }

  if (!llvm::is_sorted(indices)) {
    llvm::sort(indices);
    changedIndices = true;
  }

  if (!changedIndices)
    return failure();

  // Keep the original indices dtype.
  TensorValue newIndices = ConstantOp::create(rewriter, op->getLoc(), indices,
                                              inputShapeType.getDeviceRef());
  newIndices = CastOp::create(rewriter, op->getLoc(), newIndices,
                              indicesValue.getType().getDtype());
  rewriter.replaceOpWithNewOp<SqueezeShapeOp>(op, op.getResult().getType(),
                                              op.getInputShape(), newIndices);
  return success();
}

//===----------------------------------------------------------------------===//
// Static::ReshapeOp
//===----------------------------------------------------------------------===//
OpFoldResult Static::ReshapeOp::fold(FoldAdaptor adaptor) {
  if (auto inputArr =
          dyn_cast_if_present<ArrayElementsAttr>(adaptor.getInput())) {
    auto outputShape = getType().getShape();
    if (outputShape && outputShape.isStatic())
      return foldReshape(outputShape.getIntDims(), inputArr);
  }

  if (getInput().getType() == getType())
    return getInput();

  return {};
}

LogicalResult Static::ReshapeOp::canonicalize(Static::ReshapeOp op,
                                              PatternRewriter &rewriter) {
  if (auto precedingOp = op.getInput().getDefiningOp<Static::ReshapeOp>()) {
    rewriter.modifyOpInPlace(op,
                             [&] { op.setOperand(precedingOp.getInput()); });
    return success();
  }

  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();
  TensorType resTy = op.getType();
  if (!resTy.hasRank() || !inputTy.hasRank())
    return failure();

  // Fold a preceding ranked -> ranked rebind into this op.
  if (auto rebindOp = inputVal.getDefiningOp<RebindOp>()) {
    TensorValue rebIn = rebindOp.getInput();
    TensorType rebInTy = rebIn.getType();
    if (rebInTy.hasRank()) {
      rewriter.modifyOpInPlace(op, [&] { op.setOperand(rebIn); });
      return success();
    }
  }

  size_t inputRank = inputTy.getRank();
  size_t resRank = resTy.getRank();
  if (inputRank == 0 && resRank == 1) {
    // Scalar -> singleton.
    if (TypedAttr param = getParamAttrThrough<IndexToTensorOp>(inputVal)) {
      auto shape = ShapeAttr::get(SmallVector<TypedAttr>{param},
                                  rewriter.getType<ShapeType>());
      auto newConstOp =
          MOSH::ParamToValueOp::create(rewriter, op->getLoc(), shape);
      rewriter.replaceOpWithNewOp<Shape::ToTensorOp>(op, resTy, newConstOp);
      return success();
    }
  }
  if (inputRank == 1 && resRank == 0) {
    // Singleton -> scalar.
    if (TypedAttr param = getParamAttrThrough<Shape::ToTensorOp>(inputVal)) {
      if (auto shape = dyn_cast<ShapeAttr>(param);
          shape && shape.getRank() == 1) {
        auto newConstOp =
            MOSH::ParamToValueOp::create(rewriter, op->getLoc(), shape[0]);
        rewriter.replaceOpWithNewOp<IndexToTensorOp>(op, resTy, newConstOp);
        return success();
      }
    }
  }

  if (inputRank != resRank)
    return failure();

  // A reshape that changes only a single dimension is just a rebind.
  bool foundDifferentDim = false;
  for (auto [inDim, resDim] : llvm::zip(inputTy.getDims(), resTy.getDims())) {
    if (inDim != resDim) {
      if (foundDifferentDim)
        return failure();
      foundDifferentDim = true;
    }
  }

  rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, op.getInput());
  return success();
}

//===----------------------------------------------------------------------===//
// Static::BroadcastToOp
//===----------------------------------------------------------------------===//
LogicalResult Static::BroadcastToOp::canonicalize(Static::BroadcastToOp op,
                                                  PatternRewriter &rewriter) {
  TensorType resTy = op.getType();
  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();

  if (auto pastBroadcastOp =
          op.getInput().getDefiningOp<Static::BroadcastToOp>()) {
    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(
        op, op.getType(), pastBroadcastOp.getInput());
    return success();
  }

  if (resTy.getShape() == inputTy.getShape()) {
    // Be careful not to drop any non-shape info like metadata in TensorType.
    if (resTy == inputTy)
      rewriter.replaceAllUsesExcept(op, inputVal, op);
    else
      rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, inputVal);

    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank())
    return failure();

  // If a broadcast has an empty input, it is not equivalent to a reshape.
  auto resNumElems =
      MOSH::ShapeAttr::getStaticDim(resTy.getSymbolicNumElements());
  auto inputNumElems =
      MOSH::ShapeAttr::getStaticDim(inputTy.getSymbolicNumElements());
  if ((resNumElems && *resNumElems == 0) ||
      (inputNumElems && *inputNumElems == 0))
    return failure();

  // Same (symbolic) element count means it is a reshape.
  if (resTy.getSymbolicNumElements() == inputTy.getSymbolicNumElements()) {
    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, resTy, inputVal);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//
LogicalResult TileOp::canonicalize(TileOp op, PatternRewriter &rewriter) {
  TensorType inputTy = op.getInput().getType();
  if (!inputTy.hasRank())
    return failure();

  auto maybeRepeatsVec = tryGetConstantIntVector(op.getRepeats());
  if (!maybeRepeatsVec.has_value())
    return failure();
  ArrayRef<int64_t> repeatsVec(maybeRepeatsVec.value());

  // A tile is a broadcast when every repeated dimension has size 1. This
  // fires only with statically known input dimensions: the rewrite lowers
  // straight to a static broadcast, never through shape tensors, which
  // would mix the input's device with the host-resident repeats.
  for (auto [inDim, repeats] : llvm::zip(inputTy.getIntDims(), repeatsVec)) {
    if (mlir::ShapedType::isDynamic(inDim))
      return failure();
    if (repeats < 1 || (repeats > 1 && inDim != 1))
      return failure();
  }

  auto bcast = Static::BroadcastToOp::create(rewriter, op.getLoc(),
                                             op.getType(), op.getInput());
  // A static broadcast preserves the shape, so the result needs a rebind to
  // carry the tile's declared output parameters through.
  rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), bcast.getResult(),
                                        op.getOutputParamDecls());
  return success();
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//
LogicalResult TransferOp::canonicalize(TransferOp op,
                                       PatternRewriter &rewriter) {
  // transfer(transfer(input)) = input, provided devices line up.
  TypedValue<TensorType> input = op.getInput();
  TypedValue<TensorType> output = op.getResult();
  if (auto prevOp = input.getDefiningOp<TransferOp>()) {
    TypedValue<TensorType> prevOpInput = prevOp.getInput();
    if (output.getType().getDeviceRef() ==
        prevOpInput.getType().getDeviceRef()) {
      rewriter.replaceOp(op, ValueRange{prevOpInput, prevOp.getInChain()});
      if (prevOp->use_empty()) {
        rewriter.eraseOp(prevOp);
      }
      return success();
    }
  }

  if (op.getAlwaysElideSameDeviceCopy() &&
      input.getType() == output.getType()) {
    // Replace uses of `op`'s outChain with its inChain to preserve ordering.
    rewriter.replaceOp(op, ValueRange{input, op.getInChain()});
    return success();
  }

  DeviceRefAttr inputDevice = input.getType().getDeviceRef();
  // Rotate a narrowing cast after a host transfer to run on the host first, so
  // fewer bytes are transferred.
  if (output.hasOneUse() && inputDevice == getHostDevice(op)) {
    Operation *user = *output.getUsers().begin();

    if (auto cast = dyn_cast<MO::CastOp>(user)) {
      TypedValue<TensorType> castOut = cast.getResult();

      Type srcTy = output.getType().getElementType();
      Type dstTy = castOut.getType().getElementType();

      if (srcTy.isF32() && dstTy.isBF16()) {
        cast->setOperand(0, input);
        castOut.setType(castOut.getType().cloneWith(inputDevice));

        op->setOperand(0, castOut);
        output.setType(
            output.getType().cloneWith(castOut.getType().getDtype()));

        cast->moveBefore(op);
        castOut.replaceAllUsesExcept(output, op);
        return success();
      }
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
LogicalResult TransposeOp::canonicalize(TransposeOp op,
                                        PatternRewriter &rewriter) {
  if (!op.getType().hasRank())
    return failure();
  size_t opRank = op.getType().getRank();

  auto maybePermVec = tryGetConstantIntVector(op.getPerm());
  if (!maybePermVec.has_value())
    return failure();
  ArrayRef<int64_t> permVec(maybePermVec.value());

  auto identityPerm = llvm::iota_range<int64_t>(0, opRank, /*Inclusive=*/false);
  if (llvm::equal(permVec, identityPerm)) {
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), op.getInput(),
                                          op.getOutputParamDecls());
    return success();
  }

  // Fold transpose of a broadcast constant (single stored element). The
  // constant fold pass cannot run kernels on these: the backing buffer is
  // smaller than the shape implies.
  if (auto constantOp = op.getInput().getDefiningOp<ConstantOp>()) {
    if (auto arrayAttr =
            dyn_cast<ArrayElementsAttr>(constantOp.getValueAttr())) {
      if (arrayAttr.getType().getNumElements() == 1) {
        rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), arrayAttr);
        return success();
      }
    }
  }

  // NOTE The following (transposes merging or transpose-->reshape) assumes
  // the verifier checks against invalid permutations.

  if (auto inputTransposeOp = op.getInput().getDefiningOp<TransposeOp>()) {
    auto maybeInputPermVec =
        tryGetConstantIntVector(inputTransposeOp.getPerm());
    if (!maybeInputPermVec.has_value())
      return failure();
    ArrayRef<int64_t> inputPermVec(maybeInputPermVec.value());

    SmallVector<int64_t> mergedPerm = permute(inputPermVec, permVec);
    auto mergedPermVal =
        ConstantOp::create(rewriter, op.getLoc(), mergedPerm,
                           op.getPerm().getType().getDeviceRef());
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getType(), inputTransposeOp.getInput(), mergedPermVal,
        op.getOutputParamDecls());
    return success();
  }

  // Transpose == Reshape if the relative order of non-1 dims remains the same
  // e.g. transpose from shape [8, 1, 1, 5, 1, 1, 1, 50] to
  // - [8, 1, 1, 5, 1, 1, 50, 1]
  // - [1, 1, 1, 8, 5, 50, 1, 1]
  // - [8, 5, 50, 1, 1, 1, 1, 1]
  int64_t indexLastNonOneDim = -1; // should always be increasing
  MOSH::ShapeAttr inputShape = op.getInput().getType().getShape();
  bool meetsReshapeRewriteCriteria = inputShape != nullptr;
  for (int64_t inputDimIndex : permVec) {
    if (!meetsReshapeRewriteCriteria) {
      break;
    }
    int64_t dimInteger = inputShape.getIntDim(inputDimIndex);
    if (dimInteger == mlir::ShapedType::kDynamic) {
      // violates static reshape which only allows int dims
      meetsReshapeRewriteCriteria = false;
    } else if (dimInteger != 1) {
      if (inputDimIndex < indexLastNonOneDim)
        meetsReshapeRewriteCriteria = false;
      indexLastNonOneDim = inputDimIndex;
    }
  }
  if (meetsReshapeRewriteCriteria) {
    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, op.getType(),
                                                   op.getInput());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//
namespace {

struct RemoveDuplicateLoopCarriedValues : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  // Maps each duplicated operand index to its first occurrence.
  static llvm::DenseMap<unsigned, unsigned> duplicateMap(ValueRange inputs) {
    llvm::DenseMap<Value, unsigned> firstIndex;
    llvm::DenseMap<unsigned, unsigned> result;
    for (auto [index, value] : llvm::enumerate(inputs)) {
      auto [it, inserted] = firstIndex.try_emplace(value, index);
      if (!inserted)
        result[index] = it->second;
    }
    return result;
  }

  // Two values in the same block are duplicated if they are identical or are
  // block arguments for the known duplicate operand indices.
  static bool isDuplicate(Block *block, ValueRange values, unsigned index,
                          unsigned firstOccurrence) {
    auto x = values[index];
    auto y = values[firstOccurrence];

    if (x == y)
      return true;

    auto blockArgX = dyn_cast<BlockArgument>(x);
    auto blockArgY = dyn_cast<BlockArgument>(y);

    return blockArgX && blockArgY && blockArgX.getParentBlock() == block &&
           blockArgY.getParentBlock() == block &&
           blockArgX.getArgNumber() == index &&
           blockArgY.getArgNumber() == firstOccurrence;
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    WhileConditionOp condOp = op.getCondTerminator();
    YieldOp yieldOp = op.getBodyTerminator();

    ValueRange whileOpResults = op.getResults();

    Block::BlockArgListType condBlockArgs = op.getCondArguments();
    Block::BlockArgListType bodyBlockArgs = op.getBodyArguments();

    ValueRange condOpArgs = condOp.getArgs();
    ValueRange yieldOpArgs = yieldOp.getOperands();

    // Duplicate operands are candidates; each must also be duplicated on both
    // region terminators.
    auto duplicatedWhileOperands = duplicateMap(op.getOperands());
    if (duplicatedWhileOperands.empty())
      return rewriter.notifyMatchFailure(op, "no operands to remove");

    llvm::SmallVector<Value> operandsToKeep;
    llvm::SmallVector<Value> resultsToKeep;
    llvm::BitVector toErase(op.getNumOperands());

    for (unsigned index : llvm::seq(op.getNumOperands())) {
      auto it = duplicatedWhileOperands.find(index);
      if (it == duplicatedWhileOperands.end()) {
        operandsToKeep.push_back(op.getOperand(index));
        resultsToKeep.push_back(op.getResult(index));
        continue;
      }

      unsigned firstOccurrence = it->second;
      if (!isDuplicate(op.getCondBlock(), condOpArgs, index, firstOccurrence) ||
          !isDuplicate(op.getBodyBlock(), yieldOpArgs, index,
                       firstOccurrence)) {
        operandsToKeep.push_back(op.getOperand(index));
        resultsToKeep.push_back(op.getResult(index));
        continue;
      }

      toErase.set(index);
      rewriter.replaceAllUsesWith(condBlockArgs[index],
                                  condBlockArgs[firstOccurrence]);
      rewriter.replaceAllUsesWith(bodyBlockArgs[index],
                                  bodyBlockArgs[firstOccurrence]);
      rewriter.replaceAllUsesWith(whileOpResults[index],
                                  whileOpResults[firstOccurrence]);
    }

    if (toErase.none())
      return rewriter.notifyMatchFailure(op,
                                         "no duplicate loop carried values");

    // The terminators have no API to erase arguments, so rebuild them.
    SmallVector<Value> newCondArgs;
    SmallVector<Value> newYieldArgs;

    for (auto index : llvm::seq(op.getNumOperands())) {
      if (!toErase.test(index)) {
        newCondArgs.push_back(condOp.getArgs()[index]);
        newYieldArgs.push_back(yieldOpArgs[index]);
      }
    }

    op.getCondBlock()->eraseArguments(toErase);
    op.getBodyBlock()->eraseArguments(toErase);

    condOp.getArgsMutable().assign(newCondArgs);
    yieldOp.getOperandsMutable().assign(newYieldArgs);

    auto newWhileOp = WhileOp::create(
        rewriter, op.getLoc(), TypeRange(operandsToKeep), operandsToKeep);

    newWhileOp.getCondRegion().takeBody(op.getCondRegion());
    newWhileOp.getBodyRegion().takeBody(op.getBodyRegion());

    rewriter.replaceAllUsesWith(resultsToKeep, newWhileOp.getResults());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Find unused results of the while loop that are also unused in the
/// body block and remove them.
///
/// Results are produced by the `mo.while.condition` operation, which yields
/// values to the parent `mo.while` operation and to the body region.
struct RemoveLoopInvariantCarriedValues : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  /// Whether `val`, as the index'th terminator operand, is the init value or a
  /// trivial forward of the index'th block argument.
  static bool canSimplifyValue(Value val, Value initVal, Block *block,
                               unsigned index) {
    if (val == initVal)
      return true;

    auto blockArg = dyn_cast<BlockArgument>(val);
    if (blockArg && blockArg.getParentBlock() == block &&
        index == blockArg.getArgNumber())
      return true;

    return false;
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    Block::BlockArgListType condBlockArgs = op.getCondArguments();
    Block::BlockArgListType bodyBlockArgs = op.getBodyArguments();
    WhileConditionOp condOp = op.getCondTerminator();
    OperandRange condOpArgs = condOp.getArgs();
    YieldOp yieldOp = op.getBodyTerminator();
    ValueRange yieldOpArgs = yieldOp.getOperands();

    // Check first to avoid allocating the data structures below.
    bool canSimplify = false;
    for (const auto [index, initVal, yieldOpArg, condOpArg] :
         llvm::enumerate(op.getOperands(), yieldOpArgs, condOpArgs)) {

      canSimplify |=
          canSimplifyValue(yieldOpArg, initVal, yieldOp->getBlock(), index) &&
          canSimplifyValue(condOpArg, initVal, condOp->getBlock(), index);
    }

    if (!canSimplify)
      return rewriter.notifyMatchFailure(
          op, "no loop carried values can be simplified");

    SmallVector<Value> newInitArgs, newYieldOpArgs, newCondOpArgs;
    SmallVector<Location> newCondBlockArgLocs, newBodyBlockArgLocs;

    SmallVector<Value> blockInitValMap(condOpArgs.size());
    for (const auto [index, initVal, yieldOpArg, condOpArg] :
         llvm::enumerate(op.getOperands(), yieldOpArgs, condOpArgs)) {

      bool simplifyArg =
          canSimplifyValue(yieldOpArg, initVal, yieldOp->getBlock(), index) &&
          canSimplifyValue(condOpArg, initVal, condOp->getBlock(), index);

      if (simplifyArg) {
        blockInitValMap[index] = initVal;
        continue;
      }

      newInitArgs.emplace_back(initVal);
      newYieldOpArgs.emplace_back(yieldOpArg);
      newCondOpArgs.emplace_back(condOpArg);
      newCondBlockArgLocs.emplace_back(condBlockArgs[index].getLoc());
      newBodyBlockArgLocs.emplace_back(bodyBlockArgs[index].getLoc());
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, newYieldOpArgs);
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(condOp);
      rewriter.replaceOpWithNewOp<WhileConditionOp>(
          condOp, condOp.getCondition(), newCondOpArgs);
    }

    auto newWhile = WhileOp::create(rewriter, op.getLoc(),
                                    TypeRange(newInitArgs), newInitArgs);

    Block &newCondBlock = *rewriter.createBlock(
        &newWhile.getCondRegion(), /*insertPt*/ {},
        ValueRange(newYieldOpArgs).getTypes(), newCondBlockArgLocs);

    Block &newBodyBlock = *rewriter.createBlock(
        &newWhile.getBodyRegion(), /*insertPt*/ {},
        ValueRange(newYieldOpArgs).getTypes(), newBodyBlockArgLocs);

    Block &condBlock = *op.getCondBlock();
    Block &bodyBlock = *op.getBodyBlock();

    SmallVector<Value> newCondBlockArgs(condBlock.getNumArguments());
    SmallVector<Value> newBodyBlockArgs(condBlock.getNumArguments());
    SmallVector<Value> newResults(condBlock.getNumArguments());

    // A loop-invariant i-th argument is replaced by its init value; the rest
    // map in order to the j-th new block argument.
    for (unsigned i = 0, j = 0, n = condBlock.getNumArguments(); i < n; ++i) {
      if (blockInitValMap[i] != nullptr) {
        newCondBlockArgs[i] = blockInitValMap[i];
        newBodyBlockArgs[i] = blockInitValMap[i];
        newResults[i] = blockInitValMap[i];
      } else {
        newCondBlockArgs[i] = newCondBlock.getArgument(j);
        newBodyBlockArgs[i] = newBodyBlock.getArgument(j);
        newResults[i] = newWhile.getResult(j);
        ++j;
      }
    }

    rewriter.mergeBlocks(&condBlock, &newCondBlock, newCondBlockArgs);
    rewriter.mergeBlocks(&bodyBlock, &newBodyBlock, newBodyBlockArgs);

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

} // end namespace

void WhileOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns
      .add<RemoveLoopInvariantCarriedValues, RemoveDuplicateLoopCarriedValues>(
          ctx);
}

} // namespace M::MO

//===----------------------------------------------------------------------===//
// UnsqueezeShapeOp
//===----------------------------------------------------------------------===//
LogicalResult UnsqueezeShapeOp::canonicalize(UnsqueezeShapeOp op,
                                             PatternRewriter &rewriter) {
  // Normalize negative indices and make result in sorted order
  TensorValue indicesValue = op.getPaddingIndices();
  TensorType inputShapeType = op.getInputShape().getType();
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(indicesValue);
  if (!maybeIndices || !inputShapeType.getNumElements().has_value())
    return failure();
  SmallVector<int64_t> &indices = maybeIndices.value();
  int64_t outputRank = inputShapeType.getNumElements().value() + indices.size();
  bool changedIndices = false;
  for (int64_t &index : indices) {
    // Leave an out-of-range index for the verifier to reject; normalizing it
    // would land on a valid axis after enough rewrites.
    if (index < -outputRank || index >= outputRank)
      return failure();
    if (index < 0) {
      index += outputRank;
      changedIndices = true;
    }
  }
  if (!llvm::is_sorted(indices)) {
    llvm::sort(indices);
    changedIndices = true;
  }
  if (!changedIndices)
    return failure();
  // Keep the original indices dtype.
  TensorValue newIndices = ConstantOp::create(rewriter, op->getLoc(), indices,
                                              inputShapeType.getDeviceRef());
  newIndices = CastOp::create(rewriter, op->getLoc(), newIndices,
                              indicesValue.getType().getDtype());
  rewriter.replaceOpWithNewOp<UnsqueezeShapeOp>(op, op.getResult().getType(),
                                                op.getInputShape(), newIndices);
  return success();
}

namespace M::MO {

namespace {
/// A pad-like operation with all-zero paddings is a no-op.
struct FoldNoOpPadLikeOp : public OpInterfaceRewritePattern<PadLike> {
  using OpInterfaceRewritePattern<PadLike>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(PadLike op,
                                PatternRewriter &rewriter) const override {
    auto maybePaddings = tryGetConstantIntVector(op.getPaddings());
    if (!maybePaddings.has_value())
      return failure();
    ArrayRef<int64_t> paddings = maybePaddings.value();
    // Rebind rather than forward: the output shape may be less refined than
    // the input, e.g. fully dynamic from the frontends:
    //     mo.pad(input, zeros, val) : ... -> tensor<[?, ?, ?], f32>
    if (llvm::all_of(paddings, [](int64_t n) { return n == 0; })) {
      rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), op.getInput(),
                                            op.getOutputParamDecls());
      return success();
    }
    return failure();
  }
};

/// A scatter-like operation with an empty indices tensor is a no-op.
struct FoldNoOpScatterLikeOp : public OpInterfaceRewritePattern<ScatterLike> {
  using OpInterfaceRewritePattern<ScatterLike>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ScatterLike op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getType() != op.getResult().getType())
      return rewriter.notifyMatchFailure(op,
                                         "input and result types do not match");

    std::optional<int64_t> numIndices =
        op.getIndices().getType().getNumElements();

    if (numIndices != 0)
      return rewriter.notifyMatchFailure(op, "non-empty indices tensor");

    rewriter.replaceAllOpUsesWith(op, op.getInput());
    return success();
  }
};

// `mo.scatter` carries its axis as an attribute, so it is not `ScatterLike`.
struct FoldNoOpScatterOp : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getType() != op.getResult().getType())
      return rewriter.notifyMatchFailure(op,
                                         "input and result types do not match");

    std::optional<int64_t> numIndices =
        op.getIndices().getType().getNumElements();

    if (numIndices != 0)
      return rewriter.notifyMatchFailure(op, "non-empty indices tensor");

    rewriter.replaceAllOpUsesWith(op, op.getInput());
    return success();
  }
};
} // namespace

void MODialect::getCanonicalizationPatterns(RewritePatternSet &patterns) const {
  patterns.add<FoldNoOpPadLikeOp, FoldNoOpScatterLikeOp, FoldNoOpScatterOp>(
      patterns.getContext());
}

} // namespace M::MO
