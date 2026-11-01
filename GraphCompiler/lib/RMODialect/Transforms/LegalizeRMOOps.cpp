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
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/RMODialect/RMOOps.h"
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/DeviceSpecs.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;

namespace M::RMO {
#define GEN_PASS_DEF_LEGALIZERMOOPS
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h.inc"
} // namespace M::RMO

namespace {

/// Dedupes asserts by (condition, message, block).
using AssertCache = llvm::DenseSet<std::tuple<TypedAttr, StringRef, Block *>>;

TensorValue shapeToIndicesTensor(OpBuilder &builder, Location loc,
                                 ShapeAttr shape, DeviceRefAttr device) {
  auto rank = static_cast<int64_t>(shape.getRank());

  auto indicesType =
      TensorType::getVectorSI64Type(builder.getContext(), rank, device);

  auto shapeConstant = MOSH::ParamToValueOp::create(builder, loc, shape);
  return Shape::ToTensorOp::create(builder, loc, indicesType, shapeConstant);
}

// Return whether the shapes in `a` and `b` can be rebound to each other.
bool isCompatibleShape(TensorType a, TensorType b) {
  if (a.getRank() != b.getRank())
    return false;

  // Cannot rebind two different integer dims
  for (size_t i = 0; i < a.getRank(); ++i) {
    if (a.isIntegerDim(i) && b.isIntegerDim(i) &&
        a.getIntDim(i) != b.getIntDim(i))
      return false;
  }

  return true;
}

/// Helpers for the shape_of / broadcast_shape / broadcast_to chain that
/// composite elementwise ops lower to. Results get explicit types whose dims
/// are user-declared on the composite op, so no `RMO_D*` symbols are minted.

/// Build a rank-1 si64 shape tensor describing `tensor.getType()`'s shape.
static TensorValue buildShapeOf(OpBuilder &b, Location loc,
                                TensorValue tensor) {
  TensorType inputTy = tensor.getType();
  auto shapeTy = TensorType::get({static_cast<int64_t>(inputTy.getRank())},
                                 b.getIntegerType(64, /*isSigned=*/true),
                                 inputTy.getDeviceRef());
  return RMO::MOShapeOfOp::create(b, loc, shapeTy, tensor);
}

/// Builds an `rmo.mo.broadcast_shape` with result length `outputRank`.
static TensorValue buildBroadcastShape(OpBuilder &b, Location loc,
                                       TensorValue lhsShape,
                                       TensorValue rhsShape,
                                       int64_t outputRank) {
  auto resultTy =
      TensorType::get({outputRank}, b.getIntegerType(64, /*isSigned=*/true),
                      lhsShape.getType().getDeviceRef());
  return RMO::MOBroadcastShapeOp::create(b, loc, resultTy, lhsShape, rhsShape);
}

/// Broadcasts `tensor` to runtime shape `shape`, typed as `targetType` with
/// `tensor`'s dtype.
static TensorValue buildBroadcastTo(OpBuilder &b, Location loc,
                                    TensorValue tensor, TensorValue shape,
                                    TensorType targetType) {
  return RMO::MOBroadcastToOp::create(
      b, loc, targetType.cloneWith(tensor.getType().getDtype()), tensor, shape);
}

template <typename T>
using ToTensorValueType = TensorValue;

/// Convenience function for promoting the dtype of a sequence of tensor values.
template <typename... Tensors>
static std::tuple<ToTensorValueType<Tensors>...>
promoteDTypes(OpBuilder &builder, Location loc, DType targetDType,
              Tensors... values) {
  auto promote = [&](auto value) -> TensorValue {
    auto tensor = cast<TensorValue>(value);

    if (tensor.getType().getDtype() != targetDType) {
      auto targetType = tensor.getType().cloneWith(targetDType);
      value = RMO::MOCastOp::create(builder, loc, targetType, value);
    }
    return value;
  };

  return {cast<TensorValue>(promote(values))...};
}

/// NOTE: RMO's rebinds are asserting in nature!
std::pair<TensorValue, bool>
_rebindToShape(MO::GraphOp graphOp, TensorValue input, TensorType newTy,
               mlir::RewriterBase &rewriter,
               ArrayRef<ParamDeclAttr> newOutputParamDecl = {}) {
  auto res = RMO::RebindTensorShapeOp::create(rewriter, input.getLoc(), newTy,
                                              input, newOutputParamDecl,
                                              rewriter.getStringAttr(""));

  return {res.getResult(), isCompatibleShape(input.getType(), newTy)};
}

TensorValue rebindToShape(MO::GraphOp graphOp, OpOperand &input,
                          TensorType newTy, mlir::RewriterBase &rewriter,
                          ArrayRef<ParamDeclAttr> newOutputParamDecl = {}) {
  if (input.get().getType() == newTy)
    return cast<TensorValue>(input.get());

  TensorValue val;
  bool verifierSuccess;
  std::tie(val, verifierSuccess) =
      _rebindToShape(graphOp, cast<TensorValue>(input.get()), newTy, rewriter,
                     newOutputParamDecl);

  if (!verifierSuccess) {
    Operation *parent = input.getOwner();
    parent->emitError() << "[LegalizeRMOOps] RMO rebinding failed for op "
                           "in operand index #"
                        << input.getOperandNumber()
                        << " original and expected types are not compatible. "
                           "\n\toriginal type is     : "
                        << input.get().getType()
                        << "\n\tRMO expected type is : " << newTy;
  }

  return val;
}

TensorValue rebindToShape(MO::GraphOp graphOp, OpResult input, TensorType newTy,
                          mlir::RewriterBase &rewriter,
                          ArrayRef<ParamDeclAttr> newOutputParamDecl = {}) {
  if (input.getType() == newTy)
    return cast<TensorValue>(input);

  TensorValue val;
  bool verifierSuccess;
  std::tie(val, verifierSuccess) = _rebindToShape(
      graphOp, cast<TensorValue>(input), newTy, rewriter, newOutputParamDecl);

  if (!verifierSuccess) {
    Operation *parent = input.getOwner();
    parent->emitError()
        << "[LegalizeRMOOps] RMO rebinding failed for op in result index #"
        << input.getResultNumber()
        << " original and expected types are not compatible. \n\toriginal type "
           "is     : "
        << input.getType() << "\n\tRMO expected type is : " << newTy;
  }

  return val;
}

/// Collects the decls in `allParamDecls` that `targetTy` references, so
/// multi-result ops don't redeclare params.
void getParamDeclsFor(TensorType targetTy,
                      ArrayRef<KGEN::ParamDeclAttr> allParamDecls,
                      llvm::SmallVectorImpl<ParamDeclAttr> &outParamDecls) {
  auto addAttrIfDecl = [&](TypedAttr attr) {
    KGEN::ParamDeclRefAttr paramRefAttr =
        dyn_cast<KGEN::ParamDeclRefAttr>(attr);
    if (!paramRefAttr)
      return;
    for (KGEN::ParamDeclAttr decl : allParamDecls)
      if (decl.getName() == paramRefAttr.getName()) {
        outParamDecls.push_back(decl);
      }
  };

  addAttrIfDecl(targetTy.getShape());
  for (TypedAttr dimAttr : targetTy.getShape().getValues())
    addAttrIfDecl(dimAttr);
}

/// Lower the given RMO::MOAnalogue op to its MO analogue given by `generator`.
LogicalResult
lowerMOAnalogue(RMO::MOAnalogue op, PatternRewriter &rewriter,
                const std::function<MO::ParamDeclarationInterface(
                    PatternRewriter &rewriter, const Location &, ArrayRef<Type>,
                    ArrayRef<Value>)> &generator) {
  auto graphOp = op->getParentOfType<MO::GraphOp>();

  size_t numOperands = op->getNumOperands();
  size_t numResults = op->getNumResults();

  SmallVector<Type> moOperandTypes, moResultTypes;
  SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;
  SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;
  op.inferMOTypes(graphOp, moOperandTypes, moResultTypes, moInputParamDecls,
                  moOutputParamDecls);

  // Rebind inputs to what the MO op would expect
  SmallVector<Value> newOperands;
  for (size_t i = 0; i < numOperands; ++i) {
    Value val = op->getOperand(i);
    Type targetType = moOperandTypes[i];
    TensorValue tensorVal = dyn_cast<TensorValue>(val);
    if (!tensorVal) {
      ASSERT_STREAM(targetType == val.getType(),
                    << "Disagreement on type for non-tensor argument.");
      newOperands.push_back(val);
      continue;
    }
    SmallVector<KGEN::ParamDeclAttr> &curInputDecls = moInputParamDecls[i];

    TensorValue reboundValue =
        rebindToShape(graphOp, op->getOpOperand(i),
                      cast<TensorType>(targetType), rewriter, curInputDecls);

    newOperands.push_back(reboundValue);
  }

  // If `inferMOTypes` returned the declared result types verbatim (see
  // `requireRmoResults`), the MO op inherits the RMO op's
  // `output_param_decls`, since the RMO op is about to be erased. Otherwise it
  // owns the decls from `inferMOTypes`.
  MO::ParamDeclarationInterface newOp =
      generator(rewriter, op->getLoc(), moResultTypes, newOperands);

  bool resultTypesPassthrough = true;
  for (size_t i = 0; i < numResults; ++i) {
    if (moResultTypes[i] != op->getResult(i).getType()) {
      resultTypesPassthrough = false;
      break;
    }
  }
  if (resultTypesPassthrough) {
    newOp.setOutputParamDecls(op.getMOAnalogueOutputParamDecls());
  } else {
    SmallVector<KGEN::ParamDeclAttr> moOutputParamDeclsFlattened;
    for (const auto &resultDecls : moOutputParamDecls)
      moOutputParamDeclsFlattened.append(resultDecls);
    newOp.setOutputParamDecls(moOutputParamDeclsFlattened);
  }

  // Rebind the output of the MO op to what the original op had and replace
  for (size_t i = 0; i < numResults; ++i) {
    Value newValue = newOp->getResult(i);
    Value oldValue = op->getResult(i);
    TensorType targetType = cast<TensorType>(op->getResultTypes()[i]);

    TensorValue newTensorValue = dyn_cast<TensorValue>(newValue);

    // Device verifiers are weak; copy the declared device to avoid rebinds
    // that change it.
    newTensorValue.setType(
        newTensorValue.getType().cloneWith(targetType.getDeviceRef()));

    if (!newTensorValue) {
      rewriter.replaceAllUsesWith(oldValue, newValue);
      continue;
    }

    TensorValue reboundValue;
    if (numResults == 1) {
      reboundValue =
          rebindToShape(graphOp, newOp->getOpResult(i), targetType, rewriter,
                        op.getMOAnalogueOutputParamDecls());
    } else {
      // The decls cover all outputs; keep only this output's to avoid
      // redefinitions.
      llvm::SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
      getParamDeclsFor(targetType, op.getMOAnalogueOutputParamDecls(),
                       outputParamDecls);
      reboundValue = rebindToShape(graphOp, newOp->getOpResult(i), targetType,
                                   rewriter, outputParamDecls);
    }
    rewriter.replaceAllUsesWith(oldValue, reboundValue);
  }

  return success();
}

struct LegalizeRMOMOAnalogue
    : public OpInterfaceRewritePattern<RMO::MOAnalogue> {

  LegalizeRMOMOAnalogue(MLIRContext *ctx, mlir::PatternBenefit benefit = 0)
      : OpInterfaceRewritePattern<RMO::MOAnalogue>(ctx, benefit) {}

  LogicalResult matchAndRewrite(RMO::MOAnalogue op,
                                PatternRewriter &rewriter) const override {
    auto generator =
        [&op](PatternRewriter &rewriter, const Location &loc,
              ArrayRef<Type> resultTypes,
              ArrayRef<Value> operands) -> MO::ParamDeclarationInterface {
      // All rmo ops have the form "rmo.mo.opname".
      StringAttr name = rewriter.getStringAttr(op->getName().stripDialect());

      auto result = rewriter.create(loc, name, ValueRange(operands),
                                    TypeRange(resultTypes));

      NamedAttrList newAttrs(result->getAttrs());
      // An inherent RMO attribute the MO op lacks (e.g. a reduction's
      // `axis`) is carried by the MO result type instead.
      // TODO(GEX-3348): only reductions need this. Drop `axis` from the RMO
      // reductions too, then copy every attribute again.
      ArrayRef<StringAttr> rmoInherent =
          op->getName().getRegisteredInfo()->getAttributeNames();
      ArrayRef<StringAttr> moInherent =
          result->getName().getRegisteredInfo()->getAttributeNames();

      for (auto attr : op->getAttrs()) {
        if (attr.getName() == op.getMOAnalogueOutputParamDeclsAttrName())
          continue;
        if (llvm::is_contained(rmoInherent, attr.getName()) &&
            !llvm::is_contained(moInherent, attr.getName()))
          continue;
        newAttrs.set(attr.getName(), attr.getValue());
      }

      rewriter.modifyOpInPlace(result, [&] { result->setAttrs(newAttrs); });
      return cast<MO::ParamDeclarationInterface>(result);
    };
    return lowerMOAnalogue(op, rewriter, generator);
  }
};

struct LegalizeMutableLoadOp : public OpRewritePattern<RMO::MOMutableLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MOMutableLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->getParentOfType<MO::GraphOp>();

    SmallVector<Type> moOperandTypes, moResultTypes;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;

    RMO::MOAnalogue rma = op;
    rma.inferMOTypes(graphOp, moOperandTypes, moResultTypes, moInputParamDecls,
                     moOutputParamDecls);

    SmallVector<Value> moOperands;
    moOperands.push_back(op.getInBuffer());
    moOperands.push_back(op.getInChain());

    auto moOp = MO::MutableLoadOp::create(rewriter, op->getLoc(), moResultTypes,
                                          moOperands);
    moOp->setDiscardableAttrs(op->getRawDictionaryAttrs());
    OpResult outTensorRes = moOp->getOpResult(0);
    auto reboundOutput = rebindToShape(graphOp, outTensorRes,
                                       op.getOutTensor().getType(), rewriter);
    rewriter.replaceAllUsesWith(op.getOutTensor(), reboundOutput);
    rewriter.replaceAllUsesWith(op.getOutChain(), moOp.getOutChain());

    op->erase();

    return success();
  }
};

struct LegalizeMutableStoreOp : public OpRewritePattern<RMO::MOMutableStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MOMutableStoreOp op,
                                PatternRewriter &rewriter) const override {

    auto graphOp = op->getParentOfType<MO::GraphOp>();

    SmallVector<Type> moOperandTypes, moResultTypes;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;

    RMO::MOAnalogue rmaOp = op;
    rmaOp.inferMOTypes(graphOp, moOperandTypes, moResultTypes,
                       moInputParamDecls, moOutputParamDecls);

    SmallVector<Value> moOperands;

    auto reboundInTensor =
        rebindToShape(graphOp, op.getInTensorMutable(),
                      cast<TensorType>(moOperandTypes[1]), rewriter);

    moOperands.push_back(op.getInBuffer());
    moOperands.push_back(reboundInTensor);
    moOperands.push_back(op.getInChain());

    auto moOp = MO::MutableStoreOp::create(rewriter, op->getLoc(),
                                           moResultTypes, moOperands);
    moOp->setDiscardableAttrs(op->getRawDictionaryAttrs());
    rewriter.replaceAllUsesWith(op.getOutChain(), moOp.getOutChain());

    op->erase();

    return success();
  }
};

struct LegalizeTransferOp : public OpRewritePattern<RMO::MOTransferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MOTransferOp op,
                                PatternRewriter &rewriter) const override {
    // No dynamic dims permitted in RMO at this point: check that explicitly.
    [[maybe_unused]] TensorType inTy =
        cast<TensorType>(op.getInput().getType());
    [[maybe_unused]] TensorType outTy =
        cast<TensorType>(op.getResult().getType());

    rewriter.replaceOpWithNewOp<MO::TransferOp>(
        op, op.getInput(), outTy.getDeviceRef(), op.getInChain());

    return success();
  }
};

struct LegalizeMutableStoreSliceOp
    : public OpRewritePattern<RMO::MOMutableStoreSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MOMutableStoreSliceOp op,
                                PatternRewriter &rewriter) const override {

    auto graphOp = op->getParentOfType<MO::GraphOp>();

    SmallVector<Type> moOperandTypes, moResultTypes;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;
    SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;

    RMO::MOAnalogue rmaOp = op;
    rmaOp.inferMOTypes(graphOp, moOperandTypes, moResultTypes,
                       moInputParamDecls, moOutputParamDecls);

    SmallVector<Value> moOperands;

    moOperands.push_back(op.getInBuffer());

    for (size_t i = 1; i < moOperandTypes.size() - 1; ++i) {
      SmallVector<KGEN::ParamDeclAttr> &curInputDecls = moInputParamDecls[i];

      auto rebound = rebindToShape(graphOp, op->getOpOperand(i),
                                   cast<TensorType>(moOperandTypes[i]),
                                   rewriter, curInputDecls);

      moOperands.push_back(rebound);
    }

    moOperands.push_back(op.getInChain());

    // Only the inherent `outputParamDecls` carries over into properties; the
    // discardable half is replaced wholesale just below.
    MO::MutableStoreSliceOp::Properties properties;
    if (auto decls = op->getAttrOfType<KGEN::ParamDeclArrayAttr>(
            MO::MutableStoreSliceOp::getAttributeNames().front()))
      properties.setOutputParamDecls(decls);

    auto moOp = MO::MutableStoreSliceOp::create(
        rewriter, op->getLoc(), moResultTypes, moOperands, properties);
    moOp->setDiscardableAttrs(op->getRawDictionaryAttrs());
    rewriter.replaceAllUsesWith(op.getOutChain(), moOp.getOutChain());

    op->erase();

    return success();
  }
};

struct LowerRebindTensorShapeOp
    : public OpRewritePattern<RMO::RebindTensorShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::RebindTensorShapeOp op,
                                PatternRewriter &rewriter) const override {

    TensorValue operand = op.getInput();
    TensorValue result = op.getResult();
    TypedAttr operandShape = operand.getType().getShape();
    TypedAttr resultShape = result.getType().getShape();

    MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    auto boolTy =
        KGEN::SIMDType::get(op.getContext(), 1, KGEN::KGENDType::kBool);
    TypedAttr cmpAttr =
        KGEN::emitMLIROperationCall(MOSH::EqualOp::getOperationName(), {},
                                    {operandShape, resultShape}, boolTy);
    KGEN::ParamDeclAttr condAttr =
        graphOp.getNewParamOfType("RMO_SH_EQ", boolTy);
    KGEN::ParamDeclareOp::create(rewriter, loc, condAttr, cmpAttr);

    std::string fullMsg = "rebind shape check failed: expected shape ";
    llvm::raw_string_ostream ss(fullMsg);
    ss << MOSH::shapeTypedAttrToStr(resultShape) << " but got "
       << MOSH::shapeTypedAttrToStr(operandShape);
    StringAttr msg = mlir::cast<StringAttr>(op.getMessageAttr());
    if (!msg.empty()) {
      ss << "\n" << msg;
    }

    TypedAttr assertAttr = MO::AssertOp::emitStaticCall(
        KGEN::ParamDeclRefAttr::get(condAttr), fullMsg);

    auto chainTy = rewriter.getType<ChainType>();
    KGEN::ParamDeclAttr chainAttr =
        graphOp.getNewParamOfType("RMO_SH_CH", chainTy);
    KGEN::ParamDeclareOp::create(rewriter, loc, chainAttr, assertAttr);

    auto moRebind = MO::RebindOp::create(rewriter, loc, result.getType(),
                                         operand, op.getOutputParamDecls());
    rewriter.replaceOp(op, moRebind);

    return success();
  }
};

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerCompositeBinaryElemwiseOp : public OpRewritePattern<RMOOpT> {
  using OpRewritePattern<RMOOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(RMOOpT op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->template getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue x = op.getInputX();
    TensorValue y = op.getInputY();
    auto xDtype = x.getType().getDtype();
    auto yDtype = y.getType().getDtype();
    auto resultDtype = op.getResult().getType().getDtype();

    if (xDtype != resultDtype) {
      x = RMO::MOCastOp::create(rewriter, loc,
                                x.getType().cloneWith(resultDtype), x);
    }
    if (yDtype != resultDtype) {
      y = RMO::MOCastOp::create(rewriter, loc,
                                y.getType().cloneWith(resultDtype), y);
    }

    TensorValue xFinal = x;
    TensorValue yFinal = y;
    if (x.getType().getShape() != y.getType().getShape()) {
      // `op.getType()` is the user-declared broadcasted shape; reuse its dims
      // for the broadcast destination so we don't mint fresh symbols.
      TensorType bcastType = op.getType();
      TensorValue xShape = buildShapeOf(rewriter, loc, x);
      TensorValue yShape = buildShapeOf(rewriter, loc, y);
      TensorValue commonShape = buildBroadcastShape(
          rewriter, loc, xShape, yShape, bcastType.getRank());
      xFinal = buildBroadcastTo(rewriter, loc, x, commonShape, bcastType);
      yFinal = buildBroadcastTo(rewriter, loc, y, commonShape, bcastType);
    }

    auto answer = MOAnalogueOpT::create(rewriter, loc,
                                        ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

template <typename RMOOpT, typename MOAnalogueOpT>
struct LowerCompositeBinaryComparisonOp : public OpRewritePattern<RMOOpT> {
  using OpRewritePattern<RMOOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(RMOOpT op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->template getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue x = op.getInputX();
    TensorValue y = op.getInputY();
    auto xDtype = x.getType().getDtype();
    auto yDtype = y.getType().getDtype();

    ErrorOr<DType> promotedDtype = M::RMO::promoteDtype(xDtype, yDtype);
    ASSERT_STREAM(
        !promotedDtype.isError(),
        << "failed to calculate valid promotion dtype in lowering. This should "
           "be impossible due to it being calculated when inferring the return "
           "type"
        << promotedDtype.takeError());
    DType comparisonDtype = promotedDtype.takeValue();
    if (xDtype != comparisonDtype) {
      x = RMO::MOCastOp::create(rewriter, loc,
                                x.getType().cloneWith(comparisonDtype), x);
    }
    if (yDtype != comparisonDtype) {
      y = RMO::MOCastOp::create(rewriter, loc,
                                y.getType().cloneWith(comparisonDtype), y);
    }

    TensorValue xFinal = x;
    TensorValue yFinal = y;
    if (x.getType().getShape() != y.getType().getShape()) {
      // The comparison op's result shape carries the broadcast destination's
      // user-declared dim names; reuse them for the broadcasts (with each
      // operand's pre-comparison dtype).
      TensorType bcastType = op.getType();
      TensorValue xShape = buildShapeOf(rewriter, loc, x);
      TensorValue yShape = buildShapeOf(rewriter, loc, y);
      TensorValue commonShape = buildBroadcastShape(
          rewriter, loc, xShape, yShape, bcastType.getRank());
      xFinal = buildBroadcastTo(rewriter, loc, x, commonShape, bcastType);
      yFinal = buildBroadcastTo(rewriter, loc, y, commonShape, bcastType);
    }

    auto answer = MOAnalogueOpT::create(rewriter, loc,
                                        ValueRange{xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerBroadcastLike : public OpRewritePattern<RMO::BroadcastLikeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::BroadcastLikeOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue shapeLike =
        buildShapeOf(rewriter, loc, cast<TensorValue>(op.getShapeLike()));
    TensorValue answer =
        buildBroadcastTo(rewriter, loc, cast<TensorValue>(op.getInput()),
                         shapeLike, op.getType());
    TensorValue reboundAnswer =
        rebindToShape(graphOp, cast<OpResult>(answer), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerConcat : public OpRewritePattern<RMO::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ConcatOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    int64_t axisInt = op.getAxis().getInt();

    TensorValue concat = MO::ConcatOp::create(
        rewriter, loc, op.getResult().getType(), op.getAxis().getValue(),
        op.getInputs(), op.getOutputParamDecls());

    // If there is a user specified rebind for the axis dim, generate the
    // required assert.
    if (TypedAttr outDimAttr =
            mlir::dyn_cast_or_null<TypedAttr>(op.getOutDimAttr())) {
      MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
      SmallVector<ParamBinding> paramBindings;
      ParamExprBuilder b(graphOp, loc, &paramBindings);

      int64_t rank = op.getResult().getType().getRank();
      axisInt = (axisInt < 0) ? axisInt + rank : axisInt;

      SmallVector<TypedAttr> inputDims;
      for (auto input : op.getOperands())
        inputDims.push_back(
            mlir::cast<TensorType>(input.getType()).getShape()[axisInt]);

      Var concatDimSum =
          b.createVar(KGEN::ParamOperatorAttr::get(KGEN::POC::Add, inputDims));
      Var newDim = b.createVar(outDimAttr);
      b.createAssert(concatDimSum == newDim,
                     "[concat] out_dim must be equivalent to the sum of the "
                     "input dims along the concat axis");

      for (auto binding : paramBindings)
        binding.toOp(loc, rewriter);
    }

    rewriter.replaceOp(op, concat);

    return success();
  }
};

struct LowerReshape : public OpRewritePattern<RMO::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // The shape is fully resolved, so lower to the static op.
    rewriter.replaceOpWithNewOp<MO::Static::ReshapeOp>(
        op, op.getResult().getType(), op.getInput());
    return success();
  }
};

struct LowerBroadcastTo : public OpRewritePattern<RMO::BroadcastToOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::BroadcastToOp op,
                                PatternRewriter &rewriter) const override {
    // The shape is fully resolved, so lower to the static op.
    rewriter.replaceOpWithNewOp<MO::Static::BroadcastToOp>(
        op, op.getResult().getType(), op.getInput());
    return success();
  }
};

struct LowerMatmul : public OpRewritePattern<RMO::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    TensorValue lhs = op.getInputX();
    TensorValue rhs = op.getInputY();
    TensorType lhsType = lhs.getType();
    TensorType rhsType = rhs.getType();
    ShapeAttr lhsShape = lhsType.getShape();
    ShapeAttr rhsShape = rhsType.getShape();

    size_t lhsRank = lhsShape.getRank();
    size_t rhsRank = rhsShape.getRank();
    size_t fullRank = std::max(size_t(2), std::max(lhsRank, rhsRank));

    // Reshape to ensure both tensors have same rank.
    auto [lhsReshapeTarget, rhsReshapeTarget] =
        expandMatmulInputsForBroadcasting(lhsShape, rhsShape);

    auto lhsReshapeShape = ShapeAttr::get(lhsReshapeTarget, lhsShape.getType());
    if (lhsReshapeShape != lhsShape)
      lhs = MO::Static::ReshapeOp::create(
          rewriter, loc, lhsType.cloneWith(lhsReshapeShape), lhs);

    auto rhsReshapeShape = ShapeAttr::get(rhsReshapeTarget, rhsShape.getType());
    if (rhsReshapeShape != rhsShape)
      rhs = MO::Static::ReshapeOp::create(
          rewriter, loc, rhsType.cloneWith(rhsReshapeShape), rhs);

    // Calculated broadcast shapes and broadcast outer dims.
    auto intDim1 = MOSH::ShapeAttr::getDim(lhsShape.getContext(), 1ll);
    SmallVector<TypedAttr> lhsBroadcastTarget{lhsReshapeTarget};
    SmallVector<TypedAttr> rhsBroadcastTarget{rhsReshapeTarget};
    for (size_t i = 0; i < fullRank - 2; ++i) {
      TypedAttr dim;
      if (lhsReshapeTarget[i] == intDim1)
        dim = rhsReshapeTarget[i];
      else
        dim = lhsReshapeTarget[i];

      lhsBroadcastTarget[i] = dim;
      rhsBroadcastTarget[i] = dim;
    }

    auto lhsBroadcastShape =
        ShapeAttr::get(lhsBroadcastTarget, lhsShape.getType());
    if (lhsBroadcastShape != lhsReshapeShape)
      lhs = MO::Static::BroadcastToOp::create(
          rewriter, loc, lhsType.cloneWith(lhsBroadcastShape), lhs);

    auto rhsBroadcastShape =
        ShapeAttr::get(rhsBroadcastTarget, rhsShape.getType());
    if (rhsBroadcastShape != rhsReshapeShape)
      rhs = MO::Static::BroadcastToOp::create(
          rewriter, loc, rhsType.cloneWith(rhsBroadcastShape), rhs);

    SmallVector<TypedAttr> matmulTarget{lhsBroadcastTarget};
    matmulTarget[fullRank - 1] = rhsBroadcastTarget[fullRank - 1];
    auto matmulShape = ShapeAttr::get(matmulTarget, rhsShape.getType());
    auto matmulType = op.getType().cloneWith(matmulShape);
    TensorValue matmul;
    if (fullRank == 2)
      matmul = MO::MatmulOp::create(rewriter, loc, matmulType, lhs, rhs);
    else
      matmul = MO::BatchMatmulOp::create(rewriter, loc, matmulType, lhs, rhs);

    // Reshape to remove extra one dimensions if they were introduced by vector
    // inputs.
    if (matmulShape != op.getType().getShape())
      matmul =
          MO::Static::ReshapeOp::create(rewriter, loc, op.getType(), matmul);

    rewriter.replaceAllUsesWith(op, matmul);
    return success();
  }
};

struct LowerConv : public OpRewritePattern<RMO::ConvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ConvOp conv,
                                PatternRewriter &rewriter) const override {
    Location loc = conv.getLoc();

    ShapeAttr strides = conv.getStrides();
    ShapeAttr paddings = conv.getPaddings();
    ShapeAttr dilations = conv.getDilations();
    int64_t groups = conv.getNumGroups();
    StringAttr inputLayout = conv.getInputLayoutAttr();

    auto resultType = conv.getType();

    auto [promotedInput, promotedFilter] =
        promoteDTypes(rewriter, conv.getLoc(), resultType.getDtype(),
                      conv.getInput(), conv.getFilter());

    auto device = getHostDevice(conv.getContext());
    auto stridesConstant = shapeToIndicesTensor(rewriter, loc, strides, device);

    auto paddingConstant =
        shapeToIndicesTensor(rewriter, loc, paddings, device);

    auto dilationsConstant =
        shapeToIndicesTensor(rewriter, loc, dilations, device);

    auto groupsConstant = MO::ConstantOp::create(rewriter, loc, groups, device);

    rewriter.replaceOpWithNewOp<RMO::MOConvOp>(
        conv, conv.getType(), promotedInput, promotedFilter, stridesConstant,
        dilationsConstant, paddingConstant, groupsConstant, inputLayout,
        rewriter.getStringAttr(""));
    return success();
  }
};

struct LowerConvTranspose : public OpRewritePattern<RMO::ConvTransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ConvTransposeOp conv,
                                PatternRewriter &rewriter) const override {
    Location loc = conv.getLoc();

    ShapeAttr strides = conv.getStrides();
    ShapeAttr paddings = conv.getPaddings();
    ShapeAttr dilations = conv.getDilations();
    ShapeAttr outputPaddings = conv.getOutputPaddings();
    StringAttr inputLayout = conv.getInputLayoutAttr();

    TensorType resultType = conv.getType();

    auto [promotedInput, promotedFilter] =
        promoteDTypes(rewriter, conv.getLoc(), resultType.getDtype(),
                      conv.getInput(), conv.getFilter());

    auto device = getHostDevice(conv.getContext());
    TensorValue stridesConstant =
        shapeToIndicesTensor(rewriter, loc, strides, device);

    TensorValue paddingConstant =
        shapeToIndicesTensor(rewriter, loc, paddings, device);

    TensorValue outputPaddingConstant =
        shapeToIndicesTensor(rewriter, loc, outputPaddings, device);

    TensorValue dilationsConstant =
        shapeToIndicesTensor(rewriter, loc, dilations, device);

    rewriter.replaceOpWithNewOp<RMO::MOConvTransposeOp>(
        conv, conv.getType(), promotedInput, promotedFilter, stridesConstant,
        dilationsConstant, paddingConstant, outputPaddingConstant, inputLayout,
        rewriter.getStringAttr(""));
    return success();
  }
};

struct LowerAvgPool : public OpRewritePattern<RMO::AvgPoolOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::AvgPoolOp avg_pool,
                                PatternRewriter &rewriter) const override {
    Location loc = avg_pool.getLoc();

    ShapeAttr filterShape = avg_pool.getFilterShape();
    ShapeAttr strides = avg_pool.getStrides();
    ShapeAttr paddings = avg_pool.getPaddings();
    ShapeAttr dilations = avg_pool.getDilations();
    auto ceilMode = avg_pool.getCeilMode();
    auto countBoundary = avg_pool.getCountBoundary();
    TensorType resultType = avg_pool.getType();

    auto [promotedInput] =
        promoteDTypes(rewriter, avg_pool.getLoc(), resultType.getDtype(),
                      avg_pool.getInput());

    auto device = getHostDevice(avg_pool.getContext());
    TensorValue filterShapeConstant =
        shapeToIndicesTensor(rewriter, loc, filterShape, device);

    TensorValue stridesConstant =
        shapeToIndicesTensor(rewriter, loc, strides, device);

    TensorValue paddingConstant =
        shapeToIndicesTensor(rewriter, loc, paddings, device);

    TensorValue dilationsConstant =
        shapeToIndicesTensor(rewriter, loc, dilations, device);

    if (ceilMode) {
      rewriter.replaceOpWithNewOp<RMO::MOAvgPoolCeilOp>(
          avg_pool, avg_pool.getType(), promotedInput, filterShapeConstant,
          stridesConstant, dilationsConstant, paddingConstant, countBoundary);
    } else {
      rewriter.replaceOpWithNewOp<RMO::MOAvgPoolOp>(
          avg_pool, avg_pool.getType(), promotedInput, filterShapeConstant,
          stridesConstant, dilationsConstant, paddingConstant, countBoundary);
    }

    return success();
  }
};

struct LowerMaxPool : public OpRewritePattern<RMO::MaxPoolOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::MaxPoolOp max_pool,
                                PatternRewriter &rewriter) const override {
    Location loc = max_pool.getLoc();

    ShapeAttr filterShape = max_pool.getFilterShape();
    ShapeAttr strides = max_pool.getStrides();
    ShapeAttr paddings = max_pool.getPaddings();
    ShapeAttr dilations = max_pool.getDilations();
    auto ceilMode = max_pool.getCeilMode();

    TensorType resultType = max_pool.getType();

    auto [promotedInput] =
        promoteDTypes(rewriter, max_pool.getLoc(), resultType.getDtype(),
                      max_pool.getInput());

    auto device = getHostDevice(max_pool.getContext());
    TensorValue filterShapeConstant =
        shapeToIndicesTensor(rewriter, loc, filterShape, device);

    TensorValue stridesConstant =
        shapeToIndicesTensor(rewriter, loc, strides, device);

    TensorValue paddingConstant =
        shapeToIndicesTensor(rewriter, loc, paddings, device);

    TensorValue dilationsConstant =
        shapeToIndicesTensor(rewriter, loc, dilations, device);

    if (ceilMode) {
      rewriter.replaceOpWithNewOp<RMO::MOMaxPoolCeilOp>(
          max_pool, max_pool.getType(), promotedInput, filterShapeConstant,
          stridesConstant, dilationsConstant, paddingConstant);
    } else {
      rewriter.replaceOpWithNewOp<RMO::MOMaxPoolOp>(
          max_pool, max_pool.getType(), promotedInput, filterShapeConstant,
          stridesConstant, dilationsConstant, paddingConstant);
    }

    return success();
  }
};

struct LowerSelect : public OpRewritePattern<RMO::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto graphOp = op->template getParentOfType<MO::GraphOp>();
    Location loc = op->getLoc();

    TensorValue cond = op.getCond();
    TensorValue x = op.getInputX();
    TensorValue y = op.getInputY();
    auto xDtype = x.getType().getDtype();
    auto yDtype = y.getType().getDtype();
    auto resultDtype = op.getResult().getType().getDtype();

    if (xDtype != resultDtype) {
      x = RMO::MOCastOp::create(rewriter, loc,
                                x.getType().cloneWith(resultDtype), x);
    }
    if (yDtype != resultDtype) {
      y = RMO::MOCastOp::create(rewriter, loc,
                                y.getType().cloneWith(resultDtype), y);
    }

    TensorValue condFinal = cond;
    TensorValue xFinal = x;
    TensorValue yFinal = y;
    if (x.getType().getShape() != y.getType().getShape() ||
        x.getType().getShape() != cond.getType().getShape()) {
      // The select op's result shape is the broadcasted destination's
      // user-declared shape; reuse its dims for each broadcast.
      TensorType bcastType = op.getType();
      int64_t outputRank = bcastType.getRank();
      TensorValue condShape = buildShapeOf(rewriter, loc, cond);
      TensorValue xShape = buildShapeOf(rewriter, loc, x);
      TensorValue yShape = buildShapeOf(rewriter, loc, y);

      // The intermediate broadcast of x and y has its own (possibly smaller)
      // rank; the final broadcast against `cond` has the destination rank.
      int64_t xyRank = std::max(x.getType().getRank(), y.getType().getRank());
      TensorValue xyShape =
          buildBroadcastShape(rewriter, loc, xShape, yShape, xyRank);
      TensorValue commonShape =
          buildBroadcastShape(rewriter, loc, condShape, xyShape, outputRank);

      condFinal = buildBroadcastTo(rewriter, loc, cond, commonShape, bcastType);
      xFinal = buildBroadcastTo(rewriter, loc, x, commonShape, bcastType);
      yFinal = buildBroadcastTo(rewriter, loc, y, commonShape, bcastType);
    }

    auto answer = RMO::MOSelectOp::create(
        rewriter, loc, ValueRange{condFinal, xFinal, yFinal}, graphOp);
    TensorValue reboundAnswer =
        rebindToShape(graphOp, answer->getOpResult(0), op.getType(), rewriter,
                      op.getOutputParamDecls());

    rewriter.replaceAllUsesWith(op, reboundAnswer);
    return success();
  }
};

struct LowerTopK : public OpRewritePattern<RMO::TopKOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::TopKOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();

    TensorValue input = op.getInput();
    DeviceRefAttr deviceRef = input.getType().getDeviceRef();
    if (!deviceRef)
      return op.emitError() << "rmo.top_k lowering requires input tensor to "
                               "have an assigned device";

    auto inputType = input.getType();
    auto inputShape = inputType.getShape();
    const int64_t kVal = op.getK();
    const int64_t rank = inputShape.getRank();
    // Normalize axis. `inferReturnTypes` checks -rank <= axis < rank already.
    int64_t axisVal = op.getAxis();
    axisVal = axisVal < 0 ? axisVal + rank : axisVal;

    MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
    SmallVector<ParamBinding> paramBindings;
    ParamExprBuilder b(graphOp, loc, &paramBindings);

    TypedAttr axisDimAttr = inputShape[axisVal];
    Var axisDim = b.createVar(axisDimAttr);
    Var kVar = b.createVar(ShapeAttr::getDim(ctx, kVal));

    b.createAssert(kVar <= axisDim,
                   "k value exceeds dimension size along specified axis");

    for (const ParamBinding &binding : paramBindings)
      binding.toOp(loc, rewriter);

    // Convert attributes to tensor constants expected by `mo.top_k`.
    auto k =
        MO::ConstantOp::create(rewriter, loc, op.getK(), getHostDevice(ctx));
    auto axis =
        MO::ConstantOp::create(rewriter, loc, op.getAxis(), getHostDevice(ctx));
    auto sorted = MO::ConstantOp::create(rewriter, loc, /*value=*/true,
                                         getHostDevice(ctx));

    auto topkOp =
        MO::TopKOp::create(rewriter, op.getLoc(), op.getValues().getType(),
                           op.getIndices().getType(), input, k, axis, sorted,
                           op.getOutputParamDecls());

    rewriter.replaceOp(op, topkOp->getResults());
    return success();
  }
};

struct LowerSlice : public OpRewritePattern<RMO::SliceOp> {
  LowerSlice(MLIRContext *ctx, AssertCache &cache)
      : OpRewritePattern<RMO::SliceOp>(ctx), assertCache(cache) {}

  AssertCache &assertCache;

  void createAssert(ParamExprBuilder &builder, RMO::SliceOp op,
                    const Var &condAttr, StringRef msg) const {
    auto block = op->getBlock();
    auto attr = condAttr.getAttr();

    if (!assertCache.insert({attr, msg, block}).second)
      return;

    builder.createAssert(condAttr, msg);
  }

  LogicalResult matchAndRewrite(RMO::SliceOp op,
                                PatternRewriter &rewriter) const override {
    // Note that rmo.slice has special restrictions compared to mo.slice:
    // - Requires start to be in [-dim, dim) and stop in [-dim, dim].
    // - Requires steps to be a statically known positive integer.
    // These keep the algebraic dims in the rmo.slice type small.
    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();
    TensorValue input = op.getInput();
    auto inputShape = input.getType().getShape();
    ShapeAttr startsAttr = op.getStarts();
    ShapeAttr stopsAttr = op.getStops();
    ShapeAttr stepsAttr = op.getSteps();
    auto maybeSteps = MO::tryGetConstantIntVector(stepsAttr);
    if (!maybeSteps ||
        llvm::any_of(*maybeSteps, [](int64_t step) { return step == 0; }))
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("rmo.slice expects static steps != 0 but got {0}",
                             stepsAttr));

    // Convert start/stop/step to tensors expected by mo.slice.
    auto indicesType = TensorType::getVectorSI64Type(
        ctx, startsAttr.getRank(),
        // Always put indices on CPU, as expected by the `mo.slice` kernel.
        DeviceRefAttr::get(ctx, kCPULabel, /*id=*/0));
    auto toTensorOp = [loc = loc, indicesType = indicesType,
                       &rewriter](ShapeAttr attr) -> Value {
      auto constOp = MOSH::ParamToValueOp::create(rewriter, loc, attr);
      return Shape::ToTensorOp::create(rewriter, loc, indicesType, constOp);
    };

    auto starts = toTensorOp(startsAttr);
    auto stops = toTensorOp(stopsAttr);
    auto steps = toTensorOp(stepsAttr);

    MO::GraphOp graphOp = op->getParentOfType<MO::GraphOp>();
    SmallVector<ParamBinding> paramBindings;
    ParamExprBuilder b(graphOp, loc, &paramBindings);
    for (size_t axis = 0; axis < input.getType().getRank(); ++axis) {
      // The mo.slice shape function checks step != 0 and the start/stop
      // ordering implied by the step's sign.
      Var start = b.createVar(startsAttr[axis]);
      Var stop = b.createVar(stopsAttr[axis]);
      auto maybeStartConst = tryGetConstantInt(startsAttr[axis]);
      if (!maybeStartConst)
        createAssert(b, op, start >= 0,
                     "[rmo.slice] requires positive start when symbolic");
      auto maybeStopConst = tryGetConstantInt(stopsAttr[axis]);
      if (!maybeStopConst)
        createAssert(b, op, stop >= 0,
                     "[rmo.slice] requires positive stop when symbolic");

      // Stop's upper bound is inclusive since stop = dim is the natural
      // exclusive end.
      Var dim = b.createVar(inputShape[axis]);
      createAssert(b, op, start >= -dim && start < dim,
                   "[rmo.slice] requires start in [-dim, dim)");
      createAssert(b, op, stop >= -dim && stop <= dim,
                   "[rmo.slice] requires stop in [-dim, dim]");
    }

    auto res = MO::SliceOp::create(rewriter, op->getLoc(), op.getType(), input,
                                   starts, stops, steps);
    res->setDiscardableAttrs(op->getRawDictionaryAttrs());
    rewriter.replaceAllOpUsesWith(op, res);
    return success();
  }
};

struct LowerShapeToTensor : public OpRewritePattern<RMO::ShapeToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RMO::ShapeToTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypedAttr shape = op.getShape();
    TensorType resTy = op.getType();

    auto shConst = MOSH::ParamToValueOp::create(rewriter, loc, shape);
    rewriter.replaceOpWithNewOp<Shape::ToTensorOp>(op, resTy, shConst);
    return success();
  }
};

struct LegalizeRMOOps
    : public M::RMO::impl::LegalizeRMOOpsBase<LegalizeRMOOps> {
  using LegalizeRMOOpsBase::LegalizeRMOOpsBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet blockPatterns(context);

    // Benefit 0 so more specific patterns override the default lowering.
    blockPatterns.add<LegalizeRMOMOAnalogue>(context, /*benefit=*/0);

    // Override for rmo.mo.transfer to handle chain and avoid param decls.
    blockPatterns.add<LegalizeTransferOp>(context);

    blockPatterns.add<LegalizeMutableLoadOp, LegalizeMutableStoreOp,
                      LegalizeMutableStoreSliceOp>(context);

    if (!skipRMORebind)
      blockPatterns.add<LowerRebindTensorShapeOp>(context);

    blockPatterns.add<
        LowerCompositeBinaryElemwiseOp<RMO::AddOp, RMO::MOAddOp>,
        LowerCompositeBinaryElemwiseOp<RMO::SubOp, RMO::MOSubOp>,
        LowerCompositeBinaryElemwiseOp<RMO::MulOp, RMO::MOMulOp>,
        LowerCompositeBinaryElemwiseOp<RMO::DivOp, RMO::MODivOp>,
        LowerCompositeBinaryElemwiseOp<RMO::ModOp, RMO::MOModOp>,
        LowerCompositeBinaryElemwiseOp<RMO::PowOp, RMO::MOPowOp>,
        LowerCompositeBinaryElemwiseOp<RMO::MinOp, RMO::MOMinOp>,
        LowerCompositeBinaryElemwiseOp<RMO::MaxOp, RMO::MOMaxOp>,
        LowerCompositeBinaryComparisonOp<RMO::EqualOp, RMO::MOEqualOp>,
        LowerCompositeBinaryComparisonOp<RMO::NotEqualOp, RMO::MONotEqualOp>,
        LowerCompositeBinaryComparisonOp<RMO::GreaterOp, RMO::MOGreaterOp>,
        LowerCompositeBinaryComparisonOp<RMO::GreaterEqualOp,
                                         RMO::MOGreaterEqualOp>,
        LowerCompositeBinaryComparisonOp<RMO::AndOp, RMO::MOAndOp>,
        LowerCompositeBinaryComparisonOp<RMO::OrOp, RMO::MOOrOp>,
        LowerCompositeBinaryComparisonOp<RMO::XorOp, RMO::MOXorOp>,
        LowerBroadcastLike, LowerConv, LowerConvTranspose, LowerAvgPool,
        LowerMaxPool, LowerConcat, LowerReshape, LowerMatmul, LowerBroadcastTo,
        LowerSelect, LowerTopK, LowerShapeToTensor>(context);

    AssertCache assertCache;
    blockPatterns.add<LowerSlice>(context, assertCache);

    if (failed(applyPatternsGreedily(getOperation(), std::move(blockPatterns))))
      return signalPassFailure();
  }
};

} // namespace
