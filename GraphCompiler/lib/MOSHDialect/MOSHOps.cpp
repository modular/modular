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
// This file implements the MOSH dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// ParamToValueOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamToValueOpValue(OpAsmParser &p, TypedAttr &value,
                                            Type &resultType) {
  auto defaultType = ShapeAttr::getDimType(p.getContext());
  if (KGEN::parseColonTypeOrDefault(p, resultType, defaultType) ||
      p.parseEqual() || p.parseLess() ||
      KGEN::parseParamValue(p, value, resultType) || p.parseGreater())
    return failure();
  return success();
}

static void printParamToValueOpValue(OpAsmPrinter &p, Operation *,
                                     TypedAttr value, Type resultType) {
  auto defaultType = ShapeAttr::getDimType(value.getContext());
  KGEN::printColonTypeOrDefault(p, value.getType(), defaultType);
  p << " = <";
  KGEN::printParamValue(p, value);
  p << ">";
}

void ParamToValueOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

//===----------------------------------------------------------------------===//
// ParamFromValueOp support
//===----------------------------------------------------------------------===//

/// Also handles mosh.param.to_value, which can't be an MLIR constant op.
static TypedAttr getConstantValue(Value value) {
  TypedAttr constant;
  if (matchPattern(value, m_Constant(&constant)))
    return constant;

  if (auto paramToValue = value.getDefiningOp<ParamToValueOp>())
    return paramToValue.getValue();

  return {};
}

void M::MOSH::ParamFromValueOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), ParamDefValue());
}

LogicalResult
M::MOSH::ParamFromValueOp::canonicalize(ParamFromValueOp op,
                                        PatternRewriter &rewriter) {
  // Fold the declaration of a constant to a parameter declaration.
  if (auto constant = getConstantValue(op.getValue())) {
    rewriter.replaceOpWithNewOp<KGEN::ParamDeclareOp>(op, op.getParamDeclAttr(),
                                                      constant);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Shape ops
//===----------------------------------------------------------------------===//

// Symbolic simplification (`add(X, 0) --> X`) lives in the constructors of
// `KGEN::ParamOperatorAttr`, which only knows KGEN's built-in ops. Shape ops
// like concat are expressed as `KGEN::MLIROpAttr`, so their simplification must
// be MLIR folders.
//
// Folder results can't be materialized into `MOSH::ParamToValueOp`, so the
// canonicalizers below reuse the folders to make `--canonicalize` behave as if
// `ParamToValueOp` were ConstantLike.
ErrorOr<TypedAttr> M::MOSH::tryFoldOp(Operation *op) {
  ASSERT_STREAM(isa<MOSHDialect>(op->getDialect()), << "expected MOSH op");

  // MOSH op inputs are always parametric shapes/dims or index constants.
  SmallVector<Attribute> attrs;
  attrs.reserve(op->getNumOperands());

  for (auto input : op->getOperands()) {
    if (auto paramToValueOp = getConstantValue(input)) {
      attrs.push_back(paramToValueOp);
    } else {
      return Error("An input isn't a constant");
    }
  }

  SmallVector<OpFoldResult> foldResults;
  if (failed(op->fold(attrs, foldResults)))
    return Error("Failed to fold the op");

  if (foldResults.size() != 1)
    return Error("Multiple results from fold");

  if (auto foldResultAttr = dyn_cast<Attribute>(foldResults.front()))
    if (auto foldResultTypedAttr = dyn_cast<TypedAttr>(foldResultAttr))
      return foldResultTypedAttr;

  return Error("Fold result isn't a typed attribute");
}

// Replaces `op` with a `ParamToValueOp` holding its folded result.
static LogicalResult genericCanonicalize(Operation *op,
                                         PatternRewriter &rewriter) {
  ASSERT_STREAM(isa<MOSHDialect>(op->getDialect()), << "expected MOSH op");
  Location loc = op->getLoc();

  ErrorOr<TypedAttr> resultAttrOr = tryFoldOp(op);
  if (resultAttrOr.isError())
    return rewriter.notifyMatchFailure(loc, resultAttrOr.getError());

  TypedAttr resultAttr = resultAttrOr.get();
  if (isa<KGEN::ParamDeclRefAttr, KGEN::ParamOperatorAttr, IntegerAttr,
          KGEN::SIMDAttr, ShapeAttr>(resultAttr)) {
    rewriter.replaceOpWithNewOp<ParamToValueOp>(op, resultAttr);
    return success();
  }

  return rewriter.notifyMatchFailure(
      loc, "fold result attribute type isn't supported");
}

LogicalResult BroadcastOp::canonicalize(BroadcastOp op,
                                        PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult EqualOp::canonicalize(EqualOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult GetDimOp::canonicalize(GetDimOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult GetRankOp::canonicalize(GetRankOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult NewOp::canonicalize(NewOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult NumElementsOp::canonicalize(NumElementsOp op,
                                          PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult SliceOp::canonicalize(SliceOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  auto xShape = dyn_cast_or_null<ShapeAttr>(adaptor.getInputX());
  auto yShape = dyn_cast_or_null<ShapeAttr>(adaptor.getInputY());
  if (xShape && yShape) {
    ErrorOr<ShapeAttr> shapeOr = ShapeAttr::broadcastShapes(xShape, yShape);
    if (!shapeOr.isError())
      return *shapeOr;
  }
  return {};
}

void ConcatOp::build(OpBuilder &builder, OperationState &state,
                     TypedValue<ShapeType> input1,
                     TypedValue<ShapeType> input2) {
  return build(builder, state, {input1, input2});
}

OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getInputs().size() == 1)
    return adaptor.getInputs().front();

  SmallVector<TypedAttr> finalDims;
  for (Attribute input : adaptor.getInputs()) {
    if (auto inputDims = dyn_cast_or_null<ShapeAttr>(input))
      llvm::append_range(finalDims, inputDims.getValues());
    else
      return {};
  }
  auto shapeTy = ShapeType::get(getContext());
  return ShapeAttr::get(finalDims, shapeTy);
}

LogicalResult ConcatOp::canonicalize(ConcatOp op, PatternRewriter &rewriter) {
  if (genericCanonicalize(op, rewriter).succeeded())
    return success();

  auto inputs = op.getInputs();
  if (inputs.size() == 1) {
    rewriter.replaceAllUsesWith(op, inputs[0]);
    return success();
  }

  SmallVector<Value> newInputs;
  for (Value arg : inputs) {
    if (auto precedingOp = arg.getDefiningOp<ConcatOp>()) {
      auto preInputs = precedingOp.getInputs();
      newInputs.append(preInputs.begin(), preInputs.end());
    } else {
      newInputs.emplace_back(arg);
    }
  }

  if (llvm::equal(inputs, newInputs))
    return failure();

  rewriter.modifyOpInPlace(op,
                           [&]() { op.getInputsMutable().assign(newInputs); });
  return success();
}

TypedAttr EqualOp::asExpression(ShapeAttr shape1, ShapeAttr shape2) {
  ASSERT_STREAM(shape1.getRank() == shape2.getRank(),
                << "Shapes must have same rank but they are "
                << shape1.getRank() << " and " << shape2.getRank());
  SmallVector<TypedAttr> andOperands;
  for (auto [xDim, yDim] : llvm::zip(shape1.getValues(), shape2.getValues())) {
    andOperands.emplace_back(
        KGEN::ParamOperatorAttr::get(KGEN::POC::EQ, {xDim, yDim}));
  }
  return KGEN::ParamOperatorAttr::get(KGEN::POC::And, andOperands);
}

OpFoldResult EqualOp::fold(FoldAdaptor adaptor) {
  auto input1 = adaptor.getInput1();
  auto input2 = adaptor.getInput2();
  if (!input1 || !input2)
    return {};

  if (input1 == input2) // fast path
    return KGEN::splatIntLiteralToSIMD(true, cast<KGEN::SIMDType>(getType()));

  auto shape1 = dyn_cast_or_null<ShapeAttr>(input1);
  auto shape2 = dyn_cast_or_null<ShapeAttr>(input2);
  if (!shape1 || !shape2)
    return {};

  if (shape1.getRank() != shape2.getRank())
    return KGEN::splatIntLiteralToSIMD(false, cast<KGEN::SIMDType>(getType()));

  return asExpression(shape1, shape2);
}

TypedAttr GetDimOp::emitStaticCall(TypedAttr inputShape, int64_t axis) {
  auto indexTy = ShapeAttr::getDimType(inputShape.getContext());
  return KGEN::emitMLIROperationCall(
      GetDimOp::getOperationName(), {},
      {inputShape, IntegerAttr::get(indexTy, axis)}, indexTy);
}

OpFoldResult GetDimOp::fold(FoldAdaptor adaptor) {
  auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput());

  std::optional<int64_t> axisOr = ShapeAttr::getStaticDim(adaptor.getDim());
  if (!input || !axisOr)
    return {};

  auto normalizedAxis = normalizeAxis(*axisOr, input.getRank());
  ArrayRef<TypedAttr> values = input.getValues();
  ASSERT_STREAM((normalizedAxis >= 0) &&
                    (normalizedAxis < static_cast<int64_t>(values.size())),
                "Normalized axis must be in range [0, "
                    << values.size() << "] but it is " << normalizedAxis);
  TypedAttr value = values[normalizedAxis];
  Type resTy = getType();
  if (value.getType() == resTy)
    return value;
  return {};
}

OpFoldResult GetRankOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput()))
    return MOSH::ShapeAttr::getDim(getContext(), input.getRank());

  return {};
}

OpFoldResult NewOp::fold(FoldAdaptor adaptor) {
  SmallVector<TypedAttr> newDims;
  for (Attribute dimAttr : adaptor.getDims()) {
    if (auto dim = dyn_cast_or_null<TypedAttr>(dimAttr))
      newDims.emplace_back(dim);
    else
      return {};
  }
  auto shapeTy = ShapeType::get(getContext());
  return ShapeAttr::get(newDims, shapeTy);
}

OpFoldResult NumElementsOp::fold(FoldAdaptor adaptor) {
  if (auto inputDimAttrs = dyn_cast_or_null<ShapeAttr>(adaptor.getInput()))
    return inputDimAttrs.getSymbolicNumElements();
  return {};
}

OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  auto inputAttr = dyn_cast_or_null<ShapeAttr>(adaptor.getInput());
  std::optional<int64_t> startOr = ShapeAttr::getStaticDim(adaptor.getStart());
  std::optional<int64_t> endOr = ShapeAttr::getStaticDim(adaptor.getEnd());
  if (!inputAttr || !startOr || !endOr)
    return {};

  // Slice semantics allow `start`/`end` outside [0, rank].
  // TODO: is this really needed? Maybe we should just enforce this in IR.
  int64_t rank = inputAttr.getRank();
  int64_t end = clampAxis(*endOr, rank);
  int64_t start = clampAxis(*startOr, rank);

  if (start > end) {
    ASSERT_STREAM(false, << "start index (" << start
                         << ") must be less than end index (" << end << ")");
    return {};
  }

  auto it = inputAttr.getValues().begin();
  SmallVector<TypedAttr> slicedDims(it + start, it + end);
  auto shapeTy = ShapeType::get(getContext());
  return ShapeAttr::get(slicedDims, shapeTy);
}

using KGEN::parseIndexParamValue;
using KGEN::parseStringParam;
using KGEN::printIndexParamValue;
using KGEN::printStringParam;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOSHDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MOSHDialect/MOSH.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GraphCompiler/MOSHDialect/MOSH.cpp.inc"
