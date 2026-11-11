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
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/Permutation.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

using namespace M;
using namespace MO;
SmallVector<ConstantOp> M::MO::tryGetConstantOpsOfRank(TensorValue val,
                                                       size_t rank) {
  SmallVector<ConstantOp> constantOps;
  Operation *inputOp = val.getDefiningOp();

  // Handle identity chains.
  auto isIdentityLike = [&](Operation *op) {
    if (isa_and_nonnull<RebindOp>(op))
      return true;
    if (isa_and_nonnull<mlir::UnrealizedConversionCastOp>(op))
      return true;
    if (isa_and_nonnull<ReshapeOp>(op)) {
      // After a ReshapeOp, accept constants of any rank.
      rank = std::numeric_limits<size_t>::max();
      return true;
    }
    return false;
  };

  // TODO(#37571)
  while (isIdentityLike(inputOp))
    inputOp = inputOp->getOperand(0).getDefiningOp();

  // Handle concatenations of constant tensors.
  if (auto concatOp = dyn_cast_if_present<ConcatOp>(inputOp)) {
    for (Value operand : concatOp.getInputs()) {
      TensorValue tensorOperand = cast<TensorValue>(operand);
      SmallVector<ConstantOp> operandConstantOps =
          tryGetConstantOpsOfRank(tensorOperand, rank);
      if (operandConstantOps.empty())
        return {};
      llvm::append_range(constantOps, operandConstantOps);
    }
    return constantOps;
  }

  auto constantOp = dyn_cast_if_present<ConstantOp>(inputOp);
  if (!constantOp)
    return {};

  MO::TensorType type = constantOp.getType();
  if (rank != std::numeric_limits<size_t>::max() &&
      (!type.hasRank() || type.getRank() != rank))
    return {};

  constantOps.push_back(constantOp);

  return constantOps;
}

std::optional<float> M::MO::tryGetConstantFloat(TensorValue val) {
  if (auto maybeFloatBlob = tryGetConstantOfRank<float>(val, 0))
    return maybeFloatBlob.value().front();
  return std::nullopt;
}

std::optional<double> M::MO::tryGetConstantDouble(TensorValue val) {
  if (auto maybeDoubleBlob = tryGetConstantOfRank<double>(val, 0))
    return maybeDoubleBlob.value().front();
  return std::nullopt;
}

std::optional<int64_t> M::MO::tryGetConstantInt(TensorValue val) {
  if (auto maybeIntBlob = tryGetConstantOfRank<int64_t>(val, 0))
    return maybeIntBlob.value().front();
  return std::nullopt;
}

std::optional<bool> M::MO::tryGetConstantBool(TensorValue val) {
  if (auto maybeIntBlob = tryGetConstantOfRank<int64_t>(val, 0))
    return maybeIntBlob.value().front();
  return std::nullopt;
}

std::optional<APFloat> M::MO::tryGetConstantAPFloat(TensorValue val) {
  if (auto maybeBlob = tryGetConstantOfRank<APFloat>(val, 0))
    return maybeBlob.value().front();
  return std::nullopt;
}

std::optional<APInt> M::MO::tryGetConstantAPInt(TensorValue val) {
  if (auto maybeBlob = tryGetConstantOfRank<APInt>(val, 0))
    return maybeBlob.value().front();
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
M::MO::tryGetConstantIntVector(TensorValue val) {
  if (val.getType().getElementType().isIntOrIndex())
    return tryGetConstantOfRank<int64_t>(val, 1);
  return std::nullopt;
}

std::optional<SmallVector<bool>>
M::MO::tryGetConstantBoolVector(TensorValue val) {
  if (val.getType().getElementType().isInteger(1))
    return tryGetConstantOfRank<bool>(val, 1);
  return std::nullopt;
}

/// Returns the `MOSH::ShapeType` value that `val` (an `MO::TensorType`) was
/// created from, or null.
static Value tryGetShapeValue(Value val) {
  // Skip any number of rebinds.
  while (auto rebindOp = val.getDefiningOp<RebindOp>())
    val = rebindOp.getInput();

  if (auto toTensorOp = val.getDefiningOp<Shape::ToTensorOp>())
    return toTensorOp.getInput();

  return {};
}

/// Returns a `ShapeAttr` of the constant integers that created `val` (an
/// `MO::TensorType`), or null.
static ShapeAttr tryGetConstantShapeAttr(Value val) {
  // Skip at most a single intermediate rebind
  if (auto rebindOp = val.getDefiningOp<RebindOp>())
    val = rebindOp.getInput();

  if (auto tensorVal = dyn_cast<TensorValue>(val)) {
    if (auto maybeIntVec = tryGetConstantIntVector(tensorVal)) {
      MLIRContext *ctx = val.getContext();
      auto shapeType = ShapeType::get(ctx);
      return ShapeAttr::get(maybeIntVec.value(), shapeType);
    }
  }
  return {};
}

ShapeAttr M::MO::tryGetShapeAttr(Value val) {
  if (Value shapeVal = tryGetShapeValue(val)) {
    if (auto paramToValueOp = shapeVal.getDefiningOp<MOSH::ParamToValueOp>())
      return dyn_cast<ShapeAttr>(paramToValueOp.getValue());

    return {};
  }

  return tryGetConstantShapeAttr(val);
}

TypedAttr M::MO::tryGetDimAttr(Value val) {
  auto tensorVal = dyn_cast<TensorValue>(val);
  if (!tensorVal)
    return {};

  if (TypedAttr param = getParamAttrThrough<IndexToTensorOp>(tensorVal))
    return param;

  std::optional<int64_t> maybeConst = tryGetConstantInt(tensorVal);
  if (!maybeConst.has_value())
    return {};

  return MOSH::ShapeAttr::getDim(val.getContext(), *maybeConst);
}

void M::MO::synchronizeMetadataUpdates(GraphOp op) {
  MLIRContext *ctx = op.getContext();
  TypeRange newArgTypes = op.getBody().getArgumentTypes();
  SmallVector<Type> newResTypes;

  for (auto [newOutTy, funcOutTy] :
       llvm::zip(op.getOutputOp().getOperandTypes(), op.getResultTypes())) {
    // By default, assume nothing changed.
    Type newResTy = funcOutTy;

    if (auto funcOutTensorTy = dyn_cast<TensorType>(funcOutTy)) {
      auto newOutTensorTy = cast<TensorType>(newOutTy);
      newResTy = TensorType::get(
          funcOutTensorTy.getShape(), funcOutTensorTy.getDtype(),
          newOutTensorTy.getDeviceRef(), newOutTensorTy.getMetadata());
    }

    if (auto funcOutBufferTy = dyn_cast<BufferType>(funcOutTy)) {
      auto newOutBufferTy = cast<BufferType>(newOutTy);

      newResTy = BufferType::get(
          funcOutBufferTy.getShape(), funcOutBufferTy.getDtype(),
          newOutBufferTy.getDeviceRef(), newOutBufferTy.getMetadata());
    }

    if (auto funcOutListTy = dyn_cast<ListType>(funcOutTy)) {
      auto newOutListTy = cast<ListType>(newOutTy);
      newResTy = ListType::get(funcOutListTy.getContext(),
                               newOutListTy.getElementType());
    }

    newResTypes.push_back(newResTy);
  }

  auto funcType = FunctionType::get(ctx, newArgTypes, newResTypes);
  FuncTypeGeneratorType signature =
      KGEN::FuncTypeGeneratorType::remapToFuncTypeGenerator(
          op.getInputParameters(), funcType);
  op.setSignature(signature);
  op.setFunctionType(funcType);
}

SmallVector<OpOperand *> CallOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getOperandsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

KGEN::ParamDeclAttr GraphOp::getNewDimParam(const Twine &prefix) {
  auto dimType = MOSH::ShapeAttr::getDimType(getContext());
  return getNewParamOfType(prefix, dimType);
}

KGEN::ParamDeclAttr GraphOp::getNewParamOfType(const Twine &prefix, Type type) {
  return KGEN::ParamDeclAttr::get(getNewParamName(prefix), type);
}

std::string GraphOp::getNewParamName(const Twine &prefix) {
  // TODO(#8829) avoid duplication with existing parameters in the input graph
  int64_t newCounter = getCounter() + 1;
  setCounter(newCounter);
  return (prefix + Twine(newCounter)).str();
}

TargetInfoAttr GraphOp::getDeviceTargetInfo() {
  return dyn_cast_if_present<TargetInfoAttr>(
      getOperation()->getAttr(kGraphTargetInfoAttrName));
}

Block *IfOp::getThenBlock() { return &getThenRegion().front(); }

Block *IfOp::getElseBlock() { return &getElseRegion().front(); }

OperandRange
WhileOp::getEntrySuccessorOperands(mlir::RegionSuccessor successor) {
  assert(successor.getSuccessor() == &getCondRegion() &&
         "WhileOp is expected to branch only to the first region");
  return getOperands();
}

MutableArrayRef<OpOperand> WhileOp::getInitsMutable() {
  return getInputsMutable();
}

std::optional<MutableArrayRef<OpOperand>> WhileOp::getYieldedValuesMutable() {
  return getBodyTerminator().getOperandsMutable();
}

Block::BlockArgListType WhileOp::getRegionIterArgs() {
  return getCondRegion().getArguments();
}

SmallVector<Region *> WhileOp::getLoopRegions() {
  return {&getCondRegion(), &getBodyRegion()};
}

Block *WhileOp::getCondBlock() { return &getCondRegion().front(); }

Block *WhileOp::getBodyBlock() { return &getBodyRegion().front(); }

Block::BlockArgListType WhileOp::getCondArguments() {
  return getCondRegion().getArguments();
}

Block::BlockArgListType WhileOp::getBodyArguments() {
  return getBodyRegion().getArguments();
}

SmallVector<OpOperand *> WhileOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getInputsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

MutableOperandRange
WhileConditionOp::getMutableSuccessorOperands(mlir::RegionSuccessor successor) {
  assert((successor.isOperation() ||
          successor.getSuccessor() == &getParentOp().getBodyRegion()) &&
         "condition op can only exit the loop or branch to the after"
         "region");
  // Pass all operands except the condition to the successor region.
  return getArgsMutable();
}

SmallVector<OpOperand *> ParallelOp::getInChainsMutable() {
  SmallVector<OpOperand *> result;
  if (getInChain()) {
    auto mutableRange = getInChainMutable();
    for (auto &operand : mutableRange)
      result.push_back(&operand);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// ReduceAddOp, ReduceMinOp, ReduceMaxOp

TypedAttr M::MO::emitStaticCallImpl(MLIRContext *ctx,
                                    ArrayRef<TypedAttr> operands,
                                    const Twine &message) {
  SmallVector<std::pair<StringAttr (*)(mlir::OperationName), Attribute>>
      msgAttrNameFuncAndArg = {
          {AssertOp::getMessageAttrName,
           StringAttr::get(message, KGEN::StringType::get(ctx))}};
  auto chainTy = ChainType::get(ctx);
  return KGEN::emitMLIROperationCall(AssertOp::getOperationName(),
                                     msgAttrNameFuncAndArg, operands, chainTy);
}

TypedAttr AssertOp::emitStaticCall(TypedAttr cond, const Twine &message) {
  return emitStaticCallImpl(cond.getContext(), {cond}, message);
}

SmallVector<OpOperand *> CustomOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getOperandsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

void MODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MODialect/MO.cpp.inc"
      >();
}
