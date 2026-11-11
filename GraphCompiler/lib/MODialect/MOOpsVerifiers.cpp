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
// This file implements the verifiers of the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
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
/// Verifies that the first block of the given `region` is terminated by a
/// TerminatorTy. Reports errors on the given operation if it is not the case.
template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(Operation *op, Region &region,
                                           StringRef errorMessage) {
  Operation *terminatorOperation = nullptr;
  if (!region.empty() && !region.front().empty()) {
    terminatorOperation = &region.front().back();
    if (auto yield = dyn_cast_if_present<TerminatorTy>(terminatorOperation))
      return yield;
  }
  auto diag = op->emitOpError(errorMessage);
  if (terminatorOperation)
    diag.attachNote(terminatorOperation->getLoc()) << "terminator here";
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult
CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto graphOp =
      symbolTable.lookupNearestSymbolFrom<GraphOp>(*this, getCalleeAttr());
  if (!graphOp)
    return emitError("'") << getCalleeAttr()
                          << "' does not reference a valid graph";

  if (!graphOp.getIsSubgraph())
    return emitError() << "Only subgraphs can be called";

  auto fnType = graphOp.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitError() << "Subgraph " << graphOp.getName()
                       << " has wrong number of arguments";

  auto emitErrorFn = [&](size_t idx, Type graphArgType, Type operandType) {
    auto diag = emitError() << "Subgraph " << graphOp.getName()
                            << " has wrong type for argument " << idx
                            << " (function type: " << graphArgType
                            << ", operand type: " << operandType << ")";
    diag.attachNote(graphOp.getLoc()) << "graph op here";
    return failure();
  };

  for (auto [idx, graphArgType, operandType] : llvm::enumerate(
           graphOp.getFunctionType().getInputs(), getOperandTypes())) {
    if (isa<TensorType>(graphArgType) && isa<TensorType>(operandType)) {
      auto graphArgTensorType = cast<TensorType>(graphArgType);
      auto operandTensorType = cast<TensorType>(operandType);
      if (graphArgTensorType.cloneWithoutLayout() !=
          operandTensorType.cloneWithoutLayout()) {
        return emitErrorFn(idx, graphArgType, operandType);
      }
    } else if (graphArgType != operandType) {
      return emitErrorFn(idx, graphArgType, operandType);
    }
  }

  return success();
}

LogicalResult CallOp::verify() {
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();

  // Add this logic here to ensure eager verification of call operations
  // at build time in the Graph API.
  mlir::SymbolTableAnalysis symTabAnalysis(moduleOp);
  return verifySymbolUses(symTabAnalysis.getSymbolTables());
}

LogicalResult TransferOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType resultType = getResult().getType();
  TensorType resTypeWithInputDevice =
      resultType.cloneWith(inputType.getDeviceRef());
  if (resTypeWithInputDevice != inputType) {
    return emitError()
           << "expected result type to only differ from input type in device ("
           << inputType << " vs " << resultType << ").";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StableCreateOp
//===----------------------------------------------------------------------===//

LogicalResult StableCreateOp::verify() {
  if (!getResult().getType().isStable())
    return emitOpError("result must be marked `stable`");
  return success();
}

LogicalResult GraphOp::verify() {
  KGEN::ParameterCollector::Analysis cache;
  KGEN::ParameterUseDefGraph useDefGraph(getRegion());

  // check that no expressions in the input parameters
  mlir::AttrTypeWalker walker;
  bool found = false;
  walker.addWalk([&](KGEN::ParamOperatorAttr attr) {
    found = true;
    return WalkResult::interrupt();
  });
  for (auto ty : getArgumentTypes()) {
    walker.walk(ty);
    if (found)
      break;
  }
  if (found) {
    return emitError(
        "Expressions are not (currently) allowed in function signatures");
  }

  return useDefGraph.verify(nullptr, cache);
}

LogicalResult OutputOp::verify() {
  // ODS checks the parent is a GraphOp.
  auto graph = cast<GraphOp>(getOperation()->getParentOp());
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParameters();
  ArrayRef<TypedAttr> defs = this->getParameters();
  if (decls.size() != defs.size()) {
    return emitError(
        "output bindings and result parameters must have the same size");
  }

  // Build a mapping from result param references to their binding attribute,
  // example:
  //  { (D1, 42), (Sh0, [D0, 10]) }
  DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> resParamRefToBindingAttr;
  for (auto [resParamDecl, bindingParam] : llvm::zip(decls, defs)) {
    resParamRefToBindingAttr.insert(
        {KGEN::ParamDeclRefAttr::get(resParamDecl), bindingParam});
  }

  // Substitute param references inside graph result types using the map, which
  // gives us the expected output operands type.
  SmallVector<Type> expectedOutputOperandTypes;
  for (Type ty : graph.getResultTypes()) {
    if (auto tensorTy = dyn_cast<TensorType>(ty)) {
      auto newShape = cast<MOSH::ShapeAttr>(tensorTy.getShape().replace(
          [&resParamRefToBindingAttr](KGEN::ParamDeclRefAttr paramRef) {
            TypedAttr resolvedAttr = resParamRefToBindingAttr.lookup(paramRef);
            return resolvedAttr ? resolvedAttr : paramRef;
          }));
      expectedOutputOperandTypes.emplace_back(tensorTy.cloneWith(newShape));
    } else {
      expectedOutputOperandTypes.emplace_back(ty);
    }
  }

  return checkOperandTypes(getOperation(), expectedOutputOperandTypes);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  Operation *parentOp = getOperation()->getParentOp();
  if (auto ifOp = dyn_cast<IfOp>(parentOp)) {
    ArrayRef<KGEN::ParamDeclAttr> decls = ifOp.getOutputParamDecls();
    ArrayRef<TypedAttr> defs = this->getParameters();
    if (decls.size() != defs.size()) {
      return emitOpError(
                 "[YieldOp::walkDefinitions] yield parameter bindings (")
             << defs.size() << ") and if output parameters (" << decls.size()
             << ") must have the same size";
    }
  }
  return success();
}

LogicalResult WhileOp::verify() {
  auto beforeTerminator = verifyAndGetTerminator<WhileConditionOp>(
      *this, getCondRegion(),
      "expects the 'before' region to terminate with 'mo.while.condition'");
  if (!beforeTerminator)
    return failure();

  auto afterTerminator = verifyAndGetTerminator<YieldOp>(
      *this, getBodyRegion(),
      "expects the 'after' region to terminate with 'mo.yield'");
  return success(afterTerminator != nullptr);
}

//===----------------------------------------------------------------------===//
// Bundle / Unbundle ops
//===----------------------------------------------------------------------===//

LogicalResult TensorBundleOp::verify() {
  BundleType resultType = getResult().getType();
  if (getInputs().size() != resultType.getNumElements())
    return emitOpError("input count (")
           << getInputs().size() << ") must match bundle element count ("
           << resultType.getNumElements() << ")";
  for (auto [i, input, expected] :
       llvm::enumerate(getInputs(), resultType.getElementTypes()))
    if (input.getType() != expected)
      return emitOpError("input #")
             << i << " type " << input.getType()
             << " does not match bundle element type " << expected;
  return success();
}

LogicalResult TensorUnbundleOp::verify() {
  BundleType inputType = getInput().getType();
  if (getOutputs().getTypes().size() != inputType.getNumElements())
    return emitOpError("output count (")
           << getOutputs().getTypes().size()
           << ") must match bundle element count ("
           << inputType.getNumElements() << ")";
  for (auto [i, output, expected] :
       llvm::enumerate(getOutputs().getTypes(), inputType.getElementTypes()))
    if (output != expected)
      return emitOpError("output #")
             << i << " type " << output
             << " does not match bundle element type " << expected;
  return success();
}
namespace {

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

// Returns success if `op` is allowed inside `mo.parallel`, else emits a
// diagnostic. Allowed ops are:
//
//  - Computational ops that lower to a MOGG kernel via `MOToMOGG` (caught
//    here by exclusion from the rejection arms below).
//  - Per-launch entry points for bundled collectives (`mo.bundled.expand`,
//    `mo.bundled.allreduce.sum`), tagged with `MO_BundledCollectiveInterface`.
LogicalResult verifyParallelBodyOp(Operation &op, ParallelOp parallelOp) {
  // Bundled collective ops are only supported within parallel regions.
  // They may contain mutable signal buffers.
  if (isa<MOBundledCollectiveInterface>(&op))
    return success();

  // HACK (see GEX-3701):
  // Prior to HoistParameterExpressionsPass, rebinds may exist in the parallel
  // body. We currently assume they will be moved out before lowering.
  if (isa<MO::RebindOp>(&op))
    return success();

  // mo.buffer is not supported in parallel regions, except signal buffers in
  // bundled collectives and conditionally-in-place ops (`mo.custom`) that are
  // not in-place, which lower to kernels like any computational op.
  if (isa<MOMutableOpInterface>(&op)) {
    auto cond = dyn_cast<MOConditionallyInPlaceInterface>(&op);
    if (!cond || cond.isInPlace())
      return op.emitOpError()
             << "is not allowed inside mo.parallel: mutable buffer ops and "
                "multi-device collectives are not supported; see GEX-3704";
  }

  // `MO_PreservedDuringKernelLowering` tags ops the kernel-lowering pass leaves
  // in place (constants, control flow, transfers, host-only, bundle /
  // unbundle, nested mo.parallel, ...). None are allowed in the body yet.
  if (isa<MO::PreservedDuringKernelLowering>(&op))
    return op.emitOpError()
           << "is not allowed inside mo.parallel: only kernels and bundled "
              "collectives are currently supported in the body; see "
              "GEX-3701, GEX-3704, GEX-3705";

  return success();
}
} // namespace

/// Returns true if device labels match; IDs may differ.
static bool deviceLabelsMatch(DeviceRefAttr a, DeviceRefAttr b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  return a.getLabel() == b.getLabel();
}

/// Returns true if two types match ignoring device ID.
static bool typesMatchIgnoringDeviceId(Type a, Type b) {
  if (auto tensorA = dyn_cast<TensorType>(a)) {
    auto tensorB = dyn_cast<TensorType>(b);
    if (!tensorB)
      return false;
    return tensorA.getShape() == tensorB.getShape() &&
           tensorA.getDtype() == tensorB.getDtype() &&
           deviceLabelsMatch(tensorA.getDeviceRef(), tensorB.getDeviceRef());
  }
  if (auto bufferA = dyn_cast<BufferType>(a)) {
    auto bufferB = dyn_cast<BufferType>(b);
    if (!bufferB)
      return false;
    return bufferA.getShape() == bufferB.getShape() &&
           bufferA.getDtype() == bufferB.getDtype() &&
           deviceLabelsMatch(bufferA.getDeviceRef(), bufferB.getDeviceRef());
  }
  return a == b;
}

LogicalResult ParallelOp::verify() {
  auto inputs = getInputs();
  SmallVector<DeviceRefAttr> launchDevices = getLaunchDeviceList();

  // Launch count comes from launchDevices (the source of truth).  When
  // input bundles are present they must agree on width.
  size_t numLaunches = launchDevices.size();
  if (numLaunches == 0)
    return emitOpError("launchDevices must have at least one entry");

  // Each input bundle's elements must match launchDevices: same length and
  // same per-launch device.
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    auto bType = cast<BundleType>(input.getType());
    if (bType.getNumElements() != numLaunches)
      return emitOpError("bundle input #")
             << idx << " has " << bType.getNumElements()
             << " elements, expected " << numLaunches;
    for (auto [i, elemType] : llvm::enumerate(bType.getElementTypes())) {
      auto tensorTy = dyn_cast<TensorType>(elemType);
      if (!tensorTy)
        continue;
      DeviceRefAttr bundleDevice = tensorTy.getDeviceRef();
      if (bundleDevice != launchDevices[i])
        return emitOpError("bundle input #")
               << idx << " element #" << i << "'s device " << bundleDevice
               << " does not match launchDevices[" << i
               << "] = " << launchDevices[i];
    }
  }

  if (hasBuffers() != hasChain())
    return emitOpError("buffers and chain must be both present or both absent");

  if (hasBuffers()) {
    if (getBuffers().size() != numLaunches)
      return emitOpError("buffers count (")
             << getBuffers().size() << ") must match number of launches ("
             << numLaunches << ")";
  }

  Block &body = getBodyRegion().front();
  size_t expectedArgs = inputs.size() + (hasChain() ? 1 : 0);
  if (body.getNumArguments() != expectedArgs)
    return emitOpError("body must have ")
           << expectedArgs << " block argument(s) (one per bundle input"
           << (hasChain() ? " plus a trailing chain arg" : "") << "), got "
           << body.getNumArguments();

  // Chain threading contract: a chained parallel takes its in-chain through a
  // trailing !mo.chain block argument and yields the threaded out-chain as the
  // terminator's last operand. A chainless parallel does neither.
  auto yield = cast<YieldOp>(body.getTerminator());
  if (hasChain()) {
    if (!isa<ChainType>(body.getArguments().back().getType()))
      return emitOpError(
          "trailing body block argument must be !mo.chain when the parallel "
          "is chained");
    if (yield.getNumOperands() == 0 ||
        !isa<ChainType>(yield.getOperands().back().getType()))
      return emitOpError(
          "mo.yield must produce a trailing !mo.chain when the parallel is "
          "chained");
  } else {
    for (Value operand : yield.getOperands())
      if (isa<ChainType>(operand.getType()))
        return emitOpError("chainless parallel must not yield a chain");
  }

  // GEX-3701 restricts the body to kernels (or ops that will become kernels),
  // `mo.bundled.expand`, and a small whitelist of view/chain ops.
  for (Operation &op : body.without_terminator()) {
    if (failed(verifyParallelBodyOp(op, *this)))
      return failure();
  }

  // Ensure that the types within each input bundle match the type of the
  // corresponding block argument.
  for (auto [group, bundleInput] : llvm::enumerate(inputs)) {
    Type blockArgType = body.getArgument(group).getType();
    auto bundleType = cast<BundleType>(bundleInput.getType());
    for (auto [launch, elemType] :
         llvm::enumerate(bundleType.getElementTypes())) {
      if (!typesMatchIgnoringDeviceId(elemType, blockArgType))
        return emitOpError("bundle input #")
               << group << " element #" << launch << " type " << elemType
               << " does not match block argument type " << blockArgType;
    }
  }

  auto yieldOp = cast<YieldOp>(body.getTerminator());
  size_t numYields = yieldOp.getNumOperands();
  // The yielded chain (when chained) maps to the trailing chain result, not a
  // bundle result.
  size_t numBundleYields = numYields - (hasChain() ? 1 : 0);
  if (numBundleYields < 1)
    return emitOpError("yield must produce at least one bundle result");

  size_t expectedResults = numBundleYields + (hasChain() ? 1 : 0);
  if (getNumResults() != expectedResults)
    return emitOpError("expected ")
           << expectedResults << " results, got " << getNumResults();

  for (size_t yieldIdx = 0; yieldIdx < numBundleYields; ++yieldIdx) {
    auto resultType = dyn_cast<BundleType>(getResult(yieldIdx).getType());
    if (!resultType)
      return emitOpError("result #")
             << yieldIdx << " must be an mo.bundle type";
    if (resultType.getNumElements() != numLaunches)
      return emitOpError("result bundle #")
             << yieldIdx << " has " << resultType.getNumElements()
             << " elements, expected " << numLaunches;
    for (auto [i, elemType] : llvm::enumerate(resultType.getElementTypes())) {
      auto tensorTy = dyn_cast<TensorType>(elemType);
      if (!tensorTy)
        continue;
      DeviceRefAttr bundleDevice = tensorTy.getDeviceRef();
      if (bundleDevice != launchDevices[i])
        return emitOpError("result bundle #")
               << yieldIdx << " element #" << i << "'s device " << bundleDevice
               << " does not match launchDevices[" << i
               << "] = " << launchDevices[i];
    }
  }

  return success();
}

LogicalResult SequenceOp::verify() {
  Block &body = getBodyRegion().front();

  // Inputs map 1:1 to block arguments (plain variadic list, no bundles).
  if (body.getNumArguments() != getInputs().size())
    return emitOpError("body must have ")
           << getInputs().size() << " block argument(s) (one per input), got "
           << body.getNumArguments();
  for (auto [idx, input] : llvm::enumerate(getInputs())) {
    if (body.getArgument(idx).getType() != input.getType())
      return emitOpError("input #") << idx << " type " << input.getType()
                                    << " does not match block argument type "
                                    << body.getArgument(idx).getType();
  }

  // Yield operands map 1:1 to results.
  auto yieldOp = cast<YieldOp>(body.getTerminator());
  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError("yield has ")
           << yieldOp.getNumOperands() << " operand(s) but op has "
           << getNumResults() << " result(s)";
  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    if (yielded.getType() != getResult(idx).getType())
      return emitOpError("yield operand #")
             << idx << " type " << yielded.getType()
             << " does not match result type " << getResult(idx).getType();
  }

  // Nested `mo.sequence` is unsupported: the side-stream device-context
  // override installed during MOGG->MGP lowering is single-level per device.
  if (getOperation()->getParentOfType<MO::SequenceOp>())
    return emitOpError("nested `mo.sequence` is unsupported");

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastShapeOp::verify() {
  TensorType inputXType = getInputX().getType();
  TensorType inputYType = getInputY().getType();
  TensorType resultType = getResult().getType();
  // For dynamically ranked/shaped result, we defer to shape inference
  if (!resultType.hasStaticShape())
    return success();

  bool inputXIsStatic = inputXType.hasStaticShape();
  bool inputYIsStatic = inputYType.hasStaticShape();
  // Note that inputs and output are rank-1 Shape tensors
  int64_t inputXRank = inputXIsStatic ? inputXType.getIntDim(0) : 0;
  int64_t inputYRank = inputYIsStatic ? inputYType.getIntDim(0) : 0;
  int64_t resultRank = resultType.getIntDim(0);

  if (inputXIsStatic && inputXRank > resultRank) {
    return emitError("result rank must be no less than first input rank if "
                     "both are statically ranked");
  }

  if (inputYIsStatic && inputYRank > resultRank) {
    return emitError("result rank must be no less than second input rank if "
                     "both are statically ranked");
  }

  if (inputXIsStatic && inputYIsStatic &&
      resultRank != std::max(inputXRank, inputYRank)) {
    return emitError("output rank must be the max between 2 input ranks, if "
                     "all are statically ranked");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

/// Check if the input type can be broadcasted to the result type, assuming both
/// have static ranks.
static ErrorOrSuccess verifyBroadcastStaticRanks(TensorType inputTy,
                                                 TensorType resTy) {
  if (inputTy.hasRank() && resTy.hasRank()) {
    if (inputTy.getRank() > resTy.getRank())
      return Error("result rank must be at least the input rank");

    ArrayRef<TypedAttr> fromDims = inputTy.getShape().getValues();
    ArrayRef<TypedAttr> toDims = resTy.getShape().getValues();
    for (auto [fromDim, toDim] :
         llvm::zip(llvm::reverse(fromDims), llvm::reverse(toDims))) {
      auto fromIntDim = tryGetConstantInt(fromDim);
      auto toIntDim = tryGetConstantInt(toDim);
      if (fromIntDim && toIntDim) {
        // We can only assert anything if both dimensions are known constants.
        int64_t inDim = *fromIntDim;
        if (inDim > 1 && (inDim != *toIntDim))
          return Error("input shape not broadcastable to result shape");
      }
    }
  }
  return success();
}

LogicalResult BroadcastToOp::verify() {
  TensorType resTy = getType();
  TensorType newShapeTy = getNewShape().getType();

  if (!newShapeTy.hasStaticShape()) {
    if (!resTy.hasRank())
      return success();
    return emitError(
        "result must be dynamically ranked if new shape has unknown size");
  }

  if (!resTy.hasRank()) {
    return emitError(
        "result must be statically ranked if new shape has static size");
  }

  if ((int64_t)resTy.getRank() != newShapeTy.getIntDim(0))
    return emitError("new shape size and output rank must match");

  if (auto err = verifyBroadcastStaticRanks(getInput().getType(), resTy))
    return emitError(err.getError());

  return success();
}

//===----------------------------------------------------------------------===//
// Static::BroadcastToOp
//===----------------------------------------------------------------------===//

LogicalResult Static::BroadcastToOp::verify() {
  if (auto err = verifyBroadcastStaticRanks(getInput().getType(), getType()))
    return emitError(err.getError());
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto valType = cast<ShapedType>(getValue().getType()).getElementType();
  auto resType = getResult().getType().getElementType();
  // TODO(#28523): Restrict the value type of booleans to i8.
  if (resType.isInteger(1)) {
    if (!(valType.isInteger(1) || valType.isInteger(8) ||
          valType.isInteger(32) || valType.isInteger(64))) {
      return emitOpError("for boolean tensors, the element type of the value "
                         "must be an integer type, but got ")
             << valType;
    }
    return success();
  }
  if (valType != resType) {
    return emitOpError("element type of the value (")
           << valType << ") must match the element type of the result ("
           << resType << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantScalarOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOp::verify() {
  auto scalarDType = cast<MO::ScalarType>(getType()).getDType();
  if (auto value = dyn_cast<IntegerAttr>(getValue())) {
    if (!isa<IntegerType>(scalarDType))
      return emitError(
          "mo.constant.scalar expected the dtype to be an integer");
    if (value.getType().getIntOrFloatBitWidth() !=
        scalarDType.getIntOrFloatBitWidth())
      return emitError("mo.constant.scalar expected bitwidth of the dtype and "
                       "attribute to match");
  } else if (auto value = dyn_cast<FloatAttr>(getValue())) {
    if (!isa<FloatType>(scalarDType))
      return emitError("mo.constant.scalar expected the dtype to be a float");
    if (value.getType().getIntOrFloatBitWidth() !=
        scalarDType.getIntOrFloatBitWidth())
      return emitError("mo.constant.scalar expected bitwidth of the dtype and "
                       "attribute to match");
  }
  return success();
}

LogicalResult BufferCreateOp::verify() {
  Attribute initValue = getInitValueAttr();
  if (!initValue)
    return success();

  Type bufferElementType =
      castFromTensorLike(getResult().getType()).getElementType();

  // The init value is a single-element elements attribute (the form the Python
  // API emits) or a scalar typed attribute.
  Type valueElementType;
  if (auto elts = dyn_cast<ElementsAttr>(initValue)) {
    if (elts.getNumElements() != 1)
      return emitError(
          "mo.buffer.create initValue must be a single scalar element");
    valueElementType = elts.getElementType();
  } else if (auto typed = dyn_cast<TypedAttr>(initValue)) {
    valueElementType = typed.getType();
  } else {
    return emitError("mo.buffer.create initValue must be a typed scalar or "
                     "elements attribute");
  }

  if (isa<FloatType>(bufferElementType) != isa<FloatType>(valueElementType) ||
      bufferElementType.getIntOrFloatBitWidth() !=
          valueElementType.getIntOrFloatBitWidth())
    return emitError("mo.buffer.create initValue element type must match the "
                     "buffer's element type");
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs < 1)
    return emitError("concat must have at least 1 input tensor");

  SmallVector<TensorValue> tensors;
  for (Value input : getInputs())
    tensors.emplace_back(cast<TensorValue>(input));

  tensors.emplace_back(getResult());
  if (failed(::Impl::verifyMatchingRanks(tensors))) {
    return emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  auto elementTypes = llvm::map_range(
      tensors, [](TensorValue val) { return val.getType().getElementType(); });

  if (!llvm::all_equal(elementTypes)) {
    return emitError("input and output element types must match");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

/// Verify that the sum of `outputs` shapes along `splitAxis` matches the input
/// shape, and that the other dimensions match.
static ErrorOrSuccess verifyMatchingSplitShapes(ArrayRef<TensorValue> outputs,
                                                TensorType inputTy,
                                                int64_t splitAxis) {
  int64_t splitDimSum = 0;
  for (auto res : outputs) {
    TensorType resTy = res.getType();
    for (auto [i, dimSize] : llvm::enumerate(resTy.getIntDims())) {
      if ((int64_t)i == splitAxis) {
        splitDimSum += dimSize;
      } else if (dimSize != inputTy.getIntDim(i)) {
        return Error(
            "input and output static shapes must match except split axis");
      }
    }
  }

  if (splitDimSum != inputTy.getIntDim(splitAxis)) {
    return Error(
        "sum of output shapes along split axis must match input shape");
  }

  return success();
}

LogicalResult SplitOp::verify() {
  SmallVector<TensorValue> tensors;
  tensors.emplace_back(getInput());
  for (Value output : getResults())
    tensors.emplace_back(cast<TensorValue>(output));

  if (failed(::Impl::verifyMatchingRanks(tensors))) {
    return emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  TensorType inputTy = getInput().getType();
  TensorType resultTy = cast<TensorType>(getResults().front().getType());
  if (inputTy.getDtype() != resultTy.getDtype())
    return emitError("input and output element types must match");

  // `axis` is a compile-time `index` attribute.
  if (inputTy.hasStaticShape()) {
    int64_t axis = getAxisAttr().getInt();
    int64_t inputRank = inputTy.getRank();
    int64_t normalizedAxis = normalizeAxis(axis, inputRank);
    if ((normalizedAxis < 0) || (inputRank <= normalizedAxis))
      return emitError("split axis must be in range [-rank, rank)");

    if (resultTy.hasStaticShape()) {
      ErrorOrSuccess result = verifyMatchingSplitShapes(
          ArrayRef<TensorValue>(tensors).drop_front(), inputTy, normalizedAxis);
      if (result.isError())
        return emitError(result.getError());
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verify() {
  TensorType paramsType = getInput().getType();
  TensorType indicesType = getIndices().getType();
  TensorType resultType = getResult().getType();

  if (paramsType.getDtype() != resultType.getDtype())
    return emitError("input and output must have the same element type");

  // Nothing we can say if either input or indices is dynamically ranked
  if (!paramsType.hasRank() || !indicesType.hasRank())
    return success();

  if (!resultType.hasRank()) {
    return emitError(
        "result must be ranked if both param and indices are ranked");
  }

  if (paramsType.getRank() - 1 + indicesType.getRank() !=
      resultType.getRank()) {
    return emitError("result rank must be 1 less than the sum of param and "
                     "indices ranks if they are all static");
  }

  return success();
}

LogicalResult GatherSumOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType indicesType = getIndices().getType();
  TensorType resultType = getResult().getType();

  // Operand constraints (rank-2 table, rank-2 si32 indices) are enforced
  // by the TableGen confined operand types; this verifier pins only the
  // result shape: the reduce.add output [batch, 1, ...] with the
  // multi-hot axis kept at size 1 (or the rank-2 equivalent).
  if (resultType.getRank() != 2 && resultType.getRank() != 3)
    return emitError("gather_sum result must be rank 2 or 3");

  // Pin the result to the kernel's derived shape: batch from the indices,
  // multi-hot axis kept at 1, embedding dim from the table. A too-small
  // declared result would otherwise write out of bounds, and unlike most
  // ops this one has no shape inference that would catch the mismatch.
  // TypedAttr equality covers both constant dims and matching parametric
  // dims (e.g. result and table both `N`); a constant-vs-parametric pair
  // is left for shape materialization to decide.
  bool batchOk = resultType.getDim(0) == indicesType.getDim(0);
  bool midAxisKept = resultType.getRank() == 2 ||
                     MOSH::ShapeAttr::getStaticDim(resultType.getDim(1)) == 1;
  auto embDim =
      resultType.getRank() == 2 ? resultType.getDim(1) : resultType.getDim(2);
  if (!batchOk || !midAxisKept || embDim != inputType.getDim(1))
    return emitError("gather_sum result must be [batch, 1, emb_dim] "
                     "(or [batch, emb_dim]) matching the indices batch "
                     "and table width");

  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  // `axis` is a compile-time `index` attribute; its range is validated during
  // shape materialization.
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();

  if (inputTy.hasStaticShape() && resTy.hasStaticShape() &&
      inputTy.getShape() != resTy.getShape()) {
    return emitError("input and result shapes must match if both static");
  }

  // Input, updates, indices and result must share a rank when ranked.
  uint64_t expectedRank = 0;
  bool haveExpected = false;
  auto rankMatches = [&](TensorValue tensor) -> bool {
    TensorType ty = tensor.getType();
    if (!ty.hasRank())
      return true;
    if (!haveExpected) {
      expectedRank = ty.getRank();
      haveExpected = true;
      return true;
    }
    return ty.getRank() == expectedRank;
  };
  if (!rankMatches(getInput()) || !rankMatches(getUpdates()) ||
      !rankMatches(getIndices()) || !rankMatches(getResult())) {
    return emitError(
        "output and inputs (except axis) of a scatter op must have the "
        "same rank");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  DType xDType = getX().getType().getDtype();
  DType yDType = getY().getType().getDtype();
  DType resultDType = getResult().getType().getDtype();

  if (xDType != yDType)
    return emitError("operands must have the same element type");
  if (xDType != resultDType)
    return emitError(
        "operands and result type must have the same element type");

  return success();
}

//===----------------------------------------------------------------------===//
// ArgNonzeroOp
//===----------------------------------------------------------------------===//

LogicalResult ArgNonzeroOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();

  if (!resTy.hasRank() || resTy.getRank() != 2)
    return emitError("mo.arg_nonzero: invalid output shape: rank must be 2");

  if ((inputTy.hasRank() && (int64_t)inputTy.getRank() > 0 &&
       resTy.getIntDim(1) != (int64_t)inputTy.getRank()) ||
      (!inputTy.hasRank() && !ShapedType::isDynamic(resTy.getIntDim(1))))
    return emitError(
        "mo.arg_nonzero: invalid output shape: inner_dim != input_rank");

  return success();
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult
verifyWithRankedInput(Op *op, int64_t axis, bool isReductionAxis,
                      int64_t resDim, TensorType inputTy,
                      size_t &dynamicOutDims, size_t &reducedDims) {
  auto inDim = inputTy.getShape().getIntDim(axis);
  if (mlir::ShapedType::isDynamic(resDim) ||
      mlir::ShapedType::isDynamic(inDim)) {
    ++dynamicOutDims;
    return success();
  }
  if (inDim != resDim) {
    if (resDim != 1)
      return op->emitOpError(
          "mo.reduce.arg_max: output dimensions must match input "
          "dimensions, except at the reduction axis where the "
          "output dimension must be 1");
    ++reducedDims;
  } else if (isReductionAxis && resDim == 1) {
    // Reducing an axis of length 1 keeps the shape.
    ++reducedDims;
  }
  return success();
}

static void verifyWithUnrankedInput(int64_t axis, int64_t resDim,
                                    size_t &dynamicOutDims,
                                    size_t &reducedDims) {
  if (mlir::ShapedType::isDynamic(resDim)) {
    ++dynamicOutDims;
    return;
  }
  if (resDim == 1)
    ++reducedDims;
}

template <typename Op>
static LogicalResult argVerify(Op *op) {
  static_assert(std::is_same_v<Op, ArgMaxOp> || std::is_same_v<Op, ArgMinOp>,
                "input op must be either an ArgMax or ArgMin op");

  TensorType resTy = op->getResult().getType();
  TensorType inputTy = op->getInput().getType();

  // TODO(#12939): Refactor this into verifyReduction.

  if ((inputTy.hasRank() && resTy.hasRank()) &&
      (inputTy.getRank() != resTy.getRank()))
    return op->emitOpError(Op::getOperationName() +
                           ": output rank must match input rank");

  if (resTy.hasRank()) {
    // `axis` is a compile-time `index` attribute and is always known.
    int64_t optAxis = op->getAxisAttr().getInt();
    size_t reducedDims = 0;
    size_t dynamicOutDims = 0;

    for (int64_t axis = 0, rank = resTy.getRank(); axis < rank; ++axis) {
      int64_t resDim = resTy.getShape().getIntDim(axis);

      if ((axis == optAxis) && (resDim != 1) &&
          !mlir::ShapedType::isDynamic(resDim))
        return op->emitOpError(
            Op::getOperationName() +
            ": output dimensions must be 1 at the reduction axis");

      // Note: we don't check that non-reduction axes match directly. The code
      // below does it, which handles the dynamic cases as well.

      if (inputTy.hasRank()) {
        if (failed(verifyWithRankedInput(op, axis, axis == optAxis, resDim,
                                         inputTy, dynamicOutDims, reducedDims)))
          return failure();
      } else {
        verifyWithUnrankedInput(axis, resDim, dynamicOutDims, reducedDims);
      }
    }

    if ((reducedDims > 1) || (!reducedDims && !dynamicOutDims)) {
      return op->emitOpError(
          Op::getOperationName() +
          ": output dimensions must match input "
          "dimensions, except for exactly one dimension, the reduction axis");
    }
  }

  return success();
}

LogicalResult ArgMaxOp::verify() { return argVerify(this); }

//===----------------------------------------------------------------------===//
// ArgMinOp
//===----------------------------------------------------------------------===//

LogicalResult ArgMinOp::verify() { return argVerify(this); }

//===----------------------------------------------------------------------===//
// CumSumOp
//===----------------------------------------------------------------------===//

LogicalResult CumSumOp::verify() {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (inputType.getDtype() != resultType.getDtype())
    return emitOpError("Expect input and result DTypes to be the same.");

  auto inDimAttrs = inputType.getShape();
  auto outDimAttrs = resultType.getShape();
  if (inDimAttrs && outDimAttrs) {
    for (auto [inDimAttr, outDimAttr] :
         llvm::zip(inDimAttrs.getValues(), outDimAttrs.getValues())) {
      auto inDimInt = tryGetConstantInt(inDimAttr);
      auto outDimInt = tryGetConstantInt(outDimAttr);
      if (inDimInt && outDimInt && *inDimInt != *outDimInt)
        return emitError("statically known dimensions must match");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Windowed ops
//===----------------------------------------------------------------------===//

LogicalResult AvgPoolOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult AvgPoolCeilOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult MaxPoolOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult MaxPoolCeilOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

LogicalResult ConvOp::verify() {
  // The sliding window interface already checks input/result ranks; this
  // gives a more specific error.
  if (failed(
          ::Impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
    return emitError("filter, input and output ranks must be all dynamic, or "
                     "all static and match");

  auto op = cast<SlidingWindow>(this->getOperation());
  return op.verifyWindowParamSizes({getDilations(), getStrides()});
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

LogicalResult ConvTransposeOp::verify() {
  // The sliding window interface already checks input/result ranks; this
  // gives a more specific error.
  if (failed(
          ::Impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
    return emitError("filter, input and output ranks must be all dynamic, or "
                     "all static and match");

  auto op = cast<SlidingWindow>(this->getOperation());
  return op.verifyWindowParamSizes(
      {getDilations(), getStrides(), getOutputPaddings()});
}

LogicalResult BatchMatmulOp::verify() {
  TensorType aType = getTensorA().getType();
  TensorType bType = getTensorB().getType();
  TensorType cType = getTensorResult().getType();

  // Note MO_MatmulLike interface verifies ranks are consistent between ops.
  auto violatedRankProperty = [&](TensorType ty) -> bool {
    return ty.hasRank() && ty.getRank() < 3;
  };
  if (violatedRankProperty(aType) || violatedRankProperty(bType) ||
      violatedRankProperty(cType)) {
    return emitError("All operands and results must be of at least rank 3.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

LogicalResult RangeOp::verify() {
  ArrayRef<KGEN::ParamDeclAttr> paramDecls = getOutputParamDecls();
  if (!paramDecls.empty()) {
    if (paramDecls.size() != 1)
      return emitError("cannot declare more than one parameter");
    if (!ShapeAttr::isDimType(paramDecls[0].getType()))
      return emitError("can only declare dimension parameters");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

LogicalResult RebindOp::verify() {
  TensorType inTy = getInput().getType();
  TensorType outTy = getType();
  // Disallow rebind to a different device.
  if (inTy.getDeviceRef() && outTy.getDeviceRef() &&
      (inTy.getDeviceRef() != outTy.getDeviceRef()))
    return emitError("input device " + inTy.getDeviceRef().toString() +
                     " must match result device " +
                     outTy.getDeviceRef().toString() + " in rebind");

  // Nothing to check if either type is dynamically ranked
  if (!inTy.hasRank() || !outTy.hasRank())
    return success();

  if (inTy.getRank() != outTy.getRank())
    return emitError("input and output static ranks must match");

  SmallDenseMap<KGEN::ParamDeclRefAttr, TypedAttr> dimParamToConst;
  auto tryUnifyParamToConstDim = [&](TypedAttr srcDimAttr,
                                     TypedAttr dstDimAttr) -> LogicalResult {
    auto srcDimParamRef = dyn_cast<KGEN::ParamDeclRefAttr>(srcDimAttr);
    if (srcDimParamRef && tryGetConstantInt(dstDimAttr)) {
      auto [cachedEntry, _] =
          dimParamToConst.try_emplace(srcDimParamRef, dstDimAttr);
      if (cachedEntry->second != dstDimAttr)
        return failure();
    }
    return success();
  };
  auto inDimAttrs = inTy.getShape();
  auto outDimAttrs = outTy.getShape();
  if (inDimAttrs && outDimAttrs) {
    for (auto [inDimAttr, outDimAttr] :
         llvm::zip(inDimAttrs.getValues(), outDimAttrs.getValues())) {
      auto inDimInt = tryGetConstantInt(inDimAttr);
      auto outDimInt = tryGetConstantInt(outDimAttr);
      if (inDimInt && outDimInt && *inDimInt != *outDimInt)
        return emitError("statically known dimensions must match");
      else if (failed(tryUnifyParamToConstDim(inDimAttr, outDimAttr)) ||
               failed(tryUnifyParamToConstDim(outDimAttr, inDimAttr)))
        return emitError("one dimension parameter cannot bind to different "
                         "static dimensions");
    }
  }
  return success();
}

#undef DEFINE_IMPLICIT_AXIS_REDUCTION

//===----------------------------------------------------------------------===//
// TopK / BottomK
//===----------------------------------------------------------------------===//

template <typename OrderKOpT>
LogicalResult verifyOrderedKOp(OrderKOpT op) {
  auto output1Shape = op.getValues().getType().getShape();
  auto output2Shape = op.getIndices().getType().getShape();
  if (output1Shape != output2Shape) {
    return op.emitOpError(
               "values and indices should have the same shape but got: ")
           << output1Shape << " vs " << output2Shape << "\n";
  }
  return success();
}

LogicalResult TopKOp::verify() { return verifyOrderedKOp(*this); }

LogicalResult BottomKOp::verify() { return verifyOrderedKOp(*this); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

template <typename T>
inline LogicalResult checkReshapeInOutTypesCompatible(T op) {
  TensorType inputTy = op.getInput().getType();
  TensorType resultTy = op.getType();
  return checkReshapeShapesCompatible(inputTy.getShape(), resultTy.getShape(),
                                      [&]() { return op.emitOpError(); });
}

LogicalResult ReshapeOp::verify() {
  LogicalResult result = checkReshapeInOutTypesCompatible(*this);
  if (result.failed())
    return result;

  auto shapeInput =
      dyn_cast_if_present<ConstantOp>(getNewShape().getDefiningOp());
  if (!shapeInput)
    return success();

  ElementsAttr attr = shapeInput.getValue();
  auto dimIntegers =
      tryGetConstantBlob<int64_t>(attr, getNewShape().getType().getDtype());
  if (!dimIntegers.has_value())
    return success();

  int64_t requestedElements = 1;
  for (int64_t dim : dimIntegers.value()) {
    if (dim == -1)
      return emitOpError("-1 size unsupported in requested shape");
    requestedElements *= dim;
  }

  std::optional<int64_t> resultElements =
      getResult().getType().getNumElements();
  if (resultElements.has_value() &&
      resultElements.value() != requestedElements) {
    return emitOpError("Expected requested shape and output shape to have the "
                       "same number of elements requested ")
           << requestedElements << " but expected " << resultElements.value();
  }

  std::optional<int64_t> inputElements = getInput().getType().getNumElements();
  if (inputElements.has_value() && inputElements.value() != requestedElements) {
    return emitOpError("Expected requested shape and input shape to have the "
                       "same number of elements requested ")
           << requestedElements << " but expected " << inputElements.value();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnsqueezeShapeOp
//===----------------------------------------------------------------------===//
LogicalResult UnsqueezeShapeOp::verify() {
  if (getInputShape().getType().getDtype() != getResult().getType().getDtype())
    return emitOpError("Expect in and out DTypes to be the same type");

  std::optional<int64_t> inputShapeSize =
      getInputShape().getType().getNumElements();

  // Check indices don't point to the same index. Negative indices are
  // normalized first, so that -1 and 5 on a rank-6 output count as the same.
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(getPaddingIndices());
  if (maybeIndices) {
    SmallVector<int64_t> &indices = maybeIndices.value();
    if (inputShapeSize) {
      int64_t outputRank = *inputShapeSize + indices.size();
      for (int64_t &index : indices) {
        if (index < -outputRank || index >= outputRank)
          return emitOpError("Unsqueeze index not in [-out_rank, out_rank)");
        if (index < 0)
          index += outputRank;
      }
    }
    llvm::sort(indices);
    for (size_t i = 1; i < indices.size(); ++i) {
      if (indices[i - 1] == indices[i])
        return emitOpError("Padding indices pad the same dimension twice");
    }
  }

  std::optional<int64_t> paddingIndicesSize =
      getPaddingIndices().getType().getNumElements();
  std::optional<int64_t> outputSize = getResult().getType().getNumElements();
  if (inputShapeSize.has_value() && paddingIndicesSize.has_value() &&
      outputSize.has_value() &&
      inputShapeSize.value() + paddingIndicesSize.value() !=
          outputSize.value()) {
    return emitOpError("Output size is not the sum of the input sizes");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SqueezeShapeOp
//===----------------------------------------------------------------------===//

LogicalResult SqueezeShapeOp::verify() {
  if (getInputShape().getType().getDtype() != getResult().getType().getDtype())
    return emitOpError("Expect in and out DTypes to be the same type");

  std::optional<int64_t> inputShapeSize =
      getInputShape().getType().getNumElements();

  // Check indices don't point to the same index. Negative indices are
  // normalized first, so that -1 and 3 on a rank-4 input count as the same.
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(getRemoveIndices());
  if (maybeIndices) {
    SmallVector<int64_t> &indices = maybeIndices.value();
    if (inputShapeSize) {
      int64_t inputRank = *inputShapeSize;
      for (int64_t &index : indices) {
        if (index < -inputRank || index >= inputRank)
          return emitOpError("Squeeze index not in [-in_rank, in_rank)");
        if (index < 0)
          index += inputRank;
      }
    }
    llvm::sort(indices);
    for (size_t i = 1, e = indices.size(); i < e; ++i)
      if (indices[i - 1] == indices[i])
        return emitOpError("Padding indices pad the same dimension twice");
  }

  std::optional<int64_t> paddingIndicesSize =
      getRemoveIndices().getType().getNumElements();
  std::optional<int64_t> outputSize = getResult().getType().getNumElements();
  if (inputShapeSize && paddingIndicesSize && outputSize &&
      inputShapeSize.value() - paddingIndicesSize.value() !=
          outputSize.value()) {
    return emitOpError("Output size is not the difference of the input sizes");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Static::ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult Static::ReshapeOp::verify() {
  return checkReshapeInOutTypesCompatible(*this);
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeOfOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();
  if (resTy.hasStaticShape()) {
    if (!inputTy.hasRank())
      return emitError("input must have static rank if result has known size");
    if ((int64_t)inputTy.getRank() != resTy.getIntDim(0))
      return emitError("input rank and output size must match");
    return success();
  }

  if (inputTy.hasRank())
    return emitError("input must have dynamic rank if result has unknown size");
  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::verify() {
  TensorType inputTy = getInput().getType();
  TensorType startTy = getStart().getType();
  TensorType stopTy = getStop().getType();
  TensorType stepTy = getStep().getType();
  if (inputTy.hasRank()) {
    int64_t inputRank = inputTy.getRank();
    if (startTy.hasStaticShape() && startTy.getIntDim(0) != inputRank)
      return emitError("start shape must be [inputRank] if it's static");

    if (stopTy.hasStaticShape() && stopTy.getIntDim(0) != inputRank)
      return emitError("stop shape must be [inputRank] if it's static");

    if (stepTy.hasStaticShape() && stepTy.getIntDim(0) != inputRank)
      return emitError("step shape must be [inputRank] if it's static");

  } else {
    if (startTy.hasStaticShape()) {
      return emitError(
          "start shape must be dynamic if input is has dynamic rank");
    }

    if (stopTy.hasStaticShape()) {
      return emitError(
          "stop shape must be dynamic if input is has dynamic rank");
    }

    if (stepTy.hasStaticShape()) {
      return emitError(
          "step shape must be dynamic if input is has dynamic rank");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  // The permutation has known shape iff the argument and result are
  // statically ranked.
  TensorType permTy = getPerm().getType();
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();
  if (!permTy.hasStaticShape()) {
    if (inputTy.hasRank() || resTy.hasRank()) {
      return emitError("argument and result must be dynamically ranked if perm "
                       "has unknown shape");
    }
    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank()) {
    return emitError("argument and result must be statically ranked if perm "
                     "has known shape");
  }

  size_t opRank = resTy.getRank();
  if (inputTy.getRank() != opRank)
    return emitError("input and result ranks do not match");

  // ODS constraints ensure that perm has rank = 1.
  if ((int64_t)opRank != permTy.getIntDim(0))
    return emitError("input and result ranks do not match perm size");

  // A valid perm must be a permutation of [0, ..., rank - 1].
  if (auto maybePermVec = tryGetConstantIntVector(getPerm())) {
    auto permVec = maybePermVec.value();
    auto identityPerm =
        llvm::iota_range<int64_t>(0, opRank, /*Inclusive=*/false);
    SmallVector<int64_t> sortedPerm(permVec);
    llvm::sort(sortedPerm);
    if (!llvm::equal(sortedPerm, identityPerm))
      return emitError("invalid permutations");

    if (inputTy.hasStaticShape() && resTy.hasStaticShape() &&
        !llvm::equal(resTy.getShape().getValues(),
                     permute(inputTy.getShape().getValues(), permVec)))
      return emitError("output shape is invalid based on constant permutation");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

LogicalResult TileOp::verify() {
  // `repeats` has known shape iff the argument and result are statically
  // ranked.
  TensorType repeatsTy = getRepeats().getType();
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();
  if (!repeatsTy.hasStaticShape()) {
    if (inputTy.hasRank() || resTy.hasRank()) {
      return emitError("argument and result must be dynamically ranked if "
                       "repeats has unknown shape");
    }
    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank()) {
    return emitError("argument and result must be statically ranked if "
                     "repeats has known shape");
  }

  if (inputTy.getRank() != resTy.getRank())
    return emitError("input and result ranks do not match");

  // ODS constraints ensure that repeats has rank = 1.
  if ((int64_t)resTy.getRank() != repeatsTy.getIntDim(0))
    return emitError("input and result ranks do not match repeats size");
  return success();
}

//===----------------------------------------------------------------------===//
// Linear Algebra Ops
//===----------------------------------------------------------------------===//

LogicalResult LinAlg::BandPartOp::verify() {
  // TODO: Refactor this into a trait when we see another op that does this
  if (getInput().getType() != getResult().getType())
    return emitError("first operand and the result must have the same type");
  return success();
}

//===----------------------------------------------------------------------===//
// InvokeShapeFuncOp
//===----------------------------------------------------------------------===//

LogicalResult InvokeShapeFuncOp::verify() {
  if (llvm::any_of(getOutputParamDecls(), [&](ParamDeclAttr decl) {
        return isa<MOSH::ShapeType>(decl.getType());
      }))
    return emitError() << "dynamic rank isn't supported in shape function";
  return success();
}

//===----------------------------------------------------------------------===//
// LayoutTransformOp
//===----------------------------------------------------------------------===//

LogicalResult LayoutTransformOp::verify() {
  if (!getInput().getType().getLayoutAttr())
    return emitError() << "input type must contain layout annotation";

  if (!getResult().getType().getLayoutAttr())
    return emitError() << "output type must contain layout annotation";

  return success();
}

LogicalResult MutableStoreSliceOp::verify() {

  // Copied wholesale from the mo.slice verifier
  TensorType inputTy = castFromTensorLike(getInBuffer().getType());
  TensorType startTy = getStart().getType();
  TensorType stopTy = getStop().getType();
  TensorType stepTy = getStep().getType();
  int64_t inputRank = inputTy.getRank();
  if (startTy.hasStaticShape() && startTy.getIntDim(0) != inputRank)
    return emitError("start shape must be [inputRank] if it's static");

  if (stopTy.hasStaticShape() && stopTy.getIntDim(0) != inputRank)
    return emitError("stop shape must be [inputRank] if it's static");

  if (stepTy.hasStaticShape() && stepTy.getIntDim(0) != inputRank)
    return emitError("step shape must be [inputRank] if it's static");

  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceSumOp
//===----------------------------------------------------------------------===//

LogicalResult AllReduceSumOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty())
    return emitError("allreduce requires at least one input");

  // Zero (the attribute default) means all devices form one group.
  int64_t ngpus = inputs.size();
  int64_t groupSize = getGroupSizeAttr().getInt();
  if (groupSize < 0)
    return emitError("group_size must be non-negative");
  if (groupSize == 0)
    groupSize = ngpus;
  if (ngpus % groupSize != 0)
    return emitError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        ngpus, groupSize));

  // Groups may differ in shape (e.g. per-DP-replica row counts). Inputs
  // differ in device (one per GPU), so AllTypesMatch can't be used.
  // Output-input type matching and signal buffer count are TableGen traits.
  auto firstInputType = getInputTensor(0).getType();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
  }
  for (int64_t groupStart = 0; groupStart < ngpus; groupStart += groupSize) {
    auto groupInputType = getInputTensor(groupStart).getType();
    for (int64_t i = groupStart + 1, e = groupStart + groupSize; i < e; ++i) {
      auto inputType = getInputTensor(i).getType();
      if (inputType.getShape() != groupInputType.getShape())
        return emitError(
            "all inputs in each allreduce group must have the same shape");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BundledExpandOp
//===----------------------------------------------------------------------===//

LogicalResult BundledExpandOp::verify() {
  auto parallelOp = (*this)->getParentOfType<ParallelOp>();
  if (!parallelOp)
    return emitOpError("must be inside an mo.parallel region");

  size_t numLaunches = parallelOp.getNumLaunches();
  if (getNumResults() != numLaunches)
    return emitOpError("expected ")
           << numLaunches << " results (one per launch), got "
           << getNumResults();

  SmallVector<DeviceRefAttr> launchDevices = parallelOp.getLaunchDeviceList();
  for (auto [i, resultType] : llvm::enumerate(getResultTypes())) {
    if (!typesMatchIgnoringDeviceId(resultType, getInput().getType()))
      return emitOpError("result #")
             << i << " type " << resultType
             << " does not match input type (ignoring device) "
             << getInput().getType();

    auto expectedDevice = launchDevices[i];
    auto actualDevice = cast<TensorType>(resultType).getDeviceRef();
    if (actualDevice != expectedDevice)
      return emitOpError("result #")
             << i << " device " << actualDevice
             << " does not match launch device " << expectedDevice;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Bundled collectives
//===----------------------------------------------------------------------===//

/// Checks a bundled collective's `peer_rank` against its world. Absent, the
/// kernel takes the rank from the device id; set, it must name the signal
/// buffer on this op's device.
static LogicalResult verifyPeerRank(Operation *op,
                                    std::optional<int64_t> peerRank,
                                    ValueRange signalBuffers,
                                    DeviceRefAttr device) {
  if (!peerRank)
    return success();
  int64_t numDevices = signalBuffers.size();
  if (*peerRank < 0 || *peerRank >= numDevices)
    return op->emitOpError(llvm::formatv(
        "peer_rank {0} out of range for {1} devices", *peerRank, numDevices));
  auto peerDevice =
      cast<BufferType>(signalBuffers[*peerRank].getType()).getDeviceRef();
  if (device && peerDevice && device != peerDevice)
    return op->emitOpError(llvm::formatv(
        "peer_rank {0} names the signal buffer on {1}, but the op runs on {2}",
        *peerRank, peerDevice, device));
  return success();
}

//===----------------------------------------------------------------------===//
// BundledAllReduceSumOp
//===----------------------------------------------------------------------===//

LogicalResult BundledAllReduceSumOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty())
    return emitOpError("requires at least one input");

  // Zero (the attribute default) means all devices form one group.
  int64_t numDevices = inputs.size();
  int64_t groupSize = getGroupSize();
  if (groupSize < 0)
    return emitOpError("group_size must be non-negative");
  if (groupSize == 0)
    groupSize = numDevices;
  if (numDevices % groupSize != 0)
    return emitOpError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        numDevices, groupSize));

  auto outputType = cast<TensorType>(getOutput().getType());
  if (failed(verifyPeerRank(getOperation(), getPeerRank(), getSignalBuffers(),
                            outputType.getDeviceRef())))
    return failure();

  // Groups may differ in shape (e.g. per-DP-replica row counts), so shapes
  // are checked within a group and element types across the world.
  auto firstType = cast<TensorType>(inputs[0].getType());
  for (Value input : inputs.drop_front())
    if (cast<TensorType>(input.getType()).getElementType() !=
        firstType.getElementType())
      return emitOpError("all inputs must have the same element type");
  for (int64_t groupStart = 0; groupStart < numDevices;
       groupStart += groupSize) {
    auto groupType = cast<TensorType>(inputs[groupStart].getType());
    for (int64_t i = groupStart + 1, e = groupStart + groupSize; i < e; ++i)
      if (cast<TensorType>(inputs[i].getType()).getShape() !=
          groupType.getShape())
        return emitOpError(
            "all inputs in each allreduce group must have the same shape");
  }

  if (outputType.getElementType() != firstType.getElementType())
    return emitOpError("the output must have the inputs' element type");
  if (std::optional<int64_t> peerRank = getPeerRank()) {
    auto peerType = cast<TensorType>(inputs[*peerRank].getType());
    if (outputType.getShape() != peerType.getShape())
      return emitOpError(llvm::formatv(
          "the output must have the same shape as input {0}, this device's "
          "own input",
          *peerRank));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BundledBroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BundledBroadcastOp::verify() {
  auto signalBuffers = getSignalBuffers();
  if (signalBuffers.empty())
    return emitError("bundled broadcast requires at least one signal buffer");

  int64_t root = getRoot();
  int64_t numDevices = signalBuffers.size();
  if (root < 0 || root >= numDevices)
    return emitError(llvm::formatv(
        "root index {0} out of range for {1} devices", root, numDevices));

  auto inputType = cast<TensorType>(getInput().getType());
  // The input is the root's copy of the tensor, so it must already live on
  // the root's device. Devices can still be unassigned in graphs built in
  // memory, so only compare when both are known.
  auto rootDevice =
      cast<BufferType>(signalBuffers[root].getType()).getDeviceRef();
  auto inputDevice = inputType.getDeviceRef();
  if (inputDevice && rootDevice && inputDevice != rootDevice)
    return emitError(llvm::formatv(
        "the input must be on the root's device, {0}, but is on {1}",
        rootDevice, inputDevice));
  auto outputType = cast<TensorType>(getOutput().getType());
  if (outputType.getShape() != inputType.getShape())
    return emitError("the output must have the same shape as the input");
  if (outputType.getElementType() != inputType.getElementType())
    return emitError("the output must have the same element type as the input");
  return verifyPeerRank(getOperation(), getPeerRank(), signalBuffers,
                        outputType.getDeviceRef());
}

//===----------------------------------------------------------------------===//
// BundledAllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult BundledAllGatherOp::verify() {
  auto inputs = getInputs();
  auto outputs = getOutputs();
  if (inputs.empty())
    return emitError("bundled allgather requires at least one input");
  if (outputs.size() != inputs.size())
    return emitError(llvm::formatv(
        "number of outputs must match number of inputs (one gathered piece "
        "per peer); expected {0} outputs but got {1}",
        inputs.size(), outputs.size()));

  auto device = cast<TensorType>(outputs[0].getType()).getDeviceRef();
  if (failed(verifyPeerRank(getOperation(), getPeerRank(), getSignalBuffers(),
                            device)))
    return failure();
  for (auto [i, pair] : llvm::enumerate(llvm::zip(inputs, outputs))) {
    auto [input, output] = pair;
    auto inputType = cast<TensorType>(input.getType());
    auto outputType = cast<TensorType>(output.getType());
    if (outputType.getShape() != inputType.getShape())
      return emitError(
          llvm::formatv("output {0} must have the same shape as input {0}", i));
    if (outputType.getElementType() != inputType.getElementType())
      return emitError(llvm::formatv(
          "output {0} must have the same element type as input {0}", i));
    if (outputType.getDeviceRef() != device)
      return emitError(
          llvm::formatv("output {0} must be on this device's output device, "
                        "{1}, but is on {2}",
                        i, device, outputType.getDeviceRef()));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DistributedScatterOp
//===----------------------------------------------------------------------===//

LogicalResult DistributedScatterOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty())
    return emitError("scatter requires at least one input");

  auto outputs = getOutputs();
  if (outputs.size() != inputs.size())
    return emitError("number of outputs must match number of inputs");

  if (getSignalBuffers().size() != inputs.size())
    return emitError("number of signal buffers must match number of inputs");

  auto firstInputType = getInputTensor(0).getType();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
  }

  int64_t root = getRoot();
  if (root < 0 || root >= static_cast<int64_t>(inputs.size()))
    return emitError("root must be in range [0, ngpus)");

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceScatterSumOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterSumOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty())
    return emitError("reducescatter requires at least one input");

  // Verify input ranks and element types. Shapes are checked per
  // reduce-scatter group below.
  auto firstInputType = getInputTensor(0).getType();
  int64_t rank = firstInputType.getRank();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
    if (inputType.getRank() != firstInputType.getRank())
      return emitError("all inputs must have the same rank");
  }

  auto outputs = getOutputs();
  if (outputs.size() != inputs.size())
    return emitError("number of outputs must match number of inputs");

  if (getSignalBuffers().size() != inputs.size())
    return emitError("number of signal buffers must match number of inputs");

  int64_t ngpus = inputs.size();
  int64_t groupSize = getGroupSizeAttr().getInt();
  if (groupSize < 1)
    return emitError("group_size must be at least 1");
  if (ngpus % groupSize != 0)
    return emitError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        ngpus, groupSize));

  // Resolve scatter axis (negative indexing supported).
  int64_t axis = normalizeAxis(getAxis(), rank);

  if (axis < 0 || axis >= rank)
    return emitError(
        llvm::formatv("axis {0} out of bounds for rank {1}", getAxis(), rank));

  // Inputs in a group reduce together and therefore must have matching shapes.
  for (int64_t groupStart = 0; groupStart < ngpus; groupStart += groupSize) {
    auto groupInputType = getInputTensor(groupStart).getType();
    for (int64_t i = groupStart + 1, e = groupStart + groupSize; i < e; ++i) {
      auto inputType = getInputTensor(i).getType();
      if (inputType.getShape() != groupInputType.getShape())
        return emitError(
            "all inputs in each reduce-scatter group must have the same shape");
    }
  }

  for (int64_t devIdx = 0, e = outputs.size(); devIdx < e; ++devIdx) {
    auto outputType = getOutputTensor(devIdx).getType();
    auto inputType = getInputTensor(devIdx).getType();

    if (outputType.getElementType() != firstInputType.getElementType())
      return emitError("output element type must match input element type");

    if (static_cast<int64_t>(outputType.getRank()) != rank)
      return emitError("output rank must match input rank");

    for (int64_t i = 0; i < rank; ++i) {
      int64_t inputDim = inputType.getIntDim(i);
      int64_t outputDim = outputType.getIntDim(i);

      if (mlir::ShapedType::isDynamic(inputDim) ||
          mlir::ShapedType::isDynamic(outputDim))
        continue;

      if (i == axis) {
        // Scatter axis: verify partitioned size
        int64_t base = inputDim / groupSize;
        int64_t remainder = inputDim % groupSize;
        int64_t localRank = devIdx % groupSize;
        int64_t expectedSize = base + (localRank < remainder ? 1 : 0);

        if (outputDim != expectedSize)
          return emitError(llvm::formatv(
              "output dimension {0} should be {1} for device {2} "
              "(input dim {3} / group_size {4}), but got {5}",
              i, expectedSize, devIdx, inputDim, groupSize, outputDim));
      } else {
        // Non-scatter axes: must match exactly
        if (outputDim != inputDim)
          return emitError(llvm::formatv(
              "output dimension {0} must match input ({1} vs {2})", i, inputDim,
              outputDim));
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty())
    return emitError("all_to_all requires at least one input");

  auto outputs = getOutputs();
  if (outputs.size() != inputs.size())
    return emitError("number of outputs must match number of inputs");
  if (getSignalBuffers().size() != inputs.size())
    return emitError("number of signal buffers must match number of inputs");

  auto firstInputType = getInputTensor(0).getType();
  int64_t rank = firstInputType.getRank();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
    if (static_cast<int64_t>(inputType.getRank()) != rank)
      return emitError("all inputs must have the same rank");
  }

  int64_t ngpus = inputs.size();
  int64_t groupSize = getGroupSize();
  if (groupSize < 1)
    return emitError("group_size must be at least 1");
  if (ngpus % groupSize != 0)
    return emitError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        ngpus, groupSize));

  int64_t splitAxis = getSplitAxis();
  int64_t concatAxis = getConcatAxis();
  if (splitAxis < 0 || splitAxis >= rank)
    return emitError(llvm::formatv("invalid split_axis {0}", splitAxis));
  if (concatAxis < 0 || concatAxis >= rank)
    return emitError(llvm::formatv("invalid concat_axis {0}", concatAxis));

  // Inputs in a group exchange chunks and therefore must match in shape.
  for (int64_t groupStart = 0; groupStart < ngpus; groupStart += groupSize) {
    auto groupInputType = getInputTensor(groupStart).getType();
    for (int64_t i = groupStart + 1, e = groupStart + groupSize; i < e; ++i) {
      if (getInputTensor(i).getType().getShape() != groupInputType.getShape())
        return emitError(
            "all inputs in each all_to_all group must have the same shape");
    }
  }

  for (int64_t devIdx = 0; devIdx < ngpus; ++devIdx) {
    auto inputType = getInputTensor(devIdx).getType();
    auto outputType = getOutputTensor(devIdx).getType();
    if (outputType.getElementType() != firstInputType.getElementType())
      return emitError("output element type must match input element type");
    if (static_cast<int64_t>(outputType.getRank()) != rank)
      return emitError("output rank must match input rank");

    // The input's shape with the split axis divided by the group size and
    // the concat axis multiplied by it; dynamic dims stay unchecked.
    SmallVector<int64_t> expected;
    for (int64_t i = 0; i < rank; ++i)
      expected.push_back(inputType.getIntDim(i));
    if (!mlir::ShapedType::isDynamic(expected[splitAxis])) {
      if (expected[splitAxis] % groupSize != 0)
        return emitError(llvm::formatv(
            "split axis of size {0} does not divide into {1} chunks",
            expected[splitAxis], groupSize));
      expected[splitAxis] /= groupSize;
    }
    if (!mlir::ShapedType::isDynamic(expected[concatAxis]))
      expected[concatAxis] *= groupSize;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t outputDim = outputType.getIntDim(i);
      if (mlir::ShapedType::isDynamic(expected[i]) ||
          mlir::ShapedType::isDynamic(outputDim))
        continue;
      if (outputDim != expected[i])
        return emitError(llvm::formatv("output {0} dim {1} should be {2}",
                                       devIdx, i, expected[i]));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EPDispatchOp (BF16)
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchOp::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.dispatch requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

//===----------------------------------------------------------------------===//
// EPDispatchFP8Op
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchFP8Op::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.dispatch.fp8 requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getOutputScales().size() != numDevices)
    return emitError("outputScales count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

//===----------------------------------------------------------------------===//
// EPDispatchBlockScaledNVOp
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchBlockScaledNVOp::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError(
        "ep.dispatch.block.scaled.nv requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getInputScales().size() != numDevices)
    return emitError("inputScales count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getOutputScales().size() != numDevices)
    return emitError("outputScales count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getScalesOffsets().size() != numDevices)
    return emitError("scalesOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

//===----------------------------------------------------------------------===//
// EPDispatchBlockScaledNVDynGlobalScalesOp
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchBlockScaledNVDynGlobalScalesOp::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.dispatch.block.scaled.nv.dyn_global_scales requires "
                     "at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getOutputScales().size() != numDevices)
    return emitError("outputScales count must match inputTokens count");
  if (getOutputRowwiseScales().size() != numDevices)
    return emitError("outputRowwiseScales count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getScalesOffsets().size() != numDevices)
    return emitError("scalesOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

//===----------------------------------------------------------------------===//
// EPDispatchMXFP4Op
//===----------------------------------------------------------------------===//

LogicalResult EPDispatchMXFP4Op::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.dispatch.mxfp4 requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getTopkIds().size() != numDevices)
    return emitError("topkIds count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");
  if (getOutputScales().size() != numDevices)
    return emitError("outputScales count must match inputTokens count");
  if (getRowOffsets().size() != numDevices)
    return emitError("rowOffsets count must match inputTokens count");
  if (getExpertIds().size() != numDevices)
    return emitError("expertIds count must match inputTokens count");
  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");

  return success();
}

//===----------------------------------------------------------------------===//
// EPCombineOp
//===----------------------------------------------------------------------===//

LogicalResult EPCombineOp::verify() {
  auto inputTokens = getInputTokens();
  if (inputTokens.empty())
    return emitError("ep.combine requires at least one device");

  size_t numDevices = inputTokens.size();

  if (getSrcInfo().size() != numDevices)
    return emitError("srcInfo count must match inputTokens count");
  if (getSendPtrs().size() != numDevices)
    return emitError("sendPtrs count must match inputTokens count");
  if (getRecvPtrs().size() != numDevices)
    return emitError("recvPtrs count must match inputTokens count");
  if (getRecvCountPtrs().size() != numDevices)
    return emitError("recvCountPtrs count must match inputTokens count");
  if (getRouterWeights().size() != numDevices)
    return emitError("routerWeights count must match inputTokens count");
  if (getAtomicCounters().size() != numDevices)
    return emitError("atomicCounters count must match inputTokens count");

  if (getOutputTokens().size() != numDevices)
    return emitError("outputTokens count must match inputTokens count");

  return success();
}

// Composite::DistributedMatmulReduceScatterSumOp
//===----------------------------------------------------------------------===//

LogicalResult Composite::DistributedMatmulReduceScatterSumOp::verify() {
  auto inputsA = getInputsA();
  auto inputsB = getInputsB();
  auto outputs = getOutputs();
  auto signalBuffers = getSignalBuffers();

  if (inputsA.empty())
    return emitError("matmul_reduce_scatter requires at least one device");

  if (inputsB.size() != inputsA.size())
    return emitError(llvm::formatv(
        "number of B inputs ({0}) must match number of A inputs ({1})",
        inputsB.size(), inputsA.size()));

  if (outputs.size() != inputsA.size())
    return emitError(llvm::formatv(
        "number of outputs ({0}) must match number of inputs ({1})",
        outputs.size(), inputsA.size()));

  if (signalBuffers.size() != inputsA.size())
    return emitError(llvm::formatv(
        "number of signal buffers ({0}) must match number of inputs ({1})",
        signalBuffers.size(), inputsA.size()));

  auto firstAType = cast<TensorType>(inputsA[0].getType());
  if (!firstAType.hasRank() || firstAType.getRank() != 2)
    return emitError("A inputs must be rank-2 tensors");

  for (size_t i = 1; i < inputsA.size(); ++i) {
    auto aType = cast<TensorType>(inputsA[i].getType());
    if (aType.getShape() != firstAType.getShape())
      return emitError("all A inputs must have the same shape");
    if (aType.getElementType() != firstAType.getElementType())
      return emitError("all A inputs must have the same element type");
  }

  auto firstBType = cast<TensorType>(inputsB[0].getType());
  if (!firstBType.hasRank() || firstBType.getRank() != 2)
    return emitError("B inputs must be rank-2 tensors");

  for (size_t i = 1; i < inputsB.size(); ++i) {
    auto bType = cast<TensorType>(inputsB[i].getType());
    if (bType.getShape() != firstBType.getShape())
      return emitError("all B inputs must have the same shape");
    if (bType.getElementType() != firstBType.getElementType())
      return emitError("all B inputs must have the same element type");
  }

  // Verify matmul shape compatibility: A is [M, K], B is [N, K] (transposed).
  int64_t aK = firstAType.getIntDim(1);
  int64_t bK = firstBType.getIntDim(1);
  if (!mlir::ShapedType::isDynamic(aK) && !mlir::ShapedType::isDynamic(bK) &&
      aK != bK)
    return emitError(llvm::formatv(
        "matmul K dimension mismatch: A has K={0}, B has K={1}", aK, bK));

  if (getHasResidual()) {
    int64_t residualPeer = getResidualPeer();
    if (residualPeer < 0 ||
        residualPeer >= static_cast<int64_t>(inputsA.size()))
      return emitError(
          llvm::formatv("residual_peer ({0}) must be in range [0, {1})",
                        residualPeer, inputsA.size()));

    auto residualType = cast<TensorType>(getResidual().getType());
    if (!residualType.hasRank() || residualType.getRank() != 2)
      return emitError("residual must be a rank-2 tensor");

    // The residual must live on the same device as the peer that applies it,
    // since that peer's kernel reads the residual tensor directly.
    auto residualDevice = residualType.getDeviceRef();
    auto peerDevice =
        cast<TensorType>(inputsA[residualPeer].getType()).getDeviceRef();
    if (residualDevice && peerDevice && residualDevice != peerDevice)
      return emitError(llvm::formatv(
          "residual device must match the device of input {0}", residualPeer));

    // Residual shape must match matmul output: [M, N].
    int64_t aM = firstAType.getIntDim(0);
    int64_t bN = firstBType.getIntDim(0);
    int64_t rM = residualType.getIntDim(0);
    int64_t rN = residualType.getIntDim(1);
    if (!mlir::ShapedType::isDynamic(aM) && !mlir::ShapedType::isDynamic(rM) &&
        aM != rM)
      return emitError(llvm::formatv(
          "residual M dimension ({0}) must match A M dimension ({1})", rM, aM));
    if (!mlir::ShapedType::isDynamic(bN) && !mlir::ShapedType::isDynamic(rN) &&
        bN != rN)
      return emitError(llvm::formatv(
          "residual N dimension ({0}) must match B N dimension ({1})", rN, bN));

    if (residualType.getElementType() != firstAType.getElementType())
      return emitError("residual element type must match A element type");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  auto outputs = getOutputs();
  auto inputs = getInputs();
  auto signalBuffers = getSignalBuffers();

  if (signalBuffers.size() != inputs.size())
    return emitError(
        llvm::formatv("number of signal buffers should match number of inputs. "
                      "Expected {0} signal buffers but got {1}",
                      inputs.size(), signalBuffers.size()));

  int64_t numInputs = inputs.size();
  int64_t groupSize = getGroupSize();
  if (groupSize < 1)
    return emitError("group_size must be at least 1");
  if (numInputs % groupSize != 0)
    return emitError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        numInputs, groupSize));

  int64_t expectedOutputs = numInputs * groupSize;
  if (static_cast<int64_t>(outputs.size()) != expectedOutputs)
    return emitError(llvm::formatv(
        "number of outputs must be num_inputs * group_size for all-gather; "
        "expected {0} outputs, found {1} (num_inputs={2}, group_size={3})",
        expectedOutputs, outputs.size(), inputs.size(), groupSize));
  if (getSliceRows() && groupSize != numInputs)
    return emitError(llvm::formatv(
        "slice_rows requires one full-world group, but group_size is {0} "
        "for {1} inputs",
        groupSize, numInputs));
  return success();
}

//===----------------------------------------------------------------------===//
// AllGatherConcatOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherConcatOp::verify() {
  auto outputs = getOutputs();
  auto inputs = getInputs();
  auto signalBuffers = getSignalBuffers();

  if (inputs.empty())
    return emitError("allgather requires at least one input");
  if (signalBuffers.size() != inputs.size())
    return emitError(
        llvm::formatv("number of signal buffers should match number of inputs. "
                      "Expected {0} signal buffers but got {1}",
                      inputs.size(), signalBuffers.size()));
  if (outputs.size() != inputs.size())
    return emitError(llvm::formatv(
        "number of outputs must match number of inputs; expected {0} outputs, "
        "found {1}",
        inputs.size(), outputs.size()));

  int64_t numInputs = inputs.size();
  int64_t groupSize = getGroupSize();
  if (groupSize < 1)
    return emitError("group_size must be at least 1");
  if (numInputs % groupSize != 0)
    return emitError(llvm::formatv(
        "group_size must evenly divide the number of inputs ({0}), but got {1}",
        numInputs, groupSize));

  auto firstInputType = getInputTensor(0).getType();
  int64_t rank = firstInputType.getRank();
  for (int64_t i = 1; i < numInputs; ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
    if (static_cast<int64_t>(inputType.getRank()) != rank)
      return emitError("all inputs must have the same rank");
  }

  int64_t axis = normalizeAxis(getAxis(), rank);
  if (axis < 0 || axis >= rank)
    return emitError(
        llvm::formatv("axis {0} out of bounds for rank {1}", getAxis(), rank));

  // Dynamic dims are left to the kernel.
  for (int64_t devIdx = 0; devIdx < numInputs; ++devIdx) {
    auto outputType = getOutputTensor(devIdx).getType();
    if (outputType.getElementType() != firstInputType.getElementType())
      return emitError("output element type must match input element type");
    if (static_cast<int64_t>(outputType.getRank()) != rank)
      return emitError("output rank must match input rank");

    int64_t groupStart = devIdx / groupSize * groupSize;
    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t outputDim = outputType.getIntDim(dim);
      bool dynamic = mlir::ShapedType::isDynamic(outputDim);
      int64_t gathered = 0;
      for (int64_t src = groupStart; src < groupStart + groupSize; ++src) {
        int64_t inputDim = getInputTensor(src).getType().getIntDim(dim);
        if (mlir::ShapedType::isDynamic(inputDim)) {
          dynamic = true;
          continue;
        }
        if (dim == axis) {
          gathered += inputDim;
          continue;
        }
        if (!mlir::ShapedType::isDynamic(outputDim) && inputDim != outputDim)
          return emitError(llvm::formatv(
              "output {0} dimension {1} must match input {2} ({3} vs {4})",
              devIdx, dim, src, outputDim, inputDim));
      }
      if (dim == axis && !dynamic && gathered != outputDim)
        return emitError(llvm::formatv(
            "output {0} dimension {1} must be the sum of its group's inputs "
            "along the axis ({2}), but got {3}",
            devIdx, dim, gathered, outputDim));
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DistributedBroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult DistributedBroadcastOp::verify() {
  auto outputs = getOutputs();
  auto signalBuffers = getSignalBuffers();

  if (signalBuffers.empty())
    return emitError("broadcast requires at least one signal buffer");

  if (outputs.size() != signalBuffers.size())
    return emitError(
        llvm::formatv("number of outputs must match number of signal buffers. "
                      "Expected {0} outputs but got {1}",
                      signalBuffers.size(), outputs.size()));

  int64_t root = getRoot();
  int64_t numDevices = signalBuffers.size();
  if (root < 0 || root >= numDevices)
    return emitError(llvm::formatv(
        "root index {0} out of range for {1} devices", root, numDevices));

  auto inputType = cast<TensorType>(getInput().getType());
  // The input is the root's copy of the tensor, so it must already live on
  // the root's device. Devices can still be unassigned in graphs built in
  // memory, so only compare when both are known.
  auto rootDevice =
      cast<BufferType>(signalBuffers[root].getType()).getDeviceRef();
  auto inputDevice = inputType.getDeviceRef();
  if (inputDevice && rootDevice && inputDevice != rootDevice)
    return emitError(llvm::formatv(
        "the input must be on the root's device, {0}, but is on {1}",
        rootDevice, inputDevice));

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto outputType = getOutputTensor(i).getType();
    if (outputType.getShape() != inputType.getShape())
      return emitError("all outputs must have the same shape as the input");
    if (outputType.getElementType() != inputType.getElementType())
      return emitError(
          "all outputs must have the same element type as the input");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceRMSNormOp
//===----------------------------------------------------------------------===//

LogicalResult Composite::ReduceRMSNormOp::verify() {
  if (cast<TensorType>(getWeight().getType()).getRank() != 1)
    return emitOpError("weight must be rank 1");
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceRMSNormRoPEOp
//===----------------------------------------------------------------------===//

LogicalResult Composite::ReduceRMSNormRoPEOp::verify() {
  auto inputTy = cast<TensorType>(getInput().getType());
  if (!inputTy.getDeviceRef())
    return emitOpError("input must be on a device");
  if (inputTy.getDeviceRef().getLabel() != M::kGPULabel)
    return emitOpError("mo.composite.rms_norm_rope is only supported on GPU");
  return success();
}

//===----------------------------------------------------------------------===//
// QuantizeDynamicScaledFP8Op
//===----------------------------------------------------------------------===//

LogicalResult Composite::QuantizeDynamicScaledFP8Op::verify() {
  auto inputTy = cast<TensorType>(getInput().getType());
  if (inputTy.getRank() != 2)
    return emitOpError("the input must be [rows, cols]");
  if (cast<TensorType>(getOutput().getType()).getShape() != inputTy.getShape())
    return emitOpError("the output must have the same shape as the input");
  if (cast<TensorType>(getScales().getType()).getRank() != 2)
    return emitOpError("the scales must be rank 2");

  int64_t groupSize = getGroupSizeOrPerToken().getSExtValue();
  if (groupSize != -1 && groupSize <= 0)
    return emitOpError("group_size_or_per_token must be -1 or positive");

  if (std::optional<ElementsAttr> floor = getAmaxFloor()) {
    auto values = floor->tryGetValues<Attribute>();
    auto value = values ? dyn_cast<FloatAttr>(*values->begin()) : nullptr;
    if (!value || !value.getValue().isFinite() || value.getValue().isNegative())
      return emitOpError("amax_floor must be finite and non-negative");
  }
  return success();
}
