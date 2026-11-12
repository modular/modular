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
// This file implements the verifiers of the MOGG dialect operations.
//
//===----------------------------------------------------------------------===//
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOGGDialect/Support/MojoTraits.h"
#include "GraphCompiler/MOGGDialect/Support/ProfileScopeLabels.h"
#include "GraphCompiler/MOGGDialect/Support/SymbolNames.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/NameMangling.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include <cmath>
#include <cstdlib>
#include <future>
#include <vector>

using namespace M;
using namespace MOGG;

/// Common verifySymbolUses logic for ops implementing KernelCallInterface.
/// Looks up the KernelDeclOp and delegates to `check` for the op-specific
/// function presence check.
static LogicalResult verifySymbolUsesForKernelDeclCall(
    KernelCallInterface op, SymbolTableCollection &symbolTable,
    llvm::function_ref<LogicalResult(KernelDeclOp)> check) {
  auto symAttr = op.getKernelSymNameAttr();
  auto decl = symbolTable.lookupNearestSymbolFrom<KernelDeclOp>(op, symAttr);
  if (!decl)
    return op->emitOpError("'")
           << symAttr.getValue() << "' does not refer to a kernel.decl op";
  return check(decl);
}

// A launch takes a static or a run-time width limit, not both.
template <typename LaunchOpT>
static LogicalResult verifyMaxSimdWidth(LaunchOpT op) {
  if (op.getMaxSimdWidth() && op.getDynamicMaxSimdWidth())
    return op.emitOpError("expected a static or a dynamic max_simd_width, not "
                          "both");
  return success();
}

LogicalResult CallForeachOp::verify() { return verifyMaxSimdWidth(*this); }

LogicalResult RoutineOp::verify() {
  // A chain result, if present, is last (mirrors `IterElementwiseOp::verify()`
  // / `RoutineOp::maybeGetOutChain()`) and never gets a body block argument.
  mlir::ResultRange outputs = getOutputs();
  for (auto [i, output] : llvm::enumerate(outputs))
    if (isa<MO::ChainType>(output.getType()) && i + 1 != outputs.size())
      return emitOpError() << "only the last output may be a chain";

  // A by-value opaque result is allocated inside the kernel body and returned;
  // `lower` maps a single one onto the region's results.
  if (llvm::count_if(outputs, [](Value output) {
        return isa<MO::OpaqueType>(output.getType());
      }) > 1)
    return emitOpError("returns more than one opaque result");

  // The body block arguments are the codegen handles: one per input then
  // destination-passing output, then the !mogg.dims, then one !mogg.context.
  // A tensor-like entry gets a !mogg.pointer; an opaque input carries its own
  // type across. Results that leave by value get none -- a shape function's
  // `!mogg.shape_func_result`, or an `!mo.opaque`.
  SmallVector<Type> boundaryTypes(getInputs().getTypes());
  for (Value output : outputs)
    if (MO::dynCastFromTensorLike(output.getType()))
      boundaryTypes.push_back(output.getType());
  const size_t numTensors = boundaryTypes.size();
  auto args = getBody().front().getArguments();

  if (args.size() < numTensors + 1)
    return emitOpError() << "expected at least " << (numTensors + 1)
                         << " body block arguments (one per input "
                            "and output plus a trailing !mogg.context), "
                            "got "
                         << args.size();

  for (size_t i = 0; i < numTensors; ++i) {
    if (isa<MO::OpaqueType>(boundaryTypes[i])) {
      if (args[i].getType() != boundaryTypes[i])
        return emitOpError()
               << "body block argument #" << i << " must be "
               << boundaryTypes[i] << " to match its opaque input/output, got "
               << args[i].getType();
      continue;
    }
    if (!isa<PointerType>(args[i].getType()))
      return emitOpError() << "body block argument #" << i
                           << " must be a !mogg.pointer (one per input then "
                              "output tensor)";
  }

  size_t i = numTensors;
  while (i < args.size() && isa<DimType>(args[i].getType()))
    ++i;
  // After the dims come one or more trailing `!mogg.context`: exactly one for
  // an ordinary kernel, one per device for a distributed kernel that takes a
  // `DeviceContextArray`.
  if (i >= args.size())
    return emitOpError() << "body block arguments after the pointers must be "
                            "zero or more !mogg.dim followed by one or more "
                            "!mogg.context";
  for (; i < args.size(); ++i)
    if (!isa<ContextType>(args[i].getType()))
      return emitOpError() << "body block argument #" << i
                           << " must be a !mogg.context (the trailing args "
                              "after the dims are all contexts)";

  // One kernel, one call -- counting the `mogg.call` outlining leaves.
  size_t numCalls = 0;
  for (Operation &bodyOp : getBody().front())
    if (bodyOp.hasTrait<CallLike>())
      ++numCalls;
  if (numCalls != 1)
    return emitOpError() << "expected exactly one call in the body, got "
                         << numCalls;

  // A shape function returns exactly the dims it declares.
  size_t numShapeReturning = 0;
  uint64_t shapeRank = 0;
  for (Operation &bodyOp : getBody().front())
    for (Value result : bodyOp.getResults())
      if (auto shapeTy = dyn_cast<ShapeFuncResultType>(result.getType())) {
        ++numShapeReturning;
        shapeRank = shapeTy.getRank();
      }
  if (numShapeReturning > 1)
    return emitOpError() << "expected at most one shape-returning call in the "
                            "body, got "
                         << numShapeReturning;
  if (getOutputParamDecls().empty() != (numShapeReturning == 0))
    return emitOpError() << "outputParamDecls and a shape-returning call must "
                            "appear together";
  if (numShapeReturning == 1 && shapeRank != getOutputParamDecls().size())
    return emitOpError() << "shape result rank " << shapeRank
                         << " does not match " << getOutputParamDecls().size()
                         << " declared output parameter(s)";

  return success();
}

//===----------------------------------------------------------------------===//
// CallBuildGraphOp
//===----------------------------------------------------------------------===//

LogicalResult CallBuildGraphOp::verify() {
  // The launch argument is the dedicated `$builder` operand, so a context must
  // not appear among the kernel inputs/outputs (that is the `call.execute`
  // shape this op replaces).
  auto isContext = [](Value v) { return isa<MOGG::ContextType>(v.getType()); };
  if (llvm::any_of(getInputs(), isContext) ||
      llvm::any_of(getOutputs(), isContext))
    return emitError("mogg.call.build_graph takes a device-graph builder as "
                     "its launch argument, not a context");
  return success();
}

LogicalResult
CallBuildGraphOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForKernelDeclCall(
      *this, symbolTable, [this](KernelDeclOp decl) -> LogicalResult {
        if (!decl.getBuildGraph())
          return emitOpError("kernel '")
                 << getKernelSymName() << "' has no 'build_graph' function";
        return success();
      });
}

LogicalResult FuncOp::verify() {
  // FunctionOpInterface already verifies the arguments.
  return success();
}

// Shared verifier backing `MOGG_TensorCreateInterface::verifyTrait` (declared
// in MOGGInterfaces.h). Lives here, not in the generated interface header,
// because it names `!mogg.buffer`. See the interface for why only the
// buffer-implies-mutable direction is enforced.
llvm::LogicalResult
M::MOGG::verifyTensorCreateInterfaceMutability(Operation *op) {
  auto iface = cast<TensorCreateInterface>(op);
  if (isa<BufferType>(op->getResult(0).getType()) && !iface.getMut())
    return op->emitOpError("!mogg.buffer result requires mut = true");
  return success();
}

//===----------------------------------------------------------------------===//
// CallExecuteDeviceOp
//===----------------------------------------------------------------------===//

LogicalResult
CallExecuteDeviceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl =
      symbolTable.lookupNearestSymbolFrom<KernelDeclOp>(*this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a kernel.decl op";
  if (!decl.getExecuteDevice())
    return emitOpError("kernel '")
           << getCallee() << "' has no 'execute_device' function";
  return success();
}

// Each `mogg.call.execute_device` checks its own callee, so this needs no
// symbol lookup.
LogicalResult GraphBlobOp::verifyRegions() {
  Value ctx;
  for (Operation &op : getBody().getOps()) {
    if (!op.hasTrait<MOGG::CallLike>())
      continue;
    auto call = dyn_cast<CallExecuteDeviceOp>(op);
    if (!call)
      return emitOpError("calls kernels only through "
                         "'mogg.call.execute_device', but holds '")
             << op.getName() << "'";
    if (!ctx)
      ctx = call.getDeviceContext();
    else if (call.getDeviceContext() != ctx)
      return emitOpError("calls kernels on more than one device context");
  }
  if (!ctx)
    return emitOpError("calls no kernel");
  return success();
}

/// Checks that `tile` moves elements of `tensor`'s type. A capture is a block
/// argument with no creating op, so only a tensor whose creating op is in
/// view is checked.
static LogicalResult verifyTileElementType(Operation *op, Value tensor,
                                           TileType tile) {
  auto create = tensor.getDefiningOp<TensorCreateInterface>();
  if (!create || create.getElementType() == tile.getElementType())
    return success();
  return op->emitOpError("tile element type ")
         << tile.getElementType() << " does not match tensor element type "
         << create.getElementType();
}

LogicalResult TileLoadOp::verify() {
  return verifyTileElementType(*this, getTensor(), getResult().getType());
}

LogicalResult TileStoreOp::verify() {
  return verifyTileElementType(*this, getTensor(), getValue().getType());
}

LogicalResult CallRowwiseOp::verify() {
  if (failed(verifyMaxSimdWidth(*this)))
    return failure();
  Block &body = getBody().front();
  bool afterMap = false;
  for (Operation &op : body) {
    if (auto store = dyn_cast<TensorStoreOp>(op);
        store && store.getIndex() != body.getArguments().back())
      return emitOpError("stores outside its row index, its emit and its map "
                         "phases");
    // A fold after a map phase would need a Row API pass per phase.
    if (afterMap && isa<RowwiseReduceOp>(op))
      return emitOpError("has a fold after a map phase");
    afterMap |= isa<RowwiseMapOp>(op);
  }
  // `Row.emit` isn't phase-aware, so it can't run in a body that writes
  // every element.
  if (afterMap && !body.getOps<RowwiseEmitOp>().empty())
    return emitOpError("has both per-row stores and a map phase");
  // A body without a map phase launches as one phase, whose split-row tiers
  // join one fold per row.
  if (!afterMap && llvm::range_size(body.getOps<RowwiseReduceOp>()) > 1)
    return emitOpError("has several folds and no map phase");
  return success();
}

LogicalResult RowwiseReduceOp::verify() {
  if (getNumResults() == 0)
    return emitOpError("must have at least one result");
  Block &body = getBody().front();
  if (body.getNumArguments() != 1)
    return emitOpError("body takes exactly one argument, the index");
  auto yield = body.empty() ? YieldOp() : dyn_cast<YieldOp>(body.back());
  if (!yield || yield.getNumOperands() != 1)
    return emitOpError("body must yield one contribution");
  if (!isa<SimdType>(yield.getOperand(0).getType()))
    return emitOpError("body must yield a !mogg.simd contribution, got ")
           << yield.getOperand(0).getType();
  if (getBody()
          .walk([](TensorStoreOp) { return WalkResult::interrupt(); })
          .wasInterrupted())
    return emitOpError("stores; only the emit and the map phases store");
  return success();
}

LogicalResult RowwiseTileOp::verify() {
  Block &body = getBody().front();
  if (body.getNumArguments() != 1)
    return emitOpError("body takes exactly one argument, the index");
  auto yield = body.empty() ? YieldOp() : dyn_cast<YieldOp>(body.back());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0).getType() != getType())
    return emitOpError("body must yield one value of the result type");
  if (body.walk([](TensorStoreOp) { return WalkResult::interrupt(); })
          .wasInterrupted())
    return emitOpError("must not store; it only reads the row");
  for (Operation *op = getOperation()->getPrevNode(); op;
       op = op->getPrevNode())
    if (!isa<ConstantScalarOp>(op))
      return emitOpError("must come first in the row, after constants only");
  return success();
}

LogicalResult RowwiseMapOp::verify() {
  if (getBody().front().getNumArguments() != 1)
    return emitOpError("body takes exactly one argument, the index");
  return success();
}

LogicalResult TensorCreatePrologueOp::verify() {
  Block &body = getBody().front();
  auto yield = body.empty() ? YieldOp() : dyn_cast<YieldOp>(body.back());
  if (!yield)
    return success();
  for (Type type : yield.getOperandTypes())
    if (!isa<SimdType>(type))
      return emitOpError("body must yield the !mogg.simd value the kernel "
                         "reads, got ")
             << type;
  return success();
}

LogicalResult TensorCreateEpilogueOp::verify() {
  Block &body = getBody().front();
  if (body.empty())
    return emitOpError("requires a non-empty body");
  // Both shapes need the store-target capture: the store lane writes through
  // it, the compute lane leaves it for emit to bind onto.
  if (getCaptures().empty())
    return emitOpError("requires at least one (store-target) capture");
  if (body.getNumArguments() <
      2 + getCaptures().size() + getShapeCaptures().size())
    return emitOpError("body takes a value, an index, one arg per capture, and "
                       "one arg per shape capture");
  if (!isa<SimdType>(body.getArgument(0).getType()))
    return emitOpError("the value argument must be a !mogg.simd, got ")
           << body.getArgument(0).getType();

  // `mogg.yield` is a terminator, so it can only be the last op: its presence
  // is what makes this a compute lambda rather than a store one.
  auto yieldOp = dyn_cast<YieldOp>(&body.back());
  if (!yieldOp)
    return success();

  if (!getComputeOutputArg())
    return emitOpError("a yield-terminated epilogue body is a compute lambda, "
                       "which requires `computeOutputArg`");
  if (yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0).getType() != body.getArgument(0).getType())
    return emitOpError("a compute-lambda epilogue must yield exactly the "
                       "value type it was handed");
  return success();
}

/// Whether `value` is one of `body`'s capture arguments, which start at
/// `firstCapture`.
static bool isCaptureArgOf(Block &body, unsigned firstCapture, Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  return arg && arg.getOwner() == &body && arg.getArgNumber() >= firstCapture;
}

/// A tile body addresses whole tiles at its own index, so every load reads a
/// capture at `index`, and there is no index transform to read through.
static LogicalResult verifyTileBodyLoads(Block &body, unsigned firstCapture,
                                         Value index, StringRef container) {
  for (Operation &op : body) {
    if (isa<TensorLoadOp, TensorStoreOp>(op))
      return op.emitOpError() << "in a " << container
                              << " must move a tile, with tile.load or "
                                 "tile.store";
    if (auto load = dyn_cast<TileLoadOp>(op)) {
      if (!isCaptureArgOf(body, firstCapture, load.getTensor()) ||
          load.getCoord() != index)
        return load.emitOpError() << "in a " << container
                                  << " must read a capture at the loop index";
    } else if (isa<IndexReshapeOp, IndexSliceSelectOp>(op)) {
      return op.emitOpError()
             << "cannot transform the index of a " << container;
    }
  }
  return success();
}

LogicalResult TileCreateEpilogueOp::verify() {
  Block &body = getBody().front();
  if (body.empty())
    return emitOpError("requires a non-empty body");
  if (getCaptures().empty())
    return emitOpError("requires at least one (store-target) capture");
  if (body.getNumArguments() != 2 + getCaptures().size())
    return emitOpError("body takes a value, an index, and one arg per capture");
  constexpr unsigned kFirstCapture = 2;
  Value index = body.getArgument(1);
  Value storeTarget = body.getArgument(kFirstCapture);
  if (failed(verifyTileBodyLoads(body, kFirstCapture, index, "tile epilogue")))
    return failure();
  Type tileType = TileType::get(getElementType());
  Value value = body.getArgument(0);
  if (value.getType() != tileType)
    return emitOpError("the kernel's tile is a ") << tileType;

  // The lane fixes the body shape, so the two are cross-checked.
  Operation &last = body.back();
  if (getLane() == TileFusionLane::Compute) {
    auto yield = dyn_cast<YieldOp>(last);
    if (!yield || yield.getNumOperands() != 1)
      return emitOpError("a compute lane ends in a yield of one value");
    if (yield.getOperand(0).getType() != tileType)
      return emitOpError("a compute lane yields a ") << tileType;
    return success();
  }
  if (!body.getOps<YieldOp>().empty())
    return emitOpError("a store lane does not yield");
  auto store = dyn_cast<TileStoreOp>(last);
  if (!store || store.getTensor() != storeTarget || store.getCoord() != index)
    return emitOpError("a store lane ends in a store to the store-target "
                       "capture at the loop index");
  return success();
}

LogicalResult CallForeachTileOp::verify() {
  Block &body = getBody().front();
  size_t numCaptures = getCapturing().size();
  if (body.getNumArguments() != numCaptures + 1)
    return emitOpError("body takes one arg per capture, then the index");
  if (numCaptures < 2)
    return emitOpError(
        "requires an output capture and at least one input capture");
  Value index = body.getArguments().back();
  if (failed(
          verifyTileBodyLoads(body, /*firstCapture=*/0, index, "tile foreach")))
    return failure();
  unsigned numStores = 0;
  for (auto store : body.getOps<TileStoreOp>()) {
    ++numStores;
    if (!isCaptureArgOf(body, /*firstCapture=*/0, store.getTensor()) ||
        store.getCoord() != index)
      return store.emitOpError(
          "in a tile foreach must write a capture at the loop index");
  }
  if (numStores != 1)
    return emitOpError("body stores exactly once, got ") << numStores;
  return success();
}

LogicalResult AsyncRoutineOp::verify() {
  Region &body = getBody();

  if (getOperands().size() != body.getNumArguments())
    return emitError("number of operands and body arguments must match");

  // There are a large number of unittests which use an empty
  // InOutSignatureAttr. For now, allow them to pass verification.
  // TODO: Update unittests and remove this guard.
  if (getInOutSignature().size() != 0) {
    if (getOperands().size() != getInOutSignature().size()) {
      return emitError() << "inOutSignature size ("
                         << getInOutSignature().size()
                         << ") does not match number of operands ("
                         << getOperands().size() << ")";
    }
  }

  // Each body block argument is a `!mogg.async<T>` wrapping its operand type
  // `T`, unpacked via `mogg.async.unpack`; one consumed as the raw async slot
  // (e.g. a context) may be retyped to plain `T` during ABI lowering.
  for (auto [operandType, argType] :
       llvm::zip(getOperandTypes(), body.getArgumentTypes())) {
    if (argType == operandType)
      continue;
    auto asyncType = dyn_cast<AsyncType>(argType);
    if (!asyncType || asyncType.getElementType() != operandType)
      return emitError("each region body argument must be the corresponding "
                       "operand type or a mogg.async of it");
  }

  if (!getExecutionDevice())
    return emitError("Compiled regions must have a device");

  return success();
}

//===----------------------------------------------------------------------===//
// Shared verification for CallOp / CallAsyncOp
//===----------------------------------------------------------------------===//

static FailureOr<FuncOp>
verifyKernelStubCall(Operation *op, SymbolTableCollection &symbolTable,
                     FlatSymbolRefAttr calleeSymbol,
                     TypeRange callerOperandTypes) {
  auto calleeFunc =
      symbolTable.lookupNearestSymbolFrom<FuncOp>(op, calleeSymbol);
  if (!calleeFunc) {
    return op->emitError(llvm::formatv("There is no kernel stub named '{}'",
                                       calleeSymbol.getValue()));
  }

  auto calleeArgumentTypes = calleeFunc.getArgumentTypes();

  if (callerOperandTypes.size() != calleeArgumentTypes.size()) {
    return op->emitError(
        llvm::formatv("Kernel stub '{0}' expects {1} arguments, but got {2}",
                      calleeSymbol.getValue(), calleeArgumentTypes.size(),
                      callerOperandTypes.size()));
  }

  for (auto [i, calleeArgType, callerOperandType] :
       llvm::enumerate(calleeArgumentTypes, callerOperandTypes)) {
    if (calleeArgType == callerOperandType)
      continue;

    if (isa<MOGG::ContextType>(calleeArgType) &&
        isa<MGP::DeviceContextType>(callerOperandType))
      continue;

    return op->emitError(llvm::formatv(
        "Kernel stub '{0}' expects argument {1} to have type {2}, but got "
        "{3}",
        calleeSymbol.getValue(), i, calleeArgType, callerOperandType));
  }

  return calleeFunc;
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto calleeFunc = verifyKernelStubCall(*this, symbolTable, getCalleeSymbol(),
                                         getOperandTypes());
  if (failed(calleeFunc))
    return failure();

  // The MOGG ABI is the advanced-fusion outlining path (a stub outlined from a
  // `mogg.routine`); it reuses `mogg.call` as its call site.
  if (calleeFunc->getAbi() != ABIKind::MOGG)
    return emitError("Call site should be in the MOGG ABI");

  return success();
}

LogicalResult
CallAsyncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyKernelStubCall(*this, symbolTable, getCalleeSymbol(),
                              getOperandTypes());
}

//===----------------------------------------------------------------------===//
// CallViewOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CallElementwiseOp
//===----------------------------------------------------------------------===//

LogicalResult CallElementwiseOp::verify() {
  auto isTile = [](Value v) { return isa<TileType>(v.getType()); };
  if (llvm::none_of(getOutputs(), isTile)) {
    if (llvm::any_of(getInputs(), isTile))
      return emitOpError("a tile input requires a tile result");
    for (Value v : llvm::concat<Value>(getInputs(), getOutputs()))
      if (!isa<SimdType>(v.getType()))
        return emitOpError("a SIMD call takes and returns !mogg.simd values, "
                           "got ")
               << v.getType();
    return success();
  }
  // The `TileTensor` overload computes on whole tiles. A literal is the one
  // scalar it takes, which emit passes alongside the tiles.
  if (!llvm::all_of(getOutputs(), isTile))
    return emitOpError("cannot mix tile and SIMD results");
  for (Value input : getInputs())
    if (!isTile(input) && (!input.getDefiningOp<ConstantScalarOp>() ||
                           isa<SimdType>(input.getType())))
      return emitOpError("a tile call takes tiles or mogg.constant.scalar "
                         "literals, got ")
             << input.getType();
  if (getIndex())
    return emitOpError("a tile call takes no index");
  return success();
}

LogicalResult
CallElementwiseOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifySymbolUsesForKernelDeclCall(
      *this, symbolTable, [this](KernelDeclOp decl) -> LogicalResult {
        if (!decl.getElementwise())
          return emitOpError("kernel '")
                 << getCallee() << "' has no 'elementwise' function";
        return success();
      });
}

//===----------------------------------------------------------------------===//
// AsTupleOp
//===----------------------------------------------------------------------===//

// MOGG_KernelCallInterface methods

LogicalResult
CallShapeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<ShapeFunctionDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a shape_func.decl op";
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeResultGetItemOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeResultGetItemOp::verify() {
  auto shapeType = dyn_cast<MOGG::ShapeFuncResultType>(getShape().getType());
  if (!shapeType) {
    return emitOpError("shape operand must be of ShapeFuncResultType");
  }

  int64_t index = getIndex().getValue().getSExtValue();
  int64_t rank = shapeType.getRank();

  if (index < 0 || index >= rank) {
    return emitOpError("index ")
           << index << " is out of bounds for shape result with rank " << rank;
  }

  return success();
}

LogicalResult ConstantScalarOp::verify() {
  Type resultType = getResult().getType();
  auto simd = dyn_cast<SimdType>(resultType);
  Type elementType = simd ? simd.getElementType() : resultType;
  if (getValue().getType() != elementType)
    return emitOpError("value attribute type ")
           << getValue().getType() << " does not match element type "
           << elementType;
  if (simd)
    return success();
  // Only the literal a tile call takes next to its tiles stays a scalar.
  for (Operation *user : getResult().getUsers()) {
    auto call = dyn_cast<CallElementwiseOp>(user);
    if (!call || !llvm::all_of(call.getOutputs(), [](Value v) {
          return isa<TileType>(v.getType());
        }))
      return emitOpError("a scalar result may only feed a tile call; use "
                         "!mogg.simd in a SIMD body");
  }
  return success();
}
