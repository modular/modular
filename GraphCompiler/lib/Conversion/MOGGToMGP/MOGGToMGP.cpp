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
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MGPDialect/Transforms/FixupIsolatedFromAbove.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/TransformUtils/SCCUtils.h"
#include "Support/AssertStream.h"
#include "Support/ML/CommonAttributes.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringMap.h"

#include <deque>
#include <numeric>

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_MOGGTOMGP
#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h.inc"

} // namespace M::MGP

//===----------------------------------------------------------------------===//
// MOGraphLowering implementation
//===----------------------------------------------------------------------===//

namespace {

struct IndexLowering {
  MGP::MOOpLoweringState &opLoweringState;

  // Arithmetic helpers
  auto constant(int64_t val) -> Value {
    return opLoweringState.create<mlir::index::ConstantOp>(val);
  }

  auto mul(Value a, Value b) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    return opLoweringState.create<mlir::index::MulOp>(indexTy, a, b);
  }

  auto add(Value a, Value b) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    return opLoweringState.create<mlir::index::AddOp>(indexTy, a, b);
  }

  auto div(Value a, Value b) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    return opLoweringState.create<mlir::index::DivSOp>(indexTy, a, b);
  }

  // `pop.select` takes a `!kgen.scalar<bool>` condition, but `index.cmp`
  // produces an `i1`. Convert the builtin `i1` to `!kgen.scalar<bool>`.
  auto asScalarBool(Value i1Cond) -> Value {
    auto boolTy = KGEN::SIMDType::get(opLoweringState.builder.getContext(), 1,
                                      KGEN::KGENDType::kBool);
    return opLoweringState.create<KGEN::POP::CastFromBuiltinOp>(boolTy, i1Cond);
  }

  auto ifPositive(Value check, Value yes, Value no) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    auto cond = opLoweringState.create<mlir::index::CmpOp>(
        mlir::index::IndexCmpPredicate::SLT, check, constant(0));
    return opLoweringState.create<KGEN::POP::SelectOp>(
        indexTy, asScalarBool(cond), no, yes);
  }

  auto ifZero(Value check, Value yes, Value no) -> Value {
    auto indexTy = opLoweringState.builder.getIndexType();
    auto cond = opLoweringState.create<mlir::index::CmpOp>(
        mlir::index::IndexCmpPredicate::EQ, check, constant(0));
    return opLoweringState.create<KGEN::POP::SelectOp>(
        indexTy, asScalarBool(cond), yes, no);
  }
};

/// Lowers one mo.graph to an mgp.model with an 'init' phase, an 'execute'
/// phase, and a model descriptor.
class MOGraphLowering {
public:
  MOGraphLowering(MO::GraphOp graph, llvm::StringRef compiledFrameworkLabel,
                  size_t minimumCpuAlignment,
                  mlir::SymbolTable &moduleSymbolTable)
      : loweringState(graph, compiledFrameworkLabel, minimumCpuAlignment,
                      moduleSymbolTable) {}

  /// Lower the mo.graph to mgp.model.
  LogicalResult run();

  // Establish type builders for all the (virtual) devices needed by the model.
  LogicalResult setupTypeBuildersForAllDevices(MO::GraphOp graph);

  // Establish buffer builders for all the (virtual) devices needed by the
  // model. Each device will have its 'runtime' created in the 'init' block
  // and held in the !mgp.context.
  CompileDiagnosticOrSuccess setupBufferBuildersForAllDevices();

  LogicalResult createModelOp();

  /// Lowers a MO::ListType to a MGP::ListType.
  MGP::ListType lowerListType(MO::ListType listTy);

  /// Clones a trivial operations that does not need transformation. E.g.,
  /// `index.constant`
  LogicalResult cloneTrivialOp(Operation *trivialOp);

  /// Returns an attribute to represent elementAttr for the remainder of
  /// compilation:
  ///  - We commit to a specific byte layout, ie from here on the attribute
  ///    is considered an ArrayRef<uint8_t> with no further interpretation.
  ///  - Large attributes are outlined as dense_resources.
  ///  - The required runtime alignment is captured as an !M.aligned_bytes type.
  ///  - The runtime alignment will be at least 'minimumCpuAlignment' from
  ///    the pass options.
  ///  - If elementAttr already has an !M.aligned_bytes type annotation, that
  ///    may increase the alignment. Otherwise, if the attribute itself has some
  ///    indication of alignment (eg a dense_resource's blob's alignment) that
  ///    may increase the alignment.
  ///  - However the result's underlying data may not respect the alignment.
  ///    Instead we assume alignment is enforced on serialization to MEF.
  ///    This helps us avoid gratuitous memory copies.
  ///  - No data is copied.
  ErrorOr<ElementsAttr> lowerElementsAttr(ElementsAttr elementsAttr) {
    return loweringState.lowerElementsAttr(elementsAttr);
  }

  /// Per-op lowering overloads. Each concrete MO/MOGG/KGEN op type has a
  /// corresponding `lower()` overload; `lowerOp()` dispatches to these via
  /// TypeSwitch.
  LogicalResult lower(MO::OutputOp op);
  LogicalResult lower(MO::YieldOp op);
  LogicalResult lower(MO::WhileConditionOp op);
  LogicalResult lower(MO::IndexToTensorOp op);
  LogicalResult lower(MO::Shape::ToTensorOp op);
  LogicalResult lower(MO::Shape::FromTensorOp op);
  LogicalResult lower(MO::RebindOp op);
  LogicalResult lower(MO::ConstantOp op);
  LogicalResult lower(MO::ConstantExternalOp op);
  LogicalResult lower(MO::ConstantScalarOp op);
  LogicalResult lower(MO::DebugPrintOp op);
  LogicalResult lower(MO::DebugTensorPrintOp op);
  LogicalResult lower(MO::TransferOp op);
  LogicalResult lower(MO::StableCreateOp op);
  LogicalResult lower(MO::IfOp op);
  LogicalResult lower(MO::WhileOp op);
  LogicalResult lower(MO::GuardOp op);
  LogicalResult lower(MO::ChainCreateOp op);
  LogicalResult lower(MO::MutableStoreOp op);
  LogicalResult lower(MO::MutableLoadOp op);
  LogicalResult lower(MO::BufferCreateOp op);
  LogicalResult lower(MO::BufferTransferOp op);
  LogicalResult lower(MO::CallOp op);
  LogicalResult lower(MO::ParallelOp op);
  LogicalResult lower(MO::SequenceOp op);
  LogicalResult lower(MO::TensorBundleOp op);
  LogicalResult lower(MO::TensorUnbundleOp op);

  /// Param ops need the enclosing region to resolve parameter expressions.
  LogicalResult lowerParam(Region *parentRegion, KGEN::ParamDeclareOp op);
  LogicalResult lowerParam(Region *parentRegion, MOSH::ParamToValueOp op);
  LogicalResult lowerParam(Region *parentRegion, MOSH::ParamFromValueOp op);

  /// Interface-based lowering for externally-defined ops.
  LogicalResult lowerInterface(MGP::LowerableToMGPInterface op);

private:
  /// Lowers a single Operation via type-based dispatch.
  LogicalResult lowerOp(Operation &op, mlir::Region *parentRegion);

  /// Lowers a Block
  LogicalResult lowerBlock(mlir::Block *block);

  MGP::MOGraphLoweringState loweringState;
};

} // namespace

static SmallVector<DeviceRefAttr> getAllDevicesUsedInGraph(MO::GraphOp graph) {
  llvm::DenseSet<DeviceRefAttr> devices;

  // The host device may not appear in the graph.
  DeviceRefAttr hostDevice = DeviceRefAttr::getDefault(graph->getContext());
  devices.insert(hostDevice);

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](DeviceRefAttr attr) { devices.insert(attr); });
  graph->walk([&](Operation *op) {
    walker.walk(op->getAttrDictionary());
    for (Type type : op->getOperands().getTypes())
      walker.walk(type);
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions()) {
      for (Type type : region.getArgumentTypes())
        walker.walk(type);
    }
  });

  // Host first, then the rest by string representation, for a stable order.
  SmallVector<DeviceRefAttr> sorted = llvm::to_vector(devices);
  llvm::sort(sorted.begin(), sorted.end(),
             [hostDevice](DeviceRefAttr left, DeviceRefAttr right) {
               if (left == hostDevice)
                 return true;
               if (right == hostDevice)
                 return false;
               return left.toString() < right.toString();
             });
  return sorted;
}

LogicalResult
MOGraphLowering::setupTypeBuildersForAllDevices(MO::GraphOp graph) {
  for (DeviceRefAttr device : getAllDevicesUsedInGraph(graph))
    loweringState.registerTypeBuilder(device);
  return success();
}

CompileDiagnosticOrSuccess MOGraphLowering::setupBufferBuildersForAllDevices() {
  DeviceRefAttr hostDevice =
      DeviceRefAttr::getDefault(loweringState.getModelOp()->getContext());

  auto makeBufferBuilder =
      [&](DeviceRefAttr targetDevice) -> CompileDiagnosticOrSuccess {
    MGP::MGPBufferBuilder *hostBufferBuilder;
    auto bufferBuilder = std::make_unique<MGP::MGPBufferBuilder>(targetDevice);
    if (hostDevice == targetDevice) {
      hostBufferBuilder = bufferBuilder.get();
    } else {
      hostBufferBuilder = &loweringState.getBufferBuilder(hostDevice);
    }
    auto inChain =
        cast<TypedValue<MEF::ChainType>>(loweringState.getInitChain());
    auto chain = bufferBuilder->setupForModel(
        loweringState.getModelOp(), hostBufferBuilder,
        loweringState.getInitOpBuilder(), loweringState.getGraph()->getLoc(),
        inChain);
    loweringState.setInitChain(chain);
    loweringState.registerBufferBuilder(targetDevice, std::move(bufferBuilder));
    return success();
  };

  if (auto err = makeBufferBuilder(hostDevice); err.isError())
    return err;

  for (auto device : getAllDevicesUsedInGraph(loweringState.getGraph())) {
    if (device != hostDevice) {
      if (auto err = makeBufferBuilder(device); err.isError())
        return err;
    }
  }

  return success();
}

// Create the mgp.model and the regions for the init/execute phases.
LogicalResult MOGraphLowering::createModelOp() {
  MO::GraphOp &graph = loweringState.getGraph();
  OpBuilder b(graph);

  // Establish type builders for all the devices.
  // CAUTION: Must be done before any types are lowered, eg the next
  // paragraph.
  if (failed(setupTypeBuildersForAllDevices(graph)))
    return failure();

  // Convert the tensor types for the model signature over.
  SmallVector<Type> inputTypes, outputTypes;
  if (failed(loweringState.getTypeConverter().convertTypes(
          graph.getArgumentTypes(), inputTypes))) {
    return graph.emitOpError("failed to convert mo graph input types")
           << graph.getArgumentTypes() << " is not supported";
  } else if (failed(loweringState.getTypeConverter().convertTypes(
                 graph.getResultTypes(), outputTypes))) {
    return graph.emitOpError("failed to convert mo graph output types -- ")
           << graph.getResultTypes() << " is not supported";
  }

  // Preserve the MO::GraphOp argument locations in the MGP::ModelOp 'execute'
  // block argument locations.
  SmallVector<Location> inputLocs;
  for (BlockArgument arg : graph.getBody().front().getArguments())
    inputLocs.push_back(arg.getLoc());

  // Collect additional attributes to include with the MGP::ModelOp to guide
  // the MEF executor.
  SmallVector<NamedAttribute> extraAttrs;

  ArrayAttr argNamesAttr =
      graph->getAttrOfType<ArrayAttr>(kMgpModelArgumentNames);
  ArrayAttr resultNamesAttr =
      graph->getAttrOfType<ArrayAttr>(kMgpModelResultNames);
  ASSERT_STREAM(
      (argNamesAttr == nullptr) == (resultNamesAttr == nullptr),
      << "expected either both or neither of the `argument_names` and "
         "`result_names` attributes.");
  if (argNamesAttr) {
    extraAttrs.emplace_back(b.getStringAttr(kMgpModelArgumentNames),
                            argNamesAttr);
    extraAttrs.emplace_back(b.getStringAttr(kMgpModelResultNames),
                            resultNamesAttr);
  }

  ErrorOrSuccess err(success());
  auto devices = llvm::map_to_vector(getAllDevicesUsedInGraph(graph),
                                     [&err](DeviceRefAttr attr) {
                                       auto refOr = toRuntimeDeviceRef(attr);
                                       if (refOr.isError()) {
                                         err = refOr.takeError();
                                         return DeviceRef();
                                       } else {
                                         return refOr.takeValue();
                                       }
                                     });
  if (err.isError())
    return err;

  auto ctx = graph->getContext();
  SmallDenseMap<DeviceRefAttr, IntegerAttr> deviceIndexMap;
  auto idxType = IntegerType::get(ctx, 32, IntegerType::Unsigned);
  for (const auto &[idx, deviceRef] :
       llvm::enumerate(getAllDevicesUsedInGraph(graph)))
    deviceIndexMap[deviceRef] = IntegerAttr::get(idxType, idx);
  auto populateDeviceIndicesExtraAttr = [&](const char *label,
                                            SmallVector<Type> types) {
    llvm::SmallVector<Attribute> deviceIndices;
    for (const auto &type : types) {
      if (auto hasDeviceType = dyn_cast<MGP::MGPHasDeviceTypeInterface>(type))
        deviceIndices.push_back(deviceIndexMap[hasDeviceType.getDevice()]);
    }
    if (!deviceIndices.empty()) {
      ArrayAttr deviceIndicesAttr = ArrayAttr::get(ctx, deviceIndices);
      extraAttrs.emplace_back(b.getStringAttr(label), deviceIndicesAttr);
    }
  };
  populateDeviceIndicesExtraAttr(kMgpModelArgumentDeviceIndices, inputTypes);
  populateDeviceIndicesExtraAttr(kMgpModelResultDeviceIndices, outputTypes);

  b.setInsertionPoint(graph);
  auto modelOp = MGP::ModelOp::create(
      b, graph.getLoc(),
      /*name=*/graph.getNameAttr(),
      /*frameworkName=*/loweringState.getCompiledFrameworkLabel(),
      /*devices=*/devices,
      /*inputTypes=*/inputTypes,
      /*outputTypes=*/outputTypes,
      /*inputArgLocs=*/inputLocs,
      /*isSubgraph=*/graph.getIsSubgraph(),
      /*isDeviceGraph=*/graph.getIsDeviceGraph(),
      /*extraAttrs=*/extraAttrs);

  loweringState.setModelOp(modelOp);

  // The model takes over the graph's symbol name. The cached
  // `SymbolTableAnalysis` is never invalidated by the pass manager, so update
  // it by hand; the graph stays in the IR until the end of `run()`.
  mlir::SymbolTable &moduleSymbolTable = loweringState.getModuleSymbolTable();
  moduleSymbolTable.remove(graph);
  moduleSymbolTable.insert(modelOp);

  loweringState.getInitOpBuilder().setInsertionPointToEnd(
      loweringState.getModelOp().getInitBlock());
  // Tracks the current init output chain.
  loweringState.setInitChain(GRT::ChainCreateOp::create(
      loweringState.getInitOpBuilder(), graph.getLoc()));

  // Create the !mgp.context. Its numSlots attribute will be incremented
  // as needed.
  loweringState.setInitMgpContext(MGP::ContextCreateOp::create(
      loweringState.getInitOpBuilder(), graph.getLoc(), /*numSlots=*/0));

  // Initialize the execute state.
  Block *executeBlock = loweringState.getModelOp().getExecuteBlock();
  loweringState.getExecuteOpBuilder().setInsertionPointToEnd(executeBlock);

  // Extract chain and MGP context from arguments
  loweringState.setRootExecuteChain(executeBlock->getArgument(0));
  loweringState.setCurrentExecuteChain(executeBlock->getArgument(0));
  loweringState.setExecuteMgpContext(executeBlock->getArgument(1));

  for (auto [devRef, devArg] :
       llvm::zip(modelOp.getDevices(), modelOp.getInitDevices())) {
    loweringState.setInitDevice(
        dyn_cast<DeviceRefAttr>(devRef),
        dyn_cast<TypedValue<MGP::DeviceContextType>>(devArg));
  }

  // Establish buffer builders for all the devices. This may require some
  // extra device-specific setup in the 'init' block.
  // CAUTION: Must be done after the 'init' and 'execute' mgp.contexts have
  // been established, but before we lower any device-specific ops.
  if (auto err = setupBufferBuildersForAllDevices(); err.isError())
    return err.takeError();

  return success();
}

MGP::ListType MOGraphLowering::lowerListType(MO::ListType listTy) {
  auto outputTensorTy = dyn_cast<MO::TensorType>(listTy.getElementType());
  // TODO(#29947): Support other element types
  ASSERT_STREAM(outputTensorTy, << "only support lowering list of tensors");

  MLIRContext *ctx = listTy.getContext();
  MGP::TensorSpecType outputTensorSpecTy =
      MGP::createTensorSpecType(outputTensorTy);
  DeviceRefAttr tensorDeviceRef = MO::getDeviceOrDefault(outputTensorTy);
  auto mgpOutputTensorTy = loweringState.getTypeBuilder(tensorDeviceRef)
                               .tensorType(outputTensorSpecTy);
  return MGP::ListType::get(ctx, mgpOutputTensorTy);
}

LogicalResult MOGraphLowering::cloneTrivialOp(Operation *trivialOp) {
  Operation *cloned = loweringState.getExecuteOpBuilder().clone(*trivialOp);
  for (auto [from, to] :
       llvm::zip_equal(trivialOp->getResults(), cloned->getResults()))
    loweringState.emplaceExecuteLoweredValue(from, to);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::OutputOp output) {
  // Init phase.
  SmallVector<Value> outputs;
  outputs.push_back(loweringState.getInitChain());
  outputs.push_back(loweringState.getInitMgpContext());
  outputs.push_back(loweringState.getModelOp().getInitWeightsRegistry());
  for (auto dev : loweringState.getModelOp().getDevices())
    outputs.push_back(
        loweringState.getInitDevice(dyn_cast<DeviceRefAttr>(dev)));
  MEF::OutputOp::create(loweringState.getInitOpBuilder(), output.getLoc(),
                        outputs);

  // Execute phase. Always returns a chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getCurrentExecuteChain());

  for (auto [idx, operand] : llvm::enumerate(output.getOperands())) {
    Value loweredValue = loweringState.lookupExecuteLoweredValue(operand);
    if (!loweredValue)
      return output.emitOpError("operand #")
             << idx << " of type " << operand.getType()
             << " was never lowered into the execute block";

    // The model's results expect mgp.opaque. ToMGPFromOpaqueOp is a type
    // rebind, folded away once the input becomes mgp.opaque (e.g. after
    // OutlineAsyncRoutines).
    if (auto moOpaqueType = dyn_cast<MO::OpaqueType>(loweredValue.getType())) {
      Type mgpOpaqueType =
          MGP::OpaqueType::get(output->getContext(), moOpaqueType.getSymbol(),
                               moOpaqueType.getOptionalParameters());
      loweredValue = MOGG::ToMGPFromOpaqueOp::create(
          loweringState.getExecuteOpBuilder(), operand.getLoc(),
          TypeRange{mgpOpaqueType}, loweredValue);
    }

    operands.push_back(loweredValue);
  }

  MEF::OutputOp::create(loweringState.getExecuteOpBuilder(), output.getLoc(),
                        operands);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::YieldOp yield) {
  // Always implicitly returns the current 'execute' block chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getCurrentExecuteChain());

  for (Value operand : yield.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredValue(operand));

  for (TypedAttr param : yield.getParameters()) {
    auto valueOr = loweringState.lowerExpression(yield->getParentRegion(),
                                                 yield.getLoc(), param);
    if (valueOr.isError())
      return yield->emitOpError(valueOr.getError());
    operands.push_back(*valueOr);
  }

  MEF::OutputOp::create(loweringState.getExecuteOpBuilder(), yield.getLoc(),
                        operands);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::WhileConditionOp condition) {
  // Always implicitly returns the current 'execute' block chain.
  SmallVector<Value> operands;
  operands.push_back(loweringState.getCurrentExecuteChain());

  for (Value operand : condition.getOperands())
    operands.push_back(loweringState.lookupExecuteLoweredValue(operand));

  MEF::OutputOp::create(loweringState.getExecuteOpBuilder(), condition.getLoc(),
                        operands);
  return success();
}

LogicalResult MOGraphLowering::lowerParam(Region *parentRegion,
                                          KGEN::ParamDeclareOp declOp) {
  auto valueOr = loweringState.lowerExpression(parentRegion, declOp->getLoc(),
                                               declOp.getValue());
  if (valueOr.isError())
    return declOp->emitOpError(valueOr.getError());

  Value value = *valueOr;
  if (auto err = loweringState.lowerParamDecl(parentRegion,
                                              declOp.getParamDecl(), value))
    return declOp->emitOpError(err.getError());

  return success();
}

LogicalResult MOGraphLowering::lowerParam(Region *parentRegion,
                                          MOSH::ParamToValueOp paramToValueOp) {
  auto valueOr = loweringState.lowerExpression(
      parentRegion, paramToValueOp.getLoc(), paramToValueOp.getValue());
  if (valueOr.isError())
    return paramToValueOp->emitOpError(valueOr.getError());

  loweringState.emplaceExecuteLoweredValue(paramToValueOp.getResult(),
                                           *valueOr);
  return success();
}

LogicalResult
MOGraphLowering::lowerParam(Region *parentRegion,
                            MOSH::ParamFromValueOp paramFromValueOp) {
  auto input =
      loweringState.lookupExecuteLoweredValue(paramFromValueOp.getValue());

  if (auto err = loweringState.lowerParamDecl(
          parentRegion, paramFromValueOp.getParamDecl(), input))
    return paramFromValueOp->emitOpError(err.getError());

  return success();
}

LogicalResult MOGraphLowering::lowerInterface(MGP::LowerableToMGPInterface op) {
  return op.lower(loweringState);
}

LogicalResult MOGraphLowering::lower(MO::IndexToTensorOp toTensorOp) {
  ASSERT_STREAM(MO::isOnHost(toTensorOp),
                << "mo.index.to_tensor is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp);

  // Set up output buffer.
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputStorageOr))
    return failure();
  auto &outputStorage = (*loweredOutputStorageOr)[0];
  [[maybe_unused]] auto bufType =
      dyn_cast<MGP::DeviceBufferType>(outputStorage.buffer.getType());
  ASSERT_STREAM(bufType && bufType.getDevice().getLabel() == kCPULabel,
                << "expecting !mgp.buffer on host but it is on "
                << bufType.getDevice());

  // Fill the buffer with the index value.
  Value loweredIndexVal =
      loweringState.lookupExecuteLoweredValue(toTensorOp.getInput());

  // The input to mo.index.to_tensor is an scalar<si64>. Cast to index to match
  // the expectation of mgp.set_with_index.
  auto &builder = loweringState.getExecuteOpBuilder();
  loweredIndexVal =
      MGP::dimensionToIndex(builder, toTensorOp.getLoc(), loweredIndexVal);

  auto outChain = opLoweringState.bufferSetWithIndex(
      outputStorage.buffer, {loweredIndexVal},
      toTensorOp.getResult().getType());

  // Set up output tensor.
  opLoweringState.createOutputs(outChain, *loweredOutputStorageOr);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::Shape::ToTensorOp toTensorOp) {
  ASSERT_STREAM(MO::isOnHost(toTensorOp),
                << "mo.shape.to_tensor is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, toTensorOp);

  // Set up output buffer.
  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &outputStorage = (*loweredOutputsStorageOr)[0];

  // Fill the buffer with the runtime dimensions.
  auto loweredShape =
      loweringState.lookupExecuteLoweredValue(toTensorOp.getInput());

  auto loweredTensorSpec = cast<TypedValue<MGP::TensorSpecType>>(loweredShape);
  // Note loweredTensorSpec is the encoding for the shape we are trying to
  // reify as a tensor, whereas outputStorage.tensorSpec just describes
  // the statically known shape of our output tensor.
  SmallVector<Value> values(loweredTensorSpec.getType().getRank());
  for (size_t i = 0; i < values.size(); ++i)
    values[i] =
        opLoweringState.create<MGP::TensorSpecGetDimOp>(loweredTensorSpec, i);

  auto outChain = opLoweringState.bufferSetWithIndex(
      outputStorage.buffer, values, toTensorOp.getResult().getType());

  // Set up output tensor.
  opLoweringState.createOutputs(outChain, *loweredOutputsStorageOr);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::Shape::FromTensorOp fromTensorOp) {
  ASSERT_STREAM(MO::isOnHost(fromTensorOp),
                << "mo.shape.to_tensor is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, fromTensorOp);

  MO::TensorType tensorType = fromTensorOp.getInput().getType();
  std::optional<int64_t> rank = tensorType.getNumElements();

  ASSERT_STREAM(rank, << "unable to determine rank of result shape");

  MGP::BufferizedTensor inputBuffer =
      opLoweringState.getBufferizedInput(fromTensorOp.getInput());

  Value inChain =
      opLoweringState.getGuardChain(loweringState.getExecuteChain());

  auto sizeInBytes = opLoweringState.create<mlir::index::ConstantOp>(
      tensorType.getDtype().getSizeInBytes());

  SmallVector<Value> indices;
  for (auto dim : llvm::seq(*rank)) {
    IndexLowering indexLowering{opLoweringState};

    Value offset = indexLowering.constant(dim);
    Value offsetInBytes = indexLowering.mul(offset, sizeInBytes);

    Value sliced = opLoweringState.bufferSlice(
        inputBuffer.buffer, offsetInBytes, sizeInBytes, tensorType);

    Value indexValue =
        opLoweringState.create<MGP::BufferToIndexOp>(inChain, sliced);

    indices.push_back(indexValue);
  }

  SmallVector<int64_t> shape(*rank, mlir::ShapedType::kDynamic);
  auto tensorSpecType =
      MGP::TensorSpecType::get(shape, tensorType.getElementType());

  auto result =
      opLoweringState.create<MGP::TensorSpecCreateOp>(tensorSpecType, indices);

  loweringState.emplaceExecuteLoweredValue(fromTensorOp.getResult(), result);

  return success();
}

/// Returns the lowered tensor that the rebind's result should map to.
static Value getRebindResTensor(MO::RebindOp rebindOp,
                                MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, rebindOp);

  MO::TensorValue inputTensor = rebindOp.getInput();
  MO::TensorType inputTy = inputTensor.getType();
  MO::TensorType resTy = rebindOp.getResult().getType();

  auto outSpecTy = MGP::createTensorSpecType(resTy);
  auto inSpecTy = MGP::createTensorSpecType(inputTy);
  Value loweredInput = loweringState.lookupExecuteLoweredValue(inputTensor);
  if (outSpecTy == inSpecTy)
    return loweredInput;

  SmallVector<Value> runtimeDims;
  Location loc = opLoweringState.loc;
  for (auto [idx, inDim, resDim] :
       llvm::enumerate(inputTy.getDims(), resTy.getDims())) {

    if (MOSH::ShapeAttr::getStaticDim(resDim))
      continue;

    if (std::optional<int64_t> intDim = MOSH::ShapeAttr::getStaticDim(inDim)) {
      auto constOp = opLoweringState.create<mlir::index::ConstantOp>(*intDim);
      runtimeDims.emplace_back(constOp.getResult());
      continue;
    }

    // Both dims are dynamic.
    auto getDimOp = loweringState.getDim(loc, loweredInput, idx,
                                         getDeviceOrDefault(inputTy));
    runtimeDims.emplace_back(getDimOp);
  }

  Value buffer = opLoweringState.tensorExtract(loweredInput, inputTy).first;
  TypedValue<MGP::TensorSpecType> tensorSpec =
      opLoweringState.create<MGP::TensorSpecCreateOp>(outSpecTy, runtimeDims);
  return opLoweringState.tensorCreate(loweringState.getExecuteChain(), buffer,
                                      tensorSpec, inputTy);
}

LogicalResult MOGraphLowering::lower(MO::RebindOp rebindOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, rebindOp);

  MO::TensorValue inputTensor = rebindOp.getInput();
  MO::TensorValue resTensor = rebindOp.getResult();

  MO::TensorType inputTy = inputTensor.getType();
  MO::TensorType resTy = resTensor.getType();
  if (inputTy.hasRank() != resTy.hasRank())
    return rebindOp.emitOpError(
        "rebind lowering doesn't support unknown shapes");

  // The input chain also guards the users of the reshaped buffer.
  Value inChain =
      opLoweringState.getGuardChain(loweringState.getExecuteChain());

  Value loweredOutput = getRebindResTensor(rebindOp, loweringState);
  loweringState.emplaceExecuteLoweredBufferValue(resTensor, loweredOutput,
                                                 inChain);

  // Establish the lowered form of the rebind's result parameters.
  // TODO: Note this may end up extracting the dynamic dimensions from the
  // result tensor even though the same dimension values were just passed
  // as operands to a tensor_spec.create op.
  ArrayRef<KGEN::ParamDeclAttr> decls = rebindOp.getOutputParamDecls();
  if (!decls.empty()) {
    MOSH::ShapeAttr resShape = resTy.getShape();
    for (KGEN::ParamDeclAttr decl : decls) {
      int64_t idx = findDeclIdx(resShape, decl);
      if (idx == -1) {
        return rebindOp.emitOpError(
            "rebind declares parameter not appearing in its result");
      }

      auto getDimOp = loweringState.getDim(rebindOp.getLoc(), loweredOutput,
                                           idx, getDeviceOrDefault(resTy));

      auto asDimension = MGP::indexToDimension(
          loweringState.getExecuteOpBuilder(), rebindOp.getLoc(), getDimOp);

      if (auto err = loweringState.lowerParamDecl(rebindOp->getParentRegion(),
                                                  decl, asDimension))
        return rebindOp->emitOpError(err.getError());
    }
  }

  return success();
}

/// Takes a constant op's result buffer on host and:
/// - allocates device memory,
/// - copies from host to device, then,
/// - creates the output tensor.
/// If the target device is host, then just creates the output tensor.
static LogicalResult
constHostBufferToTarget(MGP::MOOpLoweringState &opLoweringState,
                        MGP::MOGraphLoweringState &graphLoweringState,
                        Value hostBuffer, TypedValue<MEF::ChainType> chain,
                        TypedValue<MO::TensorType> opResult,
                        std::optional<uint64_t> maybeAlignment = std::nullopt) {
  MO::TensorType moTensorType = opResult.getType();
  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(moTensorType);
  if (failed(tensorSpecOr))
    return failure();
  TypedValue<MGP::TensorSpecType> tensorSpec = *tensorSpecOr;

  DeviceRefAttr device = MO::getDeviceOrDefault(moTensorType);
  DeviceRefAttr hostDevice = graphLoweringState.getHostDevice();

  // Off host, allocate a device buffer and transfer HtoD.
  if (device == hostDevice) {
    opLoweringState.createOutputTensor(opResult, chain, hostBuffer, tensorSpec);
    return success();
  }

  uint64_t alignment = maybeAlignment.value_or(
      M::Driver::getPreferredAlignment(device.getLabel().str()));

  uint64_t sizeInBytes = *tensorSpec.getType().getSizeInBytes();
  Value bufferSize =
      opLoweringState.create<mlir::index::ConstantOp>(sizeInBytes);
  Value deviceBuffer =
      opLoweringState.bufferAlloc(chain, bufferSize, alignment, moTensorType);
  // bufferTransferFromHost also handles the stream sync, which is needlessly
  // conservative since a constant external's lifetime is outside the graph.
  Value outChain = opLoweringState.bufferTransferFromHost(
      chain, cast<TypedValue<MGP::DeviceBufferType>>(hostBuffer), deviceBuffer,
      moTensorType);

  opLoweringState.createOutputTensor(opResult, outChain, deviceBuffer,
                                     tensorSpec);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::ConstantOp constOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constOp);

  ErrorOr<ElementsAttr> elementsOr = lowerElementsAttr(constOp.getValue());
  if (elementsOr)
    return constOp->emitOpError(elementsOr.getError());

  TypedValue<MO::TensorType> opResult = constOp.getResult();
  TypedValue<MEF::ChainType> chain = loweringState.getExecuteChain();

  Value hostBuffer = opLoweringState.hostBufferConstant(chain, *elementsOr,
                                                        opResult.getType());
  return constHostBufferToTarget(opLoweringState, loweringState, hostBuffer,
                                 chain, opResult);
}

LogicalResult MOGraphLowering::lower(MO::ConstantExternalOp constExternalOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constExternalOp);
  OpBuilder &builder = opLoweringState.builder;
  Location loc = constExternalOp.getLoc();

  TypedValue<MEF::ChainType> chain = loweringState.getExecuteChain();
  TypedValue<MO::TensorType> opResult = constExternalOp.getResult();
  uint64_t sizeInBytes = opResult.getType().getDtype().getSizeInBytes() *
                         (*opResult.getType().getNumElements());
  uint64_t alignment = constExternalOp.getAlign();
  bool hasAlias = constExternalOp.getHasAlias();
  bool isPlaceholder = constExternalOp.getIsPlaceholder();

  auto mojoStringType = MGP::OpaqueType::get(
      builder.getContext(), builder.getStringAttr("mojo_string"), {});
  auto name = MGP::StringCreateOp::create(
      builder, loc, mojoStringType,
      builder.getStringAttr(constExternalOp.getName()));

  DeviceRefAttr opDevice = constExternalOp.getExecutionDevice();
  opDevice =
      opDevice ? opDevice : DeviceRefAttr::getDefault(chain.getContext());
  if (opDevice.getLabel() == kCPULabel) {
    // Lowered like mo.constant.
    Value hostBuffer = opLoweringState.hostBufferConstantExternal(
        chain, name, sizeInBytes, alignment, hasAlias, isPlaceholder);

    return constHostBufferToTarget(opLoweringState, loweringState, hostBuffer,
                                   chain, opResult, alignment);
  }

  // A device external constant stays on device, with no copy.
  DeviceRefAttr resultDevice = MO::getDeviceOrDefault(opResult.getType());
  if (resultDevice != opDevice)
    return constExternalOp->emitOpError(
        Twine("expected op device ") + opDevice.getLabel() +
        " and result device " + resultDevice.getLabel() + " to match");

  Value deviceBuffer = opLoweringState.deviceBufferConstantExternal(
      opDevice, chain, name, sizeInBytes, alignment, hasAlias, isPlaceholder);

  MO::TensorType moTensorType = opResult.getType();
  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(moTensorType);
  if (failed(tensorSpecOr))
    return failure();
  TypedValue<MGP::TensorSpecType> tensorSpec = *tensorSpecOr;

  opLoweringState.createOutputTensor(opResult, chain, deviceBuffer, tensorSpec);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::ConstantScalarOp constOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, constOp);

  // currently only si32, si64, and i1 constants are supported
  if (auto boolAttr = dyn_cast<mlir::BoolAttr>(constOp.getValue())) {
    IntegerAttr intAttr = boolAttr;
    auto grtConstOp =
        opLoweringState.create<GRT::ConstantI1Op>(intAttr.getType(), intAttr);
    loweringState.emplaceExecuteLoweredValue(constOp, grtConstOp);

  } else if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
    if (intAttr.getType().isSignedInteger(32)) {
      auto grtConstOp = opLoweringState.create<GRT::ConstantSI32Op>(
          intAttr.getType(), intAttr);
      loweringState.emplaceExecuteLoweredValue(constOp, grtConstOp);
    } else if (intAttr.getType().isSignedInteger(64)) {
      auto mipConstOp =
          opLoweringState.create<mlir::index::ConstantOp>(intAttr.getSInt());
      loweringState.emplaceExecuteLoweredValue(constOp, mipConstOp);
    }
  } else {
    return constOp->emitOpError(
        "Only si32, si64 and i1 constant scalar are currently lowerable");
  }

  return success();
}

static LogicalResult lowerReshapeImpl(MGP::MOOpLoweringState &opLoweringState,
                                      Value input, Value output, Value chain,
                                      MO::TensorType outputType) {
  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(input);
  Value buffer = bufferizedInput.buffer;

  FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
      opLoweringState.createTensorSpec(outputType);
  if (failed(tensorSpecOr))
    return failure();

  // The input chain also guards the users of the reshaped buffer.
  Value inChain = opLoweringState.getGuardChain(chain);

  opLoweringState.createOutputTensor(output, inChain, buffer, *tensorSpecOr);

  return success();
}

static LogicalResult lowerMoggSliceImpl(MGP::MOOpLoweringState &opLoweringState,
                                        MOGG::SliceOp sliceOp, Value chain) {
  MO::TensorType outputType = sliceOp.getType();

  FailureOr<TypedValue<MGP::TensorSpecType>> outputSpecOr =
      opLoweringState.createTensorSpec(outputType);
  if (failed(outputSpecOr))
    return failure();

  Value loweredInput = opLoweringState.loweringState.lookupExecuteLoweredValue(
      sliceOp.getInput());
  Value loweredStart = opLoweringState.loweringState.lookupExecuteLoweredValue(
      sliceOp.getStart());

  // The Mojo primitive computes the sliced dimension, stride, and start offset
  // at runtime.
  auto inputDevice =
      cast<MGP::DeviceTensorType>(loweredInput.getType()).getDevice();
  TypedValue<MGP::DeviceContextType> deviceContext =
      opLoweringState.loweringState.getExecuteDevice(inputDevice);
  Value slicedTensor = opLoweringState.create<MGP::TensorSliceOp>(
      loweredInput, *outputSpecOr, loweredStart, deviceContext);

  Value inChain = opLoweringState.getGuardChain(chain);
  opLoweringState.loweringState.emplaceExecuteLoweredBufferValue(
      sliceOp.getResult(), slicedTensor, inChain);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::DebugPrintOp printOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, printOp);

  OpBuilder b(printOp);

  DeviceRefAttr targetDevice = printOp.getExecutionDevice();
  DeviceRefAttr hostDevice = loweringState.getHostDevice();

  auto chain = opLoweringState.getGuardChain(
      loweringState.lookupExecuteLoweredValue(printOp.getInChain()));

  auto outChain =
      cast<TypedValue<MEF::ChainType>>(opLoweringState.hostDebugPrint(
          chain, printOp.getValueAttr(), printOp.getLabelAttr()));

  chain = opLoweringState.sync(chain);

  if (targetDevice != hostDevice)
    chain = opLoweringState.sync(chain);

  loweringState.setCurrentExecuteChain(
      cast<TypedValue<MEF::ChainType>>(outChain));

  loweringState.emplaceExecuteLoweredValue(printOp.getOutChain(), outChain);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::DebugTensorPrintOp printOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, printOp);

  OpBuilder b(printOp);
  auto moTensorType = printOp.getInput().getType();
  DeviceRefAttr targetDevice = MO::getDeviceOrDefault(moTensorType);
  DeviceRefAttr hostDevice = loweringState.getHostDevice();

  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(printOp.getInput());

  auto &buffer = bufferizedInput.buffer;
  auto &tensorSpec = bufferizedInput.tensorSpec;
  auto chain = opLoweringState.getGuardChain(
      loweringState.lookupExecuteLoweredValue(printOp.getInChain()));

  if (printOp.getLabelAttr().empty() || targetDevice == hostDevice) {
    auto outChain = opLoweringState.debugTensorPrint(
        chain, buffer, tensorSpec, printOp.getLabelAttr(),
        printOp.getInput().getType());
    loweringState.setCurrentExecuteChain(
        cast<TypedValue<MEF::ChainType>>(outChain));

    loweringState.emplaceExecuteLoweredValue(printOp.getOutChain(), outChain);
    return success();
  }

  // Off host, copy DtoH and print on the host, as the device primitive does.
  auto maybeTensorSizeInBytes = tensorSpec.getType().getSizeInBytes();
  Value bufferSize;
  if (maybeTensorSizeInBytes.has_value())
    bufferSize = opLoweringState.create<mlir::index::ConstantOp>(
        *maybeTensorSizeInBytes);
  else
    bufferSize = opLoweringState.bufferSizeBytes(tensorSpec, moTensorType);
  Value hostBuffer = opLoweringState.hostBufferAlloc(
      cast<TypedValue<MEF::ChainType>>(chain), bufferSize,
      M::Driver::getPreferredAlignment(hostDevice.getLabel().str()));
  auto transferChain = opLoweringState.bufferTransferToHost(
      chain, buffer, cast<TypedValue<MGP::DeviceBufferType>>(hostBuffer),
      moTensorType);
  auto outChain = opLoweringState.hostDebugTensorPrint(
      transferChain, hostBuffer, tensorSpec, printOp.getLabelAttr());
  loweringState.setCurrentExecuteChain(
      cast<TypedValue<MEF::ChainType>>(outChain));
  loweringState.emplaceExecuteLoweredValue(printOp.getOutChain(), outChain);
  return success();
}

// Returns null for an unsupported device pair.
static Value lowerTransferImpl(MGP::MOOpLoweringState &opLoweringState,
                               DeviceRefAttr hostDevice,
                               MGP::BufferizedTensor &srcBuffer,
                               Value dstBuffer, MO::TensorType srcType,
                               MO::TensorType dstType, Value inChain) {
  DeviceRefAttr srcDevice = srcType.getDeviceRef();
  DeviceRefAttr dstDevice = dstType.getDeviceRef();

  if (srcDevice == hostDevice) {
    return opLoweringState.bufferTransferFromHost(
        inChain, cast<TypedValue<MGP::DeviceBufferType>>(srcBuffer.buffer),
        dstBuffer, dstType);
  }

  if (dstDevice == hostDevice) {
    return opLoweringState.bufferTransferToHost(
        inChain, srcBuffer.buffer,
        cast<TypedValue<MGP::DeviceBufferType>>(dstBuffer), srcType);
  }

  if (srcDevice.getLabel() == dstDevice.getLabel()) {
    return opLoweringState.bufferTransferDevice(inChain, srcBuffer.buffer,
                                                dstBuffer, srcType, dstType);
  }

  return nullptr;
}

LogicalResult MOGraphLowering::lower(MO::TransferOp transferOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, transferOp);

  auto srcType = transferOp.getInput().getType();
  auto dstType = transferOp.getResult().getType();
  if (!srcType.getDeviceRef() || !dstType.getDeviceRef())
    return transferOp->emitOpError(
        "should have explicit devices before lowering");

  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(transferOp.getInput());

  Value inChain =
      loweringState.lookupExecuteLoweredValue(transferOp.getInChain());
  inChain = opLoweringState.getGuardChain(inChain);

  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(inChain,
                                          {bufferizedInput.tensorSpec});
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &loweredOutputsStorage = *loweredOutputsStorageOr;

  Value outChain = lowerTransferImpl(
      opLoweringState, loweringState.getHostDevice(), bufferizedInput,
      loweredOutputsStorage[0].buffer, srcType, dstType, inChain);

  if (!outChain)
    return transferOp->emitOpError("mo.transfer from device '")
           << srcType.getDeviceRef() << "' to device '"
           << dstType.getDeviceRef() << "' is not supported.";

  opLoweringState.createOutputs(outChain, loweredOutputsStorage);

  loweringState.emplaceExecuteLoweredValue(transferOp.getOutChain(), outChain);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::StableCreateOp stableOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, stableOp);

  MO::TensorValue input = stableOp.getInput();
  Value loweredInput = loweringState.lookupExecuteLoweredValue(input);
  ASSERT_STREAM(loweredInput, << "stable.create input not lowered before use");

  // The MGP operation takes the dynamic dimensions as inputs. Extract them from
  // the lowered shape attribute.
  ErrorOr<SmallVector<Value>> runtimeDimsOr =
      loweringState.lowerShapeAttrDynDims(stableOp->getParentRegion(),
                                          input.getType().getShape(),
                                          stableOp.getLoc());
  if (runtimeDimsOr.isError())
    return stableOp.emitOpError(runtimeDimsOr.getError());

  auto stable = opLoweringState.create<MGP::DeviceGraphStableCreateOp>(
      loweredInput, *runtimeDimsOr);

  loweringState.emplaceExecuteLoweredValue(stableOp.getResult(), stable);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::BufferTransferOp transferOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, transferOp);

  auto srcType = MO::castFromTensorLike(transferOp.getSrc().getType());
  auto dstType = MO::castFromTensorLike(transferOp.getDst().getType());
  if (!srcType.getDeviceRef() || !dstType.getDeviceRef())
    return transferOp->emitOpError(
        "should have explicit devices before lowering");

  MGP::BufferizedTensor bufferizedInput =
      opLoweringState.getBufferizedInput(transferOp.getSrc());
  MGP::BufferizedTensor bufferizedOutput =
      opLoweringState.getBufferizedInput(transferOp.getDst());

  auto inChain =
      loweringState.lookupExecuteLoweredValue(transferOp.getInChain());
  inChain = opLoweringState.getGuardChain(inChain);

  Value outChain = lowerTransferImpl(
      opLoweringState, loweringState.getHostDevice(), bufferizedInput,
      bufferizedOutput.buffer, srcType, dstType, inChain);

  if (!outChain)
    return transferOp->emitOpError("mo.buffer.transfer from device '")
           << srcType.getDeviceRef() << "' to device '"
           << dstType.getDeviceRef() << "' is not supported.";

  loweringState.emplaceExecuteLoweredValue(transferOp.getOutChain(), outChain);
  return success();
}

LogicalResult MOGraphLowering::lower(MO::GuardOp guardOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, guardOp);

  Value loweredChain =
      loweringState.lookupExecuteLoweredValue(guardOp.getChain());

  SmallVector<Value> loweredInputs;

  for (Value val : guardOp.getInputs()) {
    Value loweredInput = loweringState.lookupExecuteLoweredValue(val);
    loweredInputs.push_back(loweredInput);
  }

  auto mgpGuardOp =
      opLoweringState.create<MGP::GuardOp>(loweredChain, loweredInputs);

  for (auto &&[origRes, loweredRes] :
       llvm::zip(guardOp.getResults(), mgpGuardOp.getResults())) {
    loweringState.emplaceExecuteLoweredValue(origRes, loweredRes);
  }

  return success();
}

LogicalResult MOGraphLowering::lower(MO::ChainCreateOp chainCreateOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, chainCreateOp);

  SmallVector<Value> loweredOperands;

  for (Value val : chainCreateOp.getOperands()) {
    Value loweredOperand = loweringState.lookupExecuteLoweredValue(val);
    loweredOperands.push_back(loweredOperand);
  }

  auto grtChainCreateOp =
      opLoweringState.create<GRT::ChainCreateOp>(loweredOperands);

  loweringState.emplaceExecuteLoweredValue(chainCreateOp.getResult(),
                                           grtChainCreateOp.getResult());

  return success();
}

LogicalResult MOGraphLowering::lower(MO::MutableStoreOp mutableStoreOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, mutableStoreOp);

  MO::TensorType srcType = mutableStoreOp.getInTensor().getType();
  MO::TensorType dstType =
      MO::castFromTensorLike(mutableStoreOp.getInBuffer().getType());

  auto srcBufferized =
      opLoweringState.getBufferizedInput(mutableStoreOp.getInTensor());
  auto dstBufferized =
      opLoweringState.getBufferizedInput(mutableStoreOp.getInBuffer());

  auto inChain =
      loweringState.lookupExecuteLoweredValue(mutableStoreOp.getInChain());
  inChain = opLoweringState.getGuardChain(inChain);

  auto loweredChain = opLoweringState.bufferTransferDevice(
      inChain, srcBufferized.buffer, dstBufferized.buffer, srcType, dstType);

  loweringState.emplaceExecuteLoweredValue(mutableStoreOp.getResult(),
                                           loweredChain);

  return success();
}

/// The little-endian bit pattern of a scalar `initValue`, zero-extended to 64
/// bits, for mgp.buffer.memset's `valueBits`. Accepts a single-element
/// elements attribute (as the Python API emits) or a scalar integer/bool/float.
static FailureOr<uint64_t> initValueBits(Attribute initValue) {
  // `M::ArrayElementsAttr` only iterates as `Attribute`.
  if (auto elts = dyn_cast<ElementsAttr>(initValue)) {
    if (elts.getNumElements() != 1)
      return failure();
    initValue = *elts.value_begin<Attribute>();
  }
  // Also covers BoolAttr.
  if (auto intAttr = dyn_cast<IntegerAttr>(initValue))
    return intAttr.getValue().zextOrTrunc(64).getZExtValue();
  if (auto floatAttr = dyn_cast<FloatAttr>(initValue))
    return floatAttr.getValue().bitcastToAPInt().zextOrTrunc(64).getZExtValue();
  return failure();
}

LogicalResult MOGraphLowering::lower(MO::BufferCreateOp bufferCreateOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, bufferCreateOp);

  Type resultType = bufferCreateOp.getResult().getType();
  auto resTensorType = MO::castFromTensorLike(resultType);

  Attribute initValue = bufferCreateOp.getInitValueAttr();
  if (!initValue) {
    // Uninitialized per-call scratch buffer, allocated on every 'execute'.
    Value inChain = loweringState.getCurrentExecuteChain();
    inChain = opLoweringState.getGuardChain(inChain);

    auto tensorSpecOr = opLoweringState.createTensorSpec(resTensorType);
    if (failed(tensorSpecOr))
      return failure();

    auto bufferOr =
        opLoweringState.createBuffer(inChain, *tensorSpecOr, resultType);
    if (failed(bufferOr))
      return failure();

    opLoweringState.createOutputTensor(bufferCreateOp.getResult(), inChain,
                                       bufferOr->buffer, bufferOr->tensorSpec);

    return success();
  }

  // Persistent state buffer: allocated and filled with `initValue` once in
  // 'init', cached in the context, and retrieved on every 'execute' so kernel
  // mutations persist across calls. A single memset needs a static size.
  std::optional<int64_t> numElements = resTensorType.getNumElements();
  if (!numElements)
    return bufferCreateOp->emitOpError(
        "persistent buffer with an 'initValue' must have a statically known "
        "shape (dynamic dimensions are unsupported)");
  uint64_t sizeOfDtype = resTensorType.getDtype().getSizeInBytes();
  uint64_t sizeInBytes = sizeOfDtype * static_cast<uint64_t>(*numElements);

  FailureOr<uint64_t> valueBitsOr = initValueBits(initValue);
  if (failed(valueBitsOr))
    return bufferCreateOp->emitOpError(
        "unsupported 'initValue' attribute kind");
  uint64_t valueBits = *valueBitsOr;

  DeviceRefAttr device = MO::getDeviceOrDefault(resTensorType);
  MGP::MGPBufferBuilder &bufferBuilder = loweringState.getBufferBuilder(device);

  Location loc = bufferCreateOp.getLoc();

  //===--- Init region: alloc -> memset -> cache. ---===//
  OpBuilder &initBuilder = loweringState.getInitOpBuilder();
  TypedValue<MGP::ContextType> initContext =
      loweringState.getInitMgpContext().getResult();
  TypedValue<MGP::DeviceContextType> initDevCtx =
      loweringState.getInitDevice(device);
  auto initChain =
      cast<TypedValue<MEF::ChainType>>(loweringState.getInitChain());

  // Match the alignment logic in MOOpLoweringState::createBuffer.
  uint64_t alignment =
      M::Driver::getPreferredAlignment(device.getLabel().str());
  if (device.getLabel() == kCPULabel)
    alignment = std::gcd(alignment, opLoweringState.minimumCpuAlignment);

  Value sizeVal = mlir::index::ConstantOp::create(
      initBuilder, loc, static_cast<int64_t>(sizeInBytes));

  Value allocBuffer = bufferBuilder.bufferAlloc(
      initBuilder, loc, initContext, /*inChain=*/initChain, /*size=*/sizeVal,
      /*deviceContext=*/initDevCtx, alignment);

  auto memsetChain = MGP::BufferMemsetOp::create(
                         initBuilder, loc, /*inChain=*/initChain,
                         /*buffer=*/allocBuffer, /*deviceContext=*/initDevCtx,
                         /*valueBits=*/valueBits, /*elemSize=*/sizeOfDtype)
                         .getOutChain();

  auto [cacheChain, slot] = bufferBuilder.cache(
      initBuilder, loc, initContext, /*inChain=*/memsetChain,
      /*contextCreateOp=*/loweringState.getInitMgpContext(),
      /*buffer=*/allocBuffer);
  loweringState.setInitChain(cacheChain);

  //===--- Execute region: retrieve the cached buffer, build the tensor.
  //---===//
  OpBuilder &execBuilder = loweringState.getExecuteOpBuilder();
  Value execChain = loweringState.getCurrentExecuteChain();
  execChain = opLoweringState.getGuardChain(execChain);

  Value cachedBuffer = bufferBuilder.getCached(
      execBuilder, loc, /*context=*/loweringState.getExecuteMgpContext(),
      /*inChain=*/cast<TypedValue<MEF::ChainType>>(execChain), slot,
      /*expectedBufferType=*/allocBuffer.getType());

  auto tensorSpecOr = opLoweringState.createTensorSpec(resTensorType);
  if (failed(tensorSpecOr))
    return failure();

  opLoweringState.createOutputTensor(bufferCreateOp.getResult(), execChain,
                                     cachedBuffer, *tensorSpecOr);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::MutableLoadOp mutableLoadOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, mutableLoadOp);

  auto inChain =
      loweringState.lookupExecuteLoweredValue(mutableLoadOp.getInChain());

  auto bufferizedInput =
      opLoweringState.getBufferizedInput(mutableLoadOp.getInBuffer());

  inChain = opLoweringState.getGuardChain(inChain);
  auto bufOr =
      opLoweringState.createBuffer(inChain, bufferizedInput.tensorSpec,
                                   mutableLoadOp.getOutTensor().getType());
  if (failed(bufOr))
    return failure();

  auto &outputStorage = *bufOr;

  auto loweredChain = opLoweringState.bufferTransferDevice(
      inChain, bufferizedInput.buffer, outputStorage.buffer,
      MO::castFromTensorLike(mutableLoadOp.getInBuffer().getType()),
      mutableLoadOp.getOutTensor().getType());

  opLoweringState.createOutputTensor(mutableLoadOp.getOutTensor(), loweredChain,
                                     outputStorage.buffer,
                                     outputStorage.tensorSpec);

  loweringState.emplaceExecuteLoweredValue(mutableLoadOp.getOutChain(),
                                           loweredChain);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::IfOp ifOp) {
  ASSERT_STREAM(MO::isOnHost(ifOp), << "mo.if is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, ifOp);

  // Thread the current 'execute' chain through both branches.
  Value currentChain = loweringState.getCurrentExecuteChain();
  currentChain = opLoweringState.getGuardChain(currentChain);

  Value cond = ifOp.getCond();

  bool isScalarCond = isa<MO::ScalarType>(cond.getType());

  Value condBool;
  if (isScalarCond) {
    condBool = loweringState.lookupExecuteLoweredValue(cond);
  } else {
    MGP::BufferizedTensor bufferizedCond =
        opLoweringState.getBufferizedInput(cond);
    condBool = opLoweringState.bufferToBool(currentChain, bufferizedCond.buffer,
                                            cond.getType());
  }

  SmallVector<Value> operands;
  operands.push_back(condBool);

  SmallVector<Type> resultTypes;
  // Always return the current 'execute' chain
  resultTypes.emplace_back(currentChain.getType());

  for (Type type : ifOp.getResultTypes())
    resultTypes.emplace_back(
        loweringState.getTypeConverter().convertType(type));

  for (ParamDeclAttr decl : ifOp.getOutputParamDecls()) {
    Type declType = decl.getType();
    Type converted = loweringState.getTypeConverter().convertType(declType);
    resultTypes.emplace_back(converted ? converted : declType);
  }

  auto outputIf = opLoweringState.create<GRT::IfOp>(resultTypes, operands);

  size_t paramStartIdx = 1 + ifOp.getNumResults(); // After chain + tensors
  for (auto [decl, result] :
       llvm::zip(ifOp.getOutputParamDecls(),
                 outputIf.getResults().drop_front(paramStartIdx))) {
    if (auto err =
            loweringState.lowerParamDecl(ifOp->getParentRegion(), decl, result))
      return ifOp->emitOpError(err.getError());
  }

  auto enterLowerBlock = [&](Region &inputRegion,
                             Region &outputRegion) -> LogicalResult {
    // Enter the lowered region's unique block.
    Block &newBlock = outputRegion.emplaceBlock();
    opLoweringState.builder.setInsertionPointToEnd(&newBlock);

    if (failed(lowerBlock(&inputRegion.getBlocks().front())))
      return failure();
    return success();
  };

  if (failed(enterLowerBlock(ifOp.getThenRegion(), outputIf.getThenRegion())))
    return failure();

  loweringState.setCurrentExecuteChain(currentChain);

  if (failed(enterLowerBlock(ifOp.getElseRegion(), outputIf.getElseRegion())))
    return failure();

  Value outChain = outputIf.getResult(0);

  loweringState.setCurrentExecuteChain(outChain);

  for (auto [origResult, loweredResult] :
       llvm::zip(ifOp.getResults(), llvm::drop_begin(outputIf.getResults()))) {

    if (isa<MGP::DeviceTensorType>(loweredResult.getType())) {
      loweringState.emplaceExecuteLoweredBufferValue(origResult, loweredResult,
                                                     outChain);
    } else {
      loweringState.emplaceExecuteLoweredValue(origResult, loweredResult);
    }
  }

  opLoweringState.builder.setInsertionPointAfter(outputIf);

  return success();
}

LogicalResult MOGraphLowering::lower(MO::WhileOp whileOp) {
  ASSERT_STREAM(MO::isOnHost(whileOp), << "mo.while is for host only");
  MGP::MOOpLoweringState opLoweringState(loweringState, whileOp);

  // Thread the current 'execute' chain through 'cond' and 'body'.
  Value rootChain = loweringState.getExecuteChain();
  Value currentChain = loweringState.getCurrentExecuteChain();
  bool currentIsRoot = currentChain == rootChain;
  currentChain = opLoweringState.getGuardChain(currentChain);

  // Collect the lowered form of all the outer SSA values which (may) appear
  // in the final grt.while 'cond' and 'body' blocks.
  SmallVector<Value> loweredFreeValues;
  loweredFreeValues.reserve(1 + whileOp.getInputs().size());
  loweredFreeValues.emplace_back(currentChain);
  for (Value value : whileOp.getInputs())
    loweredFreeValues.emplace_back(
        loweringState.lookupExecuteLoweredValue(value));

  SmallVector<Value> operands;
  operands.reserve(loweredFreeValues.size());
  SmallVector<Type> resultTypes;
  resultTypes.reserve(loweredFreeValues.size());
  for (Value loweredFreeValue : loweredFreeValues) {
    operands.emplace_back(loweredFreeValue);
    resultTypes.emplace_back(loweredFreeValue.getType());
  }
  auto loweredWhileOp =
      opLoweringState.create<GRT::WhileOp>(resultTypes, operands);

  auto enterLowerBlock = [&](Region &inputRegion,
                             Region &outputRegion) -> LogicalResult {
    // Enter the lowered region's unique block.
    Block &newBlock = outputRegion.emplaceBlock();
    opLoweringState.builder.setInsertionPointToEnd(&newBlock);

    // The current 'execute' chain comes first.
    auto loweredChainArg =
        outputRegion.addArgument(currentChain.getType(), whileOp.getLoc());
    loweringState.setCurrentExecuteChain(loweredChainArg);
    if (currentIsRoot)
      loweringState.setRootExecuteChain(loweredChainArg);
    // Otherwise fixupIsolatedFromAbove adds a root-chain argument if needed.

    // Then one arg per original region arg, typed like loweredFreeValues.
    for (auto [loweredFreeValue, arg] : llvm::zip(
             llvm::drop_begin(loweredFreeValues), inputRegion.getArguments())) {
      Value loweredArg =
          outputRegion.addArgument(loweredFreeValue.getType(), arg.getLoc());
      if (isa<MGP::DeviceTensorType>(loweredArg.getType())) {
        loweringState.emplaceExecuteLoweredBufferValue(arg, loweredArg,
                                                       loweredChainArg);
      } else {
        loweringState.emplaceExecuteLoweredValue(arg, loweredArg);
      }
    }

    if (failed(lowerBlock(&inputRegion.getBlocks().front())))
      return failure();

    return success();
  };

  // The lowered 'cond' mef.output is fixed up below: grt.while needs every
  // signature value plus an i1 condition, not a mgp.tensor<bool>.
  if (failed(enterLowerBlock(whileOp.getCondRegion(),
                             loweredWhileOp.getCondRegion())))
    return failure();

  if (failed(enterLowerBlock(whileOp.getBodyRegion(),
                             loweredWhileOp.getBodyRegion())))
    return failure();

  {
    MO::WhileConditionOp origYieldOp = whileOp.getCondTerminator();

    Value origTensor = origYieldOp.getCondition();
    Type condMoTensorType = origTensor.getType();

    MEF::OutputOp loweredCondOutputOp = loweredWhileOp.getCondTerminator();

    // At this point, the lowered OutputOp operands are the form:
    // 1. the execute chain
    // 2. the lowered condition tensor
    // 3. the lowered arguments of the mo.yield
    Value loweredOutputChain = loweredCondOutputOp->getOperand(0);
    Value loweredCondTensor = loweredCondOutputOp->getOperand(1);
    ValueRange loweredArgs = loweredCondOutputOp->getOperands().drop_front(2);

    SmallVector<Value> condOutputOpOperands;
    condOutputOpOperands.reserve(loweredWhileOp->getNumResults() + 1);

    condOutputOpOperands.push_back(loweredOutputChain);
    llvm::append_range(condOutputOpOperands, loweredArgs);

    // i1 condition
    opLoweringState.builder.setInsertionPoint(loweredCondOutputOp);
    Value condBuffer =
        opLoweringState.tensorExtract(loweredCondTensor, condMoTensorType)
            .first;

    Value chain = loweredOutputChain;
    if (Value guardChain =
            loweringState.getChainGuardingValueOrNull(origTensor)) {
      chain =
          GRT::ChainCreateOp::create(opLoweringState.builder, whileOp.getLoc(),
                                     ValueRange{guardChain, chain})
              .getResult();
    }
    Value condBool =
        opLoweringState.bufferToBool(chain, condBuffer, condMoTensorType);
    condOutputOpOperands.push_back(condBool);

    loweredCondOutputOp.getOperandsMutable().clear();
    loweredCondOutputOp.getOperandsMutable().append(condOutputOpOperands);
  }

  assert(loweredWhileOp.getNumResults() == 1 + whileOp.getNumResults());

  Value outChain = loweredWhileOp.getResult(0);

  loweringState.setCurrentExecuteChain(outChain);
  loweringState.setRootExecuteChain(rootChain);

  for (auto [origResult, loweredResult] :
       llvm::zip(whileOp.getResults(),
                 llvm::drop_begin(loweredWhileOp.getResults()))) {
    if (isa<MGP::DeviceTensorType>(loweredResult.getType())) {
      loweringState.emplaceExecuteLoweredBufferValue(origResult, loweredResult,
                                                     outChain);
    } else {
      loweringState.emplaceExecuteLoweredValue(origResult, loweredResult);
    }
  }

  opLoweringState.builder.setInsertionPointAfter(loweredWhileOp);

  return success();
}

// level-wise topological sort
static LogicalResult levelwiseTopologicalSort(Block *block) {
  llvm::iterator_range<Block::iterator> ops = *block;
  if (block->back().hasTrait<OpTrait::IsTerminator>())
    ops = block->without_terminator();

  if (ops.empty())
    return success();

  llvm::SmallVector<Operation *> allOps;
  llvm::DenseMap<Operation *, unsigned> inDegree;
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> dependents;

  for (Operation &op : ops) {
    allOps.push_back(&op);
    inDegree[&op] = 0;
  }

  for (Operation *op : allOps) {
    llvm::SetVector<Operation *> dependencies;

    // Walk through operands of all nested operations to find dependencies
    op->walk</*Order=*/mlir::WalkOrder::PreOrder>([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        Operation *parent = operand.getDefiningOp();
        if (!parent)
          continue; // Block argument

        while (parent->getParentOp() &&
               parent->getParentOp() != block->getParentOp()) {
          parent = parent->getParentOp();
        }

        if (parent->getBlock() == block && parent != op) {
          dependencies.insert(parent);
        }
      }

      if (nestedOp->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
        return WalkResult::skip();
      return WalkResult::advance();
    });

    for (Operation *dep : dependencies) {
      dependents[dep].push_back(op);
      inDegree[op]++;
    }
  }

  Block::iterator nextScheduledOp = block->begin();
  [[maybe_unused]] unsigned sortedOps = 0;

  llvm::SmallVector<Operation *> readyQueue;
  for (Operation *op : allOps) {
    if (inDegree[op] == 0) {
      readyQueue.push_back(op);
    }
  }

  while (!readyQueue.empty()) {
    llvm::SmallVector<Operation *> currentLevel = std::move(readyQueue);
    readyQueue.clear();

    for (Operation *op : currentLevel) {
      ++sortedOps;
      op->moveBefore(block, nextScheduledOp);

      if (op == &*nextScheduledOp) {
        ++nextScheduledOp;
      }

      for (Operation *dependent : dependents[op]) {
        inDegree[dependent]--;
        if (inDegree[dependent] == 0) {
          readyQueue.push_back(dependent);
        }
      }
    }
  }

  ASSERT_STREAM(sortedOps == allOps.size(),
                << "Failed to sort the MO operations topologically.");

  return success();
}

LogicalResult MOGraphLowering::lowerOp(Operation &op, Region *parentRegion) {
  if (llvm::isa_and_nonnull<mlir::index::IndexDialect>(op.getDialect()))
    return cloneTrivialOp(&op);

  // clang-format off
  return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      // Param ops need the enclosing region.
      .Case<KGEN::ParamDeclareOp, MOSH::ParamToValueOp,
            MOSH::ParamFromValueOp>(
          [this, parentRegion](auto op) { return lowerParam(parentRegion, op); })
      // Reshape/slice ops need a local MOOpLoweringState.
      .Case<MO::Static::ReshapeOp>([this](auto op) {
        MGP::MOOpLoweringState opLoweringState(loweringState, op);
        return lowerReshapeImpl(opLoweringState, op.getInput(), op.getResult(),
                                loweringState.getExecuteChain(), op.getType());
      })
      .Case<MOGG::SliceOp>([this](auto op) {
        MGP::MOOpLoweringState opLoweringState(loweringState, op);
        return lowerMoggSliceImpl(opLoweringState, op,
                                  loweringState.getExecuteChain());
      })
      // All other concrete op types dispatch to lower() overloads.
      .Case<MO::IndexToTensorOp, MO::Shape::ToTensorOp,
            MO::Shape::FromTensorOp, MO::RebindOp, MO::ConstantOp,
            MO::ConstantExternalOp, MO::ConstantScalarOp, MO::DebugPrintOp,
            MO::DebugTensorPrintOp, MO::TransferOp, MO::StableCreateOp,
            MO::OutputOp, MO::IfOp, MO::WhileOp, MO::TensorBundleOp,
            MO::TensorUnbundleOp, MO::ParallelOp, MO::SequenceOp, MO::YieldOp,
            MO::WhileConditionOp, MO::GuardOp, MO::ChainCreateOp,
            MO::MutableStoreOp, MO::MutableLoadOp, MO::BufferCreateOp,
            MO::BufferTransferOp, MO::CallOp>(
          [this](auto op) { return lower(op); })
      // Externally-defined ops, e.g. `mogg.routine`.
      .Case<MGP::LowerableToMGPInterface>(
          [this](auto op) { return lowerInterface(op); })
      .Default([](Operation *op) -> LogicalResult {
        return op->emitOpError("currently not supported");
      });
  // clang-format on
}

LogicalResult MOGraphLowering::lowerBlock(Block *block) {
  if (failed(levelwiseTopologicalSort(block)))
    return failure();

  for (Operation &op : *block) {
    if (failed(lowerOp(op, block->getParent())))
      return failure();
  }
  return success();
}

LogicalResult MOGraphLowering::lower(MO::SequenceOp seq) {
  // Lower mo.sequence by inlining its body into the execute block, bound to a
  // per-device side-stream view of the device context. The region was a fusion
  // boundary (enforced at the MOGG stage) and a marker for which ops get the
  // side stream. Cross-stream ordering is expressed purely through buffer/
  // tensor guard chains -- never the running execute chain: the fork gates on
  // the guard chains of the body's inputs (chain-guarded captures are promoted
  // to inputs first), and the body's outputs are re-guarded on the join chain
  // so downstream consumers wait for the side stream.
  OpBuilder &b = loweringState.getExecuteOpBuilder();
  Location loc = seq.getLoc();
  Block &body = seq.getBodyRegion().front();
  Value execContext = loweringState.getExecuteMgpContext();
  Value rootChain = loweringState.getExecuteChain();

  // Combine guard chains (dropping nulls); the root chain stands in for "ready
  // immediately" when there are no producers (e.g. model inputs).
  auto combineChains = [&](ArrayRef<Value> chains) -> Value {
    SmallVector<Value> nonNull;
    for (Value c : chains)
      if (c)
        nonNull.push_back(c);
    if (nonNull.empty())
      return rootChain;
    if (nonNull.size() == 1)
      return nonNull.front();
    return GRT::ChainCreateOp::create(b, loc, nonNull).getResult();
  };
  // Device of a value, or null for non-tensor (e.g. chain) values: those carry
  // no device and must not fabricate a default one.
  auto deviceOf = [](Value v) -> DeviceRefAttr {
    if (auto devTy = dyn_cast<MO::MOHasDeviceTypeInterface>(v.getType()))
      return MO::getDeviceOrDefault(devTy);
    return {};
  };
  // A value may be guarded under its MO key or its lowered key (see
  // lower(MO::TensorUnbundleOp)).
  auto guardOf = [&](Value moVal) -> Value {
    if (Value g = loweringState.getChainGuardingValueOrNull(moVal))
      return g;
    if (Value lowered = loweringState.lookupExecuteLoweredValue(moVal))
      return loweringState.getChainGuardingValueOrNull(lowered);
    return {};
  };

  // Collect the unique accelerator devices the body touches. A replicated body
  // (e.g. the EP shared expert) spans every device, one side-stream view each.
  // streamId 0 (default stream) or a host-only body adds no binding.
  uint64_t streamId = seq.getStreamId();
  SmallVector<DeviceRefAttr> devices;
  auto addDevice = [&](Value v) {
    DeviceRefAttr dev = deviceOf(v);
    if (dev && !MO::isOnHost(v.getType()) && !llvm::is_contained(devices, dev))
      devices.push_back(dev);
  };
  for (Value in : seq.getInputs())
    addDevice(in);
  for (Value res : seq.getResults())
    addDevice(res);
  const bool bindStream = streamId != 0 && !devices.empty();

  // mo.sequence is not IsolatedFromAbove so the body can read constants and
  // weights directly. A capture with a producer chain is promoted to a block
  // argument so it is fork-guarded like a listed input.
  SmallVector<Value> inputs(seq.getInputs());
  llvm::SetVector<Value> captured;
  mlir::getUsedValuesDefinedAbove(seq.getBodyRegion(), captured);
  for (Value cap : captured) {
    if (!guardOf(cap))
      continue;
    BlockArgument arg = body.addArgument(cap.getType(), cap.getLoc());
    cap.replaceUsesWithIf(arg, [&](OpOperand &use) {
      return seq.getBodyRegion().isAncestor(use.getOwner()->getParentRegion());
    });
    inputs.push_back(cap);
  }

  // Fork: per device, create the side-stream view and a device_wait that makes
  // the side stream wait for the base-stream producers of that device's inputs.
  llvm::SmallDenseMap<DeviceRefAttr, TypedValue<MGP::DeviceContextType>>
      viewFor;
  llvm::SmallDenseMap<DeviceRefAttr, Value> forkChainFor;
  if (bindStream) {
    for (DeviceRefAttr dev : devices) {
      TypedValue<MGP::DeviceContextType> base =
          loweringState.getExecuteDevice(dev);
      auto view =
          MGP::DeviceContextSelectStreamOp::create(b, loc, base, streamId);
      viewFor[dev] = view;
      SmallVector<Value> inGuards;
      for (Value in : inputs)
        if (deviceOf(in) == dev)
          inGuards.push_back(guardOf(in));
      forkChainFor[dev] =
          MGP::DeviceWaitOp::create(b, loc, /*inChain=*/combineChains(inGuards),
                                    execContext, /*waitingDeviceContext=*/view,
                                    /*signalingDeviceContext=*/base,
                                    /*bDevice=*/dev)
              .getResult();
      loweringState.setExecuteDeviceOverride(dev, view);
    }
  }

  // Guard each body block argument (not the shared input, so unrelated
  // main-stream consumers are not serialized) on its device's fork chain. A
  // device with no fork keeps producer ordering.
  for (auto [arg, operand] : llvm::zip_equal(body.getArguments(), inputs)) {
    loweringState.emplaceExecuteLoweredValue(
        arg, loweringState.lookupExecuteLoweredValue(operand));
    Value guard = bindStream ? forkChainFor.lookup(deviceOf(operand)) : Value();
    if (!guard)
      guard = guardOf(operand);
    if (guard)
      loweringState.replaceChainGuard(arg, guard);
  }

  // Lower the body ops (excluding the yield terminator) in place; with the
  // overrides installed, getExecuteDevice routes them onto the side streams.
  if (failed(levelwiseTopologicalSort(&body)))
    return failure();
  for (Operation &op : body.without_terminator())
    if (failed(lowerOp(op, &seq.getBodyRegion())))
      return failure();

  // Join: per device, a device_wait that makes the base stream wait for the
  // side stream, gated on the guard chains of that device's body outputs.
  // deferSideStreamJoins later moves it to the first consumer of the results.
  auto yieldOp = cast<MO::YieldOp>(body.getTerminator());
  llvm::SmallDenseMap<DeviceRefAttr, Value> joinChainFor;
  if (bindStream) {
    for (DeviceRefAttr dev : devices)
      loweringState.clearExecuteDeviceOverride(dev);
    for (DeviceRefAttr dev : devices) {
      TypedValue<MGP::DeviceContextType> base =
          loweringState.getExecuteDevice(dev);
      SmallVector<Value> outGuards;
      for (Value out : yieldOp.getOperands())
        if (deviceOf(out) == dev)
          outGuards.push_back(guardOf(out));
      auto join = MGP::DeviceWaitOp::create(
          b, loc, /*inChain=*/combineChains(outGuards), execContext,
          /*waitingDeviceContext=*/base,
          /*signalingDeviceContext=*/viewFor[dev], /*bDevice=*/dev);
      join.setIsSideStreamJoin(true);
      joinChainFor[dev] = join.getResult();

      // A freed buffer goes back to its owner stream's free list, which hands
      // it out again right away without knowing the side stream still reads
      // it. Hold this device's inputs until the join has queued the base
      // stream's wait on the side stream.
      SmallVector<Value> held;
      for (Value in : inputs)
        if (deviceOf(in) == dev)
          if (Value lowered = loweringState.lookupExecuteLoweredValue(in)) {
            if (auto view = lowered.getDefiningOp<MGP::TensorCreateOp>())
              lowered = view.getBuffer();
            held.push_back(lowered);
          }
      if (!held.empty())
        MGP::ExtendLifetimeOp::create(b, loc, joinChainFor[dev], held);
    }
  }

  // Map each result to its lowered yield value and guard the result on the
  // device's join chain, so downstream consumers wait for the side stream.
  for (auto [res, yielded] :
       llvm::zip_equal(seq.getResults(), yieldOp.getOperands())) {
    loweringState.emplaceExecuteLoweredValue(
        res, loweringState.lookupExecuteLoweredValue(yielded));
    Value guard =
        bindStream ? joinChainFor.lookup(deviceOf(res)) : guardOf(yielded);
    if (guard)
      loweringState.replaceChainGuard(res, guard);
  }

  return success();
}

LogicalResult MOGraphLowering::run() {
  MO::GraphOp &graph = loweringState.getGraph();

  if (failed(createModelOp()))
    return failure();

  for (auto [graphArg, exeArg] :
       llvm::zip(graph.getBody().front().getArguments(),
                 loweringState.getModelOp().getExecuteInputs())) {
    auto loweringResult =
        loweringState.lowerArg(&graph.getRegion(), graphArg, exeArg);
    if (loweringResult.isError())
      return graph->emitOpError(loweringResult.getError());

    if (!loweringResult->hasParameters && graphArg.use_empty())
      continue;

    loweringState.emplaceExecuteLoweredValue(graphArg,
                                             loweringResult->loweredArg);
  }

  if (failed(lowerBlock(graph.getBodyBlock())))
    return failure();

  OpBuilder b(graph);
  MGP::ModelOp model = loweringState.getModelOp();

  // Binds free variables in grt.if/grt.while and adds a new model to the
  // module, replacing this one.
  (void)MGP::fixupIsolatedFromAbove(b, model);
  model.erase();
  graph.erase();

  return success();
}

/// Fuse a set of synchronous compiled regions into one, then convert the
/// synchronous call stubs to async dispatch with a join barrier.
///
/// After this function:
///   - The fused region contains CallStubAsyncOps, an AsyncJoinOp, and a
///     single AsyncReadyOp producing the output chain.
///   - The original regions in `groupRegions` have been erased.
///   - The fused region is inserted after `groupRegions.back()`.
static MOGG::AsyncRoutineOp
fuseAndConvertToAsync(mlir::MLIRContext *ctx, Location loc,
                      ArrayRef<Operation *> groupRegions) {
  MOGG::AsyncRoutineOp fusedRegion = MOGG::fuseAndReplaceCompiledRegions(
      /*allowDuplication=*/false, groupRegions);

  // fuseAndReplaceCompiledRegions leaves InOutSignatureAttr empty because it
  // normally runs after ABI lowering. We call it earlier, so downstream passes
  // still need a valid signature.
  SmallVector<InOutSignatureAttr::InOutSemantics> sig(
      fusedRegion->getNumOperands(), InOutSignatureAttr::kNone);
  fusedRegion.setInOutSignatureAttr(InOutSignatureAttr::get(ctx, sig));

  // Forward the affinity device ordinal (set only on GPU launches) so the
  // emitter can choose an affinity-aware create_task.
  Block &body = fusedRegion.getBody().front();
  SmallVector<Value> taskHandles;
  body.walk([&](MOGG::CallOp callStub) {
    OpBuilder b(ctx);
    b.setInsertionPoint(callStub);
    MOGG::CallAsyncOp asyncCall = MOGG::CallAsyncOp::create(
        b, callStub.getLoc(), callStub.getName(), callStub.getInputs(),
        callStub.getAffinityDeviceIdAttr());
    taskHandles.push_back(asyncCall.getTaskHandle());
    callStub.erase();
  });

  Operation *outputOp = body.getTerminator();
  OpBuilder b(ctx);
  b.setInsertionPoint(outputOp);
  MOGG::AsyncJoinOp::create(b, loc, taskHandles);

  // Fusion cloned one AsyncReadyOp per original region. The first one (which
  // fuseAndReplaceCompiledRegions wired into the OutputOp) is the canonical
  // chain; erase the rest and move the survivor after the join.
  MOGG::AsyncReadyOp canonical;
  body.walk([&](MOGG::AsyncReadyOp readyOp) {
    if (!canonical) {
      canonical = readyOp;
      return;
    }
    readyOp.getResult().replaceAllUsesWith(canonical.getResult());
    readyOp.erase();
  });
  ASSERT_STREAM(canonical, << "expected at least one AsyncReadyOp in fused "
                              "parallel region body");
  canonical->moveBefore(outputOp);

  for (Operation *region : groupRegions)
    region->erase();

  return fusedRegion;
}

LogicalResult MOGraphLowering::lower(MO::ParallelOp parallelOp) {
  mlir::MLIRContext *ctx = parallelOp->getContext();
  Location loc = parallelOp.getLoc();

  Block &bodyBlock = parallelOp.getBodyRegion().front();
  size_t numLaunches = parallelOp.getNumLaunches();
  size_t numBundleGroups = parallelOp.getInputs().size();

  // The parallel op threads its chain explicitly: the in-chain enters the body
  // through a trailing block argument, the body's buffer ops thread it, and the
  // result is yielded as the terminator's last operand and becomes the op's
  // out-chain. The lowering is a pass-through: it maps the chain block arg to
  // the lowered in-chain and the out-chain to the lowered yielded chain.

  // Per-launch MO-level values of each bundle input, from
  // lower(TensorBundleOp).
  size_t numGroups = numBundleGroups;
  SmallVector<SmallVector<Value>> bundleElements(numGroups);

  for (size_t group = 0; group < numBundleGroups; ++group) {
    Value bundleVal = parallelOp.getInputs()[group];
    auto moElements = loweringState.lookupBundleElements(bundleVal);
    for (size_t launch = 0; launch < numLaunches; ++launch)
      bundleElements[group].push_back(moElements[launch]);
  }

  // Only kernels and bundled.expand should remain by now; the verifier's other
  // transient ops (see ParallelOp::verify) mean a missing cleanup.
  SmallVector<Operation *> kernelOps;
  SmallVector<MO::BundledExpandOp> expandOps;
  for (Operation &op : bodyBlock.without_terminator()) {
    if (isa<MOGG::RoutineOp>(&op)) {
      kernelOps.push_back(&op);
    } else if (auto expand = dyn_cast<MO::BundledExpandOp>(&op)) {
      expandOps.push_back(expand);
    } else {
      return op.emitOpError()
             << "lowering of this op inside mo.parallel is not yet "
                "supported; see GEX-3701";
    }
  }
  if (kernelOps.empty())
    return parallelOp.emitOpError("body must contain at least one kernel");

  // Each kernel is cloned per launch, lowered via
  // `RoutineOp::lowerToAsync`, then fused and made async. Fusion is per
  // kernel group, since a later group's buffer allocs may depend on an earlier
  // group's chain. Bundled kernels (collectives) are N-expanded so each clone
  // sees every launch's inputs.
  //
  // perLaunchCloneResults maps each body result to its per-launch clone
  // results for later clones' IRMapping; clones stay alive until all kernels
  // are processed.
  DenseMap<Value, SmallVector<Value>> perLaunchCloneResults;
  SmallVector<Operation *> allClones;

  // A chained parallel's in-chain is the body's trailing block argument.
  Value loweredInChain;
  if (parallelOp.hasChain()) {
    loweredInChain =
        loweringState.lookupExecuteLoweredValue(parallelOp.getInChain());
    ASSERT_STREAM(loweredInChain,
                  << "parallel in-chain not lowered before parallel op");
  }
  // Each kernel group's fused completion chain, keyed by the group's MO-level
  // result values, so the yielded out-chain can be resolved to it.
  DenseMap<Value, Value> resultToFusedChain;

  // Each mo.bundled.expand result maps to the per-launch elements of its
  // source bundle.
  for (MO::BundledExpandOp expandOp : expandOps) {
    Value input = expandOp.getInput();
    for (size_t group = 0; group < numBundleGroups; ++group) {
      if (input != bodyBlock.getArgument(group))
        continue;
      for (size_t i = 0; i < expandOp.getNumResults(); ++i) {
        SmallVector<Value> &cloneResults =
            perLaunchCloneResults[expandOp.getResult(i)];
        cloneResults.resize(numLaunches);
        // Expand result i always resolves to launch i's value, regardless
        // of which per-launch clone is being built.
        for (size_t launch = 0; launch < numLaunches; ++launch)
          cloneResults[launch] = bundleElements[group][i];
      }
      break;
    }
  }

  for (Operation *kernelOp : kernelOps) {
    SmallVector<Operation *> kernelClones;
    SmallVector<Operation *> groupRegions;

    for (size_t launch = 0; launch < numLaunches; ++launch) {
      IRMapping mapper;
      for (size_t group = 0; group < numGroups; ++group)
        mapper.map(bodyBlock.getArgument(group), bundleElements[group][launch]);
      // Mapped to the MO-level in-chain; the compiled-region lowering
      // resolves its lowered value.
      if (parallelOp.hasChain())
        mapper.map(bodyBlock.getArgument(numBundleGroups),
                   parallelOp.getInChain());
      for (auto &[origResult, cloneResults] : perLaunchCloneResults)
        mapper.map(origResult, cloneResults[launch]);

      OpBuilder cloneBuilder(ctx);
      cloneBuilder.setInsertionPointAfter(parallelOp);
      Operation *clonedOp = cloneBuilder.clone(*kernelOp, mapper);

      // The body is typed for the first input's device; retarget the clone's
      // result types and device to this launch's device.
      DeviceRefAttr launchDevice = parallelOp.getLaunchDeviceList()[launch];
      const bool retarget = launchDevice != MO::getDeviceOrDefaultAny(clonedOp);
      for (OpResult result : clonedOp->getResults()) {
        if (auto resTy = dyn_cast<MO::TensorType>(result.getType()))
          if (resTy.getDeviceRef() != launchDevice)
            result.setType(resTy.cloneWith(launchDevice));
      }

      auto clonedKernel = cast<MOGG::RoutineOp>(clonedOp);
      if (retarget)
        clonedKernel.setDeviceRef(launchDevice);
      MOGG::AsyncRoutineOp launchRegion =
          clonedKernel.lowerToAsync(loweringState);
      if (!launchRegion)
        return failure();

      // Stamp the GPU device ordinal on each CallOp for
      // fuseAndConvertToAsync. A plain I64Attr, since kernel-layer device refs
      // are label-only.
      if (launchDevice && launchDevice.getLabel() == M::kGPULabel) {
        auto affinityId =
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                   static_cast<int64_t>(launchDevice.getId()));
        launchRegion->walk([&](MOGG::CallOp callStub) {
          callStub.setAffinityDeviceIdAttr(affinityId);
        });
      }

      groupRegions.push_back(launchRegion);
      kernelClones.push_back(clonedOp);
    }

    for (auto [idx, origResult] : llvm::enumerate(kernelOp->getResults())) {
      SmallVector<Value> &cloneResults = perLaunchCloneResults[origResult];
      for (size_t launch = 0; launch < numLaunches; ++launch)
        cloneResults.push_back(kernelClones[launch]->getResult(idx));
    }
    allClones.append(kernelClones);

    MOGG::AsyncRoutineOp fusedRegion =
        fuseAndConvertToAsync(ctx, loc, groupRegions);
    Value fusedChain = fusedRegion.getResult(0);
    for (Value res : kernelOp->getResults())
      resultToFusedChain[res] = fusedChain;

    // Repoint guards from the erased per-launch region chains to the fused
    // chain.
    for (Operation *clone : kernelClones) {
      for (Value cloneRes : clone->getResults()) {
        loweringState.replaceChainGuard(cloneRes, fusedChain);
        Value lowered = loweringState.lookupExecuteLoweredValue(cloneRes);
        if (lowered)
          loweringState.replaceChainGuard(lowered, fusedChain);
      }
    }
  }

  // Register clone results as each result's bundle elements for
  // lower(TensorUnbundleOp).
  auto yieldOp = cast<MO::YieldOp>(bodyBlock.getTerminator());
  size_t resultIdx = 0;
  for (Value yieldOperand : yieldOp.getOperands()) {
    if (!isa<MO::TensorType>(yieldOperand.getType()))
      continue;
    Value bundleResult = parallelOp.getResult(resultIdx++);
    SmallVector<Value> &cloneResults = perLaunchCloneResults[yieldOperand];
    loweringState.emplaceBundleElements(bundleResult,
                                        SmallVector<Value>(cloneResults));
  }

  // The out-chain is the lowered completion of the body's yielded chain.
  if (parallelOp.hasChain()) {
    Value yieldedChain = yieldOp.getOperands().back();
    ASSERT_STREAM(
        isa<MO::ChainType>(yieldedChain.getType()),
        << "chained parallel must yield a trailing chain (enforced by verify)");
    auto it = resultToFusedChain.find(yieldedChain);
    // Or the in-chain, if the body passes it through.
    Value outChain =
        it != resultToFusedChain.end() ? it->second : loweredInChain;

    loweringState.emplaceExecuteLoweredValue(parallelOp.getOutChain(),
                                             outChain);
  }

  // Clones must stay alive, since loweredValues references their results.
  // Moving them into the body keeps lowerBlock from revisiting them; they die
  // with the MO graph.
  Block &bodyFront = parallelOp.getBodyRegion().front();
  for (Operation *clone : allClones)
    clone->moveBefore(&bodyFront, bodyFront.begin());

  return success();
}

LogicalResult MOGraphLowering::lower(MO::TensorBundleOp bundleOp) {
  SmallVector<Value> moElements = llvm::to_vector(bundleOp.getInputs());
  loweringState.emplaceBundleElements(bundleOp.getResult(),
                                      std::move(moElements));
  return success();
}

LogicalResult MOGraphLowering::lower(MO::TensorUnbundleOp unbundleOp) {
  auto moElements = loweringState.lookupBundleElements(unbundleOp.getInput());
  for (auto [result, moElem] :
       llvm::zip_equal(unbundleOp.getOutputs(), moElements)) {
    Value lowered = loweringState.lookupExecuteLoweredValue(moElem);
    Value chain = loweringState.getChainGuardingValueOrNull(moElem);
    if (!chain)
      chain = loweringState.getChainGuardingValueOrNull(lowered);
    if (chain)
      loweringState.emplaceExecuteLoweredBufferValue(result, lowered, chain);
    else
      loweringState.emplaceExecuteLoweredValue(result, lowered);
  }
  return success();
}

LogicalResult MOGraphLowering::lower(MO::CallOp callOp) {
  MGP::MOOpLoweringState opLoweringState(loweringState, callOp);
  OpBuilder b(callOp);

  StringRef targetModelName = callOp.getCallee();

  auto moduleOp = callOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return callOp.emitError() << "Failed to find parent ModuleOp";

  auto targetModel = moduleOp.lookupSymbol<MGP::ModelOp>(targetModelName);
  if (!targetModel)
    return callOp.emitError()
           << "Failed to find target model with name: " << targetModelName;

  // Implicit args first: chain, context, weights registry, devices.
  SmallVector<Value> callOperands;
  callOperands.push_back(loweringState.getCurrentExecuteChain());
  callOperands.push_back(loweringState.getExecuteMgpContext());
  callOperands.push_back(loweringState.getExecuteWeightsRegistry());

  for (auto deviceAttr : targetModel.getDevices()) {
    DeviceRefAttr deviceRef = cast<DeviceRefAttr>(deviceAttr);
    callOperands.push_back(loweringState.getExecuteDevice(deviceRef));
  }

  for (auto operand : callOp.getOperands()) {
    Value loweredValue = loweringState.lookupExecuteLoweredValue(operand);
    if (!loweredValue)
      return callOp.emitError() << "Operand not lowered yet: " << operand;

    callOperands.push_back(loweredValue);
  }

  // The chain, then the explicit results.
  SmallVector<Type> resultTypes;
  resultTypes.push_back(b.getType<MEF::ChainType>());
  for (auto result : callOp.getResults()) {
    Type loweredType =
        loweringState.getTypeConverter().convertType(result.getType());
    if (!loweredType) {
      return callOp.emitError()
             << "Failed to lower result type: " << result.getType();
    }
    resultTypes.push_back(loweredType);
  }

  auto targetSymbol = SymbolRefAttr::get(targetModel.getSymNameAttr());
  auto mgpCallOp = opLoweringState.create<MGP::CallOp>(
      resultTypes, targetSymbol, callOp.getPrefixAttr(), callOperands);
  Value callOutChain = mgpCallOp.getResult(0);

  for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
    loweringState.emplaceExecuteLoweredValue(callOp.getResult(i),
                                             mgpCallOp.getResult(i + 1));
    loweringState.guardValueWithChain(mgpCallOp.getResult(i + 1), callOutChain);
  }

  loweringState.setCurrentExecuteChain(callOutChain);

  return success();
}

//===----------------------------------------------------------------------===//
// MOGGToMGPPass
//===----------------------------------------------------------------------===//

/// A side-stream join is created at the end of its region, gated only on the
/// body's outputs, so the host enqueues the base stream's wait as soon as the
/// body is enqueued. Every base-stream op enqueued after that waits for the
/// side stream, data dependency or not. Move each join to the first consumer of
/// the region's results and gate it on that consumer's other chains, so the
/// host enqueues the independent base-stream work first.
static void deferSideStreamJoins(ModuleOp module) {
  SmallVector<MGP::DeviceWaitOp> joins;
  module.walk([&](MGP::DeviceWaitOp join) {
    if (join.getIsSideStreamJoin())
      joins.push_back(join);
  });
  llvm::SmallPtrSet<Value, 16> joinChains;
  for (MGP::DeviceWaitOp join : joins)
    joinChains.insert(join.getResult());

  for (MGP::DeviceWaitOp join : joins) {
    Block *block = join->getBlock();
    // The input holds made with the join move with it; they are not consumers.
    SmallVector<Operation *> holds;
    Operation *first = nullptr;
    for (Operation *user : join.getResult().getUsers()) {
      if (isa<MGP::ExtendLifetimeOp>(user)) {
        holds.push_back(user);
        continue;
      }
      Operation *op = block->findAncestorOpInBlock(*user);
      if (op && (!first || op->isBeforeInBlock(first)))
        first = op;
    }
    if (!first)
      continue;

    // Other joins in the consumer's chain are left out: gating joins on each
    // other would form a cycle.
    SmallVector<Value> chains{join.getInChain()};
    if (auto create = dyn_cast<GRT::ChainCreateOp>(first))
      for (Value chain : create->getOperands())
        if (!joinChains.contains(chain) && !llvm::is_contained(chains, chain))
          chains.push_back(chain);

    join->moveBefore(first);
    for (Operation *hold : holds)
      hold->moveAfter(join);
    if (chains.size() > 1) {
      OpBuilder b(join);
      join.getInChainMutable().assign(
          GRT::ChainCreateOp::create(b, join.getLoc(), chains).getResult());
    }
  }
}

namespace {
struct MOGGToMGPPass : public MGP::impl::MOGGToMGPBase<MOGGToMGPPass> {
  using MOGGToMGPBase::MOGGToMGPBase;

  void runOnOperation() override;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override;
};
} // namespace

void MOGGToMGPPass::getDependentDialects(DialectRegistry &registry) const {
  MOGGToMGPBase::getDependentDialects(registry);
}

void MOGGToMGPPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Kernel stubs created during lowering are registered through the cached
  // `SymbolTableAnalysis`, which the pass manager never invalidates.
  SymbolTable &moduleSymbolTable =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  std::deque<MO::GraphOp> worklist;

  DenseSet<StringAttr> completedGraphs;
  DenseMap<StringAttr, DenseSet<StringAttr>> graphDependencies;

  for (auto graphOp : theModule.getOps<MO::GraphOp>()) {
    worklist.push_back(graphOp);

    auto &dependencies = graphDependencies[graphOp.getNameAttr()];
    graphOp.walk([&](MO::CallOp callOp) {
      auto calleeName = callOp.getCalleeAttr().getAttr();
      dependencies.insert(calleeName);
    });
  }

  size_t lastProcessed = 0;
  // Process graphs in dependency order.
  while (!worklist.empty()) {
    // A full pass with no progress means a cycle.
    if (lastProcessed > worklist.size()) {
      theModule.emitError("cycle detected in graph ops");
      return signalPassFailure();
    }

    ++lastProcessed;

    MO::GraphOp graph = worklist.back();
    worklist.pop_back();

    auto &dependencies = graphDependencies[graph.getNameAttr()];

    if (!llvm::set_is_subset(dependencies, completedGraphs)) {
      worklist.push_front(graph);
      continue;
    }

    completedGraphs.insert(graph.getNameAttr());
    if (failed(MOGraphLowering(graph, compiledFrameworkLabel.getValue(),
                               minimumCpuAlignment.getValue(),
                               moduleSymbolTable)
                   .run()))
      return signalPassFailure();

    lastProcessed = 0;
  }

  deferSideStreamJoins(theModule);

  // A named dim on a stub's boundary would block MergeIdenticalFuncs
  // from deduplicating identical stubs. A nested pass so it stays testable on
  // its own.
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      &getContext(), mlir::OpPassManager::Nesting::Implicit);
  pm.addPass(MOGG::createAnonymizeDimNames());
  if (pm.run(theModule).failed())
    return signalPassFailure();
}
