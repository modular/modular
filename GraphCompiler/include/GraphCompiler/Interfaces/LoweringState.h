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
// Helper for lowering MO graphs into device-dependent buffer-semantic form.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_INTERFACES_LOWERINGSTATE_H
#define GENERICML_INTERFACES_LOWERINGSTATE_H

#include "GraphCompiler/Conversion/MGPSupport.h"
#include "GraphCompiler/Conversion/ParameterLowering.h"
#include "GraphCompiler/Interfaces/ConvertibleToMGPTypeInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Support/Diagnostic.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"

namespace M::MOGG {
class AsyncRoutineOp;
}

namespace M::MGP {

//===----------------------------------------------------------------------===//
// MOGraphLoweringState
//===----------------------------------------------------------------------===//

/// State for lowering one mo.graph to a mgp.model, shared by MOGGToMGP and
/// LowerableToMGPInterface implementers and built up as ops are lowered.
class MOGraphLoweringState {
public:
  MOGraphLoweringState(M::MO::GraphOp graph,
                       llvm::StringRef compiledFrameworkLabel,
                       size_t minimumCpuAlignment,
                       mlir::SymbolTable &moduleSymbolTable)
      : graph(graph), common(compiledFrameworkLabel, minimumCpuAlignment),
        init(graph.getContext()), execute(graph.getContext()),
        moduleSymbolTable(moduleSymbolTable) {
    builderContext.hostDevice = M::getHostDevice(graph);

    typeConverter.addConversion(
        [this](MO::TensorType type) -> std::optional<Type> {
          return getTypeBuilder(MO::getDeviceOrDefault(type))
              .tensorType(MGP::createTensorSpecType(type), /*mut=*/false,
                          type.isStable());
        });
    typeConverter.addConversion(
        [this](MO::BufferType bufferType) -> std::optional<Type> {
          // Buffers lower to mutable tensors: the `mut` marker is the only
          // record of in-place semantics past this boundary.
          auto type = MO::BufferType::toTensorType(bufferType);
          return getTypeBuilder(MO::getDeviceOrDefault(type))
              .tensorType(MGP::createTensorSpecType(type), /*mut=*/true);
        });
    typeConverter.addConversion([this](
                                    MO::ListType type) -> std::optional<Type> {
      auto tensorTy = dyn_cast<MO::TensorType>(type.getElementType());
      // TODO(#29947): Support other element types
      ASSERT_STREAM(tensorTy, << "only support lowering list of tensors");

      MLIRContext *ctx = type.getContext();
      MGP::TensorSpecType tensorSpecTy = MGP::createTensorSpecType(tensorTy);
      DeviceRefAttr tensorDeviceRef = MO::getDeviceOrDefault(tensorTy);
      auto mgpTensorTy =
          getTypeBuilder(tensorDeviceRef).tensorType(tensorSpecTy);
      return MGP::ListType::get(ctx, mgpTensorTy);
    });
    typeConverter.addConversion([](MO::ScalarType type) -> std::optional<Type> {
      auto errOrDtype = MO::toMLIRType(type.getContext(), type.getDtype());

      if (errOrDtype.isError()) {
        return std::nullopt;
      }

      return *errOrDtype;
    });

    typeConverter.addConversion([](MO::OpaqueType type) -> std::optional<Type> {
      return OpaqueType::get(type.getContext(), type.getSymbol(), std::nullopt);
    });

    typeConverter.addConversion([](MO::ChainType type) -> std::optional<Type> {
      return MEF::ChainType::get(type.getContext());
    });

    // Foreign types.
    typeConverter.addConversion([](Type type) -> std::optional<Type> {
      if (auto interface = dyn_cast<ConvertibleToMGPTypeInterface>(type))
        return interface.convertType();
      return {};
    });
  }

  ErrorOr<ElementsAttr> lowerElementsAttr(ElementsAttr elementsAttr);

  /// Common state used by both the init and execute phases.
  struct CommonState {
    explicit CommonState(llvm::StringRef compiledFrameworkLabel,
                         size_t minimumCpuAlignment)
        : compiledFrameworkLabel(compiledFrameworkLabel),
          minimumCpuAlignment(minimumCpuAlignment) {}

    MGP::ModelOp modelOp;

    /// The name of the target framework, e.g. "mgp", "tf", etc.
    llvm::StringRef compiledFrameworkLabel;

    /// For all CPU buffers, constant and allocated.
    size_t minimumCpuAlignment;

    CommonState(const CommonState &) = delete;
    void operator=(const CommonState &) = delete;
  };

  /// State used when building an invokable for the 'init' phase.
  struct InitState {
    InitState(MLIRContext *ctx) : builder(ctx) {}

    Value chain;

    MGP::ContextCreateOp mgpContext;

    /// This is a builder appending operations to the end of the invokable.
    OpBuilder builder;

    InitState(const InitState &) = delete;
    void operator=(const InitState &) = delete;
  };

  /// State used when building an invokable for the 'execute' phase.
  struct ExecuteState {
    ExecuteState(MLIRContext *ctx) : builder(ctx) {}

    /// The input chain passed as the first argument to the 'execute' block,
    /// or an equivalent SSA value in a control flow sub-block.
    TypedValue<MEF::ChainType> rootChain;

    /// Sequences side-effecting operations such as printing.
    TypedValue<MEF::ChainType> currentChain;

    /// Passed in as the second argument.
    TypedValue<MGP::ContextType> mgpContext;

    /// Keeps track of what runtime SSA values the graph SSA values lower to.
    DenseMap<Value, Value> loweredValues;

    /// Maps !mo.bundle SSA values to their per-element MO-level values, so
    /// consumers needn't look through the defining TensorBundleOp.
    DenseMap<Value, SmallVector<Value>> bundleElements;

    /// The chain guarding each buffer / tensor's first write; readers must
    /// depend on it.
    DenseMap<Value, Value> loweredBufferChains;

    /// Names dense_resource attributes capturing MO constants.
    size_t nextConstId = 0;

    /// This is a builder appending operations to the end of the invokable.
    OpBuilder builder;

    /// Maps a lowered assert to the lowered assert it was chained to in MO.
    DenseMap<Operation *, Operation *> parentAssert;

    ExecuteState(const ExecuteState &) = delete;
    void operator=(const ExecuteState &) = delete;
  };

  // Base class for clients that want to inject their own custom state.
  struct ClientState {
    virtual ~ClientState() = default;
  };

  bool hasClientState() const { return bool(clientState); }

  // Only to be called once.
  template <typename T>
  void initClientState(T &&initState) {
    assert(!hasClientState());
    clientState = std::make_unique<T>(std::forward<T>(initState));
  }

  template <typename T>
  T &getClientState() {
    return static_cast<T &>(*clientState);
  }

  MGP::ModelOp getModelOp() { return common.modelOp; }

  /// New module-scope symbols (e.g. kernel stubs) must be inserted through
  /// this: the pass manager never invalidates `SymbolTableAnalysis`, so
  /// downstream passes wouldn't see them otherwise.
  mlir::SymbolTable &getModuleSymbolTable() { return moduleSymbolTable; }

  void setModelOp(MGP::ModelOp modelOp) { common.modelOp = modelOp; }

  llvm::StringRef getCompiledFrameworkLabel() {
    return common.compiledFrameworkLabel;
  }

  size_t getMinimumCpuAlignment() { return common.minimumCpuAlignment; }

  MO::GraphOp &getGraph() { return graph; }

  TypeConverter &getTypeConverter() { return typeConverter; }

  Value getInitChain() { return init.chain; }

  void setInitChain(Value chain) { init.chain = chain; }

  TypedValue<DeviceContextType> getInitDevice(DeviceRefAttr dev) {
    return initDevices[dev];
  }

  void setInitDevice(DeviceRefAttr dev, TypedValue<DeviceContextType> val) {
    initDevices[dev] = val;
  }

  OpBuilder &getInitOpBuilder() { return init.builder; }

  void setInitMgpContext(MGP::ContextCreateOp mgpContext) {
    init.mgpContext = mgpContext;
  }

  MGP::ContextCreateOp &getInitMgpContext() { return init.mgpContext; }

  TypedValue<MGP::OpaqueType> getInitWeightsRegistry() {
    return cast<TypedValue<MGP::OpaqueType>>(
        common.modelOp.getInitWeightsRegistry());
  }

  OpBuilder &getExecuteOpBuilder() { return execute.builder; }

  void setExecuteMgpContext(Value mgpContext) {
    execute.mgpContext = cast<TypedValue<MGP::ContextType>>(mgpContext);
  }

  TypedValue<MGP::ContextType> getExecuteMgpContext() {
    return execute.mgpContext;
  }

  TypedValue<DeviceContextType> getExecuteDevice(DeviceRefAttr dev) {
    if (!dev)
      dev = DeviceRefAttr::getDefault(execute.builder.getContext());

    // While lowering an `mo.sequence` body onto a side stream, every op
    // lowered through here picks up the side-stream context.
    if (auto it = execDevices.find(dev); it != execDevices.end())
      return it->second;

    for (auto [deviceRefAttr, deviceContext] : llvm::zip(
             common.modelOp.getDevices(), common.modelOp.getExecuteDevices())) {
      if (deviceRefAttr == dev)
        return cast<TypedValue<MGP::DeviceContextType>>(deviceContext);
    }
    llvm_unreachable("Could not find device in model args");
  }

  /// Make `getExecuteDevice(dev)` return `ctx` (e.g. a side-stream view) until
  /// cleared. Not nestable; the MO verifier forbids nested `mo.sequence`.
  void setExecuteDeviceOverride(DeviceRefAttr dev,
                                TypedValue<DeviceContextType> ctx) {
    assert(!execDevices.count(dev) &&
           "nested execute-device override for the same device");
    execDevices[dev] = ctx;
  }

  void clearExecuteDeviceOverride(DeviceRefAttr dev) { execDevices.erase(dev); }

  TypedValue<MGP::OpaqueType> getExecuteWeightsRegistry() {
    return cast<TypedValue<MGP::OpaqueType>>(
        common.modelOp.getExecuteWeightsRegistry());
  }

  void setRootExecuteChain(Value chain) {
    execute.rootChain = cast<TypedValue<MEF::ChainType>>(chain);
  }

  /// The 'root' chain of the 'execute' block, for non-side-effecting
  /// operations.
  TypedValue<MEF::ChainType> getExecuteChain() { return execute.rootChain; }

  /// The 'current' chain of the 'execute' block, for side-effecting
  /// operations.
  Value getCurrentExecuteChain() { return execute.currentChain; }

  void setCurrentExecuteChain(Value chain) {
    execute.currentChain = cast<TypedValue<MEF::ChainType>>(chain);
  }

  /// The runtime value a mo.graph SSA value (often a `mo.tensor`) lowers to.
  Value lookupExecuteLoweredValue(Value key) {
    return execute.loweredValues.lookup(key);
  }

  Value getChainGuardingValueOrNull(Value key) {
    return execute.loweredBufferChains.lookup(key);
  }

  void guardValueWithChain(Value key, Value chain) {
    execute.loweredBufferChains.try_emplace(key, chain);
  }

  /// Overwrite an existing chain guard, e.g. after region fusion.
  void replaceChainGuard(Value key, Value newChain) {
    execute.loweredBufferChains[key] = newChain;
  }

  using ParamPlaceholderOp = ParameterLoweringState::PlaceholderOp;

  /// Record the runtime value a mo.graph SSA value lowers to.
  void emplaceExecuteLoweredValue(Value key, Value value) {
    if (auto placeholder = value.getDefiningOp<ParamPlaceholderOp>())
      placeholderMap.find(placeholder)->getSecond().emplace_back(key);

    execute.loweredValues.try_emplace(key, value);
  }

  void emplaceExecuteLoweredBufferValue(Value key, Value value, Value chain) {
    guardValueWithChain(key, chain);
    guardValueWithChain(value, chain);
    execute.loweredValues.try_emplace(key, value);
  }

  void emplaceBundleElements(Value bundleVal, SmallVector<Value> elements) {
    execute.bundleElements.try_emplace(bundleVal, std::move(elements));
  }

  SmallVector<Value> lookupBundleElements(Value bundleVal) {
    auto it = execute.bundleElements.find(bundleVal);
    ASSERT_STREAM(it != execute.bundleElements.end(),
                  << "bundle value has no registered elements");
    return it->second;
  }

  size_t getExecuteNextConstId() { return execute.nextConstId++; }

  /// Returns the shape of a lowered tensor which is on the given device.
  TypedValue<MGP::TensorSpecType> getShape(Location loc, Value loweredTensor,
                                           DeviceRefAttr device);

  /// Assumes the rank is known.
  SmallVector<Value> getDynamicDims(Location loc, Value loweredTensor,
                                    DeviceRefAttr device);

  MGP::TensorSpecGetDimOp getDim(Location loc, Value loweredTensor,
                                 uint64_t axis, DeviceRefAttr device);

  struct ArgLoweringResult {
    bool hasParameters;
    Value loweredArg;
  };

  /// Maps the dynamic shape parameters of `inArg` (a !mo.tensor lowered to the
  /// !mgp.tensor `outArg`) to their MGP values. Returns whether there were
  /// parameters, and `outArg` guarded by any runtime assertions.
  ErrorOr<ArgLoweringResult> lowerArg(Region *parentRegion, BlockArgument inArg,
                                      BlockArgument outArg);

  /// Lowers a parameter expression: unseen references become placeholders,
  /// operators recurse, and other attributes go to the hook.
  ErrorOr<Value> lowerExpression(Region *parentRegion, Location loc,
                                 TypedAttr attr);

  /// Lower a parameter declaration once, resolving its placeholder if any.
  ErrorOrSuccess lowerParamDecl(Region *parentRegion,
                                KGEN::ParamDeclAttr declAttr, Value value);

  /// The values of `shape`'s parametric dimensions, in order.
  ErrorOr<SmallVector<Value>> lowerShapeAttrDynDims(Region *parentRegion,
                                                    MOSH::ShapeAttr shapeAttr,
                                                    Location loc);

  const BufferBuilderContext &getBufferBuilderContext() const {
    return builderContext;
  }

  /// Register a type builder for the given device.
  void registerTypeBuilder(DeviceRefAttr device) {
    builderContext.registerTypeBuilder(device);
  }

  /// Returns the type builder to use for device.
  MGP::MGPTypeBuilder getTypeBuilder(DeviceRefAttr device) {
    return builderContext.getTypeBuilder(device);
  }

  /// Register the given buffer builder for the given device.
  void
  registerBufferBuilder(DeviceRefAttr device,
                        std::unique_ptr<MGP::MGPBufferBuilder> bufferBuilder) {
    builderContext.registerBufferBuilder(device, std::move(bufferBuilder));
  }

  /// The returned builder is owned by this object.
  MGP::MGPBufferBuilder &getBufferBuilder(DeviceRefAttr device) {
    return builderContext.getBufferBuilder(device);
  }

  /// Return the 'host' device for this graph.
  DeviceRefAttr getHostDevice() { return builderContext.hostDevice; }

  /// Returns the buffer builder to use for the 'host' device.
  MGP::MGPBufferBuilder &getHostBufferBuilder() {
    return builderContext.getHostBufferBuilder();
  }

  /// Packs outputs and unpacks inputs of a MOGG::AsyncRoutineOp.
  static void lowerMOGGABI(MOGG::AsyncRoutineOp kernelOp);

private:
  //===-----------------------------------------------------------------===//
  // Parameter lowering helpers.
  //===-----------------------------------------------------------------===//

  /// Lowers an operator attribute to the appropriate mop index op.
  static ErrorOr<Value> lowerParamOperator(OpBuilder &builder, Location loc,
                                           KGEN::ParamOperatorAttr opAttr,
                                           ValueRange operands);

  /// Lowers an integer attribute to a index constant op.
  static Operation *lowerKGENConstantAttr(OpBuilder &builder, Location loc,
                                          TypedAttr intAttr);

  /// Lower a parameter expression.
  ErrorOr<Value> lowerExpressionImpl(ParameterLoweringState &state,
                                     OpBuilder &builder, Location loc,
                                     Region *parentRegion, TypedAttr attr,
                                     ValueRange operands);

  /// Lower a parameter expression used as a dimension.
  ErrorOr<Value> lowerDimensionExpression(ParameterLoweringState &state,
                                          OpBuilder &builder, Location loc,
                                          Region *parentRegion, TypedAttr dim);

private:
  //===-----------------------------------------------------------------===//
  // Lowering state
  //===-----------------------------------------------------------------===//

  /// The graph being lowered.
  MO::GraphOp graph;
  /// Type converter for lowering from MO types to MGP and other
  /// device-specific types.
  TypeConverter typeConverter;

  /// All the type and buffer builders we've registered.
  BufferBuilderContext builderContext;

  CommonState common;
  InitState init;
  ExecuteState execute;

  /// Owned by the pass's `SymbolTableAnalysis`.
  mlir::SymbolTable &moduleSymbolTable;

  // Arbitrary client state.
  std::unique_ptr<ClientState> clientState;

  /// The state and interface for parameter lowering.
  ParameterLoweringState params;

  DenseMap<Value, SmallVector<Value>> placeholderMap;

  /// Mapping from device ref to value in init.
  SmallDenseMap<DeviceRefAttr, TypedValue<DeviceContextType>> initDevices;

  /// Mapping from device ref to value in exec.
  SmallDenseMap<DeviceRefAttr, TypedValue<DeviceContextType>> execDevices;
};

//===----------------------------------------------------------------------===//
// MOOpLoweringState
//===----------------------------------------------------------------------===//

/// MO ops only return !mo.tensors; other result types would need a variant.
using MOOutputStorage = BufferizedTensor;

/// Manages the value-to-buffer-semantic transition when lowering one MO op.
struct MOOpLoweringState {
  MOGraphLoweringState &loweringState;
  OpBuilder &builder;
  mlir::Operation *opBeingLowered;
  mlir::Location loc;
  TypedValue<MGP::ContextType> context;
  TypedValue<MGP::OpaqueType> weightsRegistry;
  uint64_t minimumCpuAlignment;

  MOOpLoweringState(MGP::MOGraphLoweringState &loweringState,
                    mlir::Operation *op)
      : loweringState(loweringState),
        builder(loweringState.getExecuteOpBuilder()), opBeingLowered(op),
        loc(op->getLoc()), context(loweringState.getExecuteMgpContext()),
        weightsRegistry(loweringState.getExecuteWeightsRegistry()),
        minimumCpuAlignment(loweringState.getMinimumCpuAlignment()) {}

  //===-----------------------------------------------------------------===//
  // Main API
  //===-----------------------------------------------------------------===//

  /// Returns the lowered operands: !mo.tensors as their buffer on the type's
  /// storage device, everything else as is.
  SmallVector<Value> getLoweredOperands();

  /// Merges `chain` with any other chains this op must wait on.
  Value getGuardChain(Value chain);

  /// Derives the output specs from the op's result types.
  FailureOr<SmallVector<MOOutputStorage>> createOutputStorage(Value inChain);

  /// Returns destination-passing-style storage for the op's !mo.tensor
  /// results: a fresh buffer on the storage device, paired with a tensor spec.
  FailureOr<SmallVector<MOOutputStorage>>
  createOutputStorage(Value inChain,
                      ArrayRef<TypedValue<MGP::TensorSpecType>> outSpecs);

  /// Create the outputs from the storage and map the MO op results to them.
  void createOutputs(Value inChain,
                     SmallVector<MOOutputStorage> &loweredOutputsStorage);

  //===-----------------------------------------------------------------===//
  // Input and Output helpers
  //===-----------------------------------------------------------------===//

  /// Returns the bufferized form for the MO tensor value.
  BufferizedTensor getBufferizedInput(Value moTensor);

  FailureOr<TypedValue<MGP::TensorSpecType>>
  createTensorSpec(MO::TensorType moTensorType);

  /// Returns the tensor specs to represent all the op's outputs.
  FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>>
  createOutputTensorSpecs();

  /// Allocates a buffer sized for `spec` on moTensorType's storage device.
  FailureOr<BufferizedTensor>
  createBuffer(Value inChain, TypedValue<TensorSpecType> tensorSpec,
               Type moTensorType);

  /// Creates a tensor on moTensor's storage device and maps moTensor to it.
  void createOutputTensor(Value moTensor, Value inChain, Value buffer,
                          TypedValue<MGP::TensorSpecType> tensorSpec);

  void createOutputTensors(
      Value inChain, SmallVector<Value> &buffers,
      SmallVector<TypedValue<MGP::TensorSpecType>> &tensorSpecs);

  //===-----------------------------------------------------------------===//
  // MLIR create helpers
  //===-----------------------------------------------------------------===//

  /// Creates an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    return OpTy::create(builder, loc, std::forward<Args>(args)...);
  }

  /// The buffer builder for moTensorType's storage device.
  MGP::MGPBufferBuilder &getStorageBufferBuilder(Type moTensorType);

  // These redirect to MGPBufferBuilder for moTensorType's storage device.

  std::pair<Value, TypedValue<TensorSpecType>>
  tensorExtract(Value loweredTensor, Type moTensorType) {
    MGP::MGPBufferBuilder &bufferBuilder =
        getStorageBufferBuilder(moTensorType);
    // Separate statements fix the op order; argument evaluation order is
    // unspecified.
    auto buffer =
        bufferBuilder.tensorExtractBuffer(builder, loc, loweredTensor);
    auto spec =
        bufferBuilder.tensorExtractTensorSpec(builder, loc, loweredTensor);
    loweringState.guardValueWithChain(
        buffer, loweringState.getChainGuardingValueOrNull(loweredTensor));
    return std::make_pair(buffer, spec);
  }

  Value tensorCreate(Value inChain, Value buffer,
                     TypedValue<TensorSpecType> tensorSpec, Type moTensorType) {
    bool isMut = isa<MO::BufferType>(moTensorType);

    Value tensor =
        getStorageBufferBuilder(moTensorType)
            .tensorCreate(builder, loc, context, inChain, buffer, tensorSpec,
                          /*mut=*/isMut);
    loweringState.guardValueWithChain(
        tensor, loweringState.getChainGuardingValueOrNull(buffer));
    return tensor;
  }

  Value hostBufferConstant(TypedValue<MEF::ChainType> inChain,
                           ElementsAttr value, Type moTensorType) {
    return loweringState.getHostBufferBuilder().bufferConstant(
        builder, loc, inChain, context, value);
  }

  Value hostBufferConstantExternal(TypedValue<MEF::ChainType> inChain,
                                   TypedValue<MGP::OpaqueType> name,
                                   uint64_t size, uint64_t align, bool hasAlias,
                                   bool isPlaceholder) {
    return loweringState.getHostBufferBuilder().bufferConstantExternal(
        builder, loc, inChain, context, weightsRegistry, name, size, align,
        hasAlias, isPlaceholder);
  }

  Value deviceBufferConstantExternal(DeviceRefAttr device,
                                     TypedValue<MEF::ChainType> inChain,
                                     TypedValue<MGP::OpaqueType> name,
                                     uint64_t size, uint64_t align,
                                     bool hasAlias, bool isPlaceholder) {
    return loweringState.getBufferBuilder(device).bufferConstantExternal(
        builder, loc, inChain, context, weightsRegistry, name, size, align,
        hasAlias, isPlaceholder);
  }

  Value hostDebugPrint(Value inChain, StringAttr data, StringAttr label) {
    return loweringState.getHostBufferBuilder().debugPrint(
        builder, loc, context, inChain, data, label);
  }

  Value debugTensorPrint(Value inChain, Value buffer,
                         TypedValue<TensorSpecType> tensorSpec,
                         StringAttr label, Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        .debugTensorPrint(builder, loc, context, inChain, buffer, tensorSpec,
                          label);
  }
  Value hostDebugTensorPrint(Value inChain, Value buffer,
                             TypedValue<TensorSpecType> tensorSpec,
                             StringAttr label) {
    return loweringState.getHostBufferBuilder().debugTensorPrint(
        builder, loc, context, inChain, buffer, tensorSpec, label);
  }

  Value bufferTransferFromHost(Value inChain,
                               TypedValue<MGP::DeviceBufferType> srcBuffer,
                               Value dstBuffer, Type dstMoTensorType) {
    auto deviceRef = getStorageBufferBuilder(dstMoTensorType).getDevice();
    auto deviceContext = loweringState.getExecuteDevice(deviceRef);
    auto chain =
        getStorageBufferBuilder(dstMoTensorType)
            .bufferTransferFromHost(builder, loc, context, inChain,
                                    &loweringState.getHostBufferBuilder(),
                                    srcBuffer, dstBuffer, deviceContext);
    loweringState.guardValueWithChain(dstBuffer, chain);
    return chain;
  }
  Value bufferTransferToHost(Value inChain, Value srcBuffer,
                             TypedValue<MGP::DeviceBufferType> dstBuffer,
                             Type srcMoTensorType) {
    auto deviceRef = getStorageBufferBuilder(srcMoTensorType).getDevice();
    auto deviceContext = loweringState.getExecuteDevice(deviceRef);
    return getStorageBufferBuilder(srcMoTensorType)
        .bufferTransferToHost(builder, loc, context, inChain, srcBuffer,
                              &loweringState.getHostBufferBuilder(), dstBuffer,
                              deviceContext);
  }
  Value bufferTransferDevice(Value inChain, Value srcBuffer, Value dstBuffer,
                             Type srcMoTensorType, Type dstMoTensorType) {
    // The only op spanning two devices. The D2D device context synchronizes
    // them. TODO(GEX-1859): Explicit sync to enable compute overlap.

    DeviceRefAttr srcDeviceRef =
        getStorageBufferBuilder(srcMoTensorType).getDevice();
    DeviceRefAttr dstDeviceRef =
        getStorageBufferBuilder(dstMoTensorType).getDevice();
    auto dstDeviceContext = loweringState.getExecuteDevice(dstDeviceRef);
    auto srcDeviceContext = loweringState.getExecuteDevice(srcDeviceRef);

    return getStorageBufferBuilder(srcMoTensorType)
        .bufferTransferDevice(builder, loc, context, inChain, srcBuffer,
                              &getStorageBufferBuilder(dstMoTensorType),
                              dstBuffer, srcDeviceContext, dstDeviceContext);
  }

  Value bufferConcat(Value inChain, Value outBuffer, ArrayRef<Value> inBuffers,
                     Type outMoTensorType) {
    auto ty = cast<MO::TensorType>(outMoTensorType);
    auto deviceContext = loweringState.getExecuteDevice(ty.getDeviceRef());
    return getStorageBufferBuilder(outMoTensorType)
        .bufferConcat(builder, loc, deviceContext, inChain, outBuffer,
                      inBuffers);
  }

  Value bufferSlice(Value inBuffer, Value offset, Value size,
                    Type outMoTensorType) {
    auto deviceRef = getStorageBufferBuilder(outMoTensorType).getDevice();
    auto deviceContext = loweringState.getExecuteDevice(deviceRef);
    return getStorageBufferBuilder(outMoTensorType)
        .bufferSlice(builder, loc, inBuffer, offset, size, deviceContext);
  }

  Value bufferToBool(Value inChain, Value buffer, Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        .bufferToBool(builder, loc, context, inChain, buffer);
  }

  Value bufferSetWithIndex(Value buffer, ArrayRef<Value> values,
                           Type moTensorType) {
    return getStorageBufferBuilder(moTensorType)
        .bufferSetWithIndex(builder, loc, context, buffer, values);
  }

  Value hostBufferAlloc(TypedValue<MEF::ChainType> inChain, Value size,
                        uint64_t align) {
    auto deviceContext =
        loweringState.getExecuteDevice(loweringState.getHostDevice());
    return loweringState.getHostBufferBuilder().bufferAlloc(
        builder, loc, context, inChain, size, deviceContext, align);
  }

  Value bufferAlloc(TypedValue<MEF::ChainType> inChain, Value size,
                    uint64_t align, Type moTensorType) {
    auto deviceRef = getStorageBufferBuilder(moTensorType).getDevice();
    auto deviceContext = loweringState.getExecuteDevice(deviceRef);
    return getStorageBufferBuilder(moTensorType)
        .bufferAlloc(builder, loc, context, inChain, size, deviceContext,
                     align);
  }

  Value bufferSizeBytes(TypedValue<TensorSpecType> tensorSpec,
                        Type moTensorType) {
    int64_t dtypeSize =
        dyn_cast<MO::TensorType>(moTensorType).getDtype().getSizeInBytes();
    Value numBytes = mlir::index::ConstantOp::create(builder, loc, dtypeSize);
    for (size_t i = 0; i < tensorSpec.getType().getRank(); ++i) {
      Value dimVal =
          builder.createOrFold<MGP::TensorSpecGetDimOp>(loc, tensorSpec, i);
      numBytes = builder.createOrFold<mlir::index::MulOp>(
          loc, builder.getIndexType(), numBytes, dimVal);
    }
    return numBytes;
  }

  Value sync(Value chain) {
    auto device = loweringState.getHostDevice();
    auto deviceContext = loweringState.getExecuteDevice(device);
    return loweringState.getBufferBuilder(device).sync(builder, loc, context,
                                                       deviceContext, chain);
  }
};

} // namespace M::MGP

#endif // GENERICML_INTERFACES_LOWERINGSTATE_H
