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
#ifndef GENERICML_INTERFACES_EMITMOJOSTATE_H
#define GENERICML_INTERFACES_EMITMOJOSTATE_H

#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGEnums.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GraphCompiler/MOGGDialect/Support/EmitMojoRegion.h"
#include "GraphCompiler/MOGGDialect/Support/MojoFile.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"

#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Support/AssertStream.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TypeName.h"

#include <memory>
#include <utility>

namespace M::MOGG {
// Forward-declared to break MOGGOps.h -> EmitsMojoInterface.h ->
// EmitMojoState.h -> {KernelDeclAdaptor,MOGGOps}.h -> MOGGOps.h.
class MojoRegistry;
struct KernelDeclFunctionAdaptor;
struct KernelDeclAdaptor;
class PrimitiveDeclOp;
class ShapeFunctionDeclOp;
class GraphBlobRegion;

/// A fused load/store receiver's `ManagedTensorSlice` decomposed into a bare
/// pointer, runtime shape/strides, and comptime `StaticTensorSpec`, all
/// device-passable, so fusion lanes capture these instead of the slice and
/// `TensorLoadOp`/`TensorStoreOp::emitMojo` go through
/// `simd_{load_from,store_into}_tensor_pointer`.
struct FusedTensorArg {
  MojoValue ptr;
  MojoValue shape;
  MojoValue strides;
  MojoValue spec;
};

/// State for lowering MOGG operations to Mojo text.
struct EmitMojoState {
  EmitMojoState(MojoRegistry &registry, bool enableOpTraceComments = false,
                bool enableComputeLambdas = false,
                bool enableDebugDumpFusedTensorSpecs = false)
      : registry(registry), enableComputeLambdas(enableComputeLambdas),
        enableOpTraceComments(enableOpTraceComments),
        enableDebugDumpFusedTensorSpecs(enableDebugDumpFusedTensorSpecs) {}

  /// The Mojo source file being emitted: the function bodies in order, then
  /// the module-scope declarations they need. Emitters write through
  /// `file.output()`, which is the declarations section while a declaration
  /// is being emitted.
  MojoFile file;

  /// Emits `definition` into the declarations section unless this file already
  /// has it. `definition` is the text of the declaration itself, which is what
  /// identifies it: there is no separate id to keep in sync.
  void declare(llvm::StringRef definition) {
    // Inside a declaration, this text would be written at the declaration's
    // indentation and then never at file scope, since the set below records it
    // as written first.
    ASSERT_STREAM(!file.inDeclarations(),
                  << "a declaration cannot be written from inside another; "
                     "leave the declaration scope first");
    if (!emittedDeclarations.insert(definition).second)
      return; // already in this file.
    file.declarations().verbatim(MojoValue(definition));
  }

  /// The `mogg.routine.async` return values, set by ops feeding
  /// `mogg.output` (e.g. AsyncReadyOp or AsyncPack*).
  SmallVector<MojoValue> mojoReturnValues;
  /// Dense resources, passed in the void** input after inputs and outputs.
  SmallVector<std::tuple<MojoValue, MojoValue>> mojoConstantResources;

  MojoRegistry &registry;

  /// Values kept alive with `_ = someVar` at function end, where the Mojo
  /// compiler misjudges lifetimes (mostly lambdas, until the new closures).
  SmallVector<MojoValue> keepAliveValues;

  /// Emitted Mojo names per TaskHandle, from CallAsyncOp to AsyncJoinOp.
  struct TaskHandleInfo {
    MojoValue task;  // The Task variable from create_task()
    MojoValue error; // The Optional[Error] variable
    // The task's coroutine points into this closure, so it must outlive the
    // task's wait.
    MojoValue wrapper;
  };
  llvm::DenseMap<Value, TaskHandleInfo> taskHandleInfos;

  /// Pushes a region built from `args` for the ops lowered while the scope
  /// lives, and pops it when the scope ends.
  template <typename Region, typename... Args>
  [[nodiscard]] EmitMojoRegionScope enterRegion(Args &&...args) {
    // The size before this region, so the scope can check on exit that nothing
    // pushed after it is still alive.
    size_t regionCount = regions.size();
    regions.push_back(std::make_unique<Region>(std::forward<Args>(args)...));
    return EmitMojoRegionScope(regions, regionCount);
  }

  /// The innermost region of this type; asserts if none was pushed.
  template <typename Region>
  Region &currentRegion() {
    if (Region *region = findRegion<Region>())
      return *region;
    ASSERT_STREAM(false, << "no " << llvm::getTypeName<Region>()
                         << " in scope");
    llvm_unreachable("no region of this type in scope");
  }

  /// The innermost region of this type, if one was pushed.
  template <typename Region>
  Region *findRegion() {
    for (auto frame = regions.rbegin(); frame != regions.rend(); ++frame)
      if ((*frame)->kind() == Region::kKind)
        return static_cast<Region *>(frame->get());
    return nullptr;
  }

  /// Drops the bindings made while it lives, and restores the outer ones they
  /// shadowed. A value lowered inside a Mojo lambda is local to it (a constant
  /// is splat to the lambda's width), so another lambda lowers the same value
  /// again.
  class ValueScope {
  public:
    explicit ValueScope(EmitMojoState &state)
        : scope(state.mlirValueToMojoValue) {}

  private:
    llvm::ScopedHashTableScope<Value, MojoValue> scope;
  };

  /// If true, generate compute lambdas when possible.
  bool enableComputeLambdas;

private:
  /// The declarations already written into `file`.
  llvm::StringSet<> emittedDeclarations;

  /// The regions of the constructs currently being emitted, innermost last.
  EmitMojoRegionStack regions;

  llvm::ScopedHashTable<Value, MojoValue> mlirValueToMojoValue;
  // Holds the bindings made outside any `ValueScope`.
  llvm::ScopedHashTableScope<Value, MojoValue> outermostValueScope{
      mlirValueToMojoValue};

  /// The decomposed `ManagedTensorSlice` components of a fused load/store
  /// receiver (see `setFusedTensorArgFor`).
  mlir::DenseMap<Value, FusedTensorArg> fusedTensorArgs;

  /// Emit comments naming the IR op being processed.
  bool enableOpTraceComments;

  /// Print the fused tensor specs inside fused lambdas.
  bool enableDebugDumpFusedTensorSpecs;

public:
  using KernelBodyBuilder =
      llvm::function_ref<void(const MOGG::FunctionBody &)>;

  MojoValue getMojoValueFor(Value val) const {
    ASSERT_STREAM(mlirValueToMojoValue.count(val),
                  << "value not bound to any MojoValue");
    return mlirValueToMojoValue.lookup(val);
  }

  /// Binds `val`. Inside a `ValueScope`, a binding from an outer scope is
  /// shadowed until this scope ends.
  void setMojoValueFor(Value val, MojoValue mojoVal) {
    ASSERT_STREAM(!mlirValueToMojoValue.count(val) ||
                      mlirValueToMojoValue.getCurScope() !=
                          &outermostValueScope,
                  << "value already bound to a MojoValue");
    mlirValueToMojoValue.insert(val, std::move(mojoVal));
  }

  /// Record a fused load/store receiver's `FusedTensorArg`, keyed by its tensor
  /// operand.
  void setFusedTensorArgFor(Value val, FusedTensorArg arg) {
    fusedTensorArgs[val] = std::move(arg);
  }

  const FusedTensorArg &getFusedTensorArgFor(Value val) const {
    auto it = fusedTensorArgs.find(val);
    ASSERT_STREAM(it != fusedTensorArgs.end(),
                  << "tensor value has no recorded fused-arg components");
    return it->second;
  }

  /// Generates `_ = value` at the end of the function.
  void markAsKeepAlive(MojoValue value) { keepAliveValues.push_back(value); }

  void genAllKeepAliveStatements() {
    for (const MojoValue &val : keepAliveValues)
      file->keepAlive(val);
    keepAliveValues.clear();
  }

  /// Lower `op` to Mojo using the EmitMojoInterface.
  LogicalResult lowerOpToMojo(Operation *op);

  /// Convert an mlir::Type to Mojo text. Parameterized opaque types become
  /// LegacyOpaquePointer; see mlirOpaqueTypeToMojoType.
  static MojoValue mlirTypeToMojoType(Type mlirType) {
    if (isa<MGP::DeviceBufferType>(mlirType)) {
      // Every `!mgp.buffer` value lowers to the storage-carrying composite;
      // producers build it and consumers unwrap its `.view` internally.
      return kStructOwnedByteBuffer;
    } else if (auto deviceTensorType =
                   dyn_cast<MGP::DeviceTensorType>(mlirType)) {
      // Rare: `!mgp.tensor` types are usually inferred, not spelled. Scalars
      // are rank-1, matching `unpack_tensor`.
      MGP::TensorSpecType tensorSpec = deviceTensorType.getTensorSpec();
      auto bufferRank =
          int64_t(tensorSpec.getRank() == 0 ? 1 : tensorSpec.getRank());
      SmallVector<MojoValue> params = {
          mlirDTypeToMojoType(tensorSpec.getOptDType().value()),
          MojoValue(bufferRank)};
      if (deviceTensorType.getMut())
        params.push_back(MojoValue(kParameterKind)
                             .equal(MojoValue(kStructDeviceGraphInputKind)
                                        .get(kInputKindMutable)));
      else if (M::isHostPlaced(deviceTensorType.getDevice()))
        params.push_back(MojoValue(kParameterKind)
                             .equal(MojoValue(kStructDeviceGraphInputKind)
                                        .get(kInputKindHost)));
      return MojoValue(kStructOwnedTensor)[params];
    } else if (isa<MEF::ChainType>(mlirType)) {
      return kComptimeOpaquePointer;
    } else if (isa<MGP::DeviceContextType, ContextType>(mlirType)) {
      return kStructDeviceContext;
    } else if (isa<PointerType>(mlirType)) {
      // A raw data pointer at the MOGG-ABI stub boundary.
      return kComptimeOpaquePointer;
    } else if (isa<DimType>(mlirType)) {
      // A runtime dynamic dimension is passed as a plain `Int`.
      return kStructInt;
    } else if (isa<MGP::DeviceGraphType>(mlirType)) {
      return kStructDeviceGraph;
    } else if (isa<MGP::DeviceGraphBuilderType>(mlirType)) {
      return kStructDeviceGraphBuilder;
    } else if (isa<MGP::DeviceGraphNodeType>(mlirType)) {
      return kStructDeviceGraphNode;
    } else if (isa<MGP::DeviceGraphCacheType>(mlirType)) {
      return kStructDeviceGraphCache;
    } else if (auto integerTy = dyn_cast<IntegerType>(mlirType)) {
      if (integerTy.getWidth() == 1)
        return kStructBool;
      return kStructInt;
    } else if (auto tensorType = dyn_cast<MO::TensorType>(mlirType)) {
      return mlirTensorTypeToMojoType(tensorType);
    } else if (auto tensorSpecType = dyn_cast<MGP::TensorSpecType>(mlirType)) {
      return MojoValue(kStructIndexList)[int64_t(tensorSpecType.getRank())];
    } else if (isa<MGP::ContextType>(mlirType)) {
      return kStructStateContext;
    } else if (isa<IndexType>(mlirType)) {
      return kStructInt;
    } else if (auto scalarTy = dyn_cast<MO::ScalarType>(mlirType)) {
      // MO-dialect scalars captured by fusion struct fields (e.g. constants
      // from ConstantScalarOp); the SIMDType branch only covers KGEN types.
      DType dtype = scalarTy.getDtype();
      if (dtype.isBool())
        return kStructBool;
      assert(dtype.isInt() &&
             "only bool/integer MO::ScalarType expected in fusion captures; "
             "extend this to handle float/other dtypes if needed");
      return kStructInt;
    } else if (auto simdType = dyn_cast<KGEN::SIMDType>(mlirType)) {
      // FIXME(GEX-2867): SIMD types only appear for shape math on index/Int.
      if (simdType.getResolvedDType()->isBool())
        return kStructBool;
      return kStructInt;
    } else if (auto shapeResultType = dyn_cast<ShapeFuncResultType>(mlirType)) {
      MOGG::MojoValue rankParam = MOGG::MojoValue(shapeResultType.getRank());
      return MOGG::MojoValue(MOGG::kStructIndexList)[rankParam];
    } else if (isa<MGP::OpaqueType, MO::OpaqueType>(mlirType)) {
      return kComptimeOpaquePointer;
    } else {
      ASSERT_STREAM(false, << "Unsupported mlir type " << mlirType);
      llvm_unreachable("Unsupported mlir type");
    }
  }

  /// Convert an MO::OpaqueType to its qualified Mojo name without parameters,
  /// e.g. !mo.opaque<"root::SIMDPair", ...> to "root.SIMDPair". Apply
  /// getOpaqueTypeParameters() with MojoValue::operator[].
  static MojoValue mlirOpaqueTypeToMojoType(MO::OpaqueType opaqueType);

  /// E.g. !mo.opaque<"SIMDPair", {S0 = 4, S1 = 8}> returns [4, 8].
  static SmallVector<MojoValue>
  getOpaqueTypeParameters(MO::OpaqueType opaqueType);

  static MojoValue mlirDTypeToMojoType(DType dtype) {
    KGEN::KGENDType kgenDtype(dtype);
    return MojoValue(kStructDType).get(kgenDtype.getAsString(/*libForm*/ true));
  }

  static MojoValue mlirTensorTypeToMojoType(MO::TensorType tensorType) {
    MO::IOKindAttr ioKind = tensorType.getIOKindAttr();
    auto rank = tensorType.getRank();
    // Always use rank 1 for scalar tensors.
    if (rank == 0)
      rank = 1;
    auto dtype = mlirDTypeToMojoType(tensorType.getDtype()).text;

    // Whether the tensor is a mutable input; the io-kind branches below fold
    // it into the alias they pick.
    BoolAttr isMutable = tensorType.getIsMutableAttr();
    bool mutableInput = ioKind && ioKind.getValue() == MO::IOKind::Input &&
                        isMutable && isMutable.getValue();

    auto tensorTypeName = [&]() {
      if (!ioKind)
        return kStructManagedTensorSlice;
      if (ioKind.getValue() == MO::IOKind::Input)
        return mutableInput ? kComptimeMutableInputTensor
                            : kComptimeInputTensor;
      if (ioKind.getValue() == MO::IOKind::Output)
        return kComptimeOutputTensor;
      ASSERT_STREAM(false, << "Unsupported ioKind " << ioKind);
      return kStructManagedTensorSlice;
    }();
    return llvm::formatv("{}[dtype = {}, rank = {}, ...]", tensorTypeName,
                         dtype, rank)
        .str();
  }

  /// Dims as a `Coord` Mojo value: static dims as `Idx[N]` components,
  /// dynamic dims (kDynamic) as runtime `Int(-1)` components. Rank-0
  /// tensors become rank-1, as everywhere else.
  static MojoValue getAsCoordMojoValue(ArrayRef<int64_t> dims) {
    SmallVector<MojoValue> dimsAsMojoValues;
    for (auto dim : dims) {
      if (dim == mlir::ShapedType::kDynamic)
        dimsAsMojoValues.push_back(MojoValue(kStructInt)({}, {MojoValue(-1)}));
      else
        dimsAsMojoValues.push_back(getComptimeIdx(dim));
    }
    if (dimsAsMojoValues.empty())
      dimsAsMojoValues.push_back(getComptimeIdx(1));
    return MojoValue(kStructCoord)({}, dimsAsMojoValues);
  }

  MojoValue getShapeAsIndexListMojoValue(MOSH::ShapeAttr shapeAttr,
                                         ValueRange dynamicDims = {});

  /// Emit the shape as a `Coord`, encoding static dims as compile-time `Idx[N]`
  /// components and dynamic dims as runtime `Int(<dim>)` components.
  MojoValue getShapeAsCoordMojoValue(MOSH::ShapeAttr shapeAttr,
                                     ValueRange dynamicDims = {});

  SmallVector<MojoValue> getMojoValuesFor(ValueRange operands) {
    return llvm::map_to_vector(
        operands, [&](Value operand) { return getMojoValueFor(operand); });
  }

  /// Split e.g. "max.kernels.AddOp.execute" into ("max.kernels.AddOp",
  /// "execute"), stripping a leading "__init__." from the struct name.
  static std::pair<MojoValue, MojoValue>
  splitFunctionName(llvm::StringRef fullyResolvedName) {
    std::string fullName = fullyResolvedName.str();
    size_t lastDot = fullName.rfind('.');
    std::string structName = fullName.substr(0, lastDot);
    std::string fnName = fullName.substr(lastDot + 1);
    constexpr llvm::StringLiteral kInitPrefix = "__init__.";
    if (llvm::StringRef(structName).starts_with(kInitPrefix))
      structName.erase(0, kInitPrefix.size());
    return {MojoValue(std::move(structName)), MojoValue(std::move(fnName))};
  }

  /// A compile-time index, `Idx[value]`.
  static MojoValue getComptimeIdx(int64_t value) {
    return MojoValue(kAliasIdx)[MojoValue(value)];
  }

  static MojoValue getSIMDType(DType dtype, MojoValue width) {
    return MojoValue(kStructSIMD)[{mlirDTypeToMojoType(dtype), width}];
  }

  /// Looks up a parameter name in a function adaptor's result-parameter
  /// mappings.
  static std::optional<StringRef>
  getResultParamNameFor(const KernelDeclFunctionAdaptor &fnAdaptor,
                        StringRef paramName);

  /// Base function to infer all parameter values of a kernel.
  ErrorOr<MojoValue>
  inferDefaultKernelParamValue(Operation *callOp, StringRef paramName,
                               DictionaryAttr customParamDict = nullptr);

  /// Debug print the spec of `tensorValue` if enabled.
  void addDebugPrintTensorSpecs(const MojoValue &tensorValue,
                                const MojoValue &indicesValue, bool isInput,
                                unsigned operandIndex);

  /// Debug print the parameters of a fused lambda if enabled.
  void addDebugPrintLambdaParam(MOGG::BindLambdaKind lambdaKind,
                                const MojoValue &widthParam,
                                const MojoValue &alignmentParam,
                                const MojoValue &indicesValue);
};
} // namespace M::MOGG

#endif // GENERICML_INTERFACES_EMITMOJOSTATE_H
