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
// Central catalog of every Mojo-side identifier the GraphCompiler depends on
// and the matching C++ wrappers for high-traffic Mojo types.
//
// A rename or file-move on the Mojo side should be a one-line edit here.
//
// Prefixes communicate role:
//       kStruct*    Mojo struct names emitted into source.
//       kFn*        Mojo free-function names emitted into source.
//       kField*     Mojo field/method names accessed on values (`.foo`).
//       kEnum*      Mojo enum values (qualified or unqualified).
//       kTrait*     Mojo trait names used in conformance declarations.
//       kComptime*  Pre-rendered comptime type expressions.
//       kAlias*     Mojo alias names.
//       kDecorator* Mojo decorator strings (`@inline(.always)`, ...).
//       kParameter* Mojo parameter names (kwargs in emitted calls, generic
//                   parameter names, kernel parameter inference keys).
//       kPackage*   Mojo package names.
//       kAttr*      MLIR attribute keys carried on KGEN/MOGG ops.
//       kMogg*      MOGG-internal labels and `mogg.`-prefixed attribute keys.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOIDENTIFIERS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOIDENTIFIERS_H

#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <utility>

namespace M::MOGG {

//===----------------------------------------------------------------------===//
// MLIR attribute keys (carried on MOGG ops)
//===----------------------------------------------------------------------===//

/// Module-level map from simple struct names ("MyStruct") to fully-qualified
/// Mojo names ("foo.bar.MyStruct"). Written by `mogg-import-packages`, read by
/// `mogg-rewrite-opaque-types`; it travels with the module so callers needn't
/// re-walk packages.
constexpr llvm::StringLiteral kOpaqueTypeMappingAttrName =
    "mogg.opaque_type_mapping";

/// The `@register` decorators, owned by KGEN's `LITUtils.h`, which the Mojo
/// parser also uses. A kernel struct's method names are
/// `stringifyKernelFunctionKind`'s.
using KGEN::kFnRegister;
using KGEN::kFnRegisterInternal;

/// Indices into `getDecoratorLambdaArgument` for `@extensibility.register`.
constexpr size_t kRegisterDecoratorArgName = 0;
constexpr size_t kRegisterDecoratorArgType = 1;
constexpr size_t kRegisterDecoratorArgApi = 2;
constexpr size_t kRegisterDecoratorArgArch = 3;
constexpr size_t kRegisterDecoratorArgModel = 4;

constexpr llvm::StringLiteral kMoggTargetParameterName = "target";

/// Lowered specially.
constexpr llvm::StringLiteral kMoggOutputRankParameterName = "output_rank";

/// Per-stub-argument metadata: `{dtype}` for a trait tensor, else "". The
/// layout `L` of `_MoggTensorBufferCapture[dtype, L]` is inferred from the
/// argument, since metadata cannot encode dynamic extents.
constexpr llvm::StringLiteral kTraitStubBufferCaptureAttrName =
    "mogg.trait_stub_buffer_capture";

/// Generic trait bounds for shape-function stubs with generic `TileTensorable`
/// parameters.
constexpr llvm::StringLiteral kTraitStubArgBoundsAttrName =
    "mogg.trait_stub_arg_bounds";

/// On accumulation-style fused kernels.
constexpr llvm::StringLiteral kMoggHasEpilogueFusionParameterName =
    "has_epilogue_fusion";

constexpr llvm::StringLiteral kMoggTraceNameParameterName = "_trace_name";

/// Package roots used to recognise `@register_internal` and friends without a
/// symbol-table lookup into dependency packages.
constexpr llvm::StringLiteral kPackageMax = "max";
using KGEN::kPackageExtensibility;
using KGEN::kPackageStd;
constexpr llvm::StringLiteral kPackageManagedTensorSlice =
    "managed_tensor_slice";
constexpr llvm::StringLiteral kPackageStableTensor = "stable_tensor";

/// Last segments of `SymbolRefAttr` paths, for symbol-tree matching.
using KGEN::kLeafDeviceContext;
using KGEN::kLeafManagedTensorSlice;
constexpr llvm::StringLiteral kLeafVariadicTensors = "VariadicTensors";
constexpr llvm::StringLiteral kLeafVariadicPack = "VariadicPack";
constexpr llvm::StringLiteral kLeafTensorTuple = "TensorTuple";
constexpr llvm::StringLiteral kLeafFusedInputVariadicTensors =
    "_FusedInputVariadicTensors";
constexpr llvm::StringLiteral kLeafFusedOutputVariadicTensors =
    "_FusedOutputVariadicTensors";
constexpr llvm::StringLiteral kLeafStableTensor = "StableTensor";
constexpr llvm::StringLiteral kLeafStableVariadicTensors =
    "StableVariadicTensors";
constexpr llvm::StringLiteral kLeafSIMD = "SIMD";
constexpr llvm::StringLiteral kLeafIndexList = "IndexList";
constexpr llvm::StringLiteral kLeafTileTensor = "TileTensor";
constexpr llvm::StringLiteral kLeafDeviceContextArray = "DeviceContextArray";
constexpr llvm::StringLiteral kLeafDeviceGraphBuilder = "DeviceGraphBuilder";

/// The `ReduceOp` monoid trait (`algorithm.reduce_op.ReduceOp`) and its
/// `width` alias, which names the struct parameter that is the SIMD width.
constexpr llvm::StringLiteral kReduceOpTraitRoot = "algorithm";
constexpr llvm::StringLiteral kReduceOpTraitLeaf = "ReduceOp";
constexpr llvm::StringLiteral kReduceOpWidthAlias = "width";

/// Mojo module names the MojoRegistry builds its registry from. These are
/// required: every build has them, and one that fails to resolve leaves no
/// registry to compile a graph against, so it is fatal.
///
/// Order is not precedence. A registration these packages share is an error
/// rather than an override (see `emitDecls` in ImportPackages.cpp); only a
/// user-supplied package can override a built-in one.
constexpr llvm::StringLiteral kMoggRequiredKernelModuleNames[] = {
    "builtin_kernels",
    "builtin_primitives",
};

/// Mojo module names holding the registrations for the closed-source kernels
/// under `//Kernels/lib`. These are optional, which is why they are not in the
/// list above: an open-source checkout has neither the kernels nor their
/// registrations, and so does any install shipping only the open-source kernel
/// packages. Failing to resolve one costs the ops it registers rather than the
/// whole registry.
constexpr llvm::StringLiteral kMoggOptionalKernelModuleNames[] = {
    "graph_compiler",
};

/// Prefixes an async error with its region name. Called from
/// `AsyncErrorOp::emitMojo` besides the op's own primitive, so it can't come
/// from `getOperationName()`.
constexpr llvm::StringLiteral kMoggPrimitiveFormatRegionError =
    "mogg.format_region_error";

/// `@register_internal` name of the primitive that builds a `StableTensor`
/// for a stable tensor init. Not an op name: `mogg.tensor.__init__` selects
/// it by its result type, `mogg.tensor.create` by its `$stable` attr.
constexpr llvm::StringLiteral kMoggPrimitiveStableTensorInit =
    "mogg.tensor.__init__.stable";

//===----------------------------------------------------------------------===//
// Mojo type symbol names (as stored in MLIR by the Mojo compiler)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kMojoDpsTensorTypeName =
    "extensibility::managed_tensor_slice::ManagedTensorSlice";

constexpr llvm::StringLiteral kMojoInternalDpsSimdTypeName = "std::SIMD";
constexpr llvm::StringLiteral kMojoInternalDpsIntTypeName = "std::Int";
constexpr llvm::StringLiteral kMojoInternalDpsUintTypeName = "std::UInt";
constexpr llvm::StringLiteral kMojoInternalDpsBoolTypeName = "std::Bool";
constexpr llvm::StringLiteral kMojoInternalDpsIndexListTypeName =
    "std::IndexList";

/// In `buildOpaqueSymbol`'s dotted form, for `isOpaqueWithTypeName`.
constexpr llvm::StringLiteral kMojoInternalDpsIndexListTypeQualifiedName =
    "std.utils.index.IndexList";

/// The fused variants carry per-element fusion structs in a heterogeneous
/// Tuple.
constexpr llvm::StringLiteral kMojoVariadicTensorsName =
    "extensibility::managed_tensor_slice::VariadicTensors";
constexpr llvm::StringLiteral kMojoFusedInputVariadicTensorsName =
    "extensibility::managed_tensor_slice::_FusedInputVariadicTensors";
constexpr llvm::StringLiteral kMojoFusedOutputVariadicTensorsName =
    "extensibility::managed_tensor_slice::_FusedOutputVariadicTensors";

constexpr llvm::StringLiteral kMojoExtensibilityApiDeviceContextTypeName =
    "max::DeviceContext";

/// Used by multi-device custom ops.
constexpr llvm::StringLiteral kMojoExtensibilityApiDeviceContextArrayTypeName =
    "max::DeviceContextArray";

/// Inside the Graph Compiler these types show up as:
///   - StaticString -> std::StringSpan
///   - String -> std::String
///   - StringSpan -> std::Origin
constexpr llvm::StringLiteral kMojoInternalStaticStringTypeName =
    "std::StringSpan";
constexpr llvm::StringLiteral kMojoStaticStringTypeName = "std::StaticString";

/// In the emit-mojo path, DType parameters appear as LIT struct types with
/// type name "std::DType" rather than KGEN::DTypeType.
constexpr llvm::StringLiteral kMojoDtypeTypeName = "std::DType";

constexpr llvm::StringLiteral kMojoStringTypeName = "std::String";
constexpr llvm::StringLiteral kMojoInternalStringSliceTypeName = "std::Origin";

//===----------------------------------------------------------------------===//
// Mojo struct names (dotted, as emitted into Mojo source)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kStructIO = "extensibility.IO";
constexpr llvm::StringLiteral kStructIndexList = "std.utils.IndexList";
constexpr llvm::StringLiteral kStructPointer = "Pointer";
/// `UntrackedOrigin[mut=...]`, for where `mut` must stay generic (unlike
/// `kComptimeMutUntrackedOrigin`).
constexpr llvm::StringLiteral kStructUntrackedOrigin = "UntrackedOrigin";
constexpr llvm::StringLiteral kStructOptional = "Optional";
constexpr llvm::StringLiteral kStructString = "String";
constexpr llvm::StringLiteral kStructDeviceContext =
    "max.gpu.host.DeviceContext";
constexpr llvm::StringLiteral kStructDeviceContextArray =
    "max.gpu.host.DeviceContextArray";
constexpr llvm::StringLiteral kStructDeviceGraph = "max.gpu.host.DeviceGraph";
constexpr llvm::StringLiteral kStructDeviceGraphBuilder =
    "max.gpu.host.DeviceGraphBuilder";
/// `DeviceGraphInputKind`: how an `OwnedTensor` participates as a device graph
/// input. The `data` member is the Mojo default and is never spelled out.
constexpr llvm::StringLiteral kStructDeviceGraphInputKind =
    "max.gpu.host.DeviceGraphInputKind";
constexpr llvm::StringLiteral kInputKindMutable = "mutable";
constexpr llvm::StringLiteral kInputKindHost = "host";
constexpr llvm::StringLiteral kInputKindShapeOnly = "shape_only";
/// `OwnedTensor.with_kind[kind]()`: the same tensor under another input kind.
constexpr llvm::StringLiteral kFnWithKind = "with_kind";
constexpr llvm::StringLiteral kStructDeviceGraphNode =
    "max.gpu.host.DeviceGraphNode";
constexpr llvm::StringLiteral kStructDeviceGraphCache =
    "max.gpu.host.DeviceGraphCache";
constexpr llvm::StringLiteral kStructStableAddr = "max.gpu.host.StableAddr";
constexpr llvm::StringLiteral kStructManagedTensorSlice =
    "extensibility.ManagedTensorSlice";
constexpr llvm::StringLiteral kStructStableTensor =
    "extensibility.StableTensor";
constexpr llvm::StringLiteral kStructInt = "Int";
constexpr llvm::StringLiteral kStructSIMDLength = "SIMDLength";
constexpr llvm::StringLiteral kStructUInt = "UInt";
constexpr llvm::StringLiteral kStructBool = "Bool";
constexpr llvm::StringLiteral kStructAddressSpace = "std.memory.AddressSpace";
constexpr llvm::StringLiteral kStructDType = "DType";
constexpr llvm::StringLiteral kStructSIMD = "SIMD";
// Qualified because the generated module imports packages qualified
// (`import layout`), so bare names don't bind.
constexpr llvm::StringLiteral kStructTileTensor = "layout.TileTensor";
constexpr llvm::StringLiteral kStructTensorLayout = "layout.TensorLayout";
constexpr llvm::StringLiteral kStructTileCopier = "layout.tile_io.TileCopier";
constexpr llvm::StringLiteral kStructTileLayout = "layout.tile_layout.Layout";
constexpr llvm::StringLiteral kStructMoggOutputStoreFusionTensor =
    "_MoggOutputStoreFusionTensor";
constexpr llvm::StringLiteral kStructStateContext =
    "builtin_primitives.primitives.StateContext";
constexpr llvm::StringLiteral kStructStaticTuple = "std.utils.StaticTuple";
constexpr llvm::StringLiteral kStructArray = "std.collections.Array";

inline MojoValue makeMojoArrayType(llvm::StringRef elementType, size_t size) {
  return MojoValue(
      llvm::formatv("{}[{}, {}]", kStructArray, elementType, size).str());
}

constexpr llvm::StringLiteral kStructScalar = "Scalar";
constexpr llvm::StringLiteral kStructVariadic = "Variadic";
constexpr llvm::StringLiteral kStructParameterList = "ParameterList";
constexpr llvm::StringLiteral kStructStaticTensorSpec =
    "managed_tensor_slice.StaticTensorSpec";
constexpr llvm::StringLiteral kStructStaticTensorSpecList =
    "extensibility.StaticTensorSpecList";
constexpr llvm::StringLiteral kStructVariadicTensors =
    "extensibility.VariadicTensors";
constexpr llvm::StringLiteral kStructStableVariadicTensors =
    "extensibility.StableVariadicTensors";
constexpr llvm::StringLiteral kStructTensorTuple = "extensibility.TensorTuple";
constexpr llvm::StringLiteral kStructStaticString = "StaticString";
constexpr llvm::StringLiteral kStructMsFusionPack =
    "managed_tensor_slice._FusionPack";
constexpr llvm::StringLiteral kStructMsFusedInputVariadicTensors =
    "managed_tensor_slice._FusedInputVariadicTensors";
constexpr llvm::StringLiteral kStructMsFusedOutputVariadicTensors =
    "managed_tensor_slice._FusedOutputVariadicTensors";
constexpr llvm::StringLiteral kStructTrace = "max.runtime.tracing.Trace";
constexpr llvm::StringLiteral kStructAnyAsyncValueRef =
    "max.runtime.async_value.AnyAsyncValueRef";
/// Re-exported through `max.runtime.tracing`, avoiding an `_tracing` import.
constexpr llvm::StringLiteral kStructColor = "max.runtime.tracing.Color";
constexpr llvm::StringLiteral kFnTraceStart = "start";
constexpr llvm::StringLiteral kFnTraceEnd = "end";
constexpr llvm::StringLiteral kParameterColor = "color";
constexpr llvm::StringLiteral kStructIntTuple = "layout.IntTuple";
constexpr llvm::StringLiteral kStructCoord = "layout.Coord";
constexpr llvm::StringLiteral kAliasIdx = "layout.Idx";
constexpr llvm::StringLiteral kFnRowMajor = "layout.row_major";

//===----------------------------------------------------------------------===//
// Mojo enum values
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kEnumAddressSpaceGeneric = "GENERIC";
constexpr llvm::StringLiteral kEnumIoInput = "extensibility.IO.Input";
constexpr llvm::StringLiteral kEnumTraceLevelOp =
    "max.runtime.tracing.TraceLevel.OP";
constexpr llvm::StringLiteral kEnumDTypeInt8 = "int8";
/// The index-sized dtype of dynamic layout `Coord` elements.
constexpr llvm::StringLiteral kEnumDTypeInt = "int";

//===----------------------------------------------------------------------===//
// Mojo free-function names (emitted into Mojo source)
//===----------------------------------------------------------------------===//

// Qualified because std.memory is imported qualified.
constexpr llvm::StringLiteral kFnStackAllocation =
    "std.memory.stack_allocation";
/// For special float values with no Mojo literal spelling.
constexpr llvm::StringLiteral kFnNumericsNan = "std.utils.numerics.nan";
constexpr llvm::StringLiteral kFnNumericsInf = "std.utils.numerics.inf";
constexpr llvm::StringLiteral kFnSimdLoadFromManagedTensorSlice =
    "managed_tensor_slice.simd_load_from_managed_tensor_slice";
constexpr llvm::StringLiteral kFnSimdLoadFromTensorPointer =
    "managed_tensor_slice.simd_load_from_tensor_pointer";
constexpr llvm::StringLiteral kFnSimdStoreIntoManagedTensorSlice =
    "managed_tensor_slice.simd_store_into_managed_tensor_slice";
constexpr llvm::StringLiteral kFnSimdStoreIntoTensorPointer =
    "managed_tensor_slice.simd_store_into_tensor_pointer";
constexpr llvm::StringLiteral kFnUnpackTensorSpec =
    "builtin_primitives.primitives.unpack_tensor_spec";
constexpr llvm::StringLiteral kFnGetSimdWidthForDtypes =
    "builtin_primitives.primitives.get_simd_width_for_dtypes";
constexpr llvm::StringLiteral kFnUnpackBufferRef =
    "builtin_primitives.primitives.unpack_buffer_ref";
constexpr llvm::StringLiteral kFnUnpackDeviceCtx =
    "builtin_primitives.primitives.unpack_device_ctx";
constexpr llvm::StringLiteral kFnUnpackStateCtx =
    "builtin_primitives.primitives.unpack_state_ctx";
constexpr llvm::StringLiteral kFnUnpackTensor =
    "builtin_primitives.primitives.unpack_tensor";
constexpr llvm::StringLiteral kFnAllZeros =
    "builtin_primitives.primitives.all_zeros";
constexpr llvm::StringLiteral kFnMoggAsyncPackValue =
    "builtin_primitives.primitives.mogg_async_pack_value";
constexpr llvm::StringLiteral kFnMoggAsyncPackOwnedTensorValue =
    "builtin_primitives.primitives.mogg_async_pack_owned_tensor_value";
constexpr llvm::StringLiteral kFnMoggAsyncReadyValue =
    "builtin_primitives.primitives.mogg_async_ready_value";
constexpr llvm::StringLiteral kFnMoggAsyncErrorValue =
    "builtin_primitives.primitives.mogg_async_error_value";
constexpr llvm::StringLiteral kFnMoggAssignAsyncValue =
    "builtin_primitives.primitives.mogg_assign_async_value";
constexpr llvm::StringLiteral kFnMsSimdLoadFromTensorPointer =
    "managed_tensor_slice.simd_load_from_tensor_pointer";
constexpr llvm::StringLiteral kFnMsSimdStoreIntoTensorPointer =
    "managed_tensor_slice.simd_store_into_tensor_pointer";
/// Returns a tile kernel's `(rows, cols)` tile shape.
constexpr llvm::StringLiteral kFnGetKernelTileShape =
    "managed_tensor_slice.get_kernel_tile_shape";
constexpr llvm::StringLiteral kFnToIndexList = "layout.to_index_list";
/// Fully qualified because `std.utils` does not re-export `coord`.
constexpr llvm::StringLiteral kFnCoordToIndexList =
    "std.utils.coord.coord_to_index_list";
constexpr llvm::StringLiteral kStructRowwiseRow = "algorithm.rowwise.Row";
constexpr llvm::StringLiteral kStructRowwiseContextParams =
    "algorithm.rowwise_types.ContextParams";
constexpr llvm::StringLiteral kStructRowwiseContext =
    "algorithm.rowwise_types.Context";
constexpr llvm::StringLiteral kStructRowwiseRowCoord =
    "algorithm.rowwise_types.RowCoord";
/// Package-private, so qualified with `_asyncrt`.
constexpr llvm::StringLiteral kFnCreateTask =
    "std.runtime._asyncrt.create_task";

/// Affinity-aware variant (`desired_worker_id=<id>`), for GPU-device launches.
constexpr llvm::StringLiteral kFnCreateTaskWithAffinity =
    "std.runtime._asyncrt._create_task";
constexpr llvm::StringLiteral kFnTaskIdForDevice =
    "max.runtime.asyncrt.task_id_for_device";
constexpr llvm::StringLiteral kFnGetSafeTaskId =
    "max.runtime.tracing.get_safe_task_id";

constexpr llvm::StringLiteral kFnMax = "std.math.max";
constexpr llvm::StringLiteral kFnPrint = "print";

// Field / method names accessed via `.foo` live on the wrapper classes below.

//===----------------------------------------------------------------------===//
// Trait names (used in struct conformance declarations)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kTraitInputFusion =
    "managed_tensor_slice.InputFusion";
constexpr llvm::StringLiteral kTraitOutputFusion =
    "managed_tensor_slice.OutputFusion";
constexpr llvm::StringLiteral kTraitComputeOutputFusion =
    "managed_tensor_slice.ComputeOutputFusion";
constexpr llvm::StringLiteral kTraitComputeOutputFusionTile =
    "managed_tensor_slice.ComputeOutputFusionTile";
constexpr llvm::StringLiteral kTraitOutputFusionTile =
    "managed_tensor_slice.OutputFusionTile";
constexpr llvm::StringLiteral kTraitElementwiseFusion =
    "managed_tensor_slice.ElementwiseFusion";
constexpr llvm::StringLiteral kTraitElementwiseFusionTile =
    "managed_tensor_slice.ElementwiseFusionTile";

//===----------------------------------------------------------------------===//
// Decorator strings
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kDecoratorAlwaysInline = "@inline(.always)";
constexpr llvm::StringLiteral kDecoratorNoInline = "@inline(.never)";

//===----------------------------------------------------------------------===//
// Pre-rendered comptime type expressions
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kComptimeOpaquePointer =
    "OpaquePointer[MutAnyOrigin]";
constexpr llvm::StringLiteral kComptimeInputTensor =
    "extensibility.InputTensor";
constexpr llvm::StringLiteral kComptimeOutputTensor =
    "extensibility.OutputTensor";
constexpr llvm::StringLiteral kComptimeMutableInputTensor =
    "managed_tensor_slice._MutableInputTensor";
constexpr llvm::StringLiteral kComptimeStableInputTensor =
    "extensibility.StableInputTensor";
constexpr llvm::StringLiteral kComptimeStableMutableInputTensor =
    "stable_tensor._StableMutableInputTensor";
constexpr llvm::StringLiteral kComptimeUInt8 = "UInt8";
constexpr llvm::StringLiteral kComptimeMutAnyOrigin = "MutAnyOrigin";
constexpr llvm::StringLiteral kComptimeMutUntrackedOrigin =
    "MutUntrackedOrigin";
constexpr llvm::StringLiteral kComptimeNone = "None";
constexpr llvm::StringLiteral kComptimeOptionalErrorEmpty = "Optional[Error]()";
constexpr llvm::StringLiteral kComptimeMutByteBuffer =
    "builtin_primitives.primitives.MutByteBuffer";
constexpr llvm::StringLiteral kStructOwnedByteBuffer =
    "builtin_primitives.primitives.OwnedByteBuffer";
constexpr llvm::StringLiteral kStructOwnedTensor =
    "builtin_primitives.primitives.OwnedTensor";
/// Builds an `AsyncValue` over an `OwnedByteBuffer` / `OwnedTensor`'s memory,
/// to register a device-graph output.
constexpr llvm::StringLiteral kFnAsyncPack = "async_pack";
constexpr llvm::StringLiteral kComptimeDTypeNoneType = "NoneType";

//===----------------------------------------------------------------------===//
// Parameter names (kernel param inference keys, kwargs in emitted calls)
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kParameterMut = "mut";
constexpr llvm::StringLiteral kParameterKind = "kind";
constexpr llvm::StringLiteral kParameterInput = "input";
constexpr llvm::StringLiteral kParameterDType = "dtype";
constexpr llvm::StringLiteral kParameterRank = "rank";
constexpr llvm::StringLiteral kParameterWidth = "width";
constexpr llvm::StringLiteral kParameterIOSpec = "io_spec";
constexpr llvm::StringLiteral kParameterStaticSpec = "static_spec";
constexpr llvm::StringLiteral kParameterSize = "size";
constexpr llvm::StringLiteral kParameterLength = "length";
constexpr llvm::StringLiteral kParameterStaticSpecs = "static_specs";
constexpr llvm::StringLiteral kParameterSimdWidth = "simd_width";
constexpr llvm::StringLiteral kParameterElementAlignment = "element_alignment";
/// The source tensor's static alignment, which a `TileTensor` does not carry.
constexpr llvm::StringLiteral kParameterTensorAlignment = "tensor_alignment";
constexpr llvm::StringLiteral kParameterElementSize = "element_size";
constexpr llvm::StringLiteral kParameterOutputStaticShape =
    "output_static_shape";
constexpr llvm::StringLiteral kParameterStaticStarts = "static_starts";
constexpr llvm::StringLiteral kParameterStaticSteps = "static_steps";
constexpr llvm::StringLiteral kParameterStaticPermutations =
    "static_permutations";
constexpr llvm::StringLiteral kParameterLayoutType = "LayoutType";
constexpr llvm::StringLiteral kParameterOrigin = "origin";
constexpr llvm::StringLiteral kParameterAddressSpace = "address_space";
constexpr llvm::StringLiteral kParameterLinearIdxType = "linear_idx_type";
constexpr llvm::StringLiteral kParameterStorage = "Storage";
constexpr llvm::StringLiteral kParameterTaskId = "task_id";
constexpr llvm::StringLiteral kParameterOutput = "output";
constexpr llvm::StringLiteral kParameterElementType = "element_type";
constexpr llvm::StringLiteral kParameterTo = "to";
/// Keep in sync with `foreach_fusion` in
/// `max/kernels/src/graph_compiler/builtin_primitives/primitives.mojo`.
constexpr llvm::StringLiteral kParameterForeachTensor = "tensor";
constexpr llvm::StringLiteral kParameterForeachElem = "elem";
constexpr llvm::StringLiteral kParameterForeachCtx = "ctx";
constexpr llvm::StringLiteral kMoggForeachTileRegistration =
    "mogg.call.foreach.tile";
constexpr llvm::StringLiteral kParameterSelf = "self";
constexpr llvm::StringLiteral kParameterIdx = "idx";
constexpr llvm::StringLiteral kParameterVal = "val";
constexpr llvm::StringLiteral kParameterDst = "dst";
constexpr llvm::StringLiteral kParameterTileCoords = "tile_coords";
constexpr llvm::StringLiteral kParameterCopier = "copier";
/// Must differ from `kParameterCopier`, or the emitted method redefines
/// `copier`.
constexpr llvm::StringLiteral kParameterCopierType = "Copier";
constexpr llvm::StringLiteral kParameterStoreCopier = "store_copier";
constexpr llvm::StringLiteral kParameterStoreCopierType = "StoreCopier";

//===----------------------------------------------------------------------===//
// Stdlib package names
//===----------------------------------------------------------------------===//

/// Imported into every generated Mojo module. TODO(GEX-2488): Derive from the
/// Modular config rather than hardcoding.
constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral>
    kPackagesToImport[] = {
        {"builtin_primitives.primitives", ""},
        {"builtin_kernels", "builtin_kernels"},
        {"extensibility", ""},
        {"extensibility.managed_tensor_slice", "managed_tensor_slice"},
        {"extensibility.stable_tensor", "stable_tensor"},
        {"max.gpu.host", ""},
        {"std.utils", ""},
        {"std.utils.coord", ""},
        {"std.utils.numerics", ""},
        {"std.math", ""},
        {"std.runtime", ""},
        {"std.runtime._asyncrt", ""},
        {"std.memory", ""},
        {"std.collections", ""},
        {"max.runtime.tracing", ""},
        {"max.runtime.async_value", ""},
        {"max.runtime.asyncrt", ""},
        {"layout", ""},
        // For the Row API that `mogg.call.rowwise` prints.
        {"algorithm.rowwise", ""},
        {"algorithm.rowwise_types", ""},
        // For `layout.tile_io.TileCopier`.
        {"layout.tile_io", ""},
};

/// TODO(GEX-2488): Find another way to detect that a type is an `IOSpec`
/// without relying on package / type name.
inline bool isMojoIOSpecTypeName(llvm::ArrayRef<mlir::FlatSymbolRefAttr> refs) {
  constexpr llvm::StringLiteral mojoTypeName = "IOSpec";
  if (refs.empty())
    return false;
  return refs.back().getValue() == mojoTypeName &&
         refs.front().getValue() == kPackageExtensibility;
}

//===----------------------------------------------------------------------===//
// IOSpec encoding constants and Mojo enum names
//===----------------------------------------------------------------------===//

constexpr bool kIOSpecImmutable = false;
constexpr bool kIOSpecMutable = true;

constexpr int64_t kIOSpecIOOutput = 0;
constexpr int64_t kIOSpecIOInput = 1;
constexpr int64_t kIOSpecIOFusedInput = 2;
constexpr int64_t kIOSpecIOFusedOutput = 3;
constexpr int64_t kIOSpecIOFusedComputeOutput = 31;
constexpr int64_t kIOSpecIOFusedComputeOutputTile = 32;
constexpr int64_t kIOSpecIOFusedOutputTile = 33;
constexpr int64_t kIOSpecIOUnknown = -1;

constexpr llvm::StringLiteral kInputTensor = "InputTensor";
constexpr llvm::StringLiteral kOutputTensor = "OutputTensor";
constexpr llvm::StringLiteral kMutableInputTensor = "MutableInputTensor";
constexpr llvm::StringLiteral kFusedInputTensor = "FusedInputTensor";
constexpr llvm::StringLiteral kFusedOutputTensor = "FusedOutputTensor";
constexpr llvm::StringLiteral kFusedComputeOutputTensor =
    "FusedComputeOutputTensor";
constexpr llvm::StringLiteral kFusedComputeOutputTileTensor =
    "FusedComputeOutputTileTensor";
constexpr llvm::StringLiteral kFusedOutputTileTensor = "FusedOutputTileTensor";

/// Mojo `IO` member names, as they appear in `KGEN::SugarKind::MemberAlias`
/// attributes of a stored `IOSpec` parameter.
constexpr llvm::StringLiteral kIoInput = "Input";
constexpr llvm::StringLiteral kIoOutput = "Output";
constexpr llvm::StringLiteral kIoFusedInput = "FusedInput";
constexpr llvm::StringLiteral kIoFusedOutput = "FusedOutput";
constexpr llvm::StringLiteral kIoFusedComputeOutput = "FusedComputeOutput";
constexpr llvm::StringLiteral kIoFusedComputeOutputTile =
    "_FusedComputeOutputTile";
constexpr llvm::StringLiteral kIoFusedOutputTile = "_FusedOutputTile";

//===----------------------------------------------------------------------===//
// Wrapper classes for high-traffic Mojo types
//===----------------------------------------------------------------------===//
//
// Thin typed accessors over a `MojoValue` (e.g. `mts.unsafePtr()` instead of
// `value.get("unsafe_ptr")()`), so a Mojo rename is a one-line edit.
//
//===----------------------------------------------------------------------===//

class ManagedTensorSliceRef {
public:
  explicit ManagedTensorSliceRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kUnsafePtr = "unsafe_ptr";
  static constexpr llvm::StringLiteral kShape = "shape";
  static constexpr llvm::StringLiteral kStrides = "strides";
  static constexpr llvm::StringLiteral kToTileTensor = "to_tile_tensor";
  static constexpr llvm::StringLiteral kStaticSpec = "static_spec";
  static constexpr llvm::StringLiteral kStaticShapeTuple =
      "_static_shape_tuple";
  static constexpr llvm::StringLiteral kStaticStridesTuple =
      "_static_strides_tuple";
  static constexpr llvm::StringLiteral kAlignment = "alignment";
  /// Storage pointer of VariadicTensors elements.
  static constexpr llvm::StringLiteral kPtr = "_ptr";
  static constexpr llvm::StringLiteral kIoSpec = "io_spec";
  /// Bind methods for fusion lambdas.
  static constexpr llvm::StringLiteral kBindToFusedInput =
      "_bind_to_fused_input";
  static constexpr llvm::StringLiteral kBindToFusedOutput =
      "_bind_to_fused_output";
  static constexpr llvm::StringLiteral kBindToFusedComputeOutput =
      "_bind_to_fused_compute_output";
  static constexpr llvm::StringLiteral kBindToFusedComputeOutputTile =
      "_bind_to_fused_compute_output_tile";
  static constexpr llvm::StringLiteral kBindToFusedOutputTile =
      "_bind_to_fused_output_tile";

  /// Fusion functor fields and the matching type parameters.
  static constexpr llvm::StringLiteral kInFusion = "in_fusion";
  static constexpr llvm::StringLiteral kOutFusion = "out_fusion";
  static constexpr llvm::StringLiteral kInFusionParam = "InFusion";
  static constexpr llvm::StringLiteral kOutFusionParam = "OutFusion";

  MojoValue staticSpec() const { return value.get(kStaticSpec); }
  MojoValue staticShapeTuple() const { return value.get(kStaticShapeTuple); }
  MojoValue staticStridesTuple() const {
    return value.get(kStaticStridesTuple);
  }
  MojoValue alignment() const { return value.get(kAlignment); }
  MojoValue ioSpec() const { return value.get(kIoSpec); }
  MojoValue rawPtr() const { return value.get(kPtr); }
  MojoValue inFusion() const { return value.get(kInFusion); }
  MojoValue outFusion() const { return value.get(kOutFusion); }
  MojoValue inFusionType() const {
    return MojoValue("type_of")({}, {value}).get(kInFusionParam);
  }
  MojoValue outFusionType() const {
    return MojoValue("type_of")({}, {value}).get(kOutFusionParam);
  }

  MojoValue unsafePtr() const { return value.get(kUnsafePtr)(); }
  MojoValue shape() const { return value.get(kShape)(); }
  MojoValue strides() const { return value.get(kStrides)(); }
  MojoValue toTileTensor() const { return value.get(kToTileTensor)(); }

  /// Return the uncalled method handle.
  MojoValue bindToFusedInput() const { return value.get(kBindToFusedInput); }
  MojoValue bindToFusedOutput() const { return value.get(kBindToFusedOutput); }
  MojoValue bindToFusedComputeOutput() const {
    return value.get(kBindToFusedComputeOutput);
  }

  MojoValue value;
};

/// Wrapper around a Mojo `StableTensor` value: a tensor view whose data
/// pointer lives in a stable address slot.
class StableTensorRef {
public:
  explicit StableTensorRef(MojoValue v) : value(std::move(v)) {}

  /// Mojo field / method symbol names on `StableTensor`.
  static constexpr llvm::StringLiteral kAddr = "_addr";
  static constexpr llvm::StringLiteral kRuntimeLayout = "_runtime_layout";
  static constexpr llvm::StringLiteral kShape = "shape";
  static constexpr llvm::StringLiteral kStaticSpec = "static_spec";
  static constexpr llvm::StringLiteral kIoSpec = "io_spec";
  /// `StableAddr.ptr`: the slot address itself, a pointer to the data pointer.
  static constexpr llvm::StringLiteral kSlotPtr = "ptr";

  /// Field accessors (return the raw field value, not a call).
  MojoValue addr() const { return value.get(kAddr); }
  MojoValue slotPtr() const { return addr().get(kSlotPtr); }
  MojoValue runtimeLayout() const { return value.get(kRuntimeLayout); }
  MojoValue staticSpec() const { return value.get(kStaticSpec); }
  MojoValue ioSpec() const { return value.get(kIoSpec); }

  /// Method calls (parens included).
  MojoValue shape() const { return value.get(kShape)(); }

  MojoValue value;
};

class StaticTensorSpecRef {
public:
  explicit StaticTensorSpecRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kStaticLayout = "static_layout";
  static constexpr llvm::StringLiteral kAlignment = "alignment";
  static constexpr llvm::StringLiteral kRank = "rank";
  static constexpr llvm::StringLiteral kGetInternals = "get_internals";
  /// Comptime members of `StaticTensorSpec`, not of `get_internals()`.
  static constexpr llvm::StringLiteral kShapeTuple = "shape_tuple";
  static constexpr llvm::StringLiteral kStridesTuple = "strides_tuple";

  MojoValue staticLayout() const { return value.get(kStaticLayout); }
  MojoValue alignment() const { return value.get(kAlignment); }
  MojoValue rank() const { return value.get(kRank); }
  MojoValue shapeTuple() const { return value.get(kShapeTuple); }
  MojoValue stridesTuple() const { return value.get(kStridesTuple); }

  /// Carries the runtime `alignment` / `address_space`.
  MojoValue getInternals() const { return value.get(kGetInternals)(); }

  MojoValue value;
};

/// What `!mgp.buffer` lowers to: a `MutByteBuffer` view plus an
/// `AnyAsyncValueRef` storage handle. Unwrap `.view` before passing to a
/// kernel stub or register-passable container.
class OwnedByteBufferRef {
public:
  explicit OwnedByteBufferRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kView = "view";
  static constexpr llvm::StringLiteral kUnsafePtr = "unsafe_ptr";

  MojoValue view() const { return value.get(kView); }

  /// `Pointer[Int8, MutAnyOrigin]`.
  MojoValue unsafePtr() const { return value.get(kUnsafePtr)(); }

  MojoValue value;
};

/// What `!mgp.tensor` lowers to: a `DynamicTensor` view plus a storage
/// handle. Unwrap `.tensor` before passing to a kernel stub.
class OwnedTensorRef {
public:
  explicit OwnedTensorRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kTensor = "tensor";

  MojoValue tensor() const { return value.get(kTensor); }

  MojoValue value;
};

/// The recorder handed to the build closure of `mgp.device_graph.create`.
class DeviceGraphBuilderRef {
public:
  explicit DeviceGraphBuilderRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kAddInput = "add_input";
  static constexpr llvm::StringLiteral kAddOutput = "add_output";
  static constexpr llvm::StringLiteral kRegion = "region";
  static constexpr llvm::StringLiteral kContext = "context";
  static constexpr llvm::StringLiteral kRecordingContext = "recording_context";
  static constexpr llvm::StringLiteral kDependencies = "dependencies";
  static constexpr llvm::StringLiteral kGetStable = "get_stable";
  static constexpr llvm::StringLiteral kPlaceStable = "place_stable";

  /// A graph-owned stable location recorded in place of the caller's buffer,
  /// so replay does not depend on the caller's address.
  MojoValue addInput(MojoValue input) const {
    return value.get(kAddInput)(/*parameters=*/{}, {std::move(input)});
  }

  /// `builder.get_stable(index)`.
  MojoValue getStable(MojoValue index) const {
    return value.get(kGetStable)(/*parameters=*/{}, {std::move(index)});
  }

  /// `builder.place_stable(index, ptr)`.
  MojoValue placeStable(MojoValue index, MojoValue ptr) const {
    return value.get(kPlaceStable)(/*parameters=*/{},
                                   {std::move(index), std::move(ptr)});
  }

  /// `builder.add_output`.
  MojoValue addOutput() const { return value.get(kAddOutput); }

  /// Chains every node `work` records and returns the last as a single
  /// dependency.
  MojoValue region(MojoValue work, MojoValue dependencies) const {
    return value.get(kRegion)(
        /*parameters=*/{},
        {std::move(work),
         MojoValue(kDependencies).equal(std::move(dependencies))});
  }

  MojoValue context() const { return value.get(kContext)(); }

  /// A context that records enqueued work instead of executing it.
  MojoValue recordingContext() const { return value.get(kRecordingContext)(); }

  MojoValue value;
};

/// An instantiated, replayable graph.
class DeviceGraphRef {
public:
  explicit DeviceGraphRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kCreate = "create";
  static constexpr llvm::StringLiteral kCreateCollective = "create_collective";
  static constexpr llvm::StringLiteral kCache = "cache";
  static constexpr llvm::StringLiteral kStableDevices = "stable_devices";

  /// `args`: device context, build closure, then cache key inputs.
  static MojoValue create(llvm::ArrayRef<MojoValue> args) {
    return MojoValue(kStructDeviceGraph).get(kCreate)(/*parameters=*/{}, args);
  }

  /// Builds one graph per device from a build callback indexed by ordinal.
  /// `args`: device contexts array, build callback, cache-key callback, and
  /// `cache = ...`.
  static MojoValue createCollective(llvm::ArrayRef<MojoValue> args) {
    return MojoValue(kStructDeviceGraph)
        .get(kCreateCollective)(/*parameters=*/{}, args);
  }

  /// Selects the caching overload of `create`.
  static MojoValue cacheArg(MojoValue cache) {
    return MojoValue(kCache).equal(std::move(cache));
  }

  /// `stable_devices = [<ordinal>, ...]` — for each stable address slot of a
  /// collective, the index into the contexts array of the device whose memory
  /// holds it.
  static MojoValue stableDevicesArg(MojoValue devices) {
    return MojoValue(kStableDevices).equal(std::move(devices));
  }

  MojoValue value;
};

/// Per-model state holding cached buffers and the device graph cache.
class StateContextRef {
public:
  explicit StateContextRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kGetDeviceGraphCache =
      "get_device_graph_cache";

  /// Created on first use. Not copyable, so callers take a pointer to it.
  MojoValue getDeviceGraphCache() const {
    return value.get(kGetDeviceGraphCache)();
  }

  MojoValue value;
};

class OpaquePointerRef {
public:
  explicit OpaquePointerRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kBitcast = "unsafe_bitcast";
  static constexpr llvm::StringLiteral kTakePointee = "unsafe_take_pointee";
  /// Invoked on the type, so it has no accessor.
  static constexpr llvm::StringLiteral kUnsafeDangling = "unsafe_dangling";

  /// Returns the uncalled `unsafe_bitcast[Type]`.
  MojoValue bitcast(const MojoValue &targetType) const {
    return value.get(kBitcast)[targetType];
  }

  MojoValue takePointee() const { return value.get(kTakePointee)(); }

  MojoValue value;
};

class MojoTaskRef {
public:
  explicit MojoTaskRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kWait = "wait";

  MojoValue wait() const { return value.get(kWait)(); }

  MojoValue value;
};

class MojoStringRef {
public:
  explicit MojoStringRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kUnsafePtr = "unsafe_ptr";
  static constexpr llvm::StringLiteral kByteLength = "byte_length";

  MojoValue unsafePtr() const { return value.get(kUnsafePtr)(); }
  MojoValue byteLength() const { return value.get(kByteLength)(); }

  MojoValue value;
};

class TileTensorRef {
public:
  explicit TileTensorRef(MojoValue v) : value(std::move(v)) {}

  static constexpr llvm::StringLiteral kLayout = "layout";
  static constexpr llvm::StringLiteral kPtr = "ptr";
  static constexpr llvm::StringLiteral kUnsafeOriginCast = "unsafe_origin_cast";
  static constexpr llvm::StringLiteral kDstAddressSpace = "dst_address_space";
  static constexpr llvm::StringLiteral kSrcAddressSpace = "src_address_space";
  static constexpr llvm::StringLiteral kAddressSpace = "address_space";
  static constexpr llvm::StringLiteral kAddressSpaceCast =
      "unsafe_address_space_cast";
  static constexpr llvm::StringLiteral kStaticProduct = "static_product";

  MojoValue layout() const { return value.get(kLayout); }

  /// A `__getattr_param__` field, not a method: do not add `()`.
  MojoValue ptr() const { return value.get(kPtr); }

  /// Casts to `Copier.src_address_space`, as `Copier.copy`'s `src` requires.
  /// The staging tile is already in the dst space (see `makeStorage`).
  MojoValue castToSrcAddressSpace() const {
    MojoValue space = MojoValue(kParameterCopierType).get(kSrcAddressSpace);
    return value.get(kAddressSpaceCast)[space]();
  }

  /// A GENERIC-address-space view over staging storage, as the tile
  /// `*.elementwise` overload expects. The pointer is cast before wrapping so
  /// `linear_idx_type` is recomputed for GENERIC; casting the whole tile would
  /// keep the dst-space type and fail to match:
  ///   `TileTensor(storage.unsafe_address_space_cast[AddressSpace.GENERIC]()
  ///               .unsafe_origin_cast[MutAnyOrigin](), layout)`
  static MojoValue fromGenericStorage(const MojoValue &storage,
                                      const MojoValue &layout) {
    MojoValue genericSpace =
        MojoValue(kStructAddressSpace).get(kEnumAddressSpaceGeneric);
    MojoValue ptr = storage.get(kAddressSpaceCast)[genericSpace]().get(
        kUnsafeOriginCast)[kComptimeMutAnyOrigin]();
    return MojoValue(kStructTileTensor)({}, {ptr, layout});
  }

  /// `Int(val.dim[d]())`: `dim()` returns a `Scalar`, but tile-offset
  /// arithmetic mixes it with `Int`.
  MojoValue dim(unsigned d) const {
    return MojoValue(kStructInt)({}, {value.get("dim")[d]()});
  }

  /// Stack storage for a local tile in the copier's staging address space:
  ///   `stack_allocation[LayoutType.static_product, dtype,
  ///   address_space = copier.dst_address_space]()`
  static MojoValue makeStorage() {
    MojoValue layoutProduct =
        MojoValue(kParameterLayoutType).get(kStaticProduct);
    // A comptime field, so read it from `Copier`, not `copier`.
    MojoValue addressSpace =
        MojoValue(kAddressSpace)
            .equal(MojoValue(kParameterCopierType).get(kDstAddressSpace));
    return MojoValue(kFnStackAllocation)[{
        layoutProduct, MojoValue(kParameterDType), addressSpace}]();
  }

  ///   `TileTensor(storage.unsafe_origin_cast[MutAnyOrigin](), layout)`
  static MojoValue fromMutStorage(const MojoValue &storage,
                                  const MojoValue &layout) {
    MojoValue ptr = storage.get(kUnsafeOriginCast)[kComptimeMutAnyOrigin]();
    return MojoValue(kStructTileTensor)({}, {ptr, layout});
  }

  /// From a global-memory pointer, parenthesised for compound expressions:
  ///   `TileTensor((ptr).unsafe_origin_cast[MutUntrackedOrigin](), layout)`
  static MojoValue fromExternalPtr(const MojoValue &ptr,
                                   const MojoValue &layout) {
    MojoValue parenPtr(llvm::formatv("({})", ptr.text).str());
    MojoValue castPtr =
        parenPtr.get(kUnsafeOriginCast)[kComptimeMutUntrackedOrigin]();
    return MojoValue(kStructTileTensor)({}, {castPtr, layout});
  }

  /// Field type capturing a tensor in a tile fusion struct, with the layout
  /// rebuilt from `specParam` (e.g. `Self.spec_0`). Matches
  /// `ManagedTensorSlice.to_tile_tensor()`.
  static MojoValue captureFieldType(llvm::StringRef dtypeText,
                                    llvm::StringRef specParam) {
    return MojoValue(
        llvm::formatv("{0}[{1}, {2}[shape_types = "
                      "{3}.static_layout._shape_types, stride_types = "
                      "{3}.static_layout._stride_types], {4}]",
                      kStructTileTensor, dtypeText, kStructTileLayout,
                      specParam, kComptimeMutUntrackedOrigin)
            .str());
  }

  /// Like `fromExternalPtr` but with explicit dtype/LayoutType/mut:
  ///   `TileTensor[dtype=<dtype>, LayoutType=<layoutType>,
  ///   origin=UntrackedOrigin[mut=<mut>]](ptr=<ptr>, layout=<layout>)`
  static MojoValue fromExternalPtrExplicit(const MojoValue &ptr,
                                           const MojoValue &layout,
                                           const MojoValue &dtype,
                                           const MojoValue &layoutType,
                                           const MojoValue &mut) {
    MojoValue origin = MojoValue(
        kStructUntrackedOrigin)[{MojoValue(kParameterMut).equal(mut)}];
    MojoValue tileType = MojoValue(
        kStructTileTensor)[{MojoValue(kParameterDType).equal(dtype),
                            MojoValue(kParameterLayoutType).equal(layoutType),
                            MojoValue(kParameterOrigin).equal(origin)}];
    return tileType(
        {}, {MojoValue("ptr").equal(ptr), MojoValue(kLayout).equal(layout)});
  }

  MojoValue value;
};

/// Packs heterogeneous comptime values into a `ParameterList`. Holds no
/// instance, since `of` is invoked on the type.
class ParameterListRef {
public:
  static constexpr llvm::StringLiteral kOf = "of";

  static MojoValue of(llvm::ArrayRef<MojoValue> elements) {
    return MojoValue(kStructParameterList).get(kOf)[elements]();
  }
};

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOIDENTIFIERS_H
