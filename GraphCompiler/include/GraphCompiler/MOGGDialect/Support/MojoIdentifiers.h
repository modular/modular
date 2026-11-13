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
constexpr MojoStaticValue kOpaqueTypeMappingAttrName =
    "mogg.opaque_type_mapping";

/// The module of the primitives a graph blob's kernel calls, and the name a
/// graph with a blob imports it as.
constexpr MojoStaticValue kGraphBlobModule = "max._graph_blob";
constexpr MojoStaticValue kGraphBlobModuleAlias = "_graph_blob";

/// The `@register` decorators, owned by KGEN's `LITUtils.h`, which the Mojo
/// parser also uses. A kernel struct's method names are
/// `stringifyKernelFunctionKind`'s.
using KGEN::kFnRegister;
using KGEN::kFnRegisterInternal;

/// The prefixes of the mangled leaf names of `@extensibility.register(name)`
/// and `@extensibility.register(name, cond)`.
constexpr MojoStaticValue kRegisterDecoratorPrefix = "register(";
constexpr MojoStaticValue kRegisterCondDecoratorPrefix = "register[";

constexpr MojoStaticValue kMoggTargetParameterName = "target";

/// Lowered specially.
constexpr MojoStaticValue kMoggOutputRankParameterName = "output_rank";

/// Per-stub-argument metadata: `{dtype}` for a trait tensor, else "". The
/// layout `L` of `_MoggTensorBufferCapture[dtype, L]` is inferred from the
/// argument, since metadata cannot encode dynamic extents.
constexpr MojoStaticValue kTraitStubBufferCaptureAttrName =
    "mogg.trait_stub_buffer_capture";

/// Generic trait bounds for shape-function stubs with generic `TileTensorable`
/// parameters.
constexpr MojoStaticValue kTraitStubArgBoundsAttrName =
    "mogg.trait_stub_arg_bounds";

/// On accumulation-style fused kernels.
constexpr MojoStaticValue kMoggHasEpilogueFusionParameterName =
    "has_epilogue_fusion";

constexpr MojoStaticValue kMoggTraceNameParameterName = "_trace_name";

/// Package roots used to recognise `@register_internal` and friends without a
/// symbol-table lookup into dependency packages.
constexpr MojoStaticValue kPackageMax = "max";
using KGEN::kPackageExtensibility;
using KGEN::kPackageStd;
constexpr MojoStaticValue kPackageManagedTensorSlice = "managed_tensor_slice";

/// Last segments of `SymbolRefAttr` paths, for symbol-tree matching.
using KGEN::kLeafDeviceContext;
using KGEN::kLeafManagedTensorSlice;
constexpr MojoStaticValue kLeafVariadicTensors = "VariadicTensors";
constexpr MojoStaticValue kLeafVariadicPack = "VariadicPack";
constexpr MojoStaticValue kLeafTensorTuple = "TensorTuple";
constexpr MojoStaticValue kLeafFusedInputVariadicTensors =
    "_FusedInputVariadicTensors";
constexpr MojoStaticValue kLeafFusedOutputVariadicTensors =
    "_FusedOutputVariadicTensors";
constexpr MojoStaticValue kLeafSIMD = "SIMD";
constexpr MojoStaticValue kLeafIndexList = "IndexList";
constexpr MojoStaticValue kLeafTileTensor = "TileTensor";
constexpr MojoStaticValue kLeafDeviceContextArray = "DeviceContextArray";
constexpr MojoStaticValue kLeafDeviceGraphBuilder = "DeviceGraphBuilder";

/// The `ReduceOp` monoid trait (`algorithm.reduce_op.ReduceOp`) and its
/// `width` alias, which names the struct parameter that is the SIMD width.
constexpr MojoStaticValue kReduceOpTraitRoot = "algorithm";
constexpr MojoStaticValue kReduceOpTraitLeaf = "ReduceOp";
constexpr MojoStaticValue kReduceOpWidthAlias = "width";

/// Mojo module names the MojoRegistry builds its registry from. These are
/// required: every build has them, and one that fails to resolve leaves no
/// registry to compile a graph against, so it is fatal.
///
/// Order is not precedence. A registration these packages share is an error
/// rather than an override (see `emitDecls` in ImportPackages.cpp); only a
/// user-supplied package can override a built-in one.
constexpr MojoStaticValue kMoggRequiredKernelModuleNames[] = {
    "builtin_kernels",
    "builtin_primitives",
};

/// Mojo module names holding the registrations for the closed-source kernels
/// under `//Kernels/lib`. These are optional, which is why they are not in the
/// list above: an open-source checkout has neither the kernels nor their
/// registrations, and so does any install shipping only the open-source kernel
/// packages. Failing to resolve one costs the ops it registers rather than the
/// whole registry.
constexpr MojoStaticValue kMoggOptionalKernelModuleNames[] = {
    "graph_compiler",
};

/// Prefixes an async error with its region name. Called from
/// `AsyncErrorOp::emitMojo` besides the op's own primitive, so it can't come
/// from `getOperationName()`.
constexpr MojoStaticValue kMoggPrimitiveFormatRegionError =
    "mogg.format_region_error";

//===----------------------------------------------------------------------===//
// Mojo type symbol names (as stored in MLIR by the Mojo compiler)
//===----------------------------------------------------------------------===//

constexpr MojoStaticValue kMojoDpsTensorTypeName =
    "extensibility::managed_tensor_slice::ManagedTensorSlice";

constexpr MojoStaticValue kMojoInternalDpsSimdTypeName = "std::SIMD";
constexpr MojoStaticValue kMojoInternalDpsIntTypeName = "std::Int";
constexpr MojoStaticValue kMojoInternalDpsUintTypeName = "std::UInt";
constexpr MojoStaticValue kMojoInternalDpsBoolTypeName = "std::Bool";
constexpr MojoStaticValue kMojoInternalDpsIndexListTypeName = "std::IndexList";

/// In `buildOpaqueSymbol`'s dotted form, for `isOpaqueWithTypeName`.
constexpr MojoStaticValue kMojoInternalDpsIndexListTypeQualifiedName =
    "std.utils.index.IndexList";

/// The fused variants carry per-element fusion structs in a heterogeneous
/// Tuple.
constexpr MojoStaticValue kMojoVariadicTensorsName =
    "extensibility::managed_tensor_slice::VariadicTensors";
constexpr MojoStaticValue kMojoFusedInputVariadicTensorsName =
    "extensibility::managed_tensor_slice::_FusedInputVariadicTensors";
constexpr MojoStaticValue kMojoFusedOutputVariadicTensorsName =
    "extensibility::managed_tensor_slice::_FusedOutputVariadicTensors";

constexpr MojoStaticValue kMojoExtensibilityApiDeviceContextTypeName =
    "max::DeviceContext";

/// Used by multi-device custom ops.
constexpr MojoStaticValue kMojoExtensibilityApiDeviceContextArrayTypeName =
    "max::DeviceContextArray";

/// Inside the Graph Compiler these types show up as:
///   - StaticString -> std::StringSpan
///   - String -> std::String
///   - StringSpan -> std::Origin
constexpr MojoStaticValue kMojoInternalStaticStringTypeName = "std::StringSpan";
constexpr MojoStaticValue kMojoStaticStringTypeName = "std::StaticString";

/// In the emit-mojo path, DType parameters appear as LIT struct types with
/// type name "std::DType" rather than KGEN::DTypeType.
constexpr MojoStaticValue kMojoDtypeTypeName = "std::DType";

constexpr MojoStaticValue kMojoStringTypeName = "std::String";
constexpr MojoStaticValue kMojoInternalStringSliceTypeName = "std::Origin";

//===----------------------------------------------------------------------===//
// Mojo struct names (dotted, as emitted into Mojo source)
//===----------------------------------------------------------------------===//

constexpr MojoStaticValue kStructIO = "extensibility.IO";
constexpr MojoStaticValue kStructIndexList = "std.utils.IndexList";
constexpr MojoStaticValue kStructPointer = "Pointer";
/// `UntrackedOrigin[mut=...]`, for where `mut` must stay generic (unlike
/// `kComptimeMutUntrackedOrigin`).
constexpr MojoStaticValue kStructUntrackedOrigin = "UntrackedOrigin";
constexpr MojoStaticValue kStructOptional = "Optional";
constexpr MojoStaticValue kStructString = "String";
constexpr MojoStaticValue kStructDeviceContext = "max.gpu.host.DeviceContext";
constexpr MojoStaticValue kStructDeviceContextArray =
    "max.gpu.host.DeviceContextArray";
/// `DeviceContext.assert_default_target()`: this context as a
/// `DeviceContext[]`. TODO(DRIV-655): Remove along with its uses.
constexpr MojoStaticValue kFnAssertDefaultTarget = "assert_default_target";
constexpr MojoStaticValue kStructDeviceGraph = "max.gpu.host.DeviceGraph";
constexpr MojoStaticValue kStructDeviceGraphBuilder =
    "max.gpu.host.DeviceGraphBuilder";
/// `DeviceGraphInputKind`: how an `OwnedTensor` participates as a device graph
/// input. The `data` member is the Mojo default and is never spelled out.
constexpr MojoStaticValue kStructDeviceGraphInputKind =
    "max.gpu.host.DeviceGraphInputKind";
constexpr MojoStaticValue kInputKindMutable = "mutable";
constexpr MojoStaticValue kInputKindHost = "host";
constexpr MojoStaticValue kInputKindShapeOnly = "shape_only";
/// `OwnedTensor.with_kind[kind]()`: the same tensor under another input kind.
constexpr MojoStaticValue kFnWithKind = "with_kind";
constexpr MojoStaticValue kStructDeviceGraphNode =
    "max.gpu.host.DeviceGraphNode";
constexpr MojoStaticValue kStructDeviceGraphCache =
    "max.gpu.host.DeviceGraphCache";
constexpr MojoStaticValue kStructStableAddr = "max.gpu.host.StableAddr";
constexpr MojoStaticValue kStructManagedTensorSlice =
    "extensibility.ManagedTensorSlice";
constexpr MojoStaticValue kStructInt = "Int";
constexpr MojoStaticValue kStructSIMDLength = "SIMDLength";
constexpr MojoStaticValue kStructUInt = "UInt";
constexpr MojoStaticValue kStructBool = "Bool";
constexpr MojoStaticValue kStructAddressSpace = "std.memory.AddressSpace";
constexpr MojoStaticValue kStructDType = "DType";
constexpr MojoStaticValue kStructSIMD = "SIMD";
// Qualified because the generated module imports packages qualified
// (`import layout`), so bare names don't bind.
constexpr MojoStaticValue kStructTileTensor = "layout.TileTensor";
constexpr MojoStaticValue kStructTensorLayout = "layout.TensorLayout";
constexpr MojoStaticValue kStructTileCopier = "layout.tile_io.TileCopier";
constexpr MojoStaticValue kStructTileLayout = "layout.MixedLayout";
constexpr MojoStaticValue kStructMoggOutputStoreFusionTensor =
    "_MoggOutputStoreFusionTensor";
constexpr MojoStaticValue kStructMoggTransformFusionTensor =
    "_MoggTransformFusionTensor";
constexpr MojoStaticValue kStructMoggInputLoadFusionTensor =
    "_MoggInputLoadFusionTensor";
constexpr MojoStaticValue kStructStateContext =
    "builtin_primitives.primitives.StateContext";
constexpr MojoStaticValue kStructStaticTuple = "std.utils.StaticTuple";
constexpr MojoStaticValue kStructArray = "std.collections.Array";

inline MojoValue makeMojoArrayType(llvm::StringRef elementType, size_t size) {
  return llvm::formatv("{}[{}, {}]", kStructArray, elementType, size).str();
}

constexpr MojoStaticValue kStructScalar = "Scalar";
constexpr MojoStaticValue kStructVariadic = "Variadic";
constexpr MojoStaticValue kStructParameterList = "ParameterList";
constexpr MojoStaticValue kStructStaticTensorSpec =
    "managed_tensor_slice.StaticTensorSpec";
constexpr MojoStaticValue kStructStaticTensorSpecList =
    "extensibility.StaticTensorSpecList";
constexpr MojoStaticValue kStructVariadicTensors =
    "extensibility.VariadicTensors";
constexpr MojoStaticValue kStructTensorTuple = "extensibility.TensorTuple";
constexpr MojoStaticValue kStructStaticString = "StaticString";
constexpr MojoStaticValue kStructMsNoComputeFusionTile =
    "managed_tensor_slice._NoComputeFusionTile";
constexpr MojoStaticValue kStructMsNoOutputFusionTile =
    "managed_tensor_slice._NoOutputFusionTile";
constexpr MojoStaticValue kStructMsFusionPack =
    "managed_tensor_slice._FusionPack";
constexpr MojoStaticValue kStructMsFusedInputVariadicTensors =
    "managed_tensor_slice._FusedInputVariadicTensors";
constexpr MojoStaticValue kStructMsFusedOutputVariadicTensors =
    "managed_tensor_slice._FusedOutputVariadicTensors";
constexpr MojoStaticValue kStructTrace = "max.runtime.tracing.Trace";
constexpr MojoStaticValue kStructAnyAsyncValueRef =
    "max.runtime.async_value.AnyAsyncValueRef";
/// Re-exported through `max.runtime.tracing`, avoiding an `_tracing` import.
constexpr MojoStaticValue kStructColor = "max.runtime.tracing.Color";
constexpr MojoStaticValue kFnTraceStart = "start";
constexpr MojoStaticValue kFnTraceEnd = "end";
constexpr MojoStaticValue kParameterColor = "color";
constexpr MojoStaticValue kStructCoord = "layout.Coord";
constexpr MojoStaticValue kAliasIdx = "layout.Idx";
constexpr MojoStaticValue kFnRowMajor = "layout.row_major";
constexpr MojoStaticValue kFnTypeOf = "type_of";

//===----------------------------------------------------------------------===//
// Mojo enum values
//===----------------------------------------------------------------------===//

constexpr MojoStaticValue kEnumAddressSpaceGeneric = "GENERIC";
constexpr MojoStaticValue kEnumIoInput = "extensibility.IO.Input";
constexpr MojoStaticValue kEnumTraceLevelOp =
    "max.runtime.tracing.TraceLevel.OP";
constexpr MojoStaticValue kEnumDTypeInt8 = "int8";
/// The index-sized dtype of dynamic layout `Coord` elements.
constexpr MojoStaticValue kEnumDTypeInt = "int";

//===----------------------------------------------------------------------===//
// Mojo free-function names (emitted into Mojo source)
//===----------------------------------------------------------------------===//

// Qualified because std.memory is imported qualified.
constexpr MojoStaticValue kFnStackAllocation = "std.memory.stack_allocation";
/// For special float values with no Mojo literal spelling.
constexpr MojoStaticValue kFnNumericsNan = "std.utils.numerics.nan";
constexpr MojoStaticValue kFnNumericsInf = "std.utils.numerics.inf";
constexpr MojoStaticValue kFnSimdLoadFromManagedTensorSlice =
    "managed_tensor_slice.simd_load_from_managed_tensor_slice";
constexpr MojoStaticValue kFnSimdLoadFromTensorPointer =
    "managed_tensor_slice.simd_load_from_tensor_pointer";
constexpr MojoStaticValue kFnSimdStoreIntoManagedTensorSlice =
    "managed_tensor_slice.simd_store_into_managed_tensor_slice";
constexpr MojoStaticValue kFnSimdStoreIntoTensorPointer =
    "managed_tensor_slice.simd_store_into_tensor_pointer";
constexpr MojoStaticValue kFnUnpackTensorSpec =
    "builtin_primitives.primitives.unpack_tensor_spec";
constexpr MojoStaticValue kFnGetSimdWidthForDtypes =
    "builtin_primitives.primitives.get_simd_width_for_dtypes";
constexpr MojoStaticValue kFnPow2DivisorOfDims =
    "builtin_primitives.primitives.pow2_divisor_of_dims";
constexpr MojoStaticValue kFnUnpackBufferRef =
    "builtin_primitives.primitives.unpack_buffer_ref";
constexpr MojoStaticValue kFnUnpackDeviceCtx =
    "builtin_primitives.primitives.unpack_device_ctx";
constexpr MojoStaticValue kFnUnpackStateCtx =
    "builtin_primitives.primitives.unpack_state_ctx";
constexpr MojoStaticValue kFnUnpackTensor =
    "builtin_primitives.primitives.unpack_tensor";
constexpr MojoStaticValue kFnAllZeros =
    "builtin_primitives.primitives.all_zeros";
constexpr MojoStaticValue kFnMoggLaunchDeclarative =
    "builtin_primitives.primitives.mogg_launch_declarative";
constexpr MojoStaticValue kFnMoggAsyncPackValue =
    "builtin_primitives.primitives.mogg_async_pack_value";
constexpr MojoStaticValue kFnMoggAsyncPackOwnedTensorValue =
    "builtin_primitives.primitives.mogg_async_pack_owned_tensor_value";
constexpr MojoStaticValue kFnMoggAsyncReadyValue =
    "builtin_primitives.primitives.mogg_async_ready_value";
constexpr MojoStaticValue kFnMoggAsyncErrorValue =
    "builtin_primitives.primitives.mogg_async_error_value";
constexpr MojoStaticValue kFnMoggAssignAsyncValue =
    "builtin_primitives.primitives.mogg_assign_async_value";
constexpr MojoStaticValue kFnMsSimdLoadFromTensorPointer =
    "managed_tensor_slice.simd_load_from_tensor_pointer";
constexpr MojoStaticValue kFnMsSimdStoreIntoTensorPointer =
    "managed_tensor_slice.simd_store_into_tensor_pointer";
/// Returns a tile kernel's `(rows, cols)` tile shape.
constexpr MojoStaticValue kFnGetKernelTileShape =
    "managed_tensor_slice.get_kernel_tile_shape";
/// Fully qualified because `std.utils` does not re-export `coord`.
constexpr MojoStaticValue kFnCoordToIndexList =
    "std.utils.coord.coord_to_index_list";
constexpr MojoStaticValue kStructRowwiseRow = "algorithm.rowwise.Row";
constexpr MojoStaticValue kStructRowwiseContextParams =
    "algorithm.rowwise_types.ContextParams";
constexpr MojoStaticValue kStructRowwiseContext =
    "algorithm.rowwise_types.Context";
constexpr MojoStaticValue kFuncRowwiseAtAxis =
    "algorithm.rowwise_types.at_axis";
/// Fully qualified because `std.utils` does not re-export `coord`.
constexpr MojoStaticValue kAliasDynamicCoord = "std.utils.coord.DynamicCoord";
/// Package-private, so qualified with `_asyncrt`.
constexpr MojoStaticValue kFnCreateTask = "std.runtime._asyncrt.create_task";

/// Affinity-aware variant (`desired_worker_id=<id>`), for GPU-device launches.
constexpr MojoStaticValue kFnCreateTaskWithAffinity =
    "std.runtime._asyncrt._create_task";
constexpr MojoStaticValue kFnTaskIdForDevice =
    "max.runtime.asyncrt.task_id_for_device";
constexpr MojoStaticValue kFnGetSafeTaskId =
    "max.runtime.tracing.get_safe_task_id";

constexpr MojoStaticValue kFnMax = "std.math.max";
constexpr MojoStaticValue kFnPrint = "print";

// Field / method names accessed via `.foo` live on the wrapper classes below.

//===----------------------------------------------------------------------===//
// Trait names (used in struct conformance declarations)
//===----------------------------------------------------------------------===//

constexpr MojoStaticValue kTraitInputFusion =
    "managed_tensor_slice.InputFusion";
constexpr MojoStaticValue kTraitOutputFusion =
    "managed_tensor_slice.OutputFusion";
constexpr MojoStaticValue kTraitComputeOutputFusion =
    "managed_tensor_slice.ComputeOutputFusion";
constexpr MojoStaticValue kTraitComputeOutputFusionTile =
    "managed_tensor_slice.ComputeOutputFusionTile";
constexpr MojoStaticValue kTraitOutputFusionTile =
    "managed_tensor_slice.OutputFusionTile";
constexpr MojoStaticValue kTraitElementwiseFusion =
    "managed_tensor_slice.ElementwiseFusion";
constexpr MojoStaticValue kTraitElementwiseFusionTile =
    "managed_tensor_slice.ElementwiseFusionTile";

//===----------------------------------------------------------------------===//
// Decorator strings
//===----------------------------------------------------------------------===//

constexpr MojoStaticValue kDecoratorAlwaysInline = "@inline(.always)";
constexpr MojoStaticValue kDecoratorNoInline = "@inline(.never)";

//===----------------------------------------------------------------------===//
// Pre-rendered comptime type expressions
//===----------------------------------------------------------------------===//

constexpr MojoStaticValue kComptimeOpaquePointer =
    "OpaquePointer[MutAnyOrigin]";
constexpr MojoStaticValue kComptimeInputTensor = "extensibility.InputTensor";
constexpr MojoStaticValue kComptimeOutputTensor = "extensibility.OutputTensor";
constexpr MojoStaticValue kComptimeMutableInputTensor =
    "managed_tensor_slice._MutableInputTensor";
constexpr MojoStaticValue kComptimeUInt8 = "UInt8";
constexpr MojoStaticValue kComptimeMutAnyOrigin = "MutAnyOrigin";
constexpr MojoStaticValue kComptimeMutUntrackedOrigin = "MutUntrackedOrigin";
constexpr MojoStaticValue kComptimeNone = "None";
constexpr MojoStaticValue kComptimeOptionalErrorEmpty = "Optional[Error]()";
constexpr MojoStaticValue kComptimeMutByteBuffer =
    "builtin_primitives.primitives.MutByteBuffer";
constexpr MojoStaticValue kStructOwnedByteBuffer =
    "builtin_primitives.primitives.OwnedByteBuffer";
constexpr MojoStaticValue kStructOwnedTensor =
    "builtin_primitives.primitives.OwnedTensor";
/// Builds an `AsyncValue` over an `OwnedByteBuffer` / `OwnedTensor`'s memory,
/// to register a device-graph output.
constexpr MojoStaticValue kFnAsyncPack = "async_pack";
constexpr MojoStaticValue kComptimeDTypeNoneType = "NoneType";

//===----------------------------------------------------------------------===//
// Parameter names (kernel param inference keys, kwargs in emitted calls)
//===----------------------------------------------------------------------===//

constexpr MojoStaticValue kParameterMut = "mut";
constexpr MojoStaticValue kParameterKind = "kind";
constexpr MojoStaticValue kParameterInput = "input";
constexpr MojoStaticValue kParameterDType = "dtype";
constexpr MojoStaticValue kParameterRank = "rank";
constexpr MojoStaticValue kParameterWidth = "width";
constexpr MojoStaticValue kParameterIOSpec = "io_spec";
constexpr MojoStaticValue kParameterStaticSpec = "static_spec";
constexpr MojoStaticValue kParameterSize = "size";
constexpr MojoStaticValue kParameterLength = "length";
constexpr MojoStaticValue kParameterStaticSpecs = "static_specs";
constexpr MojoStaticValue kParameterSimdWidth = "simd_width";
constexpr MojoStaticValue kParameterPreferredSimdWidth = "simd_width";
constexpr MojoStaticValue kArgMaxSimdWidth = "max_simd_width";
constexpr MojoStaticValue kParameterElementAlignment = "element_alignment";
/// The rowwise cache lambda's alignment parameter.
constexpr MojoStaticValue kParameterAlignment = "alignment";
/// The rowwise phase lambda's width parameter (`def contribute[w]`).
constexpr MojoStaticValue kParameterTileWidth = "w";
/// The source tensor's static alignment, which a `TileTensor` does not carry.
constexpr MojoStaticValue kParameterTensorAlignment = "tensor_alignment";
constexpr MojoStaticValue kParameterElementSize = "element_size";
constexpr MojoStaticValue kParameterOutputStaticShape = "output_static_shape";
constexpr MojoStaticValue kParameterStaticStarts = "static_starts";
constexpr MojoStaticValue kParameterStaticSteps = "static_steps";
constexpr MojoStaticValue kParameterStaticPermutations = "static_permutations";
constexpr MojoStaticValue kParameterLayoutType = "LayoutType";
constexpr MojoStaticValue kParameterOrigin = "origin";
constexpr MojoStaticValue kParameterAddressSpace = "address_space";
constexpr MojoStaticValue kParameterLinearIdxType = "linear_idx_type";
constexpr MojoStaticValue kParameterStorage = "Storage";
constexpr MojoStaticValue kParameterTaskId = "task_id";
constexpr MojoStaticValue kParameterOutput = "output";
constexpr MojoStaticValue kParameterElementType = "element_type";
constexpr MojoStaticValue kParameterTo = "to";
/// Keep in sync with `foreach_fusion` in
/// `max/kernels/src/graph_compiler/builtin_primitives/primitives.mojo`.
constexpr MojoStaticValue kParameterForeachTensor = "tensor";
constexpr MojoStaticValue kParameterForeachElem = "elem";
constexpr MojoStaticValue kParameterForeachCtx = "ctx";
constexpr MojoStaticValue kMoggForeachTileRegistration =
    "mogg.call.foreach.tile";
constexpr MojoStaticValue kParameterSelf = "self";
constexpr MojoStaticValue kParameterIdx = "idx";
constexpr MojoStaticValue kParameterVal = "val";
constexpr MojoStaticValue kParameterDst = "dst";
constexpr MojoStaticValue kParameterTileCoords = "tile_coords";
constexpr MojoStaticValue kParameterCopier = "copier";
/// Must differ from `kParameterCopier`, or the emitted method redefines
/// `copier`.
constexpr MojoStaticValue kParameterCopierType = "Copier";
constexpr MojoStaticValue kParameterStoreCopier = "store_copier";
constexpr MojoStaticValue kParameterStoreCopierType = "StoreCopier";

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

constexpr MojoStaticValue kInputTensor = "InputTensor";
constexpr MojoStaticValue kOutputTensor = "OutputTensor";
constexpr MojoStaticValue kMutableInputTensor = "MutableInputTensor";
constexpr MojoStaticValue kFusedInputTensor = "FusedInputTensor";
constexpr MojoStaticValue kFusedOutputTensor = "FusedOutputTensor";
constexpr MojoStaticValue kFusedComputeOutputTensor =
    "FusedComputeOutputTensor";
constexpr MojoStaticValue kFusedComputeOutputTileTensor =
    "FusedComputeOutputTileTensor";
constexpr MojoStaticValue kFusedOutputTileTensor = "FusedOutputTileTensor";

/// Mojo `IO` member names, as they appear in `KGEN::SugarKind::MemberAlias`
/// attributes of a stored `IOSpec` parameter.
constexpr MojoStaticValue kIoInput = "Input";
constexpr MojoStaticValue kIoOutput = "Output";
constexpr MojoStaticValue kIoFusedInput = "FusedInput";
constexpr MojoStaticValue kIoFusedOutput = "FusedOutput";
constexpr MojoStaticValue kIoFusedComputeOutput = "FusedComputeOutput";
constexpr MojoStaticValue kIoFusedComputeOutputTile = "_FusedComputeOutputTile";
constexpr MojoStaticValue kIoFusedOutputTile = "_FusedOutputTile";

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

  static constexpr MojoStaticValue kUnsafePtr = "unsafe_ptr";
  static constexpr MojoStaticValue kShape = "shape";
  static constexpr MojoStaticValue kStrides = "strides";
  static constexpr MojoStaticValue kToTileTensor = "to_tile_tensor";
  static constexpr MojoStaticValue kStaticSpec = "static_spec";
  static constexpr MojoStaticValue kStaticShapeTuple = "_static_shape_tuple";
  static constexpr MojoStaticValue kStaticStridesTuple =
      "_static_strides_tuple";
  static constexpr MojoStaticValue kAlignment = "alignment";
  /// Storage pointer of VariadicTensors elements.
  static constexpr MojoStaticValue kPtr = "_ptr";
  static constexpr MojoStaticValue kIoSpec = "io_spec";
  /// Bind methods for fusion lambdas.
  static constexpr MojoStaticValue kBindToFusedInput = "_bind_to_fused_input";
  static constexpr MojoStaticValue kBindToFusedOutput = "_bind_to_fused_output";
  static constexpr MojoStaticValue kBindToFusedComputeOutput =
      "_bind_to_fused_compute_output";
  static constexpr MojoStaticValue kBindToFusedComputeOutputTile =
      "_bind_to_fused_compute_output_tile";
  static constexpr MojoStaticValue kBindToFusedOutputTile =
      "_bind_to_fused_output_tile";

  /// Fusion functor fields and the matching type parameters.
  static constexpr MojoStaticValue kInFusion = "in_fusion";
  static constexpr MojoStaticValue kOutFusion = "out_fusion";
  static constexpr MojoStaticValue kInFusionParam = "InFusion";
  static constexpr MojoStaticValue kOutFusionParam = "OutFusion";

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
    return kFnTypeOf({value}).get(kInFusionParam);
  }
  MojoValue outFusionType() const {
    return kFnTypeOf({value}).get(kOutFusionParam);
  }

  MojoValue unsafePtr() const { return value.get(kUnsafePtr)(); }
  MojoValue shape() const { return value.get(kShape)(); }
  MojoValue strides() const { return value.get(kStrides)(); }
  MojoValue toTileTensor() const { return value.get(kToTileTensor)(); }
  MojoValue bindToFusedComputeOutputTile() const {
    return value.get(kBindToFusedComputeOutputTile);
  }
  MojoValue bindToFusedOutputTile() const {
    return value.get(kBindToFusedOutputTile);
  }

  /// Return the uncalled method handle.
  MojoValue bindToFusedInput() const { return value.get(kBindToFusedInput); }
  MojoValue bindToFusedOutput() const { return value.get(kBindToFusedOutput); }
  MojoValue bindToFusedComputeOutput() const {
    return value.get(kBindToFusedComputeOutput);
  }

  MojoValue value;
};

class StaticTensorSpecRef {
public:
  explicit StaticTensorSpecRef(MojoValue v) : value(std::move(v)) {}

  static constexpr MojoStaticValue kStaticLayout = "static_layout";
  static constexpr MojoStaticValue kAlignment = "alignment";
  static constexpr MojoStaticValue kRank = "rank";
  static constexpr MojoStaticValue kGetInternals = "get_internals";
  /// Comptime members of `StaticTensorSpec`, not of `get_internals()`.
  static constexpr MojoStaticValue kShapeTuple = "shape_tuple";
  static constexpr MojoStaticValue kStridesTuple = "strides_tuple";

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

  static constexpr MojoStaticValue kView = "view";
  static constexpr MojoStaticValue kUnsafePtr = "unsafe_ptr";

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

  static constexpr MojoStaticValue kTensor = "tensor";

  MojoValue tensor() const { return value.get(kTensor); }

  MojoValue value;
};

/// The recorder handed to the build closure of `mgp.device_graph.create`.
class DeviceGraphBuilderRef {
public:
  explicit DeviceGraphBuilderRef(MojoValue v) : value(std::move(v)) {}

  static constexpr MojoStaticValue kAddInput = "add_input";
  static constexpr MojoStaticValue kAddOutput = "add_output";
  static constexpr MojoStaticValue kRegion = "region";
  static constexpr MojoStaticValue kContext = "context";
  static constexpr MojoStaticValue kRecordingContext = "recording_context";
  static constexpr MojoStaticValue kDependencies = "dependencies";
  static constexpr MojoStaticValue kGetStable = "get_stable";
  static constexpr MojoStaticValue kPlaceStable = "place_stable";

  /// A graph-owned stable location recorded in place of the caller's buffer,
  /// so replay does not depend on the caller's address.
  MojoValue addInput(MojoValue input) const {
    return value.get(kAddInput)({std::move(input)});
  }

  /// `builder.get_stable(index)`.
  MojoValue getStable(MojoValue index) const {
    return value.get(kGetStable)({std::move(index)});
  }

  /// `builder.place_stable(index, ptr)`.
  MojoValue placeStable(MojoValue index, MojoValue ptr) const {
    return value.get(kPlaceStable)({std::move(index), std::move(ptr)});
  }

  /// `builder.add_output`.
  MojoValue addOutput() const { return value.get(kAddOutput); }

  /// Chains every node `work` records and returns the last as a single
  /// dependency.
  MojoValue region(MojoValue work, MojoValue dependencies) const {
    return value.get(kRegion)(
        {std::move(work), kDependencies.equal(std::move(dependencies))});
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

  static constexpr MojoStaticValue kCreate = "create";
  static constexpr MojoStaticValue kCreateCollective = "create_collective";
  static constexpr MojoStaticValue kCache = "cache";
  static constexpr MojoStaticValue kStableDevices = "stable_devices";

  /// `args`: device context, build closure, then cache key inputs.
  static MojoValue create(llvm::ArrayRef<MojoValue> args) {
    return kStructDeviceGraph.get(kCreate)(args);
  }

  /// Builds one graph per device from a build callback indexed by ordinal.
  /// `args`: device contexts array, build callback, cache-key callback, and
  /// `cache = ...`.
  static MojoValue createCollective(llvm::ArrayRef<MojoValue> args) {
    return kStructDeviceGraph.get(kCreateCollective)(args);
  }

  /// Selects the caching overload of `create`.
  static MojoValue cacheArg(MojoValue cache) {
    return kCache.equal(std::move(cache));
  }

  /// `stable_devices = [<ordinal>, ...]` — for each stable address slot of a
  /// collective, the index into the contexts array of the device whose memory
  /// holds it.
  static MojoValue stableDevicesArg(MojoValue devices) {
    return kStableDevices.equal(std::move(devices));
  }

  MojoValue value;
};

/// Per-model state holding cached buffers and the device graph cache.
class StateContextRef {
public:
  explicit StateContextRef(MojoValue v) : value(std::move(v)) {}

  static constexpr MojoStaticValue kGetDeviceGraphCache =
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

  static constexpr MojoStaticValue kBitcast = "unsafe_bitcast";
  static constexpr MojoStaticValue kTakePointee = "unsafe_take_pointee";
  /// Invoked on the type, so it has no accessor.
  static constexpr MojoStaticValue kUnsafeDangling = "unsafe_dangling";

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

  static constexpr MojoStaticValue kWait = "wait";

  MojoValue wait() const { return value.get(kWait)(); }

  MojoValue value;
};

class MojoStringRef {
public:
  explicit MojoStringRef(MojoValue v) : value(std::move(v)) {}

  static constexpr MojoStaticValue kUnsafePtr = "unsafe_ptr";
  static constexpr MojoStaticValue kByteLength = "byte_length";

  MojoValue unsafePtr() const { return value.get(kUnsafePtr)(); }
  MojoValue byteLength() const { return value.get(kByteLength)(); }

  MojoValue value;
};

class TileTensorRef {
public:
  explicit TileTensorRef(MojoValue v) : value(std::move(v)) {}

  static constexpr MojoStaticValue kLayout = "layout";
  static constexpr MojoStaticValue kPtr = "ptr";
  static constexpr MojoStaticValue kUnsafeOriginCast = "unsafe_origin_cast";
  static constexpr MojoStaticValue kDstAddressSpace = "dst_address_space";
  static constexpr MojoStaticValue kSrcAddressSpace = "src_address_space";
  static constexpr MojoStaticValue kAddressSpace = "address_space";
  static constexpr MojoStaticValue kAddressSpaceCast =
      "unsafe_address_space_cast";
  static constexpr MojoStaticValue kStaticProduct = "static_product";

  MojoValue layout() const { return value.get(kLayout); }

  /// A `__getattr_param__` field, not a method: do not add `()`.
  MojoValue ptr() const { return value.get(kPtr); }

  /// Casts to `Copier.src_address_space`, as `Copier.copy`'s `src` requires.
  /// The staging tile is already in the dst space (see `makeStorage`).
  MojoValue castToSrcAddressSpace() const {
    MojoValue space = kParameterCopierType.get(kSrcAddressSpace);
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
    MojoValue genericSpace = kStructAddressSpace.get(kEnumAddressSpaceGeneric);
    MojoValue ptr = storage.get(kAddressSpaceCast)[genericSpace]().get(
        kUnsafeOriginCast)[kComptimeMutAnyOrigin]();
    return kStructTileTensor({ptr, layout});
  }

  /// `Int(val.dim[d]())`: `dim()` returns a `Scalar`, but tile-offset
  /// arithmetic mixes it with `Int`.
  MojoValue dim(unsigned d) const {
    return kStructInt({value.get("dim")[d]()});
  }

  /// Stack storage for a local tile in the copier's staging address space:
  ///   `stack_allocation[LayoutType.static_product, dtype,
  ///   address_space = copier.dst_address_space]()`
  static MojoValue makeStorage() {
    MojoValue layoutProduct = kParameterLayoutType.get(kStaticProduct);
    // A comptime field, so read it from `Copier`, not `copier`.
    MojoValue addressSpace =
        kAddressSpace.equal(kParameterCopierType.get(kDstAddressSpace));
    return kFnStackAllocation[{layoutProduct, kParameterDType, addressSpace}]();
  }

  ///   `TileTensor(storage.unsafe_origin_cast[MutAnyOrigin](), layout)`
  static MojoValue fromMutStorage(const MojoValue &storage,
                                  const MojoValue &layout) {
    MojoValue ptr = storage.get(kUnsafeOriginCast)[kComptimeMutAnyOrigin]();
    return kStructTileTensor({ptr, layout});
  }

  /// From a global-memory pointer, parenthesised for compound expressions:
  ///   `TileTensor((ptr).unsafe_origin_cast[MutUntrackedOrigin](), layout)`
  static MojoValue fromExternalPtr(const MojoValue &ptr,
                                   const MojoValue &layout) {
    MojoValue parenPtr(llvm::formatv("({})", ptr.text).str());
    MojoValue castPtr =
        parenPtr.get(kUnsafeOriginCast)[kComptimeMutUntrackedOrigin]();
    return kStructTileTensor({castPtr, layout});
  }

  /// Field type capturing a tensor in a tile fusion struct, with the layout
  /// rebuilt from `specParam` (e.g. `Self.spec_0`). Matches
  /// `ManagedTensorSlice.to_tile_tensor()`.
  static MojoValue captureFieldType(llvm::StringRef dtypeText,
                                    llvm::StringRef specParam) {
    return llvm::formatv("{0}[{1}, {2}[shape_types = "
                         "{3}.static_layout._shape_types, stride_types = "
                         "{3}.static_layout._stride_types], {4}]",
                         kStructTileTensor, dtypeText, kStructTileLayout,
                         specParam, kComptimeMutUntrackedOrigin)
        .str();
  }

  /// Like `fromExternalPtr` but with explicit dtype/LayoutType/mut:
  ///   `TileTensor[dtype=<dtype>, LayoutType=<layoutType>,
  ///   origin=UntrackedOrigin[mut=<mut>]](ptr=<ptr>, layout=<layout>)`
  static MojoValue fromExternalPtrExplicit(const MojoValue &ptr,
                                           const MojoValue &layout,
                                           const MojoValue &dtype,
                                           const MojoValue &layoutType,
                                           const MojoValue &mut) {
    MojoValue origin = kStructUntrackedOrigin[{kParameterMut.equal(mut)}];
    MojoValue tileType = kStructTileTensor[{
        kParameterDType.equal(dtype), kParameterLayoutType.equal(layoutType),
        kParameterOrigin.equal(origin)}];
    return tileType({MojoValue("ptr").equal(ptr), kLayout.equal(layout)});
  }

  MojoValue value;
};

/// Packs heterogeneous comptime values into a `ParameterList`. Holds no
/// instance, since `of` is invoked on the type.
class ParameterListRef {
public:
  static constexpr MojoStaticValue kOf = "of";

  static MojoValue of(llvm::ArrayRef<MojoValue> elements) {
    return kStructParameterList.get(kOf)[elements]();
  }
};

/// `layout.row_major(<shape>)`: the row-major layout of a shape.
inline MojoValue rowMajorLayout(const MojoValue &shape) {
  return kFnRowMajor({shape});
}

/// `coord_to_index_list(<coord>)`, optionally rebound to `IndexList[rank]`.
inline MojoValue
coordToIndexList(const MojoValue &coord,
                 std::optional<MojoValue> rank = std::nullopt) {
  MojoValue list = kFnCoordToIndexList({coord});
  return rank ? list.rebind(kStructIndexList[*rank]) : list;
}

/// `Scalar[<dtype>]`.
inline MojoValue scalarOfDType(const MojoValue &dtype) {
  return kStructScalar[dtype];
}

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SUPPORT_MOJOIDENTIFIERS_H
