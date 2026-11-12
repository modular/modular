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
// This file emits the MOGG dialect operations as Mojo text.
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

//===----------------------------------------------------------------------===//
// MOGG CallOp
//===----------------------------------------------------------------------===//

// Returns the fully-qualified Mojo source name for a PrimitiveDeclOp, which
// ImportPackages stores as its sym_name.
static llvm::StringRef primitiveMojoName(PrimitiveDeclOp decl) {
  return decl.getSymNameAttr().strref();
}

// Resolves `op`'s call to the `kind` entry point of `adaptor`'s kernel,
// inferring the parameters from `op` and `kernelParams`. Every call to a
// kernel's `execute`, `build_graph`, `execute_configuration` or
// `execute_device` resolves here, so they infer parameters the same way. Two
// calls don't: an elementwise kernel's (`emitElementwiseCallExpr`) infers its
// dtype from its output and calls a free function, and a shape function's
// (`CallShapeOp::emitMojo`) infers against a `ShapeFunctionDeclOp`, which isn't
// a kernel.
static FailureOr<KernelCallee>
resolveKernelCallee(Operation *op, EmitMojoState &state,
                    KernelDeclAdaptor &adaptor, KernelFunctionKind kind,
                    DictionaryAttr kernelParams) {
  const KernelDeclFunctionAdaptor &fn = *adaptor.getFunction(kind);
  auto params = inferKernelParameters(adaptor, kind, [&](StringRef paramName) {
    return state.inferDefaultKernelParamValue(op, paramName, kernelParams);
  });
  if (params.isError()) {
    op->emitError() << "Failed to infer all the parameters of '"
                    << fn.fullyResolvedName.strref()
                    << "': " << params.takeError();
    return failure();
  }
  auto [kernel, method] =
      EmitMojoState::splitFunctionName(fn.fullyResolvedName.strref());
  return KernelCallee{kernel[params->structParameters].get(method),
                      std::move(params->functionParameters)};
}

namespace {

/// Check if input value is a mutable tensor
bool isMutableInputTensor(Value input) {
  auto tensorType = dyn_cast<MO::TensorType>(input.getType());
  if (!tensorType)
    return false;
  auto ioKindAttr = tensorType.getIOKindAttr();
  auto isMutableAttr = tensorType.getIsMutableAttr();
  return ioKindAttr && isMutableAttr && isMutableAttr.getValue();
}

/// Generate mogg_tensor_init to transform MutableInputTensor to InputTensor
MojoValue generateMutableTensorInit(Value input, EmitMojoState &state) {
  auto tensorType = cast<MO::TensorType>(input.getType());
  MojoValue inputValue = state.getMojoValueFor(input);

  // A slot-backed view re-inits as an immutable `StableTensor` over the same
  // slot and layout; there is no data pointer to hand to the init primitive.
  if (tensorType.isStable()) {
    StableTensorRef stable(inputValue);
    MojoValue ctor = MojoValue(kComptimeStableInputTensor)[{
        MojoValue(kParameterStaticSpec)
            .equal(MojoValue("type_of")({}, {inputValue})
                       .get(StableTensorRef::kStaticSpec))}];
    return state.file->var(ctor({}, {stable.addr(), stable.runtimeLayout()}));
  }

  // Extract dtype
  ErrorOr<DType> dTypeOr = M::MO::fromMLIRType(tensorType.getElementType());
  ASSERT_STREAM(!dTypeOr.isError(),
                "Failed to extract dtype from mutable input tensor");
  MojoValue dtype = state.mlirDTypeToMojoType(*dTypeOr);

  ASSERT_STREAM(tensorType.hasRank(),
                "Mutable input tensor must have statically known rank");
  MojoValue rank(static_cast<int64_t>(tensorType.getRankNonZero()));

  ManagedTensorSliceRef mts(inputValue);
  StaticTensorSpecRef staticSpec(mts.staticSpec());

  // Build template parameters. The static layout is conveyed through the
  // layout argument's type (below), so no separate layout param.
  SmallVector<MojoValue> templateParams = {dtype, rank, MojoValue(false),
                                           MojoValue(kEnumIoInput),
                                           staticSpec.alignment()};

  // Build runtime arguments: the slice already carries its layout (shape and
  // strides as static-aware Coords), so forward it directly.
  SmallVector<MojoValue> runtimeArgs = {
      OpaquePointerRef(mts.rawPtr())
          .bitcast(MojoValue(kComptimeDTypeNoneType))(),
      inputValue.get("runtime_layout")()};

  auto tensorInitPrimDecl =
      state.registry.lookupPrimitive(TensorInitOp::getOperationName());
  MojoValue moggTensorInit(primitiveMojoName(tensorInitPrimDecl));
  return state.file->var(moggTensorInit(templateParams, runtimeArgs));
}
} // namespace

LogicalResult CallBuildGraphOp::emitMojo(EmitMojoState &state) {
  // Mirrors CallExecuteOp::emitMojo, but selects the kernel's `build_graph`
  // function, appends the device-graph builder as the launch argument (in place
  // of the device context), and captures the recorded node handle as the op's
  // result.
  KernelDeclOp buildGraphKernelDecl =
      state.registry.lookupKernelBySymName(getKernelSymName());

  KernelDeclAdaptor buildGraphAdaptor(buildGraphKernelDecl);
  if (!buildGraphAdaptor.buildGraphFunction)
    return emitOpError("kernel '")
           << getKernelSymName() << "' has no 'build_graph' function";
  const KernelDeclFunctionAdaptor &buildGraphFnAdaptor =
      *buildGraphAdaptor.buildGraphFunction;

  // Collect kernel I/Os: DPS outputs, then inputs (mirroring `execute`).
  auto kernelOperands = llvm::map_to_vector(
      getOutputs(), [&](Value val) { return state.getMojoValueFor(val); });

  auto inputs = getInputs();
  for (auto [i, input] : llvm::enumerate(inputs)) {
    MojoValue inputValue = state.getMojoValueFor(input);

    // Transform mutable inputs if the kernel expects an InputTensor.
    if (isMutableInputTensor(input) &&
        i < buildGraphFnAdaptor.inputArguments.size()) {
      const auto *tensorInfo = std::get_if<KernelDeclTensorAdaptor>(
          &buildGraphFnAdaptor.inputArguments[i].underlyingType);
      if (tensorInfo && tensorInfo->ioSpec == MOGG::IOSpec::InputTensor)
        inputValue = generateMutableTensorInit(input, state);
    }

    kernelOperands.push_back(inputValue);
  }

  // The launch argument is the device-graph builder (not a context).
  kernelOperands.push_back(state.getMojoValueFor(getGraphBuilder()));

  FailureOr<KernelCallee> buildGraph =
      resolveKernelCallee(*this, state, buildGraphAdaptor,
                          KernelFunctionKind::BuildGraph, getKernelParamAttr());
  if (failed(buildGraph))
    return failure();

  // `build_graph` records the dispatch into the builder; ordering comes from
  // the enclosing `mgp.device_graph.dependencies` op, so this is a void call.
  state.file->call(buildGraph->function, buildGraph->parameters,
                   kernelOperands);
  state.genAllKeepAliveStatements();
  return success();
}

//===----------------------------------------------------------------------===//
// Error context helpers (used by FuncOp and AsyncErrorOp)
//===----------------------------------------------------------------------===//

/// Collect source notes (Python stack traces) from a location tree. Source
/// notes are StringAttr metadata on FusedLoc nodes, set by
/// AttachPythonStackTraces.
static SmallVector<StringRef> collectSourceNotes(Location rootLoc) {
  llvm::SetVector<StringRef> notes;
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](mlir::FusedLoc loc) {
    if (auto metadata = dyn_cast_if_present<StringAttr>(loc.getMetadata()))
      if (auto val = metadata.getValue(); !val.empty())
        notes.insert(metadata.getValue());
  });
  walker.walk<mlir::WalkOrder::PreOrder>(rootLoc);
  return notes.takeVector();
}

/// Parse a single Python traceback frame line of the form:
///   File "path/to/file.py", line 42, in func_name
/// into a compact "last/components.py:42:func_name" representation, keeping
/// only the trailing `numComponents` path segments.
static std::string compactFrame(StringRef frameLine,
                                unsigned numComponents = 3) {
  // Expected format: `  File "<path>", line <N>, in <func>`
  auto filePos = frameLine.find("File \"");
  if (filePos == StringRef::npos)
    return "";
  StringRef afterFile = frameLine.substr(filePos + 6); // skip `File "`
  auto quoteEnd = afterFile.find('"');
  if (quoteEnd == StringRef::npos)
    return "";
  StringRef fullPath = afterFile.substr(0, quoteEnd);
  StringRef rest = afterFile.substr(quoteEnd + 1); // after closing quote

  // Extract line number: `, line <N>`
  auto linePos = rest.find("line ");
  StringRef lineNo;
  if (linePos != StringRef::npos) {
    StringRef afterLine = rest.substr(linePos + 5);
    lineNo = afterLine.take_while([](char c) { return c >= '0' && c <= '9'; });
  }

  // Extract function name: `, in <func>`
  StringRef func;
  auto inPos = rest.find(", in ");
  if (inPos != StringRef::npos) {
    func = rest.substr(inPos + 5).trim();
  }

  // Keep only the last N path components.
  SmallVector<StringRef, 8> components;
  fullPath.split(components, '/');
  unsigned start =
      components.size() > numComponents ? components.size() - numComponents : 0;
  std::string shortPath;
  llvm::raw_string_ostream pathOs(shortPath);
  for (unsigned i = start; i < components.size(); ++i) {
    if (i > start)
      pathOs << '/';
    pathOs << components[i];
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  os << shortPath;
  if (!lineNo.empty())
    os << ':' << lineNo;
  if (!func.empty())
    os << ':' << func;
  return result;
}

/// Return the maximum number of Python stack frames to include in NVTX trace
/// detail strings. Configurable via MODULAR_TRACE_MAX_FRAMES (default 10).
static unsigned getTraceMaxFrames() {
  static unsigned cached = [] {
    if (const char *env = std::getenv("MODULAR_TRACE_MAX_FRAMES")) {
      int val = std::atoi(env);
      if (val > 0)
        return static_cast<unsigned>(val);
    }
    return 10u;
  }();
  return cached;
}

/// Collect source notes for NVTX trace detail strings, each trace cut to its
/// bottom N frames (MODULAR_TRACE_MAX_FRAMES, default 10) in `path:line:func`
/// form. Non-empty only if `max-debug.source-tracebacks` was enabled at graph
/// construction (e.g. `MODULAR_DEBUG=source-tracebacks`).
static std::string collectCompactSourceNotes(Location rootLoc) {
  unsigned maxFrames = getTraceMaxFrames();
  auto notes = collectSourceNotes(rootLoc);
  if (notes.empty())
    return "";

  std::string result;
  llvm::raw_string_ostream os(result);

  for (auto note : notes) {
    // Parse all frames from the traceback, then take the bottom N.
    SmallVector<std::string, 8> frames;
    StringRef remaining = note;
    while (!remaining.empty()) {
      auto [line, rest] = remaining.split('\n');
      if (line.contains("File \""))
        if (auto compact = compactFrame(line); !compact.empty())
          frames.push_back(std::move(compact));
      remaining = rest;
    }

    unsigned start = frames.size() > maxFrames ? frames.size() - maxFrames : 0;
    for (unsigned i = start; i < frames.size(); ++i) {
      if (i > start)
        os << '\n';
      os << frames[i];
    }
    if (!frames.empty())
      os << '\n';
  }

  // Ops with the same traceback share a note: this counts tracebacks, not ops.
  if (notes.size() > 1)
    os << "[fused from " << notes.size() << " source locations]";

  return result;
}

LogicalResult FuncOp::emitMojo(EmitMojoState &state) {
  // Emit the fusion info tree as a top-level comment before the function.
  state.file->comment("FusionInfo:");
  std::string fusionTree = getFusionInfoAttr().toPrettyTree();
  for (auto line : llvm::split(fusionTree, '\n'))
    state.file->comment(("  " + line).str());

  // If MOGG, this is an advanced-fusion stub outlined by
  // OutlineRoutines. Otherwise it was a AsyncRoutineOp
  // which has that implicit void** ABI.
  if (getAbi() == ABIKind::MOGG)
    return emitMojoMOGGABI(state);
  return emitMojoMGPABI(state);
}

LogicalResult FuncOp::emitMojoMGPABI(EmitMojoState &state) {
  state.mojoReturnValues.clear();
  state.mojoConstantResources.clear();

  bool errOccurred = false;
  static const MOGG::MojoValue vMutAnyOrigin =
      MOGG::MojoValue(MOGG::kComptimeMutAnyOrigin);
  static const MOGG::MojoValue vPointer = MOGG::kStructPointer;
  static const MOGG::MojoValue vOpaquePointer = MOGG::kComptimeOpaquePointer;

  state.file->def(
      getSymNameAttr().strref(), {},
      {MOGG::NamedMojoDef("arg", vPointer[{vOpaquePointer, vMutAnyOrigin}])},
      std::nullopt, /*exported=*/true, /*raises=*/false,
      /*isAlwaysInline=*/false, [&](const MOGG::FunctionBody &functionBody) {
        state.file->comment(
            "This is the entry point of the kernel. The argument is a "
            "pointer to a list of inputs and outputs.");

        auto runtimeArgument = functionBody.arguments.front().resolvedName;
        auto &kernelBody = getBody().front();
        // runtimeArgument order: inputs and DPS outputs, then return values
        // (chain and non-DPS outputs).
        //
        // The runtime expects the generated Mojo to take ownership of every
        // AsyncValue input slot, even if unused:
        //
        //     var val = AnyAsyncValueRef(take_slot=args[i])
        //
        // Non-async arguments (e.g. buffer constants) remain raw pointer slots.
        for (auto argument : kernelBody.getArguments()) {
          MOGG::MojoValue slotPtr =
              runtimeArgument.pointerOffsetGet(argument.getArgNumber());
          if (isa<MOGG::AsyncType>(argument.getType())) {
            MOGG::MojoValue local = state.file->var(
                MOGG::MojoValue(llvm::formatv("{0}(take_slot={1})",
                                              MOGG::kStructAnyAsyncValueRef,
                                              slotPtr.text)
                                    .str()),
                /*cache=*/false);
            state.setMojoValueFor(argument, local);

            // The unpacked result is an owning handle to the backing AsyncValue
            // and must be kept alive for the duration of the compiled function.
            state.markAsKeepAlive(local);
          } else {
            state.setMojoValueFor(argument, slotPtr);
          }
        }
        // Return values are DPS to the runtime but value-semantic inside the
        // kernel, so they get a mapping for mogg.return and mogg.async.pack.
        uint64_t numArguments = getBodyBlock().getNumArguments();
        for (uint64_t i = 0; i < getNumResults(); ++i) {
          state.mojoReturnValues.emplace_back(
              runtimeArgument.pointerOffsetGet(i + numArguments));
        }

        // Constant resources follow in the void** array as ptr + size pairs.
        for (uint64_t i = 0; i < getConstantResources().size(); ++i) {
          uint64_t flatIndex = i * 2 + numArguments + getNumResults();
          MOGG::MojoValue ptr = runtimeArgument.pointerOffsetGet(flatIndex);
          // The ABI is that the bytecount is directly stored as an
          // OpaquePointer.
          MOGG::MojoValue byteCount = OpaquePointerRef(runtimeArgument)
                                          .bitcast(MojoValue(kStructInt))()
                                          .pointerOffsetGet(flatIndex + 1);
          state.mojoConstantResources.emplace_back(
              std::tuple<MOGG::MojoValue, MOGG::MojoValue>{ptr, byteCount});
        }

        // void** ABI entry-point functions are never `raises`, so we cannot
        // use `with Trace[...]` here (Trace.__enter__ is raises). Tracing
        // is handled in the MOGG ABI function this entry point calls.
        errOccurred = llvm::any_of(getOps(), [&](mlir::Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
        if (!errOccurred)
          state.genAllKeepAliveStatements();
      });

  if (errOccurred)
    return failure();
  return success();
}

LogicalResult FuncOp::emitMojoMOGGABI(EmitMojoState &state) {
  // The MOGG-ABI stub takes raw runtime handles (a `!mogg.pointer` per input
  // then output, the dynamic `!mogg.dim`s, a `!mogg.context`) and emits
  // straight-line Mojo, with no async wrapping or `void**` marshalling.
  // `mlirTypeToMojoType` renders an opaque as a bare `OpaquePointer`, but here
  // it crosses as the kernel's own struct (`SIMDPair[4, 8]`).
  auto typeToMojo = [&](Type type) -> MOGG::MojoValue {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(type)) {
      MOGG::MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
      auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
      return params.empty() ? baseTy : baseTy[params];
    }
    return state.mlirTypeToMojoType(type);
  };

  SmallVector<MOGG::NamedMojoDef> inputTypes;
  for (BlockArgument arg : getBody().getArguments())
    inputTypes.push_back(MOGG::NamedMojoDef("arg", typeToMojo(arg.getType())));

  // A shape function's shape, or a kernel-allocated opaque, hands back a value.
  MOGG::OutputOp bodyOutput = getBodyTerminator();
  ASSERT_STREAM(bodyOutput.getNumOperands() <= 1,
                << "a MOGG-ABI stub returns at most one value, got "
                << bodyOutput.getNumOperands());
  const bool returnsByValue = bodyOutput.getNumOperands() == 1;
  std::optional<MOGG::MojoValue> returnType;
  if (returnsByValue)
    returnType = typeToMojo(bodyOutput.getOperand(0).getType());

  bool errOccurred = false;
  state.file->def(
      getSymNameAttr().str(), /*parameters=*/{}, inputTypes, returnType,
      /*exported=*/false,
      /*raises=*/isThrows(), /*isAlwaysInline=*/false,
      [&](const MOGG::FunctionBody &functionBody) {
        for (auto &&[mlirArg, mojoArg] :
             llvm::zip(getBody().getArguments(), functionBody.arguments))
          state.setMojoValueFor(mlirArg, mojoArg.resolvedName);

        errOccurred = llvm::any_of(getOps(), [&](mlir::Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
        if (!errOccurred && returnsByValue) {
          MOGG::MojoValue value =
              state.getMojoValueFor(bodyOutput.getOperand(0));
          // An opaque struct need not be `ImplicitlyCopyable` (`SIMDPair` is
          // only `RegisterPassable`), so transfer rather than copy it out.
          if (isa<MO::OpaqueType>(bodyOutput.getOperand(0).getType()))
            value = value.transfer();
          state.file->returnsValue(value);
        }
      });

  return failure(errOccurred);
}

//===----------------------------------------------------------------------===//
// Advanced-fusion codegen op emitters (MOGG ABI)
//===----------------------------------------------------------------------===//

LogicalResult ShapeCreateOp::emitMojo(EmitMojoState &state) {
  // Encode the shape as a `Coord` -- static dims as compile-time `Idx[N]`,
  // dynamic dims as runtime `Scalar[DType.int](<dim>)` sourced from the
  // `!mogg.dim` operands. `tensor.create` builds a row-major `Layout` from it.
  MojoValue coord =
      state.getShapeAsCoordMojoValue(getShape(), getDynamicDims());
  state.setMojoValueFor(getResult(), state.file->var(coord));
  return success();
}

LogicalResult TensorCreateOp::emitMojo(EmitMojoState &state) {
  // The rank comes from the op's own static shape attr; the opaque
  // `!mogg.shape` handle carries no attributes of its own.
  MOSH::ShapeAttr shapeAttr = getShapeAttr();

  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getElementType());
  ASSERT_STREAM(!dtypeOr.isError(),
                "Failed to extract dtype from tensor.create element type");
  MojoValue dTypeParam = state.mlirDTypeToMojoType(*dtypeOr);
  MojoValue rankParam(static_cast<uint64_t>(
      shapeAttr.getRank() == 0 ? 1 : shapeAttr.getRank()));
  auto mutParam = getMut();
  auto ioParam = MojoValue(kStructIO).get(MO::stringifyIOKind(getIo()));
  // Device kind (cpu vs gpu) is on the op's own `$device` attr, so the
  // preferred alignment is picked locally without reaching out to the stub.
  MojoValue alignmentParam(static_cast<uint64_t>(
      M::Driver::getPreferredAlignment(getDevice().str())));

  // Build a row-major `Layout` from the shape Coord produced by `shape.create`.
  auto layoutVar = state.file->var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));

  // Reuse the legacy `mogg.tensor.__init__` primitives: the two ops build the
  // same `ManagedTensorSlice` (or, for a slot-backed operand, the same
  // `StableTensor` over the slot address).
  StringRef primitiveName = getStable()
                                ? StringRef(kMoggPrimitiveStableTensorInit)
                                : TensorInitOp::getOperationName();
  auto tensorInitPrimDecl = state.registry.lookupPrimitive(primitiveName);
  ASSERT_STREAM(tensorInitPrimDecl,
                "Primitive not found for " << primitiveName.str());
  MojoValue moggTensorInit(primitiveMojoName(tensorInitPrimDecl));
  state.setMojoValueFor(
      getTensor(),
      state.file->var(moggTensorInit(
          {dTypeParam, rankParam, mutParam, ioParam, alignmentParam},
          {OpaquePointerRef(state.getMojoValueFor(getPtr()))
               .bitcast(MojoValue(kComptimeDTypeNoneType))(),
           layoutVar})));

  return success();
}

LogicalResult TensorCreateSliceOp::emitMojo(EmitMojoState &state) {
  // Mojo infers dtype/rank/start_type/stop_type/step_type from the argument
  // types. output_static_shape/static_starts/static_steps carry what is
  // statically known about the view, from this op's attributes.
  auto sliceCreatePrimDecl =
      state.registry.lookupPrimitive(TensorCreateSliceOp::getOperationName());
  ASSERT_STREAM(sliceCreatePrimDecl,
                "Primitive not found for "
                    << TensorCreateSliceOp::getOperationName());
  MojoValue primitive(primitiveMojoName(sliceCreatePrimDecl));
  MojoValue outputStaticShape = state.file->comptime(
      EmitMojoState::getAsCoordMojoValue(getOutShape().getIntDims()));
  MojoValue staticStarts = state.file->comptime(
      EmitMojoState::getAsCoordMojoValue(getStaticStart()));
  MojoValue staticSteps =
      state.file->comptime(EmitMojoState::getAsCoordMojoValue(getStaticStep()));
  MojoValue result = state.file->var(primitive(
      {MojoValue(kParameterOutputStaticShape).equal(outputStaticShape),
       MojoValue(kParameterStaticStarts).equal(staticStarts),
       MojoValue(kParameterStaticSteps).equal(staticSteps)},
      {state.getMojoValueFor(getInput()), state.getMojoValueFor(getStarts()),
       state.getMojoValueFor(getStops()), state.getMojoValueFor(getSteps())}));
  state.setMojoValueFor(getResult(), result);
  return success();
}

// Prints a view whose layout comes from its output shape at runtime:
//
//   comptime static_shape = Coord(...)                                  (1)
//   var shape = coord_to_index_list(<$runtimeOutShape>)                 (2)
//   var view = <primitive>[output_static_shape = static_shape](
//       <$input>, shape)                                                (3)
//
// Mojo infers dtype/rank from $input. A broadcast reads its strides from the
// input but its runtime dims from `$runtimeOutShape`, which a slice composed on
// top clamps against; a reshape also de-linearizes its strides from it.
template <typename OpT>
static LogicalResult emitShapedView(OpT op, EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(OpT::getOperationName());
  ASSERT_STREAM(primDecl,
                "Primitive not found for " << OpT::getOperationName());
  MojoValue primitive(primitiveMojoName(primDecl));
  // (1)
  MojoValue outputStaticShape = state.file->comptime(
      EmitMojoState::getAsCoordMojoValue(op.getStaticOutShape().getIntDims()));
  // (2)
  MojoValue outputShapeValue = state.file->var(MojoValue(kFnCoordToIndexList)(
      {}, {state.getMojoValueFor(op.getRuntimeOutShape())}));
  // (3)
  MojoValue result = state.file->var(primitive(
      {MojoValue(kParameterOutputStaticShape).equal(outputStaticShape)},
      {state.getMojoValueFor(op.getInput()), outputShapeValue}));
  state.setMojoValueFor(op.getResult(), result);
  return success();
}

LogicalResult TensorCreateBroadcastOp::emitMojo(EmitMojoState &state) {
  return emitShapedView(*this, state);
}

LogicalResult TensorCreateReshapeOp::emitMojo(EmitMojoState &state) {
  return emitShapedView(*this, state);
}

LogicalResult IndexReshapeOp::emitMojo(EmitMojoState &state) {
  // Per-index reshape (store side, inside an epilogue functor): convert the
  // `Coord` index to an `IndexList`, remap it to the output shape preserving
  // the row-major offset via `mogg.index.reshape`, and wrap it back into a
  // `Coord` for `mogg.tensor.store`. The from/to shapes ride as runtime
  // `Coord` args: static dims as `Idx[N]` components, dynamic dims' real
  // values as epilogue shape captures (`$dynamicShapes`), or an all-static
  // `Coord` built here when the shape is static.
  auto primDecl =
      state.registry.lookupPrimitive(IndexReshapeOp::getOperationName());
  ASSERT_STREAM(primDecl, "Primitive not found for "
                              << IndexReshapeOp::getOperationName());
  MojoValue primitive(primitiveMojoName(primDecl));

  // Inside the epilogue functor, `output()` is the functor body being
  // emitted, as for `TensorLoadOp`/`TensorStoreOp`.
  MojoBuilder &builder = *state.file;

  // The index carries the store method's abstract rank param, which can't
  // unify with the primitive's concrete from-rank; rebind it (equal at
  // every instantiation).
  MojoValue idxList = builder.var(
      MojoValue(kFnCoordToIndexList)({}, {state.getMojoValueFor(getIndex())})
          .rebind(MojoValue(kStructIndexList)[MojoValue(
              uint64_t(getFromShape().getRankNonZero()))]));

  SmallVector<MojoValue> primArgs = {idxList};
  if (getDynamicShapes().empty()) {
    // A rank-0 shape is emitted as rank-1 extent-1, here as everywhere else,
    // so the ranks must follow: a `to_rank` of 0 contradicts its own shape.
    primArgs.push_back(
        builder.var(state.getShapeAsCoordMojoValue(getFromShape(), {})));
    primArgs.push_back(
        builder.var(state.getShapeAsCoordMojoValue(getToShape(), {})));
  } else {
    // `$dynamicShapes` is [from-shape, to-shape] (see the MAPToMOGG
    // lowering): functor shape-capture fields or kernel-scope `!mogg.shape`
    // Coords, both already `Coord` values.
    primArgs.push_back(state.getMojoValueFor(getDynamicShapes()[0]));
    primArgs.push_back(state.getMojoValueFor(getDynamicShapes()[1]));
  }

  MojoValue newIdxList =
      builder.var(primitive({MojoValue("to_rank").equal(MojoValue(
                                uint64_t(getToShape().getRankNonZero())))},
                            primArgs));
  MojoValue newCoord = builder.var(MojoValue(kStructCoord)({}, {newIdxList}));
  state.setMojoValueFor(getResult(), newCoord);
  return success();
}

LogicalResult IndexSliceSelectOp::emitMojo(EmitMojoState &state) {
  // Store-side and per index, like `IndexReshapeOp` above, whose conventions
  // this follows: statements go through the functor's own builder. The static
  // bounds ride as comptime `Coord` params so the primitive folds every
  // unsliced dim; the in/out shapes ride as runtime `Coord` args -- the
  // functor's shape-capture fields, or in a foreach the kernel-scope shapes
  // its closure captures, or an all-static `Coord` built here. The body is
  // emitted where the primitive's in-slice bit is set, with its argument
  // bound to the output index.
  auto primDecl =
      state.registry.lookupPrimitive(IndexSliceSelectOp::getOperationName());
  ASSERT_STREAM(primDecl, "Primitive not found for "
                              << IndexSliceSelectOp::getOperationName());
  MojoValue primitive(primitiveMojoName(primDecl));

  // Inside the epilogue functor, `output()` is the functor body being
  // emitted.
  MojoBuilder &builder = *state.file;

  // The index carries the store method's abstract rank param, which can't
  // unify with the primitive's concrete rank; rebind it (equal at every
  // instantiation).
  MojoValue idxList = builder.var(
      MojoValue(kFnCoordToIndexList)({}, {state.getMojoValueFor(getIndex())})
          .rebind(MojoValue(kStructIndexList)[MojoValue(
              uint64_t(getInShape().getRankNonZero()))]));
  SmallVector<MojoValue> args = {idxList};
  // The shapes are already `Coord` values: functor shape-capture fields, or
  // kernel-scope `!mogg.shape` Coords a foreach closure captures. A static
  // shape (no captures) is built here.
  if (getDynamicShapes().empty()) {
    args.push_back(
        builder.var(state.getShapeAsCoordMojoValue(getInShape(), {})));
    args.push_back(
        builder.var(state.getShapeAsCoordMojoValue(getOutShape(), {})));
  }
  for (Value shape : getDynamicShapes())
    args.push_back(state.getMojoValueFor(shape));

  auto comptimeTuple = [&](ArrayRef<int64_t> values) {
    return builder.comptime(EmitMojoState::getAsCoordMojoValue(values));
  };
  MojoValue inverse = builder.var(primitive(
      {MojoValue(kParameterStaticStarts).equal(comptimeTuple(getStaticStart())),
       MojoValue(kParameterStaticSteps).equal(comptimeTuple(getStaticStep()))},
      args));
  Block &body = getBody().front();
  state.setMojoValueFor(body.getArgument(0),
                        builder.var(MojoValue(kStructCoord)(
                            {}, {inverse[MojoValue(uint64_t(0))]})));
  MojoValue inSlice = builder.var(inverse[MojoValue(uint64_t(1))]);

  // The body runs on the false edge of `not in_slice`, the branch sense of an
  // early-return guard. LLVM keeps that sense into the PTX, and with the body
  // on the true edge instead, ptxas spent an extra register and extra moves on
  // the KV-cache window store.
  bool failed = false;
  builder.if_(MojoValue("not " + inSlice.text), [&] { builder.pass(); })
      .else_([&] {
        failed = llvm::any_of(body, [&](Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
      });
  return failure(failed);
}

LogicalResult TensorCreateTransposeOp::emitMojo(EmitMojoState &state) {
  // Mojo infers dtype/rank/perm_type from $input/$perm.
  // output_static_shape/static_permutations carry the compile-time-known parts
  // (`?` where runtime-only); the runtime `$perm` supplies the rest, and a
  // fully static perm never reads it.
  auto transposeCreatePrimDecl = state.registry.lookupPrimitive(
      TensorCreateTransposeOp::getOperationName());
  ASSERT_STREAM(transposeCreatePrimDecl,
                "Primitive not found for "
                    << TensorCreateTransposeOp::getOperationName());
  MojoValue primitive(primitiveMojoName(transposeCreatePrimDecl));
  MojoValue outputStaticShape = state.file->comptime(
      EmitMojoState::getAsCoordMojoValue(getOutShape().getIntDims()));
  MojoValue staticPermutations = state.file->comptime(
      EmitMojoState::getAsCoordMojoValue(getStaticPermutations()));
  MojoValue result = state.file->var(primitive(
      {MojoValue(kParameterOutputStaticShape).equal(outputStaticShape),
       MojoValue(kParameterStaticPermutations).equal(staticPermutations)},
      {state.getMojoValueFor(getInput()),
       state.getMojoValueFor(getPermutations())}));
  state.setMojoValueFor(getResult(), result);
  return success();
}
namespace {} // namespace

// Checks a declarative call and infers the parameters of both entry points.
// It emits nothing itself: its callees emit
//
//   Kernel[...].execute_configuration[...](operand0, operand1, ...)
//   Kernel[...].execute_device[...](operand0, operand1, ...)
static FailureOr<DeclarativeCall>
prepareDeclarativeCall(Operation *op, EmitMojoState &state, ValueRange outputs,
                       ValueRange inputs, KernelDeclAdaptor &adaptor,
                       DictionaryAttr kernelParams) {
  DeclarativeCall call;
  SmallVector<Value> operandValues(outputs.begin(), outputs.end());
  llvm::append_range(operandValues, inputs);
  for (Value value : operandValues) {
    if (!isa<MOGG::TensorType, MOGG::BufferType, MOGG::ScalarType>(
            value.getType())) {
      op->emitOpError()
          << "a declarative launch passes only tensors, buffers and scalars "
             "to the device, but got "
          << value.getType();
      return failure();
    }
    if (isa<MOGG::ScalarType>(value.getType())) {
      call.kinds.push_back(DeviceArgKind::Scalar);
      continue;
    }
    // A fused operand against a trait bound carries the contract that makes
    // its create op build a device-passable trait tensor.
    Operation *def = value.getDefiningOp();
    bool traitFused = false;
    if (auto epilogue = dyn_cast_or_null<TensorCreateEpilogueOp>(def))
      traitFused = static_cast<bool>(epilogue.getContract());
    else if (auto prologue = dyn_cast_or_null<TensorCreatePrologueOp>(def))
      traitFused = static_cast<bool>(prologue.getContract());
    if (traitFused) {
      call.kinds.push_back(DeviceArgKind::Fused);
      continue;
    }
    if (isa_and_nonnull<TensorCreateEpilogueOp, TensorCreatePrologueOp,
                        TileCreateEpilogueOp>(def)) {
      op->emitOpError("a declarative launch passes a fused tensor only to a "
                      "trait-bound operand");
      return failure();
    }
    call.kinds.push_back(DeviceArgKind::Tile);
  }
  call.operands = llvm::map_to_vector(
      operandValues, [&](Value value) { return state.getMojoValueFor(value); });
  call.values = std::move(operandValues);

  FailureOr<KernelCallee> executeConfiguration = resolveKernelCallee(
      op, state, adaptor, KernelFunctionKind::ExecuteConfiguration,
      kernelParams);
  FailureOr<KernelCallee> executeDevice = resolveKernelCallee(
      op, state, adaptor, KernelFunctionKind::ExecuteDevice, kernelParams);
  if (failed(executeConfiguration) || failed(executeDevice))
    return failure();
  call.executeConfiguration = std::move(*executeConfiguration);
  call.executeDevice = std::move(*executeDevice);
  return call;
}

// Emits a wrapper for `execute_device` (named `execute_device_N` below). The
// launch passes each operand the host built, and encodes it as its device type.
// A tensor's device type is its `TileTensor` projection, which carries no
// `Input`/`Output` role, so the wrapper rebuilds the operand from it and
// `execute_device` sees the operand types and roles the host built. A fused
// tensor has no single buffer to project: it holds a fusion functor over the
// tensors it captures. Its device type is the access-enabled view on which the
// fusion runs, which `execute_device` takes as is. `body` emits the kernel's
// statements:
//
//   def execute_device_N(a0: T0.device_type, a1: T1, a2: T2.device_type):
//       Kernel[...].execute_device[...](T0(a0), a1, a2)
static MojoValue
emitDeviceKernel(MojoBuilder &builder, const DeviceKernelArgs &args,
                 std::function<void(const FunctionBody &)> &&body) {
  return builder.lambda("execute_device_", /*parameters=*/{}, args.parameters,
                        /*returnType=*/std::nullopt,
                        MojoBuilder::LambdaInline::kAutomatic,
                        /*isAsync=*/false, /*raises=*/false,
                        /*captureConvention=*/"", std::move(body));
}

// Emits the launch of `kernel` on `ctx`. `mogg_launch_declarative` compiles
// it and enqueues it with `config`'s grid and block dims:
//
//   mogg_launch_declarative[execute_device_N](ctx, config, operand0, operand1)
static void emitLaunch(EmitMojoState &state, const MojoValue &ctx,
                       const MojoValue &kernel, const DeviceKernelArgs &args,
                       const MojoValue &config) {
  SmallVector<MojoValue> launchArgs{ctx, config};
  llvm::append_range(launchArgs, args.launchArgs);
  state.file->call(MojoValue(kFnMoggLaunchDeclarative), {kernel}, launchArgs);
  state.genAllKeepAliveStatements();
}

// Emits the launch of a declarative kernel. `execute_configuration` returns a
// `GridBlockExecutionConfig`, which provides the launch's grid and block dims:
//
//   var config = Kernel[...].execute_configuration[...](operands...)
//   comptime T0 = type_of(operand0)  # one per operand
//   def execute_device_N(a0: T0.device_type, a1: T1, ...):
//       Kernel[...].execute_device[...](T0(a0), a1, ...)
//   mogg_launch_declarative[execute_device_N](ctx, config, operand0,
//                                             operand1, ...)
static LogicalResult emitDeclarativeLaunch(CallOpaqueOp op,
                                           EmitMojoState &state,
                                           KernelDeclAdaptor &adaptor) {
  if (op.getDeviceContexts().size() != 1)
    return op.emitOpError("a declarative launch takes one device context");
  FailureOr<DeclarativeCall> call =
      prepareDeclarativeCall(op, state, op.getOutputs(), op.getInputs(),
                             adaptor, op.getKernelParamAttr());
  if (failed(call))
    return failure();

  MojoBuilder &builder = *state.file;
  MojoValue ctx = state.getMojoValueFor(op.getDeviceContexts().front());
  MojoValue config =
      builder.var(call->executeConfiguration(call->operands), /*cache=*/false);
  DeviceKernelArgs args;
  SmallVector<unsigned> indices = args.add(builder, *call);
  MojoValue kernel =
      emitDeviceKernel(builder, args, [&](const FunctionBody &body) {
        builder.call(call->executeDevice.function,
                     call->executeDevice.parameters,
                     args.rebuild(body, indices));
      });
  emitLaunch(state, ctx, kernel, args, config);
  return success();
}

// Emits a call to the kernel's `execute`, which launches the kernel itself:
//
//   Kernel[...].execute[...](output0, input0, ctx)
//
// or, for a kernel that returns an `!mo.opaque`, binds the result:
//
//   var result = Kernel[...].execute[...](input0, ctx)
static LogicalResult emitExecuteCall(CallOpaqueOp op, EmitMojoState &state,
                                     KernelDeclAdaptor &adaptor) {
  // DPS: outputs, then inputs, then `$device_contexts` (wired by `MAPToMOGG`)
  // to match the execute signature's trailing `ctx`. For a distributed kernel
  // `ConformCallOperands` packed it into one tuple that emits as a
  // `DeviceContextArray`.
  SmallVector<MojoValue> operands;
  for (Value out : op.getOutputs())
    operands.push_back(state.getMojoValueFor(out));
  for (Value in : op.getInputs())
    operands.push_back(state.getMojoValueFor(in));
  for (Value ctx : op.getDeviceContexts())
    operands.push_back(state.getMojoValueFor(ctx));

  FailureOr<KernelCallee> execute = resolveKernelCallee(
      op, state, adaptor, KernelFunctionKind::Execute, op.getKernelParamAttr());
  if (failed(execute))
    return failure();

  // A kernel returning an `!mo.opaque` returns it by value, so bind the call's
  // result to a var the enclosing `mogg.output` can forward, instead of
  // emitting the call as a bare statement.
  if (Value opaqueRes = op.getOpaqueResult())
    state.setMojoValueFor(opaqueRes, state.file->var((*execute)(operands)));
  else
    state.file->call(execute->function, execute->parameters, operands);
  state.genAllKeepAliveStatements();
  return success();
}

LogicalResult CallOpaqueOp::emitMojo(EmitMojoState &state) {
  KernelDeclAdaptor adaptor(state.registry.lookupKernelBySymName(getCallee()));
  // A call lowers one of two ways. A declarative kernel gets the launch the
  // graph compiler writes; any other kernel is a call to its `execute`, which
  // launches it itself. Both resolve their callees through
  // `resolveKernelCallee`, so they infer parameters the same way, and each owns
  // the keep-alives that must follow its launch. In a graph blob,
  // `mogg.call.execute_device` calls a declarative kernel instead.
  if (adaptor.isDeclarative())
    return emitDeclarativeLaunch(*this, state, adaptor);
  return emitExecuteCall(*this, state, adaptor);
}

// Emits the call's `execute_configuration` and adds the call to the enclosing
// blob, which launches it.
LogicalResult CallExecuteDeviceOp::emitMojo(EmitMojoState &state) {
  KernelDeclAdaptor adaptor(state.registry.lookupKernelBySymName(getCallee()));
  FailureOr<DeclarativeCall> call = prepareDeclarativeCall(
      *this, state, getOutputs(), getInputs(), adaptor, getKernelParamAttr());
  if (failed(call))
    return failure();

  // The call's parent blob pushed its context before emitting its body, and
  // the blob's verifier gave every call the same context.
  GraphBlobRegion &blob = state.currentRegion<GraphBlobRegion>();
  blob.ctx = getDeviceContext();
  MojoBuilder &builder = *state.file;
  MojoValue config =
      builder.var(call->executeConfiguration(call->operands), /*cache=*/false);
  SmallVector<unsigned> indices = blob.args.add(builder, *call);
  blob.members.push_back({std::move(*call), std::move(indices), config});
  return success();
}

// Emits the body's declarative calls as one launch on every core and thread of
// the device, the `all_cores` of the first call's configuration type. Each call
// runs its `execute_device` only on the first blocks of the launch, as many as
// its own grid has:
//
//   var config0 = Kernel0[...].execute_configuration[...](operands0...)
//   var config1 = Kernel1[...].execute_configuration[...](operands1...)
//   comptime T0 = type_of(operand0)  # one per distinct operand
//   comptime C = type_of(config0)
//   var launch = C.all_cores()
//   def execute_device_N(a0: T0.device_type, ..., a5: Int32, a6: Int32):
//       if _graph_blob.is_thread_active(a5):
//           Kernel0[...].execute_device[...](T0(a0), ...)
//       _graph_blob.barrier()
//       if _graph_blob.is_thread_active(a6):
//           Kernel1[...].execute_device[...](...)
//   mogg_launch_declarative[execute_device_N](
//       ctx, launch, operand0, ...,
//       _graph_blob.active_threads(config0.get_grid_dim(),
//                                  config0.get_block_dim(),
//                                  launch.get_grid_dim(),
//                                  launch.get_block_dim()), ...)
//
// Every thread reaches every barrier, so a barrier sits outside the guards.
LogicalResult GraphBlobOp::emitMojo(EmitMojoState &state) {
  // Each body op emits through `lowerOpToMojo`, which passes it only the state,
  // so the state carries the blob for each call to add itself to. The scope
  // pops it on every return; nothing needs the outer value back, because the
  // verifier keeps a blob out of another.
  auto blobScope = state.enterRegion<GraphBlobRegion>();
  for (Operation &op : getBody().getOps())
    if (failed(state.lowerOpToMojo(&op)))
      return failure();
  GraphBlobRegion &blob = state.currentRegion<GraphBlobRegion>();
  ASSERT_STREAM(!blob.members.empty(), "the verifier requires a call");

  MojoBuilder &builder = *state.file;
  MojoValue graphBlob(kGraphBlobModuleAlias);
  MojoValue configType =
      builder.comptime(MojoValue("type_of")({}, {blob.members.front().config}));
  MojoValue launch =
      builder.var(configType.get("all_cores")(), /*cache=*/false);

  SmallVector<unsigned> active;
  for (const GraphBlobRegion::Member &member : blob.members) {
    MojoValue threads = graphBlob.get("active_threads")(
        {}, {member.config.get("get_grid_dim")(),
             member.config.get("get_block_dim")(), launch.get("get_grid_dim")(),
             launch.get("get_block_dim")()});
    active.push_back(blob.args.addScalar(threads, MojoValue("Int32")));
  }

  MojoValue kernel =
      emitDeviceKernel(builder, blob.args, [&](const FunctionBody &body) {
        for (auto [i, member] : llvm::enumerate(blob.members)) {
          if (i > 0)
            builder.call(graphBlob.get("barrier"));
          MojoValue isActive = graphBlob.get("is_thread_active")(
              {}, {body.arguments[active[i]].resolvedName});
          const KernelCallee &executeDevice = member.call.executeDevice;
          builder.if_(isActive, [&] {
            builder.call(executeDevice.function, executeDevice.parameters,
                         blob.args.rebuild(body, member.indices));
          });
        }
      });
  emitLaunch(state, state.getMojoValueFor(blob.ctx), kernel, blob.args, launch);
  return success();
}

LogicalResult ScalarCreateOp::emitMojo(EmitMojoState &state) {
  const MojoValue &input = state.getMojoValueFor(getTensor());
  // Reuse the `mogg.as_scalar` primitive: both pack a single-element
  // tensor operand into the scalar the kernel expects.
  static constexpr llvm::StringLiteral kAsScalarPrimitive = "mogg.as_scalar";
  auto asScalarPrimDecl = state.registry.lookupPrimitive(kAsScalarPrimitive);
  ASSERT_STREAM(asScalarPrimDecl,
                "Primitive not found for " << kAsScalarPrimitive);
  MojoValue moggAsScalar(primitiveMojoName(asScalarPrimDecl));
  MojoValue scalarValue = moggAsScalar(ArrayRef<MojoValue>{}, {input});
  state.setMojoValueFor(getScalar(), state.file->var(scalarValue));
  return success();
}
namespace {

/// The aggregate a variadic run of tensor operands packs into.
enum class VariadicPackKind { Plain, Stable, FusedInput, FusedOutput };

/// Emits the aggregate for a variadic run of tensor values. A pointer-backed
/// element contributes its data pointer, a slot-backed (`StableTensor`)
/// element its slot address; shapes, static specs and `io_spec` are spelled
/// identically on both. `fusionValues` / `fusionTypeParams` are consumed only
/// by the fused kinds (one functor and its type per element).
MojoValue emitVariadicTensorPack(EmitMojoState &state,
                                 ArrayRef<MojoValue> inputValues,
                                 VariadicPackKind kind,
                                 ArrayRef<MojoValue> fusionValues,
                                 ArrayRef<MojoValue> fusionTypeParams) {
  const bool stable = kind == VariadicPackKind::Stable;

  SmallVector<MojoValue> runtimeStorage =
      llvm::map_to_vector(inputValues, [&](const MojoValue &val) {
        return stable ? StableTensorRef(val).addr()
                      : ManagedTensorSliceRef(val).rawPtr();
      });
  SmallVector<MojoValue> runtimeShapes =
      llvm::map_to_vector(inputValues, [&](const MojoValue &val) {
        return stable ? StableTensorRef(val).shape()
                      : ManagedTensorSliceRef(val).shape();
      });

  MojoValue inputTupleConstructor = MojoValue(kStructStaticTuple)[{
      MojoValue(kParameterSize).equal(std::to_string(inputValues.size()))}];
  MojoValue runtimeStorageTuple =
      state.file->var(inputTupleConstructor({}, runtimeStorage));
  MojoValue runtimeShapesTuple =
      state.file->var(inputTupleConstructor({}, runtimeShapes));

  // Create the StaticTensorSpecList from per-element internals, shapes, and
  // strides. Each spec is split into its homogeneous runtime part
  // (StaticTensorSpecInternal) and its compile-time shape/strides so that
  // heterogeneous specs can be stored in a single list. `shape_tuple` /
  // `strides_tuple` are comptime members of the `StaticTensorSpec` itself,
  // distinct from its `get_internals()` result.
  auto staticSpecOf = [&](const MojoValue &val) {
    return StaticTensorSpecRef(stable
                                   ? StableTensorRef(val).staticSpec()
                                   : ManagedTensorSliceRef(val).staticSpec());
  };
  MojoValue rank = staticSpecOf(inputValues.front()).rank();
  SmallVector<MojoValue> staticSpecInternals;
  SmallVector<MojoValue> staticSpecShapes;
  SmallVector<MojoValue> staticSpecStrides;
  for (const auto &reboundInput : inputValues) {
    StaticTensorSpecRef spec = staticSpecOf(reboundInput);
    staticSpecInternals.push_back(spec.getInternals());
    staticSpecShapes.push_back(
        MojoValue(kFnCoordToIndexList)({}, {spec.shapeTuple()}));
    staticSpecStrides.push_back(
        MojoValue(kFnCoordToIndexList)({}, {spec.stridesTuple()}));
  }
  MojoValue inputSpecsInternalsComptime =
      state.file->comptime(ParameterListRef::of(staticSpecInternals));
  MojoValue inputSpecsShapesComptime =
      state.file->comptime(ParameterListRef::of(staticSpecShapes));
  MojoValue inputSpecsStridesComptime =
      state.file->comptime(ParameterListRef::of(staticSpecStrides));
  MojoValue inputSpecsComptime =
      state.file->comptime(MojoValue(kStructStaticTensorSpecList)[{
          inputSpecsInternalsComptime, inputSpecsShapesComptime,
          inputSpecsStridesComptime}]());
  MojoValue staticSpecsParam =
      MojoValue(kParameterStaticSpecs).equal(inputSpecsComptime);

  if (kind == VariadicPackKind::Plain || stable) {
    MojoValue ioSpec = stable ? StableTensorRef(inputValues[0]).ioSpec()
                              : ManagedTensorSliceRef(inputValues[0]).ioSpec();
    StringRef structName =
        stable ? kStructStableVariadicTensors : kStructVariadicTensors;
    MojoValue ctor = MojoValue(structName)[{
        MojoValue(kParameterIOSpec).equal(ioSpec), staticSpecsParam}];
    return state.file->var(ctor({}, {runtimeStorageTuple, runtimeShapesTuple}));
  }

  // Fused: the functors and their types become a `_FusionPack` +
  // `*FusionTypes`; `static_specs` is keyword-only after the pack.
  MojoValue fusionsTuple =
      state.file->var(MojoValue(kStructMsFusionPack)({}, fusionValues));
  SmallVector<MojoValue> typeParams(fusionTypeParams);
  typeParams.push_back(staticSpecsParam);
  StringRef structName = kind == VariadicPackKind::FusedInput
                             ? kStructMsFusedInputVariadicTensors
                             : kStructMsFusedOutputVariadicTensors;
  return state.file->var(MojoValue(structName)[typeParams](
      {}, {runtimeStorageTuple, runtimeShapesTuple, fusionsTuple}));
}
} // namespace

LogicalResult TupleCreateOp::emitMojo(EmitMojoState &state) {
  // `$variadic_abi` (set by `ConformCallOperands`) selects `VariadicTensors`,
  // `_FusedInputVariadicTensors`, `_FusedOutputVariadicTensors`, or a
  // `TensorTuple` of trait arguments, so this never inspects the producing
  // ops.
  SmallVector<MojoValue> inputValues = state.getMojoValuesFor(getElements());

  // A `TensorTuple`'s elements are already trait arguments; the tuple holds
  // them by value.
  if (getVariadicAbi() == VariadicABI::TensorTuple) {
    SmallVector<MojoValue> params;
    for (const MojoValue &val : inputValues)
      params.push_back(MojoValue("type_of")({}, {val}));
    state.setMojoValueFor(
        getTuple(),
        state.file->var(MojoValue(kStructTensorTuple)(params, inputValues)));
    return success();
  }

  // Contexts pack into a `DeviceContextArray[N, _]` instead, via its variadic
  // constructor -- the per-device contexts of a distributed kernel. The `_`
  // infers the `target` parameter from the first context so contexts with a
  // non-default target are accepted. Each stub context is generic over its own
  // `target`, so the rest are rebound to the first's type for the variadic
  // constructor to unify them. Mirrors the `AsTupleOp::emitMojo` context
  // branch.
  if (isa<MOGG::ContextType>(getElements().front().getType())) {
    MojoValue deviceContextArrayConstructor =
        MojoValue(kStructDeviceContextArray)[{MojoValue(inputValues.size()),
                                              MojoValue("_")}];
    MojoValue firstContextType =
        MojoValue("type_of")({}, {inputValues.front()});
    SmallVector<MojoValue> contexts = {inputValues.front()};
    for (const MojoValue &ctx : llvm::drop_begin(inputValues))
      contexts.push_back(ctx.rebind(firstContextType));
    state.setMojoValueFor(
        getTuple(),
        state.file->var(deviceContextArrayConstructor({}, contexts)));
    return success();
  }

  // Plain: a `VariadicTensors` over the raw pointers/shapes. Stable: a
  // `StableVariadicTensors` over the elements' slot addresses; the elements
  // are `StableTensor`s, built by stable `mogg.tensor.create`s.
  if (getVariadicAbi() == VariadicABI::Plain ||
      getVariadicAbi() == VariadicABI::Stable) {
    VariadicPackKind kind = getVariadicAbi() == VariadicABI::Stable
                                ? VariadicPackKind::Stable
                                : VariadicPackKind::Plain;
    state.setMojoValueFor(getTuple(),
                          emitVariadicTensorPack(state, inputValues, kind,
                                                 /*fusionValues=*/{},
                                                 /*fusionTypeParams=*/{}));
    return success();
  }

  // Fused: gather each element's fusion functor and its type. The functor
  // lives on the element slice itself (`in_fusion` / `out_fusion`), so nothing
  // here reaches back into the producing op.
  bool isInput = getVariadicAbi() == VariadicABI::FusedInput;
  SmallVector<MojoValue> fusionValues;
  SmallVector<MojoValue> typeParams;
  for (const MojoValue &el : inputValues) {
    ManagedTensorSliceRef slice(el);
    fusionValues.push_back(isInput ? slice.inFusion() : slice.outFusion());
    typeParams.push_back(isInput ? slice.inFusionType()
                                 : slice.outFusionType());
  }
  state.setMojoValueFor(getTuple(), emitVariadicTensorPack(
                                        state, inputValues,
                                        isInput ? VariadicPackKind::FusedInput
                                                : VariadicPackKind::FusedOutput,
                                        fusionValues, typeParams));
  return success();
}

// Loads `width` elements of `arg` at `coord`, a `Coord`.
static MojoValue simdLoadFromCapture(const FusedTensorArg &arg,
                                     const MojoValue &width,
                                     const MojoValue &alignment,
                                     const MojoValue &coord) {
  // The pointer-based load wants an `IndexList[rank]`, sized by the capture's
  // own static spec.
  MojoValue indices =
      MojoValue(kFnCoordToIndexList)({}, {coord})
          .rebind(MojoValue(kStructIndexList)[arg.spec.get(kParameterRank)]);
  return MojoValue(kFnMsSimdLoadFromTensorPointer)(
      {MojoValue(kParameterStaticSpec).equal(arg.spec),
       MojoValue(kParameterSimdWidth).equal(width),
       MojoValue(kParameterElementAlignment).equal(alignment)},
      {arg.ptr, arg.shape, arg.strides, indices});
}

LogicalResult TensorLoadOp::emitMojo(EmitMojoState &state) {
  // The enclosing lambda installs its width and alignment, so they are read
  // from the state rather than assumed. Every fusion lane records its
  // receiver's decomposed `ManagedTensorSlice` (ptr/shape/strides + comptime
  // `StaticTensorSpec`) via `setFusedTensorArgFor`, so
  // `simd_load_from_tensor_pointer` honors the spec's static layout
  // (stride-correct for fused views) and alignment.
  const FusedTensorArg &arg = state.getFusedTensorArgFor(getTensor());
  const MojoValue &idx = state.getMojoValueFor(getIndex());
  const ElementwiseLambdaRegion &parameters =
      state.currentRegion<ElementwiseLambdaRegion>();
  MojoValue loaded = state.file->var(simdLoadFromCapture(
      arg, parameters.width(), parameters.alignment(), idx));
  state.setMojoValueFor(getResult(), loaded);
  return success();
}

LogicalResult TensorStoreOp::emitMojo(EmitMojoState &state) {
  // Reached from `TensorCreateEpilogueOp`'s `__call__` and from
  // `CallForeachOp`'s closure, whose comptime parameter lists differ: the
  // enclosing lambda installs the width and alignment it runs at.
  const FusedTensorArg &arg = state.getFusedTensorArgFor(getTensor());
  const MojoValue &idx = state.getMojoValueFor(getIndex());
  const MojoValue &value = state.getMojoValueFor(getValue());
  const ElementwiseLambdaRegion &parameters =
      state.currentRegion<ElementwiseLambdaRegion>();

  // The method's generic `dtype` comptime parameter isn't statically known to
  // equal the store target's own spec dtype, so the stored value needs an
  // explicit rebind to the capture spec's own dtype.
  MojoValue rebound = value.rebind(MojoValue(
      kStructSIMD)[{arg.spec.get(kParameterDType), parameters.width()}]);
  // The loop index arrives as a `Coord`; the pointer-based store wants an
  // `IndexList[rank]`, sized by the capture's own static spec.
  MojoValue indices = MojoValue(kFnCoordToIndexList)({}, {idx}).rebind(
      MojoValue(kStructIndexList)[arg.spec.get(kParameterRank)]);
  state.file->call(
      MojoValue(kFnMsSimdStoreIntoTensorPointer),
      {MojoValue(kParameterStaticSpec).equal(arg.spec),
       MojoValue(kParameterSimdWidth).equal(parameters.width()),
       MojoValue(kParameterElementAlignment).equal(parameters.alignment())},
      {arg.ptr, arg.shape, arg.strides, indices, rebound});
  return success();
}

// Tile ops have no standalone Mojo form: the tile fusion container that holds
// them emits its body as a whole.
LogicalResult TileLoadOp::emitMojo(EmitMojoState &) {
  return emitOpError("is only emitted by its tile fusion container");
}

LogicalResult TileStoreOp::emitMojo(EmitMojoState &) {
  return emitOpError("is only emitted by its tile fusion container");
}

/// One fused-region functor capture, as the enclosing function sees it.
/// `!mogg.tensor` is opaque, so shape / dtype / mutability are read from the
/// op that creates the capture, as in `CallForeachOp::emitMojo`.
struct FusedCaptureInfo {
  /// The capture's `ManagedTensorSlice`, decomposed into device-passable parts:
  /// pointer and runtime shape/strides as enclosing-function locals, and
  /// `spec`, the comptime `StaticTensorSpec`. Load/store go through
  /// `simd_{load_from,store_into}` on these.
  MojoValue ptr;
  MojoValue shape;
  MojoValue strides;
  MojoValue spec;
  /// Literals, baked directly into the functor's field/param types.
  MojoValue dtype;
  MojoValue mut;
};

/// Collects per-capture metadata for a fused-region functor. Baking dtype/mut
/// as literals rather than struct params keeps capture loads statically typed
/// in the method body, so no rebind is needed per load site.
static FailureOr<SmallVector<FusedCaptureInfo>>
collectFusedCaptureInfos(EmitMojoState &state, Operation *op,
                         ValueRange captures) {
  SmallVector<FusedCaptureInfo> infos;
  for (Value cap : captures) {
    // Element type / mutability come from the producing `mogg.tensor.create*`
    // op; everything else from the capture's emitted Mojo value.
    auto create = cap.getDefiningOp<TensorCreateInterface>();
    if (!create)
      return op->emitOpError(
          "capture must be a mogg.tensor.create(.slice|.broadcast|"
          ".transpose|.reshape) result");
    ErrorOr<DType> capDtypeOr = M::MO::fromMLIRType(create.getElementType());
    if (capDtypeOr.isError())
      return op->emitOpError("failed to convert capture dtype to Mojo");

    // Views have already folded their access pattern into the decomposed
    // pointer/shape/strides. `spec` is a comptime alias in the enclosing scope
    // so it is nameable where the functor is instantiated and inside a
    // trivially-capturing foreach closure (where `mts.static_spec` would
    // reference the uncaptured slice).
    ManagedTensorSliceRef capRef(state.getMojoValueFor(cap));
    // `unsafe_ptr()` erases the origin to `MutAnyOrigin`, matching the
    // functor's pointer field and `simd_*_tensor_pointer`'s `ptr` parameter.
    MojoValue ptr = state.file->var(capRef.unsafePtr());
    MojoValue shape = state.file->var(capRef.shape());
    MojoValue strides = state.file->var(capRef.strides());
    MojoValue spec = state.file->comptime(capRef.staticSpec());
    infos.push_back({ptr, shape, strides, spec,
                     state.mlirDTypeToMojoType(*capDtypeOr),
                     MojoValue(create.getMut())});
  }
  return infos;
}

// Builds a `ManagedTensorSlice` over `ptr` with `layout`/`dtype`/`rank`, via
// the `mogg.tensor.__init__` primitive. `ptr` may be a real base pointer (so an
// accumulating kernel can accumulate through it) or a dangling one (for a
// kernel that only reads shape and never touches the pointer).
static MojoValue emitBaseSliceFromPtr(EmitMojoState &state, MojoValue dtype,
                                      unsigned rank, MojoValue layout,
                                      MojoValue alignment, MojoValue ptr) {
  auto tensorInitPrimDecl =
      state.registry.lookupPrimitive(TensorInitOp::getOperationName());
  ASSERT_STREAM(tensorInitPrimDecl,
                "Primitive not found for " << TensorInitOp::getOperationName());
  return state.file->var(MojoValue(primitiveMojoName(tensorInitPrimDecl))(
      {dtype, MojoValue(static_cast<uint64_t>(rank)), MojoValue(false),
       MojoValue(kStructIO).get(MO::stringifyIOKind(MO::IOKind::Input)),
       alignment},
      {OpaquePointerRef(ptr).bitcast(MojoValue(kComptimeDTypeNoneType))(),
       layout}));
}

/// A dangling `ManagedTensorSlice` that only carries a type; every access goes
/// through the fusion functor bound onto it. A dtype-changing epilogue needs
/// it: the kernel gets a slice of its compute dtype (f32), while the real
/// output (si32) reaches the functor as capture 0, since
/// `_bind_to_fused_output` preserves its receiver's dtype.
static MojoValue emitDanglingBaseSlice(EmitMojoState &state, MojoValue dtype,
                                       unsigned rank, MojoValue layout,
                                       MojoValue alignment) {
  MojoValue danglingPtr =
      MojoValue(
          kStructPointer)[{MojoValue(kStructScalar)[{
                               MojoValue(kStructDType).get(kEnumDTypeInt8)}],
                           MojoValue(kComptimeMutAnyOrigin)}]
          .get(OpaquePointerRef::kUnsafeDangling)();
  return emitBaseSliceFromPtr(state, dtype, rank, layout, alignment,
                              danglingPtr);
}

LogicalResult TensorGetPtrOp::emitMojo(EmitMojoState &state) {
  // Read the raw base pointer of the input slice. This op lives in the kernel
  // body (outside any fusion lambda), so `state.file->` is the right scope.
  ManagedTensorSliceRef mts(state.getMojoValueFor(getTensor()));
  state.setMojoValueFor(getResult(), state.file->var(mts.rawPtr()));
  return success();
}

/// The rank a fused-region op's `$shape` operand describes, recovered from the
/// feeding `shape.create` (the opaque `!mogg.shape` handle carries none).
static FailureOr<unsigned> fusedRegionRank(Operation *op, Value shape) {
  auto shapeCreate = shape.getDefiningOp<ShapeCreateOp>();
  if (!shapeCreate)
    return op->emitOpError("shape operand must come from a mogg.shape.create");
  unsigned rank = shapeCreate.getShape().getRank();
  return rank == 0 ? 1 : rank;
}

/// The `Coord` type a fused-region op's `$shape` operand has, spelled from
/// the feeding `shape.create`'s static/dynamic split: static dims as
/// `layout.ComptimeInt[n]`, dynamic ones as `Int`. The functor's shape-
/// capture fields carry it so the reshape/slice-select primitives the body
/// calls see the split in the field's own type.
static FailureOr<MojoValue> fusedRegionShapeCoordType(Operation *op,
                                                      Value shape) {
  auto shapeCreate = shape.getDefiningOp<ShapeCreateOp>();
  if (!shapeCreate)
    return op->emitOpError("shape operand must come from a mogg.shape.create");
  SmallVector<MojoValue> elementTypes;
  for (TypedAttr dim : shapeCreate.getShape().getValues()) {
    if (auto staticDim = MOSH::ShapeAttr::getStaticDim(dim))
      elementTypes.push_back(
          MojoValue("layout.ComptimeInt")[MojoValue(*staticDim)]);
    else
      elementTypes.push_back(MojoValue(kStructInt));
  }
  // Rank-0 shapes are represented as rank-1 extent-1, as everywhere else.
  if (elementTypes.empty())
    elementTypes.push_back(
        MojoValue("layout.ComptimeInt")[MojoValue(int64_t(1))]);
  return MojoValue(kStructCoord)[ArrayRef<MojoValue>(elementTypes)];
}
namespace {

/// One MOGG-ABI fusion-lambda flavor. The load side (`map.iter.prologue`) and
/// the store side (`map.iter.epilogue`) differ only in the trait, the method,
/// and whether it returns a value. Mirrors `emitFusionStruct`'s trait/method
/// selection.
struct FusionLambdaInfo {
  /// Prefix for `getIdentifier`, which mints a unique struct name from it
  /// (`EpilogueStoreFunctor_6`).
  llvm::StringRef structName;
  MojoValue traitName;
  MojoValue methodName;
  /// `OutputFusion` spells the width parameter `SIMDLength`; `InputFusion`
  /// spells it `Int`. The two correlate with `takesValue` across today's
  /// traits, but that is a coincidence of those three, not a rule.
  MojoValue widthParamType;
  /// The store lane is handed the kernel's value; the load lane manufactures
  /// one.
  bool takesValue = false;
  /// The MTS-bind traits (`managed_tensor_slice.InputFusion`/`OutputFusion`)
  /// spell the index `IndexList[rank]`; the trait-wrapper traits
  /// (`InputLoadFusion`/`OutputStoreFusion`/`TransformFusion`) spell it `Coord`
  /// and name the method `__call__`, which then takes `idx: Coord` directly.
  bool coordIndex = false;
};
} // namespace

/// Declares the fusion functor for `region` and returns an instance of it.
/// `captureInfos` is filled in with the functor-side field names as the struct
/// is minted. `valueDtype` is the dtype the incoming `value` is rebound to,
/// and is unused by a lane that takes no value.
static FailureOr<MojoValue>
emitMapFusionFunctor(EmitMojoState &state, Region &region,
                     SmallVectorImpl<FusedCaptureInfo> &captureInfos,
                     const FusionLambdaInfo &lane, MojoValue valueDtype,
                     ArrayRef<MojoValue> shapeCaptureTypes = {},
                     ArrayRef<MojoValue> shapeCaptureArgs = {}) {
  unsigned numCaptures = captureInfos.size();
  unsigned numShapeCaptures = shapeCaptureTypes.size();
  MojoValue structNameVal = state.file.declarations().newVar(
      state.file.declarations().getIdentifier(lane.structName.str()));
  // One comptime `StaticTensorSpec` param per capture with a concrete dtype
  // (so `simd_*_tensor_pointer` infers it) but inferred rank/layout (`_`: a
  // view's symbolic rank wouldn't unify with a literal). Read back as
  // `sb.parameters[j]`.
  SmallVector<NamedMojoDef> structParams;
  for (unsigned j = 0; j < numCaptures; ++j)
    structParams.push_back(NamedMojoDef(
        "spec",
        MojoValue(llvm::formatv("{0}[{1}, _, ...]", kStructStaticTensorSpec,
                                captureInfos[j].dtype.text)
                      .str())));

  // Shape captures (dynamic reshape/slice runtime shapes) become trailing
  // `Coord` fields, typed by each capture's static/dynamic split.
  SmallVector<MojoValue> shapeCaptureFields(numShapeCaptures);

  // Functor-side names per capture: the (ptr, shape, strides) fields and the
  // `Self.`-qualified spec param, minted in the struct callback below and read
  // back in the method body.
  SmallVector<MojoValue> selfPtrFields(numCaptures);
  SmallVector<MojoValue> selfShapeFields(numCaptures);
  SmallVector<MojoValue> selfStridesFields(numCaptures);
  SmallVector<MojoValue> selfSpecParams(numCaptures);

  bool errOccurred = false;
  {
    auto declarations = state.file.enterDeclarations();
    state.file.declarations().structDecl(
        structNameVal, structParams,
        {lane.traitName, MojoValue("TrivialRegisterPassable")},
        [&](const MojoBuilder::StructBody &sb) {
          for (unsigned j = 0; j < numCaptures; ++j) {
            FusedCaptureInfo &capture = captureInfos[j];
            selfSpecParams[j] = MojoValue("Self").get(sb.parameters[j]);
            // Shape/strides fields are sized by the spec's own rank, so they
            // unify with the runtime `shape()`/`strides()` instance args (both
            // from the same slice) -- symbolic rank and all.
            MojoValue indexListType = MojoValue(
                kStructIndexList)[selfSpecParams[j].get(kParameterRank)];
            MojoValue scalarType = MojoValue(kStructScalar)[{capture.dtype}];
            // A `MutAnyOrigin` pointer field (mutable-any, matching
            // `simd_*_tensor_pointer`'s `ptr` parameter) needs the legacy
            // any-origin opt-in the Mojo parser otherwise rejects.
            state.file.declarations().verbatim(
                MojoValue("@__allow_legacy_any_origin_fields"));
            selfPtrFields[j] =
                state.file.declarations().varDecl(MojoValue(kStructPointer)[{
                    scalarType, MojoValue(kComptimeMutAnyOrigin)}]);
            selfShapeFields[j] =
                state.file.declarations().varDecl(indexListType);
            selfStridesFields[j] =
                state.file.declarations().varDecl(indexListType);
          }
          for (unsigned k = 0; k < numShapeCaptures; ++k)
            shapeCaptureFields[k] =
                state.file.declarations().varDecl(shapeCaptureTypes[k]);

          // The method must match the conformed trait
          // (managed_tensor_slice.mojo) for `_bind_to_fused_*` to accept it.
          // Parameters match positionally; names follow the trait for
          // readability.
          //
          // A `mogg.yield`-terminated body returns that value; a store-lane
          // body returns nothing. Read it off the region so it can't disagree
          // with the lane.
          bool returnsValue = isa<YieldOp>(region.front().back());
          MojoValue simdType = MojoValue(kStructSIMD)[{
              MojoValue(kParameterDType), MojoValue(kParameterSimdWidth)}];
          SmallVector<NamedMojoDef> methodParameters = {
              NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
              NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
              NamedMojoDef::asVerbatim(kParameterSimdWidth,
                                       lane.widthParamType),
              NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                       MojoValue(1)),
          };
          SmallVector<NamedMojoDef> methodArgs = {
              NamedMojoDef::asVerbatim(kParameterSelf),
              NamedMojoDef::asVerbatim(
                  kParameterIdx,
                  lane.coordIndex
                      ? MojoValue(kStructCoord)
                      : MojoValue(kStructIndexList)[kParameterRank]),
          };
          if (lane.takesValue)
            methodArgs.push_back(NamedMojoDef::asVerbatim("value", simdType));

          state.file.declarations().def(
              lane.methodName, methodParameters, methodArgs,
              returnsValue ? std::optional<MojoValue>(simdType)
                           : std::optional<MojoValue>{},
              /*exported=*/false, /*raises=*/false,
              /*isAlwaysInline=*/true, [&](const FunctionBody &functionBody) {
                auto parametersScope =
                    state.enterRegion<ElementwiseLambdaRegion>(
                        /*width=*/functionBody.parameter(kParameterSimdWidth),
                        /*alignment=*/functionBody.parameter(
                            kParameterElementAlignment));

                // The stride-aware primitive takes a `Coord`, so convert `idx`
                // once via `Coord`'s `IndexList` constructor (coord.mojo). Not
                // `_index_list_to_static_coord`, which ignores the input at
                // statically shaped positions: right for a shape, wrong for a
                // varying loop index. The trait-wrapper lane already has a
                // `Coord`.
                MojoValue coordIdx =
                    lane.coordIndex
                        ? functionBody.arguments[1].resolvedName
                        : state.file.declarations().var(MojoValue(kStructCoord)(
                              {}, {functionBody.arguments[1].resolvedName}));

                // The store lanes' body leads with [value, index]; the load
                // lane's leads with [index] alone. Captures follow either way.
                unsigned captureBase = 1;
                if (lane.takesValue) {
                  // `value` is `SIMD[dtype, width]` with the method's generic
                  // `dtype`; nested elementwise calls hardcode `valueDtype`'s
                  // literal, so rebind `value` to it.
                  MojoValue reboundVal = state.file.declarations().var(
                      functionBody.arguments[2].resolvedName.rebind(
                          MojoValue(kStructSIMD)[{
                              valueDtype,
                              functionBody.parameter(kParameterSimdWidth)}]));
                  state.setMojoValueFor(region.getArgument(0), reboundVal);
                  state.setMojoValueFor(region.getArgument(1), coordIdx);
                  captureBase = 2;
                } else {
                  state.setMojoValueFor(region.getArgument(0), coordIdx);
                }

                for (unsigned j = 0; j < numCaptures; ++j) {
                  // Each capture gets a fresh block arg. Record its decomposed
                  // MTS components off the struct's fields/param for the body's
                  // loads/stores; the spec carries the static alignment.
                  Value captureArg = region.getArgument(captureBase + j);
                  state.setFusedTensorArgFor(
                      captureArg,
                      {MojoValue(kParameterSelf).get(selfPtrFields[j]),
                       MojoValue(kParameterSelf).get(selfShapeFields[j]),
                       MojoValue(kParameterSelf).get(selfStridesFields[j]),
                       selfSpecParams[j]});
                }

                // Shape captures follow the tensor captures as trailing block
                // args; each resolves straight to its own `IndexList` field.
                for (unsigned k = 0; k < numShapeCaptures; ++k)
                  state.setMojoValueFor(
                      region.getArgument(captureBase + numCaptures + k),
                      MojoValue(kParameterSelf).get(shapeCaptureFields[k]));

                Block &body = region.front();
                // A `mogg.yield` becomes the method's return, handled below.
                errOccurred = llvm::any_of(body, [&](Operation &op) {
                  if (isa<YieldOp>(op))
                    return false;
                  return state.lowerOpToMojo(&op).failed();
                });
                if (errOccurred || !returnsValue)
                  return;

                // The method's generic `dtype` isn't statically known to equal
                // the yielded value's own dtype (which comes from the capture
                // tiles' literals), so rebind on the way out; the two are equal
                // at every instantiation.
                auto yieldOp = cast<YieldOp>(body.getTerminator());
                MojoValue result =
                    state.getMojoValueFor(yieldOp.getOperands().front());
                state.file.declarations().returnsValue(result.rebind(simdType));
              });
        });
  }

  if (errOccurred)
    return failure();

  // Instantiate the functor: per capture, the comptime `StaticTensorSpec` and
  // the (ptr, shape, strides) instance fields, in the declared order. The spec
  // param's inferred rank/layout are resolved from each passed spec.
  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  for (const FusedCaptureInfo &capture : captureInfos) {
    comptimeArgs.push_back(capture.spec);
    instanceArgs.push_back(capture.ptr);
    instanceArgs.push_back(capture.shape);
    instanceArgs.push_back(capture.strides);
  }
  // Shape captures' `IndexList`s follow the (ptr, shape, strides) triples, in
  // the trailing field order the struct was declared with.
  for (MojoValue shapeArg : shapeCaptureArgs)
    instanceArgs.push_back(shapeArg);
  return state.file->var(structNameVal(comptimeArgs, instanceArgs));
}

// The monoid of `fold` at SIMD width `width`, its dtype parameters bound from
// the fold's `params`.
static FailureOr<MojoValue>
monoidType(EmitMojoState &state, RowwiseReduceOp fold, const MojoValue &width) {
  KernelDeclAdaptor adaptor(
      state.registry.lookupKernelBySymName(fold.getMonoid()));
  StringRef widthParam = adaptor.reduction->getWidthParam().getValue();
  DictionaryAttr bound = fold.getParamsAttr();
  if (widthParam.empty() || !bound)
    return fold.emitOpError("has no width parameter or bound params");
  SmallVector<MojoValue> params;
  for (NamedAttribute param : bound) {
    ErrorOr<DType> dtype =
        M::MO::fromMLIRType(cast<TypeAttr>(param.getValue()).getValue());
    if (dtype.isError())
      return fold.emitOpError("binds a parameter to a non-dtype type");
    params.push_back(MojoValue(param.getName().getValue())
                         .equal(state.mlirDTypeToMojoType(*dtype)));
  }
  params.push_back(MojoValue(widthParam).equal(width));
  return MojoValue(fold.getMonoid())[params];
}

// Binds the rowwise-body values `region` reads to their value inside a lambda
// of width `width`: a constant splats again at that width, and any other value,
// computed once per row at `params.simd_width`, reads its first `width` lanes.
// Runs inside the lambda's `ValueScope`.
static LogicalResult bindRowValues(EmitMojoState &state, Region &region,
                                   const MojoValue &width) {
  llvm::SetVector<Value> values;
  region.walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp();
          def &&
          def->getBlock() == state.currentRegion<RowwiseBodyRegion>().body)
        values.insert(operand);
  });
  for (Value value : values) {
    if (auto constant = value.getDefiningOp<ConstantScalarOp>()) {
      if (failed(state.lowerOpToMojo(constant)))
        return failure();
      continue;
    }
    // <value>.slice[width]()
    state.setMojoValueFor(
        value, state.getMojoValueFor(value).get("slice")[width]({}, {}));
  }
  return success();
}

// Lowers the body of `phase` inside a phase lambda of width `width`, at the
// element index `idx`, reading the row's tile as `tile`.
static LogicalResult lowerPhaseBody(EmitMojoState &state,
                                    RowwisePhaseOpInterface phase,
                                    const MojoValue &width,
                                    const MojoValue &alignment,
                                    const MojoValue &tile) {
  auto parametersScope =
      state.enterRegion<ElementwiseLambdaRegion>(width, alignment);
  state.setMojoValueFor(phase.getIndexArg(), MojoValue("idx"));
  if (failed(bindRowValues(state, phase->getRegion(0), width)))
    return failure();
  state.currentRegion<RowwiseBodyRegion>().tile = tile;
  for (Operation &op : phase->getRegion(0).front().without_terminator())
    if (failed(state.lowerOpToMojo(&op)))
      return failure();
  return success();
}

// Appends a launch's width limit to its arguments: a static one as an `Idx`,
// a run-time one as the largest power of two dividing its dims (widths are
// powers of two, so that is all the launch needs, not their GCD).
static void appendMaxSimdWidthArg(SmallVectorImpl<MojoValue> &arguments,
                                  std::optional<int64_t> maxSimdWidth,
                                  Value dynamicMaxSimdWidth,
                                  EmitMojoState &state) {
  if (maxSimdWidth)
    arguments.push_back(
        MojoValue(kArgMaxSimdWidth).equal(MojoValue(kAliasIdx)[*maxSimdWidth]));
  if (!dynamicMaxSimdWidth)
    return;
  MojoValue dims = MojoValue(kFnCoordToIndexList)(
      {}, {state.getMojoValueFor(dynamicMaxSimdWidth)});
  MojoValue divisor =
      state.file->var(MojoValue(kFnPow2DivisorOfDims)({}, {dims}));
  MojoValue scalarInt =
      MojoValue(kStructScalar)[MojoValue(kStructDType).get(kEnumDTypeInt)];
  arguments.push_back(
      MojoValue(kArgMaxSimdWidth).equal(scalarInt({}, {divisor})));
}

// The launch's arguments; a width limit is passed as on `_foreach`.
static SmallVector<MojoValue>
rowwiseLaunchArgs(CallRowwiseOp op, EmitMojoState &state, MojoValue bodyFn,
                  MojoValue shape, MojoValue context) {
  SmallVector<MojoValue> arguments = {bodyFn, shape, context};
  appendMaxSimdWidthArg(arguments, op.getMaxSimdWidth(),
                        op.getDynamicMaxSimdWidth(), state);
  return arguments;
}

// Prints the kernel as a Row API body and launches it. The body and its
// closures are always inlined, so the row loop makes no call per row:
//
//   @inline(.always)
//   def body[params](row_coords, mut row_ctx) {var}:                    (1)
//     <each op of the body, in order: the tile prints the `load`        (2)
//      closure and the Row; folds, map phases, per-row loads and the
//      emit print themselves; any other row-level op runs once per row,
//      at params.simd_width>
//   mogg.call.rowwise[SimdMonoid, dtype, accum, axis = ..., ...](       (3)
//       body, shape, ctx)
LogicalResult CallRowwiseOp::emitMojo(EmitMojoState &state) {
  RowwiseTileOp tile = getTile();
  if (!tile || !getAccum() || !getNumPhases() || !getDtypeSize() ||
      !getSimdMonoid())
    return emitOpError("isn't legalized for the Row API");
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(tile.getType().getElementType());
  ErrorOr<DType> accumOr = M::MO::fromMLIRType(*getAccum());
  if (dtypeOr.isError() || accumOr.isError())
    return emitOpError("has a non-dtype `dtype` or `accum`");
  MojoValue dtype = state.mlirDTypeToMojoType(*dtypeOr);
  MojoValue accum = state.mlirDTypeToMojoType(*accumOr);

  MojoValue shape = state.getMojoValueFor(getShape());
  MojoValue context = state.getMojoValueFor(getDeviceContext());
  FailureOr<SmallVector<FusedCaptureInfo>> captureInfosOr =
      collectFusedCaptureInfos(state, *this, getCapturing());
  if (failed(captureInfosOr))
    return failure();

  Block &body = getBody().front();
  ArrayRef<BlockArgument> args = body.getArguments();
  for (auto [arg, info] : llvm::zip(args, *captureInfosOr))
    state.setFusedTensorArgFor(arg,
                               {info.ptr, info.shape, info.strides, info.spec});

  SmallVector<RowwiseReduceOp> folds =
      llvm::to_vector(body.getOps<RowwiseReduceOp>());
  auto simdFold = llvm::find_if(folds, [&](RowwiseReduceOp fold) {
    return fold.getMonoidAttr() == getSimdMonoidAttr();
  });
  if (simdFold == folds.end())
    return emitOpError("has no fold for its SIMD monoid");
  FailureOr<MojoValue> simdMonoid = monoidType(state, *simdFold, MojoValue(1));
  if (failed(simdMonoid))
    return failure();

  // `Idx[cols]` when the axis is static, else `Int(shape[axis].value())`.
  int64_t axis = getAxis();
  MojoValue axisSize =
      getCols() ? EmitMojoState::getComptimeIdx(*getCols())
                : MojoValue(kStructInt)(
                      {}, {shape[MojoValue(axis)].get("value")({}, {})});

  NamedMojoDef rowCtxArg = NamedMojoDef::asVerbatim(
      "row_ctx", MojoValue(kStructRowwiseContext)[MojoValue("params")]);
  rowCtxArg.argConvention = "mut";
  bool failedToLower = false;

  // (1) def body[params](row_coords, mut row_ctx) {var}:
  MojoValue bodyFn = state.file->lambda(
      "body_",
      {NamedMojoDef::asVerbatim("params", kStructRowwiseContextParams)},
      {NamedMojoDef::asVerbatim("row_coords", kStructCoord), rowCtxArg},
      /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kAlways,
      /*isAsync=*/false, /*raises=*/false,
      /*captureConvention=*/"var", [&](const FunctionBody &) {
        EmitMojoState::ValueScope bodyScope(state);
        MojoValue params("params");
        // comptime rank = row_coords.rank
        MojoValue rank =
            state.file->comptime(MojoValue("row_coords").get("rank"));
        MojoValue rowCoord = MojoValue(kAliasDynamicCoord)[{
            MojoValue(kStructDType).get(kEnumDTypeInt), rank}];

        auto rowwiseBodyScope = state.enterRegion<RowwiseBodyRegion>(
            &body, params, rank, rowCoord, axisSize, dtype, accum);

        // (2) The body's ops, in order.
        auto rowLevelScope = state.enterRegion<ElementwiseLambdaRegion>(
            params.get(kParameterSimdWidth), MojoValue(1));
        for (Operation &op : body)
          if (failed(state.lowerOpToMojo(&op))) {
            failedToLower = true;
            return;
          }
      });
  if (failedToLower)
    return failure();

  // (3) mogg.call.rowwise[SimdMonoid, dtype, accum, ...](body, shape, ctx)
  PrimitiveDeclOp launch = state.registry.lookupPrimitive(getOperationName());
  state.file->call(MojoValue(primitiveMojoName(launch)),
                   {*simdMonoid, dtype, accum,
                    MojoValue("axis").equal(MojoValue(axis)),
                    MojoValue("num_phases").equal(MojoValue(*getNumPhases())),
                    MojoValue("computationally_expensive")
                        .equal(MojoValue(getComputationallyExpensive())),
                    MojoValue("dtype_size").equal(MojoValue(*getDtypeSize()))},
                   rowwiseLaunchArgs(*this, state, bodyFn, shape, context));
  return success();
}

// Prints the tile as the Row's `load` closure, and the Row, which reads the
// row through it:
//
//   @inline(.always)
//   def load[width, alignment](idx) -> SIMD[dtype, width]:              (1)
//     <body at idx, each load at width and alignment>
//     return <the yielded value>                                        (2)
//   var row = Row[params, accum, dtype, axis, rank, keep_tile](         (3)
//       row_coords, Idx[cols], row_ctx, load)
LogicalResult RowwiseTileOp::emitMojo(EmitMojoState &state) {
  RowwiseBodyRegion &info = state.currentRegion<RowwiseBodyRegion>();
  auto call = cast<CallRowwiseOp>(getOperation()->getParentOp());
  Block &body = getBody().front();

  // (1) def load[width, alignment](idx) -> SIMD[dtype, width]:
  bool failedToLower = false;
  info.load = state.file->lambda(
      "load_",
      {NamedMojoDef::asVerbatim(kParameterWidth, kStructInt),
       NamedMojoDef::asVerbatim(kParameterAlignment, kStructInt)},
      {NamedMojoDef::asVerbatim("idx", info.rowCoord)},
      MojoValue(kStructSIMD)[{info.dtype, MojoValue(kParameterWidth)}],
      MojoBuilder::LambdaInline::kAlways, false, false, "var",
      [&](const FunctionBody &functionBody) {
        auto parametersScope = state.enterRegion<ElementwiseLambdaRegion>(
            /*width=*/functionBody.parameter(kParameterWidth),
            /*alignment=*/functionBody.parameter(kParameterAlignment));
        EmitMojoState::ValueScope scope(state);
        state.setMojoValueFor(body.getArgument(0), MojoValue("idx"));
        MojoValue width = functionBody.parameter(kParameterWidth);
        if (failed(bindRowValues(state, getBody(), width))) {
          failedToLower = true;
          return;
        }
        for (Operation &op : body.without_terminator())
          if (failed(state.lowerOpToMojo(&op))) {
            failedToLower = true;
            return;
          }
        // (2) return <the yielded value>
        state.file->returnsValue(state.getMojoValueFor(
            cast<YieldOp>(body.getTerminator()).getOperand(0)));
      });
  if (failedToLower)
    return failure();

  // (3) var row = Row[params, accum, dtype, axis, rank, keep_tile](...)
  MojoValue axis(call.getAxis());
  info.row = state.file->var(
      MojoValue(kStructRowwiseRow)[{
          info.params, info.accum, info.dtype, axis, info.rank,
          MojoValue("keep_tile").equal(MojoValue(call.getKeepTile()))}](
          {}, {MojoValue("row_coords"), info.axisSize, MojoValue("row_ctx"),
               info.load}),
      /*cache=*/false);

  // Row-level ops index the row with the reduced coordinate pinned to 0, as
  // `emit` does: at_axis[axis](row.row_coord, 0)
  state.setMojoValueFor(info.body->getArguments().back(),
                        MojoValue(kFuncRowwiseAtAxis)[axis](
                            {}, {info.row.get("row_coord"), MojoValue(0)}));
  return success();
}

// Prints the load of each row's lane:
//
//   var v = <tensor>[row index] at params.emit_tile_width
LogicalResult RowwiseLoadLaneOp::emitMojo(EmitMojoState &state) {
  const RowwiseBodyRegion &info = state.currentRegion<RowwiseBodyRegion>();
  state.setMojoValueFor(getResult(),
                        state.file->var(simdLoadFromCapture(
                            state.getFusedTensorArgFor(getTensor()),
                            info.params.get("emit_tile_width"), MojoValue(1),
                            state.getMojoValueFor(getIndex()))));
  return success();
}

// Prints the lane spread over the folds' width with the op's primitive:
//
//   var vw = _rowwise_splat_lane(row, v)
LogicalResult RowwiseSplatLaneOp::emitMojo(EmitMojoState &state) {
  PrimitiveDeclOp splat = state.registry.lookupPrimitive(getOperationName());
  ASSERT_STREAM(splat, "Primitive not found for " << getOperationName());
  state.setMojoValueFor(getResult(),
                        state.file->var(MojoValue(primitiveMojoName(splat))(
                            {}, {state.currentRegion<RowwiseBodyRegion>().row,
                                 state.getMojoValueFor(getInput())})));
  return success();
}

// Prints the fold:
//
//   @inline(.always)
//   def contribute[w](tile: SIMD[dtype, w], idx) -> SIMD[accum, w]:     (1)
//     <fold body, reading `tile`>
//   var acc = row.reduce[Monoid[..., W = params.simd_width]](           (2)
//       contribute, load)
LogicalResult RowwiseReduceOp::emitMojo(EmitMojoState &state) {
  if (!getFieldsAttr())
    return emitOpError("has no bound fields");
  const RowwiseBodyRegion &info = state.currentRegion<RowwiseBodyRegion>();
  auto yield = cast<YieldOp>(getBody().front().getTerminator());

  // (1) def contribute[w](tile: SIMD[dtype, w], idx) -> SIMD[accum, w]:
  bool failedToLower = false;
  MojoValue contribute = state.file->lambda(
      "contribute_",
      {NamedMojoDef::asVerbatim(kParameterTileWidth, kStructInt)},
      {NamedMojoDef::asVerbatim(
           "tile",
           MojoValue(
               kStructSIMD)[{info.dtype, MojoValue(kParameterTileWidth)}]),
       NamedMojoDef::asVerbatim("idx", info.rowCoord)},
      MojoValue(kStructSIMD)[{info.accum, MojoValue(kParameterTileWidth)}],
      MojoBuilder::LambdaInline::kAlways, false, false, "var",
      [&](const FunctionBody &functionBody) {
        EmitMojoState::ValueScope scope(state);
        if (failed(lowerPhaseBody(
                state, *this, functionBody.parameter(kParameterTileWidth),
                MojoValue(1), functionBody.arguments[0].resolvedName))) {
          failedToLower = true;
          return;
        }
        // return <contribution>
        state.file->returnsValue(state.getMojoValueFor(yield.getOperand(0)));
      });
  if (failedToLower)
    return failure();

  // (2) var acc = row.reduce[Monoid[..., W = params.simd_width]](...)
  FailureOr<MojoValue> monoid =
      monoidType(state, *this, info.params.get(kParameterSimdWidth));
  if (failed(monoid))
    return failure();
  MojoValue folded = state.file->var(
      info.row.get("reduce")[*monoid]({}, {contribute, info.load}),
      /*cache=*/false);
  // acc.<field>
  for (auto [result, field] : llvm::zip(getResults(), getFieldsAttr()))
    state.setMojoValueFor(result,
                          folded.get(cast<StringAttr>(field).getValue()));
  return success();
}

// Prints the map phase:
//
//   @inline(.always)
//   def elementwise[w](tile: SIMD[dtype, w], idx):                      (1)
//     comptime al = Context[params].element_alignment[dtype, w]()
//     <map body, reading row values as .slice[w]()>
//   row.elementwise(elementwise, load)                                  (2)
LogicalResult RowwiseMapOp::emitMojo(EmitMojoState &state) {
  const RowwiseBodyRegion &info = state.currentRegion<RowwiseBodyRegion>();

  // (1) def elementwise[w](tile: SIMD[dtype, w], idx):
  bool failedToLower = false;
  MojoValue elementwise = state.file->lambda(
      "elementwise_",
      {NamedMojoDef::asVerbatim(kParameterTileWidth, kStructInt)},
      {NamedMojoDef::asVerbatim(
           "tile",
           MojoValue(
               kStructSIMD)[{info.dtype, MojoValue(kParameterTileWidth)}]),
       NamedMojoDef::asVerbatim("idx", info.rowCoord)},
      /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kAlways, false,
      false, "var", [&](const FunctionBody &functionBody) {
        EmitMojoState::ValueScope scope(state);
        MojoValue width = functionBody.parameter(kParameterTileWidth);
        // comptime al = Context[params].element_alignment[dtype, w]()
        MojoValue alignment = state.file->comptime(
            MojoValue(kStructRowwiseContext)[info.params].get(
                kParameterElementAlignment)[{info.dtype, width}]({}, {}));
        failedToLower =
            failed(lowerPhaseBody(state, *this, width, alignment,
                                  functionBody.arguments[0].resolvedName));
      });
  if (failedToLower)
    return failure();

  // (2) row.elementwise(elementwise, load)
  state.file->call(info.row.get("elementwise"), {}, {elementwise, info.load});
  return success();
}

// Prints the per-row stores:
//
//   @inline(.always)
//   def write(oc):                                                      (1)
//     <the emit's ops at oc, reading row values as
//      .slice[params.emit_tile_width]()>
//   row.emit(write)                                                     (2)
LogicalResult RowwiseEmitOp::emitMojo(EmitMojoState &state) {
  const RowwiseBodyRegion &info = state.currentRegion<RowwiseBodyRegion>();
  MojoValue emitWidth = info.params.get("emit_tile_width");

  // (1) def write(oc):
  bool failedToLower = false;
  MojoValue write = state.file->lambda(
      "write_", {}, {NamedMojoDef::asVerbatim("oc", info.rowCoord)},
      /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kAlways, false,
      false, "var", [&](const FunctionBody &) {
        auto parametersScope =
            state.enterRegion<ElementwiseLambdaRegion>(emitWidth, MojoValue(1));
        EmitMojoState::ValueScope scope(state);
        state.setMojoValueFor(info.body->getArguments().back(),
                              MojoValue("oc"));
        if (failed(bindRowValues(state, getBody(), emitWidth))) {
          failedToLower = true;
          return;
        }
        for (Operation &op : getBody().front())
          if (failed(state.lowerOpToMojo(&op))) {
            failedToLower = true;
            return;
          }
      });
  if (failedToLower)
    return failure();

  // (2) row.emit(write)
  state.file->call(info.row.get("emit"), {}, {write});
  return success();
}

// The phase's closure gets the tile as its argument; the read prints
// nothing: tile
LogicalResult RowwiseTileLoadOp::emitMojo(EmitMojoState &state) {
  state.setMojoValueFor(getResult(),
                        state.currentRegion<RowwiseBodyRegion>().tile);
  return success();
}

/// Whether an epilogue body does nothing but hand the kernel's value back
/// (compute) or store it untouched at the kernel's index into the store
/// target (store): the pass-through epilogue `MOToMAP` gives every output.
/// Body args: the kernel's value, its index, then the captures, the first of
/// which is the store target.
static bool isPassThroughEpilogueBody(Block &body) {
  constexpr unsigned kValueArg = 0;
  constexpr unsigned kIndexArg = 1;
  constexpr unsigned kStoreTargetArg = 2;
  if (!llvm::hasSingleElement(body))
    return false;
  Value value = body.getArgument(kValueArg);
  if (auto yield = dyn_cast<YieldOp>(body.front()))
    return yield.getNumOperands() == 1 && yield.getOperand(0) == value;
  auto store = dyn_cast<TensorStoreOp>(body.front());
  return store && store.getValue() == value &&
         store.getIndex() == body.getArgument(kIndexArg) &&
         store.getTensor() == body.getArgument(kStoreTargetArg);
}

/// The tile counterpart of `isPassThroughEpilogueBody`.
static bool isPassThroughTileBody(Block &body) {
  constexpr unsigned kValueArg = 0;
  constexpr unsigned kCoordArg = 1;
  constexpr unsigned kStoreTargetArg = 2;
  if (!llvm::hasSingleElement(body))
    return false;
  Value value = body.getArgument(kValueArg);
  if (auto yield = dyn_cast<YieldOp>(body.front()))
    return yield.getNumOperands() == 1 && yield.getOperand(0) == value;
  auto store = dyn_cast<TileStoreOp>(body.front());
  return store && store.getValue() == value &&
         store.getCoord() == body.getArgument(kCoordArg) &&
         body.getNumArguments() > kStoreTargetArg &&
         store.getTensor() == body.getArgument(kStoreTargetArg);
}

bool TensorCreateEpilogueOp::recomputesValue() {
  return !isPassThroughEpilogueBody(getBody().front());
}

bool TileCreateEpilogueOp::recomputesValue() {
  return !isPassThroughTileBody(getBody().front());
}

static std::optional<MojoValue> scalarConstantAttrToMojoLiteral(TypedAttr attr);

/// The call expression for `op`'s elementwise kernel on `inputs`, through the
/// kernel's `kind` function. `extraParams` follow the inferred parameters.
static FailureOr<MojoValue>
emitElementwiseCallExpr(EmitMojoState &state, CallElementwiseOp op,
                        KernelFunctionKind kind, ArrayRef<MojoValue> inputs,
                        ArrayRef<MojoValue> extraParams = {}) {
  KernelDeclAdaptor ewAdaptor(
      state.registry.lookupKernelBySymName(op.getCallee()));
  const KernelDeclFunctionAdaptor *ewFnAdaptor = ewAdaptor.getFunction(kind);
  if (!ewFnAdaptor)
    return op.emitOpError("kernel has no ")
           << stringifyKernelFunctionKind(kind) << " function";

  auto outputs = op.getOutputs();
  if (outputs.size() != 1)
    return op.emitError(
        "elementwise with outputs size != 1 not supported for EmitMojo");
  Value outputVal = outputs.front();

  std::optional<StringRef> maybeDtypeParamName =
      EmitMojoState::getResultParamNameFor(*ewFnAdaptor, kParameterDType);
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    if (maybeDtypeParamName.has_value() && paramName == *maybeDtypeParamName) {
      ErrorOr<DType> resultTyOr = M::MO::fromMLIRType(
          cast<ElementTypeInterface>(outputVal.getType()).getElementType());
      if (resultTyOr.isError())
        return Error("invalid output type can't be converted to mojo dtype");
      return state.mlirDTypeToMojoType(*resultTyOr);
    }
    return state.inferDefaultKernelParamValue(op, paramName);
  };
  auto maybeParams = inferKernelParameters(ewAdaptor, kind, inferParam);
  if (maybeParams.isError())
    return op.emitError() << "Failed to infer all the kernel parameters: "
                          << maybeParams.takeError();

  SmallVector<MojoValue> params(maybeParams->functionParameters);
  params.append(extraParams.begin(), extraParams.end());
  return MojoValue(ewFnAdaptor->fullyResolvedName.strref())(params, inputs);
}

/// A capture's static shape, from its `mogg.tensor.create*` op.
static MOSH::ShapeAttr captureShape(Value tensor) {
  Operation *def = tensor.getDefiningOp();
  if (auto op = dyn_cast_or_null<TensorCreateOp>(def))
    return op.getShapeAttr();
  if (auto op = dyn_cast_or_null<TensorCreateSliceOp>(def))
    return op.getOutShape();
  if (auto op = dyn_cast_or_null<TensorCreateBroadcastOp>(def))
    return op.getStaticOutShape();
  if (auto op = dyn_cast_or_null<TensorCreateTransposeOp>(def))
    return op.getOutShape();
  if (auto op = dyn_cast_or_null<TensorCreateReshapeOp>(def))
    return op.getStaticOutShape();
  return {};
}
namespace {

/// The tile fusion-struct flavors, one per trait in managed_tensor_slice.mojo.
enum class TileFusionKind {
  /// `ComputeOutputFusionTile`: returns the tile for the kernel to store.
  Compute,
  /// `OutputFusionTile`: owns the store.
  Store,
  /// `ElementwiseFusionTile`: a standalone body, whose driver stores the tile.
  Elementwise,
};

struct TileCapture {
  /// A `mogg.tensor.create*` result, and the body argument standing for it.
  Value tensor;
  BlockArgument arg;
  DType dtype;
  SmallVector<int64_t> dims;
  /// The `N` of the struct's `spec_N` / `tile_N` names.
  unsigned idx;

  unsigned rank() const { return dims.empty() ? 1 : dims.size(); }
};
} // namespace

static FailureOr<TileCapture> makeTileCapture(Operation *op, Value tensor,
                                              BlockArgument arg, unsigned idx) {
  auto create = tensor.getDefiningOp<TensorCreateInterface>();
  MOSH::ShapeAttr shape = captureShape(tensor);
  if (!create || !shape)
    return op->emitOpError("capture must be a mogg.tensor.create(.slice|"
                           ".broadcast|.transpose|.reshape) result");
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(create.getElementType());
  if (dtypeOr.isError())
    return op->emitOpError("failed to convert capture dtype to Mojo");
  return TileCapture{tensor, arg, *dtypeOr, shape.getIntDims(), idx};
}

/// The tile of `field` at `tile_coords`, `get_kernel_tile_shape` big. The
/// kernel must use the same shape (TODO(GEX-3905)).
static MojoValue emitCaptureTileAt(MojoBuilder &declarations,
                                   const MojoValue &field, unsigned rank,
                                   StringRef device) {
  MojoValue tileShape = declarations.comptime(MojoValue(kFnGetKernelTileShape)[{
      MojoValue(kParameterDType), MojoValue(device.str()).quote()}]());
  SmallVector<MojoValue> tileDims;
  SmallVector<MojoValue> coordArgs;
  for (unsigned d = 0; d < rank; ++d) {
    tileDims.push_back(tileShape[d]);
    coordArgs.push_back(
        MojoValue(kStructInt)({}, {MojoValue(kParameterTileCoords)[d]}));
  }
  MojoValue coord = MojoValue(kStructCoord)({}, coordArgs);
  return declarations.var(field.get("tile")[tileDims]({}, {coord}),
                          /*cache=*/false);
}

/// Stages the tile of `field` into storage shaped like `layoutSrc` with the
/// method's load copier. `cache=false`: each load needs its own storage.
static MojoValue emitTileCopierLoad(MojoBuilder &declarations,
                                    const MojoValue &field, unsigned rank,
                                    llvm::StringRef layoutSrc,
                                    StringRef device) {
  TileTensorRef dstTile{MojoValue(layoutSrc.str())};
  MojoValue storage =
      declarations.var(TileTensorRef::makeStorage(), /*cache=*/false);
  MojoValue dst =
      declarations.var(TileTensorRef::fromMutStorage(storage, dstTile.layout()),
                       /*cache=*/false);
  MojoValue src = emitCaptureTileAt(declarations, field, rank, device);
  declarations.call(MojoValue(kParameterCopier).get("copy"), {},
                    {dst, TileTensorRef(src).castToSrcAddressSpace()});
  return declarations.var(
      TileTensorRef::fromGenericStorage(storage, dstTile.layout()),
      /*cache=*/false);
}

/// The trait method's signature for `lane`, which must match the trait
/// (managed_tensor_slice.mojo).
static void
emitTileFusionMethodSignature(TileFusionKind lane,
                              SmallVectorImpl<NamedMojoDef> &methodParameters,
                              SmallVectorImpl<NamedMojoDef> &methodArgs,
                              std::optional<MojoValue> &returnType) {
  bool isStore = lane == TileFusionKind::Store;
  MojoValue tileTensorType = MojoValue(kStructTileTensor)[{
      kParameterDType, kParameterLayoutType, kComptimeMutAnyOrigin}];

  methodParameters.assign({
      NamedMojoDef::asVerbatim(kParameterDType, kStructDType),
      NamedMojoDef::asVerbatim(kParameterRank, kStructInt),
      NamedMojoDef::asVerbatim(kParameterLayoutType, kStructTensorLayout),
      // The runtime arg below is `copier: Copier`; the names must differ.
      NamedMojoDef::asVerbatim(kParameterCopierType, kStructTileCopier),
  });
  if (isStore)
    methodParameters.push_back(
        NamedMojoDef::asVerbatim(kParameterStoreCopierType, kStructTileCopier));

  methodArgs.push_back(NamedMojoDef::asVerbatim(kParameterSelf));
  methodArgs.push_back(NamedMojoDef::asVerbatim(
      kParameterTileCoords, MojoValue(kStructIndexList)[kParameterRank]));
  methodArgs.push_back(
      NamedMojoDef::asVerbatim(kParameterCopier, kParameterCopierType));
  if (isStore)
    methodArgs.push_back(NamedMojoDef::asVerbatim(kParameterStoreCopier,
                                                  kParameterStoreCopierType));
  methodArgs.push_back(NamedMojoDef::asVerbatim(
      lane == TileFusionKind::Elementwise ? kParameterDst : kParameterVal,
      tileTensorType));
  if (!isStore)
    returnType = tileTensorType;
}

/// Declares a tile fusion struct for `lane`'s trait and translates `body` into
/// its method, returning the struct's name. `value` is the kernel's incoming
/// tile (epilogue lanes), and `storeTarget` the argument the body stores to.
static FailureOr<MojoValue>
emitTileFusionStruct(Operation *op, EmitMojoState &state, TileFusionKind lane,
                     Block &body, ArrayRef<TileCapture> captures, Value index,
                     Value value, Value storeTarget, DType dtype,
                     StringRef device) {
  bool isStore = lane == TileFusionKind::Store;
  bool isElementwise = lane == TileFusionKind::Elementwise;
  StringRef dstArg = isElementwise ? kParameterDst : kParameterVal;

  // Local staging tiles are allocated with the method's `dtype`.
  for (const TileCapture &cap : captures)
    if (cap.dtype != dtype)
      return op->emitOpError("a tile fusion requires every captured operand "
                             "to share one dtype");

  // The rank is inferred: a view's spec (a broadcast, say) carries a symbolic
  // one.
  SmallVector<std::pair<MojoValue, MojoValue>> parameters;
  SmallVector<std::pair<MojoValue, MojoValue>> fields;
  DenseMap<Value, MojoValue> fieldFor;
  DenseMap<Value, const TileCapture *> captureFor;
  for (const TileCapture &cap : captures) {
    std::string ci = std::to_string(cap.idx);
    std::string dtypeText = state.mlirDTypeToMojoType(cap.dtype).text;
    parameters.push_back(
        {MojoValue("spec_" + ci),
         MojoValue(llvm::formatv("{0}[{1}, _, ...]", kStructStaticTensorSpec,
                                 dtypeText)
                       .str())});
    fields.push_back(
        {MojoValue("tile_" + ci),
         TileTensorRef::captureFieldType(dtypeText, "Self.spec_" + ci)});
    fieldFor[cap.arg] = MojoValue(kParameterSelf).get("tile_" + ci);
    captureFor[cap.arg] = &cap;
  }

  StringRef traitName = isStore         ? kTraitOutputFusionTile
                        : isElementwise ? kTraitElementwiseFusionTile
                                        : kTraitComputeOutputFusionTile;

  MojoBuilder &declarations = state.file.declarations();
  MojoValue name =
      declarations.newVar(declarations.getIdentifier("FusionStruct"));
  bool errOccurred = false;
  {
    auto declarationScope = state.file.enterDeclarations();
    declarations.structDecl(
        name, parameters, {MojoValue(traitName)}, fields, [&]() {
          SmallVector<NamedMojoDef> methodParams;
          SmallVector<NamedMojoDef> methodArgs;
          std::optional<MojoValue> returnType;
          emitTileFusionMethodSignature(lane, methodParams, methodArgs,
                                        returnType);

          declarations.def(
              MojoValue(isStore ? "store" : "compute"), methodParams,
              methodArgs, returnType, /*exported=*/false, /*raises=*/false,
              /*isAlwaysInline=*/true, [&](const FunctionBody &) {
                DenseMap<Value, MojoValue> tileFor;
                DenseMap<Value, MojoValue> scalarFor;
                if (value)
                  tileFor[value] = MojoValue(kParameterVal);

                auto unsupported = [&](Operation *bodyOp, StringRef why) {
                  bodyOp->emitOpError() << "in a tile fusion: " << why;
                  errOccurred = true;
                };
                MojoValue result;
                for (Operation &bodyOp : body) {
                  if (auto load = dyn_cast<TileLoadOp>(bodyOp)) {
                    auto fieldIt = fieldFor.find(load.getTensor());
                    if (load.getCoord() != index || fieldIt == fieldFor.end())
                      return unsupported(load, "a load must read a capture at "
                                               "the loop index");
                    tileFor[load.getResult()] = emitTileCopierLoad(
                        declarations, fieldIt->second,
                        captureFor[load.getTensor()]->rank(), dstArg, device);
                    continue;
                  }
                  if (auto constant = dyn_cast<ConstantScalarOp>(bodyOp)) {
                    std::optional<MojoValue> literal =
                        scalarConstantAttrToMojoLiteral(constant.getValue());
                    if (!literal)
                      return unsupported(constant, "unsupported scalar type");
                    scalarFor[constant.getResult()] = *literal;
                    continue;
                  }
                  if (auto call = dyn_cast<CallElementwiseOp>(bodyOp)) {
                    KernelDeclAdaptor ewAdaptor(
                        state.registry.lookupKernelBySymName(call.getCallee()));
                    if (!ewAdaptor.isTileElementwise())
                      return unsupported(call, "the kernel has no TileTensor "
                                               "elementwise overload");
                    // One tile argument more than the op's operands is a
                    // destination to fill (`mo.cast`).
                    bool takesDestination =
                        ewAdaptor.tileElementwiseFunction->inputArguments
                            .size() == call.getInputs().size() + 1;
                    SmallVector<MojoValue> inputs;
                    // A constant operand is a comptime parameter of the
                    // overload.
                    SmallVector<MojoValue> scalarParams;
                    for (Value in : call.getInputs()) {
                      if (auto scalarIt = scalarFor.find(in);
                          scalarIt != scalarFor.end()) {
                        // A bare literal would default to Float64.
                        scalarParams.push_back(
                            MojoValue("scalar").equal(MojoValue(
                                kStructScalar)[{MojoValue(kParameterDType)}](
                                {}, {scalarIt->second})));
                        continue;
                      }
                      auto it = tileFor.find(in);
                      if (it == tileFor.end())
                        return unsupported(call,
                                           "every elementwise operand must "
                                           "be a tile or a constant");
                      inputs.push_back(it->second);
                    }
                    if (takesDestination && isElementwise)
                      inputs.insert(inputs.begin(), MojoValue(kParameterDst));
                    FailureOr<MojoValue> expr = emitElementwiseCallExpr(
                        state, call, KernelFunctionKind::TileElementwise,
                        inputs, scalarParams);
                    if (failed(expr)) {
                      errOccurred = true;
                      return;
                    }
                    tileFor[call.getOutputs().front()] =
                        declarations.var(*expr);
                    continue;
                  }
                  if (auto store = dyn_cast<TileStoreOp>(bodyOp)) {
                    if (lane == TileFusionKind::Compute ||
                        store.getCoord() != index ||
                        store.getTensor() != storeTarget ||
                        !tileFor.count(store.getValue()))
                      return unsupported(store, "a store must write a tile to "
                                                "the output at the loop index");
                    result = tileFor[store.getValue()];
                    continue;
                  }
                  if (auto yieldOp = dyn_cast<YieldOp>(bodyOp)) {
                    auto it = tileFor.find(yieldOp.getOperand(0));
                    if (it == tileFor.end())
                      return unsupported(yieldOp, "must yield a tile");
                    result = it->second;
                    continue;
                  }
                  return unsupported(&bodyOp, "unsupported op");
                }
                if (errOccurred)
                  return;

                if (isStore) {
                  MojoValue outTile = emitCaptureTileAt(
                      declarations, fieldFor[storeTarget],
                      captureFor[storeTarget]->rank(), device);
                  MojoValue dstSpace =
                      MojoValue(kParameterStoreCopierType)
                          .get(TileTensorRef::kDstAddressSpace);
                  MojoValue srcSpace =
                      MojoValue(kParameterStoreCopierType)
                          .get(TileTensorRef::kSrcAddressSpace);
                  declarations.call(
                      MojoValue(kParameterStoreCopier).get("copy"), {},
                      {outTile.get(
                           TileTensorRef::kAddressSpaceCast)[dstSpace](),
                       result.get(
                           TileTensorRef::kAddressSpaceCast)[srcSpace]()});
                  return;
                }
                declarations.returnsValue(result.rebind(MojoValue(
                    kStructTileTensor)[{kParameterDType, kParameterLayoutType,
                                        kComptimeMutAnyOrigin}]));
              });
        });
  }
  if (errOccurred)
    return failure();
  return name;
}

/// Instantiates the struct from each capture's spec and `to_tile_tensor()`.
static MojoValue instantiateTileFusionStruct(EmitMojoState &state,
                                             const MojoValue &name,
                                             ArrayRef<TileCapture> captures) {
  SmallVector<MojoValue> comptimeArgs;
  SmallVector<MojoValue> instanceArgs;
  for (const TileCapture &cap : captures) {
    ManagedTensorSliceRef mts(state.getMojoValueFor(cap.tensor));
    comptimeArgs.push_back(mts.staticSpec());
    instanceArgs.push_back(state.file->var(mts.toTileTensor()));
  }
  return state.file->var(name(comptimeArgs, instanceArgs));
}

/// Binds a tile fused output (`_FusedComputeOutputTileTensor` /
/// `_FusedOutputTileTensor`) onto a slice over `$outputPtr`. A pass-through
/// body binds the "no fusion" sentinel, so the kernel stores by itself.
LogicalResult TileCreateEpilogueOp::emitMojo(EmitMojoState &state) {
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getElementType());
  if (dtypeOr.isError())
    return emitOpError("failed to convert epilogue element type to Mojo");
  Block &body = getBody().front();
  FailureOr<unsigned> rank = fusedRegionRank(*this, getShape());
  if (failed(rank))
    return failure();
  TileFusionKind lane = getLane() == TileFusionLane::Compute
                            ? TileFusionKind::Compute
                            : TileFusionKind::Store;

  MojoValue fusion;
  if (isPassThroughTileBody(body)) {
    fusion = MojoValue(lane == TileFusionKind::Compute
                           ? kStructMsNoComputeFusionTile
                           : kStructMsNoOutputFusionTile)();
  } else {
    // Body args are (value, index, captures...); capture 0 is the store
    // target.
    SmallVector<TileCapture> captures;
    for (auto [i, cap] : llvm::enumerate(getCaptures())) {
      FailureOr<TileCapture> capture =
          makeTileCapture(*this, cap, body.getArgument(2 + i), i);
      if (failed(capture))
        return failure();
      captures.push_back(*capture);
    }
    FailureOr<MojoValue> name = emitTileFusionStruct(
        *this, state, lane, body, captures, /*index=*/body.getArgument(1),
        /*value=*/body.getArgument(0), /*storeTarget=*/body.getArgument(2),
        *dtypeOr, getDevice());
    if (failed(name))
      return failure();
    fusion = instantiateTileFusionStruct(state, *name, captures);
  }

  MojoValue layout = state.file->var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
  ManagedTensorSliceRef baseSlice(emitBaseSliceFromPtr(
      state, state.mlirDTypeToMojoType(*dtypeOr), *rank, layout,
      MojoValue(static_cast<uint64_t>(
          M::Driver::getPreferredAlignment(getDevice().str()))),
      state.getMojoValueFor(getOutputPtr())));
  MojoValue bind = lane == TileFusionKind::Compute
                       ? baseSlice.bindToFusedComputeOutputTile()
                       : baseSlice.bindToFusedOutputTile();
  state.setMojoValueFor(getTensor(), state.file->var(bind({}, {fusion})));
  return success();
}

/// Drives a standalone elementwise body through an `ElementwiseFusionTile`
/// struct with `foreach_fusion_tile`, which stores each tile `compute` returns
/// into the output.
LogicalResult CallForeachTileOp::emitMojo(EmitMojoState &state) {
  Block &body = getBody().front();
  ValueRange capturing = getCapturing();
  auto storeArg = dyn_cast<BlockArgument>(
      (*body.getOps<TileStoreOp>().begin()).getTensor());
  // Body args are (captures..., index); the verifier checked the one store.
  Value output = capturing[storeArg.getArgNumber()];
  FailureOr<TileCapture> outCapture =
      makeTileCapture(*this, output, storeArg, /*idx=*/0);
  if (failed(outCapture))
    return failure();

  SmallVector<TileCapture> inputs;
  for (auto [i, cap] : llvm::enumerate(capturing)) {
    if (i == storeArg.getArgNumber())
      continue;
    FailureOr<TileCapture> input =
        makeTileCapture(*this, cap, body.getArgument(i), inputs.size());
    if (failed(input))
      return failure();
    inputs.push_back(*input);
  }

  FailureOr<MojoValue> name = emitTileFusionStruct(
      *this, state, TileFusionKind::Elementwise, body, inputs,
      /*index=*/body.getArguments().back(), /*value=*/Value(),
      /*storeTarget=*/storeArg, outCapture->dtype, getDevice());
  if (failed(name))
    return failure();
  MojoValue functor = instantiateTileFusionStruct(state, *name, inputs);

  auto driverDecl =
      state.registry.lookupPrimitive(kMoggForeachTileRegistration);
  ASSERT_STREAM(driverDecl,
                "Primitive not found for " << kMoggForeachTileRegistration);
  state.file->call(
      MojoValue(primitiveMojoName(driverDecl)), {},
      {MojoValue(kParameterForeachTensor).equal(state.getMojoValueFor(output)),
       MojoValue(kParameterForeachElem).equal(functor),
       MojoValue(kParameterForeachCtx)
           .equal(state.getMojoValueFor(getContext()))});
  return success();
}

LogicalResult TensorCreateEpilogueOp::emitMojo(EmitMojoState &state) {
  // Emit the output-store fusion functor + `_MoggOutputStoreFusionTensor`
  // wrapper for a MOGG-ABI fused output; `__call__` recursively lowers this
  // op's epilogue body (as `emitMojoMOGGABI` walks a stub body).
  if (getCaptures().empty())
    return emitOpError(
        "requires at least one (store-target) capture to fuse into");

  // Emit the shared trait + wrapper definitions (`OutputStoreFusion` /
  // `_MoggOutputStoreFusionTensor`) once, exactly as the MOGG-ABI store path.
  state.declare(kOutputStoreFusionDefinitions);

  // The epilogue's output dtype, to rebind the inbound `value` (typed by the
  // trait method's abstract `dtype`). Capture dtypes come from their defining
  // ops (see `collectFusedCaptureInfos`).
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getElementType());
  if (dtypeOr.isError())
    return emitOpError("failed to convert epilogue element type to Mojo");
  MojoValue dtypeParam = state.mlirDTypeToMojoType(*dtypeOr);

  // The dtype the kernel hands in, which after a dtype-changing epilogue
  // fusion (`mo.cast`) is not the output dtype; rebinding to `dtypeParam` would
  // emit `rebind[SIMD[si32, w]](SIMD[f32, w])`, which Mojo rejects.
  ErrorOr<DType> inDtypeOr = M::MO::fromMLIRType(
      cast<SimdType>(getBody().getArgument(0).getType()).getElementType());
  if (inDtypeOr.isError())
    return emitOpError("failed to convert epilogue input type to Mojo");
  MojoValue inputDtypeParam = state.mlirDTypeToMojoType(*inDtypeOr);

  // One `LayoutType` param and (ptr, layout) field pair per capture: the store
  // target, plus one per extra input of a fused binary elementwise op.
  FailureOr<SmallVector<FusedCaptureInfo>> captureInfosOr =
      collectFusedCaptureInfos(state, *this, getCaptures());
  if (failed(captureInfosOr))
    return failure();
  SmallVector<FusedCaptureInfo> captureInfos = *captureInfosOr;

  // `MAPToMOGG` builds a yield-terminated body exactly for a compute lambda;
  // `compute` is `store` plus a return.
  bool isCompute = isa<YieldOp>(getBody().front().back());

  const FusionLambdaInfo storeLane = {
      /*structName=*/"EpilogueStoreFunctor",
      /*traitName=*/kTraitOutputFusion,
      /*methodName=*/MojoValue("store"),
      /*widthParamType=*/kStructSIMDLength,
      /*takesValue=*/true,
  };
  const FusionLambdaInfo computeLane = {
      /*structName=*/"EpilogueComputeFunctor",
      /*traitName=*/kTraitComputeOutputFusion,
      /*methodName=*/MojoValue("compute"),
      /*widthParamType=*/kStructSIMDLength,
      /*takesValue=*/true,
  };

  // A `_LegacyComputeOutput` slot (`Some[Output & Fused &
  // _LegacyComputeOutput]`) gets a compute lambda only when `MAPToMOGG` could
  // build one, which it marks by yielding. Otherwise (a cast, reshape, or
  // multi-store epilogue) the slot is downgraded to a store fusion, like a
  // legacy `FusedComputeOutput`.
  bool isComputeOutputSlot =
      getContract() && bitEnumContainsAny(getContract()->getAccess(),
                                          MOGG::TensorArgAccessKind::transform);

  // A trait-ABI fused output (`Some[Output & Fused & StoreAccess]`, `$contract`
  // stamped at MAP lowering), or a downgraded compute-output slot, wants the
  // inline `OutputStoreFusion` functor shape (Coord `__call__`, `width: Int`)
  // in `_MoggOutputStoreFusionTensor`; the MTS API binds the
  // `managed_tensor_slice.OutputFusion` shape onto a base slice.
  bool wrapStore =
      (getContract() && bitEnumContainsAny(getContract()->getAccess(),
                                           MOGG::TensorArgAccessKind::store)) ||
      (isComputeOutputSlot && !isCompute);
  const FusionLambdaInfo traitStoreLane = {
      /*structName=*/"OutputStoreFunctor",
      /*traitName=*/MojoValue("OutputStoreFusion"),
      /*methodName=*/MojoValue("__call__"),
      /*widthParamType=*/kStructInt,
      /*takesValue=*/true,
      /*coordIndex=*/true,
  };

  // A compute-output slot that got a compute lambda wants the functor in the
  // inline `TransformFusion` shape (Coord `__call__` RETURNING the transformed
  // value) wrapped in `_MoggTransformFusionTensor`, which also holds the real
  // output buffer for the kernel's own store.
  bool wrapTransform = isComputeOutputSlot && isCompute;
  const FusionLambdaInfo traitTransformLane = {
      /*structName=*/"TransformFunctor",
      /*traitName=*/MojoValue("TransformFusion"),
      /*methodName=*/MojoValue("__call__"),
      /*widthParamType=*/kStructInt,
      /*takesValue=*/true,
      /*coordIndex=*/true,
  };
  if (wrapTransform)
    state.declare(kTransformFusionDefinitions);

  // Inner-dynamic store-index reshapes carry their runtime from/to shapes as
  // trailing shape captures: each `!mogg.shape` Coord is passed whole, and
  // the functor's field carries its static/dynamic split as its own type.
  SmallVector<MojoValue> shapeCaptureTypes;
  SmallVector<MojoValue> shapeCaptureArgs;
  for (Value shapeCap : getShapeCaptures()) {
    FailureOr<MojoValue> typeOr = fusedRegionShapeCoordType(*this, shapeCap);
    if (failed(typeOr))
      return failure();
    shapeCaptureTypes.push_back(*typeOr);
    shapeCaptureArgs.push_back(state.getMojoValueFor(shapeCap));
  }

  FailureOr<MojoValue> functorOr = emitMapFusionFunctor(
      state, getBody(), captureInfos,
      wrapTransform ? traitTransformLane
      : wrapStore   ? traitStoreLane
                    : (isCompute ? computeLane : storeLane),
      inputDtypeParam, shapeCaptureTypes, shapeCaptureArgs);
  if (failed(functorOr))
    return failure();
  MojoValue functorInstance = *functorOr;

  if (wrapTransform) {
    // Wrap the transform functor in `_MoggTransformFusionTensor`, which also
    // holds the real output buffer (`$outputPtr`): the wrapper stores the
    // transformed value through it. Sourced from this op's own device/shape.
    StringRef deviceLabel = getDevice();
    bool devicePassable = deviceLabel != kCPULabel;
    MojoValue alignmentParam(static_cast<uint64_t>(
        M::Driver::getPreferredAlignment(deviceLabel.str())));
    FailureOr<unsigned> baseRankT = fusedRegionRank(*this, getShape());
    if (failed(baseRankT))
      return failure();
    MojoValue layoutVar = state.file->var(
        MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
    MojoValue baseSlice = emitBaseSliceFromPtr(
        state, dtypeParam, *baseRankT, layoutVar, alignmentParam,
        state.getMojoValueFor(getOutputPtr()));
    // The wrapper's `handle` is `Pointer[Scalar[dtype], MutUntrackedOrigin]`;
    // a slice's `unsafe_ptr()` yields `MutAnyOrigin`, so retag the origin.
    MojoValue basePtr = state.file->var(MojoValue(
        llvm::formatv("{0}.unsafe_origin_cast[MutUntrackedOrigin]()",
                      ManagedTensorSliceRef(baseSlice).unsafePtr().text)
            .str()));
    MojoValue wrapper = MojoValue("_MoggTransformFusionTensor")(
        {dtypeParam, MojoValue("type_of")({}, {layoutVar}), alignmentParam,
         MojoValue(true), MojoValue("type_of")({}, {functorInstance}),
         MojoValue(devicePassable), MojoValue(!devicePassable)},
        {functorInstance, layoutVar, basePtr});
    state.setMojoValueFor(getTensor(), state.file->var(wrapper));
    return success();
  }

  if (wrapStore) {
    // The functor owns its store to the captured target, so no base slice or
    // `_bind_to_fused_output` is needed.
    StringRef deviceLabel = getDevice();
    bool devicePassable = deviceLabel != kCPULabel;
    MojoValue alignmentParam(static_cast<uint64_t>(
        M::Driver::getPreferredAlignment(deviceLabel.str())));
    MojoValue layoutVar = state.file->var(
        MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
    // The wrapper's dtype is what the KERNEL stores (the value it hands to
    // `store`), i.e. the computed dtype -- a dtype-changing epilogue casts to
    // the real output dtype inside the functor's own store to its capture.
    MojoValue wrapper = MojoValue("_MoggOutputStoreFusionTensor")(
        {inputDtypeParam, MojoValue("type_of")({}, {layoutVar}), alignmentParam,
         MojoValue("type_of")({}, {functorInstance}), MojoValue(devicePassable),
         MojoValue(!devicePassable), MojoValue(isComputeOutputSlot)},
        {functorInstance, layoutVar});
    state.setMojoValueFor(getTensor(), state.file->var(wrapper));
    return success();
  }

  // Bind the functor onto the store target's ManagedTensorSlice via
  // `_bind_to_fused_output`, giving a ManagedTensorSlice with a resolvable
  // `static_spec` that converts to the kernel's `FusedOutputTensor`. A
  // dtype-changing epilogue binds onto a slice of the computed dtype and
  // reaches the real output through capture 0.
  ASSERT_STREAM(!isCompute || inputDtypeParam.text == dtypeParam.text,
                "a compute lambda cannot change dtype; `MAPToMOGG` classifies "
                "a dtype-changing epilogue as a store lambda");
  // The base slice is over `$outputPtr` (the store target, or a pre-cast
  // accumulation buffer for a mixed-precision accumulating kernel): real
  // storage of the compute dtype, so an accumulating kernel accumulates
  // through it; otherwise the lambda does the store.
  FailureOr<unsigned> baseRank = fusedRegionRank(*this, getShape());
  if (failed(baseRank))
    return failure();
  MojoValue baseLayout = state.file->var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
  MojoValue baseSlice = emitBaseSliceFromPtr(
      state, inputDtypeParam, *baseRank, baseLayout,
      MojoValue(static_cast<uint64_t>(
          M::Driver::getPreferredAlignment(getDevice().str()))),
      state.getMojoValueFor(getOutputPtr()));
  // Which entry point to bind through follows the kernel's declared argument
  // type, not the lane: a `_FusedComputeOutputTensor` argument binds with
  // `_bind_to_fused_compute_output` even when a dtype change or the
  // `enable-compute-lambdas` opt-out kept the store lane above.
  ManagedTensorSliceRef bindTarget(baseSlice);
  MojoValue bindMethod = getComputeOutputArg()
                             ? bindTarget.bindToFusedComputeOutput()
                             : bindTarget.bindToFusedOutput();
  MojoValue fusedTensor = bindMethod({}, {functorInstance});
  state.setMojoValueFor(getTensor(), state.file->var(fusedTensor));
  return success();
}

LogicalResult TensorCreatePrologueOp::emitMojo(EmitMojoState &state) {
  // Emit the input-load fusion functor, the load-side sibling of
  // `TensorCreateEpilogueOp::emitMojo`: the tensor never materializes, and the
  // functor returns the loaded value, rebound to the method's generic dtype.
  // Captures are optional; a fused constant producer has none.
  //
  // `$contract` (stamped at MAP lowering) selects the consumer ABI:
  //   - trait (`Some[Input & Fused & LoadAccess]`): the inline
  //     `InputLoadFusion` shape (Coord `__call__`), wrapped in
  //     `_MoggInputLoadFusionTensor`.
  //   - MTS API (`FusedInputTensor`): the `managed_tensor_slice.InputFusion`
  //     shape (IndexList `load`), bound onto a type-only dangling base slice.

  // The dtype the kernel reads; capture dtypes come from their defining ops.
  ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(getElementType());
  if (dtypeOr.isError())
    return emitOpError("failed to convert prologue element type to Mojo");
  MojoValue dtypeParam = state.mlirDTypeToMojoType(*dtypeOr);

  // One `LayoutType` param and (ptr, layout) field pair per capture (a fused
  // binary producer captures both inputs).
  FailureOr<SmallVector<FusedCaptureInfo>> captureInfosOr =
      collectFusedCaptureInfos(state, *this, getCaptures());
  if (failed(captureInfosOr))
    return failure();
  SmallVector<FusedCaptureInfo> captureInfos = *captureInfosOr;

  bool traitFused = static_cast<bool>(getContract());

  // Both lanes manufacture the loaded value (width param `Int`); they differ in
  // the trait/method/index shape the consumer accepts.
  const FusionLambdaInfo bindLoadLane = {
      /*structName=*/"PrologueLoadFunctor",
      /*traitName=*/kTraitInputFusion,
      /*methodName=*/MojoValue("load"),
      /*widthParamType=*/kStructInt,
      /*takesValue=*/false,
      /*coordIndex=*/false,
  };
  const FusionLambdaInfo traitLoadLane = {
      /*structName=*/"InputLoadFunctor",
      /*traitName=*/MojoValue("InputLoadFusion"),
      /*methodName=*/MojoValue("__call__"),
      /*widthParamType=*/kStructInt,
      /*takesValue=*/false,
      /*coordIndex=*/true,
  };

  // The wrapper's inline `InputLoadFusion` trait must be defined before the
  // functor that conforms to it.
  if (traitFused)
    state.declare(kInputLoadFusionDefinitions);

  // Inner-dynamic load-index reshapes carry runtime shapes as trailing shape
  // captures, as in the epilogue: each `!mogg.shape` Coord is passed whole.
  SmallVector<MojoValue> shapeCaptureTypes;
  SmallVector<MojoValue> shapeCaptureArgs;
  for (Value shapeCap : getShapeCaptures()) {
    FailureOr<MojoValue> typeOr = fusedRegionShapeCoordType(*this, shapeCap);
    if (failed(typeOr))
      return failure();
    shapeCaptureTypes.push_back(*typeOr);
    shapeCaptureArgs.push_back(state.getMojoValueFor(shapeCap));
  }

  FailureOr<MojoValue> functorOr = emitMapFusionFunctor(
      state, getBody(), captureInfos, traitFused ? traitLoadLane : bindLoadLane,
      /*valueDtype=*/dtypeParam, shapeCaptureTypes, shapeCaptureArgs);
  if (failed(functorOr))
    return failure();
  MojoValue functorInstance = *functorOr;

  FailureOr<unsigned> rank = fusedRegionRank(*this, getShape());
  if (failed(rank))
    return failure();
  MojoValue layoutVar = state.file->var(
      MojoValue(kFnRowMajor)({}, {state.getMojoValueFor(getShape())}));
  MojoValue alignmentParam(static_cast<uint64_t>(
      M::Driver::getPreferredAlignment(getDevice().str())));

  if (traitFused) {
    // On CPU the host proxy runs the load (`_access_enabled`); on GPU the
    // projected `device_type` does, so the host proxy disables it.
    StringRef deviceLabel = getDevice();
    bool devicePassable = deviceLabel != kCPULabel;
    MojoValue wrapper = MojoValue("_MoggInputLoadFusionTensor")(
        {dtypeParam, MojoValue("type_of")({}, {layoutVar}), alignmentParam,
         MojoValue("type_of")({}, {functorInstance}), MojoValue(devicePassable),
         MojoValue(!devicePassable)},
        {functorInstance, layoutVar});
    state.setMojoValueFor(getTensor(), state.file->var(wrapper));
    return success();
  }

  // Bind the functor onto a type-only dangling base slice: the tensor the
  // kernel READS has no slice to bind onto (fusion replaced it), so every load
  // goes through the functor.
  MojoValue baseSlice = emitDanglingBaseSlice(state, dtypeParam, *rank,
                                              layoutVar, alignmentParam);
  MojoValue fusedTensor = baseSlice.get(
      ManagedTensorSliceRef::kBindToFusedInput)({}, {functorInstance});
  state.setMojoValueFor(getTensor(), state.file->var(fusedTensor));
  return success();
}

// Collects the distinct MO dtypes touched anywhere in `body` as a
// `StaticTuple[DType, N]` MojoValue for `get_simd_width_for_dtypes`.
static FailureOr<MojoValue> collectDtypesTuple(EmitMojoState &state,
                                               Region &body) {
  llvm::SmallSetVector<DType, 4> dtypes;
  body.walk([&](Operation *op) {
    for (Type t :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      auto element = dyn_cast<ElementTypeInterface>(t);
      if (!element)
        continue;
      ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(element.getElementType());
      ASSERT_STREAM(!dtypeOr.isError(), "verified element type " << t);
      dtypes.insert(*dtypeOr);
    }
  });
  if (dtypes.empty())
    return failure();

  SmallVector<MojoValue> dtypeMojoValues;
  for (DType dtype : dtypes)
    dtypeMojoValues.push_back(state.mlirDTypeToMojoType(dtype));
  return MojoValue(kStructStaticTuple)[{
      MojoValue(kStructDType),
      MojoValue(static_cast<uint64_t>(dtypeMojoValues.size()))}](
      {}, dtypeMojoValues);
}

LogicalResult CallForeachOp::emitMojo(EmitMojoState &state) {
  // Emits the per-index closure directly, then calls `_foreach` with it.
  // `_foreach`'s `Lambda` parameter needs a blanket `{var}` capture clause, the
  // only convention that works for a closure capturing register-passable
  // values (see `MojoBuilder::lambda`'s `captureConvention`). As in the fusion
  // functors, each source is captured as its decomposed `ManagedTensorSlice`
  // components, so loads/stores go through
  // `simd_{load_from,store_into}_tensor_pointer`.
  //
  // The closure's comptime parameters are `[width, element_alignment]`;
  // `TensorLoadOp`/`TensorStoreOp` read them from the context pushed below.
  MojoValue shape = state.getMojoValueFor(getShape());
  MojoValue context = state.getMojoValueFor(getDeviceContext());
  ValueRange capturing = getCapturing();

  // Decompose each capture into the (ptr, layout) the closure captures plus its
  // static alignment, exactly as `emitMapFusionFunctor` does. Built here, in
  // the enclosing scope, so the `{var}` closure captures the (ptr, layout)
  // locals (not the slice) and can name the comptime alignment alias.
  FailureOr<SmallVector<FusedCaptureInfo>> captureInfosOr =
      collectFusedCaptureInfos(state, *this, capturing);
  if (failed(captureInfosOr))
    return failure();
  SmallVector<FusedCaptureInfo> captureInfos = *captureInfosOr;

  // Infer simd_width from all dtypes touched anywhere in the closure body.
  FailureOr<MojoValue> dtypesTuple = collectDtypesTuple(state, getBody());
  if (failed(dtypesTuple))
    return emitOpError("mogg.call.foreach: could not determine capture dtype");
  MojoValue simdWidth = MojoValue(
      kFnGetSimdWidthForDtypes)[{*dtypesTuple, context.get("target")}]({}, {});

  bool errOccurred = false;
  // Blanket `{var}` capture -- see the capture-convention note above.
  MojoValue lambdaVal = state.file->lambda(
      "lambda_",
      {NamedMojoDef::asVerbatim(kParameterWidth, kStructInt),
       NamedMojoDef::asVerbatim(kParameterElementAlignment, kStructInt,
                                MojoValue(1))},
      {NamedMojoDef::asVerbatim("idx", kStructCoord)},
      /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kAutomatic,
      /*isAsync=*/false, /*raises=*/false, /*captureConvention=*/"var",
      [&](const FunctionBody &functionBody) {
        auto parametersScope = state.enterRegion<ElementwiseLambdaRegion>(
            /*width=*/functionBody.parameter(kParameterWidth),
            /*alignment=*/functionBody.parameter(kParameterElementAlignment));

        Block &body = getBody().front();
        ArrayRef<BlockArgument> args = body.getArguments();
        for (unsigned i = 0; i < captureInfos.size(); ++i) {
          // Record the captured decomposed MTS components for the body's
          // loads/stores; the closure captures trivial values only, so all are
          // nameable here.
          const FusedCaptureInfo &info = captureInfos[i];
          state.setFusedTensorArgFor(
              args[i], {info.ptr, info.shape, info.strides, info.spec});
        }
        state.setMojoValueFor(args.back(),
                              functionBody.arguments[0].resolvedName);

        errOccurred = llvm::any_of(body, [&](Operation &op) {
          return state.lowerOpToMojo(&op).failed();
        });
      });
  if (errOccurred)
    return failure();

  // `_foreach` picks a width that divides the limit: at compile time when it
  // is an `Idx`, at launch when it is known only at run time. Widths are
  // powers of two, so a run-time limit only needs the largest power of two
  // dividing its dims, not their GCD.
  SmallVector<MojoValue> parameters = {
      MojoValue(kParameterPreferredSimdWidth).equal(simdWidth)};
  SmallVector<MojoValue> arguments = {shape, lambdaVal, context};
  appendMaxSimdWidthArg(arguments, getMaxSimdWidth(), getDynamicMaxSimdWidth(),
                        state);
  state.file->call(MojoValue(getCallee()), parameters, arguments);
  return success();
}

//===----------------------------------------------------------------------===//
// ToMGPFromOpaqueOp
//===----------------------------------------------------------------------===//

LogicalResult ToMGPFromOpaqueOp::emitMojo(EmitMojoState &state) {
  // In the emit-mojo path mo.opaque and mgp.opaque are the same value.
  state.setMojoValueFor(getDataPointer(), state.getMojoValueFor(getInput()));
  return success();
}

LogicalResult AsyncUnpackOp::emitMojo(EmitMojoState &state) {
  MojoValue unpackedValue;
  mlir::TypedValue<mlir::Type> input = getInput();
  // The input is a `!mogg.async<T>`; switch on the wrapped element type `T`.
  mlir::Type inputType = cast<AsyncType>(input.getType()).getElementType();

  if (auto tensorSpec = dyn_cast<MGP::TensorSpecType>(inputType)) {

    MojoValue asyncUnpackFunc{kFnUnpackTensorSpec};
    unpackedValue = asyncUnpackFunc({int(tensorSpec.getRank())},
                                    {state.getMojoValueFor(input)});

  } else if (auto tensorType = dyn_cast<MGP::DeviceTensorType>(inputType)) {

    MojoValue asyncUnpackFunc{kFnUnpackTensor};
    MGP::TensorSpecType tensorSpec = tensorType.getTensorSpec();
    auto tensorRank = tensorSpec.getRank();
    auto bufferRank = tensorRank == 0 ? 1 : tensorRank;
    std::optional<DType> maybeDType = tensorSpec.getOptDType();
    if (!maybeDType)
      return emitError() << "Expected Tensor operand to have DType";

    SmallVector<MojoValue> params = {bufferRank, tensorRank,
                                     state.mlirDTypeToMojoType(*maybeDType)};
    // `kind` names how the tensor participates as a device graph input; the
    // Mojo default (`data`) is left implicit.
    if (tensorType.getMut())
      params.push_back(MojoValue(kParameterKind)
                           .equal(MojoValue(kStructDeviceGraphInputKind)
                                      .get(kInputKindMutable)));
    else if (M::isHostPlaced(tensorType.getDevice()))
      params.push_back(
          MojoValue(kParameterKind)
              .equal(
                  MojoValue(kStructDeviceGraphInputKind).get(kInputKindHost)));
    unpackedValue = asyncUnpackFunc(params, state.getMojoValueFor(input));

  } else if (isa<MGP::DeviceBufferType>(inputType)) {

    // TODO: (GEX-2732) Move away from NDBuffer in primitives.mojo
    MojoValue asyncUnpackFunc{kFnUnpackBufferRef};
    unpackedValue = asyncUnpackFunc({}, {state.getMojoValueFor(input)});

  } else if (auto context = dyn_cast<MGP::DeviceContextType>(inputType)) {

    // The graph's device is static, so type the context by it instead of
    // leaving every consumer to cast.
    MojoValue asyncUnpackFunc{kFnUnpackDeviceCtx};
    MojoValue deviceKind = MojoValue(context.getDevice().getLabel()).quote();
    unpackedValue =
        asyncUnpackFunc({deviceKind}, {state.getMojoValueFor((input))});

  } else if (isa<MGP::ContextType>(inputType)) {

    MojoValue asyncUnpackFunc{kFnUnpackStateCtx};
    unpackedValue = asyncUnpackFunc({}, {state.getMojoValueFor((input))});

  } else {

    auto unpackPrimDecl =
        state.registry.lookupPrimitive(this->getOperationName());
    ASSERT_STREAM(unpackPrimDecl,
                  "Primitive not found for " << getOperationName().str());
    MojoValue moggAsyncUnpackFunc(primitiveMojoName(unpackPrimDecl));
    unpackedValue = moggAsyncUnpackFunc({state.mlirTypeToMojoType(getType())},
                                        state.getMojoValueFor(input));
  }

  MojoValue finalValue = state.file->var(unpackedValue);

  // Opaque types: bitcast from LegacyOpaquePointer to the parameterized type,
  // then dereference at the use site:
  //   var_XX = value.unsafe_bitcast[Type]()
  //   ... later: var_XX[]
  if (auto opaqueType = dyn_cast<MO::OpaqueType>(getType())) {
    MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
    auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
    MojoValue opaqueTypeStr = params.empty() ? baseTy : baseTy[params];
    OpaquePointerRef opaquePtr(finalValue);
    MojoValue ptrVar = state.file->var(opaquePtr.bitcast(opaqueTypeStr)());
    finalValue = state.file->var(OpaquePointerRef(ptrVar).takePointee());
  }

  state.setMojoValueFor(getResult(), finalValue);
  return success();
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

LogicalResult RebindOp::emitMojo(EmitMojoState &state) {
  // Bridges an unpacked async handle (`mogg.async.unpack` result) to the
  // codegen handle a MOGG-ABI stub argument expects.
  Type resultType = getOutput().getType();

  // Context / dim / opaque casts are code-generation no-ops:
  // `mogg.async.unpack` already produced the Mojo type the stub expects (a
  // `DeviceContext`, an `Int`, or the kernel's own struct), so forward it
  // unchanged.
  if (isa<MOGG::ContextType, MOGG::DimType, MO::OpaqueType>(resultType)) {
    state.setMojoValueFor(getResult(), state.getMojoValueFor(getInput()));
    return success();
  }

  // Pointer cast: the input is the `OwnedByteBuffer` an `!mgp.buffer` unpacks
  // to, but the MOGG-ABI stub takes a raw `OpaquePointer[MutAnyOrigin]` (it
  // does `ptr.bitcast[NoneType]()` in `tensor.create`). Take the buffer's data
  // pointer and bitcast it to the opaque pointer type.
  if (isa<MOGG::PointerType>(resultType)) {
    MojoValue rawPtr =
        OwnedByteBufferRef(state.getMojoValueFor(getInput())).unsafePtr();
    MojoValue opaquePtr =
        OpaquePointerRef(rawPtr).bitcast(MojoValue(kComptimeDTypeNoneType))();
    state.setMojoValueFor(getResult(), state.file->var(opaquePtr));
    return success();
  }

  return emitOpError("mogg.rebind: unsupported result type for Mojo emission: ")
         << resultType;
}

//===----------------------------------------------------------------------===//
// AsyncPackOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncPackOp::emitMojo(EmitMojoState &state) {
  // A `!mgp.tensor` value is a parametric `OwnedTensor[dtype, rank]`; its
  // generic overload would lose resolution to the `Some[Movable & Deinitable]`
  // overload (mis-packing it as an opaque Mojo value). Use the dedicated
  // tensor pack function, passing the true spec rank so a scalar (rank 0) is
  // not packed as the promoted rank-1 buffer view.
  MojoValue packed;
  if (auto tensorType = dyn_cast<MGP::DeviceTensorType>(getInput().getType())) {
    MGP::TensorSpecType tensorSpec = tensorType.getTensorSpec();
    MojoValue specRankParam(static_cast<uint64_t>(tensorSpec.getRank()));
    packed = state.file->var(MojoValue(kFnMoggAsyncPackOwnedTensorValue)(
        {specRankParam}, {state.getMojoValueFor(getInput())}));
  } else {
    MojoValue input = state.getMojoValueFor(getInput());
    Type inputType = getInput().getType();

    // Trivial types are passed by value: transferring them produces compiler
    // warnings, or fails to compile when the emitter produced a literal
    // rather than a var.
    //
    // Opaque Mojo values may be move-only, so the pack must consume them.
    bool isMoveOnly =
        isa<MO::OpaqueType, MGP::OpaqueType, MGP::DeviceGraphType>(inputType);
    if (!isa<mlir::IndexType, mlir::IntegerType, KGEN::SIMDType>(inputType) &&
        (isMoveOnly || getInput().hasOneUse()))
      input = input.transfer();
    // For every other supported type, overload resolution on
    // `mogg_async_pack_value` selects the right value-returning packer.
    packed = state.file->var(MojoValue(kFnMoggAsyncPackValue)({}, {input}));
  }

  state.setMojoValueFor(getResult(), packed);
  return success();
}

//===----------------------------------------------------------------------===//
// TensorInitOp
//===----------------------------------------------------------------------===//

/// The data pointer of an `!mgp.buffer` value as an untyped Mojo pointer.
static MojoValue bufferDataPointer(EmitMojoState &state, Value buffer) {
  return OpaquePointerRef(
             ManagedTensorSliceRef(state.getMojoValueFor(buffer)).unsafePtr())
      .bitcast(MojoValue(kComptimeDTypeNoneType))();
}

/// Emits the call to the `mogg_tensor_init`-family primitive `primitiveName`,
/// binding `op`'s single result.
static void emitTensorInitPrimitiveCall(EmitMojoState &state, Operation *op,
                                        StringRef primitiveName,
                                        Type resultType, bool mut,
                                        MO::IOKind io, ValueRange dims,
                                        MojoValue storage) {
  auto tensorTy = MO::dynCastFromTensorLike(resultType);
  MojoValue dTypeParam = state.mlirDTypeToMojoType(tensorTy.getDtype());
  auto ioParam = MojoValue(kStructIO).get(MO::stringifyIOKind(io));
  MojoValue rankParam =
      MojoValue(static_cast<uint64_t>(tensorTy.getRankNonZero()));
  MojoValue alignmentParam(
      static_cast<uint64_t>(M::Driver::getPreferredAlignment(
          MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
  // Encode the shape as a `Coord` (static dims as `Idx[N]`, dynamic dims as
  // `Scalar[DType.int](<dim>)`) and build a row-major `Layout`, whose Coord
  // element types convey the static/dynamic structure to `mogg_tensor_init`.
  auto shapeVar = state.file->var(
      state.getShapeAsCoordMojoValue(tensorTy.getShape(), dims));
  auto layoutVar = state.file->var(MojoValue(kFnRowMajor)({}, {shapeVar}));
  auto primDecl = state.registry.lookupPrimitive(primitiveName);
  ASSERT_STREAM(primDecl, "Primitive not found for " << primitiveName.str());
  MojoValue primitive(primitiveMojoName(primDecl));
  state.setMojoValueFor(
      op->getResult(0),
      state.file->var(primitive(
          {dTypeParam, rankParam, MojoValue(mut), ioParam, alignmentParam},
          {std::move(storage), layoutVar})));
}

LogicalResult TensorInitOp::emitMojo(EmitMojoState &state) {
  // The result type, not the op, decides the kernel-facing tensor: a stable
  // `!mo.tensor` is a `StableTensor` that reads the slot the buffer points at.
  auto tensorTy = MO::dynCastFromTensorLike(getType());
  StringRef primitiveName = tensorTy.isStable()
                                ? StringRef(kMoggPrimitiveStableTensorInit)
                                : getOperationName();
  emitTensorInitPrimitiveCall(state, *this, primitiveName, getType(), getMut(),
                              getIo(), getDims(),
                              bufferDataPointer(state, getBuffer()));
  return success();
}

//===----------------------------------------------------------------------===//
// TensorArgInitOp
//===----------------------------------------------------------------------===//

static constexpr llvm::StringLiteral kDenseTensorArgAlias =
    "_MoggDenseTensorArg";

static constexpr llvm::StringLiteral kStableTensorArgAlias =
    "_MoggStableTensorArg";

// `_MoggShapeTensorArg`: shape functions run host-only, so this struct
// skips `_MoggTensorArg`'s device-passable machinery. `_data_visible`
// gates off `TileTensorable` for a bare `Tensor` bound.
// Emitted once per module, guarded by
// `EmitMojoState::declare`.
static constexpr llvm::StringLiteral kShapeTensorArgStructName =
    "_MoggShapeTensorArg";

/// Emit `_MoggShapeTensorArg[dtype, layoutType, alignment, dataVisible](ptr,
/// layout)`, the host-only shape-lane counterpart, defined once per module.
static MojoValue emitShapeTensorArg(EmitMojoState &state, MojoValue dtype,
                                    MojoValue layoutType, MojoValue alignment,
                                    MojoValue dataVisible, MojoValue ptr,
                                    MojoValue layout) {
  state.declare(kShapeTensorArgumentDefinitions);
  return state.file->var(MojoValue(kShapeTensorArgStructName.str())(
      {dtype, layoutType, alignment, dataVisible}, {ptr, layout}));
}

/// Emit `<alias>[params...](ptr, layout)` for an execute-lane trait tensor
/// argument, defining the shared struct and its lane aliases once per module.
/// `alias` is `kDenseTensorArgAlias` (`[dtype, layoutType, alignment, io,
/// devicePassable]`) or `kStableTensorArgAlias`
/// (`[dtype, layoutType, alignment, io]`). The runtime args are a
/// `Pointer[NoneType, MutAnyOrigin]` and the layout; callers source them from
/// a stub-bound capture or a codegen slice (via `mogg.trait.create`). For a
/// stable argument `ptr` is the slot address.
static MojoValue emitTensorArg(EmitMojoState &state, StringRef alias,
                               ArrayRef<MojoValue> params, MojoValue ptr,
                               MojoValue layout) {
  state.declare(kTensorArgumentDefinitions);
  return state.file->var(MojoValue(alias.str())(params, {ptr, layout}));
}

LogicalResult TensorArgInitOp::emitMojo(EmitMojoState &state) {
  auto tensorTy = MO::dynCastFromTensorLike(getType());
  if (!tensorTy)
    return emitOpError("expected a tensor-like result type");

  MojoValue dTypeParam = state.mlirDTypeToMojoType(tensorTy.getDtype());
  auto shapeVar = state.file->var(
      state.getShapeAsCoordMojoValue(tensorTy.getShape(), getDims()));
  auto layoutVar = state.file->var(MojoValue(kFnRowMajor)({}, {shapeVar}));
  MojoValue layoutType = MojoValue("type_of")({}, {layoutVar});

  // Shape-function args assemble the host-only `_MoggShapeTensorArg`;
  // unlike the execute lane, there's no device-passable flavor to probe
  // for. `_data_visible` reflects the contract's written bound.
  if (getContract().getRole() == MOGG::TensorArgRoleKind::none) {
    MOGG::TensorArgBoundKind bound = getContract().getBound();
    if (bound == MOGG::TensorArgBoundKind::unspecified)
      return emitOpError("shape-function tensor argument has an unspecified "
                         "bound; expected tensor, dense, or tile");
    bool dataVisible = bound != MOGG::TensorArgBoundKind::tensor;
    MojoValue alignmentParam(
        static_cast<uint64_t>(M::Driver::getPreferredAlignment(
            MO::getDeviceOrDefaultAny(tensorTy).getLabel().str())));
    MojoValue ptr = OpaquePointerRef(ManagedTensorSliceRef(
                                         state.getMojoValueFor(getBuffer()))
                                         .unsafePtr())
                        .bitcast(MojoValue(kComptimeDTypeNoneType))();
    state.setMojoValueFor(
        getResult(),
        emitShapeTensorArg(state, dTypeParam, layoutType, alignmentParam,
                           MojoValue(dataVisible), ptr, layoutVar));
    return success();
  }

  // Region→stub boundary: emit minimal buffer capture (pointer + layout).
  state.declare(kTensorBufferCaptureDefinitions);
  MojoValue rawUnsafePtr =
      ManagedTensorSliceRef(state.getMojoValueFor(getBuffer())).unsafePtr();
  MojoValue ptr = state.file->var(
      llvm::formatv("{0}.unsafe_bitcast[Scalar[{1}]]().unsafe_mut_cast[True]()"
                    ".unsafe_origin_cast[MutUntrackedOrigin]()",
                    rawUnsafePtr.text, dTypeParam.text)
          .str());
  MojoValue structName("_MoggTensorBufferCapture");
  state.setMojoValueFor(
      getResult(),
      state.file->var(structName({dTypeParam, layoutType}, {ptr, layoutVar})));

  return success();
}

LogicalResult TraitCreateOp::emitMojo(EmitMojoState &state) {
  // Re-present the codegen tensor (a `ManagedTensorSlice` from
  // `mogg.tensor.create`, or a view) as the trait argument, sourcing pointer
  // and layout off the slice so any producer works, including a non-row-major
  // view.
  MojoValue slice = state.getMojoValueFor(getTensor());
  MOGG::TensorArgContractAttr contract = getContract();

  // A stable `mogg.tensor.create` yields a `StableTensor` rather than a
  // slice: its layout is a field, and the pointer to hand on is the slot
  // address (`unsafe_ptr()` would read the data pointer, on the device only).
  // That address is the create's own pointer operand, typed as the legacy
  // `mogg_stable_tensor_init` receives it; the `StableTensor`'s slot field is
  // a pointer to pointer and does not convert to the wrapper's `ptr`.
  auto createOp = getTensor().getDefiningOp<TensorCreateOp>();
  bool stableSource = createOp && createOp.getStable();

  MojoValue dTypeParam(llvm::formatv("type_of({0}).dtype", slice.text).str());
  MojoValue layoutArg =
      state.file->var(stableSource ? StableTensorRef(slice).runtimeLayout()
                                   : slice.get("runtime_layout")());
  MojoValue layoutType = MojoValue("type_of")({}, {layoutArg});
  MojoValue ptrArg = state.file->var(
      stableSource
          ? llvm::formatv(
                "{0}.as_unsafe_any_origin()",
                OpaquePointerRef(state.getMojoValueFor(createOp.getPtr()))
                    .bitcast(MojoValue(kComptimeDTypeNoneType))()
                    .text)
                .str()
          : llvm::formatv(
                "{0}.unsafe_bitcast[NoneType]().as_unsafe_any_origin()",
                ManagedTensorSliceRef(slice).unsafePtr().text)
                .str());

  // Device is on the op (like `mogg.tensor.create`), so alignment and the
  // device-passable / tile-projection flavors are picked locally.
  StringRef deviceLabel = getDevice();
  MojoValue alignmentParam(static_cast<uint64_t>(
      M::Driver::getPreferredAlignment(deviceLabel.str())));

  // A role-less contract is the host-only shape lane; the written bound decides
  // whether element data is visible.
  if (contract.getRole() == MOGG::TensorArgRoleKind::none) {
    MOGG::TensorArgBoundKind bound = contract.getBound();
    if (bound == MOGG::TensorArgBoundKind::unspecified)
      return emitOpError("shape-function trait argument has an unspecified "
                         "bound; expected tensor, dense, or tile");
    bool dataVisible = bound != MOGG::TensorArgBoundKind::tensor;
    state.setMojoValueFor(
        getResult(),
        emitShapeTensorArg(state, dTypeParam, layoutType, alignmentParam,
                           MojoValue(dataVisible), ptrArg, layoutArg));
    return success();
  }

  MojoValue ioParam = [&] {
    MojoValue ioStruct("_MoggTensorIO");
    switch (contract.getRole()) {
    case MOGG::TensorArgRoleKind::input:
      return ioStruct.get("INPUT");
    case MOGG::TensorArgRoleKind::output:
      return ioStruct.get("OUTPUT");
    case MOGG::TensorArgRoleKind::mutable_input:
      return ioStruct.get("MUTABLE_INPUT");
    case MOGG::TensorArgRoleKind::none:
      llvm_unreachable("shape lane handled above");
    }
    llvm_unreachable("unhandled TensorArgRoleKind");
  }();
  // A `Stable` argument wraps the slice's pointer as the slot address (the
  // slice over a stable `!mgp.tensor` carries the slot, not the data).
  if (contract.getStable()) {
    if (deviceLabel == kCPULabel)
      return emitOpError("Stable trait argument requires a device kernel; a "
                         "stable address slot cannot be read on the host");
    state.setMojoValueFor(
        getResult(),
        emitTensorArg(state, kStableTensorArgAlias,
                      {dTypeParam, layoutType, alignmentParam, ioParam}, ptrArg,
                      layoutArg));
    return success();
  }

  bool devicePassable = deviceLabel != kCPULabel;
  state.setMojoValueFor(getResult(),
                        emitTensorArg(state, kDenseTensorArgAlias,
                                      {dTypeParam, layoutType, alignmentParam,
                                       ioParam, MojoValue(devicePassable)},
                                      ptrArg, layoutArg));
  return success();
}

// Projects a stub-call operand to the bare view the kernel consumes: a
// `!mgp.buffer` / `!mgp.tensor` value is a storage-carrying composite
// (`OwnedByteBuffer` / `OwnedTensor`), so unwrap its `.view` / `.tensor` before
// passing it to the kernel stub.
static SmallVector<MojoValue>
getStubCallOperands(EmitMojoState &state, Operation::operand_range ops) {
  SmallVector<MojoValue> operands;
  operands.reserve(llvm::range_size(ops));
  for (Value operand : ops) {
    MojoValue mojoVal = state.getMojoValueFor(operand);
    if (isa<MGP::DeviceBufferType>(operand.getType()))
      mojoVal = OwnedByteBufferRef(mojoVal).view();
    else if (isa<MGP::DeviceTensorType>(operand.getType()))
      mojoVal = OwnedTensorRef(mojoVal).tensor();
    operands.push_back(mojoVal);
  }
  return operands;
}

LogicalResult CallOp::emitMojo(EmitMojoState &state) {
  state.file->newline();
  MojoValue func(getName());
  if (getNumResults()) {
    state.setMojoValueFor(
        getResult(0),
        state.file->var(func({}, getStubCallOperands(state, getOperands()))));
  } else {
    state.file->call(func, {}, getStubCallOperands(state, getOperands()));
  }
  state.file->newline();

  return success();
}

LogicalResult CallAsyncOp::emitMojo(EmitMojoState &state) {
  state.file->newline();

  // 1. Create error slot: var error_N = Optional[Error]()
  MojoValue errorVar = state.file->var(MojoValue(kComptimeOptionalErrorEmpty),
                                       /*cache=*/false);

  // 2. Create async wrapper closure that calls the stub
  SmallVector<MojoValue> stubArgs = getStubCallOperands(state, getOperands());
  MojoValue stubFunc(getCallee().getSymNameAttr().str());

  // TODO(MOCO-5158): an async closure whose captures are all
  // register-passable is invoked with a pointer to a by-value context.
  // Capturing a String makes the context memory-only; drop it with the fix.
  MojoValue memoryOnly =
      state.file->var(MojoValue(kStructString)({}, {}), /*cache=*/false);
  std::string captures =
      llvm::formatv("var {0}, mut {1}, imm", memoryOnly.text, errorVar.text)
          .str();
  MojoValue wrapper = state.file->lambda(
      "wrapper", /*parameters=*/{}, /*arguments=*/{},
      /*returnType=*/MojoValue(kComptimeNone),
      MojoBuilder::LambdaInline::kAlways, /*isAsync=*/true,
      /*raises=*/false, captures, [&](const FunctionBody &) {
        state.file->keepAlive(memoryOnly);
        state.file
            ->try_([&]() {
              state.file->call(stubFunc, /*parameters=*/{}, stubArgs);
            })
            .except([&](const MojoValue &errVar) {
              state.file->assign(errorVar, MojoValue(errVar.text + "^"));
            });
      });

  // 3. Launch the task. GPU launches pin it to the device's NUMA-affine CPU
  // worker via `_create_task`; otherwise `create_task` (desired_worker_id=-1).
  // The device ID is `$affinity_device_id` (I64Attr) because codegen device
  // refs are label-only.
  MojoValue taskVar;
  auto affinityId = getAffinityDeviceIdAttr();
  if (affinityId) {
    int64_t deviceId = affinityId.getInt();
    // var worker_id_N = max.runtime.asyncrt.task_id_for_device(<deviceId>)
    MojoValue workerIdVar =
        state.file->var(MojoValue(kFnTaskIdForDevice)(
                            /*params=*/{}, /*args=*/{MojoValue(deviceId)}),
                        /*cache=*/false);
    // var task_N = std.runtime._asyncrt._create_task(
    //     wrapper_N(), desired_worker_id=worker_id_N)
    // MojoBuilder has no built-in keyword-argument support, so we construct
    // the call text directly.
    taskVar = state.file->var(
        MojoValue(llvm::formatv("{}({}, desired_worker_id={})",
                                kFnCreateTaskWithAffinity, wrapper({}, {}).text,
                                workerIdVar.text)
                      .str()),
        /*cache=*/false);
  } else {
    // var task_N = std.runtime._asyncrt.create_task(wrapper_N())
    taskVar = state.file->var(MojoValue(kFnCreateTask)({}, {wrapper({}, {})}),
                              /*cache=*/false);
  }

  // 4. Record task handle info for AsyncJoinOp to consume
  state.taskHandleInfos[getTaskHandle()] = {taskVar, errorVar, wrapper};

  state.file->newline();
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncReadyOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncReadyOp::emitMojo(EmitMojoState &state) {
  // Produce a fresh chain AnyAsyncValueRef.
  MojoValue ready = state.file->var(
      MojoValue(kFnMoggAsyncReadyValue)(/*parameters=*/{}, /*operands=*/{}));
  state.setMojoValueFor(getResult(), ready);
  return success();
}

LogicalResult AsyncJoinOp::emitMojo(EmitMojoState &state) {
  // Wait for all tasks to complete before checking any errors.
  for (Value taskHandle : getTaskHandles()) {
    auto it = state.taskHandleInfos.find(taskHandle);
    if (it == state.taskHandleInfos.end())
      return emitOpError("missing TaskHandleInfo for operand");
    state.file->keepAlive(MojoTaskRef(it->second.task).wait());
  }
  for (Value taskHandle : getTaskHandles())
    state.file->keepAlive(state.taskHandleInfos[taskHandle].wrapper);

  auto checkTaskErrPrimDecl =
      state.registry.lookupPrimitive(getOperationName());
  MojoValue checkErrorFunc(primitiveMojoName(checkTaskErrPrimDecl));

  for (Value taskHandle : getTaskHandles()) {
    auto &info = state.taskHandleInfos[taskHandle];
    state.file->call(checkErrorFunc, {}, {info.error});
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ChainDummyOp
//===----------------------------------------------------------------------===//

LogicalResult ChainDummyOp::emitMojo(EmitMojoState &) {
  // Do nothing because the consuming op should never be using the chain.
  return success();
}

//===----------------------------------------------------------------------===//
// ChainSinkOp
//===----------------------------------------------------------------------===//

LogicalResult ChainSinkOp::emitMojo(EmitMojoState &) {
  // The producing op never returns a chain.
  return success();
}

LogicalResult CallElementwiseOp::emitMojo(EmitMojoState &state) {
  SmallVector<MojoValue> inputs = state.getMojoValuesFor(getInputs());

  // `$index` (present only if `CallElementwiseOp::lowerToMOGG` wanted it) is a
  // `layout.Coord`, the only form `CallForeachOp::emitMojo`'s closure produces,
  // so convert it to the kernel's `IndexList`.
  if (Value index = getIndex())
    inputs.push_back(
        MojoValue(kFnCoordToIndexList)({}, {state.getMojoValueFor(index)}));

  KernelDeclAdaptor ewAdaptor(
      state.registry.lookupKernelBySymName(getCallee()));
  FailureOr<MojoValue> call = emitElementwiseCallExpr(
      state, *this,
      ewAdaptor.isElementwise() ? KernelFunctionKind::Elementwise
                                : KernelFunctionKind::Execute,
      inputs);
  if (failed(call))
    return failure();
  state.setMojoValueFor(getOutputs().front(), state.file->var(*call));
  return success();
}

//===----------------------------------------------------------------------===//
// EmptyBufferOp
//===----------------------------------------------------------------------===//

LogicalResult EmptyBufferOp::emitMojo(EmitMojoState &state) {
  // Represent the empty buffer with a dangling pointer.
  MojoValue danglingPointer =
      MojoValue(
          kStructPointer)[{MojoValue(kStructScalar)[{
                               MojoValue(kStructDType).get(kEnumDTypeInt8)}],
                           MojoValue(kComptimeMutAnyOrigin)}]
          .get(OpaquePointerRef::kUnsafeDangling)();
  MojoValue zeroShape =
      MojoValue(kStructIndexList)({MojoValue("1")}, {MojoValue("0")});
  state.setMojoValueFor(getResult(),
                        state.file->var(MojoValue(kComptimeMutByteBuffer)(
                            {}, {danglingPointer, zeroShape})));
  return success();
}

LogicalResult CallShapeOp::emitMojo(EmitMojoState &state) {
  SmallVector<MojoValue> kernelOperands;
  for (Value operand : getOperands())
    kernelOperands.push_back(state.getMojoValueFor(operand));

  // Infer any kernel parameters required by the shape function.
  auto inferParam = [&](StringRef paramName) -> ErrorOr<MojoValue> {
    return state.inferDefaultKernelParamValue(*this, paramName,
                                              getKernelParamAttr());
  };
  ShapeFunctionDeclOp shapeDecl =
      state.registry.lookupShapeFunctionBySymName(getCallee());
  auto maybeParams = inferKernelParameters(shapeDecl, inferParam);
  if (maybeParams.isError())
    return emitError() << "Failed to infer all the kernel parameters: "
                       << maybeParams.takeError();

  MojoValue shapeFn(getCallee());
  state.setMojoValueFor(getResult(),
                        state.file->var(shapeFn[maybeParams->structParameters](
                            maybeParams->functionParameters, kernelOperands)));
  state.genAllKeepAliveStatements();
  return success();
}

LogicalResult ShapeResultGetItemOp::emitMojo(EmitMojoState &state) {
  const MojoValue &shapeValue = state.getMojoValueFor(getShape());
  int64_t index = getIndex().getValue().getSExtValue();

  MojoValue indexValue = MojoValue(std::to_string(index));
  state.setMojoValueFor(getResult(), shapeValue[{indexValue}]);

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantScalarOp
//===----------------------------------------------------------------------===//

/// Convert a scalar-constant attribute (IntegerAttr or FloatAttr) to a Mojo
/// literal an elementwise call can splat (e.g.
/// `Add.elementwise[dtype = ...](lhs_simd, 2.0)`), or std::nullopt.
static std::optional<MojoValue>
scalarConstantAttrToMojoLiteral(TypedAttr attr) {
  if (auto intScalar = MO::tryGetConstantInt(attr))
    return MojoValue(*intScalar);
  // `tryGetConstantFloat` returning a value guarantees `attr` is a FloatAttr.
  auto floatScalar = MO::tryGetConstantFloat(attr);
  if (!floatScalar)
    return std::nullopt;
  auto floatAttr = cast<mlir::FloatAttr>(attr);

  // inf/nan have no Mojo float-literal spelling, so construct them via the
  // numerics helpers parameterized on the scalar's dtype.
  if (std::isinf(*floatScalar) || std::isnan(*floatScalar)) {
    ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(floatAttr.getType());
    if (dtypeOr.isError())
      return std::nullopt;
    MojoValue dtype = EmitMojoState::mlirDTypeToMojoType(*dtypeOr);
    if (std::isnan(*floatScalar))
      return MojoValue(kFnNumericsNan)[{dtype}]();
    MojoValue inf = MojoValue(kFnNumericsInf)[{dtype}]();
    if (std::signbit(*floatScalar))
      return MojoValue("-" + inf.text);
    return inf;
  }

  // Render the finite value as a Mojo float literal (formatting lives in the
  // MojoValue APFloat constructor).
  return MojoValue(floatAttr.getValue());
}

LogicalResult ConstantScalarOp::emitMojo(EmitMojoState &state) {
  std::optional<MojoValue> literal =
      scalarConstantAttrToMojoLiteral(getValue());
  if (!literal)
    return emitError("TODO: unsupported scalar type");

  // Splat a SIMD constant to the closure's width: Mojo infers an elementwise
  // call's width from its SIMD operands, so an unsplatted first-operand
  // constant would fail inference. A tile call's literal stays bare.
  if (auto simd = dyn_cast<SimdType>(getResult().getType())) {
    ErrorOr<DType> dtypeOr = M::MO::fromMLIRType(simd.getElementType());
    ASSERT_STREAM(!dtypeOr.isError(), "verified element type " << simd);
    MojoValue simdTy = EmitMojoState::getSIMDType(
        *dtypeOr, state.currentRegion<ElementwiseLambdaRegion>().width());
    literal = simdTy({}, {*literal});
  }

  state.setMojoValueFor(getResult(), *literal);
  return success();
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::emitMojo(EmitMojoState &state) {
  // Only the MGP ABI hands results back through the `void**` slot vector, so
  // only it populates `state.mojoReturnValues`.
  auto kernelStub = getOperation()->getParentOfType<MOGG::FuncOp>();
  if (!kernelStub || kernelStub.getAbi() != MOGG::ABIKind::MGP)
    return success();

  // `mogg.output` operands are the async-returned outputs, in slot order. Each
  // producer (mogg.async.pack/ready/error) has emitted an AnyAsyncValueRef
  // value; assign them into the corresponding output slots here.
  for (auto [idx, operand] : llvm::enumerate(getOperands())) {
    MojoValue value = state.getMojoValueFor(operand);
    state.file->call(MojoValue(kFnMoggAssignAsyncValue), {},
                     {state.mojoReturnValues[idx], value});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::emitMojo(EmitMojoState &state) {
  // Routes a region's produced value to one of the parent's results:
  // `mogg.try` forwarding a branch's, or `mogg.with_trace` forwarding a traced
  // dispatch's. Its other parents carry nothing out this way.
  Operation *parent = getOperation()->getParentOp();
  if (!isa<MOGG::TryOp, MOGG::WithTraceOp>(parent))
    return success();

  for (auto [operand, result] :
       llvm::zip_equal(getOperands(), parent->getResults())) {
    MojoValue value = state.getMojoValueFor(operand);
    // An opaque struct need not be `ImplicitlyCopyable` (`SIMDPair` is only
    // `RegisterPassable`), so transfer it into the result variable rather than
    // copying -- as `emitMojoMOGGABI` does when returning one.
    if (isa<MO::OpaqueType>(operand.getType()))
      value = value.transfer();
    state.file->assign(state.getMojoValueFor(result), value);
  }

  return success();
}

/// Renders a value crossing a stub boundary, where an opaque is the kernel's
/// own struct rather than the `Pointer[NoneType]` `mlirTypeToMojoType` erases
/// it to. Same mapping `emitMojoMOGGABI` builds inline.
static MojoValue boundaryTypeToMojo(EmitMojoState &state, Type type) {
  auto opaqueType = dyn_cast<MO::OpaqueType>(type);
  if (!opaqueType)
    return state.mlirTypeToMojoType(type);
  MojoValue baseTy = state.mlirOpaqueTypeToMojoType(opaqueType);
  auto params = EmitMojoState::getOpaqueTypeParameters(opaqueType);
  return params.empty() ? baseTy : baseTy[params];
}

LogicalResult WithTraceOp::emitMojo(EmitMojoState &state) {
  // Find the enclosing FuncOp to get the device and fusion info.
  auto kernelStubOp = getOperation()->getParentOfType<MOGG::FuncOp>();
  if (!kernelStubOp)
    return emitOpError("WithTraceOp must be nested in a FuncOp");

  std::string deviceLabel = kernelStubOp.getDevice().str();
  // Distinct chains mean a kernel body shared by two non-nested
  // profile_scope()s (MergeIdenticalFuncs merges their locations);
  // formatProfileScope marks that with `:` rather than `/`. The scope follows
  // the kernel name (unlike AddTryExceptBlocks' `[scope] __<stub>`) so the
  // fusion-info name leads the trace.
  std::string traceName =
      kernelStubOp.getFusionInfoAttr().toString(/*includeLineNumbers=*/false,
                                                /*quote=*/false,
                                                /*stripDialectPrefixes=*/true) +
      formatProfileScope(collectProfileScopeLayers(kernelStubOp.getLoc()),
                         false);

  // Build compact source traceback from FusedLoc metadata set by
  // AttachPythonStackTraces, to surface Python call sites in NVTX markers.
  std::string sourceDetail = collectCompactSourceNotes(kernelStubOp.getLoc());

  // Find the context argument (if present) to include task_id in Trace.
  MojoValue contextMojoArg;
  for (auto mlirArg : kernelStubOp.getBody().getArguments()) {
    if (isa<MOGG::ContextType>(mlirArg.getType())) {
      contextMojoArg = state.getMojoValueFor(mlirArg);
      break;
    }
  }

  // Build: Trace[TraceLevel.OP, target=StaticString("device")]("name"[,
  // detail=..., task_id=...])
  SmallVector<MojoValue> traceParams = {
      MojoValue(kEnumTraceLevelOp),
      MojoValue(kMoggTargetParameterName)
          .equal(MojoValue(kStructStaticString)(
              {}, {MojoValue(deviceLabel).quote()}))};
  SmallVector<MojoValue> traceOperands = {MojoValue(traceName).quote()};
  if (!sourceDetail.empty())
    traceOperands.push_back(MojoValue("\n" + sourceDetail).quote());
  if (!contextMojoArg.text.empty())
    traceOperands.push_back(
        MojoValue(kParameterTaskId)
            .equal(MojoValue(kFnGetSafeTaskId)({}, {contextMojoArg})));
  MojoValue traceExpr = MojoValue(kStructTrace)(traceParams, traceOperands);

  // Declare a variable per result before the `with`, so the yield inside can
  // assign it and the op consuming this trace's results still sees it in scope.
  // Same shape `TryOp::emitMojo` uses for a non-MGP ABI.
  for (OpResult result : getResults())
    state.setMojoValueFor(result, state.file->varDecl(boundaryTypeToMojo(
                                      state, result.getType())));

  bool errOccurred = false;
  state.file->with_(traceExpr, [&]() {
    for (auto &op : getBody().getOps()) {
      if (state.lowerOpToMojo(&op).failed()) {
        errOccurred = true;
        return;
      }
    }
  });

  if (!errOccurred)
    state.genAllKeepAliveStatements();

  return failure(errOccurred);
}

LogicalResult ProfileScopeBeginOp::emitMojo(EmitMojoState &state) {
  // Build the Trace object: Trace[TraceLevel.OP, target=StaticString(device)]
  // ("[scope] region_name") and call start() on it.
  auto kernelStubOp = getOperation()->getParentOfType<MOGG::FuncOp>();
  if (!kernelStubOp)
    return emitOpError("ProfileScopeBeginOp must be nested in a FuncOp");

  std::string deviceLabel = kernelStubOp.getDevice().str();

  SmallVector<MojoValue> traceParams = {
      MojoValue(kEnumTraceLevelOp),
      MojoValue(kMoggTargetParameterName)
          .equal(MojoValue(kStructStaticString)(
              {}, {MojoValue(deviceLabel).quote()}))};
  SmallVector<MojoValue> traceOperands = {MojoValue(getName()).quote()};
  if (auto color = getColor())
    traceOperands.push_back(
        MojoValue(kParameterColor)
            .equal(MojoValue(kStructColor)({}, {MojoValue(*color).quote()})));
  MojoValue traceExpr = MojoValue(kStructTrace)(traceParams, traceOperands);

  MojoValue traceVar = state.file->var(traceExpr, /*cache=*/false);
  state.setMojoValueFor(getRange(), traceVar);
  state.file->call(MojoValue(traceVar).get(kFnTraceStart), {}, {});
  return success();
}

LogicalResult ProfileScopeEndOp::emitMojo(EmitMojoState &state) {
  MojoValue traceVar = state.getMojoValueFor(getRange());
  state.file->call(MojoValue(traceVar).get(kFnTraceEnd), {}, {});
  return success();
}

LogicalResult RaiseOp::emitMojo(EmitMojoState &state) {
  // Find the enclosing FuncOp for kernel context (name, fusion info,
  // source tracebacks).
  auto kernelStubOp = getOperation()->getParentOfType<MOGG::FuncOp>();
  if (!kernelStubOp)
    return emitOpError("RaiseOp must be nested in a FuncOp");

  std::string stubName = kernelStubOp.getSymNameAttr().str();

  std::string fusionInfoContent =
      kernelStubOp.getFusionInfoAttr().toPrettyTree({}, /*indent=*/true);

  // Build source traceback content from FusedLoc metadata set by
  // AttachPythonStackTraces.
  auto notesList = collectSourceNotes(kernelStubOp.getLoc());
  std::string tracebackContent;
  if (!notesList.empty()) {
    llvm::raw_string_ostream os(tracebackContent);
    constexpr size_t kMaxInlineTraces = 32;
    size_t numShown = std::min(notesList.size(), kMaxInlineTraces);
    for (size_t i = 0; i < numShown; ++i)
      os << notesList[i] << '\n';
    if (notesList.size() > 1) {
      os << "\nNote: " << notesList.size()
         << " distinct Python source locations contributed ops to this fused"
            " kernel. ";
      if (numShown == notesList.size()) {
        os << "The source traceback for each location is above.\n";
      } else {
        os << "Only the first " << kMaxInlineTraces
           << " source tracebacks are shown above.\n";
      }
    }
  }

  auto fmtErrPrimDecl =
      state.registry.lookupPrimitive(this->getOperationName());
  ASSERT_STREAM(fmtErrPrimDecl,
                "Primitive not found for " << getOperationName().str());
  MojoValue formatKernelErrorFunc(primitiveMojoName(fmtErrPrimDecl));
  MojoValue errorExpr = state.getMojoValueFor(getError());
  SmallVector<MojoValue> args = {
      MojoValue(stubName).quote(),
      errorExpr,
      MojoValue(fusionInfoContent).escape(),
      MojoValue(tracebackContent).escape(),
  };
  auto formattedErr = state.file->var(formatKernelErrorFunc({}, args));
  state.file->raise(formattedErr.transfer());

  return success();
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

LogicalResult TryOp::emitMojo(EmitMojoState &state) {
  // Declare result variables for both branches' yields and the consuming
  // mogg.output. MGP forwards async values (its except block sets an error on
  // each), so those are `AnyAsyncValueRef`; other ABIs forward the result type
  // and raise instead, so only the try branch assigns.
  auto stubOp = getOperation()->getParentOfType<MOGG::FuncOp>();
  bool isMGPAbi = stubOp && stubOp.getAbi() == MOGG::ABIKind::MGP;
  for (OpResult result : getResults()) {
    MojoValue resultVar =
        isMGPAbi
            ? state.file->var(MojoValue(kStructAnyAsyncValueRef)(),
                              /*cache=*/false)
            : state.file->varDecl(boundaryTypeToMojo(state, result.getType()));
    state.setMojoValueFor(result, resultVar);
  }

  // MOGG ABI stubs catch, format with kernel context, and re-raise; this only
  // emits the try/except structure. AsyncErrorOp::emitMojo adds the region
  // name.
  bool errOccurred = false;
  state.file
      ->try_([&]() {
        for (auto &op : getTryRegion().getOps()) {
          if (state.lowerOpToMojo(&op).failed()) {
            errOccurred = true;
            return;
          }
        }
      })
      .except([&](const MojoValue &e) {
        state.setMojoValueFor(getExceptRegion().getArgument(0), e);
        errOccurred |=
            llvm::any_of(getExceptRegion().getOps(), [&](mlir::Operation &op) {
              return state.lowerOpToMojo(&op).failed();
            });
      });

  return failure(errOccurred);
}

//===----------------------------------------------------------------------===//
// AsyncErrorOp
//===----------------------------------------------------------------------===//

LogicalResult AsyncErrorOp::emitMojo(EmitMojoState &state) {
  MojoValue errorExpr = state.getMojoValueFor(getError());

  // Get the parent MGP region name for the error header.
  std::string regionName;
  if (auto parentKernel = getOperation()->getParentOfType<MOGG::FuncOp>())
    regionName = parentKernel.getSymNameAttr().str();

  // Format the error with the region name prefix.
  auto fmtRegionErrPrimDecl =
      state.registry.lookupPrimitive(kMoggPrimitiveFormatRegionError);
  ASSERT_STREAM(fmtRegionErrPrimDecl,
                "format_region_error primitive not found");
  MojoValue formatRegionErrorFunc(primitiveMojoName(fmtRegionErrPrimDecl));
  MojoValue formattedErr = state.file->var(
      formatRegionErrorFunc({}, {MojoValue(regionName).quote(), errorExpr}));

  // Produce a fresh error AnyAsyncValueRef. The actual output slot assignment
  // is emitted by the mogg.output terminator (which consumes the TryOp result).
  MojoValue errorValue =
      state.file->var(MojoValue(kFnMoggAsyncErrorValue)({}, {formattedErr}));
  state.setMojoValueFor(getResult(), errorValue);

  return success();
}

//===----------------------------------------------------------------------===//
// ShapeInitOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeInitOp::emitMojo(EmitMojoState &state) {
  MojoValue shapeValue =
      state.getShapeAsIndexListMojoValue(getShape(), getDynamicDims());
  state.setMojoValueFor(getResult(), shapeValue);
  return success();
}
