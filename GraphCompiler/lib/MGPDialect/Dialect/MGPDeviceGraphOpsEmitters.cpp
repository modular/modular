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
// Device-graph ops of the MGP dialect, declared in MGPDeviceGraphOps.td.
// This file holds their Mojo emission.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

using namespace M;
using namespace MGP;

/// Is `v` a device-resident graph input rather than a context operand?
/// (`inputs` also carries `!mgp.context` and `!mgp.device_context` values.)
///
/// Must agree with `graphInputData` in MGPPrimitives.cpp, or inputs pair with
/// the wrong stable buffers at replay. Mutable inputs register a null in-place
/// marker so every device input keeps one position in `graph->getInputs()`.
static bool isDeviceGraphInput(Value v) {
  return graphInputDevice(v) != nullptr;
}

// Emits `var cache_N = Pointer(to=<ctx>.get_device_graph_cache())`.
// `DeviceGraphCache` is not copyable, so the returned reference cannot be bound
// to a `var` directly; `DeviceGraph.create` takes it by pointer.
LogicalResult DeviceGraphCacheOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  MojoValue theContext = state.getMojoValueFor(getTheContext());
  MojoValue cache = state.file->var(kStructPointer(/*operands=*/{
      kParameterTo.equal(StateContextRef(theContext).getDeviceGraphCache())}));
  state.setMojoValueFor(getOutCache(), cache);
  return success();
}

// Packs each yielded value into an `AsyncValue` and passes it to `add_output`
// so its backing memory outlives the graph that references it.
LogicalResult DeviceGraphYieldOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  // The builder is the first block argument of the enclosing `create` body.
  Value builderArg = getOperation()->getBlock()->getArgument(0);
  DeviceGraphBuilderRef builder(state.getMojoValueFor(builderArg));

  for (Value output : getOutputs()) {
    MojoValue outVal = state.getMojoValueFor(output);
    MojoValue asyncValue = outVal.get(kFnAsyncPack)(/*operands=*/{});
    state.file->call(builder.addOutput(),
                     /*parameters=*/{}, /*operands=*/{asyncValue});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphAllocOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphAllocOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  // The primitive's `sym_name` (set by ImportPackages) is the Mojo function.
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primDecl.getSymNameAttr().strref());

  DeviceRefAttr hostDevice = M::getHostDevice(getContext());
  MojoValue isHostParam(getOutBuffer().getType().getDevice() == hostDevice);
  MojoValue result = state.file->var(
      function[{isHostParam}]({state.getMojoValueFor(getRuntimeSize()),
                               state.getMojoValueFor(getGraphBuilder())}));
  state.setMojoValueFor(getOutBuffer(), result);
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphRecordingContextOp
//===----------------------------------------------------------------------===//

// Emits `var ctx_N = builder.recording_context()`, which execute-only kernels
// dispatch their `execute` against.
LogicalResult
DeviceGraphRecordingContextOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue result = state.file->var(builder.recordingContext());
  state.setMojoValueFor(getContext(), result);
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphDeviceContextOp
//===----------------------------------------------------------------------===//

// Emits `var ctx_N = builder.context()`, a non-raising field read.
LogicalResult DeviceGraphDeviceContextOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue result = state.file->var(builder.context());
  state.setMojoValueFor(getDeviceContext(), result);
  return success();
}

// Emits `var node_N = <builder>.region(work_N, dependencies=[...])`, where
// `work_N` is a recording closure wrapping the body. The body captures
// enclosing values lexically; only the builder block arg is a closure
// parameter. `dependencies=` makes the edges predecessors of the first node the
// body records (`_implicit_deps`); `region` chains the rest in body order.
//
// `region` returns only the body's last node, so each in-body result is
// surfaced through a captured outer `Optional[T]` var (the type may not be
// default-constructible, e.g. a non-null buffer pointer):
//
//   var out = Optional[T](None)
//   def work(mut builder) raises {mut}:
//     ...
//     out = <yielded>
//   var node = builder.region(work, dependencies=[...])
//   var resolved = out.take()
//   // uses of the op's body result now refer to `resolved`
LogicalResult DeviceGraphDependenciesOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  Value builderArg = getBuilderBlockArgument();
  DeviceGraphDependenciesYieldOp yieldOp = getYieldOp();

  // Partition the used body results into those produced inside the body (which
  // need a captured outer var) and external pass-throughs (bound directly).
  struct SurfacedResult {
    Value result;
    Value yielded;
    MojoValue outVar;
  };

  SmallVector<SurfacedResult> surfaced;
  for (auto [result, yielded] :
       llvm::zip_equal(getBodyResults(), yieldOp.getOperands())) {
    if (result.use_empty())
      continue;

    MojoValue optionalType =
        kStructOptional[EmitMojoState::mlirTypeToMojoType(result.getType())];
    MojoValue outVar = state.file->var(optionalType({kComptimeNone}),
                                       /*cache=*/false);
    surfaced.emplace_back(result, yielded, outVar);
  }

  NamedMojoDef builderParam =
      NamedMojoDef::asVerbatim("builder", kStructDeviceGraphBuilder);
  builderParam.argConvention = "mut";

  // Surfacing closures write their `Optional` vars, so capture by `mut`. One
  // that also reads register-passable captures would need per-capture
  // conventions; current single-dispatch bodies do not.
  llvm::StringRef workConvention =
      surfaced.empty() ? llvm::StringRef("imm") : llvm::StringRef("mut");

  bool errOccurred = false;
  MojoValue workClosure = state.file->lambda(
      "work", /*parameters=*/{}, /*arguments=*/{builderParam},
      /*returnType=*/std::nullopt,
      // Never inline: single-use per-kernel bodies otherwise merge into the
      // build closure (see the build lambda in DeviceGraphCreateOp).
      MojoBuilder::LambdaInline::kNever,
      /*isAsync=*/false, /*raises=*/true, /*captureConvention=*/workConvention,
      [&](const FunctionBody &functionBody) {
        state.setMojoValueFor(builderArg,
                              functionBody.arguments.front().resolvedName);
        for (Operation &op : getBodyBlock().getOperations())
          if (state.lowerOpToMojo(&op).failed()) {
            errOccurred = true;
            return;
          }
        for (const SurfacedResult &s : surfaced)
          state.file->assign(s.outVar,
                             state.getMojoValueFor(s.yielded).transfer());
      });
  if (errOccurred)
    return failure();

  SmallVector<MojoValue> depValues = state.getMojoValuesFor(getDependencies());
  MojoValue dependenciesList = state.file->listLiteral(depValues);

  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue node =
      state.file->var(builder.region(workClosure, dependenciesList));
  state.setMojoValueFor(getNode(), node);

  for (const SurfacedResult &s : surfaced)
    state.setMojoValueFor(
        s.result, state.file->var(s.outVar.get("take")(/*operands=*/{})));

  return success();
}

// The parent op's `emitMojo` surfaces the body results.
LogicalResult DeviceGraphDependenciesYieldOp::emitMojo(MOGG::EmitMojoState &) {
  return success();
}

LogicalResult DeviceGraphStableGetOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  state.setMojoValueFor(getResult(),
                        state.file->var(builder.getStable(getIndex())));
  return success();
}

LogicalResult DeviceGraphStablePlaceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  // The slot holds the tensor's data pointer. An `!mgp.tensor` value has no
  // pointer accessor of its own, so take it from the extracted buffer, as
  // `mogg.tensor.__init__` does.
  auto extractDecl =
      state.registry.lookupPrimitive(TensorExtractBufferOp::getOperationName());
  MojoValue extractBuffer(extractDecl.getSymNameAttr().strref());
  MojoValue buffer =
      state.file->var(extractBuffer({state.getMojoValueFor(getTensor())}));
  MojoValue dataPtr =
      OpaquePointerRef(ManagedTensorSliceRef(buffer).unsafePtr())
          .bitcast(kComptimeDTypeNoneType)();

  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  state.setMojoValueFor(getPlacedAddr(), state.file->var(builder.placeStable(
                                             getIndex(), dataPtr)));
  return success();
}

LogicalResult DeviceGraphStableTensorOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primDecl.getSymNameAttr().strref());

  // Parameterized like `mgp.tensor.create`: the result is an ordinary
  // `OwnedTensor` whose data pointer happens to be the slot.
  DeviceTensorType tensorType = getResult().getType();
  MojoValue tensorRankParam = tensorType.getRank();
  MojoValue bufferRankParam =
      tensorType.getRank() == 0 ? MojoValue{1} : tensorRankParam;
  MojoValue dtypeParam = state.mlirDTypeToMojoType(
      tensorType.getTensorSpec().getOptDType().value());

  SmallVector<MojoValue> params = {tensorRankParam, bufferRankParam,
                                   dtypeParam};
  // `kind` names how the tensor participates as a device graph input; the
  // Mojo default (`data`) is left implicit.
  if (tensorType.getMut())
    params.push_back(kParameterKind.equal(
        kStructDeviceGraphInputKind.get(kInputKindMutable)));
  else if (M::isHostPlaced(tensorType.getDevice()))
    params.push_back(
        kParameterKind.equal(kStructDeviceGraphInputKind.get(kInputKindHost)));

  state.setMojoValueFor(
      getResult(),
      state.file->var(function[params](state.getMojoValuesFor(getOperands()))));
  return success();
}

/// Emits the build closure, shared by the cached and uncached paths, that
/// dispatches to the body of region `n` for a `DeviceGraphBuilder`.
static FailureOr<MOGG::MojoValue>
emitCollectiveBuildForClosure(MOGG::EmitMojoState &state,
                              DeviceGraphCollectiveCreateOp op) {
  using namespace MOGG;

  NamedMojoDef nParam("n", "Int");
  NamedMojoDef builderParam =
      NamedMojoDef::asVerbatim("builder", kStructDeviceGraphBuilder);
  builderParam.argConvention = "mut";

  MojoValue sharedContext = state.getMojoValueFor(op.getTheContext());

  llvm::SmallBitVector shapeOnlyInputs = op.shapeOnlyInputs();

  bool errOccurred = false;
  MojoValue buildFor = state.file->lambda(
      "collective_build_for", /*parameters=*/{nParam},
      /*arguments=*/{builderParam},
      /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kNever,
      /*isAsync=*/false, /*raises=*/true,
      /*captureConvention=*/"imm", [&](const FunctionBody &buildBody) {
        MojoValue n = buildBody.parameters.front().resolvedName;
        MojoValue builderArgMojo = buildBody.arguments.front().resolvedName;

        for (Region &region : op.getGraphs()) {
          state.setMojoValueFor(region.front().getArgument(1), sharedContext);
          state.setMojoValueFor(region.front().getArgument(0), builderArgMojo);
        }

        size_t inputOffset = 0;
        MojoBuilder *branch = nullptr;
        for (auto [idx, region] : llvm::enumerate(op.getGraphs())) {
          Block &block = region.front();
          size_t numRegionInputs = block.getNumArguments() - 2;
          auto regionInputs =
              op.getInputs().slice(inputOffset, numRegionInputs);

          SmallVector<Value> blockInputs(block.getArguments().begin() + 2,
                                         block.getArguments().end());

          auto regionBody = [&] {
            DeviceGraphBuilderRef builderValue(builderArgMojo);
            for (auto [i, arg, operand] :
                 llvm::enumerate(blockInputs, regionInputs)) {
              if (!isDeviceGraphInput(operand)) {
                state.setMojoValueFor(arg, state.getMojoValueFor(operand));
                continue;
              }
              MojoValue input = state.getMojoValueFor(operand);
              // The tensor was unpacked once with the kind its type implies;
              // a region that reads only its shape rebinds it so `add_input`
              // records it in place instead of copying it.
              if (shapeOnlyInputs.test(inputOffset + i)) {
                MojoValue kind =
                    kStructDeviceGraphInputKind.get(kInputKindShapeOnly);
                input = state.file->var(
                    input.get(kFnWithKind)[{kind}](/*operands=*/{}));
              }
              MojoValue stable = state.file->var(builderValue.addInput(input));
              state.setMojoValueFor(arg, stable);
            }

            for (Operation &opInBody : block.getOperations())
              if (state.lowerOpToMojo(&opInBody).failed()) {
                errOccurred = true;
                return;
              }
          };

          if (idx == 0)
            branch = &state.file->if_(n.eq(idx), regionBody,
                                      /*comptime=*/true);
          else
            branch = &branch->elif_(n.eq(idx), regionBody);

          inputOffset += numRegionInputs;
        }

        branch->else_([&] {
          state.file->raise(
              "Error(\"unexpected device index in collective device graph "
              "build\")");
        });
      });

  if (errOccurred)
    return failure();
  return buildFor;
}

/// Emits the per-device cache-key closure from that region's inputs only;
/// `DeviceGraph.create_collective` adds the closure identity separately.
static MOGG::MojoValue
emitCollectiveKeyForClosure(MOGG::EmitMojoState &state,
                            DeviceGraphCollectiveCreateOp op) {
  using namespace MOGG;

  NamedMojoDef nParam("n", "Int");

  return state.file->lambda(
      "collective_key_for", /*parameters=*/{nParam},
      /*arguments=*/{},
      /*returnType=*/kStructString, MojoBuilder::LambdaInline::kNever,
      /*isAsync=*/false, /*raises=*/true, /*captureConvention=*/"imm",
      [&](const FunctionBody &keyBody) {
        MojoValue n = keyBody.parameters.front().resolvedName;

        size_t inputOffset = 0;
        MojoBuilder *branch = nullptr;
        for (auto [idx, region] : llvm::enumerate(op.getGraphs())) {
          Block &block = region.front();
          size_t numRegionInputs = block.getNumArguments() - 2;
          auto regionInputs =
              op.getInputs().slice(inputOffset, numRegionInputs);

          auto keyRegionBody = [&] {
            SmallVector<MojoValue> keyInputs;
            for (Value input : regionInputs)
              if (isDeviceGraphInput(input))
                keyInputs.push_back(state.getMojoValueFor(input));

            MojoValue key = state.file->var(
                kStructDeviceGraphCache.get("make_key")(keyInputs));
            state.file->returnsValue(key.transfer());
          };

          if (idx == 0)
            branch = &state.file->if_(n.eq(idx), keyRegionBody,
                                      /*comptime=*/true);
          else
            branch = &branch->elif_(n.eq(idx), keyRegionBody);

          inputOffset += numRegionInputs;
        }

        branch->else_([&] {
          state.file->raise(
              "Error(\"unexpected device index in collective device graph "
              "key\")");
        });
      });
}

LogicalResult
DeviceGraphCollectiveCreateOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  FailureOr<MojoValue> buildFor = emitCollectiveBuildForClosure(state, *this);
  if (failed(buildFor))
    return failure();

  unsigned numGraphs = getGraphs().size();

  // `create_collective` takes `Array[DeviceContext[], N]`, so the element type
  // binds the default target explicitly: a bare `DeviceContext` is generic over
  // its target, which is not a concrete `Array` element type.
  // TODO(DRIV-655): Use a generic element type once `create_collective` takes
  // generic targets.
  MojoValue contextsArrayType = makeMojoArrayType(
      llvm::formatv("{}[]", kStructDeviceContext).str(), numGraphs);
  MojoValue graphsArrayType = makeMojoArrayType(kStructDeviceGraph, numGraphs);

  MojoValue graphsArray;
  if (Value cache = getCache()) {
    MojoValue keyFor = emitCollectiveKeyForClosure(state, *this);

    // Each context arrives generic over its own target, so narrow it to the
    // default target the array's element type names.
    // TODO(DRIV-655): Drop this once `create_collective` takes generic targets.
    SmallVector<MojoValue> deviceContextValues;
    for (const MojoValue &ctx : state.getMojoValuesFor(getDeviceContexts()))
      deviceContextValues.push_back(
          state.file->var(ctx.get(kFnAssertDefaultTarget)({})));
    MojoValue contextsArray = state.file->varDecl(contextsArrayType);
    state.file->assign(contextsArray,
                       state.file->listLiteral(deviceContextValues));

    SmallVector<MojoValue> collectiveArgs = {
        contextsArray, *buildFor, keyFor,
        DeviceGraphRef::cacheArg(state.getMojoValueFor(cache))};

    // Each stable address slot is allocated on the device its type names;
    // the Mojo side takes that as an ordinal into the contexts array.
    if (StableAddressArrayAttr stableTypes = getStableTypesAttr();
        stableTypes && !stableTypes.getValue().empty()) {
      SmallVector<MojoValue> stableDevices;
      for (auto [slot, type] : llvm::enumerate(stableTypes.getValue())) {
        DeviceRefAttr device = type.getElementType().getDevice();
        auto contexts = getDeviceContexts();
        auto it = llvm::find_if(contexts, [&](Value ctx) {
          return cast<DeviceContextType>(ctx.getType()).getDevice() == device;
        });
        if (!device || it == contexts.end())
          return emitOpError("stable slot #")
                 << slot << " lives on device " << device
                 << ", which is not one of the collective's device contexts";
        stableDevices.push_back(static_cast<uint64_t>(it - contexts.begin()));
      }
      collectiveArgs.push_back(DeviceGraphRef::stableDevicesArg(
          state.file->listLiteral(stableDevices)));
    }

    graphsArray =
        state.file->var(DeviceGraphRef::createCollective(collectiveArgs));
  } else {
    if (StableAddressArrayAttr stableTypes = getStableTypesAttr();
        stableTypes && !stableTypes.getValue().empty())
      return emitOpError("stable address slots require a device graph cache "
                         "to own them; the op has no `cache` operand");

    // Uncached path: build each graph individually so the op does not require
    // a shared cache.
    NamedMojoDef builderParam =
        NamedMojoDef::asVerbatim("builder", kStructDeviceGraphBuilder);
    builderParam.argConvention = "mut";

    graphsArray = state.file->varDecl(graphsArrayType);
    for (auto [idx, ctx] : llvm::enumerate(getDeviceContexts())) {
      MojoValue buildDevice = state.file->lambda(
          "collective_build_device", /*parameters=*/{},
          /*arguments=*/{builderParam},
          /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kNever,
          /*isAsync=*/false, /*raises=*/true,
          /*captureConvention=*/"var", [&](const FunctionBody &bd) {
            state.file->call(*buildFor, /*parameters=*/{idx},
                             /*operands=*/{bd.arguments.front().resolvedName});
          });

      MojoValue graph = state.file->var(
          DeviceGraphRef::create({state.getMojoValueFor(ctx), buildDevice}));
      state.file->assign(graphsArray[idx], graph);
    }
  }

  // Named per-device variables keep an origin so consumers can transfer them
  // with `^`.
  for (unsigned i = 0; i < numGraphs; ++i) {
    MojoValue graphVar = state.file->var(graphsArray[i]);
    state.setMojoValueFor(getOutGraphs()[i], graphVar);
  }

  return success();
}
