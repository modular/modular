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
// This file implements the MOGG dialect operations' MOGG-to-MGP lowering:
// the four `LowerableToMGPInterface::lower` bodies.
//
// They live in this library, not next to the MOGGToMGP pass, because the
// ODS-generated interface model in MOGG.cpp.inc references them.
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
// MOGG ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);

  SmallVector<Value> loweredOperands = opLoweringState.getLoweredOperands();

  FailureOr<SmallVector<MGP::MOOutputStorage>> loweredOutputsStorageOr =
      opLoweringState.createOutputStorage(loweringState.getExecuteChain());
  if (failed(loweredOutputsStorageOr))
    return failure();
  auto &loweredOutputsStorage = *loweredOutputsStorageOr;
  assert(loweredOutputsStorageOr->size() == 1);

  Value executeChain =
      opLoweringState.getGuardChain(loweringState.getCurrentExecuteChain());

  Value outChain = opLoweringState.bufferConcat(
      executeChain, loweredOutputsStorage[0].buffer, loweredOperands,
      getResult().getType());

  // Create the output tensors, but only once the op has completed.
  opLoweringState.createOutputs(outChain, loweredOutputsStorage);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG SplitOp
//===----------------------------------------------------------------------===//

LogicalResult SplitOp::lower(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState opLoweringState(loweringState, *this);

  Value inputBuffer = opLoweringState.getBufferizedInput(getInput()).buffer;

  // TODO Enable this.
  if (inputBuffer.getDefiningOp<MGP::BufferConstantOp>())
    return emitOpError("constant input buffer is unsupported");

  // A split needs output specs only, no buffers.
  FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>> outputSpecsOr =
      opLoweringState.createOutputTensorSpecs();
  if (failed(outputSpecsOr))
    return failure();
  auto &outputSpecs = *outputSpecsOr;

  SmallVector<Value> outputBuffers;

  // The slices stay on the input buffer's device.
  auto sliceDeviceRef =
      cast<MGP::DeviceBufferType>(inputBuffer.getType()).getDevice();
  TypedValue<MGP::DeviceContextType> sliceDeviceContext =
      loweringState.getExecuteDevice(sliceDeviceRef);

  Value offset = opLoweringState.create<mlir::index::ConstantOp>(0).getResult();
  Value sliceSize;
  for (TypedValue<MGP::TensorSpecType> outSpec : outputSpecs) {
    if (sliceSize)
      offset = opLoweringState.create<mlir::index::AddOp>(offset, sliceSize);

    MGP::TensorSpecType outSpecTy = outSpec.getType();
    std::optional<uint64_t> bufferSize = outSpecTy.getSizeInBytes();

    if (!bufferSize.has_value()) {
      ASSERT_STREAM(outSpecTy.getOptDType().has_value(),
                    << "SplitOp output spec cannot have unknown dtype.");
      sliceSize =
          opLoweringState.bufferSizeBytes(outSpec, getInput().getType());

    } else {
      sliceSize = opLoweringState.create<mlir::index::ConstantOp>(*bufferSize)
                      .getResult();
    }

    Value outSlice = opLoweringState
                         .create<MGP::BufferSliceOp>(
                             inputBuffer, offset, sliceSize, sliceDeviceContext)
                         .getResult();
    outputBuffers.push_back(outSlice);
  }

  // The slices are only accessible once the inputs are.
  Value guardChain =
      opLoweringState.getGuardChain(loweringState.getCurrentExecuteChain());

  // Create the output tensors. Note there's no new chain to wait on.
  opLoweringState.createOutputTensors(guardChain, outputBuffers, outputSpecs);
  return success();
}

//===----------------------------------------------------------------------===//
// MOGG CompiledLoadOp
//===----------------------------------------------------------------------===//

LogicalResult CompiledLoadOp::lower(MGP::MOGraphLoweringState &loweringState) {
  OpBuilder &builder = loweringState.getInitOpBuilder();
  MGP::MGPBufferBuilder &bufferBuilder =
      loweringState.getBufferBuilder(MO::getDeviceOrDefaultAny(*this));
  Value outChain = bufferBuilder.moduleLoad(
      builder, getLoc(), loweringState.getInitMgpContext(),
      // DANGER: the init chain comes from device context creation, which module
      // load doesn't depend on (and which later becomes another module load).
      // Module loads must be ordered by creation, so depend only on runtime
      // creation.
      /*inChain=*/
      cast<TypedValue<MEF::ChainType>>(
          loweringState.getInitChain()
              .getDefiningOp()
              ->getOperand(0)
              .getDefiningOp<MGP::RuntimeCreateOp>()
              .getOutChain()),
      /*image=*/getValue(),
      /*functionSymbols=*/getSymbols(),
      /*constantSymbols=*/{});
  SmallVector<Value> values = {outChain, loweringState.getInitChain()};
  auto combined = GRT::ChainCreateOp::create(builder, getLoc(), values);
  loweringState.setInitChain(combined);
  return success();
}

// The layout `verify` checks: a handle per input and tensor-like output, then
// the dims, then the contexts.
RoutineOp::BodyArgs RoutineOp::getBodyArgs() {
  Block::BlockArgListType args = getBody().front().getArguments();
  size_t numInputs = getInputs().size();
  size_t numOutputs = llvm::count_if(getOutputs().getTypes(), [](Type type) {
    return static_cast<bool>(MO::dynCastFromTensorLike(type));
  });
  Block::BlockArgListType rest = args.drop_front(numInputs + numOutputs);
  size_t numDims =
      llvm::find_if_not(
          rest, [](BlockArgument arg) { return isa<DimType>(arg.getType()); }) -
      rest.begin();
  return {args.take_front(numInputs), args.slice(numInputs, numOutputs),
          rest.take_front(numDims), rest.drop_front(numDims)};
}

// Lowers a `mogg.routine` into a `mogg.routine.async` around its
// already-outlined stub: materialize the region operands (input buffers, new
// output buffers, dims, device context), unpack + cast each async block arg to
// the stub's handle, and re-home the `mogg.call`.
//
// TODO(GEX-3965): buffer allocation / context / dim materialization should move
// to dedicated pre-MGP passes, leaving this a genuinely thin lowering.
LogicalResult RoutineOp::lower(MGP::MOGraphLoweringState &loweringState) {
  return lowerToAsync(loweringState) ? success() : failure();
}

MOGG::AsyncRoutineOp
RoutineOp::lowerToAsync(MGP::MOGraphLoweringState &loweringState) {
  MGP::MOOpLoweringState st(loweringState, getOperation());
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  DeviceRefAttr device = MO::getDeviceOrDefaultAny(getOperation());

  // The body block args are, in order: one `!mogg.pointer` per input, then per
  // output, then the `!mogg.dim<name>`s, then one trailing `!mogg.context`. A
  // trailing output chain, like the (dedicated-operand) input chain, never
  // gets one of its own.
  Block &kernelBody = getBody().front();
  unsigned numInputs = getInputs().size();
  // Only tensor-like results are allocated and get an output pointer.
  unsigned numOutputs = 0;
  for (Value output : getOutputs())
    if (MO::dynCastFromTensorLike(output.getType()))
      ++numOutputs;
  unsigned numArgs = kernelBody.getNumArguments();
  if (numArgs < numInputs + numOutputs + 1) {
    emitOpError("MOGGToMGP: malformed mogg.routine body");
    return nullptr;
  }

  // The op's own `$chain` orders it against the kernels that wrote its
  // buffers, so the region waits on it; the ambient execute chain would drop
  // that edge.
  Value inChain = loweringState.getCurrentExecuteChain();
  if (Value opChain = getChain()) {
    Value lowered = loweringState.lookupExecuteLoweredValue(opChain);
    ASSERT_STREAM(lowered, << "no lowered value for mogg.routine input chain: "
                           << opChain);
    inChain = lowered;
  }
  inChain = st.getGuardChain(inChain);

  // Region operands (plain sync values) + their in/out signature. The chain
  // leads, so block arg i is operand i + 1.
  SmallVector<Value> operands{inChain};
  SmallVector<InOutSignatureAttr::InOutSemantics> semantics{
      InOutSignatureAttr::kIn};

  // Inputs lower to `!mgp.buffer`. The input chain has no body block argument
  // (it is threaded via `inChain`), so drop it to keep operands aligned with
  // the body. A `!mo.buffer` input is written in place, so it takes the
  // mutable signature (as an input MGP would treat it as read-only and lose
  // the write), read off the operand type since body args are type-erased
  // `!mogg.pointer`s.
  auto inputs = getInputs();
  unsigned inputIdx = 0;
  for (Value buffer : st.getLoweredOperands()) {
    if (isa<MEF::ChainType>(buffer.getType()))
      continue;
    operands.push_back(buffer);
    bool isMutable = inputIdx < inputs.size() &&
                     isa<MO::BufferType>(inputs[inputIdx].getType());
    semantics.push_back(isMutable ? InOutSignatureAttr::kMut
                                  : InOutSignatureAttr::kIn);
    ++inputIdx;
  }

  // Outputs: freshly-allocated buffers (externally-allocated, DPS).
  FailureOr<SmallVector<MGP::MOOutputStorage>> storageOr =
      st.createOutputStorage(inChain);
  if (failed(storageOr))
    return nullptr;
  SmallVector<MGP::MOOutputStorage> storage = *storageOr;
  // `createOutputStorage` skips chain-typed results, so it yields `numOutputs`
  // entries.
  assert(storage.size() == numOutputs &&
         "expected one output storage per non-chain mogg.routine result");
  for (const MGP::MOOutputStorage &out : storage) {
    operands.push_back(out.buffer);
    semantics.push_back(InOutSignatureAttr::kOut);
  }

  // Dims: materialize each `!mogg.dim<name>` block arg from its param, up to
  // the trailing context(s).
  unsigned argIdx = numInputs + numOutputs;
  for (; argIdx < numArgs &&
         isa<MOGG::DimType>(kernelBody.getArgument(argIdx).getType());
       ++argIdx) {
    auto dimType =
        cast<MOGG::DimType>(kernelBody.getArgument(argIdx).getType());
    if (!dimType.getName()) {
      emitOpError("MOGGToMGP: mogg.routine dim arg lacks a name");
      return nullptr;
    }
    ErrorOr<Value> dimValue = loweringState.lowerExpression(
        getOperation()->getParentRegion(), loc, dimType.getName());
    if (dimValue.isError()) {
      emitOpError("MOGGToMGP: ") << dimValue.getError();
      return nullptr;
    }
    operands.push_back(dimValue.takeValue());
    semantics.push_back(InOutSignatureAttr::kNone);
  }

  // Context(s): one `!mogg.context` block arg, or one per device for a
  // `DeviceContextArray` kernel in `getUniqueDevicesFromKernel` order (as
  // `IterOpaqueOp::lowerToMOGG` created them). Each is its own region operand,
  // though `ConformCallOperands` packed them into a tuple at the call.
  unsigned numContexts = numArgs - argIdx;
  if (numContexts <= 1) {
    operands.push_back(loweringState.getExecuteDevice(device));
    semantics.push_back(InOutSignatureAttr::kNone);
  } else {
    SmallVector<DeviceRefAttr> devices =
        MO::getUniqueDevicesFromKernel(getOperation());
    if (devices.size() != numContexts) {
      emitOpError("MOGGToMGP: mogg.routine has ")
          << numContexts << " context args but " << devices.size()
          << " unique devices";
      return nullptr;
    }
    for (DeviceRefAttr dev : devices) {
      operands.push_back(loweringState.getExecuteDevice(dev));
      semantics.push_back(InOutSignatureAttr::kNone);
    }
  }

  // Result 0 is the chain, then one dim per declared parameter, then a
  // by-value `!mo.opaque` if there is one. Allocated inside the kernel, so it
  // leaves as a region result rather than an output buffer.
  ArrayRef<KGEN::ParamDeclAttr> decls = getOutputParamDecls();
  Type dimType = MOSH::ShapeAttr::getDimType(ctx);
  SmallVector<Type> regionResultTypes{MEF::ChainType::get(ctx)};
  regionResultTypes.append(decls.size(), dimType);

  Value opaqueKernelResult;
  for (Value output : getOutputs())
    if (isa<MO::OpaqueType>(output.getType()))
      opaqueKernelResult = output;
  const unsigned opaqueResultIdx = regionResultTypes.size();
  if (opaqueKernelResult)
    regionResultTypes.push_back(opaqueKernelResult.getType());

  OpBuilder &builder = st.builder;
  auto region = MOGG::AsyncRoutineOp::create(
      builder, loc, regionResultTypes, operands,
      /*outputParamDecls=*/KGEN::ParamDeclArrayAttr::get(ctx, decls),
      InOutSignatureAttr::get(ctx, semantics),
      /*fusionInfo=*/getFusionInfoAttr(),
      /*dispatch=*/DispatchKindAttr::get(ctx, getDispatch()),
      /*constantResources=*/ArrayAttr::get(ctx, {}),
      /*canFuseIntoOther=*/BoolAttr::get(ctx, !MO::isOnHost(getOperation())),
      /*canBeFusedInto=*/BoolAttr::get(ctx, true),
      /*canDuplicate=*/BoolAttr::get(ctx, false),
      /*isStructured=*/BoolAttr::get(ctx, true));
  region.setDeviceRef(device);

  // Region body: one `!mogg.async<T>` block arg per operand, same order.
  Block *regionBody = &region.getBody().emplaceBlock();
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  for (Value operand : operands) {
    argTypes.push_back(MOGG::AsyncType::get(operand.getType()));
    argLocs.push_back(loc);
  }
  regionBody->addArguments(argTypes, argLocs);

  // Unpack + cast each async arg to the handle the stub expects, then re-home
  // the already-outlined `mogg.call`. The leading chain arg has no stub
  // counterpart, hence the `+ 1`.
  builder.setInsertionPointToStart(regionBody);
  SmallVector<Value> stubArgs;
  for (unsigned i = 0; i < numArgs; ++i) {
    Value unpacked = MOGG::AsyncUnpackOp::create(
        builder, loc, regionBody->getArgument(i + 1));
    // `mogg.rebind` reconciles the unpacked MGP handle with the stub's codegen
    // handle. For `!mogg.pointer` it emits real Mojo (`OwnedByteBuffer` -> raw
    // opaque data pointer); context/dim casts are pass-throughs.
    // debt: drop once `mogg.async.unpack` folds directly to the typed handle.
    // A dim arg is cast to an anonymous `!mogg.dim`: the symbolic name blocks
    // kernel dedup, and the stub's dim args are stripped to match.
    Type handleType = kernelBody.getArgument(i).getType();
    if (isa<MOGG::DimType>(handleType))
      handleType = MOGG::DimType::get(ctx, /*name=*/{});
    Value handle = MOGG::RebindOp::create(builder, loc, handleType, unpacked);
    stubArgs.push_back(handle);
  }

  // Re-home `kernelBody`'s contents (the `mogg.call` plus constants kept
  // at the call site) by cloning with block args mapped to `stubArgs`.
  // `mgp-anonymize-dim-names` strips the stub's dim names once all kernels are
  // lowered.
  IRMapping mapping;
  for (unsigned i = 0; i < numArgs; ++i)
    mapping.map(kernelBody.getArgument(i), stubArgs[i]);
  SmallVector<Attribute> blobs;
  Value shapeResult;
  Value opaqueResult;
  for (Operation &calleeOp : kernelBody) {
    // The MEF alignment is only known here, so swap a `mogg.buffer.constant`
    // for the real `mgp.buffer.constant` in this region: resources reach the
    // runtime through the MGP-ABI `void**`, which the MOGG-ABI stub lacks.
    if (auto cst = dyn_cast<MOGG::BufferConstantOp>(calleeOp)) {
      ErrorOr<ElementsAttr> blob =
          loweringState.lowerElementsAttr(cst.getValue());
      if (blob.isError()) {
        cst.emitOpError("MOGGToMGP: ") << blob.getError();
        return nullptr;
      }
      blobs.push_back(*blob);
      Value buffer =
          MGP::BufferConstantOp::create(builder, cst.getLoc(), *blob);
      mapping.map(cst.getPointer(),
                  MOGG::RebindOp::create(builder, cst.getLoc(),
                                         cst.getPointer().getType(), buffer));
      continue;
    }
    Operation *clone = builder.clone(calleeOp, mapping);
    // By type, so no unrelated result can stand in.
    for (Value result : clone->getResults()) {
      if (isa<MOGG::ShapeFuncResultType>(result.getType()))
        shapeResult = result;
      else if (isa<MO::OpaqueType>(result.getType()))
        opaqueResult = result;
    }
  }
  // Only the region carries the bytes out to the runtime; the stub reads them
  // as ordinary arguments, so its own array stays empty.
  if (!blobs.empty())
    region.setConstantResourcesAttr(builder.getArrayAttr(blobs));
  Value readyChain = MOGG::AsyncReadyOp::create(builder, loc);
  SmallVector<Value> regionOutputs{readyChain};
  if (!decls.empty()) {
    ASSERT_STREAM(shapeResult,
                  << "verifier guarantees a shape-returning call whenever "
                     "outputParamDecls is non-empty");
    // Unpack the `N` dims out of the `shape_func_result<N>`.
    for (size_t i = 0, e = decls.size(); i < e; ++i) {
      auto item = MOGG::ShapeResultGetItemOp::create(
          builder, loc, dimType, shapeResult,
          builder.getIntegerAttr(builder.getIntegerType(64, true), i));
      regionOutputs.push_back(
          MOGG::AsyncPackOp::create(builder, loc, item.getResult()));
    }
  }
  if (opaqueKernelResult) {
    if (!opaqueResult) {
      emitOpError("MOGGToMGP: mogg.routine returns an opaque but its body "
                  "produces none");
      return nullptr;
    }
    regionOutputs.push_back(
        MOGG::AsyncPackOp::create(builder, loc, opaqueResult));
  }
  MOGG::OutputOp::create(builder, loc, regionOutputs);

  // Register the outputs against the graph, guarded by the region's out chain.
  builder.setInsertionPointAfter(region);

  // Bind each dim to the value the region returns. Order-independent:
  // `lowerParamDecl` back-patches any placeholder already handed out.
  for (auto [idx, decl] : llvm::enumerate(decls)) {
    Value asDimension = MGP::indexToDimension(
        loweringState.getExecuteOpBuilder(), loc, region->getResult(idx + 1));
    if (auto err = loweringState.lowerParamDecl(
            getOperation()->getParentRegion(), decl, asDimension)) {
      emitOpError("MOGGToMGP: failed to bind parameter ")
          << decl.getName() << ": " << err.getError();
      return nullptr;
    }
  }

  // No buffer storage for a by-value opaque, so `createOutputs` skips it: hand
  // the region's result straight to whoever consumes the kernel's.
  if (opaqueKernelResult)
    loweringState.emplaceExecuteLoweredValue(
        opaqueKernelResult, region->getResult(opaqueResultIdx));

  st.createOutputs(region->getResult(0), storage);

  // The region's chain result is the kernel's out chain too. Consumers look it
  // up by the kernel's own result, so without this an `mo.output` forwarding
  // that chain finds nothing lowered.
  if (Value outChain = getOutputChain())
    loweringState.emplaceExecuteLoweredValue(outChain, region->getResult(0));

  return region;
}
