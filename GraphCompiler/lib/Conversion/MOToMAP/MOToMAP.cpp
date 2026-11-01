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
// Lowers MO ops into `map` kernels. Each lowerable MO op with a registered
// Mojo kernel becomes either a `map.iter.opaque` wrapping a `map.call.opaque`
// to that kernel (with a trivial store epilogue per result), or, for
// elementwise kernels, a `map.iter.elementwise` expressing the loop body
// explicitly as scalar `map.load` / `map.call.elementwise` / `map.store` ops.
// A reduction monoid becomes a `map.iter.rowwise`.
// Contexts, variadics and scalars are materialized after fusion, not here.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Analysis/Support.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MO {

#define GEN_PASS_DEF_MOTOMAP
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h.inc"

namespace {

// The runtime tensor representation caps rank at `kMaxRank`
// (Support/ML/TensorShape.h); `MGP::TensorSpecType::get` asserts above it.
static bool hasLowerableRank(Operation *op) {
  auto rankOk = [](Type t) {
    MO::TensorType tt = MO::dynCastFromTensorLike(t);
    return !tt || !tt.hasRank() ||
           tt.getRank() <= static_cast<int64_t>(kMaxRank);
  };
  return llvm::all_of(op->getOperandTypes(), rankOk) &&
         llvm::all_of(op->getResultTypes(), rankOk);
}

bool shouldLower(Operation *op) {
  if (isa<MO::PreservedDuringKernelLowering>(op))
    return false;
  // No MAP form: returns by value. Lowered by MAPToMOGG.
  if (isa<MO::InvokeShapeFuncOp>(op))
    return false;
  Dialect *dialect = op->getDialect();
  if (!dialect || !isa<MO::MODialect>(dialect))
    return false;
  return hasLowerableRank(op);
}

MAP::IterEpilogueOp buildStoreEpilogue(OpBuilder &builder, Location loc,
                                       BlockArgument out) {
  auto outType = cast<MO::TensorType>(out.getType());
  auto indexType = MAP::IndexType::get(builder.getContext(), outType.getRank());

  // Pass a null `$intermediateTensor` explicitly; omitting it picks the
  // generic builder, which does not set `operandSegmentSizes`.
  // `AllocateAccumulatingBuffers` fills it in after fusion.
  auto epilogue =
      MAP::IterEpilogueOp::create(builder, loc, out.getType(), ValueRange{out},
                                  /*intermediateTensor=*/Value());

  OpBuilder::InsertionGuard guard(builder);
  Block *body = builder.createBlock(&epilogue.getBody());
  BlockArgument val = body->addArgument(outType.getElementType(), loc);
  BlockArgument idx = body->addArgument(indexType, loc);
  BlockArgument cap = body->addArgument(out.getType(), loc);
  builder.setInsertionPointToStart(body);
  MAP::StoreOp::create(builder, loc, /*tensor=*/cap, /*index=*/idx,
                       /*value=*/val);
  return epilogue;
}

MAP::IterPrologueOp buildLoadPrologue(OpBuilder &builder, Location loc,
                                      Value in) {
  auto inType = cast<MO::TensorType>(in.getType());
  auto indexType = MAP::IndexType::get(builder.getContext(), inType.getRank());

  auto prologue =
      MAP::IterPrologueOp::create(builder, loc, in.getType(), ValueRange{in});

  OpBuilder::InsertionGuard guard(builder);
  Block *body = builder.createBlock(&prologue.getBody());
  BlockArgument idx = body->addArgument(indexType, loc);
  BlockArgument cap = body->addArgument(in.getType(), loc);
  builder.setInsertionPointToStart(body);
  Value loaded = MAP::LoadOp::create(builder, loc, inType.getElementType(),
                                     /*tensor=*/cap, /*index=*/idx);
  MAP::YieldOp::create(builder, loc, loaded);
  return prologue;
}

// Effects for the replacing kernel to carry, instead of the conservative
// "unknown" default.
std::pair<bool, bool> getReadWriteEffects(Operation *op) {
  auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectOp)
    return {false, false};
  return {memEffectOp.hasEffect<MemoryEffects::Read>(),
          memEffectOp.hasEffect<MemoryEffects::Write>()};
}

template <typename OpT>
void stampAsyncLaunch(OpT iterOp, Operation *srcOp) {
  iterOp.setDispatch(MO::kernelRequiresSyncLaunch(srcOp)
                         ? MOGG::DispatchKind::Immediate
                         : MOGG::DispatchKind::Deferred);
}

FailureOr<Operation *>
lowerToOpaque(Operation *op, const MOGG::KernelDeclAdaptor &adaptor,
              const MOGG::KernelDeclOperandMapping &mapping) {
  llvm::DenseMap<uint64_t, bool> inputFusible;
  for (const auto &[idx, arg] : mapping.operandMapping)
    inputFusible[idx] = arg.isFusedTensorType();

  llvm::DenseMap<uint64_t, bool> outputFusible;
  for (const auto &[idx, arg] : mapping.resultMapping)
    outputFusible[idx] = arg.isFusedTensorType();

  OpBuilder builder(op);
  Location loc = op->getLoc();

  // A mutating op's chain is its last operand and last result.
  // `map.iter.opaque` threads it through `$inputChain` and a trailing result,
  // and keeps it out of the body: no kernel's `execute` reads it.
  ValueRange operands = op->getOperands();
  bool hasInputChain =
      !operands.empty() && isa<MO::ChainType>(operands.back().getType());
  Value inputChain = hasInputChain ? operands.back() : Value();
  ValueRange nonChainOperands = hasInputChain ? operands.drop_back() : operands;

  TypeRange resultTypes = op->getResultTypes();

  auto opaque = MAP::IterOpaqueOp::create(builder, loc, resultTypes,
                                          nonChainOperands, inputChain,
                                          /*fusionInfo=*/Attribute());
  // A collective executes on several devices, on which getDeviceOrDefaultAny
  // asserts. Take the first; the label only picks host or GPU codegen.
  // TODO(GEX-2278): carry all of them.
  DeviceRefAttr device;
  if (auto hasDevice = dyn_cast<MO::MOHasDeviceInterface>(op)) {
    SmallVector<DeviceRefAttr> devices = hasDevice.getExecutionDevices();
    if (!devices.empty())
      device = devices.front();
  }
  opaque.setDeviceRef(device ? device : MO::getDeviceOrDefaultAny(op));
  opaque.setFusionInfoAttr(
      MOGG::FusionInfoAttr::get(op->getContext(), MOGG::createFusionLeaf(op)));
  auto [hasReadEffect, hasWriteEffect] = getReadWriteEffects(op);
  opaque.setHasReadEffect(hasReadEffect);
  opaque.setHasWriteEffect(hasWriteEffect);
  stampAsyncLaunch(opaque, op);

  // Body block args shadow the non-chain inputs then the non-chain outputs.
  Block *body = builder.createBlock(&opaque.getBody());
  SmallVector<BlockArgument> ins;
  for (Value operand : nonChainOperands)
    ins.push_back(body->addArgument(operand.getType(), loc));
  SmallVector<BlockArgument> outs;
  Type opaqueResultType;
  for (Type resultType : resultTypes) {
    if (isa<MO::ChainType>(resultType))
      continue;
    // A `!mo.opaque` result is returned by value and carried out by
    // `map.yield`, so it has no block argument.
    if (isa<MO::OpaqueType>(resultType)) {
      opaqueResultType = resultType;
      continue;
    }
    outs.push_back(body->addArgument(resultType, loc));
  }

  builder.setInsertionPointToStart(body);
  SmallVector<Value> callInputs;
  for (auto [idx, operand] : llvm::enumerate(ins)) {
    if (!inputFusible.lookup(idx)) {
      callInputs.push_back(operand);
      continue;
    }
    if (!isa<MO::TensorType>(operand.getType())) {
      op->emitOpError(
          "MOToMAP: input fusion is only supported for tensor operands");
      return failure();
    }
    callInputs.push_back(buildLoadPrologue(builder, loc, operand).getResult());
  }

  SmallVector<Value> callOutputs;
  for (auto [idx, out] : llvm::enumerate(outs)) {
    if (!outputFusible.lookup(idx)) {
      callOutputs.push_back(out);
      continue;
    }
    if (!isa<MO::TensorType>(out.getType())) {
      op->emitOpError(
          "MOToMAP: output fusion is only supported for tensor results");
      return failure();
    }
    callOutputs.push_back(buildStoreEpilogue(builder, loc, out).getResult());
  }

  // Parameters named only by opaque types (e.g.
  // `MakeSimdPair.execute[P0, P1]() -> SIMDPair[P0, P1]`).
  DictionaryAttr params = inferParameters(op, builder);
  if (adaptor.executeFunction && adaptor.executeFunction->attr) {
    SmallVector<NamedAttribute> withOpaque(params.begin(), params.end());
    MOGG::inferOpaqueParameters(withOpaque, op, *adaptor.executeFunction,
                                adaptor.executeFunction->attr.getParams());
    params = builder.getDictionaryAttr(withOpaque);
  }

  builder.setInsertionPointToEnd(body);
  auto callOp = MAP::CallOpaqueOp::create(
      builder, loc,
      opaqueResultType ? TypeRange{opaqueResultType} : TypeRange{},
      FlatSymbolRefAttr::get(op->getContext(),
                             adaptor.fullyResolvedName.strref()),
      adaptor.registeredOpName, params,
      /*inputs=*/callInputs,
      /*outputs=*/callOutputs);
  // Needed later to split the flat operands back into equal-length variadic
  // groups (e.g. a bundled collective's N peers and N signal buffers).
  callOp.setHasSameVariadicOperandSize(
      isa<MO::SameVariadicOperandSizeInterface>(op));

  // Only a by-value opaque result needs a terminator.
  if (opaqueResultType)
    MAP::YieldOp::create(builder, loc, callOp.getOpaqueResult());

  op->replaceAllUsesWith(opaque.getResults());
  op->erase();
  return opaque.getOperation();
}

// Lowers an elementwise MO op to a `map.iter.elementwise` expressing its loop
// body explicitly: one `map.load` per input, a `map.call.elementwise` call
// to the kernel's scalar function, and a `map.store` into the output, all
// indexed by the op's `map.index<N>` block argument.
LogicalResult
lowerElementwiseToForeach(Operation *op,
                          const MOGG::KernelDeclAdaptor &adaptor) {
  if (op->getNumResults() != 1)
    return op->emitOpError(
        "MOToMAP: elementwise op must have exactly one result");

  auto resultTensorTy = dyn_cast<MO::TensorType>(op->getResult(0).getType());
  if (!resultTensorTy)
    return op->emitOpError("MOToMAP: elementwise op result must be !mo.tensor");

  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  MOSH::ShapeAttr shapeAttr = resultTensorTy.getShape();
  unsigned rank = static_cast<unsigned>(shapeAttr.getRank());
  Type elemTy = resultTensorTy.getElementType();

  auto foreachOp = MAP::IterElementwiseOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{resultTensorTy},
      /*shape=*/shapeAttr,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input tensors..., output tensor, map.index<rank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(resultTensorTy, loc);
  body->addArgument(MAP::IndexType::get(ctx, rank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument idxArg = body->getArguments().back();

  SmallVector<Value> scalarLoads;
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    BlockArgument inputArg = body->getArgument(i);
    auto inputTensorTy = cast<MO::TensorType>(operand.getType());
    Value loaded = MAP::LoadOp::create(
        bodyBuilder, loc, inputTensorTy.getElementType(), inputArg, idxArg);
    scalarLoads.push_back(loaded);
  }

  auto kernelSymbol =
      FlatSymbolRefAttr::get(ctx, adaptor.fullyResolvedName.strref());
  DictionaryAttr params = inferParameters(op, builder);
  if (params.empty())
    params = nullptr;
  Value scalar = MAP::CallElementwiseOp::create(
      bodyBuilder, loc, elemTy, kernelSymbol, params, idxArg, scalarLoads);

  BlockArgument outputArg =
      body->getArgument(static_cast<unsigned>(op->getNumOperands()));
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, scalar);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.mutable.store to a map.iter.elementwise that `map.store`s directly
// into its buffer input and threads a chain in and out. Its elementwise
// function is a scalar identity (gather_scatter.mojo's MutableStore).
// Special-cased by identity, as there is no generic mechanism (GEX-2360).
LogicalResult
lowerMutableStoreToForeach(MO::MutableStoreOp op,
                           const MOGG::KernelDeclAdaptor &adaptor) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  MO::TensorType bufferAsTensorTy = op.getInBuffer().getType().toTensorType();
  MOSH::ShapeAttr shapeAttr = bufferAsTensorTy.getShape();
  unsigned rank = static_cast<unsigned>(shapeAttr.getRank());
  Type elemTy = bufferAsTensorTy.getElementType();

  auto [hasReadEffect, hasWriteEffect] = getReadWriteEffects(op);

  auto foreachOp = MAP::IterElementwiseOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{MO::ChainType::get(ctx)},
      /*shape=*/shapeAttr,
      /*inputs=*/ValueRange{op.getInTensor(), op.getInBuffer()},
      /*chain=*/op.getInChain(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)),
      /*hasReadEffect=*/hasReadEffect,
      /*hasWriteEffect=*/hasWriteEffect);
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [tensor value, buffer, map.index<rank>]; the buffer is
  // the store target.
  Block *body = &foreachOp.getBody().emplaceBlock();
  BlockArgument valueArg = body->addArgument(op.getInTensor().getType(), loc);
  BlockArgument bufferArg = body->addArgument(op.getInBuffer().getType(), loc);
  BlockArgument idxArg = body->addArgument(MAP::IndexType::get(ctx, rank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  Value loaded =
      MAP::LoadOp::create(bodyBuilder, loc, elemTy, valueArg, idxArg);

  auto kernelSymbol =
      FlatSymbolRefAttr::get(ctx, adaptor.fullyResolvedName.strref());
  DictionaryAttr params = inferParameters(op, builder);
  if (params.empty())
    params = nullptr;
  Value scalar =
      MAP::CallElementwiseOp::create(bodyBuilder, loc, elemTy, kernelSymbol,
                                     params, idxArg, ValueRange{loaded});

  MAP::StoreOp::create(bodyBuilder, loc, bufferArg, idxArg, scalar);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.slice to a map.iter.elementwise over the *output* shape (the "pull"
// model): the body computes the input index via map.index.slice, loads, and
// stores unchanged.
LogicalResult lowerSliceToForeach(MO::SliceOp op) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  auto inputTy = cast<MO::TensorType>(op.getInput().getType());
  auto outputTy = cast<MO::TensorType>(op.getResult().getType());
  MOSH::ShapeAttr inShape = inputTy.getShape();
  MOSH::ShapeAttr outShape = outputTy.getShape();
  if (inShape.getRank() != outShape.getRank())
    return op->emitOpError("MOToMAP: slice must preserve rank");
  int64_t rank = outShape.getRank();

  // MO tracks constness per tensor, not per dimension.
  auto boundsOrDynamic = [&](Value v) {
    if (auto constVec = MO::tryGetConstantIntVector(cast<MO::TensorValue>(v)))
      return mlir::DenseI64ArrayAttr::get(ctx, *constVec);
    return mlir::DenseI64ArrayAttr::get(
        ctx, SmallVector<int64_t>(rank, ShapedType::kDynamic));
  };

  SmallVector<int64_t> inDims = inShape.getIntDims();
  SmallVector<int64_t> starts(boundsOrDynamic(op.getStart()).asArrayRef());
  SmallVector<int64_t> stops(boundsOrDynamic(op.getStop()).asArrayRef());
  SmallVector<int64_t> steps(boundsOrDynamic(op.getStep()).asArrayRef());

  // Normalize Python-style negative and out-of-range bounds, matching
  // materializeSliceShape's clamp.
  for (int64_t i = 0; i < rank; ++i) {
    if (ShapedType::isDynamic(inDims[i]) || ShapedType::isDynamic(steps[i]) ||
        steps[i] == 0)
      continue;
    int64_t dim = inDims[i];
    // A reverse slice walks down to -1, so it clamps one below the forward
    // range.
    int64_t lo = steps[i] > 0 ? 0 : -1;
    int64_t hi = steps[i] > 0 ? dim : dim - 1;
    auto normalize = [&](int64_t v) {
      if (ShapedType::isDynamic(v))
        return v;
      return std::clamp<int64_t>(v < 0 ? v + dim : v, lo, hi);
    };
    starts[i] = normalize(starts[i]);
    stops[i] = normalize(stops[i]);
  }

  // Verify first: `get` asserts on a rejected spec.
  auto emitSliceError = [&] { return op->emitOpError("MOToMAP: "); };
  if (failed(MAP::SliceSpecAttr::verify(
          emitSliceError, MAP::ViewSignatureAttr::get(ctx, inShape, outShape),
          mlir::DenseI64ArrayAttr::get(ctx, starts),
          mlir::DenseI64ArrayAttr::get(ctx, stops),
          mlir::DenseI64ArrayAttr::get(ctx, steps))))
    return failure();

  auto spec =
      MAP::SliceSpecAttr::get(ctx, inShape, outShape, starts, stops, steps);

  auto foreachOp = MAP::IterElementwiseOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{outputTy},
      /*shape=*/outShape,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input, start, stop, step, output, map.index<rank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(outputTy, loc);
  BlockArgument idxArg = body->addArgument(MAP::IndexType::get(ctx, rank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument inputArg = body->getArgument(0);
  BlockArgument startsArg = body->getArgument(1);
  BlockArgument stopsArg = body->getArgument(2);
  BlockArgument stepsArg = body->getArgument(3);
  BlockArgument outputArg = body->getArgument(4);

  Value innerIdx = MAP::SliceIndexOp::create(
      bodyBuilder, loc, MAP::IndexType::get(ctx, rank), spec, idxArg, startsArg,
      stopsArg, stepsArg);
  Value loaded = MAP::LoadOp::create(
      bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.static.broadcast_to like lowerSliceToForeach, via
// map.index.broadcast. Broadcast can add rank, so the load index rank may
// differ from the loop's.
LogicalResult lowerBroadcastToForeach(Static::BroadcastToOp op) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  auto inputTy = cast<MO::TensorType>(op.getInput().getType());
  auto outputTy = cast<MO::TensorType>(op.getResult().getType());
  MOSH::ShapeAttr inShape = inputTy.getShape();
  MOSH::ShapeAttr outShape = outputTy.getShape();
  if (inShape.getRank() > outShape.getRank())
    return op->emitOpError("MOToMAP: broadcast cannot reduce rank");
  int64_t inRank = inShape.getRank();
  int64_t outRank = outShape.getRank();

  auto spec = MAP::BroadcastSpecAttr::get(
      ctx, MAP::ViewSignatureAttr::get(ctx, inShape, outShape));

  auto foreachOp = MAP::IterElementwiseOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{outputTy},
      /*shape=*/outShape,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input, output, map.index<outRank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(outputTy, loc);
  BlockArgument idxArg =
      body->addArgument(MAP::IndexType::get(ctx, outRank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument inputArg = body->getArgument(0);
  BlockArgument outputArg = body->getArgument(1);

  Value innerIdx = MAP::BroadcastIndexOp::create(
      bodyBuilder, loc, MAP::IndexType::get(ctx, inRank), spec, idxArg);
  Value loaded = MAP::LoadOp::create(
      bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.static.reshape like lowerSliceToForeach, via map.index.reshape.
// Rank may change in either direction.
LogicalResult lowerReshapeToForeach(MO::Static::ReshapeOp op) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  auto inputTy = cast<MO::TensorType>(op.getInput().getType());
  auto outputTy = cast<MO::TensorType>(op.getResult().getType());
  MOSH::ShapeAttr inShape = inputTy.getShape();
  MOSH::ShapeAttr outShape = outputTy.getShape();
  int64_t inRank = inShape.getRank();
  int64_t outRank = outShape.getRank();

  auto spec = MAP::ReshapeSpecAttr::get(
      ctx, MAP::ViewSignatureAttr::get(ctx, inShape, outShape));

  auto foreachOp = MAP::IterElementwiseOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{outputTy},
      /*shape=*/outShape,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input, output, map.index<outRank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(outputTy, loc);
  BlockArgument idxArg =
      body->addArgument(MAP::IndexType::get(ctx, outRank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument inputArg = body->getArgument(0);
  BlockArgument outputArg = body->getArgument(1);

  Value innerIdx = MAP::ReshapeIndexOp::create(
      bodyBuilder, loc, MAP::IndexType::get(ctx, inRank), spec, idxArg);
  Value loaded = MAP::LoadOp::create(
      bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers mo.transpose like lowerSliceToForeach, via map.index.transpose. The
// perm is a runtime operand, also baked into `spec` when constant; otherwise
// (the eager interpreter's one-graph-per-op transpose) `spec` is all-dynamic.
LogicalResult lowerTransposeToForeach(MO::TransposeOp op) {
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  auto inputTy = cast<MO::TensorType>(op.getInput().getType());
  auto outputTy = cast<MO::TensorType>(op.getResult().getType());
  MOSH::ShapeAttr inShape = inputTy.getShape();
  MOSH::ShapeAttr outShape = outputTy.getShape();
  if (inShape.getRank() != outShape.getRank())
    return op->emitOpError("MOToMAP: transpose must preserve rank");
  int64_t rank = outShape.getRank();

  auto permAttr = mlir::DenseI64ArrayAttr::get(
      ctx, SmallVector<int64_t>(rank, ShapedType::kDynamic));
  if (auto permVec =
          MO::tryGetConstantIntVector(cast<MO::TensorValue>(op.getPerm())))
    permAttr = mlir::DenseI64ArrayAttr::get(ctx, *permVec);

  auto spec = MAP::TransposeSpecAttr::get(
      ctx, MAP::ViewSignatureAttr::get(ctx, inShape, outShape), permAttr);

  auto foreachOp = MAP::IterElementwiseOp::create(
      builder, loc,
      /*resultTypes=*/SmallVector<Type>{outputTy},
      /*shape=*/outShape,
      /*inputs=*/op->getOperands(),
      /*chain=*/Value(),
      /*fusionInfo=*/
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(foreachOp, op);

  // Body block args: [input, perm, output, map.index<rank>].
  Block *body = &foreachOp.getBody().emplaceBlock();
  for (Value operand : op->getOperands())
    body->addArgument(operand.getType(), loc);
  body->addArgument(outputTy, loc);
  BlockArgument idxArg = body->addArgument(MAP::IndexType::get(ctx, rank), loc);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  BlockArgument inputArg = body->getArgument(0);
  BlockArgument permArg = body->getArgument(1);
  BlockArgument outputArg = body->getArgument(2);

  Value innerIdx = MAP::TransposeIndexOp::create(
      bodyBuilder, loc, MAP::IndexType::get(ctx, rank), spec, idxArg, permArg);
  Value loaded = MAP::LoadOp::create(
      bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);

  op->replaceAllUsesWith(foreachOp.getResults());
  op->erase();
  return success();
}

// Lowers a reduction monoid to a `map.iter.rowwise`: one fold over the input,
// then a per-row store of each output field. Expects a verified reduction.
LogicalResult lowerReductionToRowwise(Operation *op,
                                      const MOGG::KernelDeclAdaptor &adaptor) {
  ErrorOr<int64_t> axis =
      MO::getReductionAxis(op->getOperand(0), op->getResultTypes());
  if (axis.isError())
    return op->emitOpError("MOToMAP: reduction ") << axis.getError();

  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder builder(op);

  Value input = op->getOperand(0);
  auto inputTy = cast<MO::TensorType>(input.getType());
  MOSH::ShapeAttr shapeAttr = inputTy.getShape();
  auto indexTy = MAP::IndexType::get(ctx, shapeAttr.getRank());

  auto rowwiseOp = MAP::IterRowwiseOp::create(
      builder, loc, op->getResultTypes(), shapeAttr,
      builder.getI64IntegerAttr(*axis), ValueRange{input},
      /*chain=*/Value(),
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)));
  stampAsyncLaunch(rowwiseOp, op);

  // Body block args: [input, outputs..., row].
  Block *body = &rowwiseOp.getBody().emplaceBlock();
  BlockArgument inputArg = body->addArgument(inputTy, loc);
  SmallVector<BlockArgument> outputArgs;
  for (Type resultTy : op->getResultTypes())
    outputArgs.push_back(body->addArgument(resultTy, loc));
  BlockArgument rowArg = body->addArgument(indexTy, loc);

  SmallVector<Type> foldTypes;
  for (Type resultTy : op->getResultTypes())
    foldTypes.push_back(cast<MO::TensorType>(resultTy).getElementType());

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  auto reduceOp = MAP::IterReduceOp::create(
      bodyBuilder, loc, foldTypes,
      FlatSymbolRefAttr::get(ctx, adaptor.fullyResolvedName.strref()));
  {
    OpBuilder::InsertionGuard guard(bodyBuilder);
    Block *foldBody = &reduceOp.getBody().emplaceBlock();
    BlockArgument idxArg = foldBody->addArgument(indexTy, loc);
    bodyBuilder.setInsertionPointToEnd(foldBody);
    Value loaded = MAP::LoadOp::create(
        bodyBuilder, loc, inputTy.getElementType(), inputArg, idxArg);
    MAP::YieldOp::create(bodyBuilder, loc, loaded);
  }

  for (auto [outputArg, field] :
       llvm::zip_equal(outputArgs, reduceOp.getResults()))
    MAP::StoreOp::create(bodyBuilder, loc, outputArg, rowArg, field);

  op->replaceAllUsesWith(rowwiseOp.getResults());
  op->erase();
  return success();
}

// See `lowerMOOpToMAP`.
FailureOr<Operation *> lowerOp(Operation *op, MOGG::MojoRegistry &registry) {
  ErrorOr<MOGG::KernelDeclAdaptor> adaptor =
      MOGG::getMojoKernelAdaptor(registry, op);
  if (adaptor.isError())
    return nullptr;

  if (auto mutableStoreOp = dyn_cast<MO::MutableStoreOp>(op)) {
    if (failed(lowerMutableStoreToForeach(mutableStoreOp, *adaptor)))
      return failure();
    return nullptr;
  }

  // Reductions have no executeFunction either.
  if (adaptor->isReduction()) {
    if (failed(lowerReductionToRowwise(op, *adaptor)))
      return failure();
    return nullptr;
  }

  // Elementwise kernels have no executeFunction.
  if (adaptor->isElementwise()) {
    if (failed(lowerElementwiseToForeach(op, *adaptor)))
      return failure();
    return nullptr;
  }

  if (!adaptor->executeFunction) {
    op->emitOpError("MOToMAP: kernel has no execute function");
    return failure();
  }

  ErrorOr<MOGG::KernelDeclOperandMapping> mapping =
      MOGG::mapMOOpToKernelImplementation(op, *adaptor->executeFunction);
  if (mapping.isError()) {
    op->emitOpError("MOToMAP: failed to map operands to kernel");
    return failure();
  }

  LogicalResult foreachResult = success();
  if (isa<MO::InvokeShapeFuncOp>(op)) {
    op->emitOpError("MOToMAP: shape-function kernels are not yet supported");
    return failure();
  } else if (auto sliceOp = dyn_cast<MO::SliceOp>(op)) {
    foreachResult = lowerSliceToForeach(sliceOp);
  } else if (auto broadcastOp = dyn_cast<MO::Static::BroadcastToOp>(op)) {
    foreachResult = lowerBroadcastToForeach(broadcastOp);
  } else if (auto reshapeOp = dyn_cast<MO::Static::ReshapeOp>(op)) {
    foreachResult = lowerReshapeToForeach(reshapeOp);
  } else if (auto transposeOp = dyn_cast<MO::TransposeOp>(op)) {
    foreachResult = lowerTransposeToForeach(transposeOp);
  } else {
    return lowerToOpaque(op, *adaptor, *mapping);
  }

  if (failed(foreachResult))
    return failure();
  return nullptr;
}

struct MOToMAPPass : public M::MO::impl::MOToMAPBase<MOToMAPPass> {
  using Base::Base;

  void runOnOperation() override {
    MOGG::MojoRegistry registry(getOperation());

    SmallVector<Operation *> worklist;
    WalkResult walkResult = getOperation()->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (dialect && isa<MO::MODialect>(dialect) && !hasLowerableRank(op)) {
        op->emitError() << "'" << op->getName().getStringRef()
                        << "' op has a tensor whose rank exceeds the maximum "
                           "supported rank of "
                        << kMaxRank << "; the graph cannot be lowered";
        return WalkResult::interrupt();
      }
      if (shouldLower(op))
        worklist.push_back(op);
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    for (Operation *op : worklist)
      if (failed(lowerOp(op, registry)))
        return signalPassFailure();
  }
};

} // namespace

FailureOr<Operation *> lowerMOOpToMAP(Operation *op,
                                      MOGG::MojoRegistry &registry) {
  return lowerOp(op, registry);
}

} // namespace M::MO
