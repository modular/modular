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
// `LowerableToMOGGInterface` implementations for the `map` ops. The pass
// driver is `GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp`.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/Interfaces/MAPToMOGGState.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/AssertStream.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace M::MAP {

namespace {

// A dimension is dynamic unless it is a concrete `KGEN::SIMDAttr`.
bool isDynamicDim(Attribute dim) { return !isa<KGEN::SIMDAttr>(dim); }

// A walk collecting the dynamic dims of every nested `MOSH::ShapeAttr` into
// `state.dimArgs`, deduped.
mlir::AttrTypeWalker dynamicDimWalker(MAPToMOGGState &state) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([&state](MOSH::ShapeAttr shape) {
    for (Attribute dim : shape.getValues())
      if (isDynamicDim(dim))
        state.dimArgs.insert({dim, BlockArgument()});
  });
  return walker;
}

// The dynamic dims of `types`, deduped in first-seen order.
void collectDynamicDims(TypeRange types, MAPToMOGGState &state) {
  mlir::AttrTypeWalker walker = dynamicDimWalker(state);
  for (Type ty : types)
    walker.walk(ty);
}

// Every dynamic dim in the attributes and operand/result types of `root` and
// its nested ops.
void collectDynamicDims(Operation *root, MAPToMOGGState &state) {
  mlir::AttrTypeWalker walker = dynamicDimWalker(state);
  root->walk([&](Operation *op) {
    walker.walk(op->getAttrDictionary());
    for (Type ty : op->getOperandTypes())
      walker.walk(ty);
    for (Type ty : op->getResultTypes())
      walker.walk(ty);
  });
}

// Materializes a `!mogg.shape` via `mogg.shape.create`. Dynamic dims in the
// ATTR become a uniform `dyn` so shapes with different symbols (e.g. `[M, K]`
// vs `[N, K]`) dedup; the `!mogg.dim` OPERANDS keep the runtime values.
Value buildShape(Location loc, MOSH::ShapeAttr shapeAttr,
                 MAPToMOGGState &state) {
  MLIRContext *ctx = state.builder.getContext();
  TypedAttr canonicalDyn = KGEN::ParamDeclRefAttr::get(
      StringAttr::get(ctx, "dyn"), MOSH::ShapeAttr::getDimType(ctx));
  SmallVector<Value> dims;
  SmallVector<TypedAttr> canonicalDims;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (isDynamicDim(dim)) {
      dims.push_back(state.dimArgs.lookup(dim));
      canonicalDims.push_back(canonicalDyn);
    } else {
      canonicalDims.push_back(dim);
    }
  }
  auto canonicalShapeAttr =
      MOSH::ShapeAttr::get(canonicalDims, shapeAttr.getType());
  return MOGG::ShapeCreateOp::create(
      state.builder, loc, MOGG::ShapeType::get(ctx), dims, canonicalShapeAttr);
}

// Reconstructs a tensor or buffer (e.g. from `FuseMutableLoads`) of
// `tensorType` from the `!mogg.pointer` `ptr`.
Value buildTensor(Location loc, Value ptr, Type tensorType, MO::IOKind io,
                  bool isMut, MAPToMOGGState &state) {
  MO::TensorType tensorTy = MO::castFromTensorLike(tensorType);
  MLIRContext *ctx = state.builder.getContext();
  MOSH::ShapeAttr shapeAttr = tensorTy.getShape();
  Value shape = buildShape(loc, shapeAttr, state);
  // Buffer-ness follows the operand type, while `$mut` is store-side
  // mutability (true for buffers and output tensors alike).
  bool isBuffer = isa<MO::BufferType>(tensorType);
  Type resultType = isBuffer ? Type(MOGG::BufferType::get(ctx))
                             : Type(MOGG::TensorType::get(ctx));
  return MOGG::TensorCreateOp::create(
      state.builder, loc, resultType, ptr, shape, shapeAttr,
      state.builder.getStringAttr(state.device.getLabel()),
      TypeAttr::get(tensorTy.getElementType()),
      state.builder.getBoolAttr(isMut), MO::IOKindAttr::get(ctx, io),
      state.builder.getBoolAttr(tensorTy.isStable()));
}

} // namespace

// mo.invoke_shape_func -> mogg.routine wrapping a mogg.call.shape. Not
// destination-passing: the shape leaves by value, and the dims it defines ride
// in `outputParamDecls`.
LogicalResult lowerInvokeShapeFuncToMOGG(MO::InvokeShapeFuncOp shapeFuncOp,
                                         MAPToMOGGState &state) {
  Operation *op = shapeFuncOp.getOperation();
  MLIRContext *ctx = op->getContext();
  Location loc = op->getLoc();
  OpBuilder &builder = state.builder;
  builder.setInsertionPoint(op);

  MOGG::ShapeFunctionDeclOp shapeDecl =
      state.registry.lookupShapeFunction(shapeFuncOp.getShapeFuncName());
  if (!shapeDecl)
    return op->emitOpError("MAPToMOGG: '")
           << shapeFuncOp.getShapeFuncName()
           << "' requires a registered shape function; add "
              "'@extensibility.register_shape_function(\""
           << shapeFuncOp.getShapeFuncName() << "\")' to the kernel";

  ValueRange inputs = shapeFuncOp.getInputs();

  const uint64_t rank = shapeFuncOp.getOutputParamDecls().size();
  Type shapeResultType = MOGG::ShapeFuncResultType::get(ctx, rank);

  // Host-only.
  state.device = MO::getDeviceOrDefaultAny(op);

  auto kernel = MOGG::RoutineOp::create(
      builder, loc, /*resultTypes=*/TypeRange{shapeResultType}, inputs,
      /*chain=*/Value(),
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(op)),
      shapeFuncOp.getOutputParamDeclsAttr());
  kernel.setDeviceRef(state.device);

  collectDynamicDims(inputs.getTypes(), state);

  Block *body = builder.createBlock(&kernel.getBody());
  SmallVector<BlockArgument> inputPtrs;
  for (size_t i = 0, e = inputs.size(); i < e; ++i)
    inputPtrs.push_back(body->addArgument(MOGG::PointerType::get(ctx), loc));
  for (auto &[dim, blockArg] : state.dimArgs)
    blockArg = body->addArgument(
        MOGG::DimType::get(ctx, dyn_cast<KGEN::ParamDeclRefAttr>(dim)), loc);
  state.contextArgs.assign(
      {body->addArgument(MOGG::ContextType::get(ctx), loc)});

  builder.setInsertionPointToEnd(body);
  SmallVector<Value> callOperands;
  for (auto [i, in] : llvm::enumerate(inputs))
    callOperands.push_back(buildTensor(loc, inputPtrs[i], in.getType(),
                                       MO::IOKind::Input, /*isMut=*/false,
                                       state));

  // `kgenParams` are parameters of the shape function too, so forwarding the
  // rank alone leaves them uninferable.
  SmallVector<NamedAttribute> shapeParams{
      builder.getNamedAttr(MOGG::kMoggOutputRankParameterName,
                           builder.getIndexAttr(static_cast<int64_t>(rank)))};
  if (DictionaryAttr kgenParams = shapeFuncOp.getKgenParamsAttr())
    for (NamedAttribute param : kgenParams)
      shapeParams.push_back(builder.getNamedAttr(
          param.getName(), normalizeKernelParameter(param.getValue())));

  bool isThrows = shapeDecl.getFunction().getSignature().isThrows();
  auto call = MOGG::CallShapeOp::create(
      builder, loc, shapeResultType,
      FlatSymbolRefAttr::get(shapeDecl.getSymNameAttr()),
      isThrows ? builder.getUnitAttr() : nullptr, callOperands,
      builder.getDictionaryAttr(shapeParams));

  MOGG::OutputOp::create(builder, loc, call.getResult());

  op->erase();
  return success();
}

// map.constant -> mogg.buffer.constant + mogg.tensor.create. `MOGGToMGP` turns
// the buffer into an `mgp.buffer.constant` at the call site, where the minimum
// CPU alignment is known.
LogicalResult ConstantOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = state.builder.getContext();
  Value ptr = MOGG::BufferConstantOp::create(
      state.builder, getLoc(), MOGG::PointerType::get(ctx), getValueAttr());
  state.mapping.map(getResult(),
                    buildTensor(getLoc(), ptr, getType(), MO::IOKind::Input,
                                /*isMut=*/false, state));
  return success();
}

// map.constant.scalar -> mogg.constant.scalar.
LogicalResult ConstantScalarOp::lowerToMOGG(MAPToMOGGState &state) {
  auto constant = MOGG::ConstantScalarOp::create(
      state.builder, getLoc(), MOGG::SimdType::get(getResult().getType()),
      getValue());
  state.mapping.map(getResult(), constant.getResult());
  return success();
}

// map.load -> mogg.tensor.load.
LogicalResult LoadOp::lowerToMOGG(MAPToMOGGState &state) {
  Value result = MOGG::TensorLoadOp::create(
      state.builder, getLoc(), MOGG::SimdType::get(getResult().getType()),
      state.mapping.lookup(getTensor()), state.mapping.lookup(getIndex()));
  state.mapping.map(getResult(), result);
  return success();
}

// map.store -> mogg.tensor.store.
LogicalResult StoreOp::lowerToMOGG(MAPToMOGGState &state) {
  MOGG::TensorStoreOp::create(
      state.builder, getLoc(), state.mapping.lookup(getTensor()),
      state.mapping.lookup(getIndex()), state.mapping.lookup(getValue()));
  return success();
}

// map.index.slice.select -> mogg.index.slice.select, lowered in place inside
// the epilogue like a store-side `map.index.reshape`, and wired the same way to
// the runtime shapes the epilogue threaded in when either shape is dynamic.
// Starts stay as written, negative ones included. The body lowers into the new
// op's own, its index argument mapped across.
LogicalResult SliceSelectOp::lowerToMOGG(MAPToMOGGState &state) {
  SliceSpecAttr spec = getSpec();
  MAP::ViewSignatureAttr signature = spec.getSignature();
  ValueRange dynamicShapes;
  if (!signature.getIn().isStatic() || !signature.getOut().isStatic()) {
    auto it = state.indexShapeArgs.find(getOperation());
    if (it == state.indexShapeArgs.end())
      return emitOpError("MAPToMOGG: a dynamic slice select has no shape "
                         "captures; its kernel did not thread them in");
    dynamicShapes = ValueRange(it->second);
  }
  auto select = MOGG::IndexSliceSelectOp::create(
      state.builder, getLoc(), state.mapping.lookup(getIndex()),
      signature.getIn(), signature.getOut(), spec.getStarts(), spec.getSteps(),
      dynamicShapes);

  OpBuilder::InsertionGuard guard(state.builder);
  BlockArgument outIdx = getBody().getArgument(0);
  Block *body =
      state.builder.createBlock(&select.getBody(), select.getBody().end(),
                                {outIdx.getType()}, {outIdx.getLoc()});
  state.mapping.map(outIdx, body->getArgument(0));
  return state.lowerRegionBody(getBody().front());
}

// map.call.elementwise -> mogg.call.elementwise. `$moOpName` comes from the
// kernel decl, which `verifySymbolUses` guarantees has an `elementwise`
// function.
LogicalResult CallElementwiseOp::lowerToMOGG(MAPToMOGGState &state) {
  StringRef kernelSymName = getCallee();
  MOGG::KernelDeclOp decl = state.registry.lookupKernelBySymName(kernelSymName);
  MOGG::KernelDeclAdaptor adaptor(decl);
  SmallVector<Value> inputs;
  for (Value arg : getArgs())
    inputs.push_back(state.mapping.lookup(arg));
  // Only if the `elementwise` function declares a trailing index-list
  // parameter.
  Value index;
  const MOGG::KernelDeclFunctionAdaptor &fn = *adaptor.elementwiseFunction;
  if (!fn.inputArguments.empty() &&
      fn.inputArguments.back().isOpaqueWithTypeName(
          MOGG::kMojoInternalDpsIndexListTypeQualifiedName))
    index = state.mapping.lookup(getIndex());
  // `kernelSymName` is a plain name; the builder wraps it into the op's
  // `$callee` `FlatSymbolRefAttr`.
  auto moggCall = MOGG::CallElementwiseOp::create(
      state.builder, getLoc(), MOGG::SimdType::get(getResult().getType()),
      kernelSymName, adaptor.registeredOpName.strref(), inputs, index,
      getParamsAttr());
  state.mapping.map(getResult(), moggCall.getOutputs()[0]);
  return success();
}

// The kernel argument `mapResult` binds on its single `map.call.opaque` use;
// nullopt for any other use. `isOutput` picks the call's output vs input
// operand list.
static std::optional<MOGG::KernelDeclOperandAdaptor>
consumingKernelArg(Value mapResult, bool isOutput, MAPToMOGGState &state) {
  if (!mapResult.hasOneUse())
    return std::nullopt;
  auto callOp = dyn_cast<CallOpaqueOp>(*mapResult.getUsers().begin());
  if (!callOp)
    return std::nullopt;
  ValueRange operands = isOutput ? callOp.getOutputs() : callOp.getInputs();
  std::optional<unsigned> operandIdx;
  for (auto [i, v] : llvm::enumerate(operands))
    if (v == mapResult) {
      operandIdx = i;
      break;
    }
  if (!operandIdx)
    return std::nullopt;
  MOGG::KernelDeclAdaptor adaptor(
      state.registry.lookupKernelBySymName(callOp.getCallee()));
  if (!adaptor.executeFunction)
    return std::nullopt;
  // Not the operand's raw position: a variadic argument binds a run of
  // operands, which shifts every argument after it. Mutable-to-immutable
  // widening, as ConformCallOperands applies: an input fed by a fused
  // `mo.mutable.load` arrives as a buffer for a read-only argument, and only
  // positions are read here.
  ErrorOr<MOGG::KernelDeclOperandMapping> mapping =
      MOGG::mapToKernelImplementation(
          callOp.getInputs(), callOp.getOutputs(), *adaptor.executeFunction,
          callOp.getCallee(), callOp.getHasSameVariadicOperandSize(),
          /*hasExtraOutputShapeArgument=*/false,
          /*resultLeavesByValue=*/callOp.getOpaqueResult() != nullptr,
          /*allowMutToImmut=*/true);
  if (mapping.isError())
    return std::nullopt;
  const MOGG::KernelDeclIndexToOperandAdaptorMapping &argForOperand =
      isOutput ? mapping->resultMapping : mapping->operandMapping;
  auto arg = argForOperand.find(*operandIdx);
  if (arg == argForOperand.end())
    return std::nullopt;
  return arg->second;
}

// Which fusion-lambda kind the kernel consuming `epilogueOp` wants. Resolved
// at MAP level, where the epilogue -> `map.call.opaque` edge is still explicit.
static ErrorOr<MOGG::KernelOutputFusionLambdaType>
classifyEpilogueLambdaKind(IterEpilogueOp epilogueOp, MAPToMOGGState &state) {
  using Kind = MOGG::KernelOutputFusionLambdaType;

  // Anything but a single `map.call.opaque` output use gets a store lambda,
  // as does a `TensorTuple` element (a variadic argument).
  Value result = epilogueOp.getResult();
  std::optional<MOGG::KernelDeclOperandAdaptor> arg =
      consumingKernelArg(result, /*isOutput=*/true, state);
  if (!arg)
    return Kind::Output;
  const auto *tensorArg =
      std::get_if<MOGG::KernelDeclTensorAdaptor>(&arg->underlyingType);
  if (!tensorArg)
    return Kind::Output;

  // `EpilogueFuser` only builds the SIMD body shape.
  if (tensorArg->ioSpec == MOGG::IOSpec::FusedComputeOutputTileTensor ||
      tensorArg->ioSpec == MOGG::IOSpec::FusedOutputTileTensor)
    return Error("MAP-driven epilogue fusion into a tile fused-output kernel "
                 "argument is not yet supported");

  // A trait-lane `_LegacyComputeOutput` slot (contract access `transform`)
  // follows the same rules as a legacy `FusedComputeOutputTensor`. Its
  // `ioSpec` is the plain `OutputTensor` the contract maps to, so check the
  // contract.
  bool isComputeOutputSlot =
      tensorArg->traitContract &&
      tensorArg->traitContract->hasAccess(MOGG::TensorArgAccess::transform);
  if (tensorArg->ioSpec != MOGG::IOSpec::FusedComputeOutputTensor &&
      !isComputeOutputSlot)
    return Kind::Output;

  // A compute lambda returns one value the kernel stores at one destination,
  // so an epilogue writing several has to own its stores. The kernel operand
  // stays a compute output either way.
  if (getOutputStores(epilogueOp.getBody().front()).size() > 1)
    return Kind::OutputLegacyForCompute;

  // A store-index transform (a fused reshape) needs a store lambda: a compute
  // lambda's value is stored at the kernel's own index, dropping the
  // transform.
  if (auto storeOp = dyn_cast<StoreOp>(epilogueOp.getBody().front().back()))
    if (storeOp.getIndex() != epilogueOp.getBody().getArgument(1))
      return Kind::OutputLegacyForCompute;

  // The opt-out and a dtype-changing epilogue also force a store lambda:
  // `compute()` returns the dtype it was handed.
  if (!state.enableComputeLambdas)
    return Kind::OutputLegacyForCompute;
  Type inType = epilogueOp.getBody().getArgument(0).getType();
  Type outType = cast<MO::TensorType>(result.getType()).getElementType();
  if (inType != outType)
    return Kind::OutputLegacyForCompute;

  return Kind::Compute;
}

// The trait contract the consuming `map.call.opaque` declares for the fused
// operand `mapResult` feeds. Present only for a trait-ABI fused argument
// (`Fused & ... & Access`), whose functor emit wraps in a conforming trait
// tensor; null for the legacy `FusedInput/OutputTensor` (MTS-API) lane, which
// binds the functor onto a `ManagedTensorSlice`. Resolved at MAP level,
// like `classifyEpilogueLambdaKind`.
static MOGG::TensorArgContractAttr
consumingFusedContract(Value mapResult, bool isOutput, MAPToMOGGState &state) {
  std::optional<MOGG::KernelDeclOperandAdaptor> arg =
      consumingKernelArg(mapResult, isOutput, state);
  if (!arg)
    return {};
  std::optional<MOGG::TensorArgContract> contract = arg->getTraitContract();
  if (!contract)
    return {};
  return MOGG::tensorArgContractToAttr(state.builder.getContext(), *contract);
}

namespace {
// The runtime shapes a functor threads in as shape captures, for the dynamic
// index ops in its body.
struct IndexShapeCaptures {
  SmallVector<Value> shapes;
  // Each op with the run of `shapes` it reads.
  struct Use {
    Operation *op;
    unsigned start;
    unsigned count;
  };
  SmallVector<Use> uses;

  // Builds the shapes in kernel scope, where the dim args are live.
  static IndexShapeCaptures build(Region &body, Location loc,
                                  MAPToMOGGState &state) {
    IndexShapeCaptures captures;
    body.walk<WalkOrder::PreOrder>([&](DynamicShapeIndexOpInterface op) {
      SmallVector<MOSH::ShapeAttr> shapes = op.getDynamicShapes();
      if (shapes.empty())
        return;
      captures.uses.push_back({op.getOperation(),
                               static_cast<unsigned>(captures.shapes.size()),
                               static_cast<unsigned>(shapes.size())});
      for (MOSH::ShapeAttr shape : shapes)
        captures.shapes.push_back(buildShape(loc, shape, state));
    });
    return captures;
  }

  // Adds the shapes as trailing args of the lowered functor's `body` and
  // records each op's for its own lowering to take as `$dynamicShapes`.
  void bind(Block *body, Location loc, MAPToMOGGState &state) const {
    SmallVector<Value> args;
    for (Value shape : shapes)
      args.push_back(body->addArgument(shape.getType(), loc));
    for (const Use &use : uses)
      state.indexShapeArgs[use.op] =
          SmallVector<Value>(ArrayRef(args).slice(use.start, use.count));
  }

  // Records each op's shapes as built, for a body that sees the kernel scope:
  // a foreach's closure captures it whole, so it needs no shape captures.
  void bindInScope(MAPToMOGGState &state) const {
    for (const Use &use : uses)
      state.indexShapeArgs[use.op] =
          SmallVector<Value>(ArrayRef(shapes).slice(use.start, use.count));
  }
};
} // namespace

// map.iter.epilogue -> mogg.tensor.create.epilogue, mapped under the
// epilogue's result so `CallOpaqueOp` resolves it like a plain kernel output.
LogicalResult IterEpilogueOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  Block &epiBody = getBody().front();
  Block::BlockArgListType epiArgs = epiBody.getArguments();
  size_t numCaptures = getCaptures().size();
  if (epiArgs.size() < numCaptures)
    return emitOpError(
        "MAPToMOGG: epilogue has fewer block args than captures");
  // Leading block args are (value, index).
  size_t numParams = epiArgs.size() - numCaptures;

  // A capture is a kernel output or a hoisted view, which
  // `LoopInvariantViewMotion` places before the closure; both lower first.
  SmallVector<Value> moggCaptures;
  for (Value cap : getCaptures()) {
    if (!state.mapping.contains(cap))
      return emitOpError("MAPToMOGG: epilogue capture is not yet lowered");
    moggCaptures.push_back(state.mapping.lookup(cap));
  }

  // A dynamic index op (a store-index reshape, a select) needs its runtime
  // shapes, threaded in as shape captures; the block args are added below.
  IndexShapeCaptures shapeCaptures =
      IndexShapeCaptures::build(getBody(), loc, state);

  auto resultTy = cast<MO::TensorType>(getResult().getType());
  Value outShape = buildShape(loc, resultTy.getShape(), state);
  Type outElemType = resultTy.getElementType();

  ErrorOr<MOGG::KernelOutputFusionLambdaType> kindOr =
      classifyEpilogueLambdaKind(*this, state);
  if (kindOr.isError())
    return emitOpError() << kindOr.getError();
  MOGG::KernelOutputFusionLambdaType kind = *kindOr;
  // A downgraded `_LegacyComputeOutput` slot (`OutputLegacyForCompute`) keeps
  // a store body, which emit wraps as a store fusion.
  bool isCompute = kind == MOGG::KernelOutputFusionLambdaType::Compute;

  // A trait-ABI fused output's contract (null for the legacy MTS API).
  MOGG::TensorArgContractAttr contractAttr =
      consumingFusedContract(getResult(), /*isOutput=*/true, state);
  bool computeOutputArg = kind != MOGG::KernelOutputFusionLambdaType::Output;

  // Set by `AllocateAccumulatingBuffers`.
  Value intermediateTensor = getIntermediateTensor();
  if (!intermediateTensor)
    return emitOpError("MAPToMOGG: epilogue has no source tensor");
  Value outputPtr = MOGG::TensorGetPtrOp::create(
      state.builder, loc, MOGG::PointerType::get(ctx),
      state.mapping.lookup(intermediateTensor));

  auto epilogueOp = MOGG::TensorCreateEpilogueOp::create(
      state.builder, loc, MOGG::TensorType::get(ctx), outShape, moggCaptures,
      /*shapeCaptures=*/shapeCaptures.shapes,
      /*outputPtr=*/outputPtr,
      state.builder.getStringAttr(state.device.getLabel()),
      TypeAttr::get(outElemType), state.builder.getBoolAttr(true),
      MO::IOKindAttr::get(ctx, MO::IOKind::Output),
      state.builder.getBoolAttr(computeOutputArg),
      /*contract=*/contractAttr);

  // Block args mirror the old ones, with captures retyped to `!mogg.tensor`
  // and shape captures trailing.
  Block *body = state.builder.createBlock(&epilogueOp.getBody());
  // The value param is the SIMD vector the kernel hands in.
  for (size_t i = 0; i < numParams; ++i) {
    Type type = epiArgs[i].getType();
    if (i == 0)
      type = MOGG::SimdType::get(type);
    BlockArgument p = body->addArgument(type, epiArgs[i].getLoc());
    state.mapping.map(epiArgs[i], p);
  }
  for (size_t j = 0; j < numCaptures; ++j) {
    BlockArgument capArg = body->addArgument(moggCaptures[j].getType(),
                                             epiArgs[numParams + j].getLoc());
    state.mapping.map(epiArgs[numParams + j], capArg);
  }
  shapeCaptures.bind(body, loc, state);

  state.builder.setInsertionPointToEnd(body);
  // A compute lambda yields the value and the kernel stores it, so skip the
  // store to the store-target capture (found by tensor, not position). The
  // capture stays, since emit binds the functor onto its slice.
  Value yielded;
  Value storeTarget = isCompute ? epiBody.getArgument(numParams) : Value();
  if (failed(state.lowerRegionBody(epiBody, [&](Operation *op) {
        auto storeOp = dyn_cast<StoreOp>(op);
        if (!storeOp || storeOp.getTensor() != storeTarget)
          return true;
        ASSERT_STREAM(!yielded, "a MAP epilogue body must store to its "
                                "store-target capture exactly once");
        yielded = state.mapping.lookup(storeOp.getValue());
        return false;
      })))
    return failure();
  if (isCompute) {
    ASSERT_STREAM(yielded, "expected a MAP epilogue body to store to its "
                           "store-target capture");
    MOGG::YieldOp::create(state.builder, loc, yielded);
  }

  state.builder.setInsertionPointAfter(epilogueOp);
  state.mapping.map(getResult(), epilogueOp.getTensor());
  return success();
}

// map.iter.prologue -> mogg.tensor.create.prologue. Load-side mirror of
// `IterEpilogueOp::lowerToMOGG`.
LogicalResult IterPrologueOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  Block &proBody = getBody().front();
  Block::BlockArgListType proArgs = proBody.getArguments();
  size_t numCaptures = getCaptures().size();
  if (proArgs.size() != 1 + numCaptures)
    return emitOpError()
           << "MAPToMOGG: prologue body takes the index plus one arg per "
              "capture, got "
           << proArgs.size() << " args for " << numCaptures << " captures";

  SmallVector<Value> moggCaptures;
  for (Value cap : getCaptures()) {
    if (!state.mapping.contains(cap))
      return emitOpError("MAPToMOGG: prologue capture is not yet lowered");
    moggCaptures.push_back(state.mapping.lookup(cap));
  }

  // Dynamic index shapes, as in `IterEpilogueOp::lowerToMOGG`.
  IndexShapeCaptures shapeCaptures =
      IndexShapeCaptures::build(getBody(), loc, state);

  // From the RESULT, not a capture: after fusion the captures are the
  // producer's operands.
  auto resultTy = cast<MO::TensorType>(getResult().getType());
  auto prologueOp = MOGG::TensorCreatePrologueOp::create(
      state.builder, loc, /*tensor=*/MOGG::TensorType::get(ctx),
      /*shape=*/buildShape(loc, resultTy.getShape(), state),
      /*captures=*/moggCaptures, /*shapeCaptures=*/shapeCaptures.shapes,
      /*device=*/state.builder.getStringAttr(state.device.getLabel()),
      /*elementType=*/TypeAttr::get(resultTy.getElementType()),
      /*mut=*/state.builder.getBoolAttr(false),
      /*io=*/MO::IOKindAttr::get(ctx, MO::IOKind::Input),
      /*contract=*/
      consumingFusedContract(getResult(), /*isOutput=*/false, state));

  // As in the epilogue: captures retyped, shape captures trailing.
  Block *body = state.builder.createBlock(&prologueOp.getBody());
  for (auto [i, arg] : llvm::enumerate(proArgs)) {
    Type argTy = i == 0 ? arg.getType() : moggCaptures[i - 1].getType();
    state.mapping.map(arg, body->addArgument(argTy, arg.getLoc()));
  }
  shapeCaptures.bind(body, loc, state);

  state.builder.setInsertionPointToEnd(body);
  for (Operation &op : proBody.without_terminator())
    if (failed(state.lowerOp(&op)))
      return failure();

  auto yieldOp = cast<YieldOp>(proBody.getTerminator());
  SmallVector<Value> yielded;
  for (Value v : yieldOp.getOperands())
    yielded.push_back(state.mapping.lookup(v));
  MOGG::YieldOp::create(state.builder, yieldOp.getLoc(), yielded);

  state.builder.setInsertionPointAfter(prologueOp);
  state.mapping.map(getResult(), prologueOp.getTensor());
  return success();
}

// map.call.opaque -> mogg.call.opaque. Inputs and outputs (output block args
// or epilogue results) are already in `state.mapping`.
LogicalResult CallOpaqueOp::lowerToMOGG(MAPToMOGGState &state) {
  SmallVector<Value> callInputs;
  for (Value in : getInputs())
    callInputs.push_back(state.mapping.lookup(in));

  SmallVector<Value> callOutputs;
  for (Value callOut : getOutputs())
    callOutputs.push_back(state.mapping.lookup(callOut));
  Value opaqueOut = getOpaqueResult();
  Type opaqueResultTy = opaqueOut ? opaqueOut.getType() : Type();

  // Baked in so emit never inspects an output's producer.
  bool hasEpilogueFusion = llvm::any_of(callOutputs, [](Value out) {
    auto epilogue = out.getDefiningOp<MOGG::FusedEpilogueInterface>();
    return epilogue && epilogue.recomputesValue();
  });

  // Emit appends the contexts verbatim, so pass them only when execute
  // declares one (a host-only kernel takes none). A declarative call always
  // takes one: `execute_device` declares none, but the launch needs it.
  MOGG::KernelDeclOp decl = state.registry.lookupKernelBySymName(getCallee());
  MOGG::KernelDeclAdaptor adaptor(decl);
  ValueRange contexts;
  if (const auto &fn = adaptor.executeFunction;
      adaptor.isDeclarative() ||
      (fn && !fn->inputArguments.empty() &&
       (fn->inputArguments.back().isContextType() ||
        fn->inputArguments.back().isVariadicContextType())))
    contexts = state.contextArgs;

  auto call = MOGG::CallOpaqueOp::create(
      state.builder, getLoc(),
      opaqueResultTy ? TypeRange{opaqueResultTy} : TypeRange{}, getCalleeAttr(),
      callInputs, callOutputs, contexts, getParams(),
      state.builder.getBoolAttr(hasEpilogueFusion),
      state.builder.getBoolAttr(getHasSameVariadicOperandSize()));
  if (opaqueOut)
    state.mapping.map(opaqueOut, call.getOpaqueResult());
  return success();
}

// A view of an in-place kernel's fused buffer re-addresses mutable storage. The
// result is still a `!mogg.tensor`, since hoisted views are only loaded from.
static bool viewsMutableStorage(Value input) {
  return isa<MO::BufferType>(input.getType());
}

// map.index.slice.stride -> mogg.tensor.create.slice.
LogicalResult SliceStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      MO::castFromTensorLike(getInput().getType()).getElementType();

  // MOToMAP cannot resolve a negative start on a dynamic dim (e.g.
  // `[-window:]` off a dynamic `seq_len`), so mark those `?` for the primitive
  // to read from the runtime `starts` operand.
  SmallVector<int64_t> inDims = signature.getIn().getIntDims();
  SmallVector<int64_t> staticStart =
      llvm::to_vector(getSpec().getStarts().asArrayRef());
  for (size_t i = 0; i < staticStart.size(); ++i)
    if (staticStart[i] < 0 && ShapedType::isDynamic(inDims[i]))
      staticStart[i] = ShapedType::kDynamic;

  auto createSlice = MOGG::TensorCreateSliceOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      DenseI64ArrayAttr::get(ctx, staticStart), getSpec().getSteps(),
      TypeAttr::get(elementType),
      state.builder.getBoolAttr(viewsMutableStorage(getInput())),
      state.mapping.lookup(getInput()), state.mapping.lookup(getStarts()),
      state.mapping.lookup(getStops()), state.mapping.lookup(getSteps()));
  state.mapping.map(getResult(), createSlice.getResult());
  return success();
}

// map.index.broadcast.stride -> mogg.tensor.create.broadcast.
LogicalResult BroadcastStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      MO::castFromTensorLike(getInput().getType()).getElementType();
  auto createBroadcast = MOGG::TensorCreateBroadcastOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      TypeAttr::get(elementType),
      state.builder.getBoolAttr(viewsMutableStorage(getInput())),
      state.mapping.lookup(getInput()));
  state.mapping.map(getResult(), createBroadcast.getResult());
  return success();
}

// map.index.reshape.stride -> mogg.tensor.create.reshape. Row-major
// de-linearization needs the output shape's runtime values, so it gets a real
// `!mogg.shape` (via `buildShape`, so CSE can fold duplicates).
LogicalResult ReshapeStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      MO::castFromTensorLike(getInput().getType()).getElementType();
  Value shape = buildShape(getLoc(), signature.getOut(), state);
  auto createReshape = MOGG::TensorCreateReshapeOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      TypeAttr::get(elementType),
      state.builder.getBoolAttr(viewsMutableStorage(getInput())),
      state.mapping.lookup(getInput()), shape);
  state.mapping.map(getResult(), createReshape.getResult());
  return success();
}

// map.index.reshape -> mogg.index.reshape, lowered in place since a
// store-index reshape is not hoisted. The spec's OUT/IN shapes become
// `fromShape`/`toShape`.
LogicalResult ReshapeIndexOp::lowerToMOGG(MAPToMOGGState &state) {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  // The owning epilogue or prologue threads in the runtime from/to shapes.
  ValueRange dynamicShapes;
  if (!signature.getOut().isStatic() || !signature.getIn().isStatic()) {
    auto it = state.indexShapeArgs.find(getOperation());
    // Emit would silently use the static defaults, so fail here.
    if (it == state.indexShapeArgs.end())
      return emitOpError("MAPToMOGG: a dynamic index reshape has no shape "
                         "captures; its parent functor did not thread them in");
    dynamicShapes = ValueRange(it->second);
  }
  auto indexReshape = MOGG::IndexReshapeOp::create(
      state.builder, getLoc(), getResult().getType(),
      state.mapping.lookup(getIndex()), signature.getOut(), signature.getIn(),
      dynamicShapes);
  state.mapping.map(getResult(), indexReshape.getResult());
  return success();
}

// map.index.transpose.stride -> mogg.tensor.create.transpose.
LogicalResult TransposeStrideOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  Type elementType =
      MO::castFromTensorLike(getInput().getType()).getElementType();
  auto createTranspose = MOGG::TensorCreateTransposeOp::create(
      state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
      getSpec().getPerm(), TypeAttr::get(elementType),
      state.builder.getBoolAttr(viewsMutableStorage(getInput())),
      state.mapping.lookup(getInput()),
      state.mapping.lookup(getPermutations()));
  state.mapping.map(getResult(), createTranspose.getResult());
  return success();
}

// map.iter.elementwise.body -> mogg.call.foreach.
LogicalResult IterElementwiseBodyOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  // From `$shape`, not an I/O tensor: they can differ (e.g. a broadcast).
  Value loopShape = buildShape(loc, getShape(), state);

  MOGG::PrimitiveDeclOp foreachPrim =
      state.registry.lookupPrimitive(MOGG::CallForeachOp::getOperationName());

  // A dynamic index op (a select) reads its runtime shapes, built here in the
  // kernel scope, which the foreach's closure captures.
  IndexShapeCaptures::build(getBody(), loc, state).bindInScope(state);

  // Every capture (kernel I/O or hoisted view) is defined, and so lowered,
  // earlier in the block.
  Block &body = getBody().front();
  Value oldIndexArg = body.getArgument(0);
  ValueRange captures = getCaptures();
  Block::BlockArgListType captureBlockArgs =
      body.getArguments().take_back(captures.size());

  SmallVector<Value> capturing;
  for (Value v : captures)
    capturing.push_back(state.mapping.lookup(v));

  // A foreach kernel is single-device, so it has exactly one context.
  auto callForeach = MOGG::CallForeachOp::create(
      state.builder, loc, FlatSymbolRefAttr::get(ctx, foreachPrim.getSymName()),
      loopShape, state.contextArgs.front(), capturing,
      state.builder.getStringAttr(state.device.getLabel()));

  // One arg per capture, then the index.
  Block *foreachBlock = state.builder.createBlock(&callForeach.getBody());
  for (auto [i, v] : llvm::enumerate(captureBlockArgs))
    state.mapping.map(v,
                      foreachBlock->addArgument(capturing[i].getType(), loc));
  state.mapping.map(oldIndexArg,
                    foreachBlock->addArgument(oldIndexArg.getType(), loc));

  state.builder.setInsertionPointToStart(foreachBlock);
  if (failed(state.lowerRegionBody(body)))
    return failure();

  state.builder.setInsertionPointAfter(callForeach);
  return success();
}

// map.iter.rowwise.body -> mogg.call.rowwise.
LogicalResult IterRowwiseBodyOp::lowerToMOGG(MAPToMOGGState &state) {
  Location loc = getLoc();
  Value rowShape = buildShape(loc, getShape(), state);

  Block &body = getBody().front();
  BlockArgument oldRowArg = body.getArgument(0);
  ValueRange captures = getCaptures();
  Block::BlockArgListType captureBlockArgs =
      body.getArguments().take_back(captures.size());

  SmallVector<Value> capturing;
  for (Value v : captures)
    capturing.push_back(state.mapping.lookup(v));

  auto callRowwise = MOGG::CallRowwiseOp::create(
      state.builder, loc, rowShape, getAxisAttr(), state.contextArgs.front(),
      capturing, state.builder.getStringAttr(state.device.getLabel()),
      /*primary=*/nullptr, /*dtype=*/nullptr, /*accum=*/nullptr,
      /*is_cached=*/nullptr, /*cols=*/nullptr, /*num_phases=*/nullptr,
      /*dtype_size=*/nullptr, /*computationally_expensive=*/nullptr,
      /*simd_monoid=*/nullptr);

  // One arg per capture, then the row index.
  Block *rowwiseBlock = state.builder.createBlock(&callRowwise.getBody());
  for (auto [i, v] : llvm::enumerate(captureBlockArgs))
    state.mapping.map(v,
                      rowwiseBlock->addArgument(capturing[i].getType(), loc));
  state.mapping.map(oldRowArg,
                    rowwiseBlock->addArgument(oldRowArg.getType(), loc));

  state.builder.setInsertionPointToStart(rowwiseBlock);
  if (failed(state.lowerRegionBody(body)))
    return failure();

  state.builder.setInsertionPointAfter(callRowwise);
  return success();
}

// map.iter.reduce -> mogg.rowwise.reduce.
LogicalResult IterReduceOp::lowerToMOGG(MAPToMOGGState &state) {
  Location loc = getLoc();
  SmallVector<Type> resultTypes = llvm::map_to_vector(
      getResultTypes(), [](Type t) -> Type { return MOGG::SimdType::get(t); });
  auto reduce = MOGG::RowwiseReduceOp::create(
      state.builder, loc, resultTypes, getMonoidAttr(),
      /*params=*/nullptr, /*fields=*/nullptr);

  Block *reduceBlock = state.builder.createBlock(&reduce.getBody());
  state.mapping.map(getIndexArg(),
                    reduceBlock->addArgument(getIndexArg().getType(), loc));

  state.builder.setInsertionPointToStart(reduceBlock);
  for (Operation &op : getBody().front().without_terminator())
    if (failed(state.lowerOp(&op)))
      return failure();
  SmallVector<Value> yielded;
  for (Value v : getYield().getOperands())
    yielded.push_back(state.mapping.lookup(v));
  MOGG::YieldOp::create(state.builder, getYield().getLoc(), yielded);

  state.builder.setInsertionPointAfter(reduce);
  state.mapping.map(getResults(), reduce.getResults());
  return success();
}

// map.iter.opaque -> mogg.kernel.
LogicalResult IterOpaqueOp::lowerToMOGG(MAPToMOGGState &state) {
  MLIRContext *ctx = getContext();
  Location loc = getLoc();
  ValueRange inputs = getInputs();
  TypeRange resultTypes = getResultTypes();

  Value inputChain = getInputChain();
  bool hasOutputChain = getOutputChain() != nullptr;

  // Reconstructed tensors record only the device label; the full ref stays on
  // the `mogg.routine` for device-context resolution.
  state.device = MO::getDeviceOrDefault(*this);

  // A fresh leaf only when absent (e.g. hand-written test IR).
  Attribute rawFusionInfo = getFusionInfoAttr();
  MOGG::FusionInfoAttr fusionInfo =
      rawFusionInfo
          ? cast<MOGG::FusionInfoAttr>(rawFusionInfo)
          : MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(*this));

  OpBuilder &builder = state.builder;
  auto kernel = MOGG::RoutineOp::create(builder, loc, resultTypes, inputs,
                                        inputChain, fusionInfo);
  kernel.setDeviceRef(state.device);
  // Without these, a kernel whose only effect is a mutable store would be
  // dead.
  kernel.setHasReadEffect(getHasReadEffect());
  kernel.setHasWriteEffect(getHasWriteEffect());
  kernel.setDispatch(getDispatch());

  // Dims anywhere in the kernel, not just its I/O: e.g. the reduced axis of a
  // dynamic-perm transpose fused into a reduce exists only on the view.
  collectDynamicDims(getOperation(), state);
  getBody().walk([&](Operation *op) { collectDynamicDims(op, state); });

  // Body block args: non-chain inputs, DPS outputs, dims, contexts. A
  // tensor-like entry gets a pointer; an opaque input crosses as itself.
  Block *body = builder.createBlock(&kernel.getBody());
  auto addBodyArg = [&](Type ty) {
    return body->addArgument(
        isa<MO::OpaqueType>(ty) ? ty : Type(MOGG::PointerType::get(ctx)), loc);
  };
  SmallVector<BlockArgument> inputArgs;
  // Indexed by result number; null for a by-value opaque or the chain.
  SmallVector<BlockArgument> outputArgs(resultTypes.size(), BlockArgument());
  for (Value in : inputs)
    inputArgs.push_back(addBodyArg(in.getType()));
  for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
    if (hasOutputChain && i + 1 == e)
      break;
    if (isa<MO::OpaqueType>(resultTypes[i]))
      continue;
    outputArgs[i] = addBodyArg(resultTypes[i]);
  }
  // E.g. `!mogg.dim<batch>`. Shapes hold no dim expressions, so each dynamic
  // dim is a `ParamDeclRefAttr`.
  for (auto &[dim, blockArg] : state.dimArgs)
    blockArg = body->addArgument(
        MOGG::DimType::get(ctx, dyn_cast<KGEN::ParamDeclRefAttr>(dim)), loc);
  // A kernel whose `execute` takes a `DeviceContextArray` gets one
  // `!mogg.context` per unique device (in `getUniqueDevicesFromKernel` order);
  // every other kernel gets one, even if it names several devices (e.g. a GPU
  // op with CPU scalar operands like `pad`'s paddings).
  //
  // TODO(GEX-3701): label each context with its `DeviceRefAttr` instead of
  // relying on creation order matching the order `lowerToAsync`
  // re-derives -- any reordering of `getUniqueDevicesFromKernel` would silently
  // misassign devices.
  bool variadicContext = false;
  getBody().walk([&](CallOpaqueOp call) {
    MOGG::KernelDeclOp decl =
        state.registry.lookupKernelBySymName(call.getCallee());
    if (!decl)
      return WalkResult::advance();
    MOGG::KernelDeclAdaptor adaptor(decl);
    if (!adaptor.executeFunction)
      return WalkResult::advance();
    for (const auto &arg : adaptor.executeFunction->inputArguments)
      if (arg.isVariadicContextType()) {
        variadicContext = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  size_t numContexts =
      variadicContext
          ? std::max<size_t>(1, MO::getUniqueDevicesFromKernel(*this).size())
          : 1;
  state.contextArgs.clear();
  for (size_t i = 0; i < numContexts; ++i)
    state.contextArgs.push_back(
        body->addArgument(MOGG::ContextType::get(ctx), loc));

  builder.setInsertionPointToEnd(body);

  // Keyed on the block args, since the region is isolated from above.
  Block::BlockArgListType inputBlockArgs = getInputBlockArgs();
  for (auto [i, in] : llvm::enumerate(inputs)) {
    if (isa<MO::OpaqueType>(in.getType())) {
      state.mapping.map(inputBlockArgs[i], inputArgs[i]);
      continue;
    }
    // A `!mo.buffer` input (e.g. a collective's signal buffers) is mutable.
    bool isMut = isa<MO::BufferType>(in.getType());
    state.mapping.map(inputBlockArgs[i],
                      buildTensor(loc, inputArgs[i], in.getType(),
                                  MO::IOKind::Input, isMut, state));
  }

  Block &opaqueBody = getBody().front();
  for (auto [i, outputBlockArg] : llvm::enumerate(getOutputBlockArgs()))
    state.mapping.map(outputBlockArg,
                      buildTensor(loc, outputArgs[i], resultTypes[i],
                                  MO::IOKind::Output, /*isMut=*/true, state));

  // `map.yield` becomes the `mogg.output` forwarding the by-value result.
  auto yieldOp = dyn_cast<YieldOp>(opaqueBody.back());
  if (failed(state.lowerRegionBody(
          opaqueBody, [&](Operation *op) { return op != yieldOp; })))
    return failure();

  // A yielded DPS output block arg is redundant and not forwarded.
  if (getOpaqueResult() && yieldOp) {
    Block::BlockArgListType outputArgs = getOutputBlockArgs();
    for (Value operand : yieldOp.getOperands())
      if (!llvm::is_contained(outputArgs, operand)) {
        MOGG::OutputOp::create(builder, loc, state.mapping.lookup(operand));
        break;
      }
  }

  replaceAllUsesWith(kernel.getResults());
  erase();
  return success();
}

} // namespace M::MAP
