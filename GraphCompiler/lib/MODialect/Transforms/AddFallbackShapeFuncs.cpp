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

#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/StringSet.h"

#define DEBUG_TYPE "add-fallback-shape-funcs"

namespace M::MO {

#define GEN_PASS_DEF_ADDFALLBACKSHAPEFUNCTIONS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Fallback guard creation (to ensure all MO ops have one or error)
//===----------------------------------------------------------------------===//

// C++ workaround so `false` depends on `T`
template <class>
[[maybe_unused]] constexpr bool FALSE = false;

// Once every MO op is instantiated from the generated headers, an op without a
// fallback or NEEDS_NO_FALLBACK entry becomes a compile-time error.
template <typename T>
struct Fallback {
  static_assert(
      FALSE<T>,
      "All MO operations must have registered fallback shape "
      "functions but no function found for this MO op. (see template error)");
};

#define HAS_FALLBACK(MOOp)                                                     \
  template <>                                                                  \
  struct Fallback<MOOp> {                                                      \
    ErrorOrSuccess build(GraphOp &graph, OpBuilder &builder, MOOp op);         \
  };

#define NEEDS_NO_FALLBACK(MOOp)                                                \
  template <>                                                                  \
  struct Fallback<MOOp> {};

HAS_FALLBACK(MO::ArgMaxOp)
HAS_FALLBACK(MO::ArgMinOp)
HAS_FALLBACK(MO::AvgPoolOp)
HAS_FALLBACK(MO::AvgPoolCeilOp)
HAS_FALLBACK(MO::BatchMatmulOp)
HAS_FALLBACK(MO::BottomKOp)
HAS_FALLBACK(MO::BroadcastToOp)
HAS_FALLBACK(MO::BroadcastShapeOp)
HAS_FALLBACK(MO::ConcatOp)
HAS_FALLBACK(MO::ConvOp)
HAS_FALLBACK(MO::ConvTransposeOp)
HAS_FALLBACK(MO::GatherOp)
HAS_FALLBACK(MO::GatherNdOp)
HAS_FALLBACK(MO::ReduceGroupNormOp)
HAS_FALLBACK(MO::LayerNormOp)
HAS_FALLBACK(MO::ReduceMinAndMaxOp)
HAS_FALLBACK(MO::ReduceRMSNormOp)
HAS_FALLBACK(MO::Composite::ReduceRMSNormRoPEOp)
HAS_FALLBACK(MO::Composite::ReduceRMSNormFusedResidualAddOp)
HAS_FALLBACK(MO::Composite::RMSNormResidualAddOp)
HAS_FALLBACK(MO::MaxPoolOp)
HAS_FALLBACK(MO::MaxPoolCeilOp)
HAS_FALLBACK(MO::NonMaximumSuppressionOp)
HAS_FALLBACK(MO::PadConstantOp)
HAS_FALLBACK(MO::PadReflectOp)
HAS_FALLBACK(MO::PadRepeatOp)
HAS_FALLBACK(MO::RandomNormalOp)
HAS_FALLBACK(MO::RandomUniformOp)
HAS_FALLBACK(MO::RangeOp)
HAS_FALLBACK(MO::ReduceMulOp)
HAS_FALLBACK(MO::ReshapeOp)
HAS_FALLBACK(MO::RoiAlignOp)
HAS_FALLBACK(MO::ResizeBicubicOp)
HAS_FALLBACK(MO::ResizeLinearOp)
HAS_FALLBACK(MO::ResizeNearestOp)
HAS_FALLBACK(MO::ScatterOp)
HAS_FALLBACK(MO::ScatterAddOp)
HAS_FALLBACK(MO::ScatterMaxOp)
HAS_FALLBACK(MO::ScatterMinOp)
HAS_FALLBACK(MO::ScatterMulOp)
HAS_FALLBACK(MO::ScatterNdOp)
HAS_FALLBACK(MO::ScatterNdAddOp)
HAS_FALLBACK(MO::ScatterNdMaxOp)
HAS_FALLBACK(MO::ScatterNdMinOp)
HAS_FALLBACK(MO::ScatterNdMulOp)
HAS_FALLBACK(MO::SliceOp)
HAS_FALLBACK(MO::SplitOp)
HAS_FALLBACK(MO::SqueezeShapeOp)
HAS_FALLBACK(MO::TopKOp)
HAS_FALLBACK(MO::TransposeOp)
HAS_FALLBACK(MO::UnsqueezeShapeOp)
HAS_FALLBACK(MO::ArgNonzeroOp)
HAS_FALLBACK(MO::TileOp)

template <>
struct Fallback<MO::CustomOp> {
  ErrorOrSuccess build(GraphOp &graph, OpBuilder &builder, MO::CustomOp op,
                       MOGG::ShapeFunctionDeclOp shapeDecl,
                       StringSet<> &inputParamsNeeded);
};

NEEDS_NO_FALLBACK(MO::MatmulOp)
NEEDS_NO_FALLBACK(MO::Static::BroadcastToOp)

// Creates the fallback for this op if it has parameters still needing decls.
#define CALL_FALLBACK(MOOp)                                                    \
  if (auto asOp = dyn_cast<MOOp>(op.getOperation())) {                         \
    auto result = Fallback<MOOp>{}.build(graph, builder, asOp);                \
    if (result.isError()) {                                                    \
      asOp.emitError() << "Failed to fallback shape function with error: "     \
                       << result.takeError();                                  \
      return mlir::WalkResult::interrupt();                                    \
    } else {                                                                   \
      op.setOutputParamDecls({});                                              \
    }                                                                          \
    return mlir::WalkResult::advance();                                        \
  }

//===----------------------------------------------------------------------===//
// Fallback creation pass.
//===----------------------------------------------------------------------===//

struct AddFallbackShapeFunctions
    : public impl::AddFallbackShapeFunctionsBase<AddFallbackShapeFunctions> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graph = getOperation();
    MLIRContext *ctx = graph.getContext();
    OpBuilder builder{ctx};
    MOGG::MojoRegistry registry(graph->getParentOfType<mlir::ModuleOp>());

    WalkResult result = graph.walk([&](ParamDeclarationInterface op) {
      // Shape function was already materialized or never needed.
      if (op.getOutputParamDecls().empty())
        return mlir::WalkResult::advance();

      builder.setInsertionPoint(op);
      CALL_FALLBACK(MO::ArgMaxOp)
      CALL_FALLBACK(MO::ArgMinOp)
      CALL_FALLBACK(MO::AvgPoolOp)
      CALL_FALLBACK(MO::AvgPoolCeilOp)
      CALL_FALLBACK(MO::BatchMatmulOp)
      CALL_FALLBACK(MO::BottomKOp)
      CALL_FALLBACK(MO::BroadcastToOp)
      CALL_FALLBACK(MO::BroadcastShapeOp)
      CALL_FALLBACK(MO::ConcatOp)
      CALL_FALLBACK(MO::ConvOp)
      CALL_FALLBACK(MO::ConvTransposeOp)
      CALL_FALLBACK(MO::GatherOp)
      CALL_FALLBACK(MO::GatherNdOp)
      CALL_FALLBACK(MO::ReduceGroupNormOp)
      CALL_FALLBACK(MO::LayerNormOp)
      CALL_FALLBACK(MO::ReduceMinAndMaxOp)
      CALL_FALLBACK(MO::ReduceRMSNormOp)
      CALL_FALLBACK(MO::Composite::ReduceRMSNormRoPEOp)
      CALL_FALLBACK(MO::Composite::ReduceRMSNormFusedResidualAddOp)
      CALL_FALLBACK(MO::Composite::RMSNormResidualAddOp)
      CALL_FALLBACK(MO::MaxPoolOp)
      CALL_FALLBACK(MO::MaxPoolCeilOp)
      CALL_FALLBACK(MO::NonMaximumSuppressionOp)
      CALL_FALLBACK(MO::PadConstantOp)
      CALL_FALLBACK(MO::PadReflectOp)
      CALL_FALLBACK(MO::PadRepeatOp)
      CALL_FALLBACK(MO::RandomNormalOp)
      CALL_FALLBACK(MO::RandomUniformOp)
      CALL_FALLBACK(MO::RangeOp)
      CALL_FALLBACK(MO::ReduceMulOp)
      CALL_FALLBACK(MO::ReshapeOp)
      CALL_FALLBACK(MO::RoiAlignOp)
      CALL_FALLBACK(MO::ResizeBicubicOp)
      CALL_FALLBACK(MO::ResizeLinearOp)
      CALL_FALLBACK(MO::ResizeNearestOp)
      CALL_FALLBACK(MO::ScatterOp)
      CALL_FALLBACK(MO::ScatterAddOp)
      CALL_FALLBACK(MO::ScatterMaxOp)
      CALL_FALLBACK(MO::ScatterMinOp)
      CALL_FALLBACK(MO::ScatterMulOp)
      CALL_FALLBACK(MO::ScatterNdOp)
      CALL_FALLBACK(MO::ScatterNdAddOp)
      CALL_FALLBACK(MO::ScatterNdMaxOp)
      CALL_FALLBACK(MO::ScatterNdMinOp)
      CALL_FALLBACK(MO::ScatterNdMulOp)
      CALL_FALLBACK(MO::SliceOp)
      CALL_FALLBACK(MO::SplitOp)
      CALL_FALLBACK(MO::SqueezeShapeOp)
      CALL_FALLBACK(MO::TopKOp)
      CALL_FALLBACK(MO::TransposeOp)
      CALL_FALLBACK(MO::UnsqueezeShapeOp)
      CALL_FALLBACK(MO::ArgNonzeroOp)
      CALL_FALLBACK(MO::TileOp)

      if (auto asOp = dyn_cast<MO::CustomOp>(op.getOperation())) {
        if (!registry.containsKernel(asOp.getSymbol())) {
          asOp.emitError() << "Could not find registered mojo kernel named '"
                           << asOp.getSymbol() << "'";
          return mlir::WalkResult::interrupt();
        }

        // We only get here if the op declares its own output dims, and a shape
        // function is the only thing that can compute them.
        if (!registry.containsShapeFunction(asOp.getSymbol())) {
          asOp.emitError()
              << "custom op '" << asOp.getSymbol()
              << "' with data-dependent output dimensions requires a "
                 "registered shape function; add "
                 "'@extensibility.register_shape_function(\""
              << asOp.getSymbol() << "\")' to the kernel";
          return mlir::WalkResult::interrupt();
        }

        // Forward custom-op attributes matching the shape fn's own params,
        // which may differ from execute's (e.g. no `target`/`_trace_name`).
        StringSet<> inputParams;
        MOGG::ShapeFunctionDeclOp shapeDecl =
            registry.lookupShapeFunction(asOp.getSymbol());
        for (mlir::Attribute attr : shapeDecl.getFunction().getParams())
          inputParams.insert(mlir::cast<KGEN::ParamDeclAttr>(attr).getName());

        auto result = Fallback<MO::CustomOp>{}.build(graph, builder, asOp,
                                                     shapeDecl, inputParams);
        if (result.isError()) {
          asOp.emitError() << "Failed to fallback shape function with error: "
                           << result.takeError();
          return mlir::WalkResult::interrupt();

        } else {
          op.setOutputParamDecls({});
        }
      }

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted())
      return signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// Fallback helpers
//===----------------------------------------------------------------------===//

// Fallbacks produce a parameter for every output dim, even static ones, so the
// lowering needn't know how many there are or which dims they refer to.
KGEN::ParamDeclArrayAttr
getNewParameterMapping(GraphOp &graph, TensorType outTy,
                       ArrayRef<ParamDeclAttr> outputParamDecls) {
  MLIRContext *ctx = graph.getContext();
  if (auto rankedParamDeclRef =
          dyn_cast<KGEN::ParamDeclRefAttr>(outTy.getShape())) {
    auto paramDecl = KGEN::ParamDeclAttr::get(rankedParamDeclRef);
    return KGEN::ParamDeclArrayAttr::get(ctx, {paramDecl});
  }

  llvm::SmallSetVector<ParamDeclAttr, 10> outputParamDeclSet(
      outputParamDecls.begin(), outputParamDecls.end());

  SmallVector<KGEN::ParamDeclAttr> params;

  // The op's declared parameters move to the shape func invocation. Other dims
  // (static, or defined elsewhere) get a placeholder "F"allback param to keep
  // the shape func contract.
  auto getShapeFuncSymbol = [&](TypedAttr dim) -> KGEN::ParamDeclAttr {
    auto asSymbolic = dyn_cast<KGEN::ParamDeclRefAttr>(dim);

    if (!asSymbolic)
      return graph.getNewDimParam("F");

    auto asParamDeclAttr = KGEN::ParamDeclAttr::get(asSymbolic);
    if (!outputParamDeclSet.contains(asParamDeclAttr))
      return graph.getNewDimParam("F");

    return asParamDeclAttr;
  };

  // Parameters are positional, one per dim.
  for (auto dim : outTy.getShape().getValues()) {
    params.push_back(getShapeFuncSymbol(dim));
  }
  return KGEN::ParamDeclArrayAttr::get(graph.getContext(), params);
}

//===----------------------------------------------------------------------===//
// Helpers for fallback implementations
//===----------------------------------------------------------------------===//

static ErrorOrSuccess buildForReductionOp(GraphOp &graph, OpBuilder &builder,
                                          MO::Reduction op) {
  KGEN::ParamDeclArrayAttr shapeParams = getNewParameterMapping(
      graph, op.getResult().getType(), op.getOutputParamDecls());

  MO::InvokeShapeFuncOp::create(
      builder, op.getLoc(), op->getOperands(), op->getName().getStringRef(),
      shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({1}));

  return success();
}

static ErrorOrSuccess buildForScatterLikeOp(GraphOp &graph, OpBuilder &builder,
                                            MO::ScatterLike op) {
  KGEN::ParamDeclArrayAttr shapeParams = getNewParameterMapping(
      graph, op.getResult().getType(), op.getOutputParamDecls());

  MO::InvokeShapeFuncOp::create(
      builder, op.getLoc(), op->getOperands(),
      op.getOperation()->getName().getStringRef(), shapeParams,
      /*kgenAttrs*/ DictionaryAttr{},
      /*Data dependent input indices*/ builder.getI32ArrayAttr({3}));

  return success();
}

void buildForOpResult(GraphOp &graph, OpBuilder &builder, TensorValue res,
                      StringRef shapeFuncName, ValueRange operands,
                      ArrayRef<ParamDeclAttr> outputParamDecls,
                      ArrayRef<std::pair<StringRef, Attribute>> kgenParams,
                      ArrayRef<int32_t> dataDepInputIdxs) {
  KGEN::ParamDeclArrayAttr shapeParams =
      getNewParameterMapping(graph, res.getType(), outputParamDecls);

  // TODO we can't deduce parameter from types of variadic inputs
  DictionaryAttr kgenParamAttrs;
  if (!kgenParams.empty()) {
    SmallVector<mlir::NamedAttribute> namedAttrs;
    for (auto [name, attr] : kgenParams)
      namedAttrs.push_back(builder.getNamedAttr(name, attr));
    kgenParamAttrs = builder.getDictionaryAttr(namedAttrs);
  }

  Location loc = res.getLoc();

  // Shape functions execute on the host and read data-dependent operands
  // through their raw storage pointers. An off-host operand would hand the
  // host a device address, so copy it to the host first. The transfer is a
  // real runtime D2H: data-dependent output shapes need current input data
  // on every execution.
  DeviceRefAttr hostDevice = getHostDevice(builder.getContext());
  SmallVector<Value> invokeOperands(operands.begin(), operands.end());
  Value chain;
  for (int32_t inputIdx : dataDepInputIdxs) {
    if (static_cast<size_t>(inputIdx) >= invokeOperands.size())
      continue;
    auto tensorTy =
        dyn_cast<MO::TensorType>(invokeOperands[inputIdx].getType());
    if (!tensorTy)
      continue;
    DeviceRefAttr operandDevice = tensorTy.getDeviceRef();
    if (!operandDevice || operandDevice == hostDevice)
      continue;
    if (!chain)
      chain = MO::ChainCreateOp::create(builder, loc, ValueRange{}).getResult();
    auto transfer = MO::TransferOp::create(
        builder, loc, cast<TensorValue>(invokeOperands[inputIdx]), hostDevice,
        cast<ChainValue>(chain),
        /*alwaysElideSameDeviceCopy=*/false);
    invokeOperands[inputIdx] = transfer.getResult();
    chain = transfer.getOutChain();
  }

  MO::InvokeShapeFuncOp::create(builder, loc, invokeOperands, shapeFuncName,
                                shapeParams, kgenParamAttrs,
                                builder.getI32ArrayAttr(dataDepInputIdxs));
}

//===----------------------------------------------------------------------===//
// Actual fallback implementations
//===----------------------------------------------------------------------===//

template <typename OpT>
static ErrorOrSuccess buildForScatterNdLikeOp(GraphOp &graph,
                                              OpBuilder &builder, OpT op) {

  KGEN::ParamDeclArrayAttr shapeParams = getNewParameterMapping(
      graph, op.getResult().getType(), op.getOutputParamDecls());
  buildForOpResult(graph, builder, op, OpT::getOperationName(),
                   op.getOperands(), shapeParams,
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());

  return success();
}

ErrorOrSuccess Fallback<MO::AvgPoolOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::AvgPoolOp op) {
  buildForOpResult(graph, builder, op, MO::AvgPoolOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::AvgPoolCeilOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::AvgPoolCeilOp op) {
  buildForOpResult(graph, builder, op, MO::AvgPoolCeilOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::BatchMatmulOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::BatchMatmulOp op) {
  buildForOpResult(graph, builder, op, MO::BatchMatmulOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::BottomKOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::BottomKOp op) {
  buildForOpResult(graph, builder, op.getValues(),
                   MO::BottomKOp::getOperationName(), op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{1, 2});
  // indices() guaranteed by verifier to have the same shape as values()
  return success();
}

ErrorOrSuccess Fallback<MO::BroadcastToOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::BroadcastToOp op) {
  buildForOpResult(graph, builder, op, MO::BroadcastToOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{1});
  return success();
}

ErrorOrSuccess Fallback<MO::BroadcastShapeOp>::build(GraphOp &graph,
                                                     OpBuilder &builder,
                                                     MO::BroadcastShapeOp op) {
  buildForOpResult(graph, builder, op, MO::BroadcastShapeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::ConcatOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::ConcatOp op) {
  // TODO we can't deduce parameter from types of variadic inputs
  auto inputTensorTy = cast<TensorType>(op.getInputs().front().getType());
  auto inputDTypeAttr = mlir::TypeAttr::get(inputTensorTy.getElementType());
  auto inputRankAttr = builder.getIndexAttr(inputTensorTy.getRank());

  buildForOpResult(graph, builder, op, MO::ConcatOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {{"input_type", inputDTypeAttr},
                    {"input_rank", inputRankAttr},
                    {"axis", op.getAxisAttr()}},
                   op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ConvOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::ConvOp op) {
  buildForOpResult(graph, builder, op, MO::ConvOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ConvTransposeOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ConvTransposeOp op) {
  buildForOpResult(graph, builder, op, MO::ConvTransposeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

// A shape-function argument bound to a bare `Tensor` can't read element
// values (see `_MoggShapeTensorArg`), so only other operands need the host
// copy. Operands map to arguments as ConformCallOperands maps the shape call.
static ErrorOr<SmallVector<int32_t>> getCustomOpDataDepInputIdxs(
    MO::CustomOp op, MOGG::ShapeFunctionDeclOp shapeDecl, ValueRange operands) {
  MOGG::KernelDeclFunctionAdaptor shapeFn(shapeDecl);
  auto mapping = MOGG::mapToKernelImplementation(
      operands, /*results=*/{}, shapeFn, op.getSymbol(),
      /*hasSameVariadicOperandSize=*/false,
      /*hasExtraOutputShapeArgument=*/false, /*resultLeavesByValue=*/true);
  if (mapping.isError())
    return mapping.takeError();

  SmallVector<int32_t> dataDepInputIdxs;
  for (const auto &[idx, arg] : mapping->operandMapping) {
    std::optional<MOGG::TensorArgContract> contract = arg.getTraitContract();
    if (!contract || contract->bound != MOGG::TensorArgBound::tensor)
      dataDepInputIdxs.push_back(static_cast<int32_t>(idx));
  }
  return dataDepInputIdxs;
}

ErrorOrSuccess Fallback<MO::CustomOp>::build(
    GraphOp &graph, OpBuilder &builder, MO::CustomOp op,
    MOGG::ShapeFunctionDeclOp shapeDecl, StringSet<> &inputParamsNeeded) {
  SmallVector<Value> operands{op.getOperands().begin(), op.getOperands().end()};

  // Buffer operands are loaded, threading the chain.
  Value currentChain = nullptr;
  MO::MutableLoadOp firstLoadOp = nullptr;
  if (!operands.empty() && isa<MO::ChainType>(operands.back().getType())) {
    currentChain = operands.back();
    operands.pop_back();
  }

  for (Value &operand : operands) {
    auto bufferTy = dyn_cast<MO::BufferType>(operand.getType());
    if (!bufferTy)
      continue;

    auto bufferLoadOp = MO::MutableLoadOp::create(
        builder, op.getLoc(), bufferTy.toTensorType(), currentChain.getType(),
        operand, currentChain);
    if (!firstLoadOp)
      firstLoadOp = bufferLoadOp;
    operand = bufferLoadOp.getOutTensor();
    currentChain = bufferLoadOp.getOutChain();
  }

  ErrorOr<SmallVector<int32_t>> dataDepInputIdxs =
      getCustomOpDataDepInputIdxs(op, shapeDecl, operands);
  if (dataDepInputIdxs.isError())
    return dataDepInputIdxs.takeError();

  for (auto [idx, res] : llvm::enumerate(op.getResults())) {
    SmallVector<std::pair<StringRef, Attribute>> kgenParams;
    kgenParams.emplace_back("output_idx", builder.getIndexAttr(idx));

    if (!isa<MO::TensorType>(res.getType()))
      continue;

    for (const NamedAttribute &namedAttr : op.getParameters())
      if (inputParamsNeeded.contains(namedAttr.getName()))
        kgenParams.emplace_back(namedAttr.getName(), namedAttr.getValue());

    buildForOpResult(graph, builder, cast<TensorValue>(res), op.getSymbol(),
                     operands, op.getOutputParamDecls(),
                     /*kgenParams=*/kgenParams,
                     /*dataDepInputIdxs=*/*dataDepInputIdxs);
  }

  // The custom op consumes the chain after all buffer loads.
  if (currentChain)
    op->getOperands().back().replaceAllUsesExcept(currentChain, firstLoadOp);

  return success();
}

ErrorOrSuccess Fallback<MO::GatherOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::GatherOp op) {
  // `axis` is a compile-time `index` attribute threaded to the shape function.
  buildForOpResult(graph, builder, op, MO::GatherOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{{"axis", op.getAxisAttr()}},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::GatherNdOp>::build(GraphOp &graph,
                                               OpBuilder &builder,
                                               MO::GatherNdOp op) {
  buildForOpResult(graph, builder, op, MO::GatherNdOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {
                       {"batch_dims", op.getBatchDimsAttr()},
                   },
                   op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::LayerNormOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::LayerNormOp op) {
  buildForOpResult(graph, builder, op, MO::LayerNormOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::MaxPoolOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::MaxPoolOp op) {

  buildForOpResult(graph, builder, op, MO::MaxPoolOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::MaxPoolCeilOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::MaxPoolCeilOp op) {

  buildForOpResult(graph, builder, op, MO::MaxPoolCeilOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess
Fallback<MO::NonMaximumSuppressionOp>::build(GraphOp &graph, OpBuilder &builder,
                                             MO::NonMaximumSuppressionOp op) {
  buildForOpResult(graph, builder, op,
                   MO::NonMaximumSuppressionOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0, 1, 2, 3, 4});
  return success();
}

ErrorOrSuccess Fallback<MO::PadConstantOp>::build(GraphOp &graph,
                                                  OpBuilder &builder,
                                                  MO::PadConstantOp op) {
  buildForOpResult(graph, builder, op, MO::PadConstantOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::PadReflectOp>::build(GraphOp &graph,
                                                 OpBuilder &builder,
                                                 MO::PadReflectOp op) {
  buildForOpResult(graph, builder, op, MO::PadReflectOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::PadRepeatOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::PadRepeatOp op) {
  buildForOpResult(graph, builder, op, MO::PadRepeatOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RandomNormalOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::RandomNormalOp op) {
  buildForOpResult(graph, builder, op, MO::RandomNormalOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RandomUniformOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::RandomUniformOp op) {
  buildForOpResult(graph, builder, op, MO::RandomUniformOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RangeOp>::build(GraphOp &graph, OpBuilder &builder,
                                            MO::RangeOp op) {
  buildForOpResult(graph, builder, op, MO::RangeOp::getOperationName(),
                   op->getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0, 1, 2});
  return success();
}

// Reduction ops

#define BUILD_REDUCTION_OP(OP)                                                 \
  ErrorOrSuccess Fallback<OP>::build(GraphOp &graph, OpBuilder &builder,       \
                                     OP op) {                                  \
    return buildForReductionOp(graph, builder, op);                            \
  }

BUILD_REDUCTION_OP(MO::ArgMaxOp)
BUILD_REDUCTION_OP(MO::ArgMinOp)
BUILD_REDUCTION_OP(MO::ReduceMulOp)

#undef BUILD_REDUCTION_OP

ErrorOrSuccess Fallback<MO::ReshapeOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::ReshapeOp op) {
  buildForOpResult(graph, builder, op, MO::ReshapeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ResizeNearestOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ResizeNearestOp op) {
  buildForOpResult(graph, builder, op, MO::ResizeNearestOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ResizeBicubicOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ResizeBicubicOp op) {
  buildForOpResult(graph, builder, op, MO::ResizeBicubicOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ResizeLinearOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::ResizeLinearOp op) {
  buildForOpResult(graph, builder, op, MO::ResizeLinearOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::RoiAlignOp>::build(GraphOp &graph,
                                               OpBuilder &builder,
                                               MO::RoiAlignOp op) {
  buildForOpResult(graph, builder, op, MO::RoiAlignOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

// Unlike the scatter-like reduction variants, `mo.scatter`'s axis is a
// compile-time attribute threaded to the shape function.
ErrorOrSuccess Fallback<MO::ScatterOp>::build(GraphOp &graph,
                                              OpBuilder &builder,
                                              MO::ScatterOp op) {
  buildForOpResult(graph, builder, op, MO::ScatterOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{{"axis", op.getAxisAttr()}},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

// ScatterLike ops (axis is a scalar tensor operand)

#define BUILD_SCATTER_LIKE_OP(OP)                                              \
  ErrorOrSuccess Fallback<OP>::build(GraphOp &graph, OpBuilder &builder,       \
                                     OP op) {                                  \
    return buildForScatterLikeOp(graph, builder, op);                          \
  }

BUILD_SCATTER_LIKE_OP(MO::ScatterAddOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMaxOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMinOp)
BUILD_SCATTER_LIKE_OP(MO::ScatterMulOp)

#undef BUILD_SCATTER_LIKE_OP

// ScatterNdLike ops

#define BUILD_SCATTER_ND_LIKE_OP(OP)                                           \
  ErrorOrSuccess Fallback<OP>::build(GraphOp &graph, OpBuilder &builder,       \
                                     OP op) {                                  \
    return buildForScatterNdLikeOp(graph, builder, op);                        \
  }

BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdAddOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMaxOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMinOp)
BUILD_SCATTER_ND_LIKE_OP(MO::ScatterNdMulOp)

#undef BUILD_SCATTER_ND_LIKE_OP

ErrorOrSuccess Fallback<MO::SliceOp>::build(GraphOp &graph, OpBuilder &builder,
                                            MO::SliceOp op) {
  buildForOpResult(graph, builder, op, MO::SliceOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::SplitOp>::build(GraphOp &graph, OpBuilder &builder,
                                            MO::SplitOp op) {
  Location loc = op->getLoc();
  // `split_ith_output_shape` takes `split_axis` and `output_idx` as runtime
  // scalar operands, so materialize the axis attribute as a constant.
  SmallVector<Value> operands(op->getOperands());
  operands.push_back(
      MO::ConstantOp::create(builder, loc, op.getAxisAttr().getInt()));
  operands.push_back(nullptr); // placeholder for output_idx
  for (auto [idx, res] : llvm::enumerate(op.getResults())) {
    operands.back() =
        MO::ConstantOp::create(builder, loc, static_cast<int64_t>(idx));
    buildForOpResult(graph, builder, cast<TensorValue>(res),
                     "split_ith_output_shape", operands,
                     op.getOutputParamDecls(),
                     /*kgenParams=*/{},
                     /*dataDepInputIdxs=*/{1, 2, 3});
  }
  return success();
}

ErrorOrSuccess Fallback<MO::SqueezeShapeOp>::build(GraphOp &graph,
                                                   OpBuilder &builder,
                                                   MO::SqueezeShapeOp op) {
  buildForOpResult(graph, builder, op, MO::SqueezeShapeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::TopKOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::TopKOp op) {
  buildForOpResult(graph, builder, op.getValues(),
                   MO::TopKOp::getOperationName(), op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());

  // indices() guaranteed by verifier to have the same shape as values()
  return success();
}

ErrorOrSuccess Fallback<MO::TransposeOp>::build(GraphOp &graph,
                                                OpBuilder &builder,
                                                MO::TransposeOp op) {
  buildForOpResult(graph, builder, op, MO::TransposeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::UnsqueezeShapeOp>::build(GraphOp &graph,
                                                     OpBuilder &builder,
                                                     MO::UnsqueezeShapeOp op) {
  buildForOpResult(graph, builder, op, MO::UnsqueezeShapeOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ArgNonzeroOp>::build(GraphOp &graph,
                                                 OpBuilder &builder,
                                                 MO::ArgNonzeroOp op) {
  buildForOpResult(graph, builder, op, MO::ArgNonzeroOp::getOperationName(),
                   op->getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, /*dataDepInputIdxs=*/{0});
  return success();
}

ErrorOrSuccess Fallback<MO::TileOp>::build(GraphOp &graph, OpBuilder &builder,
                                           MO::TileOp op) {
  buildForOpResult(graph, builder, op, MO::TileOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{}, op.getDataDependentInputIndices());
  return success();
}

ErrorOrSuccess Fallback<MO::ReduceRMSNormOp>::build(GraphOp &graph,
                                                    OpBuilder &builder,
                                                    MO::ReduceRMSNormOp op) {
  buildForOpResult(graph, builder, op, MO::ReduceRMSNormOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {{"multiply_before_cast", op.getMultiplyBeforeCastAttr()}},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::Composite::ReduceRMSNormRoPEOp>::build(
    GraphOp &graph, OpBuilder &builder, MO::Composite::ReduceRMSNormRoPEOp op) {
  buildForOpResult(graph, builder, op,
                   MO::Composite::ReduceRMSNormRoPEOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/
                   {{"multiply_before_cast", op.getMultiplyBeforeCastAttr()}},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess
Fallback<MO::ReduceGroupNormOp>::build(GraphOp &graph, OpBuilder &builder,
                                       MO::ReduceGroupNormOp op) {
  buildForOpResult(graph, builder, op,
                   MO::ReduceGroupNormOp::getOperationName(), op.getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess
Fallback<MO::ReduceMinAndMaxOp>::build(GraphOp &graph, OpBuilder &builder,
                                       MO::ReduceMinAndMaxOp op) {
  // `axis` is a compile-time `index` attribute threaded to the shape function.
  buildForOpResult(graph, builder, op,
                   MO::ReduceMinAndMaxOp::getOperationName(), op->getOperands(),
                   op.getOutputParamDecls(),
                   /*kgenParams=*/{{"axis", op.getAxisAttr()}},
                   /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::Composite::ReduceRMSNormFusedResidualAddOp>::build(
    GraphOp &graph, OpBuilder &builder,
    MO::Composite::ReduceRMSNormFusedResidualAddOp op) {
  // Both output and intermediate have the same shape as input.
  buildForOpResult(
      graph, builder, op.getOutput(),
      MO::Composite::ReduceRMSNormFusedResidualAddOp::getOperationName(),
      op.getOperands(), op.getOutputParamDecls(),
      /*kgenParams=*/{},
      /*dataDepInputIdxs=*/{});
  return success();
}

ErrorOrSuccess Fallback<MO::Composite::RMSNormResidualAddOp>::build(
    GraphOp &graph, OpBuilder &builder,
    MO::Composite::RMSNormResidualAddOp op) {
  // Both output and intermediate have the same shape as input.
  buildForOpResult(graph, builder, op.getOutput(),
                   MO::Composite::RMSNormResidualAddOp::getOperationName(),
                   op.getOperands(), op.getOutputParamDecls(),
                   /*kgenParams=*/{},
                   /*dataDepInputIdxs=*/{});
  return success();
}

#undef HAS_FALLBACK
#undef NEEDS_NO_FALLBACK
#undef CALL_FALLBACK

} // namespace

} // namespace M::MO
