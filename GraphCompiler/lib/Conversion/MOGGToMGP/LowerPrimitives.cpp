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
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h"
#include "GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/IRMapping.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_LOWERPRIMITIVES
#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h.inc"

} // namespace M::MGP

//===----------------------------------------------------------------------===//
// MOGraphLowering implementation
//===----------------------------------------------------------------------===//

namespace M::MGP {

namespace {

using CompiledRegionBodyBuilder = std::function<ErrorOr<SmallVector<Value>>(
    OpBuilder &, Location, Operation *, ValueRange)>;
struct LowerPrimitivesPass
    : public MGP::impl::LowerPrimitivesBase<LowerPrimitivesPass> {
  using Base::Base;

  void runOnOperation() override;
};

// Registered POP, index and MGP primitives may fuse.
static inline bool canFuseOp(Operation *op) {
  if (isa<MGP::SyncOp>(op))
    return false;

  // A fused task starts once all of its inputs are ready, and a side-stream
  // join waits for the inputs of its consumer. Fused with the side-stream body,
  // it would hold back the body's launches until then.
  if (auto wait = dyn_cast<MGP::DeviceWaitOp>(op);
      wait && wait.getIsSideStreamJoin())
    return false;

  // Fusing buffer.constant.external would create a huge region.
  if (isa<MGP::BufferAllocOp, MGP::BufferConstantExternalOp>(*op))
    return false;

  return isa_and_nonnull<MGPDialect, KGEN::POP::POPDialect,
                         mlir::index::IndexDialect>(op->getDialect()) ||
         isa<KGEN::ParamConstantOp>(op);
}

static inline MOGG::DispatchKind getDispatch(Operation *op) {
  // Event sync should be launched in a separate thread.
  return isa<MGP::SyncOp>(op) ? MOGG::DispatchKind::Deferred
                              : MOGG::DispatchKind::Immediate;
}

static inline bool canDuplicate(Operation *op) {
  if (isa<MGP::BufferSliceOp, MGP::BufferConstantOp, MGP::TensorSpecGetDimOp,
          MGP::BufferConstantExternalOp, MGP::BufferGetCachedOp,
          MGP::TensorSpecCreateOp, MGP::TensorExtractBufferOp,
          MGP::TensorExtractTensorSpecOp>(op))
    return true;
  return isa<KGEN::KGENDialect, KGEN::POP::POPDialect,
             mlir::index::IndexDialect>(op->getDialect());
}

static ErrorOr<MOGG::AsyncRoutineOp>
wrapGenericRegionAround(OpBuilder &builder, Location loc, Operation *op,
                        CompiledRegionBodyBuilder bodyBuilder) {
  auto constResources = builder.getArrayAttr({});
  if (auto attrToValOp = dyn_cast<MGP::MGPSelectiveLowerAttrOpInterface>(op))
    constResources = builder.getArrayAttr(attrToValOp.getLiftableAttributes());

  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics(
      op->getNumOperands(), InOutSignatureAttr::InOutSemantics::kNone);
  auto newKernel = MOGG::AsyncRoutineOp::create(
      builder, loc, op->getResultTypes(), op->getOperands(),
      KGEN::ParamDeclArrayAttr::get(builder.getContext(), {}),
      InOutSignatureAttr::get(builder.getContext(), operandSemantics),
      MOGG::FusionInfoAttr::get(builder.getContext(),
                                MOGG::createFusionLeaf(op)),
      MOGG::DispatchKindAttr::get(builder.getContext(),
                                  getDispatch(op)), // dispatch
      constResources,                               // constantResources
      builder.getBoolAttr(canFuseOp(op)),           // canFuseIntoOther
      builder.getBoolAttr(canFuseOp(op)),           // canBeFusedInto
      builder.getBoolAttr(canDuplicate(op))         // canDuplicate
  );

  if (auto hasDeviceOp = dyn_cast<MGPHasDeviceOpInterface>(op))
    newKernel.setDeviceRef(hasDeviceOp.getDevice());
  else
    newKernel.setDeviceRef(MO::getDeviceOrDefaultAny(op));

  mlir::Block &kernelBody = newKernel.getBody().emplaceBlock();
  // Body block arguments are `!mogg.async<T>` wrappers around the operand
  // types. They are unpacked to `T` by `lowerMOGGABI` (called immediately after
  // this region is built), which inserts `mogg.async.unpack` and rewires uses.
  for (Value operand : newKernel->getOperands())
    kernelBody.addArgument(MOGG::AsyncType::get(operand.getType()),
                           operand.getLoc());

  builder.setInsertionPoint(&kernelBody, kernelBody.begin());
  ErrorOr<SmallVector<Value>> outputs =
      bodyBuilder(builder, loc, op, kernelBody.getArguments());
  if (outputs.isError())
    return outputs.takeError();
  SmallVector<Value> outs = outputs.takeValue();

  SmallVector<Value> returns;
  for (auto [returnVal, returnType] :
       llvm::zip(outs, newKernel.getResultTypes())) {
    if (returnVal.getType() != returnType &&
        isa<IndexType>(returnVal.getType())) {
      returnVal = mlir::index::CastSOp::create(builder, returnVal.getLoc(),
                                               returnType, returnVal);
    }
    returns.push_back(returnVal);
  }

  MOGG::OutputOp::create(builder, loc, returns);
  return newKernel;
}

void LowerPrimitivesPass::runOnOperation() {
  ModelOp model = getOperation();
  MLIRContext *ctx = model->getContext();

  auto cloneOperationBuilder =
      [](OpBuilder &builder, Location loc, Operation *op,
         ValueRange inputs) -> ErrorOr<SmallVector<Value>> {
    mlir::IRMapping mapping;
    for (auto [from, to] : llvm::zip_equal(op->getOperands(), inputs))
      mapping.map(from, to);
    Operation *newOp = builder.clone(*op, mapping);
    return newOp->getResults();
  };

  const auto walkerFn = [&](Operation *op) {
    if (isa<MOGG::AsyncRoutineOp, MOGG::FuncOp>(op))
      return mlir::WalkResult::skip();

    // Constant external ops should not be lowered to mojo so they can be
    // updated with a prefix during inlining.
    if (isa<MGP::BufferConstantExternalOp>(op))
      return mlir::WalkResult::advance();

    /// TODO(GEX-2588): Remove this.
    if (isa<MGP::BufferCacheOp, MGP::RuntimeCreateOp, MGP::ModuleLoadOp,
            MGP::ExecuteGenericOp>(op)) {
      return mlir::WalkResult::advance();
    }

    auto implementsEmitsMojoInterface = dyn_cast<MOGG::EmitsMojoInterface>(op);
    bool isAnIndexOrPOPOrParamConstantOp =
        isa_and_nonnull<mlir::index::IndexDialect, KGEN::POP::POPDialect>(
            op->getDialect()) ||
        isa<KGEN::ParamConstantOp>(op);
    if (!isAnIndexOrPOPOrParamConstantOp && !implementsEmitsMojoInterface)
      return mlir::WalkResult::advance();

    Location opLoc = op->getLoc();
    OpBuilder builder{op};

    auto newKernel =
        wrapGenericRegionAround(builder, opLoc, op, cloneOperationBuilder);
    if (newKernel.isError()) {
      op->emitError(newKernel.takeError().get());
      return mlir::WalkResult::interrupt();
    }

    MOGraphLoweringState::lowerMOGGABI(*newKernel);

    // Lower an index constant to a mgp.buffer.constant resource so kernels
    // are not hashed on the constant's value, improving compile time.
    if (isa<mlir::index::ConstantOp>(op)) {
      auto loweredIndexOp =
          *newKernel->getBodyBlock().getOps<mlir::index::ConstantOp>().begin();
      OpBuilder builder(loweredIndexOp);
      auto loc = loweredIndexOp.getLoc();
      uint64_t alignment =
          static_cast<uint64_t>(minimumCpuAlignment.getValue());

      int64_t indexValue = loweredIndexOp.getValue().getSExtValue();
      auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<int64_t>(
          {&indexValue, 1});
      auto elementsAttr =
          AlignedBytesAttr::get(ctx, blob.getDataAs<uint8_t>(), alignment);

      auto bufferConstantOp =
          MGP::BufferConstantOp::create(builder, loc, elementsAttr);
      auto dummyChain = MOGG::ChainDummyOp::create(builder, loc);
      auto newIndexOp = MGP::BufferToIndexOp::create(
          builder, loc, {dummyChain, bufferConstantOp.getResult()});
      loweredIndexOp->replaceAllUsesWith(newIndexOp);
      loweredIndexOp.erase();
      auto currentResources = newKernel->getConstantResources();
      SmallVector<Attribute> newResources(currentResources.begin(),
                                          currentResources.end());
      newResources.push_back(elementsAttr);
      newKernel->setConstantResourcesAttr(ArrayAttr::get(ctx, newResources));
    }

    op->replaceAllUsesWith(*newKernel);
    op->erase();

    return mlir::WalkResult::skip();
  };
  if (model.walk</*Order=*/mlir::WalkOrder::PreOrder>(walkerFn)
          .wasInterrupted()) {
    signalPassFailure();
  }
}

} // namespace

} // namespace M::MGP
