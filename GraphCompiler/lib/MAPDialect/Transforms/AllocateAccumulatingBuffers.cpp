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
// Gives every fused-output `map.iter.epilogue` a real store-target tensor via
// its `$intermediateTensor` operand, so `MAPToMOGG` never builds the
// kernel-facing `ManagedTensorSlice` over a null pointer.
//
// Normally the source is the tensor the epilogue stores into. A mixed-precision
// accumulating kernel (`containsMatmulOrConvOrConcat` whose epilogue changes
// dtype) must accumulate in the pre-cast dtype, so a fresh buffer of that dtype
// is added as an extra kernel result and used instead; the store target stays
// the real (post-cast) output.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"

#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_ALLOCATEACCUMULATINGBUFFERS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Matmul/conv/concat accumulate into their output, so they must accumulate in
// their own dtype even under a dtype-changing epilogue. TODO(#22349): remove
// once those respect the fusion contract.
static bool containsMatmulOrConvOrConcat(IterOpaqueOp op) {
  for (auto callOp : op.getBody().front().getOps<CallOpaqueOp>()) {
    StringRef opName = callOp.getKernelName();
    if (opName.contains(MO::ConcatOp::getOperationName()) ||
        opName.contains(MO::MatmulOp::getOperationName()) ||
        opName.contains(MO::BatchMatmulOp::getOperationName()) ||
        opName.contains(MO::ConvOp::getOperationName()) ||
        opName.contains(MO::ConvTransposeOp::getOperationName()))
      return true;
  }
  return false;
}

// The outer tensor the kernel's own output slice aliases. Any tensor the
// epilogue stores directly at its index works: the epilogue overwrites every
// element the kernel writes there. A store nested in a select writes only part
// of a smaller tensor, so it can't be the target. A tensor of the value's dtype
// is preferred, since any other one needs an accumulation buffer.
static Value findStoreTarget(IterEpilogueOp epi) {
  Block &body = epi.getBody().front();
  Type valueType = body.getArgument(0).getType();
  auto closure = cast<CapturingFusionClosure>(epi.getOperation());
  Value fallback;
  for (StoreOp store : body.getOps<StoreOp>()) {
    auto targetArg = dyn_cast<BlockArgument>(store.getTensor());
    Value target = targetArg ? closure.getCaptureFor(targetArg) : Value();
    if (!target)
      continue;
    auto targetType = dyn_cast<MO::TensorType>(target.getType());
    if (targetType && targetType.getElementType() == valueType)
      return target;
    if (!fallback)
      fallback = target;
  }
  return fallback;
}

// Replaces `op` with a copy that has one extra non-chain result of
// `scratchType` (before any trailing chain) and a matching trailing output
// block argument, returned in `scratchArg`. `map.iter.opaque` cannot grow
// results in place.
static IterOpaqueOp addScratchResult(IterOpaqueOp op, Type scratchType,
                                     BlockArgument &scratchArg) {
  OpBuilder b(op);
  Value chain = op.getOutputChain();
  SmallVector<Type> resultTypes;
  for (Value r : op.getResults())
    if (r != chain)
      resultTypes.push_back(r.getType());
  resultTypes.push_back(scratchType);
  if (chain)
    resultTypes.push_back(chain.getType());

  auto newOp = IterOpaqueOp::create(b, op.getLoc(), resultTypes, op.getInputs(),
                                    op.getInputChain(),
                                    /*fusionInfo=*/Attribute());
  newOp.setDeviceRef(MO::getDeviceOrDefaultAny(op));
  if (Attribute fusionInfo = op.getFusionInfoAttr())
    newOp.setFusionInfoAttr(fusionInfo);
  newOp.setHasReadEffect(op.getHasReadEffect());
  newOp.setHasWriteEffect(op.getHasWriteEffect());
  newOp.setDispatch(op.getDispatch());

  newOp.getBody().takeBody(op.getBody());
  scratchArg = newOp.getBody().front().addArgument(scratchType, op.getLoc());

  unsigned newIdx = 0;
  for (Value r : op.getResults()) {
    if (r == chain)
      continue;
    r.replaceAllUsesWith(newOp.getResult(newIdx++));
  }
  if (chain)
    chain.replaceAllUsesWith(newOp.getResults().back());
  op.erase();
  return newOp;
}

struct AllocateAccumulatingBuffersPass
    : public impl::AllocateAccumulatingBuffersBase<
          AllocateAccumulatingBuffersPass> {
  using Base::Base;

  // Fails only on an unconvertible scratch dtype. The mixed-precision path
  // replaces the kernel, so it recurses into the replacement.
  LogicalResult assignIntermediateTensors(IterOpaqueOp kernel) {
    for (IterEpilogueOp epi :
         kernel.getBody().front().getOps<IterEpilogueOp>()) {
      if (epi.getIntermediateTensor())
        continue;
      Value storeTarget = findStoreTarget(epi);
      if (!storeTarget)
        continue;

      Type inElemType = epi.getBody().front().getArgument(0).getType();
      auto resultTy = dyn_cast<MO::TensorType>(storeTarget.getType());
      bool dtypeChanges = resultTy && inElemType != resultTy.getElementType();

      if (dtypeChanges && containsMatmulOrConvOrConcat(kernel)) {
        ErrorOr<DType> inDType = MO::fromMLIRType(inElemType);
        if (inDType.isError())
          return epi.emitOpError("AllocateAccumulatingBuffers: ")
                 << inDType.getError();
        Type scratchType = resultTy.cloneWith(*inDType);
        BlockArgument scratchArg;
        IterOpaqueOp newKernel =
            addScratchResult(kernel, scratchType, scratchArg);
        epi.getIntermediateTensorMutable().assign(scratchArg);
        return assignIntermediateTensors(newKernel);
      }

      epi.getIntermediateTensorMutable().assign(storeTarget);
    }
    return success();
  }

  void runOnOperation() override {
    // Walk kernels, not epilogues: the mixed-precision path replaces the
    // kernel, and erasing the visited op is walk-safe while erasing an ancestor
    // is not. `skip()` keeps the walk out of the possibly replaced body.
    WalkResult result = getOperation().walk<WalkOrder::PreOrder>(
        [&](IterOpaqueOp kernel) -> WalkResult {
          if (failed(assignIntermediateTensors(kernel)))
            return WalkResult::interrupt();
          return WalkResult::skip();
        });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MAP
