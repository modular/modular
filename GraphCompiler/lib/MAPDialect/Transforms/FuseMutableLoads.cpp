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

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/Support/DataflowUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <optional>

using namespace M;

#define DEBUG_TYPE "map-fuse-mutable-loads"

namespace M::MAP {

#define GEN_PASS_DEF_FUSEMUTABLELOADS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

// Only these kernels take a fused mutable load.
bool isFusibleKernel(Operation *op) {
  return isa_and_present<MAP::IterElementwiseOp, MAP::IterOpaqueOp>(op);
}

/// The kernel to fuse `loadOp` into, or null. The kernel must be the sole
/// consumer of the load's tensor.
Operation *maybeGetCompatibleKernel(MO::MutableLoadOp loadOp) {

  auto tensor = loadOp.getOutTensor();

  if (!tensor.hasOneUse()) {
    LLVM_DEBUG(loadOp.emitRemark()
               << "Mutable load fusion is disabled for this load op as its "
                  "output tensor has more than one use.");

    return nullptr;
  }

  if (loadOp->getBlock() != (*tensor.getUsers().begin())->getBlock()) {
    LLVM_DEBUG(
        loadOp.emitRemark()
        << "Mutable load fusion is disabled for this load op as its "
           "output tensor is produced in a different block than its user.");
    return nullptr;
  }

  Operation *kernelOp = *tensor.getUsers().begin();
  if (!isFusibleKernel(kernelOp))
    return nullptr;

  // Reading the buffer in place of the load's snapshot is only sound if the
  // kernel writes only the element it reads, as map.iter.elementwise does. An
  // in-place map.iter.opaque may write any element.
  if (isa<MAP::IterOpaqueOp>(kernelOp) &&
      llvm::is_contained(kernelOp->getOperands(), loadOp.getInBuffer())) {
    LLVM_DEBUG(loadOp.emitRemark()
               << "Mutable load fusion is disabled for this load op as the "
                  "opaque kernel consuming it also writes its buffer.");
    return nullptr;
  }

  return kernelOp;
}

IterElementwiseOp fuseLoadBufferAndChains(PatternRewriter &rewriter,
                                          MO::MutableLoadOp loadOp,
                                          IterElementwiseOp kernelOp) {
  MLIRContext *ctx = rewriter.getContext();

  auto chainType = MO::ChainType::get(ctx);

  auto &tensorUse = *loadOp.getOutTensor().getUses().begin();
  auto argIdx = tensorUse.getOperandNumber();

  SmallVector<Value> newInputs(kernelOp.getInputs());
  newInputs[argIdx] = loadOp.getInBuffer();

  SmallVector<Type> newResultTypes(TypeRange(kernelOp.getResults()));
  newResultTypes.push_back(chainType);

  rewriter.setInsertionPointAfter(kernelOp);

  auto newKernelOp = IterElementwiseOp::create(
      rewriter, kernelOp->getLoc(),
      /*resultTypes=*/newResultTypes,
      /*shape=*/kernelOp.getShape(),
      /*inputs=*/newInputs,
      /*chain=*/loadOp.getInChain(),
      /*fusionInfo=*/
      MOGG::createFusionInfo<MOGG::FusionMutableLoadAttr>(
          kernelOp->getContext(), {loadOp, kernelOp}),
      /*hasReadEffect=*/true,
      /*hasWriteEffect=*/kernelOp.getHasWriteEffect());

  // Inherent, so `setDiscardableAttrs` below does not carry it.
  newKernelOp.setDispatch(kernelOp.getDispatch());

  newKernelOp->setLoc(mlir::FusedLoc::get(
      kernelOp->getContext(), {loadOp->getLoc(), kernelOp->getLoc()}));
  newKernelOp->setDiscardableAttrs(kernelOp->getDiscardableAttrDictionary());

  // Build the new block with the buffer type at argIdx, then splice the old
  // body in.
  Block &oldBody = kernelOp.getBody().front();
  Block *newBody = &newKernelOp.getBody().emplaceBlock();
  for (auto [i, arg] : llvm::enumerate(oldBody.getArguments())) {
    Type argType = i == argIdx ? loadOp.getInBuffer().getType() : arg.getType();
    newBody->addArgument(argType, arg.getLoc());
  }

  rewriter.cloneRegionBefore(kernelOp.getBody(), newKernelOp.getBody(),
                             newKernelOp.getBody().begin());
  Block &oldBodyClone = newKernelOp.getBody().front();

  SmallVector<Value> fusedArgs(newBody->getArguments().begin(),
                               newBody->getArguments().end());
  rewriter.inlineBlockBefore(&oldBodyClone, newBody, newBody->begin(),
                             fusedArgs);

  auto outChainIdx = newKernelOp.getNumResults() - 1;
  Value outChain = newKernelOp->getResult(outChainIdx);

  loadOp.getOutChain().replaceAllUsesWith(outChain);

  for (auto &&[oldRes, newRes] :
       llvm::zip(kernelOp->getResults(), newKernelOp->getResults())) {
    oldRes.replaceAllUsesWith(newRes);
  }

  kernelOp->erase();
  loadOp->erase();

  return newKernelOp;
}

// When the kernel depends on the load's out-chain, fusing would create a cycle.
// If the path is an exclusive run of single-use `mo.chain.create`s, rewrite
// them to consume the load's in-chain instead; otherwise give up.
//
// BEFORE (fusion blocked):
//   %t, %outCh = mo.mutable.load(buf, %inCh)
//   %merged = mo.chain.create(%other, %outCh)  // kernel depends on %outCh
//   %result = mo.custom(..., %t, %merged)      // can't fuse: creates cycle
//
// AFTER DETACHMENT (fusion enabled):
//   %t, %outCh = mo.mutable.load(buf, %inCh)   // %outCh becomes dead
//   %merged = mo.chain.create(%other, %inCh)   // %outCh -> %inCh
//   %result = map.iter.elementwise(buf, ..., %merged)  // fused!
//     fusionInfo = MutableLoad(...)
//
// Requires: %outCh and %merged have single use (exclusivity constraint).
static bool detachExclusiveChainPath(Value loadOutChain, Value loadInChain,
                                     Value kernelInChain) {
  if (kernelInChain == loadOutChain) {
    return false;
  }

  SmallVector<MO::ChainCreateOp> toRewrite;
  llvm::SmallPtrSet<Operation *, 16> onPath;

  // Returns:
  //   * std::nullopt - this branch never reaches the load's out-chain.
  //   * false        - the branch touches the load but fails exclusivity checks
  //                    (non-chain.create op or extra users).
  //   * true         - the branch is exclusively composed of single-use
  //                    chain.create ops and is safe to rewrite.
  std::function<std::optional<bool>(Value)> analyzeChainValue =
      [&](Value v) -> std::optional<bool> {
    if (v == loadOutChain)
      return true;

    auto chainCreate = v.getDefiningOp<MO::ChainCreateOp>();
    if (!chainCreate)
      return std::nullopt;

    bool subtreeReachesLoad = false;
    for (Value operand : chainCreate.getInputs()) {
      if (operand == loadOutChain) {
        subtreeReachesLoad = true;
        continue;
      }

      if (std::optional<bool> pathStatus = analyzeChainValue(operand)) {
        if (!*pathStatus) {
          return false;
        }
        subtreeReachesLoad = true;
      }
    }

    if (!subtreeReachesLoad) {
      return std::nullopt;
    }

    // Another user would observe the rewiring.
    if (!chainCreate.getResult().hasOneUse()) {
      return false;
    }

    if (onPath.insert(chainCreate.getOperation()).second)
      toRewrite.push_back(chainCreate);

    return true;
  };

  if (analyzeChainValue(kernelInChain) != true)
    return false;

  for (MO::ChainCreateOp chainCreate : toRewrite)
    chainCreate->replaceUsesOfWith(loadOutChain, loadInChain);

  ASSERT_STREAM(!dataflowGraphDependsOn(kernelInChain, loadOutChain),
                "Expected exclusive chain path from mutable.load to kernel op "
                "to be severed.");

  return true;
}

// A prologue capture's block argument mirrors the captured type, so it follows
// the tensor -> buffer retype.
void retypeFusedPrologueCaptures(Value bufferArg) {
  for (OpOperand &use : bufferArg.getUses()) {
    auto prologueOp = dyn_cast<MAP::IterPrologueOp>(use.getOwner());
    if (!prologueOp)
      continue;

    // Block args are [index, captures...].
    prologueOp.getBody()
        .getArgument(use.getOperandNumber() + 1)
        .setType(bufferArg.getType());
  }
}

void fuseLoadBuffer(PatternRewriter &rewriter, MO::MutableLoadOp loadOp,
                    Operation *kernelOp, bool kernelHasLoadsOutputChain) {
  auto buffer = loadOp.getInBuffer();
  auto tensor = loadOp.getOutTensor();

  ASSERT_STREAM(*tensor.getUsers().begin() == kernelOp,
                "Expected value semantic tensor from load op to only be "
                "consumed by current kernel op");

  OpOperand &operand = *tensor.getUses().begin();
  auto argIdx = operand.getOperandNumber();

  auto kernel = cast<IterOpInterface>(kernelOp);
  Block &kernelBody = kernelOp->getRegion(0).front();

  TypedValue<MO::ChainType> loadOutChain = loadOp.getOutChain();
  if (Value kernelInChain = kernel.getInputChain()) {
    if (kernelInChain != loadOutChain &&
        dataflowGraphDependsOn(kernelInChain, loadOutChain) &&
        !detachExclusiveChainPath(loadOutChain, loadOp.getInChain(),
                                  kernelInChain)) {
      LLVM_DEBUG(loadOp.emitRemark() << "Skipping mutable load fusion to avoid "
                                        "introducing a chain cycle.");
      return;
    }
  }

  kernel.insertInputsAt(argIdx, buffer);
  Value newBufferArg = kernelBody.getArgument(argIdx);

  auto oldBlockArg = kernelBody.getArgument(argIdx + 1);
  oldBlockArg.replaceAllUsesWith(newBufferArg);
  kernel.eraseInputsAt(argIdx + 1, 1);

  retypeFusedPrologueCaptures(newBufferArg);

  if (!kernelHasLoadsOutputChain) {
    loadOutChain.replaceAllUsesWith(kernel.getOutputChain());

    rewriter.setInsertionPoint(kernelOp);

    auto newChain = rewriter.createOrFold<MO::ChainCreateOp>(
        loadOp.getLoc(),
        ValueRange{kernel.getInputChain(), loadOp.getInChain()});
    kernel.setChain(newChain);
  } else {
    kernel.setChain(loadOp.getInChain());
  }

  kernel.setFusionInfoAttr(MOGG::createFusionInfo<MOGG::FusionMutableLoadAttr>(
      kernelOp->getContext(), {loadOp, kernelOp}));
  kernelOp->setLoc(mlir::FusedLoc::get(kernelOp->getContext(),
                                       {loadOp->getLoc(), kernelOp->getLoc()}));

  loadOp.erase();
}

// Fuses `loadOp` into a read-only opaque `kernelOp` with no chain, giving it a
// chain in/out pair (GEX-3962). Otherwise the load stays a full-buffer copy,
// e.g. a gather indexing a KV leaf materializes the whole leaf per read.
// `map.iter.opaque` cannot grow results in place, so the op is replaced.
IterOpaqueOp fuseLoadBufferAndChains(PatternRewriter &rewriter,
                                     MO::MutableLoadOp loadOp,
                                     IterOpaqueOp kernelOp) {
  MLIRContext *ctx = rewriter.getContext();
  auto chainType = MO::ChainType::get(ctx);

  OpOperand &tensorUse = *loadOp.getOutTensor().getUses().begin();
  auto argIdx = tensorUse.getOperandNumber();

  SmallVector<Value> newInputs(kernelOp.getInputs());
  newInputs[argIdx] = loadOp.getInBuffer();

  SmallVector<Type> newResultTypes(TypeRange(kernelOp.getResults()));
  newResultTypes.push_back(chainType);

  rewriter.setInsertionPointAfter(kernelOp);
  auto newKernelOp = IterOpaqueOp::create(rewriter, kernelOp.getLoc(),
                                          newResultTypes, newInputs,
                                          /*inputChain=*/loadOp.getInChain(),
                                          /*fusionInfo=*/Attribute());
  newKernelOp.setDeviceRef(MO::getDeviceOrDefaultAny(kernelOp));
  newKernelOp.setHasReadEffect(true);
  newKernelOp.setHasWriteEffect(kernelOp.getHasWriteEffect());
  // Inherent, so `setDiscardableAttrs` below does not carry it.
  newKernelOp.setDispatch(kernelOp.getDispatch());

  newKernelOp->setLoc(
      mlir::FusedLoc::get(ctx, {loadOp->getLoc(), kernelOp->getLoc()}));
  newKernelOp->setDiscardableAttrs(kernelOp->getDiscardableAttrDictionary());

  // The chain has no block argument, so only the fused input is retyped.
  newKernelOp.getBody().takeBody(kernelOp.getBody());
  BlockArgument fusedArg = newKernelOp.getBody().front().getArgument(argIdx);
  fusedArg.setType(loadOp.getInBuffer().getType());
  retypeFusedPrologueCaptures(fusedArg);

  loadOp.getOutChain().replaceAllUsesWith(newKernelOp.getOutputChain());

  for (auto &&[oldRes, newRes] :
       llvm::zip(kernelOp->getResults(), newKernelOp->getResults()))
    oldRes.replaceAllUsesWith(newRes);

  newKernelOp.setFusionInfoAttr(
      MOGG::createFusionInfo<MOGG::FusionMutableLoadAttr>(ctx,
                                                          {loadOp, kernelOp}));

  kernelOp->erase();
  loadOp->erase();

  return newKernelOp;
}

/// Turns a mo.mutable.load op with n uses into n mo.mutable.load ops each with
/// a single use of their value semantic tensor output.
struct ParallelizeLoadsWithMultipleUsers
    : MO::DeviceAwareRewritePattern<OpRewritePattern, MO::MutableLoadOp> {
  ParallelizeLoadsWithMultipleUsers(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MutableLoadOp loadOp,
                             MO::DeviceAwareRewriter &rewriter) const override {
    auto tensor = loadOp.getOutTensor();
    auto loadBuffer = loadOp.getInBuffer();

    if (tensor.use_empty() || tensor.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          loadOp.getLoc(),
          "Output value semantic tensor has <= 1 uses, skipping "
          "parallelization pattern.");
    }

    // After splitting, a consumer that writes the buffer would race the other
    // consumers' fused reads.
    for (OpOperand &use : tensor.getUses()) {
      Operation *consumer = use.getOwner();
      auto kernel = dyn_cast<IterOpInterface>(consumer);
      if (!kernel || !kernel.getHasWriteEffect())
        continue;

      if (llvm::is_contained(consumer->getOperands(), loadBuffer)) {
        return rewriter.notifyMatchFailure(
            loadOp.getLoc(),
            "Parallelization blocked: a consumer kernel has write effect and "
            "shares the same buffer as the load operation.");
      }
    }

    auto *loadBlock = loadOp->getBlock();

    for (OpOperand &use : tensor.getUses()) {
      if (use.getOwner()->getBlock() != loadBlock) {
        // TODO(GEX-1203): Add support for cross-block load parallelization
        return rewriter.notifyMatchFailure(
            loadOp.getLoc(),
            "Found use of output tensor in different block than load op, "
            "skipping parallelization to maintain cross-block semantics");
      }
    }

    auto chainType = MO::ChainType::get(loadOp.getContext());
    auto outTensor = loadOp.getOutTensor();

    llvm::SmallVector<Value> chains;
    llvm::SmallVector<OpOperand *> opOperands;
    for (OpOperand &use : outTensor.getUses())
      opOperands.push_back(&use);

    for (OpOperand *use : opOperands) {
      auto newLoadOp = rewriter.create<MO::MutableLoadOp>(
          loadOp.getLoc(), outTensor.getType(), chainType, loadOp.getInBuffer(),
          loadOp.getInChain());

      rewriter.modifyOpInPlace(use->getOwner(), [&newLoadOp, &use]() {
        use->set(newLoadOp.getOutTensor());
      });

      chains.push_back(newLoadOp.getOutChain());
    }

    // Want the generated mo.chain.create to show up in a semi-reasonable spot.
    for (Operation *user : loadOp.getOutChain().getUsers()) {
      rewriter.setInsertionPoint(user);
      break;
    }

    Value newChain =
        rewriter.create<MO::ChainCreateOp>(loadOp.getLoc(), ValueRange{chains});

    rewriter.replaceAllUsesWith(loadOp.getOutChain(), newChain);

    loadOp.erase();

    return success();
  }
};

bool kernelAlreadyHasDirectBufferInput(Operation *kernelOp,
                                       MO::MutableLoadOp loadOp) {
  if (!loadOp.getOutChain().hasOneUse())
    return false;

  auto *user = *loadOp.getOutChain().getUsers().begin();
  if (user != kernelOp)
    return false;

  OpOperand *tensorOperand = nullptr;
  for (OpOperand &opOperand : kernelOp->getOpOperands()) {
    if (opOperand.get() == loadOp.getOutTensor()) {
      tensorOperand = &opOperand;
      break;
    }
  }

  if (!tensorOperand)
    return false;

  unsigned tensorIdx = tensorOperand->getOperandNumber();
  for (unsigned i = 0; i < kernelOp->getNumOperands(); ++i) {
    if (kernelOp->getOperand(i) == loadOp.getInBuffer() && i != tensorIdx)
      return true;
  }

  return false;
}

bool kernelHasInOutChains(Operation *kernelOp) {
  auto kernel = cast<IterOpInterface>(kernelOp);
  return kernel.getInputChain() && kernel.getOutputChain();
}

struct FuseMutableLoads : public impl::FuseMutableLoadsBase<FuseMutableLoads> {
  using FuseMutableLoadsBase::FuseMutableLoadsBase;
  void runOnOperation() override {
    MO::GraphOp moGraph = getOperation();
    MLIRContext *ctx = moGraph->getContext();
    PatternRewriter rewriter{ctx};

    SmallVector<Operation *> opsToProcess;
    moGraph.walk(
        [&](MO::MutableLoadOp loadOp) { opsToProcess.push_back(loadOp); });

    RewritePatternSet patterns{ctx};
    patterns.add<ParallelizeLoadsWithMultipleUsers>(ctx);

    if (failed(
            mlir::applyOpPatternsGreedily(opsToProcess, std::move(patterns))))
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "Graph after load parallelization:\n";
               moGraph.print(llvm::dbgs()); llvm::dbgs() << '\n');

    SmallVector<MO::MutableLoadOp> loadOps;
    moGraph.walk([&](MO::MutableLoadOp loadOp) { loadOps.push_back(loadOp); });

    for (auto loadOp : loadOps) {
      Operation *kernelOp = maybeGetCompatibleKernel(loadOp);
      if (!kernelOp)
        continue;

      if (kernelAlreadyHasDirectBufferInput(kernelOp, loadOp)) {
        // TODO: Unify all these cases
        fuseLoadBuffer(rewriter, loadOp, kernelOp,
                       /*kernelHasLoadsOutputChain=*/true);
      } else if (!kernelHasInOutChains(kernelOp)) {
        // The kernel must now both consume and produce a chain.
        //
        // TODO(GEX-1031): Split into adding the chain, then the
        // mo.tensor -> mo.buffer rewrite.
        if (auto foreachOp = dyn_cast<MAP::IterElementwiseOp>(kernelOp))
          fuseLoadBufferAndChains(rewriter, loadOp, foreachOp);
        else
          fuseLoadBufferAndChains(rewriter, loadOp,
                                  cast<MAP::IterOpaqueOp>(kernelOp));
      } else {
        fuseLoadBuffer(rewriter, loadOp, kernelOp,
                       /*kernelHasLoadsOutputChain=*/false);
      }
    }
  }
};

} // namespace M::MAP
