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
// This file implements folding and canonicalization for the MOGG
// dialect operations.
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

// Combine two in/out semantics by unioning their read and write capabilities.
// kNone='.', kIn='i', kOut='o', kMut='m' (see Support/.../MAttrs.td). When the
// same value is passed as multiple operands, the surviving operand must reflect
// every access the body performs through any of the duplicates.
static InOutSignatureAttr::InOutSemantics
unionInOutSemantics(InOutSignatureAttr::InOutSemantics a,
                    InOutSignatureAttr::InOutSemantics b) {
  using Sem = InOutSignatureAttr::InOutSemantics;
  auto reads = [](Sem s) { return s == Sem::kIn || s == Sem::kMut; };
  auto writes = [](Sem s) { return s == Sem::kOut || s == Sem::kMut; };
  bool r = reads(a) || reads(b);
  bool w = writes(a) || writes(b);
  if (r && w)
    return Sem::kMut;
  if (r)
    return Sem::kIn;
  if (w)
    return Sem::kOut;
  return Sem::kNone;
}

// Canonicalize duplicated operands of a generic compiled region:
//
//     mogg.routine.async(%a, %b, %a) : (%x, %y, %z) { ...use %z... }
//
// to
//
//     mogg.routine.async(%a, %b) : (%x, %y) { ...use %x... }
//
// The block argument for each duplicate is redirected to the first
// occurrence's argument, the duplicate operand/argument are erased, and the
// surviving operand's in/out semantics absorb the union of the duplicates'.
static LogicalResult deduplicateOperands(AsyncRoutineOp op,
                                         PatternRewriter &rewriter) {
  Block &entry = op.getBody().front();

  // Map each distinct operand value to the index of its first occurrence and
  // mark later duplicates for removal.
  llvm::SmallDenseMap<Value, unsigned> firstIndex;
  llvm::BitVector argsToRemove(op.getNumOperands());
  for (OpOperand &operand : op.getOperandsMutable()) {
    unsigned idx = operand.getOperandNumber();
    if (!firstIndex.try_emplace(operand.get(), idx).second)
      argsToRemove.set(idx);
  }

  if (argsToRemove.none())
    return rewriter.notifyMatchFailure(op, "no duplicate operands");

  // Redirect each duplicate block argument to its first occurrence so the body
  // reads/writes the single surviving argument.
  for (unsigned idx : argsToRemove.set_bits()) {
    unsigned keptIdx = firstIndex[op.getOperand(idx)];
    rewriter.replaceAllUsesWith(entry.getArgument(idx),
                                entry.getArgument(keptIdx));
  }

  // Fold the removed entries' semantics into their first occurrence, then drop
  // the duplicate entries. Empty signatures (used by some unittests) are left
  // untouched.
  InOutSignatureAttr oldSemantics = op.getInOutSignature();
  InOutSignatureAttr newSemantics;
  if (oldSemantics.size() != 0) {
    SmallVector<InOutSignatureAttr::InOutSemantics> merged(oldSemantics.size());
    for (size_t i = 0, e = oldSemantics.size(); i < e; ++i)
      merged[i] = oldSemantics[i];
    for (unsigned idx : argsToRemove.set_bits()) {
      unsigned keptIdx = firstIndex[op.getOperand(idx)];
      merged[keptIdx] = unionInOutSemantics(merged[keptIdx], oldSemantics[idx]);
    }
    SmallVector<InOutSignatureAttr::InOutSemantics> kept;
    for (size_t i = 0, e = oldSemantics.size(); i < e; ++i)
      if (!argsToRemove.test(i))
        kept.push_back(merged[i]);
    newSemantics = InOutSignatureAttr::get(op.getContext(), kept);
  }

  rewriter.modifyOpInPlace(op, [&]() {
    op->eraseOperands(argsToRemove);
    entry.eraseArguments(argsToRemove);
    if (newSemantics)
      op.setInOutSignatureAttr(newSemantics);
  });

  return success();
}

// Canonicalize sequences of grt.chain.create + mogg.routine.async
//
//     %ch = grt.chain.create(%0, %1)
//     mogg.routine.async(%ch)
//
// to
//
//     mogg.routine.async(%0, %1)
static LogicalResult sinkChainCreateOperands(AsyncRoutineOp op,
                                             PatternRewriter &rewriter) {
  Region &body = op.getBody();

  llvm::BitVector blockArgsToRemove(op.getNumOperands());
  SmallVector<Value> operandsToKeep;
  SmallVector<Value> newOperands;

  for (OpOperand &operand : op.getOperandsMutable()) {
    BlockArgument bodyArgument = body.getArgument(operand.getOperandNumber());

    auto chainCreateOp = operand.get().getDefiningOp<GRT::ChainCreateOp>();
    // Only remove grt.chain.create operands whose corresponding body
    // argument is unused.
    if (!chainCreateOp || !chainCreateOp.getResult().hasOneUse() ||
        !bodyArgument.getUses().empty()) {
      operandsToKeep.push_back(operand.get());
      continue;
    }

    blockArgsToRemove.set(bodyArgument.getArgNumber());
    newOperands.append(chainCreateOp.getOperands().begin(),
                       chainCreateOp.getOperands().end());
  }

  if (blockArgsToRemove.none())
    return failure();

  // Body block arguments are `!mogg.async<T>` wrappers around the operand
  // types.
  SmallVector<Type> newTypes = llvm::map_to_vector(
      newOperands, [](Value v) -> Type { return AsyncType::get(v.getType()); });

  SmallVector<Location> newLocs =
      llvm::map_to_vector(newOperands, [](Value v) { return v.getLoc(); });

  rewriter.modifyOpInPlace(op, [&]() {
    // Update the operands to [operandsToKeep; newOperands]
    op.getOperandsMutable().clear();
    op.getOperandsMutable().append(operandsToKeep);
    op.getOperandsMutable().append(newOperands);

    // Erase the block args of the removed chain operands.
    Block &entryBlock = *op.getBody().begin();
    entryBlock.eraseArguments(blockArgsToRemove);
    entryBlock.addArguments(newTypes, newLocs);

    if (op.getInOutSignature().size() != 0) {
      // Update the InOutSignatureAttr by removing the entries corresponding
      // to removed arguments and appending a `kNone` for each new argument.
      SmallVector<InOutSignatureAttr::InOutSemantics> newSemantics(
          newOperands.size(), InOutSignatureAttr::InOutSemantics::kNone);

      InOutSignatureAttr oldSemantics = op.getInOutSignature();
      op.setInOutSignatureAttr(
          oldSemantics.remove(blockArgsToRemove).append(newSemantics));
    }
  });

  return success();
}

// Canonicalize redundant outputs of a generic compiled region by rebuilding it
// with a collapsed result list:
//   - drop non-chain results that have no uses, and
//   - collapse duplicate `!mef.chain` outputs into the last chain output, which
//     is always retained so error/side-effect propagation and async-completion
//     sequencing are preserved.
// Uses of removed chain results are rewired to the surviving chain.
static LogicalResult removeRedundantResults(AsyncRoutineOp op,
                                            PatternRewriter &rewriter) {
  OutputOp outputOp = op.getBodyTerminator();

  // Candidates for removal: every chain output (duplicate chains are collapsed
  // into one), plus any non-chain output whose corresponding region result is
  // unused.
  llvm::BitVector toRemoveBits(outputOp->getNumOperands());

  for (OpOperand &operand : outputOp->getOpOperands()) {
    unsigned idx = operand.getOperandNumber();
    bool isChain = isa<MEF::ChainType>(operand.get().getType());
    if (!isChain && !op->getResult(idx).use_empty())
      continue;
    toRemoveBits.set(idx);
  }

  if (toRemoveBits.none())
    return rewriter.notifyMatchFailure(op, "no removable outputs");

  // The last chain output is always kept; removed chains are rewired to it.
  std::optional<unsigned> keptChainNewIdx;
  for (OpOperand &operand : llvm::reverse(outputOp->getOpOperands())) {
    if (isa<MEF::ChainType>(operand.get().getType())) {
      unsigned origIdx = operand.getOperandNumber();
      toRemoveBits.reset(origIdx);
      // Position of the kept chain within the collapsed result list.
      unsigned newIdx = 0;
      for (unsigned i = 0; i < origIdx; ++i)
        if (!toRemoveBits.test(i))
          ++newIdx;
      keptChainNewIdx = newIdx;
      break;
    }
  }

  // Nothing to remove, or removing everything would leave no result and no
  // side-effect chain: leave the op unchanged rather than asserting.
  if (toRemoveBits.none() || toRemoveBits.all())
    return rewriter.notifyMatchFailure(op, "no redundant outputs to remove");

  // Collapse the terminator operand list, then rebuild the op with the
  // corresponding (unwrapped) result types.
  rewriter.modifyOpInPlace(outputOp,
                           [&]() { outputOp->eraseOperands(toRemoveBits); });

  SmallVector<Type> newResultTypes =
      llvm::map_to_vector(outputOp->getOperandTypes(), [](Type type) -> Type {
        if (auto asyncType = dyn_cast<AsyncType>(type))
          return asyncType.getElementType();
        return type;
      });

  // Rebuild the op with the collapsed result types, carrying over all of its
  // attributes (inherent + discardable) so nothing relied on later in the
  // pipeline (e.g. isStructured, emitsMojo) is dropped.
  rewriter.setInsertionPoint(op);
  auto newKernel = AsyncRoutineOp::create(
      rewriter, op->getLoc(), newResultTypes, op.getOperands(),
      op.getOutputParamDeclsAttr(), op.getInOutSignatureAttr(),
      op.getFusionInfoAttr(),        // fusionInfo
      op.getDispatchAttr(),          // dispatch
      op.getConstantResourcesAttr(), // constantResources
      op.getCanFuseIntoOtherAttr(),  // canFuseIntoOther
      op.getCanBeFusedIntoAttr(),    // canBeFusedInto
      op.getCanDuplicateAttr(),      // canDuplicate
      op.getIsStructuredAttr()       // isStructured
  );

  newKernel.setDeviceRef(op.getExecutionDevice());
  newKernel->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  rewriter.inlineRegionBefore(op.getBody(), newKernel.getBody(),
                              newKernel.getBody().end());

  // Rewire uses: kept results map positionally; a removed chain redirects to
  // the surviving chain; removed unused non-chain results have no uses.
  unsigned curIdx = 0;
  for (OpResult result : op->getOpResults()) {
    if (toRemoveBits.test(result.getResultNumber())) {
      if (keptChainNewIdx && isa<MEF::ChainType>(result.getType()))
        rewriter.replaceAllUsesWith(result,
                                    newKernel->getResult(*keptChainNewIdx));
    } else {
      rewriter.replaceAllUsesWith(result, newKernel.getResult(curIdx));
      ++curIdx;
    }
  }
  assert(curIdx == newKernel->getNumResults());
  rewriter.eraseOp(op);
  return success();
}

LogicalResult AsyncRoutineOp::canonicalize(AsyncRoutineOp op,
                                           PatternRewriter &rewriter) {
  if (succeeded(deduplicateOperands(op, rewriter)))
    return success();
  if (succeeded(sinkChainCreateOperands(op, rewriter)))
    return success();
  return removeRedundantResults(op, rewriter);
}

OpFoldResult ToMGPFromOpaqueOp::fold(FoldAdaptor adaptor) {
  // Fold away when input and output types are the same.
  // This can happen if the input's type was converted upstream.
  if (getInput().getType() == getDataPointer().getType())
    return getInput();
  return {};
}

OpFoldResult AsyncUnpackOp::fold(FoldAdaptor adaptor) {
  // Fold `mogg.async.unpack(mogg.async.pack(x))` -> `x`.
  if (auto packOp = getInput().getDefiningOp<AsyncPackOp>())
    return packOp.getInput();
  return {};
}

OpFoldResult AsyncPackOp::fold(FoldAdaptor adaptor) {
  // Fold `mogg.async.pack(mogg.async.unpack(x))` -> `x`.
  if (auto unpackOp = getInput().getDefiningOp<AsyncUnpackOp>())
    return unpackOp.getInput();
  return {};
}

LogicalResult ChainSinkOp::canonicalize(ChainSinkOp sinkOp,
                                        mlir::PatternRewriter &rewriter) {
  if (auto dummyOp = sinkOp.getChain().getDefiningOp<ChainDummyOp>()) {
    rewriter.eraseOp(sinkOp);
    return success();
  }

  return failure();
}
namespace {

// Pattern to remove unused results from TryOp and update yield operations.
struct RemoveUnusedTryResults : public OpRewritePattern<TryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TryOp tryOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(tryOp.getResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    if (usedResultIndices.size() == tryOp.getNumResults())
      return failure();

    auto newTryOp =
        TryOp::create(rewriter, tryOp.getLoc(), TypeRange(usedResultTypes));

    rewriter.cloneRegionBefore(tryOp.getTryRegion(), newTryOp.getTryRegion(),
                               newTryOp.getTryRegion().begin());

    rewriter.cloneRegionBefore(tryOp.getExceptRegion(),
                               newTryOp.getExceptRegion(),
                               newTryOp.getExceptRegion().begin());

    auto tryYield =
        cast<YieldOp>(newTryOp.getTryRegion().front().getTerminator());
    SmallVector<Value> newTryYieldOperands;
    for (unsigned idx : usedResultIndices) {
      newTryYieldOperands.push_back(tryYield.getOperand(idx));
    }
    rewriter.setInsertionPoint(tryYield);
    rewriter.replaceOpWithNewOp<YieldOp>(tryYield, newTryYieldOperands);

    auto exceptYield =
        cast<YieldOp>(newTryOp.getExceptRegion().front().getTerminator());
    SmallVector<Value> newExceptYieldOperands;
    SmallVector<Value> usedExceptValues;
    for (unsigned idx : usedResultIndices) {
      Value operand = exceptYield.getOperand(idx);
      newExceptYieldOperands.push_back(operand);
      usedExceptValues.push_back(operand);
    }
    rewriter.setInsertionPoint(exceptYield);
    rewriter.replaceOpWithNewOp<YieldOp>(exceptYield, newExceptYieldOperands);

    // Erase unused AsyncErrorOp operations from the except block to prevent CSE
    // from merging them. Each AsyncErrorOp must remain distinct because they
    // correspond to specific output indices for emit-mojo code generation.
    SmallVector<Operation *> opsToErase;
    for (Operation &op : newTryOp.getExceptRegion().front()) {
      if (auto asyncError = dyn_cast<AsyncErrorOp>(&op)) {
        if (!llvm::is_contained(usedExceptValues, asyncError.getResult())) {
          opsToErase.push_back(&op);
        }
      }
    }
    for (Operation *op : opsToErase) {
      rewriter.eraseOp(op);
    }

    SmallVector<Value> replacements(tryOp.getNumResults());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices)) {
      replacements[oldIdx] = newTryOp.getResult(newIdx);
    }

    rewriter.replaceOp(tryOp, replacements);
    return success();
  }
};
} // namespace

void TryOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<RemoveUnusedTryResults>(context);
}
