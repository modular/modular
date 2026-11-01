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

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "Support/AssertStream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetOperations.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_CANONICALIZECHAINSEQUENCES
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

static bool hasSingleChain(MOMutableOpInterface mutOp) {
  return mutOp.getInChains().size() == 1 && mutOp.getOutChains().size() == 1;
}

/// Collects operands with write effects for a mutable op. When effects are
/// unknown, treats all mutable operands as written so chain relaxation is
/// blocked.
static llvm::SmallDenseSet<Value>
collectWriteEffectOperands(MOMutableOpInterface mutOp) {
  std::optional<llvm::SmallVector<mlir::MemoryEffects::EffectInstance>>
      effects = mlir::getEffectsRecursively(mutOp);

  if (!effects) {
    auto range =
        llvm::make_filter_range(mutOp->getOperands(), isMutableOperand);
    return {range.begin(), range.end()};
  }

  llvm::SmallDenseSet<Value> result;

  for (auto &effect : *effects) {
    if (isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      Value val = effect.getValue();
      ASSERT_STREAM(val, "Needs to have attached value!");
      result.insert(val);
    }
  }

  return result;
}

/// Returns true if the chain linking mutOp1 and mutOp2 can be relaxed without a
/// data race, i.e. neither writes an operand the other uses.
static bool canSafelyReorderMutableOps(MOMutableOpInterface mutOp1,
                                       MOMutableOpInterface mutOp2) {
  // A multi-device op is a rendezvous. Relaxing its chain is race-free but
  // changes when the collective becomes resident on the device, which can
  // leave the other ranks spinning.
  if (mutOp1->hasTrait<MOMultiDevice>() || mutOp2->hasTrait<MOMultiDevice>())
    return false;

  auto r1 = llvm::make_filter_range(mutOp1->getOperands(), isMutableOperand);
  auto r2 = llvm::make_filter_range(mutOp2->getOperands(), isMutableOperand);

  auto mutableOperands1 = llvm::SmallDenseSet<Value>{r1.begin(), r1.end()};
  auto mutableOperands2 = llvm::SmallDenseSet<Value>{r2.begin(), r2.end()};

  auto writeEffectOperands1 = collectWriteEffectOperands(mutOp1);
  auto writeEffectOperands2 = collectWriteEffectOperands(mutOp2);

  if (llvm::set_intersects(mutableOperands1, writeEffectOperands2) ||
      llvm::set_intersects(mutableOperands2, writeEffectOperands1))
    return false;

  // A kernel may return a ref to an input opaque value. Lacking alias
  // analysis, reject ops whose opaque operands name the same type.
  llvm::SmallDenseSet<StringAttr> symbolSet;
  for (Value v : r1) {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(v.getType()))
      symbolSet.insert(opaqueType.getSymbol());
  }

  bool hasSameOpaqueType = llvm::any_of(r2, [&](Value v) {
    if (auto opaqueType = dyn_cast<MO::OpaqueType>(v.getType()))
      return symbolSet.contains(opaqueType.getSymbol());
    return false;
  });

  return !hasSameOpaqueType;
}

namespace {

struct CanonicalizeBufferOpSequences
    : public mlir::OpInterfaceRewritePattern<MOMutableOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult
  matchAndRewrite(MOMutableOpInterface mutOp,
                  mlir::PatternRewriter &origRewriter) const override {
    DeviceAwareRewriter rewriter(mutOp, origRewriter);

    std::optional<ChainValue> inChain = mutOp.getUniqueInChain();
    std::optional<ChainValue> outChain = mutOp.getUniqueOutChain();
    if (!inChain || !outChain)
      return rewriter.notifyMatchFailure(
          mutOp,
          "Mutable op has multiple chains; canonicalization not supported");

    rewriter.setInsertionPointAfter(mutOp);

    auto rawParent = inChain->getDefiningOp();

    // An op in a region body (e.g. mo.parallel) may capture an outer-scope
    // chain; bypassing it would break sequencing.
    if (rawParent && rawParent->getParentRegion() != mutOp->getParentRegion())
      return rewriter.notifyMatchFailure(mutOp,
                                         "chain crosses a region boundary");

    auto parent = dyn_cast_if_present<MOMutableOpInterface>(rawParent);

    if (parent && hasSingleChain(parent) &&
        canSafelyReorderMutableOps(mutOp, parent)) {
      auto parentInChain = parent.getInChains()[0];

      rewriter.modifyOpInPlace(mutOp.getOperation(), [&]() {
        mutOp.getUniqueInChainMutable()->set(parentInChain);
      });

      Value parentOutChain = parent.getOutChains()[0];

      auto merge = rewriter.create<ChainCreateOp>(
          mutOp->getLoc(), ValueRange({*outChain, parentOutChain}));

      rewriter.replaceAllUsesExcept(*outChain, merge.getResult(), merge);

      return success();
    } else if (auto chainCreateOp =
                   dyn_cast_if_present<ChainCreateOp>(rawParent)) {
      mlir::BitVector eraseIdxs(chainCreateOp->getNumOperands());

      SmallVector<Value> newCreateOperands;
      SmallVector<Value> residualOperands;

      for (auto &&[idx, operand] :
           llvm::enumerate(chainCreateOp->getOperands())) {
        if (!isa<ChainType>(operand.getType())) {
          residualOperands.push_back(operand);
          continue;
        }

        auto upstreamOp = operand.getDefiningOp<MOMutableOpInterface>();
        if (upstreamOp && hasSingleChain(upstreamOp) &&
            canSafelyReorderMutableOps(mutOp, upstreamOp)) {
          eraseIdxs[idx] = true;
          newCreateOperands.push_back(operand);
          llvm::append_range(residualOperands, upstreamOp.getInChains());
          continue;
        }

        residualOperands.push_back(operand);
      }

      if (newCreateOperands.empty())
        return rewriter.notifyMatchFailure(
            mutOp,
            "Upstream mo.chain.create has no chain operands derived from "
            "other in-place operations on non-aliased buffers");

      newCreateOperands.push_back(*outChain);

      auto merge =
          rewriter.create<ChainCreateOp>(mutOp->getLoc(), newCreateOperands);

      rewriter.replaceAllUsesExcept(*outChain, merge.getResult(), merge);

      auto newInChain =
          rewriter.create<ChainCreateOp>(mutOp->getLoc(), residualOperands);
      mutOp.getInChainsMutable()[0]->set(newInChain);
      return success();
    } else {
      return rewriter.notifyMatchFailure(
          mutOp, "The op upstream is not a chain.create or buffer mutating "
                 "operation.");
    }
  }
};

struct CanonicalizeChainSequencesPass
    : public impl::CanonicalizeChainSequencesBase<
          CanonicalizeChainSequencesPass> {
  using CanonicalizeChainSequencesBase::CanonicalizeChainSequencesBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns{ctx};

    patterns.add<CanonicalizeBufferOpSequences>(ctx);
    ChainCreateOp::getCanonicalizationPatterns(patterns, ctx);

    // Chains can be long.
    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(750).setUseTopDownTraversal();

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "maximum number of iterations reached in "
                      "canonicalize-chain-sequences");
      return signalPassFailure();
    }
  }
};

} // namespace
