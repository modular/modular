//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Support/Compiler/DirectConversion.h"

#include "Support/Compiler/Threading.h"
#include "mlir/IR/Iterators.h"

using namespace M;
using namespace mlir;

//===----------------------------------------------------------------------===//
// DirectRewriter
//===----------------------------------------------------------------------===//

Value DirectRewriter::remap(Value value) {
  if (!typeConverter)
    return value;
  // A type the converter rejects has no converted form, so handing the
  // original back would let a pattern build on an unconverted operand.
  Type target = typeConverter->convertType(value.getType());
  if (!target)
    return {};
  if (target == value.getType())
    return value;

  // A bridge cast created by replaceOp already carries the converted value.
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() == 1 &&
        cast.getInputs()[0].getType() == target) {
      return cast.getInputs()[0];
    }
  }

  auto it = castCache.find(value);
  if (it != castCache.end())
    return it->second;

  OpBuilder::InsertionGuard guard(*this);
  if (Operation *def = value.getDefiningOp())
    setInsertionPointAfter(def);
  else
    setInsertionPointToStart(cast<BlockArgument>(value).getOwner());
  auto castOp =
      UnrealizedConversionCastOp::create(*this, value.getLoc(), target, value);
  Value result = castOp.getResult(0);
  castCache[value] = result;
  return result;
}

LogicalResult
DirectRewriter::getRemappedValues(ValueRange values,
                                  SmallVectorImpl<Value> &result) {
  for (Value value : values) {
    Value remapped = remap(value);
    if (!remapped)
      return failure();
    result.push_back(remapped);
  }
  return success();
}

void DirectRewriter::replaceOp(Operation *op, ValueRange newValues) {
  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");
  SmallVector<Value> replacements;
  replacements.reserve(newValues.size());
  OpBuilder::InsertionGuard guard(*this);
  setInsertionPointAfter(op);
  for (auto [oldResult, newValue] : llvm::zip(op->getResults(), newValues)) {
    if (!newValue || oldResult.getType() == newValue.getType()) {
      replacements.push_back(newValue);
      continue;
    }
    auto castOp = UnrealizedConversionCastOp::create(
        *this, op->getLoc(), oldResult.getType(), newValue);
    replacements.push_back(castOp.getResult(0));
  }
  RewriterBase::replaceOp(op, replacements);
}

void DirectRewriter::replaceOp(Operation *op, Operation *newOp) {
  replaceOp(op, newOp->getResults());
}

void DirectRewriter::eraseOp(Operation *op) {
  // Purge cache entries for dying values: their addresses may be reused by
  // ops created later, and a stale hit would resurrect a freed cast.
  op->walk([&](Operation *nested) {
    if (onErase)
      onErase(nested);
    for (Value result : nested->getResults())
      castCache.erase(result);
    for (Region &region : nested->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          castCache.erase(arg);
        }
      }
    }
  });
  RewriterBase::eraseOp(op);
}

void DirectRewriter::eraseBlock(Block *block) {
  for (BlockArgument arg : block->getArguments())
    castCache.erase(arg);
  RewriterBase::eraseBlock(block);
}

Block *DirectRewriter::applySignatureConversion(
    Block *block, TypeConverter::SignatureConversion &conversion) {
  unsigned origArgCount = block->getNumArguments();

  SmallVector<Value> newArgs;
  newArgs.reserve(conversion.getConvertedTypes().size());
  for (Type type : conversion.getConvertedTypes())
    newArgs.push_back(block->addArgument(type, getUnknownLoc()));

  OpBuilder::InsertionGuard guard(*this);
  setInsertionPointToStart(block);
  for (unsigned i = 0; i != origArgCount; ++i) {
    BlockArgument orig = block->getArgument(i);
    auto mapping = conversion.getInputMapping(i);
    if (!mapping) {
      assert(orig.use_empty() && "dropped block argument still has uses");
      continue;
    }
    Value replacement;
    if (mapping->replacedWithValues()) {
      assert(mapping->replacementValues.size() == 1 &&
             "1:N replacement values are not supported");
      replacement = mapping->replacementValues.front();
    } else {
      assert(mapping->size == 1 && "1:N argument conversion not supported");
      replacement = newArgs[mapping->inputNo];
      if (auto loc = orig.getLoc(); !isa<UnknownLoc>(loc))
        cast<BlockArgument>(replacement).setLoc(loc);
    }
    if (replacement.getType() != orig.getType()) {
      replacement = UnrealizedConversionCastOp::create(
                        *this, orig.getLoc(), orig.getType(), replacement)
                        .getResult(0);
    }
    orig.replaceAllUsesWith(replacement);
  }
  for (unsigned i = 0; i != origArgCount; ++i)
    castCache.erase(block->getArgument(i));
  block->eraseArguments(0, origArgCount);
  return block;
}

FailureOr<Block *> DirectRewriter::convertRegionTypes(
    Region *region, const TypeConverter &converter,
    TypeConverter::SignatureConversion *entryConversion) {
  if (region->empty())
    return nullptr;
  for (Block &block : llvm::drop_begin(*region)) {
    std::optional<TypeConverter::SignatureConversion> conversion =
        converter.convertBlockSignature(&block);
    if (!conversion)
      return failure();
    applySignatureConversion(&block, *conversion);
  }
  Block *entry = &region->front();
  if (entryConversion) {
    applySignatureConversion(entry, *entryConversion);
    return entry;
  }
  std::optional<TypeConverter::SignatureConversion> conversion =
      converter.convertBlockSignature(entry);
  if (!conversion)
    return failure();
  return applySignatureConversion(entry, *conversion);
}

//===----------------------------------------------------------------------===//
// applyDirectConversion
//===----------------------------------------------------------------------===//

/// Returns the ops directly in `root`, in order.
static SmallVector<Operation *, 0> getTopLevelOps(Operation *root) {
  SmallVector<Operation *, 0> ops;
  for (Region &region : root->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block)
        ops.push_back(&op);
    }
  }
  return ops;
}

/// Calls `walk(topLevelOp, results)` on each of `topLevelOps` in parallel and
/// returns the results joined in the order of `topLevelOps`. Each `walk`
/// collects its results by a walk of its op, so the joined results are in the
/// order of one walk of the root.
template <typename T, typename FnT>
static SmallVector<T> walkTopLevelOps(MLIRContext *ctx,
                                      ArrayRef<Operation *> topLevelOps,
                                      FnT &&walk) {
  auto resultsPerOp = parallelMap(ctx, topLevelOps, [&](Operation *op) {
    SmallVector<T, 0> results;
    walk(op, results);
    return results;
  });
  SmallVector<T> joined;
  for (const SmallVector<T, 0> &results : resultsPerOp)
    joined.append(results.begin(), results.end());
  return joined;
}

/// Folds away the casts in `castOps` that cancel out or became dead, as
/// `mlir::reconcileUnrealizedCasts` does. `castOps` holds each cast once, in
/// the order of the walk that found it.
///
/// The upstream function copies the casts from one hash set to another in
/// bucket order, and insertions in that order collide heavily.
// TODO(MOCO-5464): Call the upstream function when it no longer makes that
// copy.
static void reconcileCasts(ArrayRef<UnrealizedConversionCastOp> castOps) {
  DenseSet<Operation *> castSet(castOps.begin(), castOps.end());

  // Returns the cast that defines all inputs of `castOp`, in the same order.
  auto getInputCast =
      [](UnrealizedConversionCastOp castOp) -> UnrealizedConversionCastOp {
    if (castOp.getInputs().empty())
      return {};
    auto inputCastOp =
        castOp.getInputs().front().getDefiningOp<UnrealizedConversionCastOp>();
    if (!inputCastOp || inputCastOp.getOutputs() != castOp.getInputs())
      return {};
    return inputCastOp;
  };

  // From the last cast to the first, replace a cast with the inputs of the
  // nearest cast up its input chain that takes the types this cast returns.
  for (UnrealizedConversionCastOp castOp : llvm::reverse(castOps)) {
    for (UnrealizedConversionCastOp inputCast = castOp; inputCast;
         inputCast = getInputCast(inputCast)) {
      if (inputCast.getInputs().getTypes() != castOp.getResultTypes())
        continue;
      bool isCycle = llvm::any_of(inputCast.getInputs(), [&](Value input) {
        return input.getDefiningOp() == castOp;
      });
      if (!isCycle)
        castOp.replaceAllUsesWith(inputCast.getInputs());
      break;
    }
  }

  // A cast is live when an op outside `castOps` uses a result of it, directly
  // or through other casts of `castOps`.
  DenseSet<Operation *> liveCasts;
  liveCasts.reserve(castOps.size());
  auto markLive = [&](Operation *liveCast) {
    SmallVector<Operation *> worklist = {liveCast};
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (!liveCasts.insert(op).second)
        continue;
      for (Value operand : op->getOperands())
        if (Operation *def = operand.getDefiningOp())
          if (castSet.contains(def))
            worklist.push_back(def);
    }
  };
  for (UnrealizedConversionCastOp castOp : castOps) {
    if (liveCasts.contains(castOp))
      continue;
    if (llvm::any_of(castOp->getUsers(),
                     [&](Operation *user) { return !castSet.contains(user); }))
      markLive(castOp);
  }

  for (UnrealizedConversionCastOp castOp : castOps) {
    if (liveCasts.contains(castOp))
      continue;
    castOp->dropAllUses();
    castOp->erase();
  }
}

LogicalResult
M::applyDirectConversion(Operation *root, const TypeConverter *typeConverter,
                         const FrozenRewritePatternSet &patterns,
                         llvm::function_ref<bool(Operation *)> skipOp,
                         llvm::function_ref<bool(Operation *)> mustConvert) {
  const auto &byName = patterns.getOpSpecificNativePatterns();
  auto getCandidates = [&](OperationName name) -> ArrayRef<RewritePattern *> {
    auto it = byName.find(name);
    return it == byName.end() ? ArrayRef<RewritePattern *>()
                              : ArrayRef<RewritePattern *>(it->second);
  };

  bool haveMatchAny = !patterns.getMatchAnyOpNativePatterns().empty();
  auto hasPatterns = [&](Operation *op) {
    return haveMatchAny || !getCandidates(op->getName()).empty();
  };

  // Collect the ops to convert up front, in def-before-use (pre-)order.
  // Patterns may erase pending ops; `pending` keeps the worklist valid
  // (a reused Operation address is not in `pending`, so it is skipped).
  MLIRContext *ctx = root->getContext();
  SmallVector<Operation *> worklist = walkTopLevelOps<Operation *>(
      ctx, getTopLevelOps(root),
      [&](Operation *topLevelOp, SmallVectorImpl<Operation *> &ops) {
        topLevelOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (hasPatterns(op) && (!skipOp || !skipOp(op)))
            ops.push_back(op);
        });
      });
  DenseMap<Operation *, size_t> pending;
  pending.reserve(worklist.size());
  for (auto [i, op] : llvm::enumerate(worklist))
    pending[op] = i;

  DirectRewriter rewriter(root->getContext(), typeConverter);
  // Erasing absent keys from `pending` churns tombstones and slows lookups.
  // The named lambdas below outlive the non-owning function_refs bound to them.
  auto onEraseFn = [&](Operation *op) {
    if (hasPatterns(op)) {
      pending.erase(op);
    }
  };
  rewriter.onErase = onEraseFn;

  // Ops created during conversion are converted too (parity with the
  // conversion framework's new-IR legalization). Moves (region inlining)
  // carry a set `previous` point and are not re-enqueued.
  struct InsertListener : RewriterBase::Listener {
    llvm::function_ref<void(Operation *)> enqueue;
    void notifyOperationInserted(Operation *op,
                                 OpBuilder::InsertPoint previous) override {
      if (!previous.isSet())
        enqueue(op);
    }
  } listener;
  auto enqueueCreated = [&](Operation *op) {
    if (!hasPatterns(op) || (skipOp && skipOp(op))) {
      return;
    }
    pending[op] = worklist.size();
    worklist.push_back(op);
  };
  listener.enqueue = enqueueCreated;
  rewriter.setListener(&listener);

  auto applyBenefitOrdered = [&](Operation *op) -> LogicalResult {
    SmallVector<RewritePattern *, 4> candidates;
    llvm::append_range(candidates, getCandidates(op->getName()));
    for (const RewritePattern &pattern : patterns.getMatchAnyOpNativePatterns())
      candidates.push_back(const_cast<RewritePattern *>(&pattern));
    llvm::stable_sort(candidates, [](RewritePattern *a, RewritePattern *b) {
      return a->getBenefit() > b->getBenefit();
    });
    for (RewritePattern *pattern : candidates) {
      if (succeeded(pattern->matchAndRewrite(op, rewriter))) {
        return success();
      }
    }
    return failure();
  };

  // Index-based: patterns may append newly created ops.
  for (size_t i = 0; i != worklist.size(); ++i) {
    Operation *op = worklist[i];
    auto it = pending.find(op);
    if (it == pending.end() || it->second != i)
      continue;
    pending.erase(it);
    rewriter.setInsertionPoint(op);
    (void)applyBenefitOrdered(op);
  }

  // The patterns replaced and erased top-level ops, so list them again.
  SmallVector<Operation *, 0> topLevelOps = getTopLevelOps(root);

  // Ops the pass declares as necessarily-converted must be gone, including
  // those no pattern matches; the wording is the conversion framework's.
  // Everything else stays unconverted (partial-conversion semantics).
  auto hasUnconvertedOp = [&](Operation *topLevelOp) {
    return topLevelOp
        ->walk<WalkOrder::PreOrder>([&](Operation *op) {
          if ((skipOp && skipOp(op)) || !mustConvert(op))
            return WalkResult::advance();
          return WalkResult::interrupt();
        })
        .wasInterrupted();
  };
  // The parallel scan finds out whether an op is left. The walk on this thread
  // then reports the first one, so the error does not depend on the threads.
  if (mustConvert && parallelAnyOf(ctx, topLevelOps, hasUnconvertedOp)) {
    WalkResult unconverted =
        root->walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (op == root || (skipOp && skipOp(op)) || !mustConvert(op)) {
            return WalkResult::advance();
          }
          op->emitError("failed to legalize operation '")
              << op->getName() << "' that was explicitly marked illegal";
          return WalkResult::interrupt();
        });
    if (unconverted.wasInterrupted())
      return failure();
  }

  // Fold away bridge casts that became dead or cancel out (A->B->A pairs),
  // so converted-consumer chains reference converted values directly.
  SmallVector<UnrealizedConversionCastOp> casts =
      walkTopLevelOps<UnrealizedConversionCastOp>(
          ctx, topLevelOps,
          [](Operation *topLevelOp,
             SmallVectorImpl<UnrealizedConversionCastOp> &found) {
            topLevelOp->walk([&](UnrealizedConversionCastOp castOp) {
              found.push_back(castOp);
            });
          });
  reconcileCasts(casts);
  return success();
}
