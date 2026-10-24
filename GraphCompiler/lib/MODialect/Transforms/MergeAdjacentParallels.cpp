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
// mo-merge-adjacent-parallels
//
// Fuses a producer and consumer `mo.parallel` when the consumer takes one or
// more producer bundle results directly as inputs (strict SSA equality, made
// possible by the upstream `bundle(unbundle(X)...) -> X` fold).
//
// With several parallel producers (e.g. a residual `mo.add` fed by two
// allreduce parallels), the first one in consumer-input order that passes the
// dependence check is merged; the rest may merge in a later iteration.
//
// Mechanics:
//
//   * A shared input's consumer block arg is remapped to the producer's
//     yield operand, skipping the bundle round-trip.
//   * Non-shared consumer inputs get a fresh merged block arg each.
//   * Producer then consumer body ops are cloned into the merged block, and a
//     new `mo.yield` joins the surviving producer yields with the consumer's.
//
// Producer results used only by the consumer are dropped from the merged
// results; otherwise MOGG epilogue fusion stalls on a spurious live output.
//
// Buffer + chain policy (strict, matching allreduce signal buffers):
//
//   * Both chained: buffer lists must be identical and the consumer's in-chain
//     must be the producer's out-chain with no other users. The merged op
//     reuses the producer's buffers and in-chain.
//   * Only one side chained: the merged op inherits that side.
//   * Neither: the merged op is chainless.
//   * Buffer mismatch: leave the pair unmerged.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "merge-adjacent-parallels"

namespace M::MO {

#define GEN_PASS_DEF_MERGEADJACENTPARALLELS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

//===----------------------------------------------------------------------===//
// Small utilities
//===----------------------------------------------------------------------===//

/// Returns the number of bundle outputs (excluding the trailing chain, if
/// present).
static size_t numBundleResults(ParallelOp op) {
  return op.getNumResults() - (op.hasChain() ? 1 : 0);
}

/// Returns true if \p producer and \p consumer have the same buffer operands
/// in the same order.
static bool buffersMatch(ParallelOp producer, ParallelOp consumer) {
  auto producerBuffers = producer.getBuffers();
  auto consumerBuffers = consumer.getBuffers();
  if (producerBuffers.size() != consumerBuffers.size())
    return false;
  for (auto [producerBuffer, consumerBuffer] :
       llvm::zip(producerBuffers, consumerBuffers))
    if (producerBuffer != consumerBuffer)
      return false;
  return true;
}

/// Returns true if \p root transitively depends on a result of \p target.
static bool dependsOn(Value root, Operation *target) {
  llvm::DenseSet<Value> visited;
  llvm::SmallVector<Value> stack;
  stack.push_back(root);

  while (!stack.empty()) {
    Value value = stack.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *def = value.getDefiningOp();
    if (!def)
      continue;
    if (def == target)
      return true;
    llvm::append_range(stack, def->getOperands());
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Candidate discovery
//===----------------------------------------------------------------------===//

/// `sharedInputs[i] = producerIdx` means consumer input #i is producer result
/// #producerIdx; nullopt means it is carried into the merged op.
struct MergeCandidate {
  ParallelOp producer;
  ParallelOp consumer;
  SmallVector<std::optional<size_t>> sharedInputs;
};

/// Returns true if a non-shared consumer input transitively depends on
/// \p producer, so merging would create a cycle. This rejects an upstream
/// producer when a downstream producer also feeds the consumer.
static bool
nonSharedInputDependsOnProducer(ParallelOp consumer, ParallelOp producer,
                                ArrayRef<std::optional<size_t>> sharedInputs) {
  for (auto [i, input] : llvm::enumerate(consumer.getInputs())) {
    if (sharedInputs[i].has_value())
      continue;
    if (dependsOn(input, producer.getOperation()))
      return true;
  }
  return false;
}

/// Returns the first producer, in consumer-input order, in the same block as
/// \p consumer and with no dependence cycle through it.
static std::optional<MergeCandidate> findMergeCandidate(ParallelOp consumer) {
  // First-seen input order keeps selection deterministic.
  SmallVector<ParallelOp> candidatesInOrder;
  llvm::SmallDenseSet<Operation *> seen;
  for (Value input : consumer.getInputs()) {
    auto prod = dyn_cast_or_null<ParallelOp>(input.getDefiningOp());
    if (!prod || prod.getOperation() == consumer.getOperation())
      continue;
    if (seen.insert(prod.getOperation()).second)
      candidatesInOrder.push_back(prod);
  }

  for (ParallelOp producer : candidatesInOrder) {
    if (producer->getBlock() != consumer->getBlock())
      continue;

    SmallVector<std::optional<size_t>> sharedInputs(
        consumer.getInputs().size());
    for (auto [i, input] : llvm::enumerate(consumer.getInputs()))
      if (input.getDefiningOp() == producer.getOperation())
        sharedInputs[i] = cast<OpResult>(input).getResultNumber();

    if (nonSharedInputDependsOnProducer(consumer, producer, sharedInputs))
      continue;

    return MergeCandidate{producer, consumer, std::move(sharedInputs)};
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Buffer + chain resolution
//===----------------------------------------------------------------------===//

struct BufferChainPlan {
  /// Ordered buffer operand list for the merged op.
  SmallVector<Value> buffers;
  /// In-chain for the merged op (empty = chainless).
  Value inChain;
};

/// Applies the buffer+chain rules from the file header, or returns
/// std::nullopt to reject the candidate. mo.buffer handling can go once
/// GEX-3527 lands.
static std::optional<BufferChainPlan>
resolveBufferChain(const MergeCandidate &candidate) {
  ParallelOp producer = candidate.producer;
  ParallelOp consumer = candidate.consumer;
  BufferChainPlan plan;

  if (producer.hasChain() && consumer.hasChain()) {
    if (!buffersMatch(producer, consumer))
      return std::nullopt;
    if (consumer.getInChain() != producer.getOutChain())
      return std::nullopt;
    if (!producer.getOutChain().hasOneUse())
      return std::nullopt;
    plan.buffers.assign(producer.getBuffers().begin(),
                        producer.getBuffers().end());
    plan.inChain = producer.getInChain();
    return plan;
  }

  if (producer.hasChain()) {
    plan.buffers.assign(producer.getBuffers().begin(),
                        producer.getBuffers().end());
    plan.inChain = producer.getInChain();
    return plan;
  }
  if (consumer.hasChain()) {
    plan.buffers.assign(consumer.getBuffers().begin(),
                        consumer.getBuffers().end());
    plan.inChain = consumer.getInChain();
    return plan;
  }

  return plan; // neither has a chain
}

//===----------------------------------------------------------------------===//
// Merge plan
//===----------------------------------------------------------------------===//

struct MergePlan {
  /// Producer inputs, then consumer non-shared inputs.
  SmallVector<Value> mergedInputs;
  /// Per shared consumer input, the producer yield operand to substitute; null
  /// for non-shared inputs.
  SmallVector<Value> consumerSharedYieldOperand;
  /// Per non-shared consumer input, its merged block-arg index.
  SmallVector<size_t> consumerAppendedArgIdx;
  /// Producer bundle results used only by the consumer.
  llvm::BitVector dropProducerYield;
  /// Kept producer bundles, consumer bundles, then optional trailing chain.
  SmallVector<Type> mergedResultTypes;
  /// Per producer bundle, its merged result index, or nullopt if dropped.
  SmallVector<std::optional<size_t>> producerMergedIdx;
  size_t consumerMergedStart = 0;
};

static MergePlan planMerge(const MergeCandidate &candidate,
                           const BufferChainPlan &bufferChain) {
  ParallelOp producer = candidate.producer;
  ParallelOp consumer = candidate.consumer;
  Block &producerBody = producer.getBodyRegion().front();
  auto producerYield = cast<YieldOp>(producerBody.getTerminator());
  size_t producerNumBundles = numBundleResults(producer);
  size_t consumerNumBundles = numBundleResults(consumer);

  MergePlan plan;
  plan.mergedInputs.assign(producer.getInputs().begin(),
                           producer.getInputs().end());

  plan.dropProducerYield.resize(producerNumBundles);
  for (size_t i = 0; i < producerNumBundles; ++i) {
    bool hasExternalUser = false;
    for (Operation *user : producer.getResult(i).getUsers())
      if (user != consumer.getOperation()) {
        hasExternalUser = true;
        break;
      }
    plan.dropProducerYield[i] = !hasExternalUser;
  }

  plan.consumerSharedYieldOperand.assign(consumer.getInputs().size(), Value{});
  plan.consumerAppendedArgIdx.assign(consumer.getInputs().size(), 0);
  // Non-shared inputs follow the producer's bundle args and precede the
  // trailing chain arg, so don't index from producerBody.getNumArguments().
  size_t appendedIdx = producer.getInputs().size();
  for (auto [bi, input] : llvm::enumerate(consumer.getInputs())) {
    if (candidate.sharedInputs[bi].has_value()) {
      plan.consumerSharedYieldOperand[bi] =
          producerYield.getOperand(*candidate.sharedInputs[bi]);
      continue;
    }
    plan.mergedInputs.push_back(input);
    plan.consumerAppendedArgIdx[bi] = appendedIdx++;
  }

  plan.producerMergedIdx.assign(producerNumBundles, std::nullopt);
  for (size_t i = 0; i < producerNumBundles; ++i) {
    if (plan.dropProducerYield[i])
      continue;
    plan.producerMergedIdx[i] = plan.mergedResultTypes.size();
    plan.mergedResultTypes.push_back(producer.getResult(i).getType());
  }
  plan.consumerMergedStart = plan.mergedResultTypes.size();
  for (size_t i = 0; i < consumerNumBundles; ++i)
    plan.mergedResultTypes.push_back(consumer.getResult(i).getType());
  if (bufferChain.inChain)
    plan.mergedResultTypes.push_back(ChainType::get(producer.getContext()));

  return plan;
}

//===----------------------------------------------------------------------===//
// Merge
//===----------------------------------------------------------------------===//

/// Clones the producer's then the consumer's body ops into \p merged's block
/// and emits the combined yield.
static void spliceBodies(ParallelOp producer, ParallelOp consumer,
                         ParallelOp merged, const MergeCandidate &candidate,
                         const MergePlan &plan) {
  Block &producerBody = producer.getBodyRegion().front();
  Block &consumerBody = consumer.getBodyRegion().front();
  Block *mergedBlock = &merged.getBodyRegion().front();
  auto producerYield = cast<YieldOp>(producerBody.getTerminator());
  auto consumerYield = cast<YieldOp>(consumerBody.getTerminator());

  size_t producerNumInputs = producer.getInputs().size();
  size_t consumerNumInputs = consumer.getInputs().size();
  size_t producerNumBundles = numBundleResults(producer);
  size_t consumerNumBundles = numBundleResults(consumer);

  // Body args are
  //   [producer bundle args..., consumer non-shared bundle args..., chain?]
  // so the chain arg is mapped explicitly, not by position.
  Value mergedChainArg =
      merged.hasChain() ? mergedBlock->getArguments().back() : Value{};

  IRMapping mapping;
  for (size_t i = 0; i < producerNumInputs; ++i)
    mapping.map(producerBody.getArgument(i), mergedBlock->getArgument(i));
  // The producer's collective threads from the merged in-chain.
  if (producer.hasChain())
    mapping.map(producerBody.getArguments().back(), mergedChainArg);

  OpBuilder bodyBuilder(merged.getContext());
  bodyBuilder.setInsertionPointToEnd(mergedBlock);
  for (Operation &op : producerBody.without_terminator())
    bodyBuilder.clone(op, mapping);

  // Shared-input args map to the *cloned* producer yield operand, so this must
  // run after the producer body is cloned.
  for (size_t bi = 0; bi < consumerNumInputs; ++bi) {
    Value consumerArg = consumerBody.getArgument(bi);
    if (candidate.sharedInputs[bi].has_value())
      mapping.map(consumerArg,
                  mapping.lookupOrDefault(plan.consumerSharedYieldOperand[bi]));
    else
      mapping.map(consumerArg,
                  mergedBlock->getArgument(plan.consumerAppendedArgIdx[bi]));
  }
  // When both are chained, resolveBufferChain guarantees the consumer's
  // in-chain is the producer's out-chain, i.e. its cloned yielded chain.
  if (consumer.hasChain()) {
    Value consumerChainArg = consumerBody.getArguments().back();
    if (producer.hasChain())
      mapping.map(consumerChainArg,
                  mapping.lookupOrDefault(producerYield.getOperands().back()));
    else
      mapping.map(consumerChainArg, mergedChainArg);
  }
  for (Operation &op : consumerBody.without_terminator())
    bodyBuilder.clone(op, mapping);

  // The merged out-chain is the downstream side's yielded chain; when both are
  // chained, the producer's is an internal edge and is not re-yielded.
  SmallVector<Value> yieldValues;
  for (size_t i = 0; i < producerNumBundles; ++i)
    if (!plan.dropProducerYield[i])
      yieldValues.push_back(
          mapping.lookupOrDefault(producerYield.getOperand(i)));
  for (size_t i = 0; i < consumerNumBundles; ++i)
    yieldValues.push_back(mapping.lookupOrDefault(consumerYield.getOperand(i)));
  if (merged.hasChain()) {
    YieldOp chainSource = consumer.hasChain() ? consumerYield : producerYield;
    yieldValues.push_back(
        mapping.lookupOrDefault(chainSource.getOperands().back()));
  }
  YieldOp::create(bodyBuilder, merged.getLoc(), yieldValues);
}

/// Routes all uses of \p producer's and \p consumer's results to the
/// corresponding merged results.
static void replaceUses(ParallelOp producer, ParallelOp consumer,
                        ParallelOp merged, const MergePlan &plan,
                        const BufferChainPlan &bufferChain) {
  size_t producerNumBundles = numBundleResults(producer);
  size_t consumerNumBundles = numBundleResults(consumer);

  for (size_t i = 0; i < producerNumBundles; ++i)
    if (plan.producerMergedIdx[i])
      producer.getResult(i).replaceAllUsesWith(
          merged.getResult(*plan.producerMergedIdx[i]));
  for (size_t i = 0; i < consumerNumBundles; ++i)
    consumer.getResult(i).replaceAllUsesWith(
        merged.getResult(plan.consumerMergedStart + i));
  if (bufferChain.inChain) {
    Value mergedOutChain = merged.getResult(merged.getNumResults() - 1);
    if (producer.hasChain())
      producer.getOutChain().replaceAllUsesWith(mergedOutChain);
    if (consumer.hasChain())
      consumer.getOutChain().replaceAllUsesWith(mergedOutChain);
  }
}

/// Merges \p candidate.consumer into \p candidate.producer.
static void mergeCandidate(MergeCandidate &candidate,
                           const BufferChainPlan &bufferChain,
                           OpBuilder &builder) {
  MergePlan plan = planMerge(candidate, bufferChain);

  builder.setInsertionPoint(candidate.producer);
  ParallelOp merged = ParallelOp::create(
      builder, candidate.producer.getLoc(), plan.mergedInputs,
      bufferChain.buffers, bufferChain.inChain, plan.mergedResultTypes);
  spliceBodies(candidate.producer, candidate.consumer, merged, candidate, plan);
  replaceUses(candidate.producer, candidate.consumer, merged, plan,
              bufferChain);
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

/// Performs at most one merge and returns whether it fired.
///
/// Continuing the walk after a merge is unsafe (GEX-3756): a producer whose
/// only yield was absorbed still has the not-yet-erased consumer as a user, so
/// it can be visited and merged again, double-erasing it. One merge per walk
/// also guarantees no op is cloned twice.
static bool runOneIteration(Operation *root, OpBuilder &builder) {
  bool changed = false;
  root->walk([&](ParallelOp parallelOp) {
    auto candidate = findMergeCandidate(parallelOp);
    if (!candidate)
      return WalkResult::advance();

    auto bufferChain = resolveBufferChain(*candidate);
    if (!bufferChain)
      return WalkResult::advance();

    mergeCandidate(*candidate, *bufferChain, builder);
    candidate->consumer.erase();
    candidate->producer.erase();
    changed = true;
    return WalkResult::interrupt();
  });
  return changed;
}

struct MergeAdjacentParallelsPass
    : public impl::MergeAdjacentParallelsBase<MergeAdjacentParallelsPass> {
  using MergeAdjacentParallelsBase::MergeAdjacentParallelsBase;

  void runOnOperation() override {
    OpBuilder builder{&getContext()};
    while (runOneIteration(getOperation(), builder)) {
    }
  }
};

} // namespace
