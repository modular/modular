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

#ifndef GENERICML_MGPTRANSFORMS_UTILS_H
#define GENERICML_MGPTRANSFORMS_UTILS_H

#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/AssertStream.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace M::MGP {

/// Returns the first value in args of type T.
template <typename T>
std::optional<Value> findFirstOf(const ValueRange &args) {
  for (const auto &arg : args) {
    if (isa<T>(arg.getType()))
      return arg;
  }
  return std::nullopt;
}

/// Returns all values in args of type T.
template <typename T>
llvm::SmallVector<Value> findAllOf(const ValueRange &args) {
  llvm::SmallVector<Value> res;
  for (const auto &arg : args) {
    if (isa<T>(arg.getType()))
      res.push_back(arg);
  }
  return res;
}

/// Inserts and returns a shallow clone of origOp using given resultTypes and
/// operands. The result will have an empty region for each region of origOp.
/// The location, attributes and properties of the result are taken from origOp.
template <typename T>
T cloneWithArgs(OpBuilder &builder, T origOp, TypeRange resultTypes,
                ValueRange operands) {
  assert(!origOp->hasSuccessors());
  Operation *newOp =
      Operation::create(/*location=*/origOp->getLoc(),
                        /*name=*/origOp->getName(),
                        /*resultTypes=*/resultTypes,
                        /*operands=*/operands,
                        /*attributes=*/origOp->getAttrDictionary(),
                        /*properties=*/origOp->getPropertiesStorage(),
                        /*successors=*/{},
                        /*numRegions=*/origOp->getNumRegions());
  builder.insert(newOp);
  T result = dyn_cast<T>(newOp);
  assert(result);
  return result;
}

/// Accumulates into freeValues all the values in block (and its sub-blocks)
/// which are bound by a proper ancestor of bindingBlock.
void accumFreeValues(mlir::SetVector<Value> &freeValues, Block *block,
                     Block *bindingBlock);

/// Returns the operations in block in a dataflow order, ie s.t. an operation
/// does not have operands defined by operations later in the vector.
///
/// See also mlir::sortTopologically, this implementation has the advantage
/// of spelling out dataflow cycles when they arise.
SmallVector<Operation *> getDataflowOrder(Block *block);

/// Returns true if op can be executed exactly once in overall model.
bool isExactlyOnce(Operation *op);

/// Returns true if op can be executed at most once in overall model.
bool isAtMostOnce(Operation *op);

/// Returns all the block arguments for blocks containing op which could be
/// 'spliced in' to the overall execution graph more than once.
SmallVector<Value> allLoopBoundArgs(Operation *op);

/// Returns all the results of kLoop control flow ops containing op.
SmallVector<Value> allLoopResults(Operation *op);

/// Returns the 'indicating results' for op. If ANY indicating result is ready
/// then the op is known to have FINISHED execution and is not holding any
/// references to any of its operands.
SmallVector<Value> getIndicatingResults(Operation *op);

/// Returns the 'gating operands' for op. The op can only BEGIN execution once
/// ALL gating operands are ready.
SmallVector<Value> getGatingOperands(Operation *op);

/// Returns true if opOperand is a 'gating operand' for its operation.
bool isGatingOperand(const OpOperand *opOperand);

/// Returns the unique OpOperand which is used with 'out' or 'mut' semantics.
/// Returns nullptr if either no mutator or multiple mutators.
OpOperand *getUniqueMutator(Value value);

/// Return the OpOperands for all input chains to op.
SmallVector<OpOperand *, 1> getInputChainOpOperands(Operation *op);

/// Returns the 'earliest' operation (in block operations order) which uses
/// one of results as an operand.
Operation *earliestUsingOp(ArrayRef<Value> results);

using ValueSetVector = llvm::SetVector<Value, SmallVector<Value>>;

/// If chain is defined by a grt.chain.create operation then accumulate
/// all of its operands into primitiveChains (recursively). Otherwise adds
/// chain directly to primitiveChains.
void accumulatePrimitiveChains(ValueSetVector &primitiveChains, Value chain);

/// Moves targetOp before anchorOp, then transitively any operand def not
/// before its user. Requires a DAG. Not needed for MEF correctness, only for
/// readable generated MLIR.
void moveAllBefore(Operation *origTargetOp, Operation *origAnchorOp);

/// Returns a chain value which will be ready only when all of chains (which
/// cannot be empty) are ready. All chains must use the 'native' chain type
/// of the given buffer builder, or !mef.chain if the buffer builder is null.
Value waitForChains(OpBuilder &builder, mlir::Location loc,
                    TypedValue<ContextType> context,
                    MGPBufferBuilder *optBufferBuilder, ArrayRef<Value> chains);

/// Mutates op in-place to ensure its input chain is ready only when all of
/// additionalChains are also ready. Returns failure if op doesn't have any
/// input chain to merge with.
///
/// It is valid to mix-and-match both !mef.chain types and device-specific
/// chain types. For example, the op could be in the cuda dialect, could be
/// gated by a !mef.chain, and additionalChains may include both !mef.chains
/// and !cuda.chains for the same device as op.
LogicalResult alsoWaitForChains(OpBuilder &builder,
                                const BufferBuilderContext &builderContext,
                                Operation *op,
                                ArrayRef<Value> additionalChains);

/// Helper for modifying the 'init' block.
struct InitState {
  InitState(ModelOp model);

  /// 'Allocates' a slot in the context.
  size_t allocSlot();

  /// Returns the current final 'init' chain result.
  TypedValue<MEF::ChainType> getFinalChain();

  /// Update the final 'init' chain result.
  void updateFinalChain(Value newChain);

  /// Returns the 'init' block's context.
  TypedValue<ContextType> getContext();

  /// Returns the devices arguments to the 'init' block.
  MutableArrayRef<mlir::BlockArgument> getDevices() {
    return model.getInitDevices();
  }

  /// Returns the weights registry argument to the 'init' block.
  TypedValue<MGP::OpaqueType> getWeightsRegistry() {
    return cast<TypedValue<MGP::OpaqueType>>(model.getInitWeightsRegistry());
  }

  Block *init;
  OpBuilder builder;
  ContextCreateOp contextCreateOp;
  ModelOp model;
};

/// Helper for modifying the 'execute' block.
struct ExecuteState {
  ExecuteState(MLIRContext *context, ModelOp model);

  /// Returns 'root' chain, ie the first argument to the 'execute' block.
  /// If any non-root chain is ready then the root chain is guaranteed to be
  /// ready, and can thus be ignored.
  TypedValue<MEF::ChainType> getRootChain() const;

  /// Returns MGP context argument to the 'execute' block.
  TypedValue<ContextType> getContext() const;

  /// Returns the devices arguments to the 'execute' block.
  MutableArrayRef<mlir::BlockArgument> getDevices() {
    return model.getExecuteDevices();
  }

  /// Returns the weights registry argument to the 'execute' block.
  TypedValue<MGP::OpaqueType> getWeightsRegistry() {
    return cast<TypedValue<MGP::OpaqueType>>(model.getExecuteWeightsRegistry());
  }

  /// Returns the additional arguments to 'execute' block
  iterator_range<mlir::BlockArgument *> getExtraArgs();

  Block *execute;
  ModelOp model;
  OpBuilder builder;
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_UTILS_H
