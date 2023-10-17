//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "kgen-sccp"

using namespace M;
using namespace M::HLCF;
using namespace KGEN;
using namespace mlir::dataflow;
using mlir::ChangeResult;

//===----------------------------------------------------------------------===//
// SCCPPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_SCCP
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class SCCPAnalysis {
public:
  explicit SCCPAnalysis() {}

  /// ConstantValue lattice element type.
  using ConstantState = Lattice<ConstantValue>;

  /// Map type from Value to ConstantValue lattice element.
  using ConstantStateType = DenseMap<Value, ConstantState>;

  /// Process a Region operation.
  LogicalResult processRegion(Region &region, ConstantStateType &state,
                              bool &hasEarlyExits,
                              bool setBlockArgToEntryState = true);

  /// Helper function to rewrite the IR with SCCP analysis results.
  LogicalResult rewrite(MLIRContext *context,
                        MutableArrayRef<Region> initialRegions);

  LogicalResult run(Operation *op);

private:
  /// ConstantValue lattice states for ControlFlow type of nodes.
  struct ControlFlowOperationState {
    /// Work list for analyzing loops, each item in the list is possible
    /// constant values mapped to the inputs to each loop iteration.
    std::queue<SmallVector<Attribute>> entryStates;

    /// Lattice for op results that will be updated by ControlFlowTerminators.
    ConstantStateType exitStates;
  };

  /// Process a ControlFlowNode operation.
  /// `state` is the entry state of the lattice analysis values.
  /// `shouldContinue` is a flag to keep track if operation traversing
  /// in the parent region should keep going or stop in case early exits
  /// happen, such as break, continue, return.
  LogicalResult processControlFlowNode(ControlFlowNode node,
                                       ConstantStateType &state,
                                       bool &shouldContinue);

  /// Process a ControlFlowTerminator operation.
  void processControlFlowTerminator(ControlFlowTerminator term,
                                    ConstantStateType &state);

  /// Visit a general operation to apply the transform function.
  static void visitOperation(Operation *op, ConstantStateType &state);

  /// Get the lattice element for value from state.
  static ConstantState *getLatticeElement(Value value,
                                          ConstantStateType &state);

  /// Get the lattice elements for op's operands.
  static void getValuesLattice(SmallVectorImpl<Attribute> &attributes,
                               ValueRange value, ConstantStateType &state);

  /// Set state to Unknown.
  static void setToEntryState(ConstantState *state);

  /// Set states to Unknown.
  static void setAllToEntryStates(ArrayRef<ConstantState *> states);

  /// Join lattice in rhs with lhs only if value exists in both maps.
  static ChangeResult joinStates(ConstantStateType &lhs,
                                 ConstantStateType &rhs);

  /// Merge lattice from state to current. Join the lattice if exist in both.
  static ChangeResult mergeStates(ConstantStateType &current,
                                  ConstantStateType &state);

  /// Update parentOp's exit states.
  static void updateParentOpOutputState(ValueRange termValues,
                                        Operation *parentOp,
                                        ConstantStateType &termState,
                                        ConstantStateType &parentOutputState);

  /// Helper function to rewrite the IR with SCCP analysis results.
  static LogicalResult replaceWithConstant(ConstantStateType &state,
                                           OpBuilder &builder,
                                           mlir::OperationFolder &folder,
                                           Value value);

  int64_t getLoopConvergeThreshold(Operation *op);

  /// Map from a ControlFlowNode/Terminator to its states (entry/exit).
  DenseMap<Operation *, ControlFlowOperationState> controlFlowOperationStates;

  /// Input attributes to current Loop that is being processed.
  /// Used to detect early convergence point when analysing loops.
  /// NOTE: This is not thread safe within the pass.
  DenseMap<Operation *, SmallVector<Attribute>> currentLoopInputs;

  /// State for the top operation that the analysis starts from.
  ConstantStateType topState;
};

} // namespace

/// Print lattice content for debugging.
static void printState(SCCPAnalysis::ConstantStateType &state,
                       llvm::raw_ostream &os) {
  for (auto &[value, lattice] : state) {
    os << "============================\n";
    os << "value: " << value << "\n";
    lattice.print(os);
    os << "\n";
  }
}

ChangeResult SCCPAnalysis::joinStates(ConstantStateType &lhs,
                                      ConstantStateType &rhs) {
  ChangeResult changed = ChangeResult::NoChange;
  for (auto &[value, lattice] : rhs) {
    if (auto iter = lhs.find(value); iter != lhs.end())
      changed |= iter->getSecond().join(lattice);
  }
  return changed;
}

ChangeResult SCCPAnalysis::mergeStates(ConstantStateType &current,
                                       ConstantStateType &state) {
  ChangeResult changed = ChangeResult::NoChange;
  for (auto &[value, lattice] : state) {
    ConstantState *currLattice = getLatticeElement(value, current);
    changed |= currLattice->join(lattice);
  }
  return changed;
}

SCCPAnalysis::ConstantState *
SCCPAnalysis::getLatticeElement(Value value, ConstantStateType &state) {
  auto iter = state.find(value);
  if (iter == state.end())
    iter = state.insert({value, ConstantState(value)}).first;

  return &iter->getSecond();
}

void SCCPAnalysis::getValuesLattice(SmallVectorImpl<Attribute> &attributes,
                                    ValueRange values,
                                    ConstantStateType &state) {
  for (Value value : values) {
    ConstantState *lattice = getLatticeElement(value, state);
    assert(!lattice->getValue().isUninitialized() &&
           "All operands should have initialized lattice value.");
    attributes.push_back(lattice->getValue().getConstantValue());
  }
}

void SCCPAnalysis::setToEntryState(ConstantState *state) {
  state->join(ConstantValue::getUnknownConstant());
}

void SCCPAnalysis::setAllToEntryStates(ArrayRef<ConstantState *> states) {
  for (ConstantState *state : states)
    setToEntryState(state);
}

void SCCPAnalysis::visitOperation(Operation *op, ConstantStateType &state) {
  SmallVector<Attribute> constantOperands;
  getValuesLattice(constantOperands, op->getOperands(), state);

  SmallVector<ConstantState *> results;
  for (Value result : op->getResults())
    results.push_back(getLatticeElement(result, state));

  // Save the original operands and attributes just in case the operation
  // folds in-place. The constant passed in may not correspond to the real
  // runtime value, so in-place updates are not allowed.
  SmallVector<Value> originalOperands(op->getOperands());
  DictionaryAttr originalAttrs = op->getAttrDictionary();

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(constantOperands, foldResults))) {
    setAllToEntryStates(results);
    return;
  }

  // If the folding was in-place, mark the results as overdefined and reset
  // the operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (foldResults.empty()) {
    op->setOperands(originalOperands);
    op->setAttrs(originalAttrs);
    setAllToEntryStates(results);
    return;
  }

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  for (const auto [lattice, foldResult] : llvm::zip(results, foldResults)) {
    // Merge in the result of the fold, either a constant or a value.
    if (Attribute attr = llvm::dyn_cast_if_present<Attribute>(foldResult)) {
      LLVM_DEBUG(llvm::dbgs() << "Folded to constant: " << attr << "\n");
      lattice->join(ConstantValue(attr, op->getDialect()));
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Folded to value: " << foldResult.get<Value>() << "\n");
      lattice->join(*getLatticeElement(foldResult.get<Value>(), state));
    }
  }
}

int64_t SCCPAnalysis::getLoopConvergeThreshold(Operation *op) {
  if (auto forOp = dyn_cast<ForOp>(op); forOp && forOp.getTripCount()) {
    // Use trip count as threshold for a for-loop.
    return forOp.getTripCount().value();
  }

  // TODO: use decorator or more sophisticated heuristics to set per loop
  // threshold.
  return 100;
}

LogicalResult SCCPAnalysis::processControlFlowNode(ControlFlowNode node,
                                                   ConstantStateType &state,
                                                   bool &shouldContinue) {
  // TODO: Add support for other ControlFlowNode, e.g. kgen.try, etc.
  // TODO: This function should work more generally for ControlFlowInterfaces.
  if (isa<IfOp, SwitchOp>(node.getOperation())) {
    // TODO: extend this logic to SwitchOp.
    SmallVector<Attribute> constantOperands;
    getValuesLattice(constantOperands, node.getOperation()->getOperands(),
                     state);
    SmallVector<ControlFlowTarget> targets;
    node.getEntryTargets(constantOperands, targets);

    // Preserve the entry state.
    ConstantStateType entryState = state;
    for (ControlFlowTarget target : targets) {
      if (target.index) {
        // Analyze region with entry state.
        ConstantStateType nestedState = entryState;
        bool hasEarlyExits = true;
        if (failed(processRegion(node->getRegion(target.index.value()),
                                 nestedState, hasEarlyExits)))
          return failure();

        // Directly merge nestedState with state here since we don't need to
        // reset local value's lattice.
        mergeStates(state, nestedState);
        if (!hasEarlyExits && targets.size() == 1) {
          // Break happened, we can break traversal if this is the only branch
          // that is running.
          shouldContinue = false;
        }
      }
    }
    mergeStates(state,
                controlFlowOperationStates[node.getOperation()].exitStates);
    controlFlowOperationStates.erase(node.getOperation());
    return success();
  }

  if (isa<LoopOp, ForOp>(node)) {
    // Prepare for initial loop inputs.
    SmallVector<Attribute> constantOperands;
    getValuesLattice(constantOperands, node->getOperands(), state);

    // Prepare the workList for analyzing the loop.
    ControlFlowOperationState &cfStates = controlFlowOperationStates[node];
    std::queue<SmallVector<Attribute>> &workList = cfStates.entryStates;
    workList.push(constantOperands);

    int64_t iter = 0;
    ConstantStateType currState = state;
    ConstantStateType mergedState = state;

    SmallVector<Attribute> &inputValues =
        currentLoopInputs[node.getOperation()];

    while (!workList.empty() &&
           iter < getLoopConvergeThreshold(node.getOperation())) {
      inputValues = std::move(workList.front());
      workList.pop();

      ConstantStateType nestedState = currState;
      // Prepare for input arguments for this iteration.
      for (auto [inputValue, blockArg] :
           llvm::zip(inputValues, node->getRegions().front().getArguments())) {
        ConstantState *lattice = getLatticeElement(blockArg, nestedState);
        if (!inputValue)
          setToEntryState(lattice);
        else
          lattice->join(ConstantValue(inputValue, node->getDialect()));
      }

      // Process loop body.
      bool hasEarlyExits = true;
      if (failed(processRegion(node->getRegions().front(), nestedState,
                               hasEarlyExits, false)))
        return failure();

      // Each loop iteration should run with clean state for values in the
      // scope of the loop body. Only join nestedState (result of current
      // iteration) with currState to carry over to the next iteration so that
      // only propagating values that are not loop scoped.
      joinStates(currState, nestedState);

      // Keep a version of merged nestedState to propagate loop scoped
      // constant values for rewriting later. If a loop scoped value is not
      // constant between iterations, it will be unknown in mergedState and
      // will not be rewritten.
      mergeStates(mergedState, nestedState);
      ++iter;
    }

    if (workList.empty()) {
      // Merge analysis states if analyze loop converges.
      mergeStates(state, cfStates.exitStates);
      mergeStates(state, mergedState);
    } else {
      // Mark loop results as Unknown.
      for (Value result : node.getOperation()->getResults())
        setToEntryState(getLatticeElement(result, state));
    }
    // Clean up states (which is being updated by this op's terminators) when
    // analyzing is done.
    controlFlowOperationStates.erase(node);
    currentLoopInputs.erase(node);
    return success();
  }

  // Otherwise, mark all results as Unknown.
  for (Value result : node.getOperation()->getResults())
    setToEntryState(getLatticeElement(result, state));

  return success();
}

void SCCPAnalysis::updateParentOpOutputState(
    ValueRange termValues, Operation *parentOp, ConstantStateType &termState,
    ConstantStateType &parentOutputState) {
  for (auto [operand, opResult] :
       llvm::zip(termValues, parentOp->getResults())) {
    ConstantState *lattice0 = getLatticeElement(operand, termState);
    ConstantState *lattice1 = getLatticeElement(opResult, parentOutputState);
    lattice1->join(*lattice0);
  }
}

void SCCPAnalysis::processControlFlowTerminator(ControlFlowTerminator term,
                                                ConstantStateType &termState) {
  // TODO: Add support for other ControlFlowTerminators, e.g. kgen.return, etc.
  if (auto breakOp = dyn_cast<BreakOp>(term.getOperation())) {
    Operation *parentLoop = getParentNode(term);
    // Update parent loop's exit state.
    ControlFlowOperationState &cfOpStates =
        controlFlowOperationStates[parentLoop];

    updateParentOpOutputState(term->getOperands(), parentLoop, termState,
                              cfOpStates.exitStates);
    return;
  }

  if (auto continueOp = dyn_cast<ContinueOp>(term.getOperation())) {
    Operation *parentLoop = getParentNode(term);
    // Prepare new inputs for parent loop.
    SmallVector<Attribute> constantOperands;
    getValuesLattice(constantOperands, term.getOperation()->getOperands(),
                     termState);

    ControlFlowOperationState &cfOpStates =
        controlFlowOperationStates[parentLoop];
    // Only push the new inputs if it is different from current one.
    if (constantOperands != currentLoopInputs[parentLoop])
      cfOpStates.entryStates.push(constantOperands);

    return;
  }

  if (auto yieldOp = dyn_cast<YieldOp>(term.getOperation())) {
    Operation *parentOp = getParentNode(term);
    ControlFlowOperationState &cfOpstates =
        controlFlowOperationStates[parentOp];
    // update parent op's exit state
    updateParentOpOutputState(term->getOperands(), parentOp, termState,
                              cfOpstates.exitStates);
    return;
  }

  if (auto forYieldOp = dyn_cast<ForYieldOp>(term.getOperation())) {
    Operation *parentOp = getParentNode(term);
    SmallVector<Attribute> constantOperands;
    getValuesLattice(constantOperands, forYieldOp.getOperation()->getOperands(),
                     termState);
    SmallVector<ControlFlowTarget> targets;
    forYieldOp.getBranchTargets(constantOperands, targets);
    ControlFlowOperationState &cfOpstates =
        controlFlowOperationStates[parentOp];

    for (ControlFlowTarget &target : targets) {
      if (target.index) {
        // Branch back to for-loop body.
        // Only push the new inputs if it is different from current one.
        if (constantOperands != currentLoopInputs[parentOp])
          cfOpstates.entryStates.push(constantOperands);
      } else
        updateParentOpOutputState(forYieldOp.getReturnValues(), parentOp,
                                  termState, cfOpstates.exitStates);
    }
    return;
  }

  // Otherwise, mark all results as Unknown.
  for (Value result : term.getOperation()->getResults())
    setToEntryState(getLatticeElement(result, termState));
}

LogicalResult SCCPAnalysis::processRegion(Region &region,
                                          ConstantStateType &state,
                                          bool &hasEarlyExits,
                                          bool setBlockArgToEntryState) {
  if (!llvm::hasSingleElement(region)) {
    return region.getParentOp()->emitError(
        "'sccp' can only be run on operations with all single block "
        "regions");
  }

  Block &block = region.front();
  if (setBlockArgToEntryState) {
    SmallVector<ConstantState *> blockArgLattice;
    for (BlockArgument &arg : block.getArguments())
      blockArgLattice.push_back(getLatticeElement(arg, state));
    setAllToEntryStates(blockArgLattice);
  }

  for (Operation &op : block) {
    if (auto node = dyn_cast<ControlFlowNode>(op)) {
      bool shouldContinue = true;

      if (failed(processControlFlowNode(node, state, shouldContinue)))
        return failure();

      if (!shouldContinue)
        break;
      continue;
    }

    if (auto term = dyn_cast<ControlFlowTerminator>(op)) {
      processControlFlowTerminator(term, state);
      // Tell parent region that there is early exit (so that the parent region
      // can decide whether to continue traverse the rest of the operation or
      // not).
      hasEarlyExits = !(isa<ContinueOp>(op) || isa<BreakOp>(op) ||
                        isa<KGEN::UnreachableOp>(op));
      break;
    }

    if (op.getNumRegions() > 0) {
      ConstantStateType entryState = state;
      for (Region &region : op.getRegions()) {
        ConstantStateType nestedState = entryState;
        bool hasEarlyExits = true;
        if (failed(processRegion(region, nestedState, hasEarlyExits)))
          return failure();
        mergeStates(state, nestedState);
      }
      continue;
    } else {
      visitOperation(&op, state);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SCCP Rewrites
//===----------------------------------------------------------------------===//

/// Replace the given value with a constant if the corresponding lattice
/// represents a constant. Returns success if the value was replaced, failure
/// otherwise.
LogicalResult SCCPAnalysis::replaceWithConstant(ConstantStateType &state,
                                                OpBuilder &builder,
                                                mlir::OperationFolder &folder,
                                                Value value) {
  ConstantState *lattice = getLatticeElement(value, state);
  if (!lattice || lattice->getValue().isUninitialized())
    return failure();
  const ConstantValue &latticeValue = lattice->getValue();
  if (!latticeValue.getConstantValue())
    return failure();

  // Attempt to materialize a constant for the given value.
  Dialect *dialect = latticeValue.getConstantDialect();
  Value constant = folder.getOrCreateConstant(
      builder.getInsertionBlock(), dialect, latticeValue.getConstantValue(),
      value.getType(), value.getLoc());
  if (!constant)
    return failure();

  value.replaceAllUsesWith(constant);
  return success();
}

/// Rewrite the given regions using the computing analysis. This replaces the
/// uses of all values that have been computed to be constant, and erases as
/// many newly dead operations.
LogicalResult SCCPAnalysis::rewrite(MLIRContext *context,
                                    MutableArrayRef<Region> initialRegions) {
  ConstantStateType &state = topState;
  SmallVector<Block *> worklist;
  auto addToWorklist = [&](MutableArrayRef<Region> regions) {
    for (Region &region : regions)
      for (Block &block : llvm::reverse(region))
        worklist.push_back(&block);
  };

  // An operation folder used to create and unique constants.
  mlir::OperationFolder folder(context);
  OpBuilder builder(context);

  addToWorklist(initialRegions);
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      builder.setInsertionPoint(&op);

      // Replace any result with constants.
      bool replacedAll = op.getNumResults() != 0;
      for (Value result : op.getResults())
        replacedAll &=
            succeeded(replaceWithConstant(state, builder, folder, result));

      // If all of the results of the operation were replaced, try to erase
      // the operation completely.
      if (replacedAll && wouldOpBeTriviallyDead(&op)) {
        assert(op.use_empty() && "expected all uses to be replaced");
        op.erase();
        continue;
      }

      // Add any the regions of this operation to the worklist.
      addToWorklist(op.getRegions());
    }

    // Replace any block arguments with constants.
    builder.setInsertionPointToStart(block);
    for (BlockArgument arg : block->getArguments()) {
      // Ignore replaceWithConstant result here. It's okay if the value is not a
      // constant, just don't rewrite it.
      (void)replaceWithConstant(state, builder, folder, arg);
    }
  }
  return success();
}

LogicalResult SCCPAnalysis::run(Operation *op) {
  for (Region &region : op->getRegions()) {
    bool hasEarlyExits = true;
    if (failed(processRegion(region, topState, hasEarlyExits)))
      return failure();
  }

  LLVM_DEBUG(printState(topState, llvm::dbgs()));
  return success();
}

namespace {
/// Sparse Conditional Constant Propagation (SCCP).
/// This pass conditionally propagates constant values following
/// the dataflow graph of the program while eliminating dead branches.
/// This pass doesn't have inter-procedural support (yet).
struct SCCP : impl::SCCPBase<SCCP> {
  explicit SCCP() : SCCPBase() {}

  /// Run SCCP on current operation for the pass.
  void runOnOperation() override;
};
} // namespace

void SCCP::runOnOperation() {
  SCCPAnalysis analysis;

  if (failed(analysis.run(getOperation()))) {
    signalPassFailure();
    return;
  }

  // Rewrite the IR with constant result.
  if (failed(analysis.rewrite(&getContext(), getOperation()->getRegions())))
    signalPassFailure();
}
