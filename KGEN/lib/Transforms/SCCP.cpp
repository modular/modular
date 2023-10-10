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
  /// Constructor
  explicit SCCPAnalysis(unsigned optimizationLevel)
      : optimizationLevel(optimizationLevel) {}

  /// ConstantValue lattice element type.
  using ConstantState = Lattice<ConstantValue>;

  /// Map type from Value to ConstantValue lattice element.
  using ConstantStateType = DenseMap<Value, ConstantState>;

  /// Process a Region operation.
  LogicalResult processRegion(Region &region, ConstantStateType &state,
                              bool &shouldJoin,
                              bool setBlockArgToEntryState = true);

  /// Helper function to rewrite the IR with SCCP analysis results.
  static void rewrite(ConstantStateType &state, MLIRContext *context,
                      MutableArrayRef<Region> initialRegions);

  /// State for the top operation that the analysis starts from.
  ConstantStateType topState;

private:
  /// ConstantValue lattice states for ControlFlow type of nodes.
  struct ControlFlowOperationState {
    std::queue<SmallVector<Attribute>> entryStates;
    ConstantStateType exitStates;
  };

  /// Process a ControlFlowNode operation.
  void processControlFlowNode(HLCF::ControlFlowNode node,
                              ConstantStateType &state, bool &shouldContinue);

  /// Process a ControlFlowTerminator operation.
  void processControlFlowTerminator(HLCF::ControlFlowTerminator term,
                                    ConstantStateType &state);

  /// Visit a general operation to apply the transform function.
  LogicalResult visitOperation(mlir::Operation *op, ConstantStateType &state);

  /// Get the lattice element for value from state.
  static ConstantState *getLatticeElement(Value value,
                                          ConstantStateType &state);

  /// Get the lattice elements for op's operands.
  static void getOperandsLattice(SmallVectorImpl<Attribute> &attributes,
                                 Operation *op, ConstantStateType &state);

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
  static void updateParentOpOutputState(HLCF::ControlFlowTerminator term,
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

  /// Compiler optimization level.
  unsigned optimizationLevel;
};

} // namespace

ChangeResult SCCPAnalysis::joinStates(ConstantStateType &lhs,
                                      ConstantStateType &rhs) {
  ChangeResult changed = ChangeResult::NoChange;
  for (auto &[value, lattice] : rhs) {
    auto iter = lhs.find(value);
    if (iter != lhs.end())
      changed |= iter->getSecond().join(lattice);
  }
  return changed;
}

ChangeResult SCCPAnalysis::mergeStates(SCCPAnalysis::ConstantStateType &current,
                                       SCCPAnalysis::ConstantStateType &state) {
  ChangeResult changed = ChangeResult::NoChange;
  for (auto &[value, lattice] : state) {
    ConstantState *lattice0 = getLatticeElement(value, current);
    changed |= lattice0->join(lattice);
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

void SCCPAnalysis::getOperandsLattice(SmallVectorImpl<Attribute> &attributes,
                                      Operation *op, ConstantStateType &state) {
  for (Value value : op->getOperands()) {
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

LogicalResult SCCPAnalysis::visitOperation(mlir::Operation *op,
                                           ConstantStateType &state) {
  SmallVector<Attribute> constantOperands;
  getOperandsLattice(constantOperands, op, state);

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
    return success();
  }

  // If the folding was in-place, mark the results as overdefined and reset
  // the operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (foldResults.empty()) {
    op->setOperands(originalOperands);
    op->setAttrs(originalAttrs);
    setAllToEntryStates(results);
    return success();
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

  return success();
}

int64_t SCCPAnalysis::getLoopConvergeThreshold(Operation *op) {
  // TODO: use decorator or more sophisticated heuristics to set per loop
  // threshold.
  if (optimizationLevel < 2)
    return 100;
  else
    return 200;
}

void SCCPAnalysis::processControlFlowNode(HLCF::ControlFlowNode node,
                                          ConstantStateType &state,
                                          bool &shouldContinue) {
  // TODO: Add support for other ControlFlowNode, e.g. hlcf.for, etc.
  // TODO: This function should work more generally for ControlFlowInterfaces.
  if (auto ifOp = dyn_cast<HLCF::IfOp>(node.getOperation())) {
    // TODO: extend this logic to SwitchOp.
    SmallVector<Attribute> constantOperands;
    getOperandsLattice(constantOperands, node.getOperation(), state);
    SmallVector<HLCF::ControlFlowTarget> targets;
    node.getEntryTargets(constantOperands, targets);

    // Preserve the entry state.
    ConstantStateType entryState = state;
    for (HLCF::ControlFlowTarget target : targets) {
      if (target.index) {
        // Analyze region with entry state.
        ConstantStateType nestedState = entryState;
        bool shouldJoin = true;
        (void)processRegion(node->getRegion(target.index.value()), nestedState,
                            shouldJoin);

        // Directly merge nestedState with state here since we don't need to
        // reset local value's lattice.
        mergeStates(state, nestedState);
        if (!shouldJoin && targets.size() == 1) {
          // break happened, we can break traversal if this is the only branch
          // that is running.
          shouldContinue = false;
        }
      }
    }
    mergeStates(state, controlFlowOperationStates[ifOp].exitStates);
    controlFlowOperationStates.erase(ifOp);
    return;
  }

  if (HLCF::LoopOp loop = dyn_cast<HLCF::LoopOp>(node.getOperation())) {
    // Prepare for initial loop inputs.
    SmallVector<Attribute> constantOperands;
    getOperandsLattice(constantOperands, node.getOperation(), state);

    // Prepare the workList for analyzing the loop.
    ControlFlowOperationState &cfStates = controlFlowOperationStates[loop];
    std::queue<SmallVector<Attribute>> &workList = cfStates.entryStates;
    workList.push(constantOperands);

    int64_t iter = 0;
    ConstantStateType currState = state;
    ConstantStateType mergedState = state;

    SmallVector<Attribute> &inputValues = currentLoopInputs[loop];

    while (!workList.empty() &&
           iter < getLoopConvergeThreshold(node.getOperation())) {
      inputValues = std::move(workList.front());
      workList.pop();

      ConstantStateType nestedState = currState;
      bool shouldJoin = true;
      // Prepare for input arguments for this iteration.
      for (auto [inputValue, blockArg] :
           llvm::zip(inputValues, loop.getRegion().front().getArguments())) {
        ConstantState *lattice = getLatticeElement(blockArg, nestedState);
        if (!inputValue)
          setToEntryState(lattice);
        else
          lattice->join(ConstantValue(inputValue, loop->getDialect()));
      }

      // Process loop body.
      (void)processRegion(loop.getRegion(), nestedState, shouldJoin, false);

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
    controlFlowOperationStates.erase(loop);
    currentLoopInputs.erase(loop);
    return;
  }

  // Otherwise, mark all results as Unknown.
  for (Value r : node.getOperation()->getResults()) {
    setToEntryState(getLatticeElement(r, state));
  }
}

void SCCPAnalysis::updateParentOpOutputState(
    HLCF::ControlFlowTerminator term, Operation *parentOp,
    ConstantStateType &termState, ConstantStateType &parentOutputState) {
  for (auto [operand, opResult] :
       llvm::zip(term.getOperation()->getOperands(), parentOp->getResults())) {
    ConstantState *lattice0 = getLatticeElement(operand, termState);
    ConstantState *lattice1 = getLatticeElement(opResult, parentOutputState);
    lattice1->join(*lattice0);
  }
}

void SCCPAnalysis::processControlFlowTerminator(
    HLCF::ControlFlowTerminator term, ConstantStateType &state) {

  // TODO: Add support for other ControlFlowTerminators, e.g. hlcf.for.yield,
  // kgen.return, etc.
  if (auto breakOp = dyn_cast<HLCF::BreakOp>(term.getOperation())) {
    Operation *parentLoop = HLCF::getParentNode(term);
    // Update parent loop's exit state.
    ControlFlowOperationState &states = controlFlowOperationStates[parentLoop];
    ConstantStateType &outputState = states.exitStates;
    updateParentOpOutputState(term, parentLoop, state, outputState);
    return;
  }

  if (auto continueOp = dyn_cast<HLCF::ContinueOp>(term.getOperation())) {
    auto parentLoop = HLCF::getParentNode(term);
    // Prepare new inputs for parent loop.
    SmallVector<Attribute> constantOperands;
    getOperandsLattice(constantOperands, term.getOperation(), state);

    ControlFlowOperationState &states = controlFlowOperationStates[parentLoop];

    // Only push the new inputs if it is different from current one.
    if (constantOperands != currentLoopInputs[parentLoop])
      states.entryStates.push(constantOperands);

    return;
  }

  if (auto yieldOp = dyn_cast<HLCF::YieldOp>(term.getOperation())) {
    Operation *parentOp = HLCF::getParentNode(term);

    ControlFlowOperationState &states = controlFlowOperationStates[parentOp];
    // update parent op's exit state
    ConstantStateType &outputState = states.exitStates;
    updateParentOpOutputState(term, parentOp, state, outputState);
    return;
  }

  // Otherwise, mark all results as Unknown.
  for (Value r : term.getOperation()->getResults()) {
    setToEntryState(getLatticeElement(r, state));
  }
}

LogicalResult SCCPAnalysis::processRegion(Region &region,
                                          ConstantStateType &state,
                                          bool &shouldJoin,
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
    if (auto node = dyn_cast<HLCF::ControlFlowNode>(op)) {
      bool shouldContinue = true;
      processControlFlowNode(node, state, shouldContinue);
      if (!shouldContinue)
        break;
      continue;
    }

    if (auto term = dyn_cast<HLCF::ControlFlowTerminator>(op)) {
      processControlFlowTerminator(term, state);
      shouldJoin = !(isa<HLCF::ContinueOp>(op) || isa<HLCF::BreakOp>(op) ||
                     isa<KGEN::UnreachableOp>(op));
      break;
    }

    if (op.getNumRegions() > 0) {
      ConstantStateType entryState = state;
      for (Region &r : op.getRegions()) {
        ConstantStateType nestedState = entryState;
        bool shouldJoin = true;
        if (failed(processRegion(r, nestedState, shouldJoin)))
          return failure();
        mergeStates(state, nestedState);
      }
      continue;
    } else {
      (void)visitOperation(&op, state);
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
void SCCPAnalysis::rewrite(ConstantStateType &state, MLIRContext *context,
                           MutableArrayRef<Region> initialRegions) {
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
      for (Value res : op.getResults())
        replacedAll &=
            succeeded(replaceWithConstant(state, builder, folder, res));

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
    for (BlockArgument arg : block->getArguments())
      (void)replaceWithConstant(state, builder, folder, arg);
  }
}

/// Print lattice content for debugging.
static void printState(SCCPAnalysis::ConstantStateType &state,
                       llvm::raw_ostream &os) {
  for (auto iter : state) {
    os << "============================\n";
    os << "value: " << iter.getFirst() << "\n";
    iter.getSecond().print(os);
    os << "\n";
  }
}

namespace {
/// Sparse Conditional Constant Propagation (SCCP).
/// This pass conditionally propagates constant values following
/// the dataflow graph of the program while eliminating dead branches.
/// This pass doesn't have inter-procedural support (yet).
struct SCCP : impl::SCCPBase<SCCP> {
  explicit SCCP(const SCCPOptions &options = {}) : SCCPBase(options) {}

  /// Run SCCP on current operation for the pass.
  void runOnOperation() override;
};
} // namespace

void SCCP::runOnOperation() {
  SCCPAnalysis analysis(optimizationLevel);

  for (Region &region : getOperation()->getRegions()) {
    bool shouldJoin = true;
    if (failed(analysis.processRegion(region, analysis.topState, shouldJoin))) {
      signalPassFailure();
      return;
    }
  }

  LLVM_DEBUG(printState(analysis.topState, llvm::dbgs()));

  // Rewrite the IR with constant result.
  analysis.rewrite(analysis.topState, &getContext(),
                   getOperation()->getRegions());
}
