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

#include "GraphCompiler/MGPDialect/Transforms/RelationDataflowAnalysis.h"

#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/EquivalenceClasses.h"

using namespace M;
using namespace M::MGP;

#define DEBUG_TYPE "mgp-dataflow"

//===----------------------------------------------------------------------===//
// ForwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

void ForwardsRelationDataflowAnalysis::flowSimpleLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 1);
  Block *bodyBlock = &cfOp->getRegion(0).front();

  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the sub-block outputs back around to sub-block arguments, but
  // combine with existing results.
  for (const auto [output, arg] :
       llvm::zip(bodyBlock->getTerminator()->getOperands(),
                 bodyBlock->getArguments()))
    leastUpperBoundEntries(domainMap, arg, getEntriesAndId(domainMap, output));
}

void ForwardsRelationDataflowAnalysis::flowConditionedLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  LLVM_DEBUG(llvm::dbgs() << "flowConditionedLoop begin\n");
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();
  Block *bodyBlock = &cfOp->getRegion(1).front();

  LLVM_DEBUG(llvm::dbgs() << "flow cond block\n");
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the cond sub-block outputs forwards to body sub-block args.
  LLVM_DEBUG(llvm::dbgs() << "set body args\n");
  for (const auto [condOutput, bodyArg] :
       llvm::zip(condBlock->getTerminator()->getOperands(),
                 bodyBlock->getArguments()))
    setEntries(domainMap, bodyArg, getEntriesAndId(domainMap, condOutput));

  LLVM_DEBUG(llvm::dbgs() << "flow body block\n");
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the body sub-block outputs back around to the cond sub-block
  // arguments, but combine with existing args.
  LLVM_DEBUG(llvm::dbgs() << "set cond args\n");
  for (const auto [bodyOutput, condArg] :
       llvm::zip(bodyBlock->getTerminator()->getOperands(),
                 condBlock->getArguments()))
    leastUpperBoundEntries(domainMap, condArg,
                           getEntriesAndId(domainMap, bodyOutput));
  LLVM_DEBUG(llvm::dbgs() << "flowConditionedLoop end\n");
}

std::pair<ForwardsRelationDataflowAnalysis::Set,
          ForwardsRelationDataflowAnalysis::Set>
ForwardsRelationDataflowAnalysis::splitByScope(const Set &set, Block *block) {
  Set within = index->emptySet();
  Set without = index->emptySet();
  for (Value value : index->values(set)) {
    if (value.getDefiningOp() && block->getParent()->isAncestor(
                                     value.getDefiningOp()->getParentRegion()))
      index->addToSet(within, value);
    else
      index->addToSet(without, value);
  }
  return {within, without};
}

void ForwardsRelationDataflowAnalysis::flowBranch(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  // Flow from control flow op inputs forwards into each sub-block's args.
  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    for (auto [input, arg] :
         llvm::zip(cfOp.getInputs(), subBlock->getArguments()))
      setEntries(domainMap, arg, getEntriesAndId(domainMap, input));
    flowBlock(subBlock, domainMap, flowOp);
  }

  // Flow from combination of the outputs of each sub-block forwards to the
  // control flow op results. However, account for the mutual-exclusivity
  // between branches if requested.
  bool shouldSplit = domainMap.handleDisjointBranches && isAtMostOnce(cfOp);
  for (size_t i = 0, n = cfOp->getNumResults(); i < n; ++i) {
    Value result = cfOp->getResult(i);
    Set glbOfWithin = domainMap.top;
    Set lubOfWithout = domainMap.bottom;
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      Value output = subBlock->getTerminator()->getOperand(i);
      Set set = getEntriesAndId(domainMap, output);
      if (shouldSplit) {
        auto [within, without] = splitByScope(set, subBlock);
        domainMap.greatestLowerBound(glbOfWithin, within);
        domainMap.leastUpperBound(lubOfWithout, without);
      } else {
        domainMap.leastUpperBound(lubOfWithout, set);
      }
    }
    if (shouldSplit)
      domainMap.greatestLowerBound(lubOfWithout, glbOfWithin);
    setEntries(domainMap, result, lubOfWithout);
  }
}

/// Implements the following rules for control flow ops, where R is
/// the relation under definition.
///
/// Rules:
///  - Given
///      ... grt.if ... (... %lhs ...) {
///        // %lhs bound as %lhs1
///        ...
///      } else {
///        // %lhs bound as %lhs2
///        ...
///      }
///    then R(%lhs{1,2}, %rhs) if R(%lhs, %rhs),
///  - Given
///      ... %lhs ... = grt.if ... {
///        ...
///        mef.output ... %x1 ...
///      } else {
///        ...
///        mef.output ... %x2 ...
///      }
///    then R(%lhs, %rhs) if R(%x1, %rhs) and R(%x2, %rhs).
///  - Given
///      ... grt.repeat.i32 ... (... %x ... ) {
///        // %x bound as %lhs
///        ...
///        mef.output ... %y ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y, %lhs).
///  - Given
///      ... %lhs ... = grt.repeat.i32 ... (... %x ...) {
//         // %x bound as %x1
///        ...
///        mef.output ... %y ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y, %x1).
///  - Given
///      ... grt.while (... %x ... ) {
///        // %x bound as %lhs
///        ...
///        mef.output ... %y1 ...
///      } do {
///        // %x bound as %x2
///        ...
///        mef.output ... %y2 ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y2, %x2) and R(%y1, %lhs).
///  - Given
///      ... grt.while (... %x ... ) {
///        // %x bound as %x1
///        ...
///        mef.output ... %y1 ...
///      } do {
///        // %x bound as %lhs
///        ...
///        mef.output ... %y2 ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y2, %lhs) and R(%y1, %x1).
///  - Given
///      ... %lhs ... = grt.while (... %x ...) {
///        %x bound as %x1
///        ...
///        mef.output ... %y1 ...
///      } do {
///        %x bound as %x2
///        ...
///        mef.output ... %y2 ...
///      }
///    then R(%lhs, %rhs) if R(%x, %rhs) and R(%y1, %x1) and R(%y2, %x2).
void ForwardsRelationDataflowAnalysis::flowControlFlow(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  switch (cfOp.getFlavor()) {
  case GRT::kLoopLike: {
    LLVM_DEBUG(llvm::dbgs() << "fixed point setup...\n");
    // For 1-subregion loops we assume all traces are of the form:
    //  loop_body*
    //
    // For 2-subregion loops with approximateLoops false we assume all traces
    // are of the form:
    //  loop_cond (loop_body loop_cond)*
    //
    // If approximateLoops is true we use the trace form:
    //  (loop_cond|loop_body)*
    //
    // In all cases we tie the knot using the first sub-block's arguments.

    // Flow from control flow inputs to args of first sub-block.
    for (auto [input, arg] :
         llvm::zip(cfOp.getInputs(), cfOp->getRegions().front().getArguments()))
      setEntries(domainMap, arg, getEntriesAndId(domainMap, input));

    [[maybe_unused]] size_t n = 0;
    bool outerChanged = domainMap.changed;
    do {
      LLVM_DEBUG(llvm::dbgs() << "fixed point iteration " << ++n << "...\n");
      domainMap.changed = false;

      if (cfOp->getNumRegions() == 1) {
        // The sub-block may be executed again.
        flowSimpleLoop(cfOp, domainMap, flowOp);
      } else {
        // The second sub-block then first-sub block may be executed again.
        flowConditionedLoop(cfOp, domainMap, flowOp);
      }
    } while (domainMap.changed);
    LLVM_DEBUG(llvm::dbgs() << "fixed point reached\n");
    domainMap.changed = outerChanged;

    // Flow from args of last sub-block to control flow outputs.
    // For 2-subregion loops with approximateLoops true both the cond and
    // body blocks will have the same outputs, so this is still correct.
    LLVM_DEBUG(llvm::dbgs() << "set control flow results\n");
    for (auto [arg, result] : llvm::zip(
             cfOp->getRegions().back().getArguments(), cfOp->getResults()))
      setEntries(domainMap, result, getEntriesAndId(domainMap, arg));

    LLVM_DEBUG(llvm::dbgs() << "fixed point done\n");
    break;
  }
  case GRT::kBranchLike: {
    flowBranch(cfOp, domainMap, flowOp);
    break;
  }
  }
}

void ForwardsRelationDataflowAnalysis::flowDefault(Operation *op,
                                                   DomainMap &domainMap) {
  for (Value result : op->getResults())
    setDefault(domainMap, result);
}

void ForwardsRelationDataflowAnalysis::flowBlock(Block *block,
                                                 DomainMap &domainMap,
                                                 const FlowOp &flowOp) {

  for (Operation *op : getDataflowOrder(block))
    flowOp(op, domainMap);
}

void ForwardsRelationDataflowAnalysis::flowModel(ModelOp modelOp,
                                                 DomainMap &domainMap,
                                                 const FlowOp &flowOp) {
  Block *initBlock = modelOp.getInitBlock();
  Block *executeBlock = modelOp.getExecuteBlock();

  // Start with the 'init' block args.
  for (Value arg : initBlock->getArguments())
    setDefault(domainMap, arg);

  flowBlock(initBlock, domainMap, flowOp);

  // Flow from 'init' outputs {0, 1} to 'execute' args {0, 1}.
  for (const auto &[output, arg] :
       llvm::zip(initBlock->getTerminator()->getOperands(),
                 modelOp.getExecuteImplicitArguments()))
    setEntries(domainMap, arg, getEntriesAndId(domainMap, output));

  // Start the remaining 'execute' args.
  for (Value arg : modelOp.getExecuteInputs())
    setDefault(domainMap, arg);

  flowBlock(executeBlock, domainMap, flowOp);
}

//===----------------------------------------------------------------------===//
// BackwardsRelationDataflowAnalysis
//===----------------------------------------------------------------------===//

BackwardsRelationDataflowAnalysis::Set
BackwardsRelationDataflowAnalysis::operandCombination(
    const DomainMap &domainMap, Value value) const {
  Set set = domainMap.bottom;
  for (OpOperand &operand : value.getUses())
    domainMap.leastUpperBound(set, getEntriesAndId(domainMap, &operand));
  return set;
}

void BackwardsRelationDataflowAnalysis::flowSimpleLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  assert(cfOp->getNumRegions() == 1);
  Block *bodyBlock = &cfOp->getRegion(0).front();

  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the sub-block arguments back around to the block outputs,
  // but combine with the existing inputs.
  for (const auto &[arg, output] :
       llvm::zip(bodyBlock->getArguments(),
                 bodyBlock->getTerminator()->getOpOperands()))
    leastUpperBoundEntries(domainMap, &output,
                           operandCombination(domainMap, arg));
}

void BackwardsRelationDataflowAnalysis::flowConditionedLoop(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  LLVM_DEBUG(llvm::dbgs() << "flowConditionedLoop begin\n");
  assert(cfOp->getNumRegions() == 2);
  Block *condBlock = &cfOp->getRegion(0).front();
  Block *bodyBlock = &cfOp->getRegion(1).front();

  LLVM_DEBUG(llvm::dbgs() << "flow cond block\n");
  flowBlock(condBlock, domainMap, flowOp);

  // Flow from the cond sub-block args backwards to outputs of body block.
  LLVM_DEBUG(llvm::dbgs() << "set body output\n");
  for (const auto &[condArg, bodyOutput] :
       llvm::zip(condBlock->getArguments(),
                 bodyBlock->getTerminator()->getOpOperands()))
    setEntries(domainMap, &bodyOutput, operandCombination(domainMap, condArg));

  LLVM_DEBUG(llvm::dbgs() << "flow body block\n");
  flowBlock(bodyBlock, domainMap, flowOp);

  // Flow from the body sub-block args backwards to the cond sub-block
  // outputs, but combine with existing outputs.
  LLVM_DEBUG(llvm::dbgs() << "set cond outputs\n");
  for (const auto &[bodyArg, condOutput] :
       llvm::zip(bodyBlock->getArguments(),
                 condBlock->getTerminator()->getOpOperands()))
    leastUpperBoundEntries(domainMap, &condOutput,
                           operandCombination(domainMap, bodyArg));
  LLVM_DEBUG(llvm::dbgs() << "flowConditionedLoop end\n");
}

std::pair<BackwardsRelationDataflowAnalysis::Set,
          BackwardsRelationDataflowAnalysis::Set>
BackwardsRelationDataflowAnalysis::splitByScope(const Set &set, Block *block) {
  Set within = index->emptySet();
  Set without = index->emptySet();
  for (const OpOperand *operand : index->values(set)) {
    if (operand->get().getDefiningOp() &&
        block->getParent()->isAncestor(
            operand->get().getDefiningOp()->getParentRegion()))
      index->addToSet(within, operand);
    else
      index->addToSet(without, operand);
  }
  return {within, without};
}

void BackwardsRelationDataflowAnalysis::flowBranch(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  for (size_t i = 0, n = cfOp->getNumResults(); i < n; ++i) {
    Value result = cfOp->getResult(i);
    // Flow from control flow op results backwards into each sub-block output.
    Set set = operandCombination(domainMap, result);
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      OpOperand *output = &subBlock->getTerminator()->getOpOperand(i);
      setEntries(domainMap, output, set);
    }
  }

  for (Region &region : cfOp->getRegions()) {
    Block *subBlock = &region.front();
    flowBlock(subBlock, domainMap, flowOp);
  }

  // Flow from the combination of all sub-block arguments backwards to
  // the control flow op inputs. However, account for the mutual-exclusivity
  // between the branches if requested.
  bool shouldSplit = domainMap.handleDisjointBranches && isAtMostOnce(cfOp);
  for (size_t i = 0, n = cfOp.getInputs().size(); i < n; ++i) {
    const OpOperand *input = &cfOp.getInputOpOperands()[i];
    Set glbOfWithin = domainMap.top;
    Set lubOfWithout = domainMap.bottom;
    for (Region &region : cfOp->getRegions()) {
      Block *subBlock = &region.front();
      Set set = operandCombination(domainMap, subBlock->getArgument(i));
      if (shouldSplit) {
        auto [within, without] = splitByScope(set, subBlock);
        domainMap.greatestLowerBound(glbOfWithin, within);
        domainMap.leastUpperBound(lubOfWithout, without);
      } else {
        domainMap.leastUpperBound(lubOfWithout, set);
      }
    }
    if (shouldSplit)
      domainMap.greatestLowerBound(lubOfWithout, glbOfWithin);
    setEntries(domainMap, input, lubOfWithout);
  }
}

void BackwardsRelationDataflowAnalysis::flowControlFlow(
    GRT::GRTControlFlowInterface cfOp, DomainMap &domainMap,
    const FlowOp &flowOp) {
  if (cfOp->getNumOperands() > cfOp.getInputs().size()) {
    // Bind the loop or condition inputs.
    setDefault(domainMap, &cfOp->getOpOperand(0));
  }

  switch (cfOp.getFlavor()) {
  case GRT::kLoopLike: {
    LLVM_DEBUG(llvm::dbgs() << "fixed point setup...\n");
    // Flow from control flow op results to outputs of first sub-block.
    Operation *firstOutput = cfOp->getRegions().front().front().getTerminator();
    for (const auto &[result, output] :
         llvm::zip(cfOp->getResults(), firstOutput->getOpOperands()))
      setEntries(domainMap, &output, operandCombination(domainMap, result));
    if (firstOutput->getNumOperands() > cfOp->getNumResults()) {
      // Bind the condition outputs.
      setDefault(domainMap, &firstOutput->getOpOperands().back());
    }

    bool outerChanged = domainMap.changed;
    [[maybe_unused]] size_t n = 0;
    do {
      LLVM_DEBUG(llvm::dbgs() << "fixed point iteration " << ++n << "...\n");
      domainMap.changed = false;

      if (cfOp->getNumRegions() == 1) {
        // The sub-block may be executed again.
        flowSimpleLoop(cfOp, domainMap, flowOp);
      } else {
        // The second sub-block then first-sub block may be executed again.
        flowConditionedLoop(cfOp, domainMap, flowOp);
      }
    } while (domainMap.changed);
    LLVM_DEBUG(llvm::dbgs() << "fixed point reached\n");
    domainMap.changed = outerChanged;

    // Flow from args of first sub-block back to control flow op inputs.
    // For 2-subregion loops with approximateLoops true both the cond and
    // body blocks will have the same args, so this is still correct.
    LLVM_DEBUG(llvm::dbgs() << "set control flow inputs\n");
    for (const auto &[arg, input] :
         llvm::zip(cfOp->getRegions().front().getArguments(),
                   cfOp.getInputOpOperands()))
      setEntries(domainMap, &input, operandCombination(domainMap, arg));

    LLVM_DEBUG(llvm::dbgs() << "fixed point done\n");

    break;
  }
  case GRT::kBranchLike: {
    flowBranch(cfOp, domainMap, flowOp);
    break;
  }
  }
}

void BackwardsRelationDataflowAnalysis::flowDefault(Operation *op,
                                                    DomainMap &domainMap) {
  if (isa<MEF::OutputOp>(op))
    // entries for the output op operands are established from the outer block.
    return;

  for (const OpOperand &operand : op->getOpOperands())
    setDefault(domainMap, &operand);
}

void BackwardsRelationDataflowAnalysis::flowBlock(Block *block,
                                                  DomainMap &domainMap,
                                                  const FlowOp &flowOp) {
  SmallVector<Operation *> ops = getDataflowOrder(block);
  for (Operation *op : llvm::reverse(ops))
    flowOp(op, domainMap);
}

void BackwardsRelationDataflowAnalysis::flowModel(ModelOp modelOp,
                                                  DomainMap &domainMap,
                                                  const FlowOp &flowOp) {
  Block *initBlock = modelOp.getInitBlock();
  Block *execBlock = modelOp.getExecuteBlock();

  // Start with the 'execute' block outputs.
  for (const OpOperand &output : execBlock->getTerminator()->getOpOperands())
    setDefault(domainMap, &output);

  flowBlock(execBlock, domainMap, flowOp);

  // Flow from 'execute' args {0, 1} backwards to 'init' outputs {0, 1}.
  for (const auto &[arg, output] :
       llvm::zip(modelOp.getExecuteChainContext(),
                 initBlock->getTerminator()->getOpOperands()))
    setEntries(domainMap, &output, operandCombination(domainMap, arg));

  // A device the init output only passes through must be set empty, not as a
  // regular entry.
  for (const auto &[arg, output] :
       llvm::zip(modelOp.getExecuteDevices(),
                 initBlock->getTerminator()->getOpOperands().drop_front(
                     kMgpModelNumFixedImplicitExecArgs))) {
    if (output.get().getDefiningOp())
      setEntries(domainMap, &output, operandCombination(domainMap, arg));
    else
      setEmpty(domainMap, &output);
  }

  // Set the weights registry to empty since the init output is its only user.
  setEmpty(domainMap,
           &initBlock->getTerminator()
                ->getOpOperands()[kMgpModelNumFixedImplicitExecArgs - 1]);

  flowBlock(initBlock, domainMap, flowOp);
}
