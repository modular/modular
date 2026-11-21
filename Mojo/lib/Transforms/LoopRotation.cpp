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
//
// Loop rotation, shared by the `rotate-loops` pass and `raise-for-loops`.
//
// An `hlcf.loop` leaves through an exiting test: a condition in the body
// guarding an `hlcf.break`. A top test is one that comes first in the body, so
// the loop decides before any of the iteration runs.
//
//   top test                            | not a top test
//   hlcf.loop (%i = %n : index) {       | hlcf.loop (%i = %n : index) {
//     %t = index.cmp eq(%i, %idx0)      |   pop.store %i, %q
//     hlcf.if %t { hlcf.break }         |   %t = index.cmp eq(%i, %idx0)
//          else { hlcf.yield }          |   hlcf.if %t { hlcf.break }
//     pop.store %i, %q                  |        else { hlcf.yield }
//     %next = index.sub %i, %idx1       |   %next = index.sub %i, %idx1
//     hlcf.continue %next : index       |   hlcf.continue %next : index
//   }                                   | }
//
// The store runs %n times on the left and %n + 1 times on the right, since the
// iteration that breaks has already run the right-hand store.
//
// Rotation is a canonicalization, not an optimization: it saves no work. It
// gives every loop a top test, which `raise-for-loops` needs because an
// `hlcf.for` checks its bound before the body. Without it, each shape of
// `hlcf.loop` would need its own check that raising it is still correct.
//
// Rotation makes the test lead by duplicating what precedes it: one copy
// ahead of the loop, one at the end of the body. The number of times each op
// runs does not change, so the loop keeps its effects.
//
//   before                              | after
//   hlcf.loop (%i = %n : index) {       | %v = pop.load %p
//     %v = pop.load %p                  | hlcf.loop (%i = %n, %c = %v) {
//     %t = index.cmp eq(%i, %idx0)      |   %t = index.cmp eq(%i, %idx0)
//     hlcf.if %t { hlcf.break }         |   hlcf.if %t { hlcf.break }
//          else { hlcf.yield }          |        else { hlcf.yield }
//     pop.store %v, %q                  |   pop.store %c, %q
//     %next = index.sub %i, %idx1       |   %next = index.sub %i, %idx1
//     hlcf.continue %next : index       |   %v2 = pop.load %p
//   }                                   |   hlcf.continue %next, %v2
//                                       | }
//
// What the moved ops define becomes loop-carried: the copy ahead of the loop
// gives the first iteration's value, the copy at the end of the body the next
// one's, and everything left behind reads the block argument in between.
//
// LLVM's loop rotation (`llvm/lib/Transforms/Utils/LoopRotationUtils.cpp`)
// duplicates the same way but runs the other direction: it moves a top test
// down to the latch, which a CFG prefers for its one branch per iteration.
// `hlcf.for` checks its bound before the body, so a top test is the only shape
// it can express and the rotation has to go that way.
//
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/LoopUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace M;
using namespace HLCF;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_ROTATELOOPS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

/// Return whether an op ahead of the exiting test must not be duplicated.
static bool blocksRotation(Operation *op) {
  // An allocation names one object, and its lifetime markers require an
  // operand that it defines, which a loop-carried copy is not.
  if (isa<POP::StackAllocationOp>(op))
    return true;

  // A jump to a loop outside `op` would run from the copy ahead of the loop,
  // leaving that outer loop before this one has started. A nested loop whose
  // own jumps stay inside it copies fine.
  bool escapes = false;
  op->walk([&](Operation *nested) {
    StringAttr label;
    if (auto breakOp = dyn_cast<BreakOp>(nested))
      label = breakOp.getLabelAttr();
    else if (auto continueOp = dyn_cast<ContinueOp>(nested))
      label = continueOp.getLabelAttr();
    else
      return;
    escapes |= !op->isAncestor(getParentLoop(nested, label));
  });
  return escapes;
}

/// Move the exiting test to the top of the body, as the file comment describes.
static LogicalResult rotate(LoopOp loop, IfOp ifOp,
                            mlir::RewriterBase &rewriter) {
  Block &body = loop.getBody().front();

  // Two static copies of a convergent op no longer bring the same threads
  // together; `infer-function-attrs` has marked such a function already.
  auto func = loop->getParentOfType<KGEN::FuncOp>();
  if (func && func.isConvergent())
    return failure();

  // What the test reads stays put, and so does anything droppable, a debug
  // record included. The rest is what moves.
  llvm::SmallPtrSet<Operation *, 8> testChain;
  SmallVector<Value> worklist{ifOp.getCond()};
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || def->getBlock() != &body || !def->isBeforeInBlock(ifOp))
      continue;
    if (testChain.insert(def).second)
      llvm::append_range(worklist, def->getOperands());
  }

  SmallVector<Operation *> leading;
  llvm::SmallPtrSet<Operation *, 8> rotatedSet;
  SmallVector<Operation *> rotated;
  for (Operation &op : llvm::make_range(body.begin(), ifOp->getIterator())) {
    leading.push_back(&op);
    if (testChain.contains(&op) || KGEN::isDroppableOnExitIteration(&op))
      continue;
    if (blocksRotation(&op))
      return failure();
    rotated.push_back(&op);
    rotatedSet.insert(&op);
  }
  if (rotated.empty())
    return success();
  // Duplicating is the cost, so it is bounded, as LLVM bounds its own. A
  // nested loop costs what it holds, not one.
  unsigned opCount = 0;
  for (Operation *op : rotated)
    op->walk([&](Operation *) { ++opCount; });
  if (opCount > *KGENPassCLOptions::rotateLoopsMaxOps())
    return failure();

  // The rest moves past the test, which must not observe it doing so.
  for (Operation *op : testChain) {
    if (!mlir::isMemoryEffectFree(op))
      return failure();
  }

  auto continueOp = cast<ContinueOp>(body.getTerminator());

  // A mover may read an op that stays, a cast of the induction variable say,
  // so the copies take that one along. Nothing else is copied: a debug record
  // belongs to the iteration it describes, and a copy of anything the movers
  // do not read would just be dead.
  llvm::SmallPtrSet<Operation *, 8> copied(rotatedSet.begin(),
                                           rotatedSet.end());
  for (Operation *op : llvm::reverse(leading)) {
    if (!copied.contains(op))
      continue;
    // A region reads through its own ops, so the walk goes inside: what one of
    // them names has to be copied too, or the copy ahead of the loop would
    // name a value the loop defines.
    op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands()) {
        if (Operation *def = operand.getDefiningOp())
          copied.insert(def);
      }
    });
  }

  auto copyLeadingOps = [&](IRMapping &map) {
    for (Operation *op : leading) {
      if (copied.contains(op))
        rewriter.clone(*op, map);
    }
  };

  // The first copy reads the values the loop starts with.
  rewriter.setInsertionPoint(loop);
  IRMapping peeled;
  for (auto [arg, init] : llvm::zip(body.getArguments(), loop.getOperands()))
    peeled.map(arg, init);
  copyLeadingOps(peeled);

  // An op that stays keeps reading this iteration's value, so what it reads
  // becomes loop-carried. That covers the body behind the test, and equally a
  // debug record or a pure consumer left ahead of it.
  auto staysInBody = [&](OpOperand &use) {
    Operation *inBody = body.findAncestorOpInBlock(*use.getOwner());
    return inBody && !rotatedSet.contains(inBody);
  };

  SmallVector<Value> carried;
  for (Operation *op : rotated) {
    for (Value result : op->getResults()) {
      if (llvm::any_of(result.getUses(), [&](OpOperand &use) {
            return staysInBody(const_cast<OpOperand &>(use));
          }))
        carried.push_back(result);
    }
  }

  SmallVector<Value> inits;
  DenseMap<Value, BlockArgument> carriedArgs;
  for (Value value : carried) {
    // The op has effects, so its copy was kept rather than dropped.
    Value init = peeled.lookup(value);
    inits.push_back(init);
    BlockArgument arg = body.addArgument(value.getType(), value.getLoc());
    carriedArgs[value] = arg;
    // The movers keep reading each other: the second copy comes from them.
    rewriter.replaceUsesWithIf(value, arg, staysInBody);
  }
  loop->insertOperands(loop->getNumOperands(), inits);

  // The second copy computes the next iteration, so it reads what
  // `hlcf.continue` carries back.
  rewriter.setInsertionPoint(continueOp);
  IRMapping next;
  for (auto [arg, carriedBack] :
       llvm::zip(body.getArguments(), continueOp.getOperands()))
    next.map(arg, carriedBack);
  // An op that stays ahead of the test now reads the block argument a mover's
  // result became, which holds this iteration's value. Its copy here reads the
  // next iteration's instead: the copy of that mover, made just before it.
  for (Operation *op : leading) {
    if (!copied.contains(op))
      continue;
    rewriter.clone(*op, next);
    for (Value result : op->getResults()) {
      if (BlockArgument arg = carriedArgs.lookup(result))
        next.map(arg, next.lookup(result));
    }
  }

  SmallVector<Value> yielded;
  for (Value value : carried)
    yielded.push_back(next.lookup(value));
  continueOp->insertOperands(continueOp->getNumOperands(), yielded);

  for (Operation *op : llvm::reverse(rotated))
    rewriter.eraseOp(op);

  // An op that stayed only to feed a mover has no reader left, and the copies
  // compute it for the iterations that need it.
  for (Operation *op : llvm::reverse(leading)) {
    if (!rotatedSet.contains(op) && op->use_empty() &&
        mlir::isMemoryEffectFree(op))
      rewriter.eraseOp(op);
  }
  return success();
}

/// Return the `hlcf.if` holding the loop's only exit, when that exit is a
/// child of the body and the only way back to the top is the terminator.
///
/// A nested loop's jumps are its own, so the walk asks each one which loop it
/// names rather than stopping at the nested loop: a labelled break reaches
/// past it and out of this one.
static IfOp findExitingTest(LoopOp loop) {
  Block &body = loop.getBody().front();
  if (!isa<ContinueOp>(body.getTerminator()))
    return {};

  BreakOp exit;
  bool rejected = false;
  loop.getBody().walk([&](Operation *op) {
    if (auto breakOp = dyn_cast<BreakOp>(op)) {
      if (getParentLoop(breakOp, breakOp.getLabelAttr()) != loop)
        return;
      // A second exit leaves on a path rotation does not duplicate.
      rejected |= static_cast<bool>(exit);
      exit = breakOp;
    } else if (auto contOp = dyn_cast<ContinueOp>(op)) {
      if (contOp.getOperation() == body.getTerminator())
        return;
      rejected |= getParentLoop(contOp, contOp.getLabelAttr()) == loop;
    }
  });
  if (rejected || !exit)
    return {};

  auto ifOp = dyn_cast<IfOp>(exit->getParentOp());
  if (!ifOp || ifOp.getNumRegions() != 2 || ifOp->getBlock() != &body)
    return {};
  return ifOp;
}

namespace {
struct RotateLoops : impl::RotateLoopsBase<RotateLoops> {
  using RotateLoopsBase::RotateLoopsBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](LoopOp loop) {
      if (IfOp ifOp = findExitingTest(loop))
        (void)rotate(loop, ifOp, rewriter);
    });
  }
};

} // namespace
