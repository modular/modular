//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers 'semantic' control flow into more structured control flow.
//
// This performs these lowerings:
//  1) It lowers statements like `lit.break` (which is not a terminator) into
//     `hlcf.break` (which is), and deletes unreachable code after it, and
//     diagnoses it if it is anything interesting.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/STLExtras.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Semantic control flow lowering.
//===----------------------------------------------------------------------===//

namespace {
/// Each function is lowered in a depth first walk through the region tree.
struct LowerSemanticCF {
  LIT::FuncOp theFunc;
  SymbolRefAttr theFuncSymbol;

  // This is the current loop that a break or continue should exit from.
  Operation *currentLoop = nullptr;

  // Each lowered hlcf.loop gets its own unique ID so we can break out of it if
  // needed.
  unsigned loopCounter = 0;

  // True if we've emitted an error.
  bool hadError = false;

  // When lowering control flow, notice any recursive calls to diagnose infinite
  // recursion after the IR is validated and rewritten.
  bool hasRecursiveCalls = false;

  LowerSemanticCF(LIT::FuncOp theFunc) : theFunc(theFunc) {
    if (!theFunc.isOptionalSymbol())
      theFuncSymbol = LIT::getFullyResolvedSymbolRef(theFunc);
  }
  void run();

private:
  void lowerBlock(Block &block, bool &doesRaise, bool &doesBreak,
                  bool &doesFallThrough);
  bool lowerLITLoop(LIT::LoopOp loopOp, bool &enclosingBlockDoesRaise,
                    bool &enclosingBlockDoesBreak);
  void lowerElif(HLCF::ElifOp elifOp, bool &doesRaise, bool &doesBreak,
                 bool &doesFallThrough);
  bool checkSelfRecursion(Block &block, bool isConditional);
};
} // end anonymous namespace

/// Mangle a ParamDeclAttr or ParamDeclRefAttr during cloning of finally blocks.
/// This scheme postpends "f{cnt-1}" to the param name, which guarantees
/// uniqueness, since the parameters we started with must already be unique
/// within their scope. It also retains the demangling rule assumed by the stack
/// because removing everything after the backtick still yields the param name.
template <typename DeclOrRef>
static DeclOrRef mangle(DeclOrRef declOrRef, size_t cnt,
                        const SmallPtrSet<StringAttr, 4> &needToMangle,
                        DenseMap<Attribute, Attribute> &manglingCache) {
  StringAttr name = declOrRef.getName();
  // If counter is zero, we don't mangle to improve readability.
  if (cnt-- == 0 || !needToMangle.contains(name))
    return declOrRef;

  // Check the cache here instead of straightforward memoization so that we
  // limit memory footprint.
  if (Attribute cached = manglingCache.lookup(declOrRef))
    return cast<DeclOrRef>(cached);

  auto mangledName = StringAttr::get(name.getContext(),
                                     name.strref() + Twine("f") + Twine(cnt));
  auto mangledDecl = DeclOrRef::get(mangledName, declOrRef.getType());
  manglingCache.try_emplace(declOrRef, mangledDecl);
  return mangledDecl;
}

/// Insert 'finally' block logic on a `lit.try` operation by finding all
/// terminators that exit the try regions and pasting the finally clause before
/// it. Try operations must be processed post-order, so that the order in which
/// the finally clauses are pasted is correct.
static LIT::TryOp lowerTryFinally(LIT::TryOp tryOp) {
  Block &finallyBlock = tryOp.getFinallyRegion().front();

  // While cloning the finally block, we need to ensure parameter names are kept
  // unique. So we first collect parameter names that have to be mangled. Decls
  // nested within another decl scope can be ignored safely, but everything else
  // we need to remember.
  SmallPtrSet<StringAttr, 4> needToMangle;
  finallyBlock.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto paramOp = dyn_cast<ParamOpInterface>(op)) {
      paramOp.walkDeclarations(
          [&](ParamDeclAttr decl) { needToMangle.insert(decl.getName()); });
    }
    if (isa<DeclInterface>(op))
      return WalkResult::skip();
    return WalkResult::advance();
  });

  // We count how many times we cloned the block and use this for mangling.
  size_t finallyCount = 0;
  auto pasteFinally = [&](Operation *term) {
    OpBuilder b(term);
    IRMapping map;

    // Set up mangling utilities.
    DenseMap<Attribute, Attribute> manglingCache;
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](ParamDeclAttr decl) {
      return mangle(decl, finallyCount, needToMangle, manglingCache);
    });
    replacer.addReplacement([&](ParamDeclRefAttr ref) {
      return mangle(ref, finallyCount, needToMangle, manglingCache);
    });

    for (Operation &op : finallyBlock.without_terminator()) {
      Operation *cloned = b.clone(op, map);

      // We mangle the parameter names in the op. This must happen recursively
      // for all ops, since references can be deeply nested.
      replacer.recursivelyReplaceElementsIn(cloned, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/true,
                                            /*replaceTypes=*/true);
    }
    // If the finally block terminator exits, then the current terminator is
    // dead code.
    if (!isa<LIT::TryYieldOp>(finallyBlock.getTerminator())) {
      b.clone(*finallyBlock.getTerminator(), map);
      term->erase();
    }
    ++finallyCount;
  };

  // FIXME: Re-traversing `lit.try` operations is N^2. This could be computed
  // in one pass over the IR.
  auto checkRegion = [&](Operation *op) {
    // Control-flow will never cross nested functions.
    if (isa<LIT::FuncOp>(op))
      return WalkResult::skip();

    // Check for a terminator that will branch past the enclosing try operation.
    auto term = dyn_cast<HLCF::ControlFlowTerminator>(op);
    if (!term)
      return WalkResult::advance();
    Operation *node = HLCF::getParentNode(term);
    if (node->isProperAncestor(tryOp))
      pasteFinally(term);
    return WalkResult::advance();
  };

  // Route exiting branches from the 'try', 'except', and 'else' regions through
  // the finally region. Nothing to do if the 'finally' region is trivial.
  if (!tryOp.hasTrivialFinally()) {
    tryOp.getTryRegion().walk(checkRegion);
    tryOp.getExceptRegion().walk(checkRegion);
    tryOp.getElseRegion().walk(checkRegion);
    // Paste the finally block at the exits of the else and except regions if
    // they are not terminated by an exit.
    if (auto yield = dyn_cast<LIT::TryYieldOp>(
            tryOp.getExceptRegion().front().getTerminator()))
      pasteFinally(yield);
    if (auto yield = dyn_cast<LIT::TryYieldOp>(
            tryOp.getElseRegion().front().getTerminator()))
      pasteFinally(yield);
  }

  // Clear the finally region by rebuilding the operation without it.
  OperationState state(tryOp.getLoc(), tryOp->getName());
  for (unsigned i = 0; i < 3; ++i) {
    state.regions.emplace_back(std::make_unique<Region>())
        ->takeBody(tryOp->getRegion(i));
  }
  OpBuilder b(tryOp);
  auto newTry = cast<LIT::TryOp>(b.create(state));
  tryOp.erase();
  return newTry;
}

/// Erase operations to the end of the block after op.
static void eraseOpToEndOfBlock(Operation *op) {
  Block *block = op->getBlock();
  // Erase bottom up to avoid deleting an op while something uses its results.
  while (&block->back() != op)
    block->back().erase();
  op->erase();
}

/// Given a semantic terminator, diagnose and remove unreachable code, and
/// return a builder at the right spot to insert a replacement.
static ImplicitLocOpBuilder handleSemanticTerminatorOp(Operation &op,
                                                       StringRef stmtKind) {
  // Warn about dead code after the semantic terminator.
  Operation *nextOp = op.getNextNode();
  // We do report an error on `parameter if` since `parameter if` serves as a
  // if preprocessor in Mojo.
  if (!isa<ParamIfOp>(op) && !nextOp->hasTrait<OpTrait::IsTerminator>()) {
    // Don't complain if the location is the same as the enclosing function,
    // it is automatically synthesized.
    auto funcOp = nextOp->getParentOfType<LIT::FuncOp>();
    if (!funcOp || funcOp->getLoc() != nextOp->getLoc())
      emitWarning(nextOp->getLoc(), "unreachable code after ") << stmtKind;
  }

  // Remove the unreachable code.
  eraseOpToEndOfBlock(nextOp);
  // Return a builder pointing to after "op".
  return ImplicitLocOpBuilder(op.getLoc(), op.getBlock(),
                              std::next(Block::iterator(&op)));
};

void LowerSemanticCF::lowerElif(HLCF::ElifOp elifOp, bool &doesRaise,
                                bool &doesBreak, bool &doesFallThrough) {
  bool elifFallsThrough = false;
  for (auto &region : elifOp->getRegions()) {
    if (region.empty())
      continue;
    bool blockRaises = false, blockBreaks = false, blockFallThroughs = false;
    lowerBlock(region.front(), blockRaises, blockBreaks, blockFallThroughs);
    doesRaise |= blockRaises;
    doesBreak |= blockBreaks;
    // Condition regions are odd indexed regions and always fallthrough to elif
    // contained regions.
    if (region.getRegionNumber() % 2 == 1)
      continue;
    elifFallsThrough |= blockFallThroughs;
  }
  doesFallThrough = elifFallsThrough;
  if (!doesFallThrough) {
    auto b = handleSemanticTerminatorOp(
        *elifOp.getOperation(),
        "if statement with then/else that do not fall through");
    b.create<UnreachableOp>(elifOp.getLoc());
  }
}

/// Lower a LIT::LoopOp to HLCF::LoopOp.  Return true if the lowering should
/// stop traversing the rest of the operations because this is an infinite loop
/// that doesn't fall through.
bool LowerSemanticCF::lowerLITLoop(LIT::LoopOp loopOp,
                                   bool &enclosingBlockDoesRaise,
                                   bool &enclosingBlockDoesBreak) {
  // Lower loop conditions.
  Block &condBlock = loopOp.getCondRegion().front();
  Block &bodyBlock = loopOp.getBodyRegion().front();
  Block &elseBlock = loopOp.getElseRegion().front();

  // Create the new HLCF::LoopOp.
  ImplicitLocOpBuilder builder(loopOp->getLoc(), loopOp);
  builder.setInsertionPointAfter(loopOp);
  auto newLoop = builder.create<HLCF::LoopOp>();
  // Each loop gets a unique label.
  newLoop.setLabelAttr(builder.getStringAttr("_loop_" + Twine(loopCounter++)));

  Block *newBody = builder.createBlock(&newLoop.getBody());

  // Move the loop condition logic to the beginning of the HLCF::LoopOp's body.
  newBody->getOperations().splice(newBody->begin(), condBlock.getOperations());

  // Create the loop condition check, replacing the LoopConditionOp.
  auto loopCondition =
      cast<LIT::LoopConditionOp>(newBody->getTerminator()).getOperand();
  newBody->getTerminator()->erase();

  // Create loop condition check: it continues when the condition is true and
  // does the the exit logic when false.
  builder.setInsertionPointToEnd(newBody);
  auto condOp = builder.create<HLCF::IfOp>(loopCondition);
  builder.createBlock(&condOp.getThenRegion());
  builder.create<HLCF::YieldOp>();

  // Move the loop's body to the HLCF::LoopOp's body.
  newBody->getOperations().splice(newBody->end(), bodyBlock.getOperations());

  // Move any 'else' code into the exit block.  If the 'else' code falls through
  // then it will break out of the loop, for now we leave it ending with
  // lit.loop.yield.
  Block *newExitBlock = builder.createBlock(&condOp.getElseRegion());
  newExitBlock->getOperations().splice(newExitBlock->end(),
                                       elseBlock.getOperations());

  // Now that the code is set up right, we can recursively lower any semantic
  // control flow ops.  Start by lowering the 'else' block since it is logically
  // NOT inside the loop even though it is nested under it in the AST.  The
  // 'currentLoop' loop is set to the parent loop so any break or continue from
  // the 'else' logic will go to the right place.
  bool blockRaises = false, blockBreaks = false, blockFallThroughs = false;
  lowerBlock(*newExitBlock, blockRaises, blockBreaks, blockFallThroughs);
  enclosingBlockDoesRaise |= blockRaises;
  enclosingBlockDoesBreak |= blockBreaks;

  // Remove the lit.loop.yield at the end of the block if present, replacing it
  // with a break from this loop.  Other exits like return/break/continue in the
  // else block will already be rewritten if they are present.
  if (blockFallThroughs) {
    assert(isa<LIT::LoopYieldOp>(newExitBlock->getTerminator()));
    newExitBlock->getTerminator()->erase();
    builder.setInsertionPointToEnd(newExitBlock);
    builder.create<HLCF::BreakOp>(ValueRange{}, newLoop.getLabelAttr());
  }

  // Now that the else logic is set, lower the entire loop body to handle the
  // control flow in the body.  This is done with the loop set to the nested
  // loop so that breaks and continues get wired up to it.
  llvm::SaveAndRestore<Operation *> currentLoopSaver(currentLoop, newLoop);
  lowerBlock(*newBody, blockRaises, blockBreaks, blockFallThroughs);
  enclosingBlockDoesRaise |= blockRaises;

  // If the loop body never breaks, then the code after it is unreachable.
  if (!blockBreaks) {
    auto b = handleSemanticTerminatorOp(*newLoop, "infinite loop");
    b.create<UnreachableOp>(loopOp.getLoc());
    loopOp.erase();
    return true;
  }

  // Erase the lit.loop.
  loopOp.erase();
  return false;
}

/// Emit the semantic control-flow IR corresponding to a raise statement.
static void emitRaise(ImplicitLocOpBuilder &b) {
  Operation *opForRaise = LIT::findOpProcessingRaise(b.getInsertionBlock());
  assert(opForRaise && "IR invalid, RaiseOp must only be in valid context");
  if (isa<LIT::FuncOp>(opForRaise)) {
    b.create<LIT::ErrorReturnOp>(
        b.create<ParamConstantOp>(b.getBoolAttr(true)));
  } else {
    assert(isa<LIT::TryOp>(opForRaise));
    b.create<LIT::TryRaiseOp>();
  }
}

/// This function adds the error branch regions to a call operation to a
/// throwing function. These are required by CheckLifetimes to understand
/// conditional initialization of the inout results.
static void addErrorRegions(Operation &op, LIT::LITSignatureType sig,
                            ValueRange operands) {
  // Clone the op and add the error regions.
  ImplicitLocOpBuilder b(op.getLoc(), OpBuilder(&op));
  b.setInsertionPointAfter(&op);
  auto ifOp = b.create<HLCF::IfOp>(op.getResult(0));

  // In the error region, initialize the error value, then raise.
  b.createBlock(&ifOp.getThenRegion());
  b.create<LIT::OwnershipMarkInitializedOp>(
      *std::prev(operands.end(), sig.getErrorSlotOffset()));
  emitRaise(b);

  b.createBlock(&ifOp.getElseRegion());
  b.create<LIT::OwnershipMarkInitializedOp>(
      sig.hasMemoryOnlyResult() ? operands.back() : operands.front());
  b.create<HLCF::YieldOp>();
}

/// This recursive function transforms the specified block:
///   1) It transforms any semantic CF ops like lit.break into terminators like
///      hlcf.break.
///   2) It removes dead code after that and reports errors.
///   3) It computes properties about the block and enclosing context.
void LowerSemanticCF::lowerBlock(Block &block, bool &doesRaise, bool &doesBreak,
                                 bool &doesFallThrough) {
  doesRaise = doesBreak = doesFallThrough = false;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    // Look for semantic terminators and turn them into real terminators.
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      auto b = handleSemanticTerminatorOp(op, "return statement");
      b.create<KGEN::ReturnOp>(returnOp.getOperands());
      op.erase();
      doesFallThrough = false;
      return;
    }

    if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      doesRaise = true;
      auto b = handleSemanticTerminatorOp(op, "raise statement");
      emitRaise(b);
      op.erase();
      return;
    }

    if (isa<LIT::BreakOp>(op)) {
      if (!currentLoop) {
        emitError(op.getLoc(), "'break' is not inside a loop");
        hadError = true;
        op.erase();
        continue;
      }

      doesBreak = true;
      auto b = handleSemanticTerminatorOp(op, "break statement");
      if (auto hlcfLoop = dyn_cast<HLCF::LoopOp>(currentLoop))
        b.create<HLCF::BreakOp>(ValueRange{}, hlcfLoop.getLabelAttr());
      else
        b.create<ParamForBreakOp>();
      op.erase();
      return;
    }

    if (isa<LIT::ContinueOp>(op)) {
      if (!currentLoop) {
        emitError(op.getLoc(), "'continue' is not inside a loop");
        hadError = true;
        op.erase();
        continue;
      }

      auto b = handleSemanticTerminatorOp(op, "continue statement");
      if (auto hlcfLoop = dyn_cast<HLCF::LoopOp>(currentLoop))
        b.create<HLCF::ContinueOp>(ValueRange{}, hlcfLoop.getLabelAttr());
      else
        b.create<ParamForContinueOp>();
      op.erase();
      return;
    }

    if (isa<LIT::LoopContinueOp>(op)) {
      OpBuilder b(&op);
      b.create<HLCF::ContinueOp>(op.getLoc());
      op.erase();
      return;
    }

    // Notice self-recursive calls so we can check them out later.
    if (auto call = dyn_cast<LIT::CallOp>(op))
      hasRecursiveCalls |= call.getDirectCallee() == theFuncSymbol;

    // Add error branches to calls to throwing functions.
    if (isa<LIT::CallOp, LIT::CallIndirectOp>(op)) {
      if (auto sig = LIT::getCalleeType(&op); sig.isThrows()) {
        doesRaise = doesFallThrough = true;
        addErrorRegions(op, sig, LIT::getCalleeArguments(&op));
      }
      continue;
    }

    // Most ops don't have regions and are just fallthrough.
    // TODO: Add support for noreturn calls.
    if (!op.getNumRegions())
      continue;

    // Coroutine await regions are fallthrough only.
    if (auto await = dyn_cast<CO::SuspendOp>(op)) {
      bool awaitRaises = false, awaitBreaks = false, awaitFallsThrough = false;
      lowerBlock(await.getBody().front(), awaitRaises, awaitBreaks,
                 awaitFallsThrough);
      // The verifier will catch any invalid control-flow structure.
      continue;
    }

    // Ignore nested functions, they are handled (and lowered) separately by the
    // outer walker, which we are recursing within post-order.
    if (isa<LIT::FuncOp>(op))
      continue;

    // Process a try op specially to identify dead code and warn.
    if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
      bool tryBodyRaises = false, tryBodyBreaks = false,
           tryBodyFallsThrough = false;
      lowerBlock(tryOp.getTryRegion().front(), tryBodyRaises, tryBodyBreaks,
                 tryBodyFallsThrough);
      doesBreak |= tryBodyBreaks;

      // The try falls through if the except block is reachable and falls
      // through, or if the body falls through and so does the else.
      bool tryFallsThrough = false;

      // Diagnose unneeded code.
      if (!tryBodyRaises) {
        Operation &firstOpInExcept = tryOp.getExceptRegion().front().front();
        // If the finally region is not empty, then this could be a
        // try-finally pattern.
        if (!tryOp.getSuppressWarnings() && tryOp.hasTrivialFinally()) {
          if (!firstOpInExcept.hasTrait<OpTrait::IsTerminator>()) {
            emitWarning(firstOpInExcept.getLoc(),
                        "'except' logic is unreachable, try doesn't raise an "
                        "exception");

          } else {
            emitWarning(tryOp->getLoc(), "try body doesn't raise an exception");
          }
        }

        Operation *firstExceptOp = &tryOp.getExceptRegion().front().front();
        OpBuilder(firstExceptOp).create<UnreachableOp>(firstExceptOp->getLoc());
        eraseOpToEndOfBlock(firstExceptOp);
      } else {
        // The except and else blocks execute without protection from the try.
        bool exceptBreaks = false;
        lowerBlock(tryOp.getExceptRegion().front(), doesRaise, exceptBreaks,
                   tryFallsThrough);
        doesBreak |= exceptBreaks;
      }

      // If there is an 'else' block that is unreachable, complain and remove
      // it, otherwise process it.
      if (!tryBodyFallsThrough) {
        Operation *firstElseOp = &tryOp.getElseRegion().front().front();
        OpBuilder(firstElseOp).create<UnreachableOp>(firstElseOp->getLoc());
        if (!isa<LIT::TryYieldOp>(firstElseOp))
          emitWarning(firstElseOp->getLoc(),
                      "'else' logic in 'try' is unreachable");
        eraseOpToEndOfBlock(firstElseOp);
      } else {
        bool elseRaises = false, elseBreaks = false, elseFallsThrough = false;
        lowerBlock(tryOp.getElseRegion().front(), elseRaises, elseBreaks,
                   elseFallsThrough);
        doesRaise |= elseRaises;
        doesBreak |= elseBreaks;
        tryFallsThrough |= elseFallsThrough;
      }

      // The 'finally' block must fallthrough for the try to fallthrough. Also,
      // it is transparent to raises and breaks.
      bool finallyFallsThrough = false, finallyRaises = false,
           finallyBreaks = false;
      lowerBlock(tryOp.getFinallyRegion().front(), finallyRaises, finallyBreaks,
                 finallyFallsThrough);
      doesRaise |= finallyRaises;
      doesBreak |= finallyBreaks;
      tryFallsThrough &= finallyFallsThrough;

      // Modify the body of the try to implement 'finally' logic.
      tryOp->setOperands({});
      tryOp = lowerTryFinally(tryOp);

      // If the try doesn't fall through, diagnose unreachable code after it.
      if (!tryFallsThrough) {
        auto b = handleSemanticTerminatorOp(
            *tryOp, "try statement that doesn't fall through");
        b.create<UnreachableOp>(tryOp.getLoc());
        return;
      }
      continue;
    }

    // Process a LIT::LoopOp.
    if (auto loopOp = dyn_cast<LIT::LoopOp>(op)) {
      if (lowerLITLoop(loopOp, doesRaise, doesBreak))
        return;
      continue;
    }

    // Process a ParamForOp
    if (auto paramFor = dyn_cast<ParamForOp>(op)) {
      // The 'else' region is not inside the loop. It is transparent to raises
      // and breaks.
      bool elseRaises = false, elseBreaks = false, elseFallsThrough = false;
      lowerBlock(paramFor.getElseRegion().front(), elseRaises, elseBreaks,
                 elseFallsThrough);
      doesRaise |= elseRaises;
      doesBreak |= elseBreaks;

      // The loop is only transparent to raises.
      bool loopRaises = false, loopBreaks = false, loopFallsThrough = false;
      llvm::SaveAndRestore<Operation *> currentLoopSaver(currentLoop, paramFor);
      lowerBlock(paramFor.getBody().front(), loopRaises, loopBreaks,
                 loopFallsThrough);
      doesRaise |= loopRaises;
      continue;
    }

    // Process a HLCF::ElifOp
    if (auto elifOp = dyn_cast<HLCF::ElifOp>(op)) {
      lowerElif(elifOp, doesRaise, doesBreak, doesFallThrough);
      if (!doesFallThrough)
        return;
      continue;
    }

    // Otherwise we must have an if operation.
    assert((isa<HLCF::IfOp, ParamIfOp>(op)) &&
           "Unknown operation with regions");

    // If this is a dynamic `if False:` or @parameter if on known condition,
    // mark the unreachable block as unreachable so we don't consider it live.
    Region *deadRegion = nullptr;
    bool constantCondValue = false;
    if (auto ifOp = dyn_cast<HLCF::IfOp>(op)) {
      BoolAttr cond;
      if (mlir::matchPattern(ifOp.getCond(), m_Constant(&cond))) {
        constantCondValue = cond.getValue();
        deadRegion =
            &(constantCondValue ? ifOp.getElseRegion() : ifOp.getThenRegion());
      }
    } else if (auto ifOp = dyn_cast<ParamIfOp>(op)) {
      if (auto cond = dyn_cast<BoolAttr>(ifOp.getCond())) {
        constantCondValue = cond.getValue();
        deadRegion =
            &(constantCondValue ? ifOp.getElseRegion() : ifOp.getThenRegion());
      }
    }

    // If either branch of the if is unreachable, diagnose any live code there
    // as unreachable and replace it with a kgen.unreachable so we don't think
    // about it for liveness' sake.
    if (deadRegion) {
      Block &deadBlock = deadRegion->front();
      Operation *firstDeadOp = &deadBlock.front();
      // Warn about unreachable code in an 'if', but not in a '@parameter if'.
      // It serves the function of ifdef's, and conditions are often
      // known-statically true/false.
      if (!isa<ParamIfOp>(op) &&
          !firstDeadOp->hasTrait<OpTrait::IsTerminator>())
        emitWarning(firstDeadOp->getLoc(), "unreachable code after 'if ")
            << (constantCondValue ? "True'" : "False'");
      eraseOpToEndOfBlock(&deadBlock.front());
      OpBuilder::atBlockBegin(&deadBlock).create<UnreachableOp>(op.getLoc());
    }

    bool ifOpFallsThrough = false;
    for (auto &region : op.getRegions()) {
      bool regionRaises = false, regionBreaks = false,
           regionFallsThrough = false;
      lowerBlock(region.front(), regionRaises, regionBreaks,
                 regionFallsThrough);
      doesRaise |= regionRaises;
      doesBreak |= regionBreaks;
      ifOpFallsThrough |= regionFallsThrough;
    }

    // If the operation doesn't fall through, cut off the code after it.
    if (!ifOpFallsThrough) {
      auto b = handleSemanticTerminatorOp(
          op, "if statement with then/else that do not fall through");
      b.create<UnreachableOp>(op.getLoc());
      return;
    }
  }

  auto *terminator = &block.back();
  if (isa<HLCF::BreakOp, ParamForBreakOp>(terminator)) {
    doesBreak = true;
    return;
  }

  // These are not fallthroughs.
  if (isa<KGEN::ReturnOp, HLCF::ContinueOp, ParamForContinueOp,
          KGEN::UnreachableOp>(terminator))
    return;

  // If we fell off the bottom, then we have a fall-through terminator.
  assert((isa<HLCF::YieldOp, HLCF::ElifYieldOp, LIT::TryYieldOp, ParamYieldOp,
              LIT::EndFuncOp, CO::SuspendEndOp, LIT::LoopConditionOp,
              LIT::LoopYieldOp>(block.back())));
  doesFallThrough = true;
}

/// This function is called to check to see if the function has any
/// unconditional self-recursive calls.  Such a call will cause an infinite
/// loop, so we generate a warning.
///
/// This function is invoked on blocks after SemanticCF lowering is done on the
/// function. The "isConditional" argument indicates whether this is being
/// called in a conditional context (e.g. under an if check).  This returns
/// `true` if the block might early return out of the enclosing function with a
/// return or throw, `false` if it will fall through.
bool LowerSemanticCF::checkSelfRecursion(Block &block, bool isConditional) {
  for (Operation &op : llvm::make_early_inc_range(block)) {
    // Notice self-recursive calls so we can check them out later.  If we are
    // invoked in an unconditional area, we can emit the warning.
    if (auto call = dyn_cast<LIT::CallOp>(op);
        call && !isConditional && theFuncSymbol &&
        call.getDirectCallee() == theFuncSymbol) {
      emitWarning(call.getLoc(),
                  "self recursive call will cause an infinite loop");
      continue;
    }

    // If this is a return out of the function, notice this and we're done.
    // LIT::TryRaiseOp/break/continue/etc are used for transfers to an enclosing
    // try, which doesn't completely exit the function.
    if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp>(op))
      return true;

    // Most ops don't have regions and are just fallthrough.
    if (!op.getNumRegions())
      continue;

    // Ignore nested functions, they are handled separately by the outer walker.
    if (isa<LIT::FuncOp>(op))
      continue;

    // If we are already in conditional code, or if this is an 'if'-like
    // operation, then the subregions are executed conditionally.
    bool isSubregionConditional =
        isConditional || isa<HLCF::IfOp, ParamIfOp, HLCF::ElifOp>(op);
    // Handle things like if statements, HLCF::Loop, try, etc.
    for (auto &region : op.getRegions()) {
      if (checkSelfRecursion(region.front(), isSubregionConditional))
        return true;
    }
  }

  // If we made it this far then we didn't early return.
  return false;
}

/// Lower all lexical terminators in the function and remove dead code.
void LowerSemanticCF::run() {
  bool doesRaise = false, doesBreak = false, doesFallThrough = false;
  lowerBlock(*theFunc.getBody(), doesRaise, doesBreak, doesFallThrough);

  // If we had an error already, don't diagnose more semantic issues.
  if (hadError)
    return;

  // A return is required at the end of function, diagnose it if missing.  The
  // parser automatically inserts a `return None` in functions that return None.
  if (LIT::EndFuncOp endFunc =
          dyn_cast<LIT::EndFuncOp>(theFunc.getBody()->getTerminator())) {
    emitError(endFunc->getLoc(),
              "return expected at end of function with results");
    hadError = true;
    return;
  }

  // If everything looks good, check whether any self-recursive calls are
  // unconditionally executed.  If so, they are infinite recursion.  We need
  // control flow information to avoid diagnosing recursive calls in if
  // statements.
  if (hasRecursiveCalls)
    (void)checkSelfRecursion(*theFunc.getBody(), /*isConditional=*/false);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERSEMANTICCF
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerSemanticCFPass : impl::LowerSemanticCFBase<LowerSemanticCFPass> {
  using LowerSemanticCFBase::LowerSemanticCFBase;

  void runOnOperation() override {
    // Walk all functions and update them.
    bool hadError = false;
    getOperation().walk<mlir::WalkOrder::PostOrder>([&](LIT::FuncOp func) {
      // Skip external functions.
      if (func.isExternal())
        return;
      // Just delete trait functions. They are no longer needed.
      if (isa<LIT::TraitFuncOp>(func.getBody()->getTerminator())) {
        func.erase();
        return;
      }

      // Lower things like lit.break into hlcf.break which are terminators,
      // and diagnose unreachable code.
      LowerSemanticCF lowerer(func);
      lowerer.run();
      hadError |= lowerer.hadError;
    });
    if (hadError)
      return signalPassFailure();
  }
};
} // namespace
