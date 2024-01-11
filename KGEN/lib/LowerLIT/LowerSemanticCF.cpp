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
//  2) It lowers 'lit.param.return' instances into a single
//     'kgen.param.result_bind' instance.
//
//===----------------------------------------------------------------------===//

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

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// 'lit.param.return' to 'kgen.param.result_bind' lowering.
//===----------------------------------------------------------------------===//

namespace {
/// This contains information about a set of result parameters.
struct ResultParams {
  /// The result parameter values.
  SmallVector<TypedAttr> values;
  /// A location value for the lit.param.return.
  Location loc;
};
} // namespace

static FailureOr<std::optional<ResultParams>>
checkParamIfResultBindings(ParamIfOp ifOp, unsigned &nameCounter);

/// Bubble `param_return` operations up to the top-level of the function. Start
/// from a post-order traversal to visit the innermost `param.if` operations
/// first.  `kgen.param.if` defines a specific scope we need to watch out for.
static LogicalResult findParamReturn(Region &region,
                                     std::optional<ResultParams> &result,
                                     unsigned &nameCounter) {
  // This handles when we find new definition of results in this scope.
  // If we already have a lit.param.return in this scope, then this is a
  // redefinition error.
  auto handleDefinition = [&](ResultParams newResults) {
    if (result)
      emitError(newResults.loc,
                "result parameters already defined in this scope")
              .attachNote(result->loc)
          << "previous parameter return is here";

    result = newResults;
  };

  auto opProcessor = [&](Operation *op) -> WalkResult {
    // If we see a placeholder kgen.param.result_bind, erase it.
    if (auto bind = dyn_cast<ParamResultBindOp>(op)) {
      bind->erase();
      return WalkResult::skip();
    }

    // lit.param.return declares the result parameters.
    if (auto bind = dyn_cast<LIT::ParamReturnOp>(op)) {
      handleDefinition({llvm::to_vector(bind.getParameters()), bind.getLoc()});

      // Don't need the lit.param.return anymore.
      op->erase();
      return WalkResult::skip();
    }

    // kgen.param.if is handled specially.
    auto ifOp = dyn_cast<ParamIfOp>(op);
    if (!ifOp)
      return WalkResult::advance();

    // Try to find a lit.param.return in each branch.  This could fail, succeed
    // with a set of parameters to return, or succeed with no returns.
    FailureOr<std::optional<ResultParams>> nestedValues =
        checkParamIfResultBindings(ifOp, nameCounter);
    if (failed(nestedValues))
      return WalkResult::interrupt();
    if (*nestedValues)
      handleDefinition(**nestedValues);

    // Don't recurse into the kgen.param.if, we've already processed it.
    return WalkResult::skip();
  };

  WalkResult walk = region.walk<mlir::WalkOrder::PreOrder>(opProcessor);
  return success(!walk.wasInterrupted());
}

// We hoist the param.return out of kgen.param.if by transforming the 'if' to
// return the parameter values for each branch and using the result of that
// outside the 'if'.
static FailureOr<std::optional<ResultParams>>
checkParamIfResultBindings(ParamIfOp ifOp, unsigned &nameCounter) {
  // If the parameter if is on a static constant, then only follow one side.
  if (auto cond = dyn_cast<BoolAttr>(ifOp.getCond())) {
    Region &liveRegion =
        cond.getValue() ? ifOp.getThenRegion() : ifOp.getElseRegion();
    std::optional<ResultParams> params;
    if (failed(findParamReturn(liveRegion, params, nameCounter)))
      return failure();
    return params;
  }

  // Try to find a param return in each branch.
  std::optional<ResultParams> thenParams, elseParams;
  if (failed(findParamReturn(ifOp.getThenRegion(), thenParams, nameCounter)) ||
      failed(findParamReturn(ifOp.getElseRegion(), elseParams, nameCounter)))
    return failure();

  // They must either both have a return, or neither.
  if (thenParams.has_value() != elseParams.has_value()) {
    // If the else block is empty, then the user probably wrote a fallthrough,
    // which isn't going to work.
    const char *message;
    if (isa<ParamYieldOp>(ifOp.getElseRegion().front().front())) {
      message =
          "result parameters in '@parameter if' may not use fall-through else";
    } else {
      message = "result parameters must be specified in 'if' and 'else' "
                "branches of '@parameter if'";
    }
    auto diag = emitError(ifOp.getLoc(), message);
    if (thenParams)
      diag.attachNote(thenParams->loc) << "one parameter return is here";
    if (elseParams)
      diag.attachNote(elseParams->loc) << "one parameter return is here";
    return failure();
  }

  // If there was no definition, it must be elsewhere.
  if (!thenParams)
    return thenParams;

  // Pass the return parameters through using result parameters on the if.
  SmallVector<ParamDeclAttr> resultParams;
  SmallVector<TypedAttr> resultValues;
  OpBuilder b(ifOp.getContext());
  for (auto [idx, param] : llvm::enumerate(thenParams->values)) {
    StringAttr name =
        b.getStringAttr("(branch_result_" + Twine(nameCounter++) + ")");
    resultParams.push_back(ParamDeclAttr::get(name, param.getType()));
    resultValues.push_back(ParamDeclRefAttr::get(name, param.getType()));
  }
  ifOp.setResultParams(resultParams);
  b.setInsertionPoint(&ifOp.getThenOps().back());
  b.create<ParamResultBindOp>(thenParams->loc, thenParams->values);
  b.setInsertionPoint(&ifOp.getElseOps().back());
  b.create<ParamResultBindOp>(elseParams->loc, elseParams->values);
  return {ResultParams{std::move(resultValues),
                       b.getFusedLoc({thenParams->loc, elseParams->loc})}};
}

/// Lower lit.param.result into kgen.result_bind.  This happens after semantic
/// returns are processed.
static LogicalResult lowerParamResults(LIT::FuncOp func) {
  // If the function has result parameters, process them here.
  if (func.getResultParams().empty())
    return success();

  // Scan the body to find a lit.param.return that specifies the return params.
  std::optional<ResultParams> resultParams;
  unsigned nameCounter = 0;
  if (failed(findParamReturn(func.getBodyRegion(), resultParams, nameCounter)))
    return failure();

  // If there is none, diagnose it.
  if (!resultParams)
    return emitError(
        func.getLoc(),
        "missing parameter return for function with result parameters");

  // Bind the result parameter values if there are any.
  OpBuilder b(func.getBody()->getTerminator());
  b.create<ParamResultBindOp>(resultParams->loc, resultParams->values);
  return success();
}

//===----------------------------------------------------------------------===//
// Semantic control flow lowering.
//===----------------------------------------------------------------------===//

/// Insert 'finally' block logic on a `lit.try` operation by finding all
/// terminators that exit the try regions and pasting the finally clause before
/// it. Try operations must be processed post-order, so that the order in which
/// the finally clauses are pasted is correct.
static LIT::TryOp lowerTryFinally(LIT::TryOp tryOp) {
  Block &finallyBlock = tryOp.getFinallyRegion().front();

  auto pasteFinally = [&](Operation *term) {
    OpBuilder b(term);
    IRMapping map;
    for (Operation &op : finallyBlock.without_terminator())
      b.clone(op, map);
    // If the finally block terminator exits, then the current terminator is
    // dead code.
    if (!isa<LIT::TryYieldOp>(finallyBlock.getTerminator())) {
      b.clone(*finallyBlock.getTerminator(), map);
      term->erase();
    }
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

static void lowerSemanticCFForBlock(Block &block, bool &doesRaise,
                                    bool &doesBreak, bool &doesFallThrough,
                                    int64_t loopLevel);

/// Get parent loop's label if there is one; otherwise, generate and set a new
/// label.
static StringAttr getOrSetParentLoopLabel(HLCF::LoopOp loop,
                                          int64_t loopLevel) {
  auto parentLoop = loop->getParentOfType<HLCF::LoopOp>();
  StringAttr label = parentLoop.getLabelAttr();
  if (!label) {
    // Parent loop of loop has level = loopLevel - 2.
    label = StringAttr::get(parentLoop->getContext(),
                            "_loop_" + Twine(loopLevel - 2));
    parentLoop.setLabelAttr(label);
  }
  return label;
}

/// Lower a LIT::LoopOp to HLCF::LoopOp.
/// Return true if the lowering should stop traversing the rest of the
/// operations.
static bool lowerLITLoop(LIT::LoopOp loopOp, bool &enclosingBlockDoesRaise,
                         int64_t loopLevel) {
  // Lower loop conditions.
  Block &condBlock = loopOp.getCondRegion().front();
  Block &bodyBlock = loopOp.getBodyRegion().front();
  Block &elseBlock = loopOp.getElseRegion().front();
  Location loopLoc = loopOp->getLoc();

  // Create the new HLCF::LoopOp.
  OpBuilder builder(loopOp);
  builder.setInsertionPointAfter(loopOp);
  auto newLoop =
      builder.create<HLCF::LoopOp>(loopLoc, loopOp.getUnrollLevelAttr());
  Block *newBody = builder.createBlock(&newLoop.getBody());
  Block *newExitBlock = nullptr;

  // Move the loop condition logic to the beginning of the HLCF::LoopOp's body.
  Operation *prevOp = nullptr;
  for (Operation &op :
       llvm::make_early_inc_range(condBlock.without_terminator())) {
    if (prevOp == nullptr)
      op.moveBefore(newBody, newBody->begin());
    else
      op.moveAfter(prevOp);
    prevOp = &op;
  }

  // Create the loop condition check.
  auto loopCondition = cast<LIT::LoopConditionOp>(condBlock.getTerminator());
  // Create loop condition check
  builder.setInsertionPointToEnd(newBody);
  auto condOp = builder.create<HLCF::IfOp>(loopLoc, loopCondition.getOperand());
  builder.createBlock(&condOp.getThenRegion());
  builder.create<HLCF::YieldOp>(loopLoc);
  newExitBlock = builder.createBlock(&condOp.getElseRegion());
  prevOp = condOp;

  // Move the loop's body to the HLCF::LoopOp's body.
  for (Operation &op : llvm::make_early_inc_range(bodyBlock.getOperations())) {
    op.moveAfter(prevOp);
    prevOp = &op;
  }

  bool createExitBlockBreakTerminator = true;
  if (elseBlock.getOperations().size() > 1) {
    builder.setInsertionPointToStart(newExitBlock);

    // Move else region logic
    prevOp = nullptr;
    for (Operation &op :
         llvm::make_early_inc_range(elseBlock.without_terminator())) {
      if (prevOp == nullptr)
        op.moveBefore(newExitBlock, newExitBlock->begin());
      else
        op.moveAfter(prevOp);
      prevOp = &op;

      // Lower semantic continue and break in the else region
      // with the right loop label.
      if (isa<LIT::ContinueOp>(op)) {
        StringAttr label = getOrSetParentLoopLabel(newLoop, loopLevel);
        builder.setInsertionPointAfter(&op);
        prevOp =
            builder.create<HLCF::ContinueOp>(op.getLoc(), ValueRange{}, label);
        createExitBlockBreakTerminator = false;
        op.erase();
      } else if (auto breakOp = dyn_cast<LIT::BreakOp>(op)) {
        StringAttr label = getOrSetParentLoopLabel(newLoop, loopLevel);
        builder.setInsertionPointAfter(&op);
        prevOp =
            builder.create<HLCF::BreakOp>(op.getLoc(), ValueRange{}, label);
        createExitBlockBreakTerminator = false;
        op.erase();
      }
    }
  }

  if (createExitBlockBreakTerminator) {
    builder.setInsertionPointToEnd(newExitBlock);
    builder.create<HLCF::BreakOp>(loopLoc);
  }

  // Process the new HLCF::LoopOp specially to propagate up the break flag.
  bool loopBodyRaises = false, loopBodyBreaks = false,
       loopBodyFallThroughs = false;
  lowerSemanticCFForBlock(*newBody, loopBodyRaises, loopBodyBreaks,
                          loopBodyFallThroughs, loopLevel);
  enclosingBlockDoesRaise |= loopBodyRaises;

  // If the loop body never breaks, then the code after it is unreachable.
  if (!loopBodyBreaks) {
    auto b = handleSemanticTerminatorOp(*newLoop, "infinite loop");
    b.create<UnreachableOp>(loopOp.getLoc());
    loopOp.erase();
    return true;
  }

  // Erase the lit.loop.
  loopOp.erase();
  return false;
}

/// This recursive function transforms the specified block:
///   1) It transforms any semantic CF ops like lit.break into terminators like
///      hlcf.break.
///   2) It removes dead code after that and reports errors.
///   3) It computes properties about the block and enclosing context.
static void lowerSemanticCFForBlock(Block &block, bool &doesRaise,
                                    bool &doesBreak, bool &doesFallThrough,
                                    int64_t loopLevel) {
  doesRaise = doesBreak = doesFallThrough = false;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    // Look for semantic terminators and turn them into real terminators.
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      auto b = handleSemanticTerminatorOp(op, "return statement");
      b.create<KGEN::ReturnOp>(returnOp.getOperands());
      op.erase();
      return;
    }

    if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      doesRaise = true;
      auto b = handleSemanticTerminatorOp(op, "raise statement");
      Operation *opForRaise = LIT::findOpProcessingRaise(b.getInsertionBlock());
      assert(opForRaise && "IR invalid, RaiseOp must only be in valid context");
      if (isa<LIT::FuncOp>(opForRaise)) {
        LIT::FuncOp funcOp = raiseOp->getParentOfType<LIT::FuncOp>();
        Type failedType = funcOp.getMLIRResultType();
        assert(isa<VariantType>(failedType));
        b.create<LIT::ErrorReturnOp>(b.create<VariantCreateOp>(
            raiseOp->getLoc(), failedType, raiseOp.getError(), 0));
      } else {
        assert(isa<LIT::TryOp>(opForRaise));
        b.create<LIT::TryRaiseOp>(raiseOp.getError());
      }
      op.erase();
      return;
    }

    if (isa<LIT::BreakOp>(op)) {
      doesBreak = true;
      auto b = handleSemanticTerminatorOp(op, "break statement");
      b.create<HLCF::BreakOp>();
      op.erase();
      return;
    }

    if (isa<LIT::ContinueOp>(op)) {
      auto b = handleSemanticTerminatorOp(op, "continue statement");
      b.create<HLCF::ContinueOp>();
      op.erase();
      return;
    }

    if (isa<LIT::LoopContinueOp>(op)) {
      OpBuilder b(&op);
      b.create<HLCF::ContinueOp>(op.getLoc());
      op.erase();
      return;
    }

    // Most ops don't have regions and are just fallthrough.
    // TODO: Add support for noreturn calls.
    if (!op.getNumRegions())
      continue;

    // Coroutine await regions are fallthrough only.
    if (auto await = dyn_cast<POP::CoroutineAwaitOp>(op)) {
      bool awaitRaises = false, awaitBreaks = false, awaitFallsThrough = false;
      lowerSemanticCFForBlock(await.getBody().front(), awaitRaises, awaitBreaks,
                              awaitFallsThrough, loopLevel);
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
      lowerSemanticCFForBlock(tryOp.getTryRegion().front(), tryBodyRaises,
                              tryBodyBreaks, tryBodyFallsThrough, loopLevel);
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
        lowerSemanticCFForBlock(tryOp.getExceptRegion().front(), doesRaise,
                                exceptBreaks, tryFallsThrough, loopLevel);
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
        lowerSemanticCFForBlock(tryOp.getElseRegion().front(), elseRaises,
                                elseBreaks, elseFallsThrough, loopLevel);
        doesRaise |= elseRaises;
        doesBreak |= elseBreaks;
        tryFallsThrough |= elseFallsThrough;
      }

      // The 'finally' block must fallthrough for the try to fallthrough. Also,
      // it is transparent to raises and breaks.
      bool finallyFallsThrough = false, finallyRaises = false,
           finallyBreaks = false;
      lowerSemanticCFForBlock(tryOp.getFinallyRegion().front(), finallyRaises,
                              finallyBreaks, finallyFallsThrough, loopLevel);
      doesRaise |= finallyRaises;
      doesBreak |= finallyBreaks;
      tryFallsThrough &= finallyFallsThrough;

      // Modify the body of the try to implement 'finally' logic.
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
      if (lowerLITLoop(loopOp, doesRaise, loopLevel + 1))
        return;
      continue;
    }

    // Otherwise we must have an if operation.
    assert((isa<HLCF::IfOp, ParamIfOp, LIT::HandleVariantOp>(op)) &&
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
      lowerSemanticCFForBlock(region.front(), regionRaises, regionBreaks,
                              regionFallsThrough, loopLevel);
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
  if (isa<HLCF::BreakOp>(terminator)) {
    doesBreak = true;
    return;
  }

  // These are not fallthroughs.
  if (isa<KGEN::ReturnOp, HLCF::ContinueOp, KGEN::UnreachableOp>(terminator))
    return;

  // If we fell off the bottom, then we have a fall-through terminator.
  assert((isa<HLCF::YieldOp, LIT::TryYieldOp, ParamYieldOp, LIT::EndFuncOp,
              LIT::YieldOp, POP::CoroutineAwaitEndOp>(block.back())));
  doesFallThrough = true;
}

/// Lower all lexical terminators in the function and remove dead code.
static LogicalResult lowerSemanticCF(LIT::FuncOp func) {
  bool doesRaise = false, doesBreak = false, doesFallThrough = false;
  lowerSemanticCFForBlock(*func.getBody(), doesRaise, doesBreak,
                          doesFallThrough, 0);

  LIT::EndFuncOp endFunc =
      dyn_cast<LIT::EndFuncOp>(func.getBody()->getTerminator());

  // we're done if explicitly terminated with a `return` or raise.
  if (!endFunc)
    return success();

  // A return is required if the function, diagnose it if missing.
  return emitError(endFunc->getLoc(),
                   "return expected at end of function with results");
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
      hadError |= failed(lowerSemanticCF(func));

      // Lower 'lit.param.return' into 'kgen.param.result_bind'.
      hadError |= failed(lowerParamResults(func));
    });
    if (hadError)
      return signalPassFailure();
  }
};
} // namespace
