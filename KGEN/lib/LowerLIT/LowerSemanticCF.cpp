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
//  3) It hoists nested functions from being lit.func to being
//     ParamDeclareRegionOp's.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/STLExtras.h"
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
  // Try to find a param return in each branch.
  std::optional<ResultParams> thenParams, elseParams;
  if (failed(findParamReturn(ifOp.getThenRegion(), thenParams, nameCounter)) ||
      failed(findParamReturn(ifOp.getElseRegion(), elseParams, nameCounter)))
    return failure();

  // They must either both have a return, or neither.
  if (thenParams.has_value() != elseParams.has_value())
    return emitError(ifOp.getLoc(),
                     "result parameters are not defined along all branches");

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

/// Lower all lexical terminators in the function and remove dead code.
static LogicalResult lowerSemanticCF(LIT::FuncOp func) {
  // While we walk the IR, we are going to determine if the top-level
  // `lit.end_func` is reachable with trivial dead-code analysis. Do this by
  // marking each `ControlFlowNode` and `ControlFlowTerminator` as live or dead,
  // and see if we can reach the end of the function body.
  DenseSet<Operation *> liveCfOps;
  auto markNextInBlockAsLive = [&](Block *block, Block::iterator it) {
    assert(it != block->end());
    for (Operation &op : llvm::make_range(it, block->end())) {
      if (!isa<HLCF::ControlFlowNode, HLCF::ControlFlowTerminator,
               LIT::ReturnOp, LIT::RaiseOp, LIT::BreakOp, LIT::ContinueOp,
               LIT::EndFuncOp>(op))
        continue;
      liveCfOps.insert(&op);
      return;
    }
    llvm_unreachable("`lower-semantic-cf` encountered unexpected terminator");
  };
  auto markNextOperationAsLive = [&](Operation *op) {
    markNextInBlockAsLive(op->getBlock(), std::next(op->getIterator()));
  };
  auto markNextInRegionAsLive = [&](Region &region) {
    markNextInBlockAsLive(&region.front(), region.front().begin());
  };
  // Start the analysis at the function entry block.
  markNextInRegionAsLive(func.getBodyRegion());

  // Collect all the terminators first to avoid iterator invalidation.
  SmallVector<Operation *> terminators;

  WalkResult result = func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Don't walk into nested functions, the walker that called lowerSemanticCF
    // will handle them separately.
    if (op != func && isa<LIT::FuncOp>(op))
      return WalkResult::skip();

    if (isa<HLCF::ControlFlowNode, HLCF::ControlFlowTerminator>(op) &&
        !isa<ReturnOp>(op) && liveCfOps.contains(op)) {
      SmallVector<Attribute> operands;
      for (Value operand : op->getOperands())
        mlir::matchPattern(operand, mlir::m_Constant(&operands.emplace_back()));
      SmallVector<HLCF::ControlFlowTarget> targets;
      // Process each control-flow target from the base operation. For nodes,
      // the base operation is itself. For terminators, it is the nearest
      // matching parent operation.
      Operation *base = op;
      if (auto node = dyn_cast<HLCF::ControlFlowNode>(op)) {
        node.getEntryTargets(operands, targets);
      } else {
        auto term = cast<HLCF::ControlFlowTerminator>(op);
        term.getBranchTargets(operands, targets);
        do {
          base = base->getParentOp();
        } while (!term.isParentNode(base));
      }
      for (const HLCF::ControlFlowTarget &target : targets) {
        if (!target.index)
          markNextOperationAsLive(base);
        else
          markNextInRegionAsLive(base->getRegion(*target.index));
      }

    } else if (isa<LIT::ReturnOp, LIT::RaiseOp, LIT::BreakOp, LIT::ContinueOp>(
                   op)) {
      terminators.push_back(op);
      // Do nothing for `return` and `continue`. The former exits the function
      // and nothing further is live. The latter must be live if the surrounding
      // loop is live, so we ignore it as a micro-optimization.
      if (isa<LIT::BreakOp>(op) && liveCfOps.contains(op)) {
        markNextOperationAsLive(op->getParentOfType<HLCF::LoopOp>());
      } else if (isa<LIT::RaiseOp>(op)) {
        auto tryOp = op->getParentOfType<LIT::TryOp>();
        if (tryOp && liveCfOps.contains(op))
          markNextInRegionAsLive(tryOp.getExceptRegion());
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // Lower all the terminators as they are encountered.
  SmallVector<Block *> deadBlocks;
  for (Operation *op : terminators) {
    // Ignore dead operations.
    if (op->getBlock() != &op->getParentRegion()->front())
      continue;

    ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      b.create<KGEN::ReturnOp>(returnOp.getOperands());
    } else if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      b.create<LIT::TryRaiseOp>(raiseOp.getError());
    } else if (auto breakOp = dyn_cast<LIT::BreakOp>(op)) {
      b.create<HLCF::BreakOp>();
    } else {
      assert(isa<LIT::ContinueOp>(op) && "unknown terminator");
      b.create<HLCF::ContinueOp>();
    }

    // Check and warn about dead code.
    if (!op->getNextNode()->hasTrait<OpTrait::IsTerminator>())
      op->getNextNode()->emitWarning("unreachable code after ")
          << op->getName().stripDialect() << " statement";

    // Mark all subsequent operations as dead.
    deadBlocks.push_back(op->getBlock()->splitBlock(op));
  }

  // Remove all dead code.
  for (Block *block : llvm::reverse(deadBlocks))
    block->erase();

  Operation *terminator = func.getBody()->getTerminator();

  // If the function is explicitly terminated with a `return` or raise, we're
  // good.
  if (!isa<LIT::EndFuncOp>(terminator))
    return success();

  Type declaredResultType = func.getResultTypeWithoutErrorVariant();

  // If the endfunc isn't live, then it doesn't matter, there must have been a
  // return/raise before this.
  if (liveCfOps.contains(terminator)) {
    // A return is required if the function has a non-none result.
    if (!isa<LIT::NoneType>(declaredResultType) ||
        func.getSignature().hasMemoryOnlyResult())
      return terminator->emitError(
          "return expected at end of function with results");
  }

  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(terminator));
  if (!isa<LIT::NoneType>(declaredResultType)) {
    b.create<KGEN::UnreachableOp>();
  } else {
    // The function returns none.
    Value retVal = b.create<ParamConstantOp>(b.getAttr<LIT::NoneAttr>());

    // Wrap the result value if necessary.
    if (func.isThrows())
      retVal = b.create<VariantCreateOp>(func.getResultType(), retVal);
    b.create<KGEN::ReturnOp>(retVal);
  }
  terminator->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// lowerNestedFunctions
//===----------------------------------------------------------------------===//

/// Given a function, check to see if it is a top-level function.  If not, lower
/// it to a ParamDeclareRegionOp.
static LogicalResult lowerNestedFunctions(LIT::FuncOp func) {
  // If this is a top-level function, leave it alone.
  if (!func->getParentOfType<LIT::FuncOp>())
    return success();

  // Process a nested function by lowering it straight to a
  // `kgen.param.declare.region`. Nested functions are denoted with an
  // parameter declaration on the function declaration.
  ParamDeclAttr decl = func.getParamDeclAttr();
  if (!decl) {
    func.emitError("nested function must have a parameter declaration");
    return failure();
  }

  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(func));
  auto region = b.create<ParamDeclareRegionOp>(
      decl, func.getSignature(), ArrayRef<ConstraintAttr>(),
      /*isolated=*/false, func.getAlwaysInlineLevel());
  region.getBodyRegion().takeBody(func.getBodyRegion());
  func.erase();
  return success();
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
      // Lower things like lit.break into hlcf.break which are terminators,
      // and diagnose unreachable code.
      hadError |= failed(lowerSemanticCF(func));
      // Lower 'lit.param.return' into 'kgen.param.result_bind'.
      hadError |= failed(lowerParamResults(func));
      // Lower nested functions by converting them to region declarations.
      // This could erase func!
      hadError |= failed(lowerNestedFunctions(func));
    });
    if (hadError)
      return signalPassFailure();
  }
};
} // namespace
