//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
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
// lowerLexicalTerminators
//===----------------------------------------------------------------------===//

namespace {
/// This contains information about a set of result parameters.
struct ResultParams {
  /// The result parameter values.
  SmallVector<TypedAttr> values;
  /// A location value.
  Location loc;
};
} // namespace

static FailureOr<std::optional<ResultParams>>
threadParamIfResultBindings(ParamIfOp ifOp, unsigned &nameCounter);

/// Bubble `param_return` operations up to the top-level return. Start from a
/// post-order traversal to visit the innermost `param.if` operations first.
LogicalResult findParamReturn(Region &region,
                              std::optional<ResultParams> &result,
                              unsigned &nameCounter) {
  WalkResult walk = region.walk<mlir::WalkOrder::PreOrder>([&](Operation *op)
                                                               -> WalkResult {
    if (auto ifOp = dyn_cast<ParamIfOp>(op)) {
      FailureOr<std::optional<ResultParams>> nestedValues =
          threadParamIfResultBindings(ifOp, nameCounter);
      if (failed(nestedValues))
        return WalkResult::interrupt();
      if (*nestedValues) {
        if (result) {
          return emitError(op->getLoc(),
                           "result parameters already defined in this branch");
        }
        result = *nestedValues;
      }
      return WalkResult::skip();
    }
    if (auto bind = dyn_cast<LIT::ParamReturnOp>(op)) {
      if (result) {
        return emitError(op->getLoc(),
                         "result parameters already defined in this branch");
      }
      result =
          ResultParams{llvm::to_vector(bind.getParameters()), bind.getLoc()};
    }
    return WalkResult::advance();
  });
  return success(!walk.wasInterrupted());
}

static FailureOr<std::optional<ResultParams>>
threadParamIfResultBindings(ParamIfOp ifOp, unsigned &nameCounter) {
  // Try to find a param return in each branch.
  std::optional<ResultParams> thenParams, elseParams;
  if (failed(findParamReturn(ifOp.getThenRegion(), thenParams, nameCounter)) ||
      failed(findParamReturn(ifOp.getElseRegion(), elseParams, nameCounter)))
    return failure();
  if (thenParams.has_value() != elseParams.has_value()) {
    return emitError(ifOp.getLoc(),
                     "result parameters are not defined along all branches");
  }
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

/// Given the result of a throwable call, generate the code to check if the
/// result type is an error, and if so, propagate the error.
static Value createUnwrapOrPropagate(ImplicitLocOpBuilder &b, LIT::FuncOp func,
                                     Value errOr, DeclRefType errType,
                                     Type type) {
  auto ifOp =
      b.create<HLCF::IfOp>(type, b.create<POP::VariantIsOp>(errOr, errType));

  b.createBlock(&ifOp.getElseRegion());
  Value value = b.create<POP::VariantGetOp>(type, errOr);
  b.create<HLCF::YieldOp>(b.getLoc(), value);

  b.createBlock(&ifOp.getThenRegion());
  Value err = b.create<POP::VariantGetOp>(errType, errOr);
  if (auto tryOp = ifOp->getParentOfType<LIT::TryOp>();
      tryOp && tryOp.getTryRegion().findAncestorOpInRegion(*ifOp)) {
    b.create<LIT::TryRaiseOp>(err);
  } else {
    Value result = b.create<POP::VariantCreateOp>(
        POP::VariantType::get({errType, func.getResultType()}), err);
    b.create<ReturnOp>(result);
  }
  return ifOp.getResult(0);
}

/// Lower all lexical terminators in the function and remove dead code.
static LogicalResult lowerLexicalTerminators(DeclRefType errType,
                                             LIT::FuncOp func) {
  if (func.isThrows() && !errType)
    return func.emitError("function throws but no 'Error' type was found");

  // If the function has result parameters, process them here.
  std::optional<ResultParams> resultParams;
  if (!func.getResultParams().empty()) {
    unsigned nameCounter = 0;
    if (failed(
            findParamReturn(func.getBodyRegion(), resultParams, nameCounter)))
      return failure();
    if (!resultParams) {
      return emitError(
          func.getBody()->getTerminator()->getLoc(),
          "missing parameter return for function with result parameters");
    }
  }

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
               LIT::EndFuncOp, KGENCallOpInterface>(op))
        continue;
      liveCfOps.insert(&op);
      return;
    }
    llvm_unreachable(
        "`lower-lit-terminators` encountered unexpected terminator");
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
  // In a pre-order walk, we cannot erase as we walk.
  SmallVector<Operation *> toErase;
  WalkResult result = func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (op != func && isa<LIT::FuncOp>(op)) {
      return WalkResult::skip();

    } else if (isa<LIT::ParamReturnOp>(op)) {
      toErase.push_back(op);

    } else if (isa<HLCF::ControlFlowNode, HLCF::ControlFlowTerminator>(op) &&
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

    } else if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
      // Subsequent operations are always live if the call is.
      bool isLive = liveCfOps.contains(op);
      if (isLive)
        markNextOperationAsLive(op);

      SignatureType signature = call.getCalleeType();
      if (!signature.isThrows())
        return WalkResult::advance();
      // FIXME: `kgen.addressof` returns a `FunctionType` (function pointer),
      // which is an important feature to have, but that means we can't globally
      // see calls to a throwing function pointer.
      if (isa<AddressOfOp>(call)) {
        call.emitError("FIXME: cannot take address of throwing function");
        return WalkResult::interrupt();
      }

      // Pre-order traversal will not visit any of the created operations. The
      // throwing call means the except region is potentially live.
      if (isLive)
        if (auto tryOp = op->getParentOfType<LIT::TryOp>())
          markNextInRegionAsLive(tryOp.getExceptRegion());

      // Nothing to do if this is an async call.
      if (signature.isAsync())
        return WalkResult::advance();

      // We need to update the result types of the function.
      ImplicitLocOpBuilder b(call.getLoc(), OpBuilder(call->getNextNode()));
      Operation *newCall = b.clone(*call);
      Type resultType = call->getResultTypes().front();
      newCall->getResult(0).setType(VariantType::get({errType, resultType}));
      call->replaceAllUsesWith(ArrayRef(createUnwrapOrPropagate(
          b, func, newCall->getResult(0), errType, resultType)));
      toErase.push_back(call);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  for (Operation *op : toErase)
    op->erase();

  // Lower all the terminators as they are encountered.
  auto errorOr = [&] {
    return VariantType::get({errType, func.getResultType()});
  };
  SmallVector<Block *> deadBlocks;
  for (Operation *op : terminators) {
    // Ignore dead operations.
    if (op->getBlock() != &op->getParentRegion()->front())
      continue;

    ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      ValueRange operands = returnOp.getOperands();
      Value result;
      if (func.isThrows()) {
        result = b.create<VariantCreateOp>(errorOr(), operands.front());
        operands = result;
      }
      b.create<KGEN::ReturnOp>(operands);

    } else if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      auto tryOp = raiseOp->getParentOfType<LIT::TryOp>();
      if (tryOp &&
          tryOp.getTryRegion().isAncestor(raiseOp->getBlock()->getParent()))
        b.create<LIT::TryRaiseOp>(raiseOp.getError());
      else
        b.create<KGEN::ReturnOp>(
            Value(b.create<VariantCreateOp>(errorOr(), raiseOp.getError())));

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
  // Bind the result parameter values if there are any.
  if (resultParams) {
    OpBuilder b(terminator);
    b.create<ParamResultBindOp>(resultParams->loc, resultParams->values);
  }

  // If the function is explicitly terminated with a `return`, we're good.
  if (!isa<LIT::EndFuncOp>(terminator))
    return success();

  // If the endfunc isn't live, then it doesn't matter, there must have been a
  // return/raise before this.
  if (liveCfOps.contains(terminator)) {
    // A return is required if the function has result parameters, or it has
    // a non-none result.
    if (!func.getResultParams().empty() ||
        !isa<LIT::NoneType>(func.getResultType()) ||
        func.getSignature().hasMemoryOnlyResult())
      return terminator->emitError(
          "return expected at end of function with results");
  }

  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(terminator));
  Value retVal;
  if (!isa<LIT::NoneType>(func.getResultType())) {
    retVal = b.create<StaticUndefOp>(func.getResultType());
  } else {
    // The function returns none.
    retVal = b.create<ParamConstantOp>(b.getAttr<LIT::NoneAttr>());
  }

  // Wrap the result value if necessary.
  if (func.isThrows())
    retVal = b.create<VariantCreateOp>(errorOr(), retVal);
  b.create<KGEN::ReturnOp>(retVal);
  terminator->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// lowerThrows
//===----------------------------------------------------------------------===//

/// Lower `<...>(...) throws -> T` to `<...>(...) -> ErrorOr<T>`. Update all
/// callsites to signature types to reflect this change.
static void lowerThrows(DeclRefType errType, Operation *op) {
  // Replace every throwing signature type with a variant result type and
  // every async signature type with a coroutine handle result type.
  Builder b(op->getContext());
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SignatureType sigType) {
    if (!sigType.isThrows())
      return sigType;
    // Wrap the result type with the appropriate type.
    Type type = VariantType::get({errType, sigType.getValueResults().front()});
    // Clear the `throws` bit.
    return SignatureType::get(
        sigType.getInputParams(), sigType.getResultParams(),
        b.getFunctionType(sigType.getValueInputs(), type),
        b.getAttr<MetadataAttr>(
            sigType.getValueInputConventions(), sigType.getDefaultArguments(),
            bitEnumClear(sigType.getFnEffects(), FnEffects::Throws)));
  });
  replacer.recursivelyReplaceElementsIn(
      op, /*replaceAttrs=*/true, /*replaceLocs=*/false, /*replaceTypes=*/true);
}

//===----------------------------------------------------------------------===//
// lowerNestedFunctions
//===----------------------------------------------------------------------===//

/// Get a top-level function, lower all functions nested inside that function.
static LogicalResult lowerNestedFunctions(LIT::FuncOp topLevelFunc,
                                          mlir::SymbolTableAnalysis &analysis) {
  WalkResult result = topLevelFunc.walk([&](LIT::FuncOp func) {
    if (func == topLevelFunc)
      return WalkResult::advance();
    // Process a nested function by lowering it straight to a
    // `kgen.param.declare.region`. Nested functions are denoted with an
    // parameter declaration on the function declaration.
    ParamDeclAttr decl = func.getParamDeclAttr();
    if (!decl) {
      func.emitError("nested function must have a parameter declaration");
      return WalkResult::interrupt();
    }
    ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(func));
    auto region = b.create<ParamDeclareRegionOp>(
        decl, func.getSignature(), ArrayRef<ConstraintAttr>(),
        /*isolated=*/false, func.getAlwaysInlineLevel());
    region.getBodyRegion().takeBody(func.getBodyRegion());
    func.erase();
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLITTERMINATORS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerTerminatorsPass
    : impl::LowerLITTerminatorsBase<LowerTerminatorsPass> {
  using LowerLITTerminatorsBase::LowerLITTerminatorsBase;

  void runOnOperation() override {
    // Look for an error type declaration.
    DeclRefType errType;
    getOperation().walk([&](LIT::StructDeclOp decl) {
      if (decl.getName() != "Error" || !decl.getInputParams().empty())
        return;
      // Reconstruct the full symbol reference.
      errType = DeclRefType::get(
          LIT::getFullyResolvedSymbolRef(cast<mlir::SymbolOpInterface>(*decl)));
    });
    // Walk all functions.
    WalkResult result = getOperation().walk([&](LIT::FuncOp func) {
      if (failed(lowerLexicalTerminators(errType, func)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    // Lower all functions that throw.
    if (errType)
      lowerThrows(errType, getOperation());

    // Lower nested functions by converting them to region declarations. Walk
    // all top-level functions and gather nested functions.
    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    auto walkFn = [&](LIT::FuncOp func) {
      if (failed(lowerNestedFunctions(func, analysis)))
        return WalkResult::interrupt();
      return WalkResult::skip();
    };
    if (getOperation()
            ->walk<mlir::WalkOrder::PreOrder>(walkFn)
            .wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
