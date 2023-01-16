//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// LowerTerminators
//===----------------------------------------------------------------------===//

/// Return true if the result type is nominally a none type.
static bool isNoneResultType(FuncInterface func) {
  Type type = func.getResultTypes().front();
  if (bitEnumContainsAny(func.getConventions().getFnEffects(),
                         FnEffects::Throws))
    type = cast<POP::VariantType>(type).getType(1);
  return isa<LIT::NoneType>(type);
}

/// Lower all lexical terminators in the function and remove dead code.
static LogicalResult lowerLexicalTerminators(FuncInterface func) {
  auto funcItf = cast<mlir::FunctionOpInterface>(*func);
  if (funcItf.getFunctionBody().empty())
    return success();

  mlir::IRRewriter b(func.getContext());

  LIT::ReturnOp firstResultParamsReturn;
  ParameterExprArrayAttr resultParams;

  SmallVector<Operation *> terminators;
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Don't walk into nested functions.
    if (op != func && isa<FuncInterface>(op))
      return WalkResult::skip();
    if (isa<LIT::ReturnOp, LIT::RaiseOp, LIT::BreakOp, LIT::ContinueOp>(op))
      terminators.push_back(op);
    return WalkResult::advance();
  });

  SmallVector<Block *> deadBlocks;
  for (Operation *op : terminators) {
    // Ignore dead operations.
    if (op->getBlock() != &op->getParentRegion()->front())
      continue;

    b.setInsertionPoint(op);
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      if (resultParams && returnOp.getParametersAttr() != resultParams) {
        return returnOp
                   .emitError("function return defines different result "
                              "meta-parameters than previous return statement")
                   .attachNote(firstResultParamsReturn.getLoc())
               << "see conflicting result meta-parameters here";
      }
      firstResultParamsReturn = returnOp;
      resultParams = returnOp.getParametersAttr();

      if (op->getParentOp() == func)
        b.create<KGEN::ReturnOp>(op->getLoc(), resultParams,
                                 returnOp.getOperands());
      else
        b.create<HLCF::ReturnOp>(op->getLoc(), returnOp.getOperands());

    } else if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      auto tryOp = raiseOp->getParentOfType<LIT::TryOp>();
      if (tryOp &&
          tryOp.getTryRegion().isAncestor(raiseOp->getBlock()->getParent())) {
        b.create<LIT::TryRaiseOp>(op->getLoc(), raiseOp.getError());
      } else {
        // TODO(#6449): Can't have result parameters in a function that raises.
        Value err = b.create<POP::VariantCreateOp>(
            op->getLoc(), func.getResultTypes().front(), raiseOp.getError());
        if (isa<LIT::FuncOp>(op->getParentOp()))
          b.create<KGEN::ReturnOp>(op->getLoc(), ArrayRef<TypedAttr>(), err);
        else
          b.create<HLCF::ReturnOp>(op->getLoc(), err);
      }

    } else if (auto breakOp = dyn_cast<LIT::BreakOp>(op)) {
      b.create<HLCF::BreakOp>(op->getLoc());

    } else if (auto continueOp = dyn_cast<LIT::ContinueOp>(op)) {
      b.create<HLCF::ContinueOp>(op->getLoc());

    } else {
      llvm_unreachable("unknown terminator");
    }

    // Check and warn about dead code.
    if (!op->getNextNode()->hasTrait<OpTrait::IsTerminator>())
      op->getNextNode()->emitWarning("unreachable code after ")
          << op->getName().stripDialect() << " statement";

    // Mark all subsequent operations as dead.
    deadBlocks.push_back(op->getBlock()->splitBlock(op));
  }

  // Remove all dead code.
  for (Block *block : deadBlocks)
    block->erase();

  // Check if the function lacks a top-level terminator. If the function
  // nominally returns `!lit.none`, then insert one. Otherwise, emit an error.
  Operation *terminator = &funcItf.getFunctionBody().front().back();
  if (!isa<LIT::EndFuncOp>(terminator))
    return success();
  if (!isNoneResultType(func) || !func.getResultParamTypes().empty())
    return terminator->emitError(
        "return expected at end of function with results");

  b.setInsertionPoint(terminator);
  Value none =
      b.create<ParamConstantOp>(func.getLoc(), b.getAttr<LIT::NoneAttr>());
  if (bitEnumContainsAny(func.getConventions().getFnEffects(),
                         FnEffects::Throws))
    none = b.create<POP::VariantCreateOp>(func.getLoc(),
                                          func.getResultTypes().front(), none);
  b.replaceOpWithNewOp<KGEN::ReturnOp>(terminator, ArrayRef<TypedAttr>(), none);
  return success();
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
    // Walk all top-level functions.
    WalkResult result = getOperation()->walk([](FuncInterface func) {
      if (failed(lowerLexicalTerminators(func)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
