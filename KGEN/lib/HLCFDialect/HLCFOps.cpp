//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace HLCF;

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

/// arrow-type-list ::= `->` (`(` (type (`,` type)*)? `)`) | type
/// loop-arg ::= value `=` value `:` type
/// loop ::= (`(` (loop-arg (`,` loop-arg)*)? `)` arrow-type-list)? region
static ParseResult
parseLoop(OpAsmParser &p,
          SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
          SmallVectorImpl<Type> &operandTypes,
          SmallVectorImpl<Type> &resultTypes, Region &body) {
  SmallVector<OpAsmParser::Argument> loopArgs;

  // Parse the optional loop signature.
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseOptionalRParen()) {
      OpAsmParser::Argument arg;
      OpAsmParser::UnresolvedOperand operand;
      auto parseEl = [&]() -> ParseResult {
        if (p.parseArgument(arg) || p.parseEqual() || p.parseOperand(operand) ||
            p.parseColonType(arg.type))
          return failure();
        loopArgs.push_back(arg);
        operands.push_back(operand);
        operandTypes.push_back(arg.type);
        return success();
      };
      if (p.parseCommaSeparatedList(parseEl) || p.parseRParen())
        return failure();
    }
    if (p.parseOptionalArrowTypeList(resultTypes))
      return failure();
  }
  return p.parseRegion(body, loopArgs);
}

static void printLoop(OpAsmPrinter &p, Operation *op, ValueRange operands,
                      TypeRange operandTypes, TypeRange resultTypes,
                      Region &body) {
  if (!operandTypes.empty() || !resultTypes.empty()) {
    p << " (";
    llvm::interleaveComma(llvm::enumerate(operands), p, [&](auto it) {
      auto [i, operand] = it;
      p << body.getArgument(i) << " = " << operand << " : " << operandTypes[i];
    });
    p << ")";
    p.printOptionalArrowTypeList(resultTypes);
  }
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult LoopOp::verify() {
  if (getOperandTypes() != getBody().getArgumentTypes())
    return emitOpError("operand types do not match body region argument types");
  return success();
}

void LoopOp::getEntryTargets(ArrayRef<Attribute> operands,
                             SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  targets.emplace_back(0, getOperands());
}

ValueRange LoopOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target == 0);
  return getBody().getArguments();
}

ErrorTreeOr<SuccessType> LoopOp::interpret(ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  state.transferControlFlowTo(&getBody().front(), operands);
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::getEntryTargets(ArrayRef<Attribute> operands,
                           SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  if (auto cond = dyn_cast_or_null<BoolAttr>(operands.front())) {
    targets.emplace_back(cond.getValue() ? 0 : 1);
  } else {
    targets.emplace_back(0);
    targets.emplace_back(1);
  }
}

ValueRange IfOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target == 0 || *target == 1);
  return {};
}

ErrorTreeOr<SuccessType> IfOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  auto cond = dyn_cast_if_present<BoolAttr>(operands[0]);
  if (!cond)
    return ErrorTree(getLoc(), "non-constant condition");

  state.transferControlFlowTo(
      &(cond.getValue() ? getThenRegion() : getElseRegion()).front(), {});
  return success();
}

OpBuilder IfOp::getThenBodyBuilder() {
  assert(!getThenRegion().empty() && "Need a then block");
  return OpBuilder::atBlockEnd(&getThenRegion().front());
}

OpBuilder IfOp::getElseBodyBuilder() {
  assert(!getElseRegion().empty() && "Need an else block");
  return OpBuilder::atBlockEnd(&getElseRegion().front());
}

/// Erase all operations following the given OP in its parent region. The OP
/// itself does not get deleted.
static void eraseOpsAfter(PatternRewriter &rewriter, Operation *op) {
  Block *toErase =
      rewriter.splitBlock(op->getBlock(), op->getNextNode()->getIterator());
  rewriter.eraseBlock(toErase);
}

Block &IfOp::getThenBlock() { return getThenRegion().front(); }

Block &IfOp::getElseBlock() { return getElseRegion().front(); }

Operation *IfOp::getThenTerminator() { return getThenBlock().getTerminator(); }

Operation *IfOp::getElseTerminator() { return getElseBlock().getTerminator(); }

/// Replace the given op with a region. If the region ends with YieldOp then
/// uses of the results of the original op will be replaced with the
/// corresponding yielded values. Otherwise, the region must be ending with a
/// Return or a similar terminator - in that case we erase all the ops after the
/// original op as dead code.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  if (isa<YieldOp>(terminator)) {
    // If the op block ends with yield, we rewire the values in the remaining of
    // the parent block to use the yielded values.
    rewriter.replaceOp(op, terminator->getOperands());
    rewriter.eraseOp(terminator);
  } else {
    // Delete all ops after the op - the block in the op ends with a terminator
    // that renders the remaining of the parent block dead.
    eraseOpsAfter(rewriter, op);
    rewriter.eraseOp(op);
  }
}

namespace {
/// If the IfOp condition is known at compile time, replace the IfOp with the
/// contents of the corresponding branch. If the block we're inserting doesn't
/// end with YieldOp, operations following the original IfOp will be discarded.
struct RemoveStaticCondition : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    BoolAttr condition;
    if (!matchPattern(op.getCond(), m_Constant(&condition)))
      return failure();

    Region &active =
        condition.getValue() ? op.getThenRegion() : op.getElseRegion();
    replaceOpWithRegion(rewriter, op, active);

    return success();
  }
};

/// If both IfOp branches end with return ops, replace the return ops with yield
/// ops and insert a new return op right after the if. All subsequent ops in the
/// basic block are erased.
///
/// Before:                    After:
/// {                          {
///   ...                        ...
///   if %cond {                 %x = if %cond {
///      A                          A
///      return %a                  yield %a
///   } else {                   } else {
///      B                          B
///      return %b                  yield %b
///   }                          }
///   C                          return %x
/// }                          }
struct HoistUnconditionalReturn : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: This should also work for BreakOp and ContinueOp (provided
    // thenTerm and elseTerm are of the same type)
    if (!isa<KGEN::ReturnOp>(op.getThenTerminator()) ||
        !isa<KGEN::ReturnOp>(op.getElseTerminator()))
      return failure();

    // Create a new IfOp and put a return right after it. We have to create new
    // op because the number of results might be different compared to the
    // original IfOp.
    auto newIfOp = rewriter.create<IfOp>(
        op.getLoc(), op.getThenTerminator()->getOperandTypes(), op.getCond());
    rewriter.create<KGEN::ReturnOp>(op.getLoc(), newIfOp->getResults());

    // Move the 'then' block from the original IfOp to the new one and replace
    // the return terminator with yield.
    rewriter.inlineRegionBefore(op.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.setInsertionPoint(newIfOp.getThenTerminator());
    rewriter.replaceOpWithNewOp<HLCF::YieldOp>(
        newIfOp.getThenTerminator(),
        newIfOp.getThenTerminator()->getOperands());

    // Same for the 'else' block.
    rewriter.inlineRegionBefore(op.getElseRegion(), newIfOp.getElseRegion(),
                                newIfOp.getElseRegion().begin());
    rewriter.setInsertionPoint(newIfOp.getElseTerminator());
    rewriter.replaceOpWithNewOp<HLCF::YieldOp>(
        newIfOp.getElseTerminator(),
        newIfOp.getElseTerminator()->getOperands());

    // Erase the original if and all the ops below it.
    eraseOpsAfter(rewriter, op);
    rewriter.eraseOp(op);

    return success();
  }
};

/// If one of the IfOp branches is Return, then we can try pulling the code
/// after the IfOp into the other branch and replace the return op with yield.
/// This allows us to hoist return to outer scopes, potentially enabling other
/// optimizations.
///
/// We can only perform this transformation if the IfOp's basic block ends with
/// a return op - in that case it is legal to insert a return after the IfOp,
/// which we want to do in this transformation.
///
///
/// Before:                    After:
/// {                          {
///   ...                        ...
///   %x = if %cond {            %x = if %cond {
///      %a = A                     %a = A
///      return %a                  yield %a
///   } else {                   } else {
///      %b = B                     %b = B
///      yield %b                   %t = C(%b)
///                                 yield %t
///   }                          }
///   %t = C(%x)                 return %x
///   return %t
/// }                          }
struct HoistConditionalReturn : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    Block &parentBlock = op->getParentRegion()->front();
    Operation *parentBlockTerm = parentBlock.getTerminator();

    Operation *thenTerm = op.getThenTerminator();
    Operation *elseTerm = op.getElseTerminator();

    // On the other hand, if neither of them is return, then we also have
    // nothing to do.
    // TODO: This should also work for BreakOp and ContinueOp (provided thenTerm
    // and elseTerm are of the same type)
    if (!isa<KGEN::ReturnOp>(thenTerm) && !isa<KGEN::ReturnOp>(elseTerm))
      return rewriter.notifyMatchFailure(
          op, "None of the branches ends with Return");

    // One of the terminators is Return, now make sure that the other one is
    // Yield.
    if (!isa<YieldOp>(thenTerm) && !isa<YieldOp>(elseTerm))
      return rewriter.notifyMatchFailure(
          op, "None of the branches ends with Yield");

    // If the parent block doesn't end with return, then we cannot return after
    // the IfOp, which is how we want to hoist return op from its branch. Hence,
    // bail out.
    if (!isa<KGEN::ReturnOp>(parentBlockTerm))
      return rewriter.notifyMatchFailure(
          op, "Parent block doesn't end with Return");

    // Now we know that we can transform this. Create a new IfOp (we can't use
    // the original IfOp because we might need a different number of result
    // values).
    auto newIfOp = rewriter.create<HLCF::IfOp>(
        op.getLoc(), parentBlockTerm->getOperandTypes(), op.getCond());

    // Move the original 'then' and 'else' basic blocks into the new IfOp.
    rewriter.inlineRegionBefore(op.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(op.getElseRegion(), newIfOp.getElseRegion(),
                                newIfOp.getElseRegion().begin());

    // Figure out which block contains Return and which one contains Yield.
    Operation *yieldTerm = nullptr, *returnTerm = nullptr;
    if (isa<KGEN::ReturnOp>(thenTerm)) {
      yieldTerm = elseTerm;
      returnTerm = thenTerm;
    } else {
      assert(isa<KGEN::ReturnOp>(elseTerm));
      yieldTerm = thenTerm;
      returnTerm = elseTerm;
    }

    // Move the ops from the parent block following the original IfOp to a
    // separate block and then move that block into the 'yield' block in the new
    // if.
    Block *remainderBlock =
        rewriter.splitBlock(op->getBlock(), op->getNextNode()->getIterator());
    rewriter.inlineBlockBefore(remainderBlock, yieldTerm->getBlock(),
                               yieldTerm->getBlock()->end());

    // The remainder block used to use return values of the original if op. We
    // now need to rewire that to values from the yield op.
    for (auto [idx, val] : llvm::enumerate(op->getResults()))
      rewriter.replaceAllUsesWith(val, yieldTerm->getOperand(idx));

    // And after that we can erase the yield op.
    rewriter.eraseOp(yieldTerm);

    // At this point our new IfOp has its then and else block constructed, but
    // ending with returns. We need to replace them with yields and insert a
    // return after the new if op.
    rewriter.create<KGEN::ReturnOp>(op.getLoc(), newIfOp->getResults());

    rewriter.setInsertionPoint(parentBlockTerm);
    rewriter.replaceOpWithNewOp<HLCF::YieldOp>(parentBlockTerm,
                                               parentBlockTerm->getOperands());
    rewriter.setInsertionPoint(returnTerm);
    rewriter.replaceOpWithNewOp<HLCF::YieldOp>(returnTerm,
                                               returnTerm->getOperands());

    // Finally, erase the original op.
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<RemoveStaticCondition, HoistUnconditionalReturn,
              HoistConditionalReturn>(context);
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

bool ContinueOp::isParentNode(Operation *op) {
  return isMatchingLoop(op, getLabelAttr());
}

void ContinueOp::getBranchTargets(ArrayRef<Attribute> operands,
                                  SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to the beginning of the body region.
  targets.emplace_back(0, getOperands());
}

ErrorTreeOr<SuccessType> ContinueOp::interpret(ArrayRef<Attribute> operands,
                                               InterpreterState &state) {
  LoopOp loop = getParentLoop(*this, getLabelAttr());
  state.transferControlFlowTo(&loop.getBody().front(), operands);
  return success();
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

void BreakOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  if (!isMatchingLoop((*this)->getParentOp(), getLabelAttr()))
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

mlir::Speculation::Speculatability BreakOp::getSpeculatability() {
  return isMatchingLoop((*this)->getParentOp(), getLabelAttr())
             ? mlir::Speculation::Speculatable
             : mlir::Speculation::NotSpeculatable;
}

bool BreakOp::isParentNode(Operation *op) {
  return isMatchingLoop(op, getLabelAttr());
}

void BreakOp::getBranchTargets(ArrayRef<Attribute> operands,
                               SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the loop operation.
  targets.emplace_back(std::nullopt, getOperands());
}

ErrorTreeOr<SuccessType> BreakOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  LoopOp loop = getParentLoop(*this, getLabelAttr());
  state.setReturnValues(operands);
  state.transferControlFlowTo(loop);
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

bool YieldOp::isParentNode(Operation *op) { return isa<IfOp>(op); }

void YieldOp::getBranchTargets(ArrayRef<Attribute> operands,
                               SmallVectorImpl<ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the if operation.
  targets.emplace_back(std::nullopt, getOperands());
}

ErrorTreeOr<SuccessType> YieldOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  auto ifOp = cast<IfOp>(getOperation()->getParentOp());
  state.setReturnValues(operands);
  state.transferControlFlowTo(ifOp);
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/HLCFDialect/HLCF.cpp.inc"
