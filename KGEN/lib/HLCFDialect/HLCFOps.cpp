//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
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
  auto range =
      llvm::make_range(op->getNextNode()->getIterator(),
                       op->getParentRegion()->front().getOperations().end());

  // We have to delete ops in reverse order to avoid dealing with value uses.
  SmallVector<Operation *> worklist;
  for (Operation &op : range)
    worklist.push_back(&op);

  while (!worklist.empty())
    rewriter.eraseOp(worklist.pop_back_val());
}

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
} // namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<RemoveStaticCondition>(context);
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

/// Return true if the operation is a loop and has a matching label.
static bool isMatchingLoop(Operation *op, StringAttr label) {
  if (auto loop = dyn_cast<LoopOp>(op))
    return !label || loop.getLabelAttr() == label;
  return false;
}

/// Return the nearest enclosing matching loop. This runs on valid IR, so it
/// must find a matching loop.
static LoopOp getParentLoop(Operation *op, StringAttr label) {
  LoopOp loop = op->getParentOfType<LoopOp>();
  while (!isMatchingLoop(loop, label))
    loop = loop->getParentOfType<LoopOp>();
  return loop;
}

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
