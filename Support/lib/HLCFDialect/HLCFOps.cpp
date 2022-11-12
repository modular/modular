//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace HLCF;

//===----------------------------------------------------------------------===//
// Control-Flow Verification
//===----------------------------------------------------------------------===//

namespace {
/// This object contains context about control-flow trees and uses that to
/// verify them. Keep a stack of control-flow scopes as we walk the tree and
/// verify that each terminator has a valid parent somewhere up the stack and
/// check that the return types match.
class ControlFlowVerifier {
public:
  explicit ControlFlowVerifier(Operation *root) : root(root) {}

  /// Verify the control-flow tree from this operation if it is a root node.
  static LogicalResult verifyIfRoot(Operation *op);

private:
  /// The root of the control-flow tree.
  Operation *root;

  /// The current stack of control-flow scopes. We only need to track loops
  /// since yield terminators are structurally required to have if parents.
  SmallVector<LoopOp> scopes;

  /// Verify a terminator.
  LogicalResult verifyTerminator(Operation *op);

  /// Verify a node.
  LogicalResult verifyNode(Operation *op);
};
} // namespace

/// Verify two type ranges match.
static LogicalResult verifyTypes(TypeRange lhs, TypeRange rhs, Operation *op,
                                 Operation *parent, StringRef kind,
                                 StringRef node) {
  if (lhs.size() != rhs.size()) {
    return (op->emitOpError("specifies ")
            << lhs.size() << ' ' << kind << "s but surrounding " << node
            << " expects " << rhs.size())
               .attachNote(parent->getLoc())
           << "see " << node << " here";
  }
  for (auto [idx, lhsType, rhsType] :
       llvm::zip(llvm::seq<unsigned>(0, lhs.size()), lhs, rhs)) {
    if (lhsType == rhsType)
      continue;
    return (op->emitOpError("operand #")
            << idx << " type " << lhsType << " does not match expected " << kind
            << " type " << rhsType)
               .attachNote(parent->getLoc())
           << "see " << node << " here";
  }
  return success();
}

LogicalResult ControlFlowVerifier::verifyTerminator(Operation *op) {
  if (isa<ReturnOp>(op)) {
    auto function = dyn_cast<mlir::FunctionOpInterface>(root);
    if (!function) {
      return op->emitOpError("is not nested within a function")
                 .attachNote(root->getLoc())
             << "see control-flow root here";
    }
    return verifyTypes(op->getOperandTypes(), function.getResultTypes(), op,
                       function, "return value", "function");
  }

  if (isa<YieldOp>(op)) {
    // The parent must be an if.
    return verifyTypes(op->getOperandTypes(),
                       op->getParentOp()->getResultTypes(), op,
                       op->getParentOp(), "result", "if");
  }

  assert((isa<BreakOp, ContinueOp>(op)));
  if (scopes.empty()) {
    return op->emitOpError("is not nested within an 'hlcf.loop' operation")
               .attachNote(root->getLoc())
           << "see control-flow root here";
  }
  if (isa<BreakOp>(op))
    return verifyTypes(op->getOperandTypes(), scopes.back().getResultTypes(),
                       op, scopes.back(), "result", "loop");
  return verifyTypes(op->getOperandTypes(), scopes.back().getOperandTypes(), op,
                     scopes.back(), "argument", "loop");
}

LogicalResult ControlFlowVerifier::verifyNode(Operation *op) {
  // Push a loop scoe if this node defines a new one.
  auto loop = dyn_cast<LoopOp>(op);
  if (loop)
    scopes.push_back(loop);

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (block.getTerminator()->hasTrait<OpTrait::ControlFlowTerminator>() &&
          failed(verifyTerminator(block.getTerminator())))
        return failure();
      for (Operation &op : block.without_terminator())
        if (op.hasTrait<OpTrait::ControlFlowNode>() && failed(verifyNode(&op)))
          return failure();
    }
  }

  // Pop a loop scope if this node defines one.
  if (loop)
    scopes.pop_back();
  return success();
}

LogicalResult ControlFlowVerifier::verifyIfRoot(Operation *op) {
  // Verify the operation if is a root operation or if it is the root of a
  // subtree rooted at a function.
  Operation *root;
  if (isa<mlir::FunctionOpInterface>(op->getParentOp()))
    root = op->getParentOp();
  else if (!op->getParentOp()->hasTrait<OpTrait::ControlFlowNode>())
    root = op;
  else
    return success();
  return ControlFlowVerifier(root).verifyNode(op);
}

//===----------------------------------------------------------------------===//
// Operation Traits
//===----------------------------------------------------------------------===//

LogicalResult HLCF::verifyControlFlowNode(Operation *op) {
  // Verify that all immediate terminators without successors are HLCF
  // terminators.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      if (!terminator->getNumSuccessors() &&
          !terminator->hasTrait<OpTrait::ControlFlowTerminator>()) {
        return (op->emitOpError("expected terminator without successors to be "
                                "a control-flow terminator but got '")
                << terminator->getName() << "'")
                   .attachNote(terminator->getLoc())
               << "see invalid terminator here";
      }
    }
  }

  // If this operation is a root, verify the tree starting from here.
  return ControlFlowVerifier::verifyIfRoot(op);
}

LogicalResult HLCF::verifyControlFlowTerminator(Operation *op) {
  // Verify that the terminator's parent is an HLCF operation.
  Operation *parent = op->getParentOp();
  if (parent->hasTrait<OpTrait::ControlFlowNode>())
    return success();
  return (op->emitOpError("expected parent operation to be a control-flow "
                          "operation but got '")
          << parent->getName() << "'")
             .attachNote(parent->getLoc())
         << "see invalid parent here";
}

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

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

void BreakOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  if (!isa<LoopOp>((*this)->getParentOp()))
    effects.emplace_back(mlir::MemoryEffects::Write::get());
}

mlir::Speculation::Speculatability BreakOp::getSpeculatability() {
  return isa<LoopOp>((*this)->getParentOp())
             ? mlir::Speculation::Speculatable
             : mlir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/HLCFDialect/HLCF.cpp.inc"
