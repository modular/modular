//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace HLCF;

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
  return success();
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
