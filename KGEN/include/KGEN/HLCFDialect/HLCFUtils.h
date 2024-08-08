//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_HLCFDIALECT_HLCFUTILS_H
#define KGEN_HLCFDIALECT_HLCFUTILS_H

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "mlir/IR/OpImplementation.h"

namespace M::HLCF {
/// Return true if the operation is a loop and has a matching label.
bool isMatchingLoop(Operation *op, StringAttr label);

/// Return the nearest enclosing matching loop. This runs on valid IR, so it
/// must find a matching loop.
LoopOp getParentLoop(Operation *op, StringAttr label);

/// Check if the child loop is nested in the parentToCheck loop.
bool isParentLoop(LoopOp child, LoopOp parentToCheck);

/// Get the parent operation of a terminator.
Operation *getParentNode(HLCF::ControlFlowTerminator term);

/// Given an elif op, transform into multiple IfOps. Return top IfOp.
IfOp replaceElifWithIfOps(ElifOp elifOp);

ParseResult parseLoop(OpAsmParser &p,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                      SmallVectorImpl<Type> &operandTypes,
                      SmallVectorImpl<Type> &resultTypes, Region &body);
void printLoop(OpAsmPrinter &p, Operation *op, ValueRange operands,
               TypeRange operandTypes, TypeRange resultTypes, Region &body);

} // namespace M::HLCF

#endif // KGEN_HLCFDIALECT_HLCFUTILS_H
