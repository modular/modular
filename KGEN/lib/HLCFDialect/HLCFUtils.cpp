//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/HLCFDialect/HLCFOps.h"

using namespace M;
using namespace HLCF;

/// Return true if the operation is a loop and has a matching label.
bool HLCF::isMatchingLoop(Operation *op, StringAttr label) {
  if (auto loop = dyn_cast<LoopOp>(op))
    return !label || loop.getLabelAttr() == label;
  return false;
}

/// Return the nearest enclosing matching loop or nullptr if nothing found.
LoopOp HLCF::getParentLoop(Operation *op, StringAttr label) {
  LoopOp loop = op->getParentOfType<LoopOp>();
  while (!isMatchingLoop(loop, label))
    loop = loop->getParentOfType<LoopOp>();
  return loop;
}

/// Check if the child loop is nested in the parentToCheck loop.
bool HLCF::isParentLoop(LoopOp child, LoopOp parentToCheck) {
  LoopOp parent = child;
  while (parent && parent != parentToCheck)
    parent = parent->getParentOfType<LoopOp>();
  return parent == parentToCheck;
}

/// Get the parent operation of a terminator.
Operation *HLCF::getParentNode(HLCF::ControlFlowTerminator term) {
  Operation *op = term->getParentOp();
  while (!term.isParentNode(op))
    op = op->getParentOp();
  return op;
}
