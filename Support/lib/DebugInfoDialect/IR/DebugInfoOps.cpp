//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Support/DebugInfoDialect/IR/DebugInfo.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

/// Implement the interpret hook for this operation. Since the operation has no
/// results, we cannot use the fold hook.
ErrorTreeOrSuccess ValueOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  return success();
}

LogicalResult ValueOp::verify() {
  DILocalVariableAttr varAttr = getValueInfo();
  if (DIScopeAttr scope = extractScope(getLoc())) {
    if (varAttr.getScope() != scope) {
      return emitOpError("location scope must match variable scope: ")
             << scope << " vs. " << varAttr.getScope();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfo.cpp.inc"
