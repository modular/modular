//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/CODialect/CODialect.h"
#include "KGEN/CODialect/COUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"

using namespace M;
using namespace KGEN;
using namespace CO;

//===----------------------------------------------------------------------===//
// CoroutineHandleOp
//===----------------------------------------------------------------------===//

LogicalResult CoroutineHandleOp::verify() {
  if (auto func = (*this)->getParentOfType<FuncOp>()) {
    if (func.getNumResults() != 1) {
      return emitOpError("surrounding function must have 1 result")
                 .attachNote(func.getLoc())
             << "see function here";
    }
    Type resultType = func.getResultTypes().front();
    if (resultType != getType()) {
      return emitOpError("surrounding function result type does not match "
                         "coroutine handle type")
                 .attachNote(func.getLoc())
             << "surrounding function returns " << resultType;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CODialect
//===----------------------------------------------------------------------===//

void CODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "KGEN/CODialect/CO.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/CODialect/CO.cpp.inc"
