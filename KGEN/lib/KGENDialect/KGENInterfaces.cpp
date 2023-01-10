//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

LogicalResult impl::verifyCallOp(KGENCallOpInterface op) {
  if (!op.getCallee())
    return success();

  // Disallow calls from within a concrete function from calling anything with
  // input or output parameters.
  auto func = op->getParentOfType<FuncOp>();
  if (func && !op.getParamValues().empty()) {
    return op.emitOpError("cannot reference generator with input parameters "
                          "from within a concrete 'kgen.func'")
               .attachNote(func.getLoc())
           << "within 'kgen.func' @" << func.getName();
  }

  if (!op.isAllowedInFunc() && func)
    return op.emitOpError("is only allowed in generators pre-elaboration");

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.cpp.inc"
