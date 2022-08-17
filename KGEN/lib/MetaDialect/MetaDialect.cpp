//===- MetaDialect.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Meta dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "Support/LLVMCompilerForwardDecls.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/MetaDialect/MetaDialect.cpp.inc"

void MetaDialect::initialize() {
  // Register types.
  registerTypes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/MetaDialect/Meta.cpp.inc"
      >();
}

/// Registered hook to materialize a constant operation from a "meta" dialect
/// op that is folded.
Operation *MetaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // Integer constants can materialize into something specific.  We need this
  // for ops that fold in the context of kgen.kernel.
  if (isValidParameterExpr(value))
    return builder.create<ParamConstantOp>(loc, type, value);
  return nullptr;
}
