//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MDIALECT_MDIALECT_H
#define SUPPORT_MDIALECT_MDIALECT_H

#include "Support/Context.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Dialect.h"

namespace M {

/// Registers a context in via the MDialect.
void registerContext(mlir::DialectRegistry &registry, context::ContextRef &ref);

/// Loads a context from the given MLIRContext. It must have been previously
/// registered via a call to registerContext or this will trigger an assertion.
context::ContextRef loadContext(mlir::MLIRContext *ctx);

} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.h.inc"

#endif // SUPPORT_MDIALECT_MDIALECT_H
