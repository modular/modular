//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_COMPILERSUPPORT_CONTEXT_H
#define ASYNCRT_COMPILERSUPPORT_CONTEXT_H

#include "Support/Context.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Dialect.h"

namespace M {

/// Registers a context in via the MDialect.
void registerContext(mlir::DialectRegistry &registry, ContextRef &ref,
                     bool enableThreadPool = true);

/// Registers a context via the MDialect; convenience wrapper.
void registerContext(mlir::MLIRContext &ctx, ContextRef &ref,
                     bool enableThreadPool = true);

/// Loads a context from the given MLIRContext. It must have been previously
/// registered via a call to registerContext or this will trigger an assertion.
ContextRef loadContext(mlir::MLIRContext *ctx);

} // namespace M

#endif // ASYNCRT_COMPILERSUPPORT_CONTEXT_H
