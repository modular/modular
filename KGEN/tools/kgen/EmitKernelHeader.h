//===- EmitKernelHeader.h -------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef EMIT_KERNEL_HEADER_H
#define EMIT_KERNEL_HEADER_H

#include "KGEN/KGENDialect/KGENOps.h"

namespace M::KGEN {
/// Emit the header for a func.
LogicalResult emitHeaderForFunc(FuncOp func, StringRef filename);
} // namespace M::KGEN

#endif // EMIT_KERNEL_HEADER_H
