//===- EmitKernelObject.h -------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef EMIT_KERNEL_OBJECT_H
#define EMIT_KERNEL_OBJECT_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/ErrorOr.h"
#include <filesystem>

namespace M::KGEN {
class ExecutionEngine;

/// Emit the object file for the kernel `k` to the path `objPath.
LogicalResult emitObjectForKernel(ExecutionEngine &engine, FuncOp k,
                                  const std::filesystem::path &objPath);

} // namespace M::KGEN

#endif // EMIT_KERNEL_OBJECT_H
