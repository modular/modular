//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef EMIT_FUNC_OBJECT_H
#define EMIT_FUNC_OBJECT_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/ErrorOr.h"
#include <filesystem>

namespace M::KGEN {
class ExecutionEngine;

/// Emit the object file for the func `fn` to the path `objPath.
LogicalResult emitObjectForFunc(ExecutionEngine &engine, FuncOp fn,
                                const std::filesystem::path &objPath);

} // namespace M::KGEN

#endif // EMIT_FUNC_OBJECT_H
