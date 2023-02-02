//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef EMIT_FUNC_HEADER_H
#define EMIT_FUNC_HEADER_H

#include "KGEN/KGENDialect/KGENOps.h"

namespace M::KGEN {
class ObjectCompiler;

/// Emit the header for a set of exported functions.
LogicalResult emitHeader(ObjectCompiler &compiler, StringRef filename);
} // namespace M::KGEN

#endif // EMIT_FUNC_HEADER_H
