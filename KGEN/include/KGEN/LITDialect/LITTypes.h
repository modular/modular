//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITTYPES_H
#define KGEN_LITDIALECT_LITTYPES_H

#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "Support/ForwardDecls.h"

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.h.inc"

namespace M::KGEN::LIT {
/// Get a reference to the standard library error type. The standard library
/// error type is `@Error` and is expected to be visible in every compilation
/// unit.
DeclRefType getLibraryErrorType(MLIRContext *ctx);
} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITTYPES_H
