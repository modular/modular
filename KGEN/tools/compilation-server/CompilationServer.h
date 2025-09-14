//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_COMPILATION_SERVER_H
#define KGEN_TOOLS_COMPILATION_SERVER_H

#include "mlir/Support/LogicalResult.h"

namespace mlir::lsp {
class JSONTransport;
} // namespace mlir::lsp

namespace M::KGEN {
/// Run the main loop using the given transport.
mlir::LogicalResult runCompilationServer(mlir::lsp::JSONTransport &transport,
                                         bool singleThreaded);
} // namespace M::KGEN

#endif // KGEN_TOOLS_COMPILATION_SERVER_H
