//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJO_LSP_LSPSERVER_H
#define KGEN_LIB_MOJO_LSP_LSPSERVER_H

#include <memory>

namespace mlir {
struct LogicalResult;

namespace lsp {
class JSONTransport;
} // namespace lsp
} // namespace mlir

namespace M::LLCL {
class WorkQueue;
} // namespace M::LLCL

namespace M::KGEN::LIT {
/// Run the main loop using the given transport.
mlir::LogicalResult runMojoLSPServer(mlir::lsp::JSONTransport &transport,
                                     bool singleThreaded, bool waitOnShutdown);
} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJO_LSP_LSPSERVER_H
