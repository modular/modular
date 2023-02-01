//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_LITLSP_LSPSERVER_H
#define KGEN_LIB_LITLSP_LSPSERVER_H

namespace mlir {
struct LogicalResult;

namespace lsp {
class JSONTransport;
} // namespace lsp
} // namespace mlir

namespace M::KGEN::LIT {
class LITServer;

/// Run the main loop using the given LIT server and transport.
mlir::LogicalResult runLitLSPServer(LITServer &server,
                                    mlir::lsp::JSONTransport &transport);
} // namespace M::KGEN::LIT

#endif // KGEN_LIB_LITLSP_LSPSERVER_H
