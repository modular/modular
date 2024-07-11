//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJO_LSP_LSPSERVER_H
#define KGEN_LIB_MOJO_LSP_LSPSERVER_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir::lsp {
class JSONTransport;
} // namespace mlir::lsp

namespace M::AsyncRT {
class WorkQueue;
} // namespace M::AsyncRT

namespace M::KGEN::LIT {
/// Run the main loop using the given transport.
mlir::LogicalResult runMojoLSPServer(mlir::lsp::JSONTransport &transport,
                                     bool singleThreaded, bool waitOnShutdown,
                                     ArrayRef<std::string> includeDirs);
} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJO_LSP_LSPSERVER_H
