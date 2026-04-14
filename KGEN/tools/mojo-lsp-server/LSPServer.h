//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJO_LSP_LSPSERVER_H
#define KGEN_LIB_MOJO_LSP_LSPSERVER_H

#include "KGEN/Support/CompilerProfiling.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace llvm::lsp {
class JSONTransport;
} // namespace llvm::lsp

namespace M::MLRT {
class WorkQueue;
} // namespace M::MLRT

namespace M::KGEN::LIT {
/// Run the main loop using the given transport.
mlir::LogicalResult
runMojoLSPServer(llvm::lsp::JSONTransport &transport, bool singleThreaded,
                 bool waitOnShutdown, ArrayRef<std::string> includeDirs,
                 std::unique_ptr<KGEN::TraceProfiler> profiler,
                 bool skipDocstringCodeBlockChecks = false);
} // namespace M::KGEN::LIT

#endif // KGEN_LIB_MOJO_LSP_LSPSERVER_H
