//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H

#include "Protocol.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

namespace M::KGEN::CSP {

/// This class implements all of the LLVM specific functionality
/// necessary for a compilation server.
class LLVMServer {
public:
  LLVMServer(LLVMServer &) = delete;
  LLVMServer(LLVMServer &&) = delete;
  ~LLVMServer();

  /// Create a new LLVMServer instance.
  static ErrorOr<std::unique_ptr<LLVMServer>> create(bool singleThreaded);

  /// Execute ObjectCompiler::emitArchive() and return the resulting
  /// archive.
  std::string emitArchive(const EmitArchiveParams &params);

  /// Receive MLIR module, convert it to an op and send it back.
  std::string echoMLIR(StringRef module);

private:
  struct Impl;
  LLVMServer(std::unique_ptr<Impl> &&);
  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::CSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
