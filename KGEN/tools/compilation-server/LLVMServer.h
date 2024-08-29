//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/FunctionExtras.h"

namespace M::KGEN::CSP {

/// This class implements all of the LLVM specific functionality
/// necessary for a compilation server.
class LLVMServer {
public:
  LLVMServer();
  ~LLVMServer();

  /// Compile LLVM bitcode represented as base64 encoded string.
  /// For testing purposes, return emitted string or "error". This
  /// will change once the implementation is completed.
  std::string emitArchive(StringRef mlirModule);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::KGEN::CSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
