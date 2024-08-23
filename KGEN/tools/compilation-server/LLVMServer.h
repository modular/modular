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
  LLVMServer() = default;
  ~LLVMServer() = default;

  /// Compile LLVM bitcode represented as base64 encoded string.
  void compileBitcode(const std::string &bitcode);
};

} // namespace M::KGEN::CSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJO_SERVER_H
