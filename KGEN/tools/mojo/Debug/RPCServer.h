//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DEBUG_RPC_SERVER_H
#define MOJO_DEBUG_RPC_SERVER_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include <optional>

namespace M {
/// Starts an `attach` debug session with an existing RPC debug server.
/// If `dryRun` is specified, then the request payload is printed to the
/// standard output instead.
ErrorOrSuccess invokeAttachRPC(bool dryRun, ArrayRef<int> rpcPorts,
                               const std::optional<StringRef> &pid,
                               const std::optional<StringRef> &processName);

/// Starts a `launch` debug session with an existing RPC debug server.
/// If `dryRun` is specified, then the request payload is printed to the
/// standard output instead.
ErrorOrSuccess invokeLaunchRPC(bool dryRun, ArrayRef<int> rpcPorts,
                               StringRef target, ArrayRef<std::string> runArgs,
                               StringRef rpcTerminal);
} // namespace M

#endif // MOJO_DEBUG_RPC_SERVER_H
