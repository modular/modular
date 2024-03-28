//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOBUILD_SERVER_H
#define KGEN_MOJOBUILD_SERVER_H

#include "Support/SymbolExport.h"
#include "llvm/Support/Compiler.h"

/// Launches the Mojo build server, which listens for messages until it is shut
/// down. Returns an exit code indicating whether the server exited without
/// error.
MODULAR_EXPORT LLVM_ATTRIBUTE_USED int mojoBuildServerMain();

#endif // KGEN_MOJOBUILD_SERVER_H
