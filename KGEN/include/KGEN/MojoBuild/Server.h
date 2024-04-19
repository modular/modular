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
/// error. When `debug` is true, the server is configured to accept JSON
/// requests delimited by `// -----`, and format JSON responses in a "pretty"
/// way for printing.
MODULAR_EXPORT LLVM_ATTRIBUTE_USED int mojoBuildServerMain(bool debug);

#endif // KGEN_MOJOBUILD_SERVER_H
