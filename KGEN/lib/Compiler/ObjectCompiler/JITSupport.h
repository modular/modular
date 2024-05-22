//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENEnums.h"
#include "llvm/ExecutionEngine/JITSymbol.h"

#ifndef KGEN_COMPILER_ORCSUPPORT_H
#define KGEN_COMPILER_ORCSUPPORT_H

namespace M::KGEN {
struct ExportedSymbol;

/// Get the OrcJIT symbol flags for an exported symbol.
llvm::JITSymbolFlags getFlagsForExportedSymbol(const ExportedSymbol &symbol);

/// These are the symbol flags to use for the global init and deinit functions.
llvm::JITSymbolFlags getGlobalFnSymbolFlags();
} // namespace M::KGEN

#endif // KGEN_COMPILER_ORCSUPPORT_H
