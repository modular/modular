//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_H
#define KGEN_ELABORATOR_H

#include "Support/LLVMCompilerForwardDecls.h"
#include <filesystem>

namespace mlir {
class SymbolTableAnalysis;
} // namespace mlir

namespace M {
class TargetInfoAttr;
namespace LLCL {
class Runtime;
} // namespace LLCL

namespace KGEN {
class GeneratorOp;
} // namespace KGEN

/// Resolve the includes in the specified module, incorporating implementation
/// logic from the included files found in `searchPaths`. `includedFiles` is
/// an optional set that is populated with the files that were included during
/// the resolution process.
LogicalResult
resolveIncludes(SymbolTable &symtab,
                ArrayRef<std::filesystem::path> searchPaths,
                SmallVectorImpl<std::string> *includedFiles = nullptr);

/// Elaborate generators in the specified module, incorporating implementation
/// logic from the specified library.  On error, diagnostics are emitted and the
/// primary file isn't completely lowered.
LogicalResult elaborateGenerators(mlir::SymbolTableAnalysis &symtab,
                                  LLCL::Runtime &runtime, TargetInfoAttr target,
                                  ArrayRef<KGEN::GeneratorOp> generators,
                                  bool useOldImpl = false,
                                  bool enableSearch = false);

} // namespace M

#endif // KGEN_ELABORATOR_H
