//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_SLICINGUTILS_H
#define KGEN_TRANSFORMUTILS_SLICINGUTILS_H

#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
/// Produce a standalone MLIR module by slicing out the dependencies of the
/// provided exported ops.
OwningOpRef<ModuleOp>
produceStandaloneModule(const SymbolTable &symtab,
                        const KGEN::ExportMap &exportedSymbols);

/// Produce a standalone MLIR module by slicing out the dependencies of the
/// provided exported ops. An `IRMapping` can be provided to be able to map
/// into the sliced module.
OwningOpRef<ModuleOp>
produceStandaloneModule(const SymbolTable &symtab,
                        const KGEN::ExportMap &exportedSymbols,
                        IRMapping &mapping);
} // namespace M

#endif // KGEN_TRANSFORMUTILS_SLICINGUTILS_H
