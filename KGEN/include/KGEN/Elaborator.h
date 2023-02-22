//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_H
#define KGEN_ELABORATOR_H

#include "KGEN/KGENDialect/KGENParameters.h"
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

/// Elaborate generators in the specified module, incorporating implementation
/// logic from the specified library.  On error, diagnostics are emitted and the
/// primary file isn't completely lowered.
LogicalResult
elaborateGenerators(mlir::SymbolTableAnalysis &symtab,
                    KGEN::ParameterCollector::Analysis &paramCache,
                    LLCL::Runtime &runtime, TargetInfoAttr target,
                    ArrayRef<KGEN::GeneratorOp> generators,
                    bool enableSearch = false);

} // namespace M

#endif // KGEN_ELABORATOR_H
