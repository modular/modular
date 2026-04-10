//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_ELABORATORHELPER_H
#define KGEN_ELABORATOR_ELABORATORHELPER_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace M::KGEN {

/// Compute the final GPU symbol name for a @__name-decorated function.
/// This is the single source of truth shared by both renameFunctions (GPU
/// rename loop) and applyLinkageName (host-side get_linkage_name evaluation).
///
/// \param resolved   The @__name string value (verbatim, unsanitized).
/// \param lna        The linkageName attribute carrying the mangle flag.
/// \param symName    The auto-mangled symbol name for hashing (mangle=true
///                   only): for non-parametric kernels this is the @__name
///                   literal; for parametric kernels it is
///                   mangleParameterValues.
/// \param funcType   The MLIR function type, used as a hash input
/// (mangle=true).
mlir::StringAttr applyGPULinkageName(mlir::StringAttr resolved,
                                     LinkageNameAttr lna,
                                     llvm::StringRef symName,
                                     mlir::FunctionType funcType);

/// Resolve linkage names and sanitize symbol names for all FuncOps in
/// `theModule` in a single pass:
///
///  1. If a FuncOp carries a `linkageName` attribute, compute the final symbol
///     name (sanitized on GPU, verbatim on host), then remove the attribute.
///  2. On GPU targets, additionally sanitize every other name to
///     alphanumeric-only characters for PTX compatibility.
///  3. Fix up all `SymbolConstantAttr` references in the module to reflect the
///     new names.
///
/// Sets `failed = true` on errors (unresolved linkage name, duplicate symbols).
void renameFunctions(mlir::ModuleOp theModule, bool isGPU, bool &failed);

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATORHELPER_H
