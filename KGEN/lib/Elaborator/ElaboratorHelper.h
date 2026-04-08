//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_ELABORATORHELPER_H
#define KGEN_ELABORATOR_ELABORATORHELPER_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/Compiler/ErrorTree.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace M::KGEN {

/// Compute the expected final symbol name of a generator from a symbol
/// constant attribute.  Returns both the mangled name and the generator.
/// If `allowParametric`, an unresolved parametric name does not produce an
/// error.  If `sanitize`, additionally sanitizes the name to alnum-only
/// characters.
/// Used by both Elaborator and ParametricElaborator to predict the name that
/// the rename pass will assign, so that host stubs can reference GPU kernels
/// by their final PTX name.
ErrorTreeOr<std::pair<mlir::StringAttr, GeneratorOp>>
getExpectedMangledName(mlir::Location errorLoc, llvm::StringRef errorContext,
                       TypedAttr symCst, mlir::SymbolTable &symTab,
                       bool allowParametric = false, bool sanitize = false);

/// Resolve linkage names and sanitize symbol names for all FuncOps in
/// `theModule` in a single pass:
///
///  1. If a FuncOp carries a `linkageName` attribute, rename the function to
///     that name, then remove the attribute.
///  2. On GPU targets, additionally sanitize every name to alphanumeric-only
///     characters for PTX compatibility.
///  3. Fix up all `SymbolConstantAttr` references in the module to reflect the
///     new names.
///
/// Sets `failed = true` on errors (unresolved linkage name, duplicate symbols).
void renameFunctions(mlir::ModuleOp theModule, bool isGPU, bool &failed);

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATORHELPER_H
