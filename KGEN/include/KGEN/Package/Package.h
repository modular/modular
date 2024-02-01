//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_PACKAGE_PACKAGE_H
#define KGEN_PACKAGE_PACKAGE_H

#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
class BytecodeReader;
} // namespace mlir

namespace M {
namespace LLCL {
class Runtime;
} // namespace LLCL
} // namespace M

namespace M::KGEN {

class CompilationOptions;
class PackageLinkOp;

/// Given the symbol table of an elaborated module for a Mojo package, as well
/// as that package's name, returns either an attribute to store the module
/// bytecode, or an error.
ErrorOr<DenseResourceElementsAttr>
createElaboratedBytecodeAttr(const SymbolTable &symtab,
                             FlatSymbolRefAttr packageName);

/// Given the symbol table of an elaborated module for a Mojo package, compiles
/// that module to a static archive, and if successful returns an archive
/// attribute. If unsuccessful, returns an error.
ErrorOr<PackageArchiveAttr> createPackageArchive(
    const SymbolTable &symtab, const ExportMap &exportedSymbols,
    TargetInfoAttr targetInfo, DenseResourceElementsAttr elaboratedBytecode,
    const CompilationOptions &compileOptions, LLCL::Runtime &runtime);

/// Loads the serialized MLIR bytecode representing a pre-elaborated module for
/// package off of the given `packageLink` op, elaborates it, and generates a
/// static archive. If successful, an archive will be set on the given op.
ErrorOr<PackageArchiveAttr>
loadAndElaborateBytecode(PackageLinkOp packageLink, TargetInfoAttr targetInfo,
                         const CompilationOptions &compileOptions,
                         LLCL::Runtime &runtime);

/// This populates the passes to produce a fully concrete KGEN module. It's the
/// equivalent of the `buildElaborateModulePipeline` function defined in
/// KGENCompiler, but with a default handler for package link ops.
void populateElaborateModulePasses(mlir::PassManager &pm,
                                   LLCL::Runtime &runtime,
                                   TargetInfoAttr target,
                                   const CompilationOptions &options);
} // namespace M::KGEN

#endif // KGEN_PACKAGE_PACKAGE_H
