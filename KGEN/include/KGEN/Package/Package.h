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

namespace M {
namespace LLCL {
class Runtime;
} // namespace LLCL
} // namespace M

namespace M::KGEN {

class CompilationOptions;

/// Given the symbol table of an elaborated module for a Mojo package, as well
/// as that package's name, returns either an attribute to store the module
/// bytecode, or an error.
ErrorOr<DenseResourceElementsAttr>
createElaboratedBytecodeAttr(const SymbolTable &symtab,
                             FlatSymbolRefAttr packageName);

/// Given the symbol table of an elaborated module for a Mojo package, compiles
/// that module to a static archive, and if successful returns an attribute
/// of the archive bytes. If unsuccessful, returns an error.
ErrorOr<DenseResourceElementsAttr> createPackageArchive(
    const SymbolTable &symtab, const ExportMap &exportedSymbols,
    const CompilationOptions &compileOptions, LLCL::Runtime &runtime);
} // namespace M::KGEN

#endif // KGEN_PACKAGE_PACKAGE_H
