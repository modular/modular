//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_TYPEEXTRACTIONUTILS_H
#define KGEN_MOJOTOOLING_TYPEEXTRACTIONUTILS_H

#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/TypeMetadata.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

// Forward declarations to avoid circular dependency
namespace M {
class MojoASTTypeRef;
class MojoASTDeclRef;
} // namespace M

namespace M {
namespace KGEN {

namespace TypeExtractionUtils {

/// Extracts the leaf name from a symbol reference.
/// For example, given "stdlib.collections.List", returns "List".
std::string extractSymbolLeafName(mlir::SymbolRefAttr symbol);

/// Gets the base type name, removing generic parameters and qualifiers.
/// For example: "stdlib.collections.List[T]" -> "List", and
/// "ref [_] SomeType[T, U]" -> "SomeType". Uses AST information when available,
/// falls back to string parsing.
std::string extractBaseTypeName(const M::MojoASTTypeRef &astType,
                                llvm::StringRef fullTypeStr);

/// Convenience for extracting base type names when no AST info is available.
std::string extractBaseTypeName(llvm::StringRef fullTypeStr);

/// Extracts the fully qualified module path from an AST declaration reference.
/// For example, a declaration in stdlib.collections would return
/// "stdlib.collections".
std::string extractModulePathFromDecl(M::MojoASTDeclRef declRef);

/// Attempts to resolve a type name (e.g., "List", "Int") to its actual AST
/// declaration. Uses two strategies:
/// 1) Walks up the scope hierarchy from the current context looking for
///    matching struct/trait/alias declarations
/// 2) Falls back to builtin trait lookup via SharedState for compiler
/// intrinsics
std::optional<M::MojoASTDeclRef>
tryResolveTypeToDecl(llvm::StringRef typeName,
                     M::KGEN::LIT::SharedState &sharedState,
                     const M::MojoASTDeclRef *contextDecl);

/// Generates a documentation path from module info for cross-linking.
/// Uses the docsBasePath and moduleStr to construct a path.
/// For aliases, adds a fragment identifier with the lowercase alias name.
///
/// Example:
/// - generateDocPath("stdlib.collections", "List", "") ->
/// "stdlib/collections/List"
std::string generateDocPath(llvm::StringRef module, llvm::StringRef typeName,
                            llvm::StringRef docsBasePath, bool isAlias = false);

/// The main function that extracts comprehensive type metadata from type names.
/// Takes a type like "List[Int]" or "stdlib.collections.Dict" and produces
/// metadata including the clean type name, module path, and doc link path. Uses
/// AST resolution when possible to get accurate paths, caches results for
/// performance, and falls back to basic name for unresolvable types.
TypeMetadata
extractLibraryInfo(llvm::StringRef typeStr,
                   const M::MojoASTDeclRef *currentDeclContext = nullptr,
                   M::KGEN::LIT::SharedState *sharedState = nullptr);

} // namespace TypeExtractionUtils
} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOTOOLING_TYPEEXTRACTIONUTILS_H
