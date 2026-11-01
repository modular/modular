//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_TYPEEXTRACTIONUTILS_H
#define KGEN_MOJOTOOLING_TYPEEXTRACTIONUTILS_H

#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/MojoTooling/TypeMetadata.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace M {
namespace KGEN {
namespace LIT {
class ASTDecl;
class ASTType;

/// Extracts the leaf name from a symbol reference.
/// For example, given "std.collections.List", returns "List".
std::string extractSymbolLeafName(mlir::SymbolRefAttr symbol);

/// Gets the base type name, removing generic parameters and qualifiers.
/// For example: "std.collections.List[T]" -> "List", and
/// "ref [_] SomeType[T, U]" -> "SomeType". Uses AST information when available,
/// falls back to string parsing.
std::string extractBaseTypeName(LIT::ASTType astType,
                                llvm::StringRef fullTypeStr);

/// Convenience for extracting base type names when no AST info is available.
std::string extractBaseTypeName(StringRef fullTypeStr);

/// Extracts the fully qualified module path from an AST declaration reference.
/// For example, a declaration in std.collections would return
/// "std.collections".
std::string extractModulePathFromDecl(ASTDecl *decl);

/// Attempts to resolve a type name (e.g., "List", "Int") to its actual AST
/// declaration. Uses two strategies:
/// 1) Walks up the scope hierarchy from the current context looking for
///    matching struct/trait/alias declarations
/// 2) Falls back to builtin trait lookup via SharedState for compiler
/// intrinsics
std::optional<M::KGEN::LIT::ASTDecl *>
tryResolveTypeToDecl(StringRef typeName, SharedState &sharedState,
                     const ASTDecl *contextDecl);

/// Generates a documentation path from module info for cross-linking.
/// Uses the docsBasePath and moduleStr to construct a path.
/// For aliases, adds a fragment identifier with the lowercase alias name.
/// Automatically removes __init__ components from module paths since APIs
/// defined in __init__.mojo files should link to their parent package/module.
///
/// Examples:
/// - generateDocPath("std.collections", "List", "") ->
/// "/std/collections/List"
/// - generateDocPath("std.collections.__init__", "List", "") ->
/// "/std/collections/List"
std::string generateDocPath(StringRef module, StringRef typeName,
                            StringRef docsBasePath, bool isAlias = false);

/// The main function that extracts comprehensive type metadata from type names.
/// Takes a type like "List[Int]" or "std.collections.Dict" and produces
/// metadata including the clean type name, module path, and doc link path. Uses
/// AST resolution when possible to get accurate paths, caches results for
/// performance, and falls back to basic name for unresolvable types.
TypeMetadata extractLibraryInfo(StringRef typeStr,
                                const ASTDecl *currentDeclContext = nullptr,
                                SharedState *sharedState = nullptr);

} // namespace LIT
} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOTOOLING_TYPEEXTRACTIONUTILS_H
