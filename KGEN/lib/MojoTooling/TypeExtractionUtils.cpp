//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/TypeExtractionUtils.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/PublicASTDecl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <map>
#include <mutex>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

namespace M {
namespace KGEN {

namespace TypeExtractionUtils {

/// Extracts the leaf name from a symbol reference.
std::string extractSymbolLeafName(mlir::SymbolRefAttr symbol) {
  if (symbol.getNestedReferences().empty()) {
    return symbol.getRootReference().getValue().str();
  }
  return symbol.getNestedReferences().back().getAttr().getValue().str();
}

/// Gets the base type name, removing generic parameters and qualifiers.
std::string extractBaseTypeName(const M::MojoASTTypeRef &astType,
                                llvm::StringRef fullTypeStr) {
  // Try AST-based extraction first
  if (astType) {
    Type mlirType = astType.getMLIRType();

    // Handle struct types
    if (auto structType = sugarDynCast<LIT::StructType>(mlirType)) {
      if (auto symbol = structType.getSymbol()) {
        return extractSymbolLeafName(symbol);
      }
    }
    // Handle reference types
    else if (auto refType = sugarDynCast<LIT::RefType>(mlirType)) {
      if (auto elementType = astType.getReferenceElementType()) {
        return extractBaseTypeName(elementType, fullTypeStr);
      }
    }
    // Handle trait types
    else if (auto traitType = sugarDynCast<LIT::TraitType>(mlirType)) {
      auto symbols = traitType.getSymbols();
      if (!symbols.empty()) {
        return extractSymbolLeafName(symbols.front());
      }
    }
    // Handle OriginSet type name
    else if (sugarIsa<LIT::OriginSetType>(mlirType)) {
      return "OriginSet";
    }
  }

  // String-based fallback: remove generics
  llvm::StringRef baseType = fullTypeStr;

  // Special case if not found in AST:
  // If the string looks like a closure function signature, keep it as-is
  if (baseType.starts_with("fn(") || baseType.starts_with("fn[") ||
      baseType.starts_with("def(") || baseType.starts_with("def[")) {
    return baseType.trim().str();
  }

  if (size_t bracketPos = baseType.find('[');
      bracketPos != llvm::StringRef::npos) {
    baseType = baseType.substr(0, bracketPos);
  }
  baseType = baseType.trim();
  return baseType.str();
}

/// Convenience for extracting base type names when no AST info is available.
std::string extractBaseTypeName(llvm::StringRef fullTypeStr) {
  return extractBaseTypeName(M::MojoASTTypeRef{}, fullTypeStr);
}

/// Extracts the fully qualified module path from an AST declaration reference.
std::string extractModulePathFromDecl(M::MojoASTDeclRef declRef) {
  llvm::SmallVector<llvm::StringRef> pathComponents;

  M::MojoASTDeclRef current = declRef;
  while (current) {
    Operation *op = current.getIfOperation();
    if (auto fileModule = dyn_cast_or_null<FileModuleOp>(op)) {
      // File module - add its name and continue up to get full hierarchy
      if (auto name = current.getName()) {
        pathComponents.push_back(*name);
      }
    } else if (auto packageOp = dyn_cast_or_null<PackageOp>(op)) {
      // Package - add its name and continue up
      if (auto name = current.getName()) {
        pathComponents.push_back(*name);
      }
    }
    current = current.getParent();
  }

  std::reverse(pathComponents.begin(), pathComponents.end());

  std::string result;
  for (size_t i = 0; i < pathComponents.size(); ++i) {
    if (i > 0)
      result += ".";
    result += pathComponents[i].str();
  }

  return result;
}

/// Attempts to resolve a type name (eg, "List") to its actual AST declaration.
std::optional<M::MojoASTDeclRef>
tryResolveTypeToDecl(llvm::StringRef typeName, SharedState &sharedState,
                     const M::MojoASTDeclRef *contextDecl) {
  // Strategy 1: If we have a context declaration, try looking up the type in
  // its scope and parent scopes
  if (contextDecl && *contextDecl) {
    M::MojoASTDeclRef current = *contextDecl;
    while (current) {
      // Look for the type in the current scope
      for (auto childEntry : current.getChildren()) {
        if (childEntry.getName() == typeName) {
          for (auto child : childEntry.getDecls()) {
            Operation *childOp = child.getIfOperation();
            if (isa_and_nonnull<StructDeclOp, TraitDeclOp, AliasDeclOp>(
                    childOp)) {
              return child;
            }
          }
        }
      }
      current = current.getParent();
    }
  }

  // Strategy 2: Try looking up builtin traits using SharedState
  if (contextDecl && *contextDecl) {
    if (ASTDecl *builtinTraitDecl =
            sharedState.lookupBuiltinTrait(typeName, SMLoc())) {
      return M::MojoASTDeclRef(builtinTraitDecl);
    }
  }

  return std::nullopt;
}

/// Generates a documentation path from module info for cross-linking.
std::string generateDocPath(llvm::StringRef module, llvm::StringRef typeName,
                            llvm::StringRef docsBasePath, bool isAlias) {
  if (typeName.empty()) {
    return "";
  }

  // Remove __init__ components from the module path since they shouldn't appear
  // in documentation URLs (APIs defined in __init__.mojo files should link to
  // the parent package/module)
  llvm::SmallVector<llvm::StringRef> components;
  module.split(components, '.');

  std::string moduleStr;
  for (size_t i = 0; i < components.size(); ++i) {
    if (components[i] != "__init__") {
      if (!moduleStr.empty()) {
        moduleStr += ".";
      }
      moduleStr += components[i].str();
    }
  }

  std::string path;
  bool isStdType = moduleStr.starts_with("std.");

  // If this is a std type, use the module path as-is
  if (isStdType) {
    path = moduleStr;
    // Add underscore at the end of ".index" name for docsite URL compatibility
    if (path.length() >= 6 && path.substr(path.length() - 6) == ".index") {
      path = path.substr(0, path.length() - 6) + ".index_";
    }
  } else {
    if (!docsBasePath.empty()) {
      path = docsBasePath.str();
      if (!moduleStr.empty()) {
        path += "/" + moduleStr;
      }
    } else {
      path = moduleStr;
    }
  }

  std::replace(path.begin(), path.end(), '.', '/');
  path = "/" + path;

  if (isAlias) {
    // For aliases, add the fragment identifier with lowercase alias name
    std::string aliasStr = typeName.str();
    std::transform(aliasStr.begin(), aliasStr.end(), aliasStr.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (!path.empty()) {
      path += "/#" + aliasStr;
    } else {
      path = "#" + aliasStr;
    }
  } else {
    // For concrete types, append the type name normally
    if (!path.empty()) {
      path += "/" + typeName.str();
    } else {
      path = typeName.str();
    }
  }

  return path;
}

/// Resolved cross-reference info for a base type name. The display string
/// (`"type"`) is the caller's full parameterized type and is not cached here,
/// so different parameterizations of the same base type (e.g. `List[Int]` vs
/// `List[String]`) share the same path resolution without overwriting each
/// other's display string.
///
/// An entry with both fields empty is a negative-cache marker — used only when
/// `extractLibraryInfo` is called without `SharedState`, since AST resolution
/// may succeed later given better context.
namespace {
struct ResolvedPath {
  std::string moduleNamespace;
  std::string docPath;
};
} // namespace

/// The main function that extracts comprehensive type metadata from type names.
TypeMetadata extractLibraryInfo(llvm::StringRef typeStr,
                                const M::MojoASTDeclRef *currentDeclContext,
                                SharedState *sharedState) {
  // Cache for resolved cross-reference paths, keyed by stripped base name.
  static std::map<std::string, ResolvedPath> pathCache;
  static std::mutex cacheMutex;

  auto makeMetadata = [&](const ResolvedPath &resolved) {
    return TypeMetadata(typeStr, resolved.moduleNamespace, resolved.docPath);
  };

  // Extract base type name: remove templates and qualified prefixes
  llvm::StringRef baseType = typeStr;

  // Remove template parameters (everything after '[')
  if (size_t bracketPos = baseType.find('[');
      bracketPos != llvm::StringRef::npos) {
    baseType = baseType.substr(0, bracketPos);
  }

  baseType = baseType.trim();

  // Check cache first (with thread safety).
  std::string cacheKey = baseType.str();
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    if (auto cachedIt = pathCache.find(cacheKey); cachedIt != pathCache.end()) {
      return makeMetadata(cachedIt->second);
    }
  }

  // Try AST-based resolution if we have shared state
  if (sharedState) {
    if (auto declRef =
            tryResolveTypeToDecl(baseType, *sharedState, currentDeclContext)) {
      std::string modulePath = extractModulePathFromDecl(*declRef);
      if (!modulePath.empty()) {
        std::string typeName;
        std::string docPath;

        // Check if this is an alias declaration
        bool isAlias = isa_and_nonnull<AliasDeclOp>(declRef->getIfOperation());
        if (isAlias) {
          // Preserve the original alias name (used to build the URL anchor)
          typeName = baseType.str();
        } else {
          // For concrete types (structs/traits), use the regular path
          typeName = extractBaseTypeName(typeStr);
        }
        docPath = generateDocPath(modulePath, typeName,
                                  sharedState->getDocsBasePath(), isAlias);

        ResolvedPath resolved{modulePath, docPath};
        {
          std::lock_guard<std::mutex> lock(cacheMutex);
          pathCache[cacheKey] = resolved;
        }
        return makeMetadata(resolved);
      }
    }
  }

  // For unknown types (not found in AST), return minimal info with no path.
  // Don't cache failures when we have SharedState - they might succeed later
  // with better context.
  //
  // Asymmetry: the lookup at the top of the function does *not* re-check
  // `sharedState`, so a negative entry inserted here will be returned on
  // subsequent calls even if a later caller has SharedState. In practice the
  // same baseType is queried with consistent SharedState availability, so
  // this hasn't bitten us; revisit if it ever does.
  if (!sharedState) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    pathCache[cacheKey] = ResolvedPath{};
  }
  return makeMetadata(ResolvedPath{});
}

namespace {

/// True if `s` is a Mojo identifier (`[A-Za-z_][A-Za-z0-9_]*`).
bool isIdentifier(llvm::StringRef s) {
  if (s.empty() || (!llvm::isAlpha(s[0]) && s[0] != '_'))
    return false;
  for (char c : s.drop_front())
    if (!llvm::isAlnum(c) && c != '_')
      return false;
  return true;
}

/// True if `param` is a member access on `argName`, i.e. `argName.X(.Y)*`.
/// Matches `output.origin` or `arg.x.y` but not `output`, `outputFoo`, or
/// `output + 1`.
bool isMemberAccessOn(llvm::StringRef param, llvm::StringRef argName) {
  param = param.trim();
  if (!param.consume_front(argName) || !param.consume_front("."))
    return false;
  while (true) {
    auto [seg, rest] = param.split('.');
    if (!isIdentifier(seg))
      return false;
    if (rest.empty())
      return true;
    param = rest;
  }
}

} // namespace

std::string stripImplicitArgParams(llvm::StringRef typeStr,
                                   llvm::StringRef argName) {
  if (argName.empty())
    return typeStr.str();

  std::string result;
  result.reserve(typeStr.size());

  for (size_t i = 0; i < typeStr.size();) {
    if (typeStr[i] != '[') {
      result += typeStr[i++];
      continue;
    }

    // Find the matching ']' and the comma positions at this bracket depth.
    size_t depth = 1;
    size_t j = i + 1;
    llvm::SmallVector<size_t> commas;
    for (; j < typeStr.size() && depth > 0; ++j) {
      char c = typeStr[j];
      if (c == '[')
        ++depth;
      else if (c == ']') {
        if (--depth == 0)
          break;
      } else if (c == ',' && depth == 1) {
        commas.push_back(j);
      }
    }
    assert(depth == 0 && "type printer emitted unbalanced brackets");
    if (depth != 0) {
      // Release-build fallback: emit the remainder verbatim.
      result += typeStr.substr(i);
      return result;
    }

    llvm::SmallVector<llvm::StringRef> params;
    size_t prev = i + 1;
    for (size_t comma : commas) {
      params.push_back(typeStr.slice(prev, comma));
      prev = comma + 1;
    }
    params.push_back(typeStr.slice(prev, j));

    llvm::SmallVector<std::string> kept;
    for (llvm::StringRef p : params) {
      if (isMemberAccessOn(p, argName))
        continue;
      kept.push_back(stripImplicitArgParams(p.trim(), argName));
    }

    if (!kept.empty()) {
      result += '[';
      llvm::interleave(
          kept, [&](const std::string &k) { result += k; },
          [&] { result += ", "; });
      result += ']';
    }

    i = j + 1;
  }
  return result;
}

} // namespace TypeExtractionUtils
} // namespace KGEN
} // namespace M
