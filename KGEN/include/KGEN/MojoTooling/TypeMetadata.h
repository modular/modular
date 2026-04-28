//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_TYPEMETADATA_H
#define KGEN_MOJOTOOLING_TYPEMETADATA_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace M {
namespace KGEN {

//===----------------------------------------------------------------------===//
// TypeMetadata
//===----------------------------------------------------------------------===//

/// Information about a type used to generate doc details in JSON
class TypeMetadata {
public:
  TypeMetadata() = default;
  TypeMetadata(llvm::StringRef typeStr, llvm::StringRef module = "",
               llvm::StringRef relativePath = "")
      : typeString(typeStr.str()), moduleNamespace(module.str()),
        relativeDocPath(relativePath.str()) {}

  /// Get the module namespace (e.g., "builtin.int", "collections.list")
  llvm::StringRef getModuleNamespace() const { return moduleNamespace; }

  /// Get the relative documentation path for cross-references
  llvm::StringRef getRelativeDocPath() const { return relativeDocPath; }

  /// Serialize the metadata to JSON with the following schema:
  /// {
  ///   "type": string,           // Full type as written in source, including
  ///                             // any parameterization (e.g. "List[Int]").
  ///   "path": string,           // Relative documentation path for the base
  ///                             // type's doc page (optional).
  /// }
  llvm::json::Object toJSON() const;

private:
  std::string typeString;
  std::string moduleNamespace; // Module namespace: "builtin.int", etc.
  std::string relativeDocPath; // Relative path for cross-reference links
};

} // namespace KGEN
} // namespace M

#endif // KGEN_MOJOTOOLING_TYPEMETADATA_H
