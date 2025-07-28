//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/TypeMetadata.h"

namespace M {
namespace KGEN {

llvm::json::Object TypeMetadata::toJSON() const {
  llvm::json::Object result;

  result["type"] = typeString;

  if (!relativeDocPath.empty()) {
    result["path"] = relativeDocPath;
  }

  return result;
}

} // namespace KGEN
} // namespace M
