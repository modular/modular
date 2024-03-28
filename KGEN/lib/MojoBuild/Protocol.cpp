//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

bool mlir::lsp::fromJSON(const llvm::json::Value &value,
                         M::Build::InitializeBuildParams &result,
                         llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("displayName", result.displayName);
}
