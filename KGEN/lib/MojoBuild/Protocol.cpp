//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

using namespace M;
using namespace M::Build;

//===----------------------------------------------------------------------===//
// build/initialize
//===----------------------------------------------------------------------===//

bool M::Build::fromJSON(const llvm::json::Value &value,
                        InitializeBuildParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("displayName", result.displayName);
}
