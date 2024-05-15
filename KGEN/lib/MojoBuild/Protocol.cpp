//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoBuild/Protocol.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

using namespace M;
using namespace M::Build;

//===----------------------------------------------------------------------===//
// build/initialize
//===----------------------------------------------------------------------===//

llvm::json::Value M::Build::toJSON(const InitializeBuildParams &value) {
  return llvm::json::Object{{"displayName", value.displayName},
                            {"version", value.version},
                            {"bspVersion", value.bspVersion},
                            {"rootUri", value.rootUri}};
}

bool M::Build::fromJSON(const llvm::json::Value &value,
                        InitializeBuildParams &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("displayName", result.displayName) &&
         o.map("version", result.version) &&
         o.map("bspVersion", result.bspVersion) &&
         o.map("rootUri", result.rootUri);
}

llvm::json::Value M::Build::toJSON(const InitializeBuildResult &value) {
  return llvm::json::Object{{"displayName", value.displayName}};
}

bool M::Build::fromJSON(const llvm::json::Value &value,
                        InitializeBuildResult &result, llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("displayName", result.displayName);
}

//===----------------------------------------------------------------------===//
// build/shutdown
//===----------------------------------------------------------------------===//

llvm::json::Value M::Build::toJSON(const NoParams &) { return nullptr; }

bool M::Build::fromJSON(const llvm::json::Value &, NoParams &,
                        llvm::json::Path) {
  return true;
}
