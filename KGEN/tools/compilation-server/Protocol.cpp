//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"

namespace CSP = M::KGEN::CSP;

//===----------------------------------------------------------------------===//
// EmitArchiveParams
//===----------------------------------------------------------------------===//

bool CSP::fromJSON(const llvm::json::Value &value, EmitArchiveParams &result,
                   llvm::json::Path path) {
  llvm::json::ObjectMapper o(value, path);
  return o && o.map("module", result.module);
}

llvm::json::Value CSP::toJSON(const EmitArchiveParams &value) {
  return llvm::json::Object{{"module", value.module}};
}

//===----------------------------------------------------------------------===//
// ObjectArchive
//===----------------------------------------------------------------------===//

llvm::json::Value CSP::toJSON(const ObjectArchive &value) {
  return llvm::json::Object{{"archive", value.archive}};
}
