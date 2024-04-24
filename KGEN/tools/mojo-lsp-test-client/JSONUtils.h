//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_JSONUTILS_H
#define KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_JSONUTILS_H

#include "llvm/Support/JSON.h"

namespace llvm::json {

/// Function similar to the typical `llvm::json::parse`, but that can operate
/// directly on a `Value` object.
template <typename T>
llvm::Expected<T> parse(const llvm::json::Value &json) {
  llvm::json::Path::Root root("");
  T result;
  if (fromJSON(json, result, root))
    return std::move(result);
  return root.getError();
}

} // namespace llvm::json

#endif // KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_JSONUTILS_H
