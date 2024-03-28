//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines in-memory representations of the JSON parameters and results of the
// build server, loosely following the build server protocol defined here:
// https://build-server-protocol.github.io/docs/specification
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOBUILD_PROTOCOL_H
#define KGEN_MOJOBUILD_PROTOCOL_H

#include <string>

namespace llvm {
namespace json {
class Path;
class Value;
} // namespace json
} // namespace llvm

namespace M {
namespace Build {

//===----------------------------------------------------------------------===//
// Protocol objects
//===----------------------------------------------------------------------===//

/// Parameters for the `build/initialize` method.
struct InitializeBuildParams {
  /// Name of the client.
  std::string displayName;
};
} // namespace Build
} // namespace M

//===----------------------------------------------------------------------===//
// JSON serialization
//===----------------------------------------------------------------------===//

namespace mlir {
namespace lsp {

bool fromJSON(const llvm::json::Value &value,
              M::Build::InitializeBuildParams &result, llvm::json::Path path);
} // namespace lsp
} // namespace mlir

#endif // KGEN_MOJOBUILD_PROTOCOL_H
