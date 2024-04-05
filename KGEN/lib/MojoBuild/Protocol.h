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
// build/initialize
//===----------------------------------------------------------------------===//

/// Parameters for the `build/initialize` method.
struct InitializeBuildParams {
  /// Name of the client.
  std::string displayName;
};

/// Deserialize a parameters object from JSON.
bool fromJSON(const llvm::json::Value &value, InitializeBuildParams &result,
              llvm::json::Path path);
} // namespace Build
} // namespace M

#endif // KGEN_MOJOBUILD_PROTOCOL_H
