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

/// Serialize a parameters object to JSON.
llvm::json::Value toJSON(const InitializeBuildParams &value);
/// Deserialize a parameters object from JSON.
bool fromJSON(const llvm::json::Value &value, InitializeBuildParams &result,
              llvm::json::Path path);

struct InitializeBuildResult {
  /// Name of the server.
  std::string displayName;
};

/// Serialize a result object to JSON.
llvm::json::Value toJSON(const InitializeBuildResult &value);
/// Deserialize a result object from JSON.
bool fromJSON(const llvm::json::Value &value, InitializeBuildResult &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// build/shutdown
//===----------------------------------------------------------------------===//

/// An empty set of parameters, such as for the `build/shutdown` method.
struct NoParams {};

/// Serialize an empty parameters object to JSON.
llvm::json::Value toJSON(const NoParams &value);
/// Deserialize an empty parameters object from JSON.
bool fromJSON(const llvm::json::Value &value, NoParams &result,
              llvm::json::Path path);

} // namespace Build
} // namespace M

#endif // KGEN_MOJOBUILD_PROTOCOL_H
