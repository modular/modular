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

#include <optional>
#include <string>
#include <vector>

namespace llvm::json {
class Path;
class Value;
} // namespace llvm::json

namespace M::Build {

//===----------------------------------------------------------------------===//
// build/initialize request params
//===----------------------------------------------------------------------===//

/// Parameters for the `build/initialize` method.
struct InitializeBuildParams {
  /// Name of the client.
  std::string displayName;
  /// The version of the client.
  std::string version;
  /// The build server protocol that the client speaks.
  std::string bspVersion;
  /// The root URI of the workspace.
  std::string rootUri;
};

/// Serialize a parameters object to JSON.
llvm::json::Value toJSON(const InitializeBuildParams &value);
/// Deserialize a parameters object from JSON.
bool fromJSON(const llvm::json::Value &value, InitializeBuildParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// build/initialize request result
//===----------------------------------------------------------------------===//

/// The languages fow which the server supports compilation via the
/// `buildTarget/compile` method.
struct CompileProvider {
  std::vector<std::string> languageIds;
};

/// Serialize a compile provider object to JSON.
llvm::json::Value toJSON(const CompileProvider &value);
/// Deserialize a compile provider object from JSON.
bool fromJSON(const llvm::json::Value &value, CompileProvider &result,
              llvm::json::Path path);

/// The capabilities of the build server.
struct BuildServerCapabilities {
  /// The languages fow which the server supports compilation via the
  /// `buildTarget/compile` method.
  std::optional<CompileProvider> compileProvider;
};

/// Serialize a capabilities object to JSON.
llvm::json::Value toJSON(const BuildServerCapabilities &value);
/// Deserialize a capabilities object from JSON.
bool fromJSON(const llvm::json::Value &value, BuildServerCapabilities &result,
              llvm::json::Path path);

/// Result included in responses to the `build/initialize` method.
struct InitializeBuildResult {
  /// Name of the server.
  std::string displayName;
  /// The version of the server.
  std::string version;
  /// The build server protocol that the client speaks.
  std::string bspVersion;
  /// The capabilities of the build server.
  BuildServerCapabilities capabilities;
};

/// Serialize a result object to JSON.
llvm::json::Value toJSON(const InitializeBuildResult &value);
/// Deserialize a result object from JSON.
bool fromJSON(const llvm::json::Value &value, InitializeBuildResult &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// build/shutdown request
//===----------------------------------------------------------------------===//

/// An empty set of parameters, such as for the `build/shutdown` method.
struct NoParams {};

/// Serialize an empty parameters object to JSON.
llvm::json::Value toJSON(const NoParams &value);
/// Deserialize an empty parameters object from JSON.
bool fromJSON(const llvm::json::Value &value, NoParams &result,
              llvm::json::Path path);
} // namespace M::Build

#endif // KGEN_MOJOBUILD_PROTOCOL_H
