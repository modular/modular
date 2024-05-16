//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOBUILD_MOJOBUILDSERVER_H
#define KGEN_MOJOBUILD_MOJOBUILDSERVER_H

#include "Support/ErrorOr.h"

#include <filesystem>

namespace M::Build {

/// The result of a build.
enum class BuildResult {
  /// One or more artifacts were successfully built.
  Success,
  /// One or more artifacts failed to be built.
  Failure,
  /// There was nothing to build.
  NothingToDo,
};

/// Implements Mojo-specific portions of a Mojo build server. This class does
/// not concern itself with the build server protocol, JSON parsing, and other
/// things unrelated to Mojo specifically.
struct MojoBuildServer {
  /// Build the default set of targets in the workspace at the given path. If an
  /// error occurs that prevents a build, return an error. Otherwise, return the
  /// result of the build.
  static ErrorOr<BuildResult> buildWorkspace(const std::filesystem::path &path);
};
} // namespace M::Build

#endif // KGEN_MOJOBUILD_MOJOBUILDSERVER_H
