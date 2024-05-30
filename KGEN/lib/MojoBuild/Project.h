//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOBUILD_PROJECT_H
#define KGEN_MOJOBUILD_PROJECT_H

#include "Support/ErrorOr.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

#include <filesystem>
#include <string>

namespace M::Build {

/// An in-memory representation of a single build target within a workspace.
/// For now, the only targets we recognize are Mojo packages that, when
/// compiled, produce a `.mojopkg` file.
struct Target {
  /// The target identifier, which must be unique within the scope of the
  /// workspace. For now, this is just the absolute path to the Mojo source
  /// package directory.
  std::string uri;
};

/// An in-memory representation of a Mojo project workspace.
class Project {
public:
  /// Returns a project for the given workspace path, or an error if one could
  /// not be created. The targets in the project are discovered based on the
  /// filesystem layout of a conventional Mojo project.
  static ErrorOr<Project> create(const std::filesystem::path &workspacePath);

  /// Returns the project's build targets.
  const llvm::StringMap<Target> &getTargets() const;

private:
  Project(const std::filesystem::path &rootDir) : rootDir(rootDir) {}

  /// Given the path to a workspace, creates a project based on the layout of
  /// the project on the filesystem, or an error if one cannot be created.
  static ErrorOr<Project>
  createDefault(const std::filesystem::path &workspacePath);

  /// The project root directory path.
  std::filesystem::path rootDir;

  /// A mapping between a URI and the build target within the project with that
  /// URI.
  llvm::StringMap<Target> targets;
};
} // namespace M::Build

#endif // KGEN_MOJOBUILD_PROJECT_H
