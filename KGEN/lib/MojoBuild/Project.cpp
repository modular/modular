//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Project.h"

#include "Support/ErrorOr.h"
#include "Support/Filesystem/Paths.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#include <filesystem>

using namespace M;
using namespace M::Build;

ErrorOr<Project> Project::create(const std::filesystem::path &workspacePath) {
  // We do not yet support project manifests, so for now, always create a
  // project based on Mojo project filesystem layout conventions.
  return createDefault(workspacePath);
}

const llvm::StringMap<Target> &Project::getTargets() const { return targets; }

/// Returns all Mojo source package directories at the top level of the given
/// `root` directory.
static ErrorOr<SmallVector<std::filesystem::path>>
getMojoSourcePackageDirectories(const std::filesystem::path &root) {
  SmallVector<std::filesystem::path> result;
  std::error_code ec;
  std::filesystem::directory_iterator it(root, ec);
  if (ec)
    return Error(llvm::formatv("could not read directory '{0}': {1}", root,
                               ec.message()));

  for (const auto &dir : it) {
    std::filesystem::path path = std::filesystem::weakly_canonical(dir, ec);
    if (ec)
      return Error(llvm::formatv("path '{0}' could not be made absolute: {1}",
                                 dir.path(), ec.message()));
    if (Filesystem::isMojoSourcePackagePath(path))
      result.push_back(path);
  }

  return result;
}

ErrorOr<Project>
Project::createDefault(const std::filesystem::path &workspacePath) {
  Project project(workspacePath);

  // We expect targets to appear in a directory named 'src' at the root of the
  // workspace. If one doesn't exist, there are no build targets.
  std::filesystem::path srcDir = workspacePath / "src";
  std::error_code ec;
  if (!std::filesystem::is_directory(srcDir, ec) || ec)
    return project;

  // Every Mojo source package directory at the top level 'src' directory is a
  // Mojo package build target.
  auto packageDirsOr = getMojoSourcePackageDirectories(srcDir);
  if (packageDirsOr.isError())
    return Error(llvm::formatv(
        "could not read Mojo source package directories in '{0}': {1}", srcDir,
        packageDirsOr.takeError()));
  for (const std::filesystem::path &path : *packageDirsOr) {
    std::string uri = path.string();
    project.targets.try_emplace(uri, Target{uri});
  }

  return project;
}
