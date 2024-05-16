//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoBuildServer.h"

#include "KGEN/Support/Configuration.h"
#include "Support/Filesystem/Paths.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

using namespace M;
using namespace M::Build;

/// Returns the path to the `mojo` executable, or an error if none could be
/// found.
static ErrorOr<std::string> getMojoDriverPath(KGEN::MojoConfig &config) {
  std::error_code ec;
  StringRef path = config.getDriverPath();
  if (!std::filesystem::exists(path.str(), ec) || ec)
    return Error(
        llvm::formatv("unable to resolve the mojo path at '{0}'", path));
  return path.str();
}

/// Returns all Mojo source package directories at the top level of the given
/// `root` directory.
static SmallVector<std::filesystem::path>
getMojoSourcePackageDirectories(const std::filesystem::path &root,
                                std::error_code &ec) {
  SmallVector<std::filesystem::path> result;
  std::filesystem::directory_iterator it(root, ec);
  if (ec)
    return result;

  for (const auto &path : it)
    if (Filesystem::isMojoSourcePackagePath(path))
      result.push_back(path);

  return result;
}

ErrorOr<BuildResult>
MojoBuildServer::buildWorkspace(const std::filesystem::path &path) {
  // Grab the path to the `mojo` driver.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (configOr.isError())
    return Error(llvm::formatv("could not read 'modular.cfg': {0}",
                               configOr.getError()));
  ErrorOr<std::string> driverPathOr = getMojoDriverPath(*configOr);
  if (driverPathOr.isError())
    return driverPathOr.takeError();

  // We do not yet define a Mojo project manifest, so we cannot be certain what
  // to build within our workspace. For the time being, look for a directory
  // named 'src' at the root of the workspace, and build a Mojo package from
  // each Mojo source package directory within 'src'. If there's nothing to
  // build, return a "cancelled" result.
  std::error_code ec;
  std::filesystem::path srcDir = path / "src";
  if (!std::filesystem::is_directory(srcDir, ec) || ec)
    return BuildResult::NothingToDo;

  SmallVector<std::filesystem::path> packageDirs =
      getMojoSourcePackageDirectories(srcDir, ec);
  if (packageDirs.empty() || ec)
    return BuildResult::NothingToDo;

  // Create a directory for build artifacts.
  std::filesystem::path buildDirectory = path / ".build";
  std::filesystem::create_directory(buildDirectory, ec);
  if (ec)
    return Error(llvm::formatv("could not create build directory '{0}': {1}",
                               buildDirectory, ec.message()));

  // For now, as a simple proof of concept, this sequentially builds the
  // packages in the workspace. In the future, this should be parallelized.
  for (const auto &packageDir : packageDirs) {
    SmallVector<StringRef> driverArgs{*driverPathOr, "package",
                                      packageDir.c_str(), "-o",
                                      buildDirectory.c_str()};
    std::string errorMessage;
    int exitCode = llvm::sys::ExecuteAndWait(
        *driverPathOr, driverArgs,
        /*Env=*/std::nullopt, /*Redirects=*/{}, /*SecondsToWait=*/0,
        /*MemoryLimit=*/0, /*ErrMsg=*/&errorMessage);

    // An error launching the driver is an internal server error, and we treat
    // it as distinct from a compilation error.
    if (!errorMessage.empty())
      return Error(errorMessage);

    // For now, we stop building as soon as any package fails to build.
    if (exitCode)
      return BuildResult::Failure;
  }

  return BuildResult::Success;
}
