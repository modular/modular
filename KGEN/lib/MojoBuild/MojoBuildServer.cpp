//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoBuildServer.h"
#include "Project.h"

#include "KGEN/Support/Configuration.h"

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

  // Create a representation of the project, collecting information about the
  // build targets defined therein.
  ErrorOr<Project> projectOr = Project::create(path);
  if (projectOr.isError())
    return projectOr.takeError();

  // Return early if there's nothing to build.
  if (projectOr->getTargets().empty())
    return BuildResult::NothingToDo;

  // Create a directory for build artifacts.
  std::filesystem::path buildDirectory = path / ".build";
  std::error_code ec;
  std::filesystem::create_directory(buildDirectory, ec);
  if (ec)
    return Error(llvm::formatv("could not create build directory '{0}': {1}",
                               buildDirectory, ec.message()));

  // For now, as a simple proof of concept, this sequentially builds all project
  // targets. In the future, this should be parallelized.
  for (const auto &target : projectOr->getTargets()) {
    // We can assume, for now, that every single project target is built the
    // same way: a `mojo package` invocation.
    SmallVector<StringRef> driverArgs{*driverPathOr, "package",
                                      target.getValue().uri, "-o",
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
