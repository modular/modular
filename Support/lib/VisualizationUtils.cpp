//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/VisualizationUtils.h"
#include "Support/ErrorOr.h"
#include <filesystem>
#include <string>

namespace M {

constexpr std::string_view vizFileExtension = ".maxviz";

std::string getFileNameWithExtension(const std::filesystem::path &inputPath,
                                     const std::string &defaultFileName) {

  if (inputPath.empty() || !inputPath.has_filename()) {
    return defaultFileName + std::string{vizFileExtension};
  }

  // We will append our file extension to it if it's not already one that is
  // compliant with ours.
  std::string currName = inputPath.filename();
  if (inputPath.filename().extension() != vizFileExtension)
    currName += vizFileExtension;

  return currName;
}

std::filesystem::path getBasePath(const std::filesystem::path &inputPath) {

  if (inputPath.empty())
    return std::filesystem::current_path();

  auto finalPath = inputPath;
  if (finalPath.has_filename()) {
    finalPath.remove_filename();
  }
  return finalPath;
}

// Returns std::string output filepath given the output directory.
// TODO(#34233): Most of this function should move to MAX CLI code once set
// up.
ErrorOr<std::string> createFilepath(const std::filesystem::path &vizOutputPath,
                                    std::string defaultFileName) {

  auto filename = getFileNameWithExtension(vizOutputPath, defaultFileName);
  auto basePath = getBasePath(vizOutputPath);

  std::filesystem::path fullPath = filename;
  if (!basePath.empty())
    fullPath = basePath / filename;

  // TODO(#29254): Also check and account for duplicate filenames.
  // Create the output directory if the directory doesn't exist.
  std::error_code error;
  if (!fullPath.empty() && !std::filesystem::exists(fullPath, error) &&
      !error) {
    if (fullPath.has_parent_path())
      std::filesystem::create_directories(fullPath.parent_path(), error);
  }
  if (error)
    return Error("Unable to create output directory");

  return fullPath.string(); // Return the string version of the path.
}

} // namespace M
