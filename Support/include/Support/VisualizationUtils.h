//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines some util functions for generating max visualize CLI
// filepaths.
//
//===----------------------------------------------------------------------===/

#ifndef SUPPORT_VISUALIZATION_UTILS_H
#define SUPPORT_VISUALIZATION_UTILS_H

#include "Support/ErrorOr.h"
#include <filesystem>
#include <string>

namespace M {

std::string getFileNameWithExtension(const std::filesystem::path &inputPath,
                                     const std::string &defaultFileName);

std::filesystem::path getBasePath(const std::filesystem::path &inputPath);

// Returns std::string output filepath given the output directory.
// TODO(#34233): Most of this function should move to MAX CLI code once set
// up.
ErrorOr<std::string> createFilepath(const std::filesystem::path &vizOutputPath,
                                    const std::string &modelPath);

ErrorOr<std::string> getDefaultFilename(const std::string &modelPath);

} // namespace M

#endif // SUPPORT_VISUALIZATION_UTILS_H
