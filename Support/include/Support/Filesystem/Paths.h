//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FILESYSTEM_PATHS_H
#define SUPPORT_FILESYSTEM_PATHS_H

#include <filesystem>

namespace M::Filesystem {

/// Returns true if the given path is a Mojo package source directory (i.e. a
/// directory that contains an `__init__.mojo` file).
bool isMojoSourcePackagePath(const std::filesystem::path &path);

} // namespace M::Filesystem

#endif // SUPPORT_FILESYSTEM_PATHS_H
