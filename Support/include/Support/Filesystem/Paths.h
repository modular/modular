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

/// Returns true if the given path is a Mojo binary package (i.e. a `.📦` or
/// `.mojopkg` file).
bool isMojoBinaryPackagePath(const std::filesystem::path &path);

/// Return if the given file path defines a mojo source file.
bool isMojoSourceFile(const std::filesystem::path &path);

/// Return if the given file path defines a MLIR bytecode file (`.mlirbc`).
bool isMLIRByteCodeFile(const std::filesystem::path &path);

} // namespace M::Filesystem

#endif // SUPPORT_FILESYSTEM_PATHS_H
