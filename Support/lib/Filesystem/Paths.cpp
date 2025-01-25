//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Filesystem/Paths.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace Filesystem;

bool M::Filesystem::isMojoSourcePackagePath(const std::filesystem::path &path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec) && !ec) {
    bool exist = std::filesystem::exists(path / "__init__.mojo", ec) ||
                 std::filesystem::exists(path / "__init__.🔥", ec);
    bool isDir = std::filesystem::is_directory(path / "__init__.mojo", ec) ||
                 std::filesystem::is_directory(path / "__init__.🔥", ec);
    return exist && !isDir;
  }
  return false;
}

bool M::Filesystem::isMojoBinaryPackagePath(const std::filesystem::path &path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec) &&
         llvm::is_contained({".mojopkg", ".📦"}, path.extension());
}

bool M::Filesystem::isMojoSourceFile(const std::filesystem::path &path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec) &&
         llvm::is_contained({".mojo", ".🔥"}, path.extension());
}

bool M::Filesystem::isMLIRByteCodeFile(const std::filesystem::path &path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec) &&
         path.extension() == ".mlirbc";
}
