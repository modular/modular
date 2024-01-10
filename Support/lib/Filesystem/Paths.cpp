//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Filesystem/Paths.h"

using namespace M;
using namespace Filesystem;

bool M::Filesystem::isMojoSourcePackagePath(const std::filesystem::path &path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec) && !ec) {
    return std::filesystem::exists(path / "__init__.mojo", ec) ||
           std::filesystem::exists(path / "__init__.🔥", ec);
  }
  return false;
}
