//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Filesystem/DiskUsage.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"

using namespace M;

M::ErrorOr<size_t> M::getAvailableDiskSpace(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::space_info info = std::filesystem::space(path, ec);
  if (ec)
    return Error(llvm::Twine(ec.message()));
  return info.available;
}
