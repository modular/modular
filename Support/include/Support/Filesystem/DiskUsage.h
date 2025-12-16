//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FILESYSTEM_DISKUSAGE_H
#define SUPPORT_FILESYSTEM_DISKUSAGE_H

#include "Support/ForwardDecls.h"

#include <cstddef>
#include <filesystem>

namespace M {
/// Returns the available disk space in the filesystem containing given path.
ErrorOr<size_t> getAvailableDiskSpace(const std::filesystem::path &path);
} // namespace M

#endif // SUPPORT_FILESYSTEM_DISKUSAGE_H
