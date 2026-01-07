//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BAZELRUNFILES_H
#define SUPPORT_BAZELRUNFILES_H

#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace M {

/// Try to find a path for a modular.cfg config key using Bazel runfiles.
///
/// This function maps configuration keys (e.g., "mojo-max.driver_path") to
/// their corresponding Bazel runfile paths and resolves them using the Bazel
/// runfiles library.
///
/// Returns std::nullopt if:
/// - The key is not expected to be loaded via runfiles (no mapping exists)
/// - Runfiles is not available (not running under Bazel)
std::optional<std::string> findConfigWithRunfiles(llvm::StringRef key);

} // namespace M

#endif // SUPPORT_BAZELRUNFILES_H
