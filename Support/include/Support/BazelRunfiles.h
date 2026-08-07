//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
