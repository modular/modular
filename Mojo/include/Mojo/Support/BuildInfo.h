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

#ifndef KGEN_SUPPORT_BUILDINFO_H
#define KGEN_SUPPORT_BUILDINFO_H

#include <string>

namespace M::KGEN {
std::string getVersionString();

/// Returns the canonical path of the binary whose build ID getVersionString()
/// embeds, or an empty string if it cannot be determined.
std::string getVersionBinaryPath();

/// Returns the build ID getVersionString() embeds, without the
/// MODULAR_NVPTX_COMPILER_PATH hash.
std::string getVersionBuildID();
} // namespace M::KGEN

#endif // KGEN_SUPPORT_BUILDINFO_H
