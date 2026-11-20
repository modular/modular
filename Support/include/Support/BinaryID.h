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

#ifndef SUPPORT_BINARYID_H
#define SUPPORT_BINARYID_H

#include <string>

namespace M {
/// Returns the build ID of the binary that contains this code, followed by a
/// hash of MODULAR_NVPTX_COMPILER_PATH when that is set.
std::string getBinaryID();

/// Returns the build ID of the binary that contains this code, without the
/// MODULAR_NVPTX_COMPILER_PATH hash.
std::string getBinaryBuildID();

/// Returns the canonical path of the binary that contains this code, or an
/// empty string if it cannot be determined.
std::string getBinaryPath();
} // namespace M

#endif // SUPPORT_BINARYID_H
