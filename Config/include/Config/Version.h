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
//
// Information about the Modular build version.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_VERSION_H
#define CONFIG_VERSION_H

#include "Support/SymbolExport.h"

namespace M {

struct ProjectVersion final {
  int major;
  int minor;
  int patch;
  const char *label;    // version label like "-rc.1"
  const char *revision; // Truncated Git SHA
  const char *buildType;
};

MODULAR_CXX_EXPORT ProjectVersion getMAXVersion();
MODULAR_CXX_EXPORT ProjectVersion getMojoVersion();
MODULAR_CXX_EXPORT const char *getMAXVersionString();
MODULAR_CXX_EXPORT const char *getMojoVersionString();

// TODO: Remove
using ModularVersion = ProjectVersion;
MODULAR_CXX_EXPORT ModularVersion getModularVersion();
MODULAR_CXX_EXPORT const char *getModularVersionString();

} // namespace M

#endif // CONFIG_VERSION_H
