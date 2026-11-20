//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// The `MODULAR_DRIVER_PLUGINS` format, parsed in one place for its C++
// readers. The HAL `DeviceContext` reads the same format in Mojo.
//
// Header-only so the compiler family's arch query can use it without
// linking a DeviceContext runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_DRIVERPLUGINSPEC_H
#define MLRT_DRIVER_DEVICECONTEXT_DRIVERPLUGINSPEC_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

namespace M::Driver {

/// One plugin named in `MODULAR_DRIVER_PLUGINS`. `name` is empty for an entry
/// given as a bare path.
struct DriverPluginEntry {
  llvm::StringRef name;
  llvm::StringRef path;
};

/// The plugins named in `spec`, a `MODULAR_DRIVER_PLUGINS` value such as
/// "CUDA@/a/CUDAPlugin.so;/b/CPUPlugin.so". Entries with no path are dropped.
inline llvm::SmallVector<DriverPluginEntry>
parseDriverPluginSpec(llvm::StringRef spec) {
  llvm::SmallVector<llvm::StringRef> entries;
  spec.split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  llvm::SmallVector<DriverPluginEntry> plugins;
  for (llvm::StringRef entry : entries) {
    auto [name, path] = entry.contains('@')
                            ? entry.split('@')
                            : std::make_pair(llvm::StringRef(), entry);
    if (!path.empty())
      plugins.push_back({name, path});
  }
  return plugins;
}

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_DRIVERPLUGINSPEC_H
