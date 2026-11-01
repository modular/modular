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

#ifndef KGEN_SUPPORT_INTERNAL_PLUGINUTILS_H
#define KGEN_SUPPORT_INTERNAL_PLUGINUTILS_H

#include "Support/ErrorOr.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

namespace M::KGEN {

/// Resolves `symbolName` in the loaded plugin `hdl` as a `FnType`.
template <typename FnType>
ErrorOr<FnType> getFunction(void *hdl, llvm::StringRef symbolName) {
  auto fnPtr = reinterpret_cast<FnType>(dlsym(hdl, symbolName.str().c_str()));
  if (!fnPtr) {
    return Error(llvm::StringRef("failed to resolve ") + symbolName +
                 dlerror());
  }
  return fnPtr;
}

/// Compiler plugin which can be loaded at runtime to extend the compiler's
/// functionality.
class Plugin {
public:
  Plugin(const std::string &path);
  ~Plugin();

  void *getHandle() const { return handle; }

  /// Check if the plugin was successfully loaded.
  bool isLoaded() const;

  const std::string &getSoPath() const { return soPath; }

private:
  void *handle = nullptr;

  /// Plugin path.
  std::string soPath;
};

/// Loads the compiler plugins named by `modular.cfg` /
/// `MODULAR_COMPILER_PLUGINS` and owns them for callers to enumerate.
class PluginManager {
public:
  PluginManager();
  ~PluginManager() = default;

  /// The loaded plugins.
  const std::vector<std::unique_ptr<Plugin>> &getPlugins() const {
    return plugins;
  }

private:
  std::vector<std::unique_ptr<Plugin>> plugins;
};

} // namespace M::KGEN

#endif // KGEN_SUPPORT_INTERNAL_PLUGINUTILS_H
