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

#include "Mojo/Support/max_license/PluginUtils.h"
#include "Mojo/Support/Configuration.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"

#include <dlfcn.h>
#include <string>

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Plugin
//===----------------------------------------------------------------------===//

Plugin::Plugin(const std::string &path)
    : handle(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)), soPath(path) {}

Plugin::~Plugin() {
  if (handle)
    dlclose(handle);
}

bool Plugin::isLoaded() const { return handle != nullptr; }

//===----------------------------------------------------------------------===//
// PluginManager
//===----------------------------------------------------------------------===//

PluginManager::PluginManager() {
  // Read the mojo configuration.
  ErrorOr<MojoConfig> configOr = MojoConfig::open();
  SmallVector<std::string> pluginPaths;
  std::string soPath = "libmojo-compiler-plugin.so";
  if (auto envPath = llvm::sys::Process::GetEnv("MODULAR_COMPILER_PLUGINS"))
    soPath = *envPath;

  // Where to find plugins:
  // 1. Try get plugin paths from modular.cfg.
  // 2. If there is no plugins found in modular.cfg, try env var
  // MODULAR_COMPILER_PLUGINS.
  // modular.cfg takes precedence over MODULAR_COMPILER_PLUGINS.
  if (!configOr.isError()) {
    MojoConfig config = std::move(*configOr);
    pluginPaths = config.getPluginPaths();
  }

  if (pluginPaths.empty()) {
    SmallVector<StringRef> strRefPaths;
    StringRef(soPath).split(strRefPaths, ';', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
    pluginPaths = llvm::map_to_vector(strRefPaths, &StringRef::str);
  }

  for (auto path : pluginPaths) {
    auto plugin = std::make_unique<Plugin>(path);
    if (plugin->isLoaded())
      plugins.push_back(std::move(plugin));
  }
}
