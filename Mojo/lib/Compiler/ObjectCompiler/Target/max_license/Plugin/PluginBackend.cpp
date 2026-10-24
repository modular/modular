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
// LLVM-codegen policy for targets serviced by a dynamically-loaded compiler
// plugin: a registered dispatcher that resolves a triple to the `TargetBackend`
// each loaded plugin hands back through its `M_KGEN_getTargetBackend` entry
// point. The plugin's registry can't be reached at static-init, so the
// dispatcher pulls the backend at resolve time.
//
//===----------------------------------------------------------------------===//

#include "Target/max_license/Plugin/PluginBackend.h"

#include "Mojo/Support/max_license/PluginUtils.h"

#include "llvm/TargetParser/Triple.h"

#include <vector>

namespace M::KGEN {
namespace {

/// Signature of a plugin's `M_KGEN_getTargetBackend` entry point.
using GetTargetBackendFn = const TargetBackend *(*)();

/// Loaded plugins and the `TargetBackend` each contributes. The PluginManager
/// owns the plugins the backends point into, so it is kept alongside them in a
/// function-local static that lives for the process.
struct LoadedPlugins {
  PluginManager manager;
  std::vector<const TargetBackend *> backends;

  LoadedPlugins() {
    for (const std::unique_ptr<Plugin> &plugin : manager.getPlugins()) {
      ErrorOr<GetTargetBackendFn> fnOr = getFunction<GetTargetBackendFn>(
          plugin->getHandle(), "M_KGEN_getTargetBackend");
      if (fnOr.isError())
        continue;
      if (const TargetBackend *backend = (*fnOr)())
        backends.push_back(backend);
    }
  }
};

const LoadedPlugins &loadedPlugins() {
  static LoadedPlugins plugins;
  return plugins;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<PluginBackend> registerPluginBackend;
#pragma GCC diagnostic pop

} // namespace

const TargetBackend *PluginBackend::resolve(const llvm::Triple &triple) const {
  for (const TargetBackend *backend : loadedPlugins().backends) {
    if (backend->matches(triple))
      return backend;
  }
  return nullptr;
}

} // namespace M::KGEN
