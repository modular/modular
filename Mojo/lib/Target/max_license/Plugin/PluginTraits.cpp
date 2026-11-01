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
// Per-target metadata for targets serviced by a dynamically-loaded compiler
// plugin: a registered dispatcher that resolves a triple to the `TargetTraits`
// each loaded plugin hands back through its `M_KGEN_getTargetTraits` entry
// point. The dlopen'd plugin cannot register with this process's registry at
// static-init, so the dispatcher pulls the traits at resolve time instead.
//
//===----------------------------------------------------------------------===//

#include "Target/TargetTraits.h"

#include "Mojo/Support/max_license/PluginUtils.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <vector>

namespace M::KGEN {
namespace {

/// Signature of a plugin's `M_KGEN_getTargetTraits` entry point.
using GetTargetTraitsFn = const TargetTraits *(*)();

/// Loaded plugins and the `TargetTraits` each contributes. The PluginManager
/// owns the plugins the traits point into, so it is kept alongside them in a
/// function-local static that lives for the process.
struct LoadedPlugins {
  PluginManager manager;
  std::vector<const TargetTraits *> traits;

  LoadedPlugins() {
    for (const std::unique_ptr<Plugin> &plugin : manager.getPlugins()) {
      ErrorOr<GetTargetTraitsFn> fnOr = getFunction<GetTargetTraitsFn>(
          plugin->getHandle(), "M_KGEN_getTargetTraits");
      if (fnOr.isError())
        continue;
      if (const TargetTraits *traitsForPlugin = (*fnOr)())
        traits.push_back(traitsForPlugin);
    }
  }
};

const LoadedPlugins &loadedPlugins() {
  static LoadedPlugins plugins;
  return plugins;
}

/// Dispatcher for plugin-contributed traits: never describes a target itself
/// (`matches()` is false); `resolve()` hands off to a plugin's traits, which
/// the registry prefers over a direct match.
class PluginTraits final : public TargetTraits {
public:
  llvm::StringRef name() const override { return "plugin"; }

  bool matches(const llvm::Triple &) const override { return false; }

  const TargetTraits *resolve(const llvm::Triple &triple) const override {
    for (const TargetTraits *traits : loadedPlugins().traits) {
      if (traits->matches(triple))
        return traits;
    }
    return nullptr;
  }

  // Never queried on the dispatcher itself; callers resolve first.
  llvm::StringRef getAsmExtension() const override {
    llvm_unreachable(
        "PluginTraits dispatcher has no extensions; resolve first");
  }
  llvm::StringRef getLLVMExtension() const override {
    llvm_unreachable(
        "PluginTraits dispatcher has no extensions; resolve first");
  }
  llvm::StringRef getObjectExtension() const override {
    llvm_unreachable(
        "PluginTraits dispatcher has no extensions; resolve first");
  }
  llvm::StringRef getBitcodeExtension() const override {
    llvm_unreachable(
        "PluginTraits dispatcher has no extensions; resolve first");
  }

  // Queried when rendering the `--help` emission-kind table, which iterates
  // every registered traits object; empty omits the dispatcher from it.
  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return {};
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<PluginTraits> registerPluginTraits;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
