//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/PluginUtils.h"
#include "KGEN/Support/Configuration.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

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

bool Plugin::isPluginForTarget(StringRef targetTriple) const {
  if (!handle)
    return false;

  ErrorOr<Plugin::IsPluginForTargetFn> fnOr =
      getFunction<IsPluginForTargetFn>(handle, "M_KGEN_isPluginForTarget");
  if (fnOr.isError())
    return false;

  ErrorOr<bool> resultOr = (*fnOr)(targetTriple);
  if (resultOr.isError())
    return false;
  return *resultOr;
}

bool Plugin::isPluginForTarget(const llvm::Triple &targetTriple) const {
  return isPluginForTarget(targetTriple.str());
}

/// Check if the plugin was successfully loaded.
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
  // Don't combine modular.cfg andMODULAR_COMPILER_PLUGINS.
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

PluginManager::PluginManager(StringRef targetTriple) : PluginManager() {
  llvm::SmallVector<std::string> paths;
  for (auto &plugin : plugins) {
    if (plugin->isPluginForTarget(targetTriple)) {
      currPlugin = plugin.get();
      paths.push_back(currPlugin->getSoPath());
    }
  }

  // Warn if there are multiple plugins for the same target.
  if (paths.size() > 1) {
    std::string str;
    llvm::interleave(
        paths, [&](const std::string &path) { str += path; },
        [&] { str += "\n"; });
    llvm::errs() << "warning: found multiple plugin for " << targetTriple
                 << "in: " << "\n"
                 << str << "\n";
  }
}

PluginManager::PluginManager(const PluginManager &other) {
  for (auto &otherPlugin : other.plugins) {
    auto plugin = std::make_unique<Plugin>(otherPlugin->getSoPath());
    if (plugin->isLoaded())
      plugins.push_back(std::move(plugin));
    if (other.currPlugin == otherPlugin.get())
      currPlugin = plugin.get();
  }
}

PluginManager::PluginManager(PluginManager &&other)
    : plugins(std::move(other.plugins)), currPlugin(other.currPlugin) {}

bool PluginManager::hasPluginForTarget(StringRef targetTriple) const {
  for (auto &plugin : plugins) {
    if (plugin->isPluginForTarget(targetTriple))
      return true;
  }

  return false;
}

bool PluginManager::hasPluginForTarget(const llvm::Triple &targetTriple) const {
  return hasPluginForTarget(targetTriple.str());
}
