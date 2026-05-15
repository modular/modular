//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/PluginUtils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <dlfcn.h>
#include <fstream>
#include <string>

using namespace M;
using namespace KGEN;

template <typename FnType>
ErrorOr<FnType> getFunction(void *hdl, llvm::StringRef symbolName) {
  // Resolve the symbol
  auto fnPtr = reinterpret_cast<FnType>(dlsym(hdl, symbolName.str().c_str()));

  if (!fnPtr) {
    return Error(llvm::StringRef("failed to resolve ") + symbolName +
                 dlerror());
  }
  return fnPtr;
}

Plugin::Plugin(StringRef targetTriple, ArrayRef<StringRef> pluginPaths) {
  // Try plugin paths first
  for (auto path : pluginPaths) {
    void *hdl = dlopen(path.str().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!hdl)
      continue;
    if (!targetTriple.empty() && isPluginForTarget(hdl, targetTriple)) {
      soHandles.push_back(hdl);
      currHandle = hdl;
      return;
    }
    soPaths.push_back(path.str());
    soHandles.push_back(hdl);
  }

  if (!soHandles.empty())
    return;

  // Load the plugin. MODULAR_COMPILER_PLUGINS overrides the path,
  // e.g. when running from a Bazel-built binary where the .so lives
  // in the runfiles tree rather than on LD_LIBRARY_PATH.
  std::string soPath = "libmojo-compiler-plugin.so";
  if (auto envPath = llvm::sys::Process::GetEnv("MODULAR_COMPILER_PLUGINS"))
    soPath = *envPath;

  currHandle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (currHandle) {
    soHandles.push_back(currHandle);
    soPaths.push_back(std::move(soPath));
  }
}

Plugin::Plugin(const std::vector<std::string> &paths) {
  soPaths = paths;
  for (auto path : soPaths) {
    void *hdl = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    soHandles.push_back(hdl);
  }

  if (!soHandles.empty())
    currHandle = soHandles.front();
}

Plugin::~Plugin() {
  for (auto hdl : soHandles) {
    dlclose(hdl);
  }
}

bool Plugin::isPluginForTarget(void *hdl, StringRef targetTriple) const {
  if (!hdl)
    return false;

  ErrorOr<Plugin::IsPluginForTargetFn> fnOr =
      getFunction<IsPluginForTargetFn>(hdl, "M_KGEN_isPluginForTarget");
  if (fnOr.isError())
    return false;

  ErrorOr<bool> resultOr = (*fnOr)(targetTriple);
  if (resultOr.isError())
    return false;
  return *resultOr;
}

bool Plugin::isPluginForTarget(StringRef targetTriple) const {

  if (soHandles.empty())
    return false;
  if (currHandle && isPluginForTarget(currHandle, targetTriple))
    return true;

  for (auto hdl : soHandles) {
    if (isPluginForTarget(hdl, targetTriple)) {
      return true;
    }
  }
  return false;
}

bool Plugin::isPluginForTarget(const llvm::Triple &targetTriple) const {
  return isPluginForTarget(targetTriple.str());
}

ErrorOrSuccess Plugin::isLoaded() const {
  if (currHandle) {
    return success();
  } else {
    std::string pathStr;
    llvm::raw_string_ostream ss(pathStr);
    llvm::interleaveComma(soPaths, ss);
    if (soHandles.empty()) {
      return Error(llvm::StringRef("failed to load plugin(s) from path(s): " +
                                   pathStr + ": ") +
                   dlerror());
    } else {
      return Error(llvm::StringRef("failed to set plugin for a target ") +
                   dlerror());
    }
  }
}

template <typename FnType>
ErrorOr<FnType> getFunction(const Plugin *plugin, llvm::StringRef symbolName) {
  if (auto loadError = plugin->isLoaded()) {
    return loadError.takeError();
  }
  return getFunction<FnType>(plugin->getHandle(), symbolName);
}

ErrorOr<Plugin::CreateSharedObjectFn> Plugin::getCreateSharedObjectFn() const {
  return getFunction<CreateSharedObjectFn>(this, "M_KGEN_createSharedObject");
}

ErrorOr<Plugin::PopluateLowerPOPToLLVMPatternsFn>
Plugin::getPopulateLowerPOPToLLVMPatternsFn() const {
  return getFunction<PopluateLowerPOPToLLVMPatternsFn>(
      this, "M_KGEN_populateLowerPOPToLLVMPatterns");
}

ErrorOr<Plugin::PopluateLowerGlobalPOPToLLVMPatternsFn>
Plugin::getPopulateLowerGlobalPOPToLLVMPatternsFn() const {
  return getFunction<PopluateLowerGlobalPOPToLLVMPatternsFn>(
      this, "M_KGEN_populateLowerGlobalPOPToLLVMPatterns");
}
