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

Plugin::Plugin() {
  // Load the plugin. MODULAR_COMPILER_PLUGINS overrides the path,
  // e.g. when running from a Bazel-built binary where the .so lives
  // in the runfiles tree rather than on LD_LIBRARY_PATH.
  soPath = "libmojo-compiler-plugin.so";
  if (auto envPath = llvm::sys::Process::GetEnv("MODULAR_COMPILER_PLUGINS"))
    soPath = *envPath;
  soHandle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
}

Plugin::~Plugin() {
  if (soHandle)
    dlclose(soHandle);
}

ErrorOrSuccess Plugin::isLoaded() const {
  if (soHandle) {
    return success();
  } else {
    return Error(llvm::StringRef("failed to load " + soPath + ": ") +
                 dlerror());
  }
}

template <typename FnType>
ErrorOr<FnType> getFunction(const Plugin *plugin, llvm::StringRef symbolName) {
  if (auto loadError = plugin->isLoaded()) {
    return loadError.takeError();
  }
  // Resolve the symbol
  auto fnPtr = reinterpret_cast<FnType>(
      dlsym(plugin->getHandle(), symbolName.str().c_str()));

  if (!fnPtr) {
    return Error(llvm::StringRef("failed to resolve ") + symbolName +
                 dlerror());
  }
  return fnPtr;
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
