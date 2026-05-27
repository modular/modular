//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_PLUGINUTILS_H
#define KGEN_SUPPORT_PLUGINUTILS_H

#include "Support/Buffer.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "llvm/ADT/StringRef.h"

#include <dlfcn.h>
#include <string>

namespace M::KGEN {

class CompilationOptions;

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

/// Declares three members inside a Plugin class for a single plugin entry
/// point named "M_KGEN_<camelName>" in the loaded .so:
///   1. `<PascalName>Fn`      — function-pointer type alias.
///   2. `get<PascalName>Fn()` — resolves the symbol via dlsym.
///   3. `<camelName>(...)`    — convenience wrapper that calls (2) and
///                              forwards the arguments.
///
/// Example — given:
///   REGISTER_GET_KGEN_PLUGIN_FN(FooBar, fooBar, int, float, bool)
///
/// Expands to:
///   using FooFn = int (*)(float, bool);
///   ErrorOr<FooBarFn> getFooBarFn() const {
///     return getFunction<FooBarFn>(handle, "M_KGEN_fooBar");
///   }
///   template <typename... Args>
///   int fooBar(Args &&...args) const {
///     auto fnOr = getFooBarFn();
///     if (fnOr.isError()) return fnOr.takeError();
///     return (*fnOr)(std::forward<Args>(args)...);
///   }
#define REGISTER_GET_KGEN_PLUGIN_FN(PascalName, camelName, Result, ...)        \
  using PascalName##Fn = Result (*)(__VA_ARGS__);                              \
  ErrorOr<PascalName##Fn> get##PascalName##Fn() const {                        \
    return getFunction<PascalName##Fn>(handle, "M_KGEN_" #camelName);          \
  }                                                                            \
  /* Forwards to the looked-up plugin function. */                             \
  template <typename... Args>                                                  \
  Result camelName(Args &&...args) const {                                     \
    auto fnOr = get##PascalName##Fn();                                         \
    if (fnOr.isError())                                                        \
      return fnOr.takeError();                                                 \
    return (*fnOr)(std::forward<Args>(args)...);                               \
  }

/// Declares one member inside a PluginManager class that forwards a call to
/// the corresponding `Plugin::<camelName>` method on the active plugin.
///
/// Example — given:
///   REGISTER_CALL_KGEN_PLUGIN_FN(FooBar, fooBar, int, float, bool)
///
/// Expands to:
///   template <typename... Args>
///   int fooBar(Args &&...args) const {
///     if (!currPlugin)
///       return Error("PluginManager is not set for the target yet");
///     return currPlugin->fooBar(args...);
///   }
#define REGISTER_CALL_KGEN_PLUGIN_FN(PascalName, camelName, Result, ...)       \
  template <typename... Args>                                                  \
  Result camelName(Args &&...args) const {                                     \
    if (!currPlugin)                                                           \
      return Error("PluginManager is not set for the target yet");             \
    return currPlugin->camelName(std::forward<Args>(args)...);                 \
  }

/// Compiler plugin which can be loaded at runtime to extend the compiler's
/// functionality.
class Plugin {
public:
  Plugin(const std::string &path);
  ~Plugin();

  void *getHandle() const { return handle; }

  /// Plugin API for create shared object file.
  using IsPluginForTargetFn = M::ErrorOr<bool> (*)(llvm::StringRef);

  /// Check if plugin is for a specific target.
  bool isPluginForTarget(StringRef targetTriple) const;
  bool isPluginForTarget(const llvm::Triple &targetTriple) const;

  /// Check if the plugin was successfully loaded.
  bool isLoaded() const;

  /// Plugin API for creating a shared object file.
  REGISTER_GET_KGEN_PLUGIN_FN(CreateSharedObject, createSharedObject,
                              M::ErrorOr<M::BufferRef>, M::BufferRef,
                              CompilationOptions, llvm::StringRef,
                              const std::string &)

  /// Plugin API for registering patterns that lower POP ops to LLVM.
  REGISTER_GET_KGEN_PLUGIN_FN(PopulateLowerPOPToLLVMPatterns,
                              populateLowerPOPToLLVMPatterns, M::ErrorOrSuccess,
                              mlir::RewritePatternSet &,
                              mlir::LLVMTypeConverter &, M::TargetInfoAttr)

  /// Plugin API for registering patterns that lower global POP ops to LLVM.
  REGISTER_GET_KGEN_PLUGIN_FN(PopulateLowerGlobalPOPToLLVMPatterns,
                              populateLowerGlobalPOPToLLVMPatterns,
                              M::ErrorOrSuccess, mlir::RewritePatternSet &,
                              mlir::LLVMTypeConverter &, mlir::SymbolTable &,
                              M::TargetInfoAttr)

  const std::string &getSoPath() const { return soPath; }

private:
  void *handle = nullptr;

  /// Plugin path.
  std::string soPath;
};

/// Compiler plugin manager
class PluginManager {
public:
  PluginManager();
  PluginManager(StringRef targetTriple);
  PluginManager(const PluginManager &);
  PluginManager(PluginManager &&);
  ~PluginManager() = default;

  bool hasPluginForTarget(StringRef targetTriple) const;
  bool hasPluginForTarget(const llvm::Triple &targetTriple) const;

  /// Pick the active plugin for `targetTriple` from the already-loaded
  /// plugins. Needed because the default ctor loads plugins without a
  /// triple, leaving `currPlugin` null. Default ctor should only be invoked
  /// when this pass is run alone for unit tests through ken-opt instead of
  /// being part of the pipeline. Plugin mutation needs to be careful with
  /// concurrency since the same pass can be run in parallel for multiple
  /// functions.
  void selectPluginForTarget(StringRef targetTriple);

  /// Plugin API for creating a shared object file.
  REGISTER_CALL_KGEN_PLUGIN_FN(CreateSharedObject, createSharedObject,
                               M::ErrorOr<M::BufferRef>, M::BufferRef,
                               CompilationOptions, llvm::StringRef,
                               const std::string &)

  /// Plugin API for registering patterns that lower POP ops to LLVM.
  REGISTER_CALL_KGEN_PLUGIN_FN(PopulateLowerPOPToLLVMPatterns,
                               populateLowerPOPToLLVMPatterns,
                               M::ErrorOrSuccess, mlir::RewritePatternSet &,
                               mlir::LLVMTypeConverter &, M::TargetInfoAttr)

  /// Plugin API for registering patterns that lower global POP ops to LLVM.
  REGISTER_CALL_KGEN_PLUGIN_FN(PopulateLowerGlobalPOPToLLVMPatterns,
                               populateLowerGlobalPOPToLLVMPatterns,
                               M::ErrorOrSuccess, mlir::RewritePatternSet &,
                               mlir::LLVMTypeConverter &, mlir::SymbolTable &,
                               M::TargetInfoAttr)

private:
  std::vector<std::unique_ptr<Plugin>> plugins;
  Plugin *currPlugin = nullptr;
};

} // namespace M::KGEN

#endif
