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

#include <string>

namespace M::KGEN {

class CompilationOptions;

/// Compiler plugin which can be loaded at runtime to extend the compiler's
/// functionality.
class Plugin {
public:
  Plugin(StringRef targetTriple = {}, ArrayRef<StringRef> pluginPaths = {});
  Plugin(const std::vector<std::string> &paths);
  ~Plugin();

  void *getHandle() const { return currHandle; }

  bool isPluginForTarget(StringRef targetTriple) const;
  bool isPluginForTarget(const llvm::Triple &targetTriple) const;

  /// Plugin API for create shared object file.
  using IsPluginForTargetFn = M::ErrorOr<bool> (*)(llvm::StringRef);

  using CreateSharedObjectFn = M::ErrorOr<M::BufferRef> (*)(
      M::BufferRef, CompilationOptions, llvm::StringRef, const std::string &);
  ErrorOr<CreateSharedObjectFn> getCreateSharedObjectFn() const;

  /// Plugin API for registering patterns that lower POP ops to LLVM
  using PopluateLowerPOPToLLVMPatternsFn = M::ErrorOrSuccess (*)(
      mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
      M::TargetInfoAttr targetInfo);
  ErrorOr<PopluateLowerPOPToLLVMPatternsFn>
  getPopulateLowerPOPToLLVMPatternsFn() const;

  /// Plugin API for registering patterns that lower global POP ops to LLVM
  using PopluateLowerGlobalPOPToLLVMPatternsFn = M::ErrorOrSuccess (*)(
      mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
      mlir::SymbolTable &symtab, M::TargetInfoAttr targetInfo);
  ErrorOr<PopluateLowerGlobalPOPToLLVMPatternsFn>
  getPopulateLowerGlobalPOPToLLVMPatternsFn() const;

  /// Check if the plugin was successfully loaded.
  ErrorOrSuccess isLoaded() const;

  const std::vector<std::string> &getSoPaths() const { return soPaths; }

private:
  /// Handle to the loaded plugin shared object. nullptr if the plugin failed to
  /// load.
  std::vector<void *> soHandles;
  void *currHandle = nullptr;

  /// Plugin path.
  std::vector<std::string> soPaths;

  bool isPluginForTarget(void *hdl, StringRef targetTriple) const;
};

} // namespace M::KGEN

#endif
