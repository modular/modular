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
  Plugin();
  ~Plugin();

  void *getHandle() const { return soHandle; }

  /// Plugin API for create shared object file.
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

private:
  /// Handle to the loaded plugin shared object. nullptr if the plugin failed to
  /// load.
  void *soHandle = nullptr;

  /// Plugin path.
  std::string soPath;
};

} // namespace M::KGEN

#endif
