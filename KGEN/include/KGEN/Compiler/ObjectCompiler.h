//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_OBJECTCOMPILER_H
#define KGEN_COMPILER_OBJECTCOMPILER_H

#include "Cache/BlobCache.h"
#include "Cache/CachedTransform.h"
#include "KGEN/Compiler/LLVMIRUtils.h"
#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/PassManagerConfigOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include <filesystem>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
class DataLayout;
namespace orc {
class ExecutionSession;
} // namespace orc
} // namespace llvm

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

/// The purpose of this class is to provide methods to lower concrete KGEN
/// functions to LLVM, and then to objects.
class ObjectCompiler {
public:
  /// Construct an ObjectCompiler that infers the exports from the module.
  static ErrorOr<std::unique_ptr<ObjectCompiler>>
  create(StringRef basePath, CompilationOptions options, bool isJIT,
         MLIRContext &context,
         PassManagerConfigOptions pmOptions = PassManagerConfigOptions());

  /// Produce a standalone MLIR module by slicing out the dependencies of the
  /// provided exported ops.
  OwningOpRef<ModuleOp>
  produceStandaloneModule(const SymbolTable &symtab,
                          const ExportMap &exportedSymbols);
  /// Produce a standalone MLIR module by slicing out the dependencies of the
  /// provided exported ops. An `IRMapping` can be provided to be able to map
  /// into the sliced module.
  OwningOpRef<ModuleOp>
  produceStandaloneModule(const SymbolTable &symtab,
                          const ExportMap &exportedSymbols, IRMapping &mapping);

  /// Emit the module to a object archive.
  ErrorOr<BufferRef> emitArchive(ModuleOp module);

  /// Emit the module to a object archive as an ElementsAttr that can be used as
  /// an attribute on another operation.
  ErrorOr<ElementsAttr> emitArchiveAttr(ModuleOp module);

  /// Lower the given module to LLVM. Returns the LLVM module on success, and
  /// nullptr on failure.
  ErrorOr<std::unique_ptr<llvm::Module>>
  lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// assembly file. The assembly output is written to the provided stream.
  ErrorOrSuccess emitAssembly(ModuleOp module, llvm::raw_pwrite_stream &os);

  /// Writes C++ function declarations for all exported symbols.
  LogicalResult emitCXXHeader(ModuleOp module, StringRef filename,
                              raw_ostream &os);

  /// Get a reference to the object compiler's transform cache.
  RCRef<Cache::TransformCache> getTransformCache() {
    return transformCache.copy();
  }

  /// Get whether compilation is for JIT.
  bool getIsJIT() const { return isJIT; }

private:
  /// Construct an ObjectCompiler with a specific set of exports.
  ObjectCompiler(
      RCRef<Cache::BlobCacheBackend> transformCache, CompilationOptions options,
      bool isJIT, MLIRContext &context,
      PassManagerConfigOptions pmOptions = PassManagerConfigOptions());

  /// Lower the given LLVM module to an object file (parLLC = false) or
  /// multiple object files per function (parLLC = true).
  LLCL::AsyncValueRef<SmallVector<BufferRef>>
  lowerLLVMModuleToObjects(LLVMModuleAndContext module, Location loc,
                           bool parLLC, std::optional<size_t> moduleIdx);

  /// The caches needed for compilation.
  RCRef<Cache::TransformCache> transformCache;

  /// The compilation options to use.
  CompilationOptions options;

  /// This is a bit odd, but since we use this layer to generate code for cases
  /// where we aren't going to immediately execute it, we need to be able to
  /// change the codegen mode.
  bool isJIT;

  /// PassManager configuration options.
  PassManagerConfigOptions pmOptions;

  /// The MLIR context.
  MLIRContext &context;

  /// The LLCL runtime.
  LLCL::Runtime &runtime;

  friend class ObjectCompilerLayer;
};

/// Setup the machine properties from the provided target.
ErrorOr<std::unique_ptr<llvm::TargetMachine>>
createTargetMachine(const CompilationOptions &options, bool isJIT);

//===----------------------------------------------------------------------===//
// compileLLVMToAssembly
//===----------------------------------------------------------------------===//

/// Compile the given LLVM module to an object file and write it to objStream.
LogicalResult compileLLVMToAssembly(LLVMModuleAndContext module,
                                    llvm::TargetMachine &targetMachine,
                                    llvm::raw_pwrite_stream &objStream,
                                    CompilationOptions &options,
                                    LLCL::Runtime &runtime);

} // namespace M::KGEN

#endif // KGEN_COMPILER_OBJECTCOMPILER_H
