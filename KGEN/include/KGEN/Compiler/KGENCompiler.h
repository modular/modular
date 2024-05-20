//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_KGENCOMPILER_H
#define KGEN_COMPILER_KGENCOMPILER_H

#include "Cache/CachedTransform.h"
#include "KGEN/Compiler/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace M::KGEN {
class PackageLinkOp;

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

/// This populates the passes to produce a fully concrete KGEN module. It's the
/// equivalent of the `buildElaborateModulePipeline` function, but with common
/// defaults for elaboration handlers.
void populateElaborateModulePasses(mlir::PassManager &pm, TargetInfoAttr target,
                                   const CompilationOptions &options);

//===----------------------------------------------------------------------===//
// Caching
//===----------------------------------------------------------------------===//

/// Returns Mojo transform backend, or an error if the backend could not be
/// created.
ErrorOr<RCRef<Cache::BlobCacheBackend>> getMojoCacheBackend();

//===----------------------------------------------------------------------===//
// KGENCompilerLayer
//===----------------------------------------------------------------------===//

/// Forward declarations for the KGENCompilerLayer.
class ObjectCompilerLayer;

/// Provide an ExecutionEngine layer for the KGEN compiler. This wraps a call to
/// the pass manager, and on materialization, it will run compilation and then
/// delegate the rest to the base layer. Under the hood it also defines a
/// MaterializationUnit and uses that to emit symbols on-demand.
class KGENCompilerLayer : public MaterializationLayer {
public:
  KGENCompilerLayer(mlir::PassManager &pm, TargetInfoAttr target,
                    const CompilationOptions &options,
                    ObjectCompilerLayer &base,
                    RCRef<Cache::BlobCacheBackend> transformCacheBackend,
                    llvm::orc::ExecutionSession &sess,
                    const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Add a module to the JIT. This module will be modified in-place as
  /// compilation occurs, and will be forwarded to the ObjectCompilerLayer.
  ErrorOrSuccess add(StringRef libName, ModuleOp theModule);

  /// Given a library name and a module, emit the code for it. This runs
  /// the passes in `populateElaborateModulePasses` and calls `emit` on the
  /// ObjectCompilerLayer with the result.
  void emit(std::unique_ptr<llvm::orc::MaterializationResponsibility> mr,
            SymbolTable &symtab, const ExportMap &exportMap);

  static bool classof(const MaterializationLayer *layer) {
    return layer->getKind() == LayerKind::kKGENCompilerLayer;
  }

private:
  /// Conform to the ORC's interface and return a map of the exported symbols.
  /// Uses the export map that is built during `add` to provide the set of
  /// symbols that can be materialized.
  llvm::orc::MaterializationUnit::Interface
  getInterface(const ExportMap &exports);

  /// Provide KGENCompilerMaterializationUnit so that we can do codegen
  /// on-demand.
  class KGENCompilerMaterializationUnit;

private:
  mlir::PassManager &pm;
  TargetInfoAttr target;
  CompilationOptions options;
  ObjectCompilerLayer &baseLayer;

  RCRef<Cache::TransformCache> transformCache;
};

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

/// Sets up an ExecutionEngine instance for compiling Mojo. It handles
/// initializing the target machine, the cache backends, and the execution
/// engine itself. On success, the execution engine is returned.
ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>> initializeExecutionEngine(
    mlir::PassManager &pm, const KGEN::CompilationOptions &compilationOptions,
    KGEN::ExecutionEngineOptions executionEngineOptions, bool isJIT,
    TargetInfoAttr target, bool isSearch = false);

/// Run the library generation pipeline on the given module. If
/// `materializeDependencies` is true, the pipeline will ensure all dependencies
/// are materialized in the final module.
ErrorOrSuccess
runLibraryGenerationPipeline(ModuleOp module,
                             const KGEN::CompilationOptions &compileOptions,
                             bool materializeDependencies = false);

/// This creates the materialize packages pass with the default library
/// generation pipeline, i.e. `runLibraryGenerationPipeline`.
std::unique_ptr<Pass>
createMaterializePackagesWithDefaultGen(const CompilationOptions &options);

/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass> createElaborateGeneratorsWithDefaultJIT();

} // namespace M::KGEN

#endif // KGEN_COMPILER_KGENCOMPILER_H
