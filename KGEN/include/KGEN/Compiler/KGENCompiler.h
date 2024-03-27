//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_KGENCOMPILER_H
#define KGEN_COMPILER_KGENCOMPILER_H

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/Compiler/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

/// This populates the passes to produce a fully concrete KGEN module. It's the
/// equivalent of the `buildElaborateModulePipeline` function, but with common
/// defaults for elaboration handlers.
void populateElaborateModulePasses(mlir::PassManager &pm,
                                   LLCL::Runtime &runtime,
                                   TargetInfoAttr target,
                                   const CompilationOptions &options,
                                   EvaluatorExecutorFn evaluatorExecutorFn,
                                   PackageGenLibraryFn packageGenLibraryFn);
void populateElaborateModulePasses(mlir::PassManager &pm,
                                   LLCL::Runtime &runtime,
                                   TargetInfoAttr target,
                                   const CompilationOptions &options,
                                   PackageGenLibraryFn packageGenLibraryFn);

//===----------------------------------------------------------------------===//
// Caching
//===----------------------------------------------------------------------===//

/// Returns Mojo transform and caching backends, or an error if the backend
/// objects could not be created.
ErrorOr<
    std::pair<RCRef<Cache::BlobCacheBackend>, RCRef<Cache::BlobCacheBackend>>>
getMojoCacheBackends(LLCL::Runtime &runtime);

//===----------------------------------------------------------------------===//
// KGENCompilerLayer
//===----------------------------------------------------------------------===//

/// Forward declarations for the KGENCompilerLayer.
class ObjectCompilerLayer;

/// Provide an ExecutionEngine layer for the KGEN compiler. This wraps a call to
/// the pass manager, and on materialization, it will run compilation and then
/// delegate the rest to the base layer. Under the hood it also defines a
/// MaterializationUnit and uses that to emit symbols on-demand.
class KGENCompilerLayer
    : public llvm::RTTIExtends<KGENCompilerLayer, MaterializationLayer> {
public:
  static char ID;

  KGENCompilerLayer(mlir::PassManager &pm, LLCL::Runtime &runtime,
                    TargetInfoAttr target, const CompilationOptions &options,
                    ObjectCompilerLayer &base,
                    RCRef<Cache::BlobCacheBackend> transformCacheBackend,
                    RCRef<Cache::BlobCacheBackend> regionCacheBackend,
                    llvm::orc::ExecutionSession &sess,
                    const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Add a module to the JIT. This module will be modified in-place as
  /// compilation occurs, and will be forwarded to the ObjectCompilerLayer.
  ErrorOrSuccess add(StringRef libName, ModuleOp theModule,
                     PackageGenLibraryFn packageGenLibraryFn);

  /// Given a library name and a module, emit the code for it. This runs
  /// the passes in `populateElaborateModulePasses` and calls `emit` on the
  /// ObjectCompilerLayer with the result.
  void emit(std::unique_ptr<llvm::orc::MaterializationResponsibility> mr,
            SymbolTable &symtab, const ExportMap &exportMap);

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
  LLCL::Runtime &runtime;
  TargetInfoAttr target;
  CompilationOptions options;
  ObjectCompilerLayer &baseLayer;

  RCRef<Cache::RegionCache> regionCache;
  RCRef<Cache::TransformCache> transformCache;
};

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

/// A default specialization evaluator that JITs and invokes the specialized
/// functions with the provided evaluator.
ErrorOr<ElaboratorSearchFn>
evaluateSpecializations(FuncOp evaluator, const SymbolTable &symtab,
                        LLCL::Runtime &runtime, TargetInfoAttr target,
                        const CompilationOptions &options,
                        ArrayRef<FuncOp> specializations);

/// Given the pre-elaboration function `func` belonging to a module with the
/// symbol table `symtab`, slice out a standalone module rooted at `func` and
/// elaborate it and compile to assembly for the provided `target.
ErrorOr<CrossDeviceFunction>
compileElaboratorAsm(GeneratorOp func, SymbolConstantAttr symbol,
                     StringAttr name, const SymbolTable &symtab,
                     LLCL::Runtime &runtime, TargetInfoAttr target,
                     EmissionKind emissionKind, CompilationOptions options);

/// Sets up an ExecutionEngine instance for compiling Mojo. It handles
/// initializing the target machine, the cache backends, and the execution
/// engine itself. On success, the execution engine is returned.
ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>>
initializeExecutionEngine(LLCL::Runtime &runtime, mlir::PassManager &pm,
                          const KGEN::CompilationOptions &compilationOptions,
                          KGEN::ExecutionEngineOptions executionEngineOptions,
                          bool isJIT, TargetInfoAttr target,
                          bool isSearch = false);

} // namespace M::KGEN

#endif // KGEN_COMPILER_KGENCOMPILER_H
