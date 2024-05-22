//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_KGENCOMPILER_H
#define KGEN_COMPILER_KGENCOMPILER_H

#include "Cache/CachedTransform.h"
#include "KGEN/ExecutionEngine/ExecutionEngine.h"
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
// Default JIT Configuration
//===----------------------------------------------------------------------===//

/// Sets up an ExecutionEngine instance for compiling Mojo. It handles
/// initializing the target machine, the cache backends, and the execution
/// engine itself. On success, the execution engine is returned.
ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>> initializeExecutionEngine(
    mlir::PassManager &pm, const KGEN::CompilationOptions &compilationOptions,
    KGEN::ExecutionEngineOptions executionEngineOptions, bool isJIT,
    TargetInfoAttr target, bool isSearch = false);

/// Run KGEN compilation pipeline, including pre-elaboration passes,
/// elaboration, and post-elaboration pass. Get the theModule ready before llvm
/// lowering.
ErrorOrSuccess
runKGENPipeline(mlir::PassManager &pm, ModuleOp theModule,
                const KGEN::CompilationOptions &compilationOptions, bool isJIT,
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
