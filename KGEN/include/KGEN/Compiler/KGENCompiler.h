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
#include "mlir/Pass/PassManager.h"

namespace M::KGEN {
class PackageLinkOp;

class KGENCompiler {
public:
  KGENCompiler(MLIRContext &context, const CompilationOptions &options)
      : options(options), pm(&context) {}

  KGENCompiler(MLIRContext &context, StringRef operationName,
               const CompilationOptions &options)
      : options(options), pm(&context, operationName) {}

  mlir::PassManager &getPassManager() { return pm; }

  /// Provide a lambda to set the configuration of the PassManager.
  void configurePassManager(std::function<void(mlir::PassManager &pm)> config);

  /// Run KGEN compilation pipeline, including pre-elaboration passes,
  /// elaboration, and post-elaboration pass. Get the theModule ready before
  /// llvm lowering.
  ErrorOrSuccess runKGENPipeline(ModuleOp theModule, bool isJIT,
                                 TargetInfoAttr target, bool isSearch = false);

  /// Run the library generation pipeline on the given module. If
  /// `materializeDependencies` is true, the pipeline will ensure all
  /// dependencies are materialized in the final module.
  ErrorOrSuccess
  runGenerateLibraryPipeline(ModuleOp module,
                             bool materializeDependencies = false);

  /// Run post-parser pipeline that checks and lowers source-level
  /// LIT constructs.
  LogicalResult runCheckLITPipeline(ModuleOp module);

  /// Run the elaboration and post-elaboration pipeline
  /// This doesn't not include check LIT and pre-elaboration passes.
  /// This allows the transform to be cached if chain is provided.
  AnyAsyncValueRef runElaborationPipeline(
      ModuleOp module, TargetInfoAttr target, LLCL::Runtime &runtime,
      std::optional<AnyAsyncValueRef> chain = std::nullopt,
      std::function<void(Operation *)> moreOnMiss = [](Operation *) {},
      std::function<void(Operation *)> moreOnHit = [](Operation *) {});

  /// Run the post-elaboration optimization and simplification passes pipeline.
  /// These passes are intended to run immediately after the elaborator.
  LogicalResult runPostElaborationOnlyPipeline(ModuleOp module);

  /// Build the pipeline to convert post-elaboration KGEN IR to LLVM IR.
  /// The pipeline runs the canonicalizer, the KGEN to LLVM conversion, a series
  /// of LLVM lowerings, and the canonicalizer again.
  LogicalResult runLowerToLLVMPipeline(ModuleOp module,
                                       const LowerToLLVMOptions &option);

private:
  CompilationOptions options;
  mlir::PassManager pm;
};

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

/// This populates the passes to produce a fully concrete KGEN module. It's the
/// equivalent of the `buildElaborateModulePipeline` function, but with common
/// defaults for elaboration handlers.
void populateElaborateModulePasses(mlir::PassManager &pm, TargetInfoAttr target,
                                   const CompilationOptions &options);

//===----------------------------------------------------------------------===//
// PostElaborationPipeline
//===----------------------------------------------------------------------===//

/// This populates the post-elaboration optimization and simplification passes.
/// These passes are intended to run immediately after the elaborator.
void buildPostElaborationPipeline(mlir::PassManager &pm,
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

/// This creates the materialize packages pass with the default library
/// generation pipeline, i.e. `runGenerateLibraryPipeline`.
std::unique_ptr<Pass>
createMaterializePackagesWithDefaultGen(const CompilationOptions &options);

/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass> createElaborateGeneratorsWithDefaultJIT();

} // namespace M::KGEN

#endif // KGEN_COMPILER_KGENCOMPILER_H
