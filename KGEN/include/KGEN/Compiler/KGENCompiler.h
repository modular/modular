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
#include "KGEN/ToolCommon/PassManagerConfigOptions.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace M::KGEN {
class PackageLinkOp;

class KGENCompiler {
public:
  KGENCompiler(
      MLIRContext &context, CompilationOptions options,
      PassManagerConfigOptions pmConfigOptions = PassManagerConfigOptions());

  /// Run KGEN compilation pipeline, including pre-elaboration passes,
  /// elaboration, and post-elaboration pass. Get the theModule ready before
  /// llvm lowering.
  AnyAsyncValueRef runKGENPipeline(
      ModuleOp theModule, TargetInfoAttr target,
      RCRef<Cache::TransformCache> transformCache, AnyAsyncValueRef chain,
      std::function<void(Operation *)> moreOnMiss = [](Operation *) {},
      std::function<void(Operation *)> moreOnHit = [](Operation *) {});

  ErrorOrSuccess runKGENPipeline(ModuleOp theModule, TargetInfoAttr target);

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

private:
  /// Compilation options.
  CompilationOptions options;

  /// PassManager configuration options.
  PassManagerConfigOptions pmConfigOptions;

  mlir::MLIRContext &context;
};

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

/// Sets up an ExecutionEngine instance for compiling Mojo. It handles
/// initializing the target machine, the cache backends, and the execution
/// engine itself. On success, the execution engine is returned.
ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>>
initializeExecutionEngine(mlir::MLIRContext &context,
                          const KGEN::CompilationOptions &compilationOptions,
                          KGEN::ExecutionEngineOptions executionEngineOptions,
                          bool isJIT, PassManagerConfigOptions pmOptions,
                          bool isSearch = false);

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
