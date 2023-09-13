//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_H
#define KGEN_COMPILER_H

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENPasses.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN {
/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass>
createElaborateGeneratorsWithDefaultJIT(LLCL::Runtime &runtime);

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

/// This populates the passes to produce a fully concrete KGEN module. It is the
/// same as the function above, but allows the user to specify their own JIT.
void populateElaborateModulePasses(mlir::PassManager &pm,
                                   LLCL::Runtime &runtime,
                                   TargetInfoAttr target, BuildInfoAttr build,
                                   const CompilationOptions &options);

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
                    TargetInfoAttr target, BuildInfoAttr build,
                    const CompilationOptions &options,
                    ObjectCompilerLayer &base,
                    RCRef<Cache::BlobCacheBackend> transformCacheBackend,
                    RCRef<Cache::BlobCacheBackend> regionCacheBackend,
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
  BuildInfoAttr build;
  CompilationOptions options;
  ObjectCompilerLayer &baseLayer;

  RCRef<Cache::RegionCache> regionCache;
  RCRef<Cache::TransformCache> transformCache;
};
} // namespace M::KGEN

#endif // KGEN_COMPILER_H
