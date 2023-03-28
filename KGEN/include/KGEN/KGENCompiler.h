//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_H
#define KGEN_COMPILER_H

#include "Cache/CacheDialect/CachedTransform.h"
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
/// This populates the pre-elaboration phase passes of the KGEN compiler. The
/// distribution format of a KGEN library is essentially what comes just before
/// elaboration because the parameter system allows significant extension.
void populateGenerateLibraryFilePasses(mlir::PassManager &pm);

/// This populates the passes to produce a fully concrete KGEN module. That
/// means it runs pre-elaboration, elaboration, and then the post-elaboration
/// cleanup passes. Its purpose is to populate the passes used to produce the
/// format that we will end up using to produce an object file.
void populateElaborateModulePasses(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    const ElaborateGeneratorsOptions &elaborateOptions);

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
                    TargetInfoAttr target,
                    ElaborateGeneratorsOptions elaborateOptions,
                    ObjectCompilerLayer &base,
                    LLCL::RCRef<Cache::BlobCacheBackend> transformCacheBackend,
                    LLCL::RCRef<Cache::BlobCacheBackend> regionCacheBackend,
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
  ElaborateGeneratorsOptions elaborateOptions;
  ObjectCompilerLayer &baseLayer;

  LLCL::RCRef<Cache::RegionCache> regionCache;
  LLCL::RCRef<Cache::TransformCache> transformCache;
};
} // namespace M::KGEN

#endif // KGEN_COMPILER_H
