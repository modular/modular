//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_OBJECTCOMPILER_H
#define KGEN_COMPILER_OBJECTCOMPILER_H

#include "Cache/BlobCache.h"
#include "Cache/CachedTransform.h"
#include "KGEN/Compiler/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "mlir/IR/BuiltinOps.h"
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
  static ErrorOr<ObjectCompiler> create(mlir::PassManager &mgr,
                                        StringRef basePath,
                                        CompilationOptions options, bool isJIT,
                                        bool isSearch = false);

  /// Lower all exported `kgen.func` to llvm. Returns the LLVM module on
  /// success, and nullptr on failure.
  std::unique_ptr<llvm::Module>
  lowerAllFuncsToLLVM(const SymbolTable &symtab,
                      const ExportMap &exportedSymbols, llvm::LLVMContext &ctx);

  /// Lower the given module to LLVM. Returns the LLVM module on success, and
  /// nullptr on failure.
  std::unique_ptr<llvm::Module> lowerAllFuncsToLLVM(llvm::LLVMContext &ctx,
                                                    ModuleOp module);

  /// Produce a standalone MLIR module by slicing out the dependencies of the
  /// provided exported ops. An `IRMapping` can be provided to be able to map
  /// into the sliced module.
  OwningOpRef<ModuleOp>
  produceStandaloneModule(const SymbolTable &symtab,
                          const ExportMap &exportedSymbols, IRMapping &mapping);

  /// Slices the call graph for all exported symbols to produce a "standalone"
  /// archive, meaning all external libraries the archive depends upon are
  /// pulled into the archive itself.
  ErrorOr<BufferRef> produceStandaloneArchive(const SymbolTable &symtab,
                                              const ExportMap &exportedSymbols);

  /// Produces a standalone archive as an ElementsAttr that can be used as an
  /// attribute on another operation. Using this function generally implies
  /// `isJIT`, which is why it defaults to `true`. Clients should prefer this
  /// method if they intend to store the compiled object in another graph.
  ErrorOr<ElementsAttr>
  produceStandaloneArchiveAttr(const SymbolTable &symtab,
                               const ExportMap &exportedSymbols,
                               TargetInfoAttr target);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// assembly file. The assembly output is written to the provided stream.
  ErrorOrSuccess produceStandaloneAssembly(const SymbolTable &symtab,
                                           const ExportMap &exportedSymbols,
                                           llvm::raw_pwrite_stream &os);

  /// Writes function declarations for all exported symbols.
  LogicalResult produceFunctionDecls(const SymbolTable &symtab,
                                     const ExportMap &exportedSymbols,
                                     StringRef filename, raw_ostream &os);

  /// Get a reference to the object compiler's transform cache.
  RCRef<Cache::TransformCache> getTransformCache() {
    return transformCache.copy();
  }

  /// Configure the object compiler to be used for search.
  void setForSearch(bool useForSearch) { isSearch = useForSearch; }

  /// Get whether compilation is for JIT.
  bool getIsJIT() const { return isJIT; }

private:
  /// Construct an ObjectCompiler with a specific set of exports.
  ObjectCompiler(mlir::PassManager &mgr,
                 RCRef<Cache::BlobCacheBackend> transformCache,
                 CompilationOptions options, bool isJIT, bool isSearch);

  /// Lower the given LLVM module to an object file (parLLC = false) or
  /// multiple object files per function (parLLC = true).
  SmallVector<LLCL::AnyAsyncValueRef>
  lowerLLVMModuleToObjects(llvm::Module &module, Location loc,
                           MLIRContext *mlirContext, bool parLLC,
                           std::optional<size_t> moduleIdx = std::nullopt);

  /// Slices the call graph for all exported symbols to produce an archive.
  /// The `standalone` argument is false by default, but if set to true, then
  /// dependent libraries are pulled into the archive itself.
  ErrorOr<BufferRef> produceArchive(const SymbolTable &symtab,
                                    const ExportMap &exportedSymbols,
                                    bool standalone = false);

  OwningOpRef<ModuleOp>
  produceStandaloneModule(const SymbolTable &symtab,
                          const ExportMap &exportedSymbols);

  /// The caches needed for compilation.
  RCRef<Cache::TransformCache> transformCache;

  /// The configured MLIR pass manager to use.
  mlir::PassManager *mgr;

  /// The compilation options to use.
  CompilationOptions options;

  /// This is a bit odd, but since we use this layer to generate code for cases
  /// where we aren't going to immediately execute it, we need to be able to
  /// change the codegen mode.
  bool isJIT;

  /// When the elaborator performs search, the IR coming reaching the
  /// ObjectCompilerLayer is post-elaboration IR.
  bool isSearch;

  friend class ObjectCompilerLayer;
};

/// Setup the machine properties from the provided target.
ErrorOr<std::unique_ptr<llvm::TargetMachine>>
createTargetMachine(const CompilationOptions &options, bool isJIT);

/// Run the llvm opt passes over `module` given `targetMachine`.
LogicalResult runLLVMOptPasses(llvm::Module &module,
                               llvm::TargetMachine &targetMachine,
                               const CompilationOptions &options,
                               LLCL::Runtime &runtime);

//===----------------------------------------------------------------------===//
// ObjectCompilerLayer
//===----------------------------------------------------------------------===//

/// Provide an ExecutionEngine layer for the ObjectCompiler. This simply wraps
/// up the ObjectCompiler in the correct APIs - under the hood it also defines a
/// MaterializationUnit and uses that to emit symbols on-demand.
class ObjectCompilerLayer : public MaterializationLayer {
public:
  ObjectCompilerLayer(ObjectCompiler &&objCompiler,
                      llvm::orc::ObjectLayer &base,
                      llvm::orc::ExecutionSession &sess,
                      const llvm::DataLayout &dl, AddToSearchOrderFn add);

  /// Add a module to the JIT. The module must stay alive long enough for
  /// codegen to happen (so long enough for a `lookup` call), but after that the
  /// object code generated is fully stable, save for the `lookupArchive`
  /// operation below.
  ///
  /// If the export map is empty, then exports are regenerated
  /// by inspecting the module. Otherwise, the exports provided are used.
  ErrorOrSuccess add(StringRef libName, const SymbolTable &symtab,
                     ExportMap &exports);

  /// Emit a given module. This will immediately run the materialization.
  void emit(std::unique_ptr<llvm::orc::MaterializationResponsibility> mr,
            const SymbolTable &symtab, const ExportMap &exports);

  /// Provide access to the underlying ObjectCompiler so that users can call its
  /// methods directly if desired (for example, to emit asm or LLVM).
  ObjectCompiler &getRawCompiler() { return objectCompiler; }

  static bool classof(const MaterializationLayer *layer) {
    return layer->getKind() == LayerKind::kObjectCompilerLayer;
  }

private:
  /// Emit a given module. This will immediately run the materialization.
  /// Returns errors rather than setting them on the materialization layer.
  ErrorOrSuccess emitImpl(llvm::orc::MaterializationResponsibility &mr,
                          const SymbolTable &symtab, const ExportMap &exports);

  /// Conform to the ORC's interface and return a map of the exported symbols.
  /// If the export map is empty, uses `getExportedSymbols` to infer them from
  /// the module.
  llvm::orc::MaterializationUnit::Interface
  getInterface(const SymbolTable &symtab, const ExportMap &exports);

  /// Provide an ObjectCompilerMaterializationUnit so that we can do codegen
  /// on-demand.
  class ObjectCompilerMaterializationUnit;

private:
  ObjectCompiler objectCompiler;
  llvm::orc::ObjectLayer &baseLayer;
};

//===----------------------------------------------------------------------===//
// compileLLVMToObject
//===----------------------------------------------------------------------===//
/// Compile the given LLVM module to an object file and write it to objStream.
LogicalResult
compileLLVMToObject(llvm::Module &module, llvm::TargetMachine &targetMachine,
                    llvm::raw_pwrite_stream &objStream,
                    CompilationOptions &options, LLCL::Runtime &runtime,
                    bool emitAssembly = false,
                    std::optional<size_t> moduleIdx = std::nullopt);
} // namespace M::KGEN

#endif // KGEN_COMPILER_OBJECTCOMPILER_H
