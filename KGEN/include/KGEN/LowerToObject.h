//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LOWERTOOBJECT_H
#define KGEN_LOWERTOOBJECT_H

#include "Cache/BlobCache.h"
#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include <filesystem>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
} // namespace llvm

namespace M::KGEN {
/// The purpose of this class is to provide methods to lower concrete KGEN
/// functions to LLVM, and then to objects.
class ObjectCompiler {
public:
  /// Construct an ObjectCompiler that infers the exports from the module.
  static ErrorOr<ObjectCompiler> create(LLCL::Runtime &runtime,
                                        mlir::PassManager &mgr,
                                        StringRef basePath,
                                        CompilationOptions options);

  /// Lower all exported `kgen.func` to llvm. Returns the LLVM module on
  /// success, and nullptr on failure.
  std::unique_ptr<llvm::Module>
  lowerAllFuncsToLLVM(SymbolTable &symtab, const ExportMap &exportedSymbols,
                      llvm::LLVMContext &ctx);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// archive.
  ErrorOr<Cache::BufferRef>
  produceStandaloneArchive(SymbolTable &symtab,
                           const ExportMap &exportedSymbols, bool isJIT);

  /// Produces a standalone archive as an ElementsAttr that can be used as an
  /// attribute on another operation. Using this function generally implies
  /// `isJIT`, which is why it defaults to `true`. Clients should prefer this
  /// method if they intend to store the compiled object in another graph.
  ErrorOr<ElementsAttr>
  produceStandaloneArchiveAttr(SymbolTable &symtab,
                               const ExportMap &exportedSymbols,
                               TargetInfoAttr target, bool isJIT = true);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// assembly file. The assembly output is written to the provided stream.
  ErrorOrSuccess produceStandaloneAssembly(SymbolTable &symtab,
                                           const ExportMap &exportedSymbols,
                                           TargetInfoAttr target,
                                           llvm::raw_pwrite_stream &os);

  /// Writes function declarations for all exported symbols.
  LogicalResult produceFunctionDecls(SymbolTable &symtab,
                                     const ExportMap &exportedSymbols,
                                     raw_ostream &os);

private:
  /// Construct an ObjectCompiler with a specific set of exports.
  ObjectCompiler(LLCL::Runtime &runtime, mlir::PassManager &mgr,
                 LLCL::RCRef<Cache::BlobCacheBackend> transformCache,
                 CompilationOptions options);

  /// Produce a standalone MLIR module by slicing out the dependencies of the
  /// provided kgen.export ops.
  OwningOpRef<ModuleOp>
  produceStandaloneModule(SymbolTable &symtab,
                          const ExportMap &exportedSymbols);

  /// Lower the given module to LLVM. Returns the LLVM module on success, and
  /// nullptr on failure.
  std::unique_ptr<llvm::Module>
  lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module, bool isJIT);

  /// Lower the given LLVM module to an object file.
  LLCL::AnyAsyncValueRef lowerLLVMModuleToObject(llvm::Module &module,
                                                 Location loc, bool isJIT);

  /// The caches needed for compilation.
  LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache;

  /// The async runtime to use during lowering.
  LLCL::Runtime &runtime;

  /// The configured MLIR pass manager to use.
  mlir::PassManager &mgr;

  /// The compilation options to use.
  CompilationOptions options;
};

/// Get the target info for the specified target.
ErrorOr<TargetInfoAttr> getTargetInfoFor(MLIRContext *ctx,
                                         StringRef targetTriple, StringRef cpu,
                                         StringRef features);
/// Setup the machine properties from the provided target.
ErrorOr<std::unique_ptr<llvm::TargetMachine>>
createTargetMachine(const CompilationOptions &options, bool isJIT);
} // namespace M::KGEN

#endif // KGEN_LOWERTOOBJECT_H
