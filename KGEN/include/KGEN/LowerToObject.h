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
#include "mlir/IR/BuiltinOps.h"
#include <filesystem>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace M::KGEN {
/// The purpose of this class is to provide methods to lower concrete KGEN
/// functions to LLVM, and then to objects.
class ObjectCompiler {
public:
  /// Construct an ObjectCompiler that infers the exports from the module.
  static ErrorOr<ObjectCompiler> create(LLCL::Runtime &runtime,
                                        StringRef basePath, SymbolTable &symtab,
                                        const CompilationOptions &options);

  /// Construct an ObjectCompiler with a specific set of exports.
  static ErrorOr<ObjectCompiler> create(LLCL::Runtime &runtime,
                                        StringRef basePath, SymbolTable &symtab,
                                        DenseSet<StringAttr> exports,
                                        const CompilationOptions &options);

  /// Lower all exported `kgen.func` to llvm. Returns the LLVM module on
  /// success, and nullptr on failure.
  std::unique_ptr<llvm::Module> lowerAllFuncsToLLVM(llvm::LLVMContext &ctx);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// object.
  ErrorOr<Cache::BufferRef> produceStandaloneObject(TargetInfoAttr target,
                                                    bool isJIT);

  /// Produces a standalone object as an ElementsAttr that can be used as an
  /// attribute on another operation. Using this function generally implies
  /// `isJIT`, which is why it defaults to `true`. Clients should prefer this
  /// method if they intend to store the compiled object in another graph.
  ErrorOr<ElementsAttr> produceStandaloneObjectAttr(TargetInfoAttr target,
                                                    bool isJIT = true);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// assembly file. The assembly output is written to the provided stream.
  ErrorOrSuccess produceStandaloneAssembly(TargetInfoAttr target,
                                           llvm::raw_pwrite_stream &os);

  /// Writes function declarations for all exported symbols.
  LogicalResult produceFunctionDecls(raw_ostream &os);

  /// Get access to the module held by the compiler.
  ModuleOp getModule() { return module; }

  /// Returns true if the symbol is exported with a `kgen.export` op in this
  /// module. This is the equivalent of a context-sensitive "public".
  bool isSymbolExported(StringAttr symbol) {
    return exportedSymbols.contains(symbol);
  }

private:
  /// Construct an ObjectCompiler with a specific set of exports.
  ObjectCompiler(LLCL::Runtime &runtime, SymbolTable &symtab,
                 DenseSet<StringAttr> exports,
                 LLCL::RCRef<Cache::BlobCacheBackend> transformCache,
                 const CompilationOptions &options);

  /// Produce a standalone MLIR module by slicing out the dependencies of the
  /// provided kgen.export op.
  OwningOpRef<ModuleOp> produceStandaloneModule();

  /// Lower the given module to LLVM. Returns the LLVM module on success, and
  /// nullptr on failure.
  std::unique_ptr<llvm::Module> lowerAllFuncsToLLVM(llvm::LLVMContext &ctx,
                                                    ModuleOp module);

  /// The caches needed for compilation.
  LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache;

  /// The async runtime to use during lowering.
  LLCL::Runtime &runtime;

  /// This is the module the compiler was created with.
  ModuleOp module;

  /// This is a symbol table we maintain for easy lookups.
  SymbolTable &symtab;

  /// This is a list of exported symbol names so we don't constantly recompute
  /// it.
  DenseSet<StringAttr> exportedSymbols;

  /// The compilation options to use.
  CompilationOptions options;
};
} // namespace M::KGEN

#endif // KGEN_LOWERTOOBJECT_H
