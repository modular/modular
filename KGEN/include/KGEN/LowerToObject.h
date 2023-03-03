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
                                        StringRef basePath, SymbolTable &symtab,
                                        const CompilationOptions &options);

  /// Construct an ObjectCompiler with a specific set of exports.
  static ErrorOr<ObjectCompiler>
  create(LLCL::Runtime &runtime, mlir::PassManager &mgr, StringRef basePath,
         SymbolTable &symtab, const DenseMap<StringAttr, StringAttr> &exports,
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

private:
  /// Construct an ObjectCompiler with a specific set of exports.
  ObjectCompiler(LLCL::Runtime &runtime, mlir::PassManager &mgr,
                 SymbolTable &symtab,
                 const DenseMap<StringAttr, StringAttr> &exports,
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

  /// The configured MLIR pass manager to use.
  mlir::PassManager &mgr;

  /// This is the module the compiler was created with.
  ModuleOp module;

  /// This is a symbol table we maintain for easy lookups.
  SymbolTable &symtab;

  /// This is a list of exported symbol names and respective aliases so we
  /// don't constantly recompute it.
  DenseMap<StringAttr, StringAttr> exportedSymbols;

  /// The compilation options to use.
  CompilationOptions options;
};

/// Get the target info for the specified target.
ErrorOr<TargetInfoAttr> getTargetInfoFor(MLIRContext *ctx,
                                         StringRef targetTriple, StringRef cpu,
                                         StringRef features);
/// Setup the machine properties from the provided target.
ErrorOr<std::unique_ptr<llvm::TargetMachine>>
createTargetMachine(TargetInfoAttr targetInfo,
                    const CompilationOptions &options, bool isJIT);
} // namespace M::KGEN

#endif // KGEN_LOWERTOOBJECT_H
