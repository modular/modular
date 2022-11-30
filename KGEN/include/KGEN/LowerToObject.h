//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LOWERTOOBJECT_H
#define KGEN_LOWERTOOBJECT_H

#include "Cache/BlobCache.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringSet.h"
#include <filesystem>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace M::KGEN {
/// Provide a way to hash an mlir::Module in order to map from the mlir::Module
/// to the corresponding llvm::Module. These modules are usually composites, but
/// don't have to be - this key simply provides a 1:1 map from mlir::Module to
/// llvm::Module.
struct LLVMCacheKeyInfo {
  using KeyTy = ModuleOp;
  static std::string hashKey(ModuleOp key);
};

/// Stores llvm::Module objects as bitcode, indexed by an MLIR module.
using LLVMCache = M::Cache::BlobCache<LLVMCacheKeyInfo>;

/// Provides a way to hash a composite mlir::Module in order to map from that
/// module to a compiled object for that module. This allows us to avoid both
/// the LLVM lowering *and* object file emission.
struct CompositeObjectCacheKeyInfo {
  using KeyTy = ModuleOp;
  static std::string hashKey(ModuleOp key);
};

/// Provides a mapping from a composite mlir::Module to the bytes of the object
/// file for that module. A composite module is created when we produce a
/// standalone object - we take several symbols and merge them together, then
/// produce a single object for that merged (composite) module.
using CompositeObjectCache = M::Cache::BlobCache<CompositeObjectCacheKeyInfo>;

/// Provides a basic way to interact with the set of caches needed
/// lowering/raising to/from LLVM and objects.
///
/// Cache Responsibilities:
///   LLVMCache - Stores a mapping from a KGEN module made up of any number of
///     symbols to the llvm::Module produced by lowering that mlir::Module to
///     LLVM.
///   CompositeObjectCache - Stores a mapping from a KGEN module made up of the
///     modules from multiple symbols to the object file produced by compiling
///     that composite.
class LoweringCacheCollection {
public:
  explicit LoweringCacheCollection(LLCL::Runtime &runtime, StringRef basePath)
      : llvm(Cache::getDefaultBackendChain(
                 runtime,
                 (std::filesystem::path(basePath.str()) / "llvm").string())
                 .takeValue()),
        composite(
            Cache::getDefaultBackendChain(
                runtime,
                (std::filesystem::path(basePath.str()) / "composite").string())
                .takeValue()) {}

  LLVMCache &getLLVM() { return llvm; }
  CompositeObjectCache &getComposite() { return composite; }

private:
  LLVMCache llvm;
  CompositeObjectCache composite;
};

/// The purpose of this class is to provide methods to lower concrete KGEN
/// functions to LLVM, and then to objects.
class ObjectCompiler {
public:
  ObjectCompiler(LLCL::Runtime &runtime, StringRef basePath, ModuleOp module,
                 const CompilationOptions &options)
      : caches(runtime, basePath), module(module), symtab(module),
        options(options) {
    for (auto e : module.getOps<ExportOp>())
      for (auto sym : e.getExports().getAsRange<FlatSymbolRefAttr>())
        exportedSymbols.insert(sym.getAttr());
  }

  /// Construct an ObjectCompiler with a specific set of exports.
  ObjectCompiler(LLCL::Runtime &runtime, StringRef basePath, ModuleOp module,
                 DenseSet<StringAttr> exports,
                 const CompilationOptions &options)
      : caches(runtime, basePath), module(module), symtab(module),
        exportedSymbols(std::move(exports)), options(options) {}

  /// Lower all exported `kgen.func` to llvm and populate the composite module
  /// in the cache. Returns the LLVM module on success, and nullptr on failure.
  std::unique_ptr<llvm::Module> lowerAllFuncsToLLVM(llvm::LLVMContext &ctx);

  /// Slices the call graph for all exported symbols to produce a standalone
  /// object.
  FailureOr<Cache::BufferRef> produceStandaloneObject(TargetInfoAttr target,
                                                      bool isJIT);

  /// Writes function declarations for all exported symbols.
  LogicalResult produceFunctionDecls(raw_ostream &os);

  /// Get access to the symbol table the compiler holds.
  mlir::SymbolTable &getSymbolTable() { return symtab; }

  /// Get access to the module held by the compiler.
  ModuleOp getModule() { return module; }

  /// Get access to the caches the compiler holds.
  LoweringCacheCollection &getCaches() { return caches; }

  /// Returns true if the symbol is exported with a `kgen.export` op in this
  /// module. This is the equivalent of a context-sensitive "public".
  bool isSymbolExported(StringAttr symbol) {
    return exportedSymbols.contains(symbol);
  }

private:
  /// Produce a standalone MLIR module by slicing out the dependencies of the
  /// provided kgen.export op.
  OwningOpRef<ModuleOp> produceStandaloneModule();

  /// Lower a KGEN module to an LLVMIR module.
  std::unique_ptr<llvm::Module> lowerKGENToLLVM(ModuleOp module,
                                                llvm::LLVMContext &ctx);

  /// The caches needed for lowering/raising.
  LoweringCacheCollection caches;

  /// This is the module the compiler was created with.
  ModuleOp module;

  /// This is a symbol table we maintain for easy lookups.
  SymbolTable symtab;

  /// This is a list of exported symbol names so we don't constantly recompute
  /// it.
  DenseSet<StringAttr> exportedSymbols;

  /// The compilation options to use.
  CompilationOptions options;
};
} // namespace M::KGEN

#endif // KGEN_LOWERTOOBJECT_H
