//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LOWERTOOBJECT_H
#define KGEN_LOWERTOOBJECT_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/BlobCache.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringSet.h"
#include <filesystem>
#include <string>

namespace llvm {
class Module;
}

namespace M::KGEN {
/// Provides a cache key for mapping from a `kgen.precompiled.llvm` back up to a
/// `kgen.func`.
struct RaisingCacheKeyInfo {
  using KeyTy = PrecompiledLLVMOp;
  static std::string hashKey(PrecompiledLLVMOp key);
};

using RaisingCache = M::BlobCache<RaisingCacheKeyInfo>;

/// Provide a way to hash `kgen.precompiled.func` and `kgen.precompiled.llvm`
/// ops to index into the LLVM cache. The key can be any of
/// `kgen.precompiled.func`, `kgen.precompiled.llvm` or an LLVM module. We have
/// such an...interesting key type because we have many different things we'd
/// like to store in the same cache - we'd like the mapping from
/// `kgen.precompiled.func` -> `kgen.precompiled.llvm` and a mapping from
/// `kgen.precompiled.llvm` or `llvm::Module` the symbol's serialized LLVM
/// bytecode.
struct LLVMCacheKeyInfo {
  using KeyTy = std::variant<FuncOp, PrecompiledLLVMOp, llvm::Module *>;
  static std::string hashKey(KeyTy key);
};

/// Provides 2 things: a mapping from `kgen.precompiled.func` to
/// `kgen.precompiled.llvm`, and storage for llvm::Module objects that we have
/// already stored.
using LLVMCache = M::BlobCache<LLVMCacheKeyInfo>;

/// Provides a way to hash a composite llvm::Module. This will be used to map a
/// compiled object for that composite. This allows us to avoid recompiling
/// composite modules whenever possible.
struct CompositeObjectCacheKeyInfo {
  using KeyTy = llvm::Module *;
  static std::string hashKey(llvm::Module *key);
};

/// Provides a mapping from composite llvm::Module to the bytes of the object
/// file for that composite. A composite module is created when we produce a
/// standalone object - we take the modules for each of the symbols and merge
/// them together, then produce a single object for that merged (composite)
/// module.
using CompositeObjectCache = M::BlobCache<CompositeObjectCacheKeyInfo>;

/// Provides a basic way to interact with the set of caches needed
/// lowering/raising to/from LLVM and objects.
///
/// Cache Responsibilities:
///   RaisingCache - Stores a mapping from a lower-level op (i.e. object) to a
///     higher-level op (i.e. llvm). This allows us to raise the IR from a
///     lowered representation to do things like walking the original function,
///     or LTO-style optimizations.
///   LLVMCache - Stores several mappings:
///     - `kgen.func` -> `kgen.precompiled.llvm`: This is so that we don't have
///       to re-emit LLVM.
///     - `kgen.precompiled.llvm` -> llvm::Module: This is because
///       `kgen.precompiled.llvm` is largely intended as a cache reference.
///     - llvm::Module -> llvm::Module: This is the main purpose of the cache -
///       to store llvm::Modules as bitcode.
///   CompositeObjectCache - Stores a mapping from a LLVM module made up of the
///     modules from multiple symbols to the object file produced by compiling
///     that composite.
class LoweringCacheCollection {
public:
  explicit LoweringCacheCollection(StringRef basePath)
      : raising(getDefaultBackendChain(
            (std::filesystem::path(basePath.str()) / "raising").string())),
        llvm(getDefaultBackendChain(
            (std::filesystem::path(basePath.str()) / "llvm").string())),
        composite(getDefaultBackendChain(
            (std::filesystem::path(basePath.str()) / "composite").string())) {}

  RaisingCache &getRaising() { return raising; }
  LLVMCache &getLLVM() { return llvm; }
  CompositeObjectCache &getComposite() { return composite; }

private:
  RaisingCache raising;
  LLVMCache llvm;
  CompositeObjectCache composite;
};

/// The purpose of this class is to provide methods to lower concrete KGEN
/// functions to LLVM, and then to objects. It also provides methods that allow
/// the user to raise from an object, to LLVM, and even back to the original
/// function.
class ObjectCompiler {
public:
  ObjectCompiler(StringRef basePath, ModuleOp module)
      : caches(basePath), module(module), symtab(module) {
    for (auto e : module.getOps<ExportOp>())
      for (auto sym : e.getExports().getAsRange<FlatSymbolRefAttr>())
        exportedSymbols.insert(sym.getAttr());
  }

  /// Given a FuncOp, lower it to LLVM and turn it into an LLVM module.
  /// At this point, the target must be provided. This function will replace
  /// `func` in the IR with a `kgen.precompiled.llvm`, and return the op for
  /// convenience. It will cache the original function in bytecode format
  /// inside `funcCache`, and store the LLVM module in LLVMCache.
  FailureOr<PrecompiledLLVMOp> lowerToLLVM(FuncOp func, TargetInfoAttr target);

  /// Lower all `kgen.func` to llvm and populate them in the cache. This
  /// modifies the compiler-held module in-place.
  LogicalResult lowerAllFuncsToLLVM(TargetInfoAttr target);

  /// Backtrack up the compilation stack - given a `kgen.precompiled.llvm`,
  /// replace it with the `kgen.func` it came from if possible.
  FailureOr<FuncOp> raiseFromLLVM(PrecompiledLLVMOp precompiled);

  /// Slices the call graph for `which` to produce a standalone object. If
  /// slicing the call graph is not possible, it simply returns the object
  /// already in the cache.
  FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
  produceStandaloneObject(ArrayRef<StringRef> symbols, bool isJIT);

  /// Collects all of the `kgen.precompiled.llvm` in the module and slices the
  /// call graph for them to produce a single standalone object. If slicing the
  /// call graph is not possible, it simply returns the object already in the
  /// cache.
  FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
  produceStandaloneObject(bool isJIT);

  /// Get access to the symbol table the compiler holds.
  mlir::SymbolTable &getSymbolTable() { return symtab; }

  /// Get access to the caches the compiler holds.
  LoweringCacheCollection &getCaches() { return caches; }

  /// Returns true if the symbol is exported with a `kgen.export` op in this
  /// module. This is the equivalent of a context-sensitive "public".
  bool isSymbolExported(StringAttr symbol) {
    return exportedSymbols.contains(symbol);
  }

private:
  /// The caches needed for lowering/raising.
  LoweringCacheCollection caches;

  /// This is the module the compiler was created with.
  ModuleOp module;

  /// This is a symbol table we maintain for easy lookups.
  SymbolTable symtab;

  /// This is a list of exported symbol names so we don't constantly recompute
  /// it.
  DenseSet<StringAttr> exportedSymbols;
};
} // namespace M::KGEN

#endif // KGEN_LOWERTOOBJECT_H
