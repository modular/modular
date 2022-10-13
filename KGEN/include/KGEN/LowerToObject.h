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
#include <filesystem>
#include <string>

namespace llvm {
class Module;
}

namespace M::KGEN {
/// Provides a cache key for mapping from a `kgen.precompiled.llvm` back up to a
/// `kgen.func`, and from a `kgen.precompiled.object` back up to a
/// `kgen.precompiled.llvm`.
struct RaisingCacheKeyInfo {
  using KeyTy = std::variant<PrecompiledLLVMOp, PrecompiledObjectOp>;
  static std::string hashKey(KeyTy key);
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

/// Provide a way to hash `kgen.precompiled.llvm` and `kgen.precompiled.object`
/// ops to index into the object cache. If the key is a `kgen.precompiled.llvm`,
/// then the cache key is the hash of the operation and its attributes. If the
/// key is a `kgen.precompiled.object` then the key is the string stored in the
/// op's attributes.
struct ObjectCacheKeyInfo {
  using KeyTy = std::variant<PrecompiledLLVMOp, llvm::MemoryBufferRef,
                             PrecompiledObjectOp>;
  static std::string hashKey(KeyTy key);
};

/// Provides a mapping from `kgen.precompiled.llvm` to a
/// `kgen.precompiled.object`. These objects are cached on a per-op basis.
using ObjectCache = M::BlobCache<ObjectCacheKeyInfo>;

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
///   ObjectCache - Stores several mappings:
///     - `kgen.precompiled.llvm` -> `kgen.precompiled.object`: This is so that
///       we don't have to re-lower to an object.
///     - `kgen.precompiled.object` -> MemoryBuffer: This is because
///       `kgen.precompiled.object` is largely intended as a cache reference.
///     - MemoryBuffer -> MemoryBuffer: This is the main purpose of the cache,
///       to store the actual objects serialized into memory buffers.
class LoweringCacheCollection {
public:
  explicit LoweringCacheCollection(StringRef basePath)
      : raising(getDefaultBackendChain(
            (std::filesystem::path(basePath.str()) / "raising").string())),
        llvm(getDefaultBackendChain(
            (std::filesystem::path(basePath.str()) / "llvm").string())),
        obj(getDefaultBackendChain(
            (std::filesystem::path(basePath.str()) / "obj").string())) {}

  RaisingCache &getRaising() { return raising; }
  LLVMCache &getLLVM() { return llvm; }
  ObjectCache &getObject() { return obj; }

private:
  RaisingCache raising;
  LLVMCache llvm;
  ObjectCache obj;
};

/// The purpose of this class is to provide methods to lower concrete KGEN
/// functions to LLVM, and then to objects. It also provides methods that allow
/// the user to raise from an object, to LLVM, and even back to the original
/// function.
class ObjectCompiler {
public:
  ObjectCompiler(StringRef basePath, ModuleOp module)
      : caches(basePath), module(module), symtab(module) {}

  /// Given a FuncOp, lower it to LLVM and turn it into an LLVM module.
  /// At this point, the target must be provided. This function will replace
  /// `func` in the IR with a `kgen.precompiled.llvm`, and return the op for
  /// convenience. It will cache the original function in bytecode format
  /// inside `funcCache`, and store the LLVM module in LLVMCache.
  FailureOr<PrecompiledLLVMOp> lowerToLLVM(FuncOp func, TargetInfoAttr target);

  /// Backtrack up the compilation stack - given a `kgen.precompiled.llvm`,
  /// replace it with the `kgen.func` it came from if possible.
  FailureOr<FuncOp> raiseFromLLVM(PrecompiledLLVMOp precompiled);

  /// Get the body of the `kgen.precompiled.llvm`, emit an object, and replace
  /// the `kgen.precompiled.llvm` with a `kgen.precompiled.object` with the same
  /// name.
  FailureOr<PrecompiledObjectOp> lowerToObject(PrecompiledLLVMOp func,
                                               bool isJIT);

  /// Backtrack up the compilation stack - given a `kgen.precompiled.object`,
  /// replace it with the `kgen.precompiled.llvm` it came from if possible.
  FailureOr<PrecompiledLLVMOp> raiseFromObject(PrecompiledObjectOp precompiled);

  /// Lower all `kgen.func` to objects and populate them in the cache. This
  /// modifies the compiler-held module in-place.
  LogicalResult lowerAllFuncsToObject(TargetInfoAttr target, bool isJIT);

  /// Slices the call graph for `which` to produce a standalone object. If
  /// slicing the call graph is not possible, it simply returns the object
  /// already in the cache. This function will also raise any
  /// `kgen.precompiled.object` to `kgen.func` for which the raising exists.
  FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
  produceStandaloneObject(ArrayRef<StringRef> symbols, bool isJIT);

  /// Collects all the `kgen.precompiled.object` in the module and slices the
  /// call graph for them to produce a single standalone object. If slicing the
  /// call graph is not possible, it simply returns the object already in the
  /// cache. This function will also raise any `kgen.precompiled.object` to
  /// `kgen.func` for which the raising exists.
  FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
  produceStandaloneObject(bool isJIT);

  /// Get access to the symbol table the compiler holds.
  mlir::SymbolTable &getSymbolTable() { return symtab; }

  /// Get access to the caches the compiler holds.
  LoweringCacheCollection &getCaches() { return caches; }

private:
  /// The caches needed for lowering/raising.
  LoweringCacheCollection caches;

  /// This is the module the compiler was created with.
  ModuleOp module;

  /// This is a symbol table we maintain for easy lookups.
  SymbolTable symtab;
};
} // namespace M::KGEN

#endif // KGEN_LOWERTOOBJECT_H
