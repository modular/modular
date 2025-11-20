//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HashUtils.h"

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "Support/Compiler/Threading.h"

#include "xxh3.h"
#include "xxhash.h"

using namespace mlir;

M::raw_xxhash_stream::raw_xxhash_stream() : raw_ostream() {
  XXH3_128bits_reset(&State);
}

std::array<uint8_t, 16> M::raw_xxhash_stream::hash() {
  flush();

  XXH128_hash_t digest = XXH3_128bits_digest(&State);
  std::array<uint8_t, 16> result;
  memcpy(&result[0], (void *)&digest.low64, 8);
  memcpy(&result[8], (void *)&digest.high64, 8);
  return result;
}

std::string M::raw_xxhash_stream::hashString() {
  SmallString<32> output;
  llvm::toHex(hash(), /*LowerCase=*/true, output);
  return std::string(output);
}

static LogicalResult writeBytecode(Operation *op, llvm::raw_ostream &os,
                                   mlir::AttrTypeReplacer &replacer) {
  OwningOpRef<Operation *> cloned = op->clone();
  replacer.recursivelyReplaceElementsIn(*cloned,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);
  return mlir::writeBytecodeToFile(*cloned, os);
}

M::BytecodeHasher::BytecodeHasher() {
  // Add replacement which strips location information.
  replacer.addReplacement(
      [](LocationAttr loc) { return UnknownLoc::get(loc.getContext()); });
}

FailureOr<std::string> M::BytecodeHasher::getBytecodeHash(Operation *op) {
  raw_xxhash_stream ostream;
  if (failed(writeBytecode(op, ostream, replacer)))
    return op->emitError("Failed to write bytecode");
  return ostream.hashString();
}

struct ModuleHashCache {
  ModuleHashCache() = default;

  LogicalResult computeHash(Operation *op) {
    auto result = hasher.getBytecodeHash(op);
    if (failed(result))
      return failure();

    hashes.push_back(std::move(*result));
    return success();
  }

  void join(ModuleHashCache &other) {
    llvm::move(other.hashes, std::back_inserter(hashes));
    other = ModuleHashCache();
  }

  M::BytecodeHasher hasher;
  llvm::SmallVector<std::string> hashes;
};

FailureOr<std::string> M::getModuleBytecodeHash(mlir::ModuleOp module) {
  auto context = module.getContext();
  auto ops =
      llvm::map_to_vector(module.getOps(), [](Operation &op) { return &op; });

  auto workFunc = [](ModuleHashCache &cache, Operation *op) -> LogicalResult {
    return cache.computeHash(op);
  };

  auto consolidateFn = [](ModuleHashCache &original,
                          MutableArrayRef<ModuleHashCache> caches) {
    for (ModuleHashCache &cache : caches)
      original.join(cache);
  };

  ModuleHashCache resultCache;
  auto result = failableParallelForEach(
      /*ctx=*/context,
      /*range=*/ops,
      /*func=*/workFunc,
      /*cache=*/resultCache,
      /*consolidate=*/consolidateFn);

  if (failed(result))
    return failure();

  // Sort results to ensure determinism. The multi-threaded processing of
  // failableParallelForEach does not ensure an particular ordering w.r.t. what
  // input elements are associated to which cache.
  llvm::sort(resultCache.hashes);

  raw_xxhash_stream hasher;
  llvm::interleave(resultCache.hashes, hasher, "");
  return hasher.hashString();
}
