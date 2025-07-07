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

FailureOr<std::string> M::getBytecodeHash(Operation *op,
                                          ReplacementFunc replace) {
  raw_xxhash_stream ostream;
  if (failed(M::writeBytecode(op, ostream, replace)))
    return op->emitError("Failed to write bytecode");
  return ostream.hashString();
}

LogicalResult M::writeBytecode(Operation *op, llvm::raw_ostream &os,
                               ReplacementFunc replacer) {
  auto unknownLoc = UnknownLoc::get(op->getContext());

  auto *builtin = op->getContext()->getLoadedDialect<mlir::BuiltinDialect>();
  BytecodeDialectInterface *iface =
      builtin->getRegisteredInterface<BytecodeDialectInterface>();

  BytecodeWriterConfig config;
  config.attachAttributeCallback(
      [&](Attribute entryValue, std::optional<StringRef> &dialectGroupName,
          DialectBytecodeWriter &writer) -> LogicalResult {
        // Map all locations attributes to UnknownLoc.
        if (isa<LocationAttr>(entryValue))
          entryValue = unknownLoc;
        else if (replacer)
          if (auto replaced = replacer(entryValue))
            entryValue = replaced;
        return iface->writeAttribute(entryValue, writer);
      });

  return mlir::writeBytecodeToFile(op, os, config);
}

FailureOr<std::string> M::getModuleBytecodeHash(mlir::ModuleOp module) {
  auto context = module.getContext();
  auto ops =
      llvm::map_to_vector(module.getOps(), [](Operation &op) { return &op; });

  using CacheT = llvm::SmallVector<std::string>;

  auto workFunc = [](CacheT &cache, Operation *op) -> LogicalResult {
    auto result = M::getBytecodeHash(op);
    if (failed(result))
      return failure();
    cache.push_back(std::move(*result));
    return success();
  };

  auto consolidateFn = [](CacheT &original, MutableArrayRef<CacheT> caches) {
    for (CacheT &cache : caches)
      llvm::move(cache, std::back_inserter(original));
  };

  CacheT resultCache;
  resultCache.reserve(ops.size());

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
  llvm::sort(resultCache);

  raw_xxhash_stream hasher;
  llvm::interleave(resultCache, hasher, "");
  return hasher.hashString();
}
