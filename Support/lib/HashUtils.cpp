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
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
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

FailureOr<std::string> M::getBytecodeHash(mlir::Operation *op) {
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
        return iface->writeAttribute(entryValue, writer);
      });

  raw_xxhash_stream ostream;
  if (failed(mlir::writeBytecodeToFile(op, ostream, config)))
    return op->emitError("Failed to write bytecode");

  return ostream.hashString();
}

FailureOr<std::string> M::getModuleBytecodeHash(mlir::ModuleOp module) {
  auto context = module.getContext();
  auto ops =
      llvm::map_to_vector(module.getOps(), [](Operation &op) { return &op; });

  using CacheT = llvm::SmallVector<std::string>;

  auto workFunc = [&](CacheT &cache, Operation *op) -> LogicalResult {
    auto result = M::getBytecodeHash(op);
    if (failed(result))
      return failure();
    cache.push_back(std::move(*result));
    return success();
  };

  auto consolidateFn = [](CacheT &original, ArrayRef<CacheT> caches) {
    raw_xxhash_stream hasher;

    for (const CacheT &cache : caches)
      llvm::interleave(cache, hasher, "");

    return hasher.hashString();
  };

  CacheT resultCache;
  auto result = failableParallelForEach(
      /*ctx=*/context,
      /*range=*/ops,
      /*func=*/workFunc,
      /*cache=*/resultCache,
      /*consolidate=*/consolidateFn);

  if (failed(result))
    return failure();

  return resultCache[0];
}
