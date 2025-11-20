//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HASHUTILS_H
#define SUPPORT_HASHUTILS_H

#include "Support/LogicalResult.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#include "xxh3.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace M {

/// A raw_ostream that hash the content using the xxhash algorithm.
class raw_xxhash_stream : public llvm::raw_ostream {
  XXH3_state_t State;

  /// See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override {
    XXH3_128bits_update(&State, (void *)Ptr, Size);
  }

public:
  raw_xxhash_stream();

  raw_xxhash_stream(raw_xxhash_stream &&other)
      : State(std::move(other.State)) {}

  std::array<uint8_t, 16> hash();
  std::string hashString();

  uint64_t current_pos() const override { return 0; }
};

/// BytecodeHasher provides stable IR hashing which ignores source location
/// information. The hash function is based on hashing MLIR's bytecode
/// format.
class BytecodeHasher {
public:
  BytecodeHasher();

  FailureOr<std::string> getBytecodeHash(mlir::Operation *op);

private:
  /// Replacer which will strip location information and any other
  /// replacements provided by the caller.
  mlir::AttrTypeReplacer replacer;
};

/// This function computes the bytecode hash similar to getBytecodeHash, but
/// does so by hashing each individual operation in the module in parallel
/// (via BytecodeHasher) and then combining the results into a single hash.
FailureOr<std::string> getModuleBytecodeHash(mlir::ModuleOp module);

} // namespace M
#endif // SUPPORT_HASHUTILS_H
