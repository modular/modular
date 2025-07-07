//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HASHUTILS_H
#define SUPPORT_HASHUTILS_H

#include "Support/LogicalResult.h"

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

/// This function serializes an operation to MLIR bytecode and hashes the result
/// using XXH3's 128-bit variant and then returns the result as a hex string.
///
/// This is useful for deduping operations in a stable way, without relying on
/// in-memory values (e.g. pointers).
using ReplacementFunc = llvm::function_ref<mlir::Attribute(mlir::Attribute)>;
FailureOr<std::string> getBytecodeHash(mlir::Operation *op,
                                       ReplacementFunc replace = nullptr);

LogicalResult writeBytecode(mlir::Operation *op, llvm::raw_ostream &os,
                            ReplacementFunc replace = nullptr);

/// This function computes the bytecode hash similar to getBytecodeHash, but
/// does so by hashing each individual operation in the module in parallel
/// (via getBytecodeHash) and then combining the results into a single hash.
FailureOr<std::string> getModuleBytecodeHash(mlir::ModuleOp module);

} // namespace M
#endif // SUPPORT_HASHUTILS_H
