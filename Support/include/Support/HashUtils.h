//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HASHUTILS_H
#define SUPPORT_HASHUTILS_H

#include "Support/LogicalResult.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace M {

// Returns a unique hash for a given block. Values will be hashed only by the
// operation which returns them and the type. So two blocks with one operation
// will return the same hash, even though the value is a different pointer.
llvm::hash_code hashBlock(mlir::Block &block);

/// This function serializes an operation to MLIR bytecode and hashes the result
/// using XXH3's 128-bit variant and then returns the result as a hex string.
///
/// This is useful for deduping operations in a stable way, without relying on
/// in-memory values (e.g. pointers).
FailureOr<std::string> getBytecodeHash(mlir::Operation *op);

} // namespace M
#endif // SUPPORT_HASHUTILS_H
