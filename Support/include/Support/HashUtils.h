//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HASHUTILS_H
#define SUPPORT_HASHUTILS_H

#include "Support/LogicalResult.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace M {

/// This function serializes an operation to MLIR bytecode and hashes the result
/// using XXH3's 128-bit variant and then returns the result as a hex string.
///
/// This is useful for deduping operations in a stable way, without relying on
/// in-memory values (e.g. pointers).
FailureOr<std::string> getBytecodeHash(mlir::Operation *op);

/// This function computes the bytecode hash similar to getBytecodeHash, but
/// does so by hashing each individual operation in the module in parallel
/// (via getBytecodeHash) and then combining the results into a single hash.
FailureOr<std::string> getModuleBytecodeHash(mlir::ModuleOp module);

} // namespace M
#endif // SUPPORT_HASHUTILS_H
