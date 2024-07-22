//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HASHUTILS_H
#define SUPPORT_HASHUTILS_H

#include "llvm/ADT/Hashing.h"

namespace mlir {
class Block;
}

namespace M {

// Returns a unique hash for a given block. Values will be hashed only by the
// operation which returns them and the type. So two blocks with one operation
// will return the same hash, even though the value is a different pointer.
llvm::hash_code hashBlock(mlir::Block &block);

} // namespace M
#endif // SUPPORT_HASHUTILS_H
